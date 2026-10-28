//! Recursive Flint native execution over original BF16 or MLX affine weights.
//!
//! Tensor contracts and equations follow `rec/engine/src/rec_mlx/model.py`,
//! revision 0e3150ca4262f572dee25c231d9ac7148414d2af (Apache-2.0),
//! Copyright 2026 Recursive Superintelligence, Inc. This independent Rust
//! implementation uses Stark's native matrix kernels; it does not execute
//! repository Python or invoke a foreign inference runtime.

use std::collections::VecDeque;

use serde_json::Value;

use super::{
    Error, Result,
    math::{rms_norm, sigmoid, silu, softmax, topk},
    weights::Weights,
};

// Allocated lazily, with a bounded, checked maximum across every layer. The
// published checkpoint needs about 14 GiB of f32 full-attention KV at 256k.
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
struct LayerConfig {
    sliding: bool,
    rope: bool,
    dense: bool,
}

#[derive(Clone, Debug)]
struct Config {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    layers: Vec<LayerConfig>,
    intermediate: usize,
    expert_intermediate: usize,
    shared_intermediate: usize,
    experts: usize,
    active: usize,
    vocab: usize,
    context: usize,
    window: usize,
    eps: f32,
    theta: f32,
}

fn integer(
    c: &Value,
    name: &str,
    maximum: usize,
) -> Result<usize> {
    let value = c.get(name).and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok());
    value.filter(|&v| v > 0 && v <= maximum).ok_or(Error::Invalid("flint_config_integer"))
}

fn number(
    c: &Value,
    name: &str,
) -> Result<f32> {
    let value = c.get(name).and_then(Value::as_f64).ok_or(Error::Invalid("flint_config_number"))? as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::Invalid("flint_config_number"));
    }
    Ok(value)
}

fn indices(
    c: &Value,
    name: &str,
    count: usize,
) -> Result<Vec<bool>> {
    let values = c.get(name).and_then(Value::as_array).ok_or(Error::Invalid("flint_layer_indices"))?;
    if values.len() > count {
        return Err(Error::Invalid("flint_layer_indices"));
    }
    let mut found = vec![false; count];
    for value in values {
        let i = value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v < count)
            .ok_or(Error::Invalid("flint_layer_indices"))?;
        if found[i] {
            return Err(Error::Invalid("flint_duplicate_layer_index"));
        }
        found[i] = true;
    }
    Ok(found)
}

impl Config {
    fn parse(c: &Value) -> Result<Self> {
        if c.get("model_type").and_then(Value::as_str) != Some("flint") {
            return Err(Error::Unsupported("flint_architecture"));
        }
        if c.get("hidden_act").and_then(Value::as_str) != Some("silu")
            || c.get("attention_bias").and_then(Value::as_bool) != Some(false)
            || c.get("tie_word_embeddings").and_then(Value::as_bool) != Some(false)
            || c.get("norm_topk_prob").and_then(Value::as_bool) != Some(true)
            || c.get("softmax_type").and_then(Value::as_str) != Some("learnable")
            || c.get("decoder_sparse_step").and_then(Value::as_u64) != Some(1)
            || c.get("use_sliding_window").and_then(Value::as_bool) != Some(true)
            || c.get("attention_dropout").and_then(Value::as_f64) != Some(0.0)
            || c.get("attention_sink").is_some_and(|v| v.as_bool() != Some(true))
            || c.get("rope_scaling").is_some_and(|v| !v.is_null())
        {
            return Err(Error::Unsupported("flint_operation_configuration"));
        }
        let router = c.get("router_score_function").or_else(|| c.get("scoring_func"));
        if router.and_then(Value::as_str) != Some("sigmoid")
            || ["router_score_function", "scoring_func"]
                .iter()
                .any(|name| c.get(name).is_some_and(|v| v.as_str() != Some("sigmoid")))
        {
            return Err(Error::Unsupported("flint_router_configuration"));
        }
        let count = integer(c, "num_hidden_layers", 128)?;
        let no_rope = indices(c, "no_rope_layer_indices", count)?;
        let dense = indices(c, "mlp_only_layers", count)?;
        let raw_layers = c.get("layer_types").and_then(Value::as_array).ok_or(Error::Invalid("flint_layer_types"))?;
        if raw_layers.len() != count {
            return Err(Error::Invalid("flint_layer_types"));
        }
        let layers = raw_layers
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let sliding = match value.as_str() {
                    Some("full_attention") => false,
                    Some("sliding_attention") => true,
                    _ => return Err(Error::Unsupported("flint_attention_type")),
                };
                Ok(LayerConfig {
                    sliding,
                    rope: !no_rope[i],
                    dense: dense[i],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let result = Self {
            hidden: integer(c, "hidden_size", 32768)?,
            heads: integer(c, "num_attention_heads", 256)?,
            kv_heads: integer(c, "num_key_value_heads", 256)?,
            dim: integer(c, "head_dim", 512)?,
            layers,
            intermediate: integer(c, "intermediate_size", 131072)?,
            expert_intermediate: integer(c, "moe_intermediate_size", 32768)?,
            shared_intermediate: integer(c, "shared_expert_intermediate_size", 32768)?,
            experts: integer(c, "num_experts", 2048)?,
            active: integer(c, "num_experts_per_tok", 64)?,
            vocab: integer(c, "vocab_size", 1_000_000)?,
            context: integer(c, "max_position_embeddings", 1_048_576)?,
            window: integer(c, "sliding_window", 1_048_576)?,
            eps: number(c, "rms_norm_eps")?,
            theta: number(c, "rope_theta")?,
        };
        if !result.heads.is_multiple_of(result.kv_heads)
            || !result.dim.is_multiple_of(2)
            || result.active > result.experts
            || result.heads * result.dim > 131072
            || c.get("num_local_experts").is_some_and(|v| v.as_u64() != Some(result.experts as u64))
        {
            return Err(Error::Invalid("flint_geometry"));
        }
        if let Some(rope) = c.get("rope_parameters")
            && (rope.get("rope_type").and_then(Value::as_str) != Some("default")
                || rope.get("rope_theta").and_then(Value::as_f64).map(|v| v as f32) != Some(result.theta)
                || rope.as_object().is_none_or(|v| v.keys().any(|k| !matches!(k.as_str(), "rope_type" | "rope_theta"))))
        {
            return Err(Error::Unsupported("flint_rope_configuration"));
        }
        let cache_rows = result.layers.iter().try_fold(0usize, |sum, layer| {
            sum.checked_add(if layer.sliding {
                result.window.min(result.context)
            } else {
                result.context
            })
        });
        if cache_rows
            .and_then(|n| n.checked_mul(result.kv_heads * result.dim * 2 * size_of::<f32>()))
            .is_none_or(|n| n > MAX_CACHE_BYTES)
        {
            return Err(Error::Invalid("flint_cache_bound"));
        }
        Ok(result)
    }
}

#[derive(Default)]
struct AttentionState {
    keys: VecDeque<Vec<f32>>,
    values: VecDeque<Vec<f32>>,
}

pub struct FlintModel {
    weights: Weights,
    config: Config,
    states: Vec<AttentionState>,
    banks: Vec<bool>,
    position: usize,
}

impl FlintModel {
    pub fn new(
        weights: Weights,
        config: &Value,
    ) -> Result<Self> {
        let config = Config::parse(config)?;
        let banks = validate_weights(&weights, &config)?;
        let states = config.layers.iter().map(|_| AttentionState::default()).collect();
        Ok(Self {
            weights,
            config,
            states,
            banks,
            position: 0,
        })
    }

    pub fn reset(&mut self) {
        self.states.iter_mut().for_each(|state| {
            state.keys.clear();
            state.values.clear();
        });
        self.position = 0;
    }

    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.advance(token, true)
    }

    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        self.advance(token, false).map(|_| ())
    }

    fn advance(
        &mut self,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        let result = self.step_inner(token, readout);
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn step_inner(
        &mut self,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        if token as usize >= c.vocab || self.position >= c.context {
            return Err(Error::Invalid("flint_token_or_context"));
        }
        let mut x = self.weights.row("model.embed_tokens.weight", token as usize)?;
        for (i, state) in self.states.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            let normalized = rms_norm(&x, &self.weights.vector(&format!("{prefix}.input_layernorm.weight"))?, c.eps)?;
            let attn = attention(&self.weights, c, i, self.position, &normalized, state)?;
            for (value, update) in x.iter_mut().zip(attn) {
                *value += update;
            }
            let normalized =
                rms_norm(&x, &self.weights.vector(&format!("{prefix}.post_attention_layernorm.weight"))?, c.eps)?;
            let mlp_prefix = format!("{prefix}.mlp");
            let output = if c.layers[i].dense {
                dense(&self.weights, &mlp_prefix, &normalized)?
            } else {
                moe(&self.weights, c, &mlp_prefix, &normalized, self.banks[i])?
            };
            for (value, update) in x.iter_mut().zip(output) {
                *value += update;
            }
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("flint_nonfinite_hidden_state"));
        }
        let output = if readout {
            let norm = rms_norm(&x, &self.weights.vector("model.norm.weight")?, c.eps)?;
            let logits = self.weights.matvec("lm_head.weight", &norm)?;
            if logits.len() != c.vocab || logits.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("flint_nonfinite_output"));
            }
            logits
        } else {
            Vec::new()
        };
        self.position += 1;
        Ok(output)
    }
}

fn shape(
    w: &Weights,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    if w.shape(name)? != expected {
        return Err(Error::Invalid("flint_weight_shape"));
    }
    Ok(())
}

fn validate_dense(
    w: &Weights,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    for kind in ["gate", "up"] {
        shape(w, &format!("{prefix}.{kind}_proj.weight"), &[intermediate, hidden])?;
    }
    shape(w, &format!("{prefix}.down_proj.weight"), &[hidden, intermediate])
}

fn validate_weights(
    w: &Weights,
    c: &Config,
) -> Result<Vec<bool>> {
    shape(w, "model.embed_tokens.weight", &[c.vocab, c.hidden])?;
    shape(w, "model.norm.weight", &[c.hidden])?;
    shape(w, "lm_head.weight", &[c.vocab, c.hidden])?;
    let mut banks = vec![false; c.layers.len()];
    for (i, layer) in c.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            shape(w, &format!("{p}.{norm}.weight"), &[c.hidden])?;
        }
        for norm in ["q", "k"] {
            shape(w, &format!("{p}.self_attn.{norm}_norm.weight"), &[c.dim])?;
        }
        shape(w, &format!("{p}.self_attn.softmax_offset"), &[c.heads])?;
        shape(w, &format!("{p}.self_attn.q_proj.weight"), &[c.heads * c.dim, c.hidden])?;
        for kind in ["k", "v"] {
            shape(w, &format!("{p}.self_attn.{kind}_proj.weight"), &[c.kv_heads * c.dim, c.hidden])?;
        }
        shape(w, &format!("{p}.self_attn.o_proj.weight"), &[c.hidden, c.heads * c.dim])?;
        let p = format!("{p}.mlp");
        if layer.dense {
            validate_dense(w, &p, c.hidden, c.intermediate)?;
        } else {
            shape(w, &format!("{p}.gate.weight"), &[c.experts, c.hidden])?;
            let bias = format!("{p}.gate.expert_bias");
            if w.has(&bias) {
                shape(w, &bias, &[c.experts])?;
            }
            validate_dense(w, &format!("{p}.shared_expert"), c.hidden, c.shared_intermediate)?;
            banks[i] = w.has(&format!("{p}.switch_mlp.gate_proj.weight"));
            if banks[i] {
                for kind in ["gate", "up"] {
                    shape(
                        w,
                        &format!("{p}.switch_mlp.{kind}_proj.weight"),
                        &[c.experts, c.expert_intermediate, c.hidden],
                    )?;
                }
                shape(w, &format!("{p}.switch_mlp.down_proj.weight"), &[c.experts, c.hidden, c.expert_intermediate])?;
            } else {
                for expert in 0..c.experts {
                    validate_dense(w, &format!("{p}.experts.{expert}"), c.hidden, c.expert_intermediate)?;
                }
            }
        }
    }
    Ok(banks)
}

fn rope(
    head: &mut [f32],
    position: usize,
    theta: f32,
) {
    let half = head.len() / 2;
    for i in 0..half {
        let angle = position as f32 * theta.powf(-((2 * i) as f32) / head.len() as f32);
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cos - b * sin;
        head[i + half] = a * sin + b * cos;
    }
}

fn attention(
    w: &Weights,
    c: &Config,
    layer: usize,
    position: usize,
    input: &[f32],
    state: &mut AttentionState,
) -> Result<Vec<f32>> {
    let p = format!("model.layers.{layer}.self_attn");
    let names = [format!("{p}.q_proj.weight"), format!("{p}.k_proj.weight"), format!("{p}.v_proj.weight")];
    let mut projections = w.matvec_many(&[(&names[0], input), (&names[1], input), (&names[2], input)])?;
    let values = projections.pop().unwrap();
    let mut keys = projections.pop().unwrap();
    let mut queries = projections.pop().unwrap();
    let qnorm = w.vector(&format!("{p}.q_norm.weight"))?;
    let knorm = w.vector(&format!("{p}.k_norm.weight"))?;
    for (values, norm) in [(&mut queries, qnorm), (&mut keys, knorm)] {
        for head in values.chunks_exact_mut(c.dim) {
            head.copy_from_slice(&rms_norm(head, &norm, c.eps)?);
            if c.layers[layer].rope {
                rope(head, position, c.theta);
            }
        }
    }
    if queries.iter().chain(&keys).chain(&values).any(|v| !v.is_finite()) {
        return Err(Error::Invalid("flint_nonfinite_attention"));
    }
    if c.layers[layer].sliding && state.keys.len() == c.window {
        state.keys.pop_front();
        state.values.pop_front();
    }
    state.keys.push_back(keys);
    state.values.push_back(values);
    let sinks = w.vector(&format!("{p}.softmax_offset"))?;
    if sinks.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("flint_attention_sinks"));
    }
    let mut output = vec![0.0; c.heads * c.dim];
    let scale = (c.dim as f32).sqrt().recip();
    for (head, q) in queries.chunks_exact(c.dim).enumerate() {
        let kv_head = head / (c.heads / c.kv_heads);
        let begin = kv_head * c.dim;
        let mut scores: Vec<f32> = state
            .keys
            .iter()
            .map(|k| q.iter().zip(&k[begin..begin + c.dim]).map(|(&q, &k)| q * k).sum::<f32>() * scale)
            .collect();
        // The learned sink is an extra softmax logit with a zero value. Its
        // probability remains in the denominator, never in the value sum.
        scores.push(sinks[head]);
        let probabilities = softmax(&scores)?;
        for (probability, values) in probabilities.iter().zip(&state.values) {
            for (dst, value) in output[head * c.dim..(head + 1) * c.dim].iter_mut().zip(&values[begin..begin + c.dim]) {
                *dst += probability * value;
            }
        }
    }
    w.matvec(&format!("{p}.o_proj.weight"), &output)
}

fn dense(
    w: &Weights,
    prefix: &str,
    input: &[f32],
) -> Result<Vec<f32>> {
    let gate = format!("{prefix}.gate_proj.weight");
    let up = format!("{prefix}.up_proj.weight");
    let mut projected = w.matvec_many(&[(&gate, input), (&up, input)])?;
    let up = projected.pop().unwrap();
    let gate = projected.pop().unwrap();
    let activated: Vec<_> = gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
    w.matvec(&format!("{prefix}.down_proj.weight"), &activated)
}

fn route(
    logits: &[f32],
    bias: &[f32],
    count: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    if logits.len() != bias.len() || logits.iter().chain(bias).any(|v| !v.is_finite()) {
        return Err(Error::Invalid("flint_router_values"));
    }
    let scores: Vec<_> = logits.iter().map(|&v| sigmoid(v)).collect();
    let selection: Vec<_> = scores.iter().zip(bias).map(|(score, bias)| score + bias).collect();
    let selected = topk(&selection, count)?;
    let sum = selected.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    let probabilities = selected.iter().map(|&i| scores[i] / sum).collect();
    Ok((selected, probabilities))
}

fn moe(
    w: &Weights,
    c: &Config,
    p: &str,
    input: &[f32],
    banks: bool,
) -> Result<Vec<f32>> {
    let logits = w.matvec(&format!("{p}.gate.weight"), input)?;
    let bias_name = format!("{p}.gate.expert_bias");
    let bias = if w.has(&bias_name) {
        w.vector(&bias_name)?
    } else {
        vec![0.0; c.experts]
    };
    let (selected, probabilities) = route(&logits, &bias, c.active)?;
    if banks {
        let gate_name = format!("{p}.switch_mlp.gate_proj.weight");
        let up_name = format!("{p}.switch_mlp.up_proj.weight");
        let down_name = format!("{p}.switch_mlp.down_proj.weight");
        #[cfg(feature = "metal")]
        {
            let shared_gate = format!("{p}.shared_expert.gate_proj.weight");
            let shared_up = format!("{p}.shared_expert.up_proj.weight");
            let mut shared = w.matvec_many(&[(&shared_gate, input), (&shared_up, input)])?;
            let up = shared.pop().unwrap();
            let gate = shared.pop().unwrap();
            w.selected_ffn(
                &gate_name,
                &up_name,
                &down_name,
                &format!("{p}.shared_expert.down_proj.weight"),
                &selected,
                input,
                &probabilities,
                1.0,
                &gate,
                &up,
                None,
            )
        }
        #[cfg(not(feature = "metal"))]
        {
            let repeated = input.repeat(selected.len());
            let gate = w.selected_matvec(&gate_name, &selected, &repeated)?;
            let up = w.selected_matvec(&up_name, &selected, &repeated)?;
            let activated: Vec<_> = gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
            let values = w.selected_matvec(&down_name, &selected, &activated)?;
            let mut result = dense(w, &format!("{p}.shared_expert"), input)?;
            for (row, probability) in values.chunks_exact(c.hidden).zip(probabilities) {
                for (dst, value) in result.iter_mut().zip(row) {
                    *dst += probability * value;
                }
            }
            Ok(result)
        }
    } else {
        let mut result = dense(w, &format!("{p}.shared_expert"), input)?;
        for (expert, probability) in selected.iter().zip(probabilities) {
            let values = dense(w, &format!("{p}.experts.{expert}"), input)?;
            for (dst, value) in result.iter_mut().zip(values) {
                *dst += probability * value;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "flint_tests.rs"]
mod tests;
