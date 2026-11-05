//! Native Ripple execution over original floating or GGUF block weights.
//!
//! Tensor contracts and equations follow Ricursive Intelligence's
//! `modeling_ricursive.py` and `configuration_ricursive.py`, revision
//! e27e171792f303d74c048184e8259ba663cfa0d5 (Apache-2.0). This independent
//! implementation executes Stark's native matrix kernels; model-provided
//! Python and foreign inference runtimes are never loaded.

use std::collections::VecDeque;

use serde_json::Value;

use super::{
    Error, Result,
    math::{rms_norm, sigmoid, silu, softmax},
    weights::Weights,
};

// The published maximum context is metadata, not an allocation request. Cache
// grows with the sequence and is checked before any layer is advanced.
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
struct Config {
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    sliding: Vec<bool>,
    vocab: usize,
    context: usize,
    window: usize,
    eps: f32,
    theta: f32,
    embed_scale: f32,
    clamp: Option<f32>,
    gated: bool,
    valres: bool,
}

fn integer(
    c: &Value,
    name: &str,
    maximum: usize,
) -> Result<usize> {
    c.get(name)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0 && v <= maximum)
        .ok_or(Error::Invalid("ripple_config_integer"))
}

fn number(
    c: &Value,
    name: &str,
) -> Result<f32> {
    c.get(name)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or(Error::Invalid("ripple_config_number"))
}

fn boolean(
    c: &Value,
    name: &str,
    default: bool,
) -> Result<bool> {
    match c.get(name) {
        Some(v) => v.as_bool().ok_or(Error::Invalid("ripple_config_boolean")),
        None => Ok(default),
    }
}

impl Config {
    fn parse(c: &Value) -> Result<Self> {
        if c.get("model_type").and_then(Value::as_str) != Some("ricursive") {
            return Err(Error::Unsupported("ripple_architecture"));
        }
        if c.get("hidden_act").and_then(Value::as_str) != Some("silu")
            || boolean(c, "tie_word_embeddings", false)?
            || boolean(c, "attention_bias", false)?
            || boolean(c, "mlp_bias", false)?
            || boolean(c, "is_encoder_decoder", false)?
            || boolean(c, "add_cross_attention", false)?
            || c.get("attention_dropout").is_some_and(|v| v.as_f64() != Some(0.0))
            || c.get("rope_scaling").is_some_and(|v| !v.is_null())
            || c.get("rope_parameters").is_some_and(|v| !v.is_null())
            || c.get("architectures").is_some_and(|v| v != &serde_json::json!(["RicursiveForCausalLM"]))
        {
            return Err(Error::Unsupported("ripple_operation_configuration"));
        }
        let count = integer(c, "num_hidden_layers", 256)?;
        let raw = c.get("layer_types").and_then(Value::as_array).ok_or(Error::Invalid("ripple_layer_types"))?;
        if raw.len() != count {
            return Err(Error::Invalid("ripple_layer_types"));
        }
        let sliding = raw
            .iter()
            .map(|v| match v.as_str() {
                Some("sliding_attention") => Ok(true),
                Some("full_attention") => Ok(false),
                _ => Err(Error::Unsupported("ripple_attention_type")),
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(pattern) = c.get("hybrid_layer_pattern") {
            let expected: Vec<_> = sliding.iter().map(|&s| usize::from(s)).collect();
            if pattern != &serde_json::json!(expected) {
                return Err(Error::Invalid("ripple_hybrid_layer_pattern"));
            }
        }
        if let Some(hybrid) = c.get("is_hybrid_swa")
            && hybrid.as_bool() != Some(sliding.contains(&true) && sliding.contains(&false))
        {
            return Err(Error::Invalid("ripple_hybrid_configuration"));
        }
        let config = Self {
            hidden: integer(c, "hidden_size", 32768)?,
            intermediate: integer(c, "intermediate_size", 131072)?,
            heads: integer(c, "num_attention_heads", 512)?,
            kv_heads: integer(c, "num_key_value_heads", 512)?,
            dim: integer(c, "head_dim", 512)?,
            sliding,
            vocab: integer(c, "vocab_size", 1_000_000)?,
            context: integer(c, "max_position_embeddings", 16_777_216)?,
            window: integer(c, "sliding_window", 1_048_576)?,
            eps: number(c, "rms_norm_eps")?,
            theta: number(c, "rope_theta")?,
            embed_scale: if c.get("embed_scale").is_none_or(Value::is_null) {
                1.0
            } else {
                number(c, "embed_scale")?
            },
            clamp: if c.get("ffn_clamp_threshold").is_none_or(Value::is_null)
                || c.get("ffn_clamp_threshold").and_then(Value::as_f64) == Some(0.0)
            {
                None
            } else {
                Some(number(c, "ffn_clamp_threshold")?)
            },
            gated: boolean(c, "use_gated_attention", true)?,
            valres: boolean(c, "use_valres", false)?,
        };
        boolean(c, "gate_full_precision", true)?;
        if !config.heads.is_multiple_of(config.kv_heads)
            || !config.dim.is_multiple_of(2)
            || config.heads * config.dim > 131072
        {
            return Err(Error::Invalid("ripple_geometry"));
        }
        Ok(config)
    }

    fn cache_bytes(
        &self,
        tokens: usize,
    ) -> Result<usize> {
        self.sliding
            .iter()
            .try_fold(0usize, |sum, &sliding| {
                sum.checked_add(if sliding {
                    tokens.min(self.window)
                } else {
                    tokens
                })
            })
            .and_then(|rows| rows.checked_mul(self.kv_heads * self.dim * 2 * size_of::<f32>()))
            .filter(|&bytes| bytes <= MAX_CACHE_BYTES)
            .ok_or(Error::Invalid("ripple_cache_bound"))
    }
}

struct Layer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: String,
    k: String,
    v: String,
    o: String,
    gate: Option<String>,
    blend: Option<f32>,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

#[derive(Default)]
struct AttentionState {
    keys: VecDeque<Vec<f32>>,
    values: VecDeque<Vec<f32>>,
}

pub struct RippleModel {
    weights: Weights,
    config: Config,
    embedding: String,
    output: String,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    states: Vec<AttentionState>,
    position: usize,
}

impl RippleModel {
    pub fn new(
        weights: Weights,
        config: &Value,
    ) -> Result<Self> {
        let config = Config::parse(config)?;
        let gguf = !weights.metadata().is_empty();
        if gguf {
            validate_metadata(&weights, &config)?;
        }
        let embedding = if gguf {
            "token_embd.weight"
        } else {
            "model.embed_tokens.weight"
        }
        .to_owned();
        let output = if gguf {
            "output.weight"
        } else {
            "lm_head.weight"
        }
        .to_owned();
        let norm = if gguf {
            "output_norm.weight"
        } else {
            "model.norm.weight"
        };
        shape(&weights, &embedding, &[config.vocab, config.hidden])?;
        shape(&weights, &output, &[config.vocab, config.hidden])?;
        let norm = vector(&weights, norm, config.hidden)?;
        let layers =
            (0..config.sliding.len()).map(|i| load_layer(&weights, &config, i, gguf)).collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "metal")]
        {
            // A dense model scans every projection at each token. Retaining
            // the complete compressed inventory avoids repeatedly registering
            // and evicting Metal wrappers when the original 8 GiB cache is
            // smaller than the model. Embeddings remain CPU-only mappings.
            let mut projections = vec![output.clone()];
            for layer in &layers {
                projections.extend([
                    layer.q.clone(),
                    layer.k.clone(),
                    layer.v.clone(),
                    layer.o.clone(),
                    layer.ffn_gate.clone(),
                    layer.ffn_up.clone(),
                    layer.ffn_down.clone(),
                ]);
                if let Some(gate) = &layer.gate {
                    projections.push(gate.clone());
                }
            }
            weights.configure_complete_dense_views(&projections, 8 * 1024 * 1024 * 1024)?;
        }
        let states = config.sliding.iter().map(|_| AttentionState::default()).collect();
        Ok(Self {
            weights,
            config,
            embedding,
            output,
            norm,
            layers,
            states,
            position: 0,
        })
    }

    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.keys.clear();
            state.values.clear();
        }
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
            return Err(Error::Invalid("ripple_token_or_context"));
        }
        c.cache_bytes(self.position + 1)?;
        let mut seed = self.weights.row(&self.embedding, token as usize)?;
        for value in &mut seed {
            *value *= c.embed_scale;
        }
        let mut x = seed.clone();
        for (i, (layer, state)) in self.layers.iter().zip(&mut self.states).enumerate() {
            let input = rms_norm(&x, &layer.input_norm, c.eps)?;
            let update = attention(&self.weights, c, layer, i, self.position, &input, &seed, state)?;
            residual(&mut x, &rms_norm(&update, &layer.post_attention_norm, c.eps)?)?;
            let input = rms_norm(&x, &layer.pre_ffn_norm, c.eps)?;
            let update = feedforward(&self.weights, c, layer, &input)?;
            residual(&mut x, &rms_norm(&update, &layer.post_ffn_norm, c.eps)?)?;
        }
        let output = if readout {
            let logits = self.weights.matvec(&self.output, &rms_norm(&x, &self.norm, c.eps)?)?;
            if logits.len() != c.vocab || logits.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("ripple_nonfinite_output"));
            }
            logits
        } else {
            Vec::new()
        };
        self.position += 1;
        Ok(output)
    }
}

fn residual(
    x: &mut [f32],
    update: &[f32],
) -> Result<()> {
    if x.len() != update.len() {
        return Err(Error::Invalid("ripple_residual_shape"));
    }
    for (value, update) in x.iter_mut().zip(update) {
        *value += update;
        if !value.is_finite() {
            return Err(Error::Invalid("ripple_nonfinite_hidden_state"));
        }
    }
    Ok(())
}

fn shape(
    w: &Weights,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    if w.shape(name)? != expected {
        return Err(Error::Invalid("ripple_weight_shape"));
    }
    Ok(())
}

fn vector(
    w: &Weights,
    name: &str,
    length: usize,
) -> Result<Vec<f32>> {
    shape(w, name, &[length])?;
    let values = w.vector(name)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("ripple_nonfinite_parameter"));
    }
    Ok(values)
}

fn load_layer(
    w: &Weights,
    c: &Config,
    i: usize,
    gguf: bool,
) -> Result<Layer> {
    let name = |hf: &str, gg: &str| {
        if gguf {
            format!("blk.{i}.{gg}")
        } else {
            format!("model.layers.{i}.{hf}")
        }
    };
    let q = name("self_attn.q_proj.weight", "attn_q.weight");
    let k = name("self_attn.k_proj.weight", "attn_k.weight");
    let v = name("self_attn.v_proj.weight", "attn_v.weight");
    let o = name("self_attn.o_proj.weight", "attn_output.weight");
    let gate = c.gated.then(|| name("self_attn.g_proj.weight", "attn_gate.weight"));
    let ffn_gate = name("mlp.gate_proj.weight", "ffn_gate.weight");
    let ffn_up = name("mlp.up_proj.weight", "ffn_up.weight");
    let ffn_down = name("mlp.down_proj.weight", "ffn_down.weight");
    shape(w, &q, &[c.heads * c.dim, c.hidden])?;
    for tensor in [&k, &v] {
        shape(w, tensor, &[c.kv_heads * c.dim, c.hidden])?;
    }
    shape(w, &o, &[c.hidden, c.heads * c.dim])?;
    if let Some(gate) = &gate {
        shape(w, gate, &[c.heads * c.dim, c.hidden])?;
    }
    for tensor in [&ffn_gate, &ffn_up] {
        shape(w, tensor, &[c.intermediate, c.hidden])?;
    }
    shape(w, &ffn_down, &[c.hidden, c.intermediate])?;
    let blend = if c.valres {
        let atomic = name("self_attn.value_res_lambda", "attn_vres_lambda.weight");
        let alternate = format!("blk.{i}.attn_value_res.weight");
        if gguf && w.has(&atomic) && w.has(&alternate) {
            return Err(Error::Invalid("ripple_duplicate_value_residual"));
        }
        let lambda = if gguf && !w.has(&atomic) {
            alternate
        } else {
            atomic
        };
        Some(sigmoid(vector(w, &lambda, 1)?[0]))
    } else {
        None
    };
    Ok(Layer {
        input_norm: vector(w, &name("input_layernorm.weight", "attn_norm.weight"), c.hidden)?,
        post_attention_norm: vector(
            w,
            &name("post_attention_layernorm.weight", "post_attention_norm.weight"),
            c.hidden,
        )?,
        pre_ffn_norm: vector(w, &name("pre_feedforward_layernorm.weight", "ffn_norm.weight"), c.hidden)?,
        post_ffn_norm: vector(w, &name("post_feedforward_layernorm.weight", "post_ffw_norm.weight"), c.hidden)?,
        q_norm: vector(w, &name("self_attn.q_norm.weight", "attn_q_norm.weight"), c.dim)?,
        k_norm: vector(w, &name("self_attn.k_norm.weight", "attn_k_norm.weight"), c.dim)?,
        q,
        k,
        v,
        o,
        gate,
        blend,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}

fn validate_metadata(
    w: &Weights,
    c: &Config,
) -> Result<()> {
    let m = w.metadata();
    if m.get("general.architecture").and_then(Value::as_str) != Some("ricursive") {
        return Err(Error::Unsupported("ripple_gguf_architecture"));
    }
    for (name, expected) in [
        ("block_count", c.sliding.len()),
        ("embedding_length", c.hidden),
        ("feed_forward_length", c.intermediate),
        ("attention.head_count", c.heads),
        ("attention.head_count_kv", c.kv_heads),
        ("attention.key_length", c.dim),
        ("attention.value_length", c.dim),
        ("context_length", c.context),
        ("attention.sliding_window", c.window),
        ("rope.dimension_count", c.dim),
    ] {
        if m.get(&format!("ricursive.{name}")).is_some_and(|v| v.as_u64() != Some(expected as u64)) {
            return Err(Error::Invalid("ripple_gguf_geometry_mismatch"));
        }
    }
    for (name, expected) in [
        ("attention.layer_norm_rms_epsilon", c.eps),
        ("rope.freq_base", c.theta),
        ("embedding_scale", c.embed_scale),
        ("swiglu_clamp", c.clamp.unwrap_or(0.0)),
    ] {
        if m.get(&format!("ricursive.{name}")).is_some_and(|v| v.as_f64().map(|n| n as f32) != Some(expected)) {
            return Err(Error::Invalid("ripple_gguf_operation_mismatch"));
        }
    }
    if let Some(pattern) = m.get("ricursive.attention.sliding_window_pattern") {
        let actual = pattern.as_array().ok_or(Error::Invalid("ripple_gguf_sliding_pattern"))?;
        if actual.len() != c.sliding.len()
            || actual.iter().zip(&c.sliding).any(|(actual, &expected)| {
                actual.as_bool().or_else(|| actual.as_u64().filter(|&v| v <= 1).map(|v| v != 0)) != Some(expected)
            })
        {
            return Err(Error::Invalid("ripple_gguf_sliding_pattern"));
        }
    }
    Ok(())
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
    layer: &Layer,
    i: usize,
    position: usize,
    input: &[f32],
    seed: &[f32],
    state: &mut AttentionState,
) -> Result<Vec<f32>> {
    let values =
        layer.blend.map(|blend| seed.iter().zip(input).map(|(&s, &h)| s + blend * (h - s)).collect::<Vec<_>>());
    let mut jobs = vec![
        (layer.q.as_str(), input),
        (layer.k.as_str(), input),
        (layer.v.as_str(), values.as_deref().unwrap_or(input)),
    ];
    if let Some(gate) = &layer.gate {
        jobs.push((gate.as_str(), input));
    }
    let mut projections = w.matvec_many(&jobs)?;
    let gates = if layer.gate.is_some() {
        projections.pop()
    } else {
        None
    };
    let values = projections.pop().unwrap();
    let mut keys = projections.pop().unwrap();
    let mut queries = projections.pop().unwrap();
    for (values, norm) in [(&mut queries, &layer.q_norm), (&mut keys, &layer.k_norm)] {
        for head in values.chunks_exact_mut(c.dim) {
            head.copy_from_slice(&rms_norm(head, norm, c.eps)?);
            if c.sliding[i] {
                rope(head, position, c.theta);
            }
        }
    }
    if queries.iter().chain(&keys).chain(&values).chain(gates.as_deref().unwrap_or_default()).any(|v| !v.is_finite()) {
        return Err(Error::Invalid("ripple_nonfinite_attention"));
    }
    if c.sliding[i] && state.keys.len() == c.window {
        state.keys.pop_front();
        state.values.pop_front();
    }
    state.keys.push_back(keys);
    state.values.push_back(values);
    let mut output = vec![0.0; c.heads * c.dim];
    let scale = (c.dim as f32).sqrt().recip();
    for (head, q) in queries.chunks_exact(c.dim).enumerate() {
        let begin = head / (c.heads / c.kv_heads) * c.dim;
        let scores: Vec<f32> = state
            .keys
            .iter()
            .map(|k| q.iter().zip(&k[begin..begin + c.dim]).map(|(&q, &k)| q * k).sum::<f32>() * scale)
            .collect();
        let probabilities = softmax(&scores)?;
        for (probability, values) in probabilities.iter().zip(&state.values) {
            for (dst, value) in output[head * c.dim..(head + 1) * c.dim].iter_mut().zip(&values[begin..begin + c.dim]) {
                *dst += probability * value;
            }
        }
    }
    if let Some(gates) = gates {
        for (value, gate) in output.iter_mut().zip(gates) {
            *value *= sigmoid(gate);
        }
    }
    w.matvec(&layer.o, &output)
}

fn feedforward(
    w: &Weights,
    c: &Config,
    layer: &Layer,
    input: &[f32],
) -> Result<Vec<f32>> {
    let mut projected = w.matvec_many(&[(&layer.ffn_gate, input), (&layer.ffn_up, input)])?;
    let up = projected.pop().unwrap();
    let gate = projected.pop().unwrap();
    let activated: Vec<_> = gate
        .iter()
        .zip(up)
        .map(|(&g, u)| {
            let (g, u) = c.clamp.map_or((g, u), |limit| (g.min(limit), u.clamp(-limit, limit)));
            silu(g) * u
        })
        .collect();
    w.matvec(&layer.ffn_down, &activated)
}

#[cfg(test)]
#[path = "ripple_tests.rs"]
mod tests;
