//! GLM-5.3-Flash token-at-a-time native reference execution.
//!
//! GGUF tensor names, absorbed MLA and transformed `ssm_a` follow llama.cpp
//! 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552, `src/models/glm5-next.cpp`.
//! That implementation is MIT licensed, copyright (c) 2023-2026 the ggml
//! authors. This is an independent Rust implementation of its tensor contract.
//! The model configuration is Z.AI `glm5_next` (MIT). There is no MTP,
//! speculative state, persistent cache, or execution of repository Python here.

use serde_json::Value;

use super::{
    Error, Result,
    math::{rms_norm, sigmoid, silu, softmax, topk},
    weights::Weights,
};

#[derive(Clone)]
struct Config {
    hidden: usize,
    heads: usize,
    kda_heads: usize,
    kda_dim: usize,
    conv_kernel: usize,
    layer_types: Vec<bool>,
    dense_layers: usize,
    intermediate: usize,
    expert_intermediate: usize,
    experts: usize,
    active_experts: usize,
    shared_experts: usize,
    hc: usize,
    hc_iters: usize,
    rms_eps: f32,
    hc_eps: f32,
    decay_lower_bound: f32,
    q_rank: usize,
    kv_rank: usize,
    qk_dim: usize,
    value_dim: usize,
    index_heads: usize,
    index_dim: usize,
    index_pool: usize,
    index_topk: usize,
    swiglu_limit: f32,
    routed_scale: f32,
    vocab: usize,
    max_context: usize,
}

fn integer(
    value: &Value,
    name: &str,
    maximum: usize,
) -> Result<usize> {
    let n = value.get(name).and_then(Value::as_u64).ok_or(Error::Invalid("glm_config_integer"))?;
    let n = usize::try_from(n).map_err(|_| Error::Invalid("glm_config_integer"))?;
    if n == 0 || n > maximum {
        return Err(Error::Invalid("glm_config_integer"));
    }
    Ok(n)
}

fn number(
    value: &Value,
    name: &str,
) -> Result<f32> {
    let n = value.get(name).and_then(Value::as_f64).ok_or(Error::Invalid("glm_config_number"))? as f32;
    if !n.is_finite() {
        return Err(Error::Invalid("glm_config_number"));
    }
    Ok(n)
}

impl Config {
    fn parse(outer: &Value) -> Result<Self> {
        if outer.get("model_type").and_then(Value::as_str) != Some("glm5_next")
            && outer.get("model_type").and_then(Value::as_str) != Some("glm5_next_text")
        {
            return Err(Error::Unsupported("glm_architecture"));
        }
        let c = outer.get("text_config").unwrap_or(outer);
        if (outer.get("text_config").is_some() && c.get("model_type").and_then(Value::as_str) != Some("glm5_next_text"))
            || c.get("hidden_act").and_then(Value::as_str) != Some("silu")
        {
            return Err(Error::Unsupported("glm_text_architecture"));
        }
        if c.get("qk_rope_head_dim").and_then(Value::as_u64) != Some(0)
            || c.get("mhc").and_then(Value::as_bool) != Some(true)
            || c.get("mla_use_nope").and_then(Value::as_bool) != Some(true)
            || c.get("attention_bias").and_then(Value::as_bool) != Some(false)
            || c.get("n_group").and_then(Value::as_u64) != Some(1)
            || c.get("topk_group").and_then(Value::as_u64) != Some(1)
            || c.get("norm_topk_prob").and_then(Value::as_bool) != Some(true)
            || c.get("scoring_func").and_then(Value::as_str) != Some("sigmoid")
            || c.get("topk_method").and_then(Value::as_str) != Some("noaux_tc")
            || c.get("index_kpool_always_select_tail").and_then(Value::as_bool) != Some(true)
            || c.get("index_kpool_compress").and_then(Value::as_bool) != Some(true)
        {
            return Err(Error::Unsupported("glm_operation_configuration"));
        }
        let count = integer(c, "num_hidden_layers", 128)?;
        let raw_types = c.get("layer_types").and_then(Value::as_array).ok_or(Error::Invalid("glm_layer_types"))?;
        if raw_types.len() != count {
            return Err(Error::Invalid("glm_layer_types"));
        }
        let layer_types = raw_types
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(true),
                Some("deepseek_sparse_attention") => Ok(false),
                _ => Err(Error::Unsupported("glm_layer_type")),
            })
            .collect::<Result<Vec<_>>>()?;
        let index_types =
            c.get("indexer_types").and_then(Value::as_array).ok_or(Error::Invalid("glm_indexer_types"))?;
        if index_types.len() != count || index_types.iter().any(|v| v.as_str() != Some("full")) {
            return Err(Error::Unsupported("glm_shared_indexer"));
        }
        // The original HF config nests these parameters; newer Transformers
        // configurations serialize their normalized aliases at the text level.
        let linear = c.get("linear_attn_config");
        let kda_heads = if let Some(l) = linear {
            integer(l, "num_heads", 256)?
        } else {
            integer(c, "linear_num_heads", 256)?
        };
        let kda_dim = if let Some(l) = linear {
            integer(l, "head_dim", 256)?
        } else {
            integer(c, "linear_head_dim", 256)?
        };
        let conv_kernel = if let Some(l) = linear {
            integer(l, "short_conv_kernel_size", 32)?
        } else {
            integer(c, "linear_conv_kernel_dim", 32)?
        };
        let decay_lower_bound = if let Some(l) = linear {
            number(l, "gate_lower_bound")?
        } else {
            number(c, "linear_lower_bound")?
        };
        let result = Self {
            hidden: integer(c, "hidden_size", 32768)?,
            heads: integer(c, "num_attention_heads", 256)?,
            kda_heads,
            kda_dim,
            conv_kernel,
            layer_types,
            dense_layers: integer(c, "first_k_dense_replace", count)?,
            intermediate: integer(c, "intermediate_size", 131072)?,
            expert_intermediate: integer(c, "moe_intermediate_size", 32768)?,
            experts: integer(c, "n_routed_experts", 2048)?,
            active_experts: integer(c, "num_experts_per_tok", 64)?,
            shared_experts: integer(c, "n_shared_experts", 16)?,
            hc: integer(c, "hc_mult", 8)?,
            hc_iters: integer(c, "hc_sinkhorn_iters", 128)?,
            rms_eps: number(c, "rms_norm_eps")?,
            hc_eps: number(c, "hc_eps")?,
            decay_lower_bound,
            q_rank: integer(c, "q_lora_rank", 8192)?,
            kv_rank: integer(c, "kv_lora_rank", 8192)?,
            qk_dim: integer(c, "qk_nope_head_dim", 512)?,
            value_dim: integer(c, "v_head_dim", 512)?,
            index_heads: integer(c, "index_n_heads", 128)?,
            index_dim: integer(c, "index_head_dim", 512)?,
            index_pool: integer(c, "index_kpool", 64)?,
            index_topk: integer(c, "index_topk", 65536)?,
            swiglu_limit: number(c, "swiglu_limit")?,
            routed_scale: number(c, "routed_scaling_factor")?,
            vocab: integer(c, "vocab_size", 1048576)?,
            max_context: integer(c, "max_position_embeddings", 1048576)?,
        };
        if integer(c, "num_key_value_heads", 256)? != result.heads {
            return Err(Error::Unsupported("glm_grouped_mla"));
        }
        if result.kda_heads != result.heads {
            return Err(Error::Unsupported("glm_kda_head_count"));
        }
        let mlp_types = c.get("mlp_layer_types").and_then(Value::as_array).ok_or(Error::Invalid("glm_mlp_types"))?;
        if mlp_types.len() != count
            || mlp_types.iter().enumerate().any(|(i, kind)| {
                kind.as_str()
                    != Some(if i < result.dense_layers {
                        "dense"
                    } else {
                        "sparse"
                    })
            })
        {
            return Err(Error::Unsupported("glm_mlp_layer_layout"));
        }
        if result.rms_eps <= 0.0
            || result.hc_eps <= 0.0
            || result.decay_lower_bound >= 0.0
            || result.swiglu_limit <= 0.0
            || result.routed_scale <= 0.0
            || result.active_experts > result.experts
            || result.conv_kernel < 2
            || result.index_pool < 2
            || !result.index_topk.is_multiple_of(result.index_pool)
        {
            return Err(Error::Invalid("glm_config_geometry"));
        }
        Ok(result)
    }
}

struct KdaState {
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    matrix: Vec<f32>,
}

impl KdaState {
    fn new(c: &Config) -> Self {
        let history_size = c.kda_heads * c.kda_dim * (c.conv_kernel - 1);
        Self {
            q_conv: vec![0.0; history_size],
            k_conv: vec![0.0; history_size],
            v_conv: vec![0.0; history_size],
            matrix: vec![0.0; c.kda_heads * c.kda_dim * c.kda_dim],
        }
    }

    fn reset(&mut self) {
        self.q_conv.fill(0.0);
        self.k_conv.fill(0.0);
        self.v_conv.fill(0.0);
        self.matrix.fill(0.0);
    }
}

#[derive(Default)]
struct AttentionState {
    latents: Vec<Vec<f32>>,
    // Only the incomplete pool retains uncompressed indexer rows. Completed
    // pools retain one key and their implicit consecutive token positions.
    tail_keys: Vec<Vec<f32>>,
    tail_gates: Vec<Vec<f32>>,
    pool_keys: Vec<Vec<f32>>,
}

enum LayerState {
    Kda(KdaState),
    Attention(AttentionState),
}

/// A complete autoregressive GLM text trunk over the original GGUF weights.
///
/// Small vector operations use the CPU reference arithmetic; matrix products
/// use the weight store's native implementation. Callers serialize whole
/// requests and call `reset` between unrelated requests.
pub struct GlmModel {
    weights: Weights,
    config: Config,
    states: Vec<LayerState>,
    position: usize,
    #[cfg(test)]
    layer_trace: Vec<Vec<f32>>,
}

impl GlmModel {
    pub fn new(
        weights: Weights,
        config: &Value,
    ) -> Result<Self> {
        let config = Config::parse(config)?;
        validate_metadata(&weights, &config)?;
        validate_weights(&weights, &config)?;
        let states = config
            .layer_types
            .iter()
            .map(|&kda| {
                if kda {
                    LayerState::Kda(KdaState::new(&config))
                } else {
                    LayerState::Attention(AttentionState::default())
                }
            })
            .collect();
        Ok(Self {
            weights,
            config,
            states,
            position: 0,
            #[cfg(test)]
            layer_trace: Vec::new(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab
    }

    pub fn max_context_length(&self) -> usize {
        self.config.max_context
    }

    pub fn speculative_decoding_supported(&self) -> bool {
        false
    }

    pub fn reset(&mut self) {
        for state in &mut self.states {
            match state {
                LayerState::Kda(s) => s.reset(),
                LayerState::Attention(s) => *s = AttentionState::default(),
            }
        }
        self.position = 0;
        #[cfg(test)]
        self.layer_trace.clear();
    }

    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.advance(token, true)
    }

    /// Advance all recurrent and attention state without projecting vocabulary
    /// logits that an intermediate prefill position will never consume.
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
        #[cfg(test)]
        self.layer_trace.clear();
        let c = &self.config;
        if token as usize >= c.vocab || self.position >= c.max_context {
            return Err(Error::Invalid("glm_token_or_context"));
        }
        let embedding = {
            let _scope = super::profile::Scope::new("glm.embedding");
            self.weights.row("token_embd.weight", token as usize)?
        };
        if embedding.len() != c.hidden {
            return Err(Error::Invalid("glm_embedding_shape"));
        }
        let mut streams = embedding.repeat(c.hc);
        for (layer, state) in self.states.iter_mut().enumerate() {
            let prefix = format!("blk.{layer}");
            let attn_mix = hc_pre(&self.weights, c, &streams, &prefix, "attn")?;
            let norm =
                rms_norm(&attn_mix.input, &self.weights.vector(&format!("{prefix}.attn_norm.weight"))?, c.rms_eps)?;
            let attention = match state {
                LayerState::Kda(s) => kda(&self.weights, c, &norm, &prefix, s)?,
                LayerState::Attention(s) => mla(&self.weights, c, &norm, &prefix, s)?,
            };
            streams = hc_post(&streams, &attention, &attn_mix, c.hidden, c.hc)?;
            let ffn_mix = hc_pre(&self.weights, c, &streams, &prefix, "ffn")?;
            let norm =
                rms_norm(&ffn_mix.input, &self.weights.vector(&format!("{prefix}.ffn_norm.weight"))?, c.rms_eps)?;
            let ffn = if layer < c.dense_layers {
                dense_ffn(&self.weights, &norm, &prefix, "", c.swiglu_limit)?
            } else {
                moe(&self.weights, c, &norm, &prefix)?
            };
            streams = hc_post(&streams, &ffn, &ffn_mix, c.hidden, c.hc)?;
            #[cfg(test)]
            self.layer_trace.push(streams.clone());
        }
        if !readout {
            if streams.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("glm_nonfinite_output"));
            }
            self.position += 1;
            return Ok(Vec::new());
        }
        let mut mean = vec![0.0; c.hidden];
        for stream in streams.chunks_exact(c.hidden) {
            for (dst, &src) in mean.iter_mut().zip(stream) {
                *dst += src;
            }
        }
        for v in &mut mean {
            *v /= c.hc as f32;
        }
        let _scope = super::profile::Scope::new("glm.readout");
        let norm = rms_norm(&mean, &self.weights.vector("output_norm.weight")?, c.rms_eps)?;
        let logits = self.weights.matvec("output.weight", &norm)?;
        if logits.len() != c.vocab || logits.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("glm_nonfinite_output"));
        }
        self.position += 1;
        Ok(logits)
    }
}

fn shape(
    weights: &Weights,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    if weights.shape(name)? != expected {
        return Err(Error::Invalid("glm_tensor_shape"));
    }
    Ok(())
}

fn validate_metadata(
    w: &Weights,
    c: &Config,
) -> Result<()> {
    let m = w.metadata();
    if m.get("general.architecture").and_then(Value::as_str) != Some("glm5-next") {
        return Err(Error::Unsupported("glm_gguf_architecture"));
    }
    let get_int = |name: &str| {
        m.get(&format!("glm5-next.{name}"))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(Error::Invalid("glm_gguf_metadata_integer"))
    };
    for (name, expected) in [
        ("embedding_length", c.hidden),
        ("context_length", c.max_context),
        ("feed_forward_length", c.intermediate),
        ("attention.head_count", c.heads),
        ("attention.q_lora_rank", c.q_rank),
        ("attention.kv_lora_rank", c.kv_rank),
        ("attention.key_length", c.kv_rank),
        ("attention.value_length", c.kv_rank),
        ("attention.key_length_mla", c.qk_dim),
        ("attention.value_length_mla", c.value_dim),
        ("rope.dimension_count", 0),
        ("expert_count", c.experts),
        ("expert_used_count", c.active_experts),
        ("expert_group_count", 1),
        ("expert_group_used_count", 1),
        ("expert_gating_func", 2),
        ("expert_feed_forward_length", c.expert_intermediate),
        ("expert_shared_feed_forward_length", c.expert_intermediate * c.shared_experts),
        ("expert_shared_count", c.shared_experts),
        ("leading_dense_block_count", c.dense_layers),
        ("vocab_size", c.vocab),
        ("ssm.conv_kernel", c.conv_kernel),
        ("kda.head_dim", c.kda_dim),
        ("attention.indexer.head_count", c.index_heads),
        ("attention.indexer.key_length", c.index_dim),
        ("attention.indexer.top_k", c.index_topk),
        ("attention.indexer.kpool", c.index_pool),
        ("hyper_connection.count", c.hc),
        ("hyper_connection.sinkhorn_iterations", c.hc_iters),
    ] {
        if get_int(name)? != expected {
            return Err(Error::Invalid("glm_gguf_config_disagreement"));
        }
    }
    // GGUF includes the optional NextN layer in block_count. The ordinary
    // autoregressive trunk never executes that layer or discovers a drafter.
    let nextn = get_int("nextn_predict_layers")?;
    let blocks = get_int("block_count")?;
    if nextn > 1 || blocks.checked_sub(nextn) != Some(c.layer_types.len()) {
        return Err(Error::Invalid("glm_gguf_trunk_layers"));
    }
    let kv_heads = m
        .get("glm5-next.attention.head_count_kv")
        .and_then(Value::as_array)
        .ok_or(Error::Invalid("glm_gguf_layer_types"))?;
    if kv_heads.len() != blocks
        || c.layer_types.iter().zip(kv_heads).any(|(&kda, value)| value.as_u64() != Some(u64::from(!kda)))
    {
        return Err(Error::Invalid("glm_gguf_layer_types"));
    }
    for (name, expected) in [
        ("attention.layer_norm_rms_epsilon", c.rms_eps),
        ("attention.layer_norm_epsilon", 1e-6),
        ("kda.gate_lower_bound", c.decay_lower_bound),
        ("hyper_connection.epsilon", c.hc_eps),
        ("expert_weights_scale", c.routed_scale),
    ] {
        let actual = m
            .get(&format!("glm5-next.{name}"))
            .and_then(Value::as_f64)
            .ok_or(Error::Invalid("glm_gguf_metadata_number"))? as f32;
        if !actual.is_finite() || (actual - expected).abs() > expected.abs().max(1e-6) * 1e-6 {
            return Err(Error::Invalid("glm_gguf_config_disagreement"));
        }
    }
    if m.get("glm5-next.expert_weights_norm").and_then(Value::as_bool) != Some(true)
        || !m.get("glm5-next.attention.indexer.kpool_select_tail").and_then(Value::as_bool).unwrap_or(true)
    {
        return Err(Error::Unsupported("glm_gguf_operation_configuration"));
    }
    for name in ["swiglu_clamp_exp", "swiglu_clamp_shexp"] {
        let values = m
            .get(&format!("glm5-next.{name}"))
            .and_then(Value::as_array)
            .ok_or(Error::Invalid("glm_gguf_swiglu_clamp"))?;
        if values.len() != blocks || values.iter().any(|v| v.as_f64().map(|v| v as f32) != Some(c.swiglu_limit)) {
            return Err(Error::Invalid("glm_gguf_swiglu_clamp"));
        }
    }
    Ok(())
}

fn validate_weights(
    w: &Weights,
    c: &Config,
) -> Result<()> {
    shape(w, "token_embd.weight", &[c.vocab, c.hidden])?;
    shape(w, "output.weight", &[c.vocab, c.hidden])?;
    shape(w, "output_norm.weight", &[c.hidden])?;
    let inner = c.kda_heads * c.kda_dim;
    for (layer, &is_kda) in c.layer_types.iter().enumerate() {
        let p = format!("blk.{layer}");
        for part in ["attn", "ffn"] {
            shape(w, &format!("{p}.{part}_norm.weight"), &[c.hidden])?;
            shape(w, &format!("{p}.hc_{part}_fn.weight"), &[(2 + c.hc) * c.hc, c.hc * c.hidden])?;
            shape(w, &format!("{p}.hc_{part}_base.weight"), &[(2 + c.hc) * c.hc])?;
            shape(w, &format!("{p}.hc_{part}_scale.weight"), &[3])?;
        }
        if is_kda {
            for part in ["q", "k", "v"] {
                shape(w, &format!("{p}.attn_{part}.weight"), &[inner, c.hidden])?;
                let dimensions = w.shape(&format!("{p}.ssm_conv1d_{part}.weight"))?;
                if dimensions != [inner, 1, c.conv_kernel] && dimensions != [inner, c.conv_kernel] {
                    return Err(Error::Invalid("glm_convolution_shape"));
                }
            }
            for part in ["f", "g"] {
                shape(w, &format!("{p}.ssm_{part}_a.weight"), &[c.kda_dim, c.hidden])?;
                shape(w, &format!("{p}.ssm_{part}_b.weight"), &[inner, c.kda_dim])?;
            }
            shape(w, &format!("{p}.ssm_beta.weight"), &[c.kda_heads, c.hidden])?;
            shape(w, &format!("{p}.ssm_a"), &[c.kda_heads])?;
            // GGUF stores negative exp(A_log), rather than the source A_log.
            if w.vector(&format!("{p}.ssm_a"))?.iter().any(|v| !v.is_finite() || *v >= 0.0) {
                return Err(Error::Invalid("glm_transformed_decay"));
            }
            shape(w, &format!("{p}.ssm_dt.bias"), &[inner])?;
            shape(w, &format!("{p}.ssm_norm.weight"), &[c.kda_dim])?;
            shape(w, &format!("{p}.attn_output.weight"), &[c.hidden, inner])?;
        } else {
            shape(w, &format!("{p}.attn_q_a.weight"), &[c.q_rank, c.hidden])?;
            shape(w, &format!("{p}.attn_q_a_norm.weight"), &[c.q_rank])?;
            shape(w, &format!("{p}.attn_q_b.weight"), &[c.heads * c.qk_dim, c.q_rank])?;
            shape(w, &format!("{p}.attn_kv_a_mqa.weight"), &[c.kv_rank, c.hidden])?;
            shape(w, &format!("{p}.attn_kv_a_norm.weight"), &[c.kv_rank])?;
            shape(w, &format!("{p}.attn_k_b.weight"), &[c.heads, c.kv_rank, c.qk_dim])?;
            shape(w, &format!("{p}.attn_v_b.weight"), &[c.heads, c.value_dim, c.kv_rank])?;
            shape(w, &format!("{p}.attn_output.weight"), &[c.hidden, c.heads * c.value_dim])?;
            shape(w, &format!("{p}.indexer.attn_q_b.weight"), &[c.index_heads * c.index_dim, c.q_rank])?;
            shape(w, &format!("{p}.indexer.attn_k.weight"), &[c.index_dim, c.hidden])?;
            shape(w, &format!("{p}.indexer.k_norm.weight"), &[c.index_dim])?;
            shape(w, &format!("{p}.indexer.k_norm.bias"), &[c.index_dim])?;
            shape(w, &format!("{p}.indexer.proj.weight"), &[c.index_heads, c.hidden])?;
            shape(w, &format!("{p}.indexer_compressor_gate.weight"), &[c.index_dim, c.hidden])?;
            shape(w, &format!("{p}.indexer_compressor_ape.weight"), &[c.index_pool, c.index_dim])?;
        }
        if layer < c.dense_layers {
            for part in ["gate", "up"] {
                shape(w, &format!("{p}.ffn_{part}.weight"), &[c.intermediate, c.hidden])?;
            }
            shape(w, &format!("{p}.ffn_down.weight"), &[c.hidden, c.intermediate])?;
        } else {
            shape(w, &format!("{p}.ffn_gate_inp.weight"), &[c.experts, c.hidden])?;
            shape(w, &format!("{p}.exp_probs_b.bias"), &[c.experts])?;
            for part in ["gate", "up"] {
                shape(w, &format!("{p}.ffn_{part}_exps.weight"), &[c.experts, c.expert_intermediate, c.hidden])?;
                shape(
                    w,
                    &format!("{p}.ffn_{part}_shexp.weight"),
                    &[c.shared_experts * c.expert_intermediate, c.hidden],
                )?;
            }
            shape(w, &format!("{p}.ffn_down_exps.weight"), &[c.experts, c.hidden, c.expert_intermediate])?;
            shape(w, &format!("{p}.ffn_down_shexp.weight"), &[c.hidden, c.shared_experts * c.expert_intermediate])?;
        }
    }
    Ok(())
}

struct HcMix {
    input: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

fn sinkhorn(
    logits: &[f32],
    count: usize,
    iterations: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if count == 0 || logits.len() != count * count || iterations == 0 || eps <= 0.0 {
        return Err(Error::Invalid("glm_sinkhorn_shape"));
    }
    let mut comb = Vec::with_capacity(logits.len());
    for row in logits.chunks_exact(count) {
        comb.extend(softmax(row)?.iter().map(|v| v + eps));
    }
    for iteration in 0..iterations {
        if iteration > 0 {
            for row in comb.chunks_exact_mut(count) {
                let denominator = row.iter().sum::<f32>() + eps;
                for v in row {
                    *v /= denominator;
                }
            }
        }
        for dst in 0..count {
            let denominator = (0..count).map(|src| comb[src * count + dst]).sum::<f32>() + eps;
            for src in 0..count {
                comb[src * count + dst] /= denominator;
            }
        }
    }
    Ok(comb)
}

fn hc_pre(
    w: &Weights,
    c: &Config,
    streams: &[f32],
    p: &str,
    part: &str,
) -> Result<HcMix> {
    let _scope = super::profile::Scope::new("glm.mhc.pre");
    let ones = vec![1.0; streams.len()];
    let norm = rms_norm(streams, &ones, c.rms_eps)?;
    let mix = w.matvec(&format!("{p}.hc_{part}_fn.weight"), &norm)?;
    let base = w.vector(&format!("{p}.hc_{part}_base.weight"))?;
    let scale = w.vector(&format!("{p}.hc_{part}_scale.weight"))?;
    if mix.len() != (2 + c.hc) * c.hc || base.len() != mix.len() || scale.len() != 3 {
        return Err(Error::Invalid("glm_hyper_connection_shape"));
    }
    let mut input = vec![0.0; c.hidden];
    let mut post = vec![0.0; c.hc];
    for stream in 0..c.hc {
        let pre = sigmoid(mix[stream] * scale[0] + base[stream]) + c.hc_eps;
        post[stream] = 2.0 * sigmoid(mix[c.hc + stream] * scale[1] + base[c.hc + stream]);
        for d in 0..c.hidden {
            input[d] += pre * streams[stream * c.hidden + d];
        }
    }
    let logits: Vec<_> = mix[2 * c.hc..].iter().zip(&base[2 * c.hc..]).map(|(&m, &b)| m * scale[2] + b).collect();
    Ok(HcMix {
        input,
        post,
        comb: sinkhorn(&logits, c.hc, c.hc_iters, c.hc_eps)?,
    })
}

fn hc_post(
    residual: &[f32],
    output: &[f32],
    mix: &HcMix,
    hidden: usize,
    hc: usize,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.mhc.post");
    if residual.len() != hidden * hc || output.len() != hidden || mix.post.len() != hc || mix.comb.len() != hc * hc {
        return Err(Error::Invalid("glm_hyper_connection_shape"));
    }
    let mut next = vec![0.0; residual.len()];
    for dst in 0..hc {
        for d in 0..hidden {
            let mut v = mix.post[dst] * output[d];
            for src in 0..hc {
                v += mix.comb[src * hc + dst] * residual[src * hidden + d];
            }
            next[dst * hidden + d] = v;
        }
    }
    Ok(next)
}

fn convolve(
    input: &[f32],
    history: &mut [f32],
    weights: &[f32],
    kernel: usize,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.kda.convolution");
    if kernel < 2 || history.len() != input.len() * (kernel - 1) || weights.len() != input.len() * kernel {
        return Err(Error::Invalid("glm_convolution_shape"));
    }
    let mut output = vec![0.0; input.len()];
    for (channel, &x) in input.iter().enumerate() {
        let old = &mut history[channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
        let coefficients = &weights[channel * kernel..(channel + 1) * kernel];
        let v = old.iter().zip(coefficients).map(|(&a, &b)| a * b).sum::<f32>() + x * coefficients[kernel - 1];
        output[channel] = silu(v);
        old.rotate_left(1);
        old[kernel - 2] = x;
    }
    Ok(output)
}

/// The state is [value, key] within each head, matching GGML's transposed
/// storage. Forget gates are per key channel, rather than per value channel.
fn kda_recur(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: f32,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.kda.recurrent");
    let dim = q.len();
    if dim == 0 || k.len() != dim || v.len() != dim || g.len() != dim || state.len() != dim * dim {
        return Err(Error::Invalid("glm_kda_state_shape"));
    }
    let decay: Vec<_> = g.iter().map(|x| x.exp()).collect();
    let scale = 1.0 / (dim as f32).sqrt();
    let mut output = vec![0.0; dim];
    for value in 0..dim {
        let row = &mut state[value * dim..(value + 1) * dim];
        for (s, &d) in row.iter_mut().zip(&decay) {
            *s *= d;
        }
        let prediction = row.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>();
        let delta = (v[value] - prediction) * beta;
        for (s, &key) in row.iter_mut().zip(k) {
            *s += key * delta;
        }
        output[value] = row.iter().zip(q).map(|(&a, &b)| a * b).sum::<f32>() * scale;
    }
    Ok(output)
}

fn l2_normalize(x: &mut [f32]) {
    let denominator = (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    for v in x {
        *v /= denominator;
    }
}

fn kda(
    w: &Weights,
    c: &Config,
    x: &[f32],
    p: &str,
    state: &mut KdaState,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.kda");
    // These projections share an immutable input. Submit their unchanged
    // kernels together before consuming any output on the CPU.
    let [q_projection, k_projection, v_projection, f_a, g_a, beta] = project_many(
        w,
        [
            (&format!("{p}.attn_q.weight"), x),
            (&format!("{p}.attn_k.weight"), x),
            (&format!("{p}.attn_v.weight"), x),
            (&format!("{p}.ssm_f_a.weight"), x),
            (&format!("{p}.ssm_g_a.weight"), x),
            (&format!("{p}.ssm_beta.weight"), x),
        ],
    )?;
    let mut q =
        convolve(&q_projection, &mut state.q_conv, &w.vector(&format!("{p}.ssm_conv1d_q.weight"))?, c.conv_kernel)?;
    let mut k =
        convolve(&k_projection, &mut state.k_conv, &w.vector(&format!("{p}.ssm_conv1d_k.weight"))?, c.conv_kernel)?;
    let v = convolve(&v_projection, &mut state.v_conv, &w.vector(&format!("{p}.ssm_conv1d_v.weight"))?, c.conv_kernel)?;
    let [forget, gate] =
        project_many(w, [(&format!("{p}.ssm_f_b.weight"), &f_a), (&format!("{p}.ssm_g_b.weight"), &g_a)])?;
    let a = w.vector(&format!("{p}.ssm_a"))?;
    let dt = w.vector(&format!("{p}.ssm_dt.bias"))?;
    let norm_weight = w.vector(&format!("{p}.ssm_norm.weight"))?;
    let mut gated = Vec::with_capacity(q.len());
    for head in 0..c.kda_heads {
        let start = head * c.kda_dim;
        let end = start + c.kda_dim;
        l2_normalize(&mut q[start..end]);
        l2_normalize(&mut k[start..end]);
        let g: Vec<_> = (start..end).map(|i| c.decay_lower_bound * sigmoid(-a[head] * (forget[i] + dt[i]))).collect();
        let output = kda_recur(
            &mut state.matrix[head * c.kda_dim * c.kda_dim..(head + 1) * c.kda_dim * c.kda_dim],
            &q[start..end],
            &k[start..end],
            &v[start..end],
            &g,
            sigmoid(beta[head]),
        )?;
        let norm = rms_norm(&output, &norm_weight, c.rms_eps)?;
        gated.extend(norm.iter().zip(&gate[start..end]).map(|(&value, &g)| value * sigmoid(g)));
    }
    w.matvec(&format!("{p}.attn_output.weight"), &gated)
}

fn layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if x.is_empty() || x.len() != weight.len() || x.len() != bias.len() {
        return Err(Error::Invalid("glm_indexer_norm_shape"));
    }
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let denominator = (variance + eps).sqrt();
    Ok(x.iter().zip(weight).zip(bias).map(|((&v, &w), &b)| (v - mean) / denominator * w + b).collect())
}

fn pool_tail(
    state: &mut AttentionState,
    ape: &[f32],
    pool: usize,
    dim: usize,
) -> Result<()> {
    let _scope = super::profile::Scope::new("glm.mla.pool");
    if state.tail_keys.len() != pool || state.tail_gates.len() != pool || ape.len() != pool * dim {
        return Err(Error::Invalid("glm_indexer_pool_shape"));
    }
    let mut pooled = vec![0.0; dim];
    for channel in 0..dim {
        let logits: Vec<_> = (0..pool).map(|i| state.tail_gates[i][channel] + ape[i * dim + channel]).collect();
        let probabilities = softmax(&logits)?;
        pooled[channel] = (0..pool).map(|i| probabilities[i] * state.tail_keys[i][channel]).sum();
    }
    state.pool_keys.push(pooled);
    state.tail_keys.clear();
    state.tail_gates.clear();
    Ok(())
}

fn select_tokens(
    state: &AttentionState,
    queries: &[f32],
    head_weights: &[f32],
    pool: usize,
    dim: usize,
    top_k: usize,
) -> Result<Vec<usize>> {
    let _scope = super::profile::Scope::new("glm.mla.index_selection");
    if queries.len() != head_weights.len() * dim || !top_k.is_multiple_of(pool) {
        return Err(Error::Invalid("glm_indexer_query_shape"));
    }
    let scale = 1.0 / ((dim * head_weights.len()) as f32).sqrt();
    let scores: Vec<_> = state
        .pool_keys
        .iter()
        .map(|key| {
            queries
                .chunks_exact(dim)
                .zip(head_weights)
                .map(|(query, &weight)| {
                    query.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>().max(0.0) * weight * scale
                })
                .sum::<f32>()
        })
        .collect();
    let mut selected = Vec::new();
    if !scores.is_empty() {
        for group in topk(&scores, scores.len().min(top_k / pool))? {
            selected.extend(group * pool..(group + 1) * pool);
        }
    }
    let tail_start = state.pool_keys.len() * pool;
    selected.extend(tail_start..tail_start + state.tail_keys.len());
    // Attention sees a set. Temporal order makes summation deterministic and
    // matches an ordinary causal additive mask over the complete latent cache.
    selected.sort_unstable();
    Ok(selected)
}

fn mla(
    w: &Weights,
    c: &Config,
    x: &[f32],
    p: &str,
    state: &mut AttentionState,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.mla");
    let [q_a, latent, index_key, index_gate, index_weights] = project_many(
        w,
        [
            (&format!("{p}.attn_q_a.weight"), x),
            (&format!("{p}.attn_kv_a_mqa.weight"), x),
            (&format!("{p}.indexer.attn_k.weight"), x),
            (&format!("{p}.indexer_compressor_gate.weight"), x),
            (&format!("{p}.indexer.proj.weight"), x),
        ],
    )?;
    let q_residual = rms_norm(&q_a, &w.vector(&format!("{p}.attn_q_a_norm.weight"))?, c.rms_eps)?;
    let [query, index_query] = project_many(
        w,
        [
            (&format!("{p}.attn_q_b.weight"), q_residual.as_slice()),
            (&format!("{p}.indexer.attn_q_b.weight"), q_residual.as_slice()),
        ],
    )?;
    let latent = rms_norm(&latent, &w.vector(&format!("{p}.attn_kv_a_norm.weight"))?, c.rms_eps)?;
    state.latents.push(latent);

    let index_key = layer_norm(
        &index_key,
        &w.vector(&format!("{p}.indexer.k_norm.weight"))?,
        &w.vector(&format!("{p}.indexer.k_norm.bias"))?,
        1e-6,
    )?;
    state.tail_keys.push(index_key);
    state.tail_gates.push(index_gate);
    if state.tail_keys.len() == c.index_pool {
        pool_tail(state, &w.vector(&format!("{p}.indexer_compressor_ape.weight"))?, c.index_pool, c.index_dim)?;
    }
    let selected = select_tokens(state, &index_query, &index_weights, c.index_pool, c.index_dim, c.index_topk)?;
    if selected.is_empty() || state.latents.len() != state.pool_keys.len() * c.index_pool + state.tail_keys.len() {
        return Err(Error::Invalid("glm_empty_or_invalid_attention"));
    }
    let absorbed_queries = w.batched_matvec(&format!("{p}.attn_k_b.weight"), &query)?;
    let mut latent_outputs = Vec::with_capacity(c.heads * c.kv_rank);
    let scale = 1.0 / (c.qk_dim as f32).sqrt();
    let _attention_scope = super::profile::Scope::new("glm.mla.attention");
    for head in 0..c.heads {
        let absorbed = &absorbed_queries[head * c.kv_rank..(head + 1) * c.kv_rank];
        let scores: Vec<_> = selected
            .iter()
            .map(|&position| absorbed.iter().zip(&state.latents[position]).map(|(&q, &k)| q * k).sum::<f32>() * scale)
            .collect();
        let probabilities = softmax(&scores)?;
        let mut latent_output = vec![0.0; c.kv_rank];
        for (&position, &probability) in selected.iter().zip(&probabilities) {
            for (dst, &value) in latent_output.iter_mut().zip(&state.latents[position]) {
                *dst += probability * value;
            }
        }
        latent_outputs.extend(latent_output);
    }
    drop(_attention_scope);
    let output = w.batched_matvec(&format!("{p}.attn_v_b.weight"), &latent_outputs)?;
    w.matvec(&format!("{p}.attn_output.weight"), &output)
}

fn swiglu(
    gate: &[f32],
    up: &[f32],
    limit: f32,
) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(Error::Invalid("glm_swiglu_shape"));
    }
    Ok(gate.iter().zip(up).map(|(&g, &u)| silu(g.min(limit)) * u.clamp(-limit, limit)).collect())
}

fn project_many<const N: usize>(
    w: &Weights,
    requests: [(&str, &[f32]); N],
) -> Result<[Vec<f32>; N]> {
    w.matvec_many(&requests)?.try_into().map_err(|_| Error::Invalid("glm_projection_batch_size"))
}

fn dense_ffn(
    w: &Weights,
    x: &[f32],
    p: &str,
    suffix: &str,
    limit: f32,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.ffn.dense");
    let [gate, up] =
        project_many(w, [(&format!("{p}.ffn_gate{suffix}.weight"), x), (&format!("{p}.ffn_up{suffix}.weight"), x)])?;
    w.matvec(&format!("{p}.ffn_down{suffix}.weight"), &swiglu(&gate, &up, limit)?)
}

fn route(
    logits: &[f32],
    bias: &[f32],
    active: usize,
    scale: f32,
) -> Result<Vec<(usize, f32)>> {
    if logits.len() != bias.len() {
        return Err(Error::Invalid("glm_router_shape"));
    }
    let probabilities: Vec<_> = logits.iter().map(|&v| sigmoid(v)).collect();
    let choices: Vec<_> = probabilities.iter().zip(bias).map(|(&a, &b)| a + b).collect();
    let selected = topk(&choices, active)?;
    let denominator = selected.iter().map(|&i| probabilities[i]).sum::<f32>() + 1e-20;
    Ok(selected.into_iter().map(|i| (i, probabilities[i] / denominator * scale)).collect())
}

fn moe(
    w: &Weights,
    c: &Config,
    x: &[f32],
    p: &str,
) -> Result<Vec<f32>> {
    let _scope = super::profile::Scope::new("glm.ffn.moe");
    let [logits, shared_gate, shared_up] = project_many(
        w,
        [
            (&format!("{p}.ffn_gate_inp.weight"), x),
            (&format!("{p}.ffn_gate_shexp.weight"), x),
            (&format!("{p}.ffn_up_shexp.weight"), x),
        ],
    )?;
    let bias = w.vector(&format!("{p}.exp_probs_b.bias"))?;
    let selected = route(&logits, &bias, c.active_experts, c.routed_scale)?;
    let experts: Vec<_> = selected.iter().map(|&(index, _)| index).collect();
    let inputs = x.repeat(experts.len());
    let gate = w.selected_matvec(&format!("{p}.ffn_gate_exps.weight"), &experts, &inputs)?;
    let up = w.selected_matvec(&format!("{p}.ffn_up_exps.weight"), &experts, &inputs)?;
    let expert_outputs =
        w.selected_matvec(&format!("{p}.ffn_down_exps.weight"), &experts, &swiglu(&gate, &up, c.swiglu_limit)?)?;
    if expert_outputs.len() != c.hidden * selected.len() {
        return Err(Error::Invalid("glm_expert_output_shape"));
    }
    let mut output = vec![0.0; c.hidden];
    for ((_, probability), expert_output) in selected.iter().zip(expert_outputs.chunks_exact(c.hidden)) {
        for (dst, &src) in output.iter_mut().zip(expert_output) {
            *dst += probability * src;
        }
    }
    let shared = w.matvec(&format!("{p}.ffn_down_shexp.weight"), &swiglu(&shared_gate, &shared_up, c.swiglu_limit)?)?;
    if shared.len() != output.len() {
        return Err(Error::Invalid("glm_shared_expert_shape"));
    }
    for (dst, &src) in output.iter_mut().zip(&shared) {
        *dst += src;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn close(
        actual: &[f32],
        expected: &[f32],
        tolerance: f32,
    ) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().zip(expected).all(|(&a, &b)| (a - b).abs() < tolerance));
    }

    #[stark_test]
    fn kda_decays_key_channels_before_delta_update() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        let result =
            kda_recur(&mut state, &[1.0, 0.0], &[0.0, 1.0], &[2.0, 4.0], &[0.5f32.ln(), 0.25f32.ln()], 0.5).unwrap();
        close(&state, &[0.5, 1.25, 1.5, 2.5], 1e-6);
        close(&result, &[0.5 / 2f32.sqrt(), 1.5 / 2f32.sqrt()], 1e-6);
    }

    #[stark_test]
    fn convolution_uses_oldest_to_newest_coefficients_per_channel() {
        let mut history = vec![0.0; 4];
        let coefficients = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        close(&convolve(&[1.0, 2.0], &mut history, &coefficients, 3).unwrap(), &[silu(3.0), silu(-6.0)], 1e-6);
        close(&history, &[0.0, 1.0, 0.0, 2.0], 1e-6);
        close(&convolve(&[4.0, 5.0], &mut history, &coefficients, 3).unwrap(), &[silu(14.0), silu(-19.0)], 1e-6);
        close(&history, &[1.0, 4.0, 2.0, 5.0], 1e-6);
    }

    #[stark_test]
    fn sinkhorn_normalizes_both_axes_and_post_uses_source_destination_order() {
        let comb = sinkhorn(&[3.0, 0.0, -2.0, 1.0], 2, 40, 1e-6).unwrap();
        for row in comb.chunks_exact(2) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 2e-5);
        }
        for dst in 0..2 {
            assert!((comb[dst] + comb[2 + dst] - 1.0).abs() < 2e-5);
        }
        let mix = HcMix {
            input: vec![],
            post: vec![2.0, 3.0],
            comb: vec![0.1, 0.2, 0.3, 0.4],
        };
        close(&hc_post(&[10.0, 20.0], &[5.0], &mix, 1, 2).unwrap(), &[17.0, 25.0], 1e-6);
    }

    #[stark_test]
    fn router_bias_changes_selection_without_changing_combine_probabilities() {
        let routed = route(&[0.0, 1.0, 2.0], &[2.0, 0.0, 0.0], 2, 2.5).unwrap();
        assert_eq!(routed.iter().map(|v| v.0).collect::<Vec<_>>(), vec![0, 2]);
        let denominator = 0.5 + sigmoid(2.0);
        close(
            &routed.iter().map(|v| v.1).collect::<Vec<_>>(),
            &[0.5 / denominator * 2.5, sigmoid(2.0) / denominator * 2.5],
            1e-6,
        );
    }

    #[stark_test]
    fn sparse_indexer_selects_complete_pools_and_always_includes_visible_tail() {
        let state = AttentionState {
            latents: vec![],
            tail_keys: vec![vec![0.0]],
            tail_gates: vec![vec![0.0]],
            pool_keys: vec![vec![3.0], vec![1.0], vec![2.0]],
        };
        assert_eq!(select_tokens(&state, &[1.0], &[1.0], 2, 1, 4).unwrap(), vec![0, 1, 4, 5, 6]);
        let mut state = AttentionState {
            latents: vec![],
            tail_keys: vec![vec![2.0], vec![8.0]],
            tail_gates: vec![vec![0.0], vec![0.0]],
            pool_keys: vec![],
        };
        pool_tail(&mut state, &[0.0, 0.0], 2, 1).unwrap();
        close(&state.pool_keys[0], &[5.0], 1e-6);
        assert!(state.tail_keys.is_empty());
    }

    #[stark_test]
    fn sparse_indexer_zero_score_ties_choose_earliest_pools_and_keep_tail() {
        let state = AttentionState {
            latents: vec![],
            tail_keys: vec![vec![0.0]],
            tail_gates: vec![vec![0.0]],
            // Every query-key dot product is negative, so ReLU gives each
            // complete pool the same zero score at the selection boundary.
            pool_keys: vec![vec![-3.0], vec![-2.0], vec![-1.0]],
        };
        let selected = select_tokens(&state, &[1.0], &[1.0], 2, 1, 4).unwrap();
        assert_eq!(selected, vec![0, 1, 2, 3, 6]);
        assert_eq!(select_tokens(&state, &[1.0], &[1.0], 2, 1, 4).unwrap(), selected);
    }

    #[stark_test]
    fn four_token_pool_uses_channel_weights_and_preserves_incomplete_tail() {
        let mut state = AttentionState {
            latents: vec![],
            tail_keys: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]],
            tail_gates: vec![
                vec![0.0, 0.0],
                vec![2.0f32.ln(), 3.0f32.ln()],
                vec![3.0f32.ln(), 2.0f32.ln()],
                vec![4.0f32.ln(), 0.0],
            ],
            pool_keys: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        // Gates and the learned positional bias give probabilities
        // [1,2,3,4]/10 in channel zero and [4,3,2,1]/10 in channel one.
        pool_tail(&mut state, &[0.0, 4.0f32.ln(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, 2).unwrap();
        close(&state.pool_keys[2], &[5.0, 4.0], 1e-6);
        assert!(state.tail_keys.is_empty() && state.tail_gates.is_empty());
        state.tail_keys = vec![vec![0.0, 0.0]; 3];
        state.tail_gates = vec![vec![0.0, 0.0]; 3];
        // One complete selected pool plus the three currently visible tail
        // tokens. The exact set excludes duplicate and future positions.
        assert_eq!(select_tokens(&state, &[1.0, 1.0], &[1.0], 4, 2, 4).unwrap(), vec![8, 9, 10, 11, 12, 13, 14]);
    }

    #[stark_test]
    fn swiglu_clips_raw_gate_before_activation() {
        close(&swiglu(&[20.0, -20.0], &[30.0, -30.0], 10.0).unwrap(), &[silu(10.0) * 10.0, silu(-20.0) * -10.0], 1e-6);
    }

    /// The source-bound synthetic fixture is deliberately private. Its oracle
    /// executes the pinned upstream Transformers modules on complete causal
    /// sequences, exercising both sparse prefill and native incremental state.
    #[stark_test]
    #[ignore]
    fn glm_complete_forward_matches_upstream_reference() {
        let path = std::env::var_os("STARK_GLM_REFERENCE_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| panic!("GLM reference fixture path is required"));
        let read = |name: &str| -> Value {
            let bytes = std::fs::read(path.join(name)).unwrap_or_else(|_| panic!("private GLM fixture read failed"));
            serde_json::from_slice(&bytes).unwrap_or_else(|_| panic!("private GLM fixture metadata is invalid"))
        };
        let config = read("config.json");
        let reference = read("reference.json");
        assert!(
            reference["source"]["revision"].as_str() == Some("95bafdcccfffe17ec383999fc39273867b11385e")
                && reference["source"]["sha256"].as_str()
                    == Some("7870e7ce6ca477d26de3ec79b31d41553b75a09a726c661f04c2e26e790bdf36"),
            "GLM oracle source identity mismatch"
        );
        let weights = Weights::open(&path).unwrap_or_else(|_| panic!("private GLM fixture weights are invalid"));
        let mut model =
            GlmModel::new(weights, &config).unwrap_or_else(|_| panic!("private GLM fixture model admission failed"));
        assert!(!model.speculative_decoding_supported());
        let cases = reference["cases"].as_array().unwrap_or_else(|| panic!("private GLM fixture cases are invalid"));
        let mut positions = 0;
        let mut maximum_error = 0.0f64;
        let mut squared_error = 0.0f64;
        let mut squared_reference = 0.0f64;
        let mut layer_errors = vec![0.0f64; model.config.layer_types.len()];
        let mut first_mismatch = None;
        for _ in 0..2 {
            for case in cases {
                model.reset();
                let tokens = case["input_token_ids"].as_array().unwrap();
                let expected = case["logits"].as_array().unwrap();
                let mut complete_readouts = Vec::with_capacity(tokens.len());
                assert!(tokens.len() == expected.len(), "GLM reference sequence length mismatch");
                for (position, (token, row)) in tokens.iter().zip(expected).enumerate() {
                    let actual = model
                        .step(token.as_u64().unwrap() as u32)
                        .unwrap_or_else(|_| panic!("GLM reference forward failed"));
                    let row = row.as_array().unwrap();
                    assert!(actual.len() == row.len(), "GLM reference vocabulary mismatch");
                    for (&a, e) in actual.iter().zip(row) {
                        let e = e.as_f64().unwrap();
                        let difference = f64::from(a) - e;
                        maximum_error = maximum_error.max(difference.abs());
                        squared_error += difference * difference;
                        squared_reference += e * e;
                        if difference.abs() > 1e-4 && first_mismatch.is_none() {
                            first_mismatch = Some(positions);
                        }
                    }
                    if let Some(reference_layers) = case["layer_outputs"].as_array() {
                        for (layer, (actual_layer, expected_layer)) in
                            model.layer_trace.iter().zip(reference_layers).enumerate()
                        {
                            let expected_layer = expected_layer[position].as_array().unwrap();
                            assert!(
                                actual_layer.len() == expected_layer.len(),
                                "GLM layer reference dimensions differ"
                            );
                            for (&a, e) in actual_layer.iter().zip(expected_layer) {
                                layer_errors[layer] =
                                    layer_errors[layer].max((f64::from(a) - e.as_f64().unwrap()).abs());
                            }
                        }
                    }
                    complete_readouts.push(actual);
                    positions += 1;
                }
                // Skipping intermediate readouts must leave every recurrent,
                // convolution, pooled-indexer and attention-cache state exact.
                let mut prefix_lengths = vec![1, 3.min(tokens.len().saturating_sub(1)), tokens.len().saturating_sub(1)];
                prefix_lengths.sort_unstable();
                prefix_lengths.dedup();
                for prefix in prefix_lengths {
                    if prefix == 0 || prefix >= tokens.len() {
                        continue;
                    }
                    model.reset();
                    for token in &tokens[..prefix] {
                        assert!(model.ingest(token.as_u64().unwrap() as u32).is_ok(), "GLM prefill ingest failed");
                    }
                    assert_eq!(model.position, prefix, "GLM ingest did not advance position");
                    let actual = model
                        .step(tokens[prefix].as_u64().unwrap() as u32)
                        .unwrap_or_else(|_| panic!("GLM readout after ingest failed"));
                    assert!(actual == complete_readouts[prefix], "GLM ingest changed subsequent readout");
                }
                assert!(model.ingest(u32::MAX).is_err(), "invalid GLM ingest token was accepted");
                let recovered = model
                    .step(tokens[0].as_u64().unwrap() as u32)
                    .unwrap_or_else(|_| panic!("GLM ingest failure reset replay failed"));
                assert!(recovered == complete_readouts[0], "GLM ingest failure did not clear state");
                if tokens.len() >= 2 {
                    let original_context = model.config.max_context;
                    model.config.max_context = 2;
                    model.reset();
                    assert!(model.ingest(tokens[0].as_u64().unwrap() as u32).is_ok(), "GLM bounded ingest failed");
                    let final_readout = model
                        .step(tokens[1].as_u64().unwrap() as u32)
                        .unwrap_or_else(|_| panic!("GLM final allowed context readout failed"));
                    assert!(final_readout == complete_readouts[1], "GLM bounded ingest changed readout");
                    assert!(model.ingest(tokens[0].as_u64().unwrap() as u32).is_err(), "GLM ingest exceeded context");
                    let recovered = model
                        .step(tokens[0].as_u64().unwrap() as u32)
                        .unwrap_or_else(|_| panic!("GLM context failure reset replay failed"));
                    assert!(recovered == complete_readouts[0], "GLM context failure did not clear state");
                    model.config.max_context = original_context;
                }
                assert!(model.step(u32::MAX).is_err(), "invalid GLM token was accepted");
                let first = model
                    .step(tokens[0].as_u64().unwrap() as u32)
                    .unwrap_or_else(|_| panic!("GLM failure reset replay failed"));
                let expected_first = expected[0].as_array().unwrap();
                assert!(
                    first.iter().zip(expected_first).all(|(&a, e)| (f64::from(a) - e.as_f64().unwrap()).abs() <= 1e-4),
                    "GLM failure reset differs from a fresh sequence"
                );
            }
        }
        let relative_rms_error = (squared_error / squared_reference).sqrt();
        println!(
            "{}",
            serde_json::json!({
                "glm_reference_positions": positions,
                "maximum_absolute_error": maximum_error,
                "relative_rms_error": relative_rms_error,
                "first_mismatch_position": first_mismatch,
                "layer_maximum_errors": layer_errors,
            })
        );
        assert!(maximum_error <= 1e-4 && relative_rms_error <= 1e-4, "GLM complete forward reference gate failed");
    }

    /// This checks only production source headers and admission. Sparse stub
    /// payloads may be supplied; this test never performs a forward operation.
    #[stark_test]
    #[ignore]
    fn glm_production_header_admission_only() {
        let path = std::env::var_os("STARK_GLM_HEADER_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| panic!("GLM header fixture path is required"));
        let config: Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json"))
                .unwrap_or_else(|_| panic!("private GLM header config read failed")),
        )
        .unwrap_or_else(|_| panic!("private GLM header config is invalid"));
        let model = GlmModel::new(
            Weights::open(&path).unwrap_or_else(|_| panic!("production GLM header admission failed")),
            &config,
        )
        .unwrap_or_else(|_| panic!("production GLM configuration admission failed"));
        assert!(model.vocab_size() == 154880 && model.max_context_length() == 1048576);
        assert!(!model.speculative_decoding_supported());
        drop(model);
        for (name, value) in [("index_topk", Value::from(1024)), ("hc_eps", Value::from(0.001))] {
            let mut changed = config.clone();
            changed["text_config"][name] = value;
            let weights = Weights::open(&path).unwrap_or_else(|_| panic!("production GLM header reopening failed"));
            assert!(GlmModel::new(weights, &changed).is_err(), "source metadata disagreement was accepted");
        }
        println!(
            "{}",
            serde_json::json!({"glm_production_header_admission": true, "metadata_disagreements_rejected": 2, "forward_executed": false})
        );
    }
}
