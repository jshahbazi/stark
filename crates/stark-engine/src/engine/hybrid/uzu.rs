//! Admission for the typed Uzu Qwen3.8 architecture extension.

use std::{fs::File, io::Read, path::Path};

use super::{Error, Result};
use crate::config::model::qwen38_language_model::{Qwen38ArchitectureConfig, Qwen38LanguageModelConfig};

pub(super) const CONFIG_TYPE: &str = "Qwen38LanguageModelConfig";

pub(super) fn read_config(path: &Path) -> Result<Qwen38LanguageModelConfig> {
    let file = File::open(path.join("config.json"))?;
    let limit = 8 * 1024 * 1024;
    if !file.metadata()?.is_file() || file.metadata()?.len() > limit {
        return Err(Error::Invalid("Uzu Qwen configuration size"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(Error::Invalid("Uzu Qwen configuration size"));
    }
    let config: Qwen38LanguageModelConfig = serde_json::from_slice(&bytes)?;
    validate_architecture(&config.decoder_config.architecture)?;
    Ok(config)
}

fn validate_architecture(a: &Qwen38ArchitectureConfig) -> Result<()> {
    let dimensions = [
        a.hidden_size,
        a.num_hidden_layers,
        a.vocab_size,
        a.max_position_embeddings,
        a.num_experts,
        a.num_experts_per_tok,
        a.moe_intermediate_size,
        a.shared_expert_intermediate_size,
        a.num_attention_heads,
        a.num_key_value_heads,
        a.head_dim,
        a.linear_num_key_heads,
        a.linear_num_value_heads,
        a.linear_key_head_dim,
        a.linear_value_head_dim,
        a.linear_conv_kernel_dim,
        a.hc_count,
        a.hc_lowrank,
        a.ple_conv_kernel_size,
        a.ngram_size,
        a.heads_per_ngram,
        a.split_ngram_parts,
        a.indexer_n_heads,
        a.indexer_kv_heads,
        a.indexer_head_dim,
        a.indexer_budget,
        a.indexer_compress_ratio,
    ];
    if dimensions.iter().any(|&value| value == 0 || value > 1_048_576)
        || a.num_hidden_layers > 256
        || a.vocab_size > 1_000_000
        || a.layer_types.len() != a.num_hidden_layers as usize
        || a.tie_word_embeddings
        || a.hidden_act != "silu"
        || !a.rms_norm_eps.is_finite()
        || a.rms_norm_eps <= 0.0
        || a.num_experts_per_tok > a.num_experts
        || !a.num_attention_heads.is_multiple_of(a.num_key_value_heads)
        || !a.linear_num_value_heads.is_multiple_of(a.linear_num_key_heads)
        || a.linear_conv_kernel_dim < 2
        || a.ple_conv_kernel_size < 2
        || a.ngram_size < 2
        || a.ngram_size > 16
        || a.heads_per_ngram > 256
        || a.ple_layer_ids.len() > 1
        || a.ple_layer_ids.iter().any(|&layer| layer == 0 || layer > a.num_hidden_layers)
        || a.eos_token_id.is_empty()
        || a.eos_token_id.len() > 32
        || a.eos_token_id.iter().any(|&id| id >= a.vocab_size)
        || a.ple_embed_dim.is_some_and(|value| value == 0 || value > 1_048_576)
        || !a.indexer_budget.is_multiple_of(a.indexer_compress_ratio)
    {
        return Err(Error::Invalid("Uzu Qwen architecture geometry"));
    }
    let rope = &a.rope_parameters;
    let rotary = a.head_dim as f32 * rope.partial_rotary_factor;
    if !rope.rope_theta.is_finite()
        || rope.rope_theta <= 0.0
        || !rope.partial_rotary_factor.is_finite()
        || rope.partial_rotary_factor <= 0.0
        || rope.partial_rotary_factor > 1.0
        || rotary.fract() != 0.0
        || !(rotary as u32).is_multiple_of(2)
        || rope.rope_type.as_deref().is_some_and(|ty| ty != "default")
    {
        return Err(Error::Invalid("Uzu Qwen rotary geometry"));
    }
    let heads = ((a.ngram_size - 1) * a.heads_per_ngram) as usize;
    let ple = &a.ple;
    if !a.ple_layer_ids.is_empty()
        && (ple.layer_multipliers.len() != a.ngram_size as usize
            || ple.ngram_heads_vocab_sizes.len() != heads
            || ple.ngram_heads_offsets.len() != heads
            || ple.padded_vocab_size == 0
            || !ple.padded_vocab_size.is_multiple_of(a.split_ngram_parts)
            || !(a.ple_embed_dim.unwrap_or(a.hidden_size) as usize).is_multiple_of(heads)
            || ple.ngram_heads_vocab_sizes.iter().zip(ple.ngram_heads_offsets.iter()).any(|(&size, &offset)| {
                size == 0 || offset.checked_add(size).is_none_or(|end| end > ple.padded_vocab_size as u64)
            }))
    {
        return Err(Error::Invalid("Uzu Qwen PLE geometry"));
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) fn core_config(
    config: &Qwen38ArchitectureConfig
) -> Result<stark_lily_core::qwen4exp::config::Qwen4ExpConfig> {
    validate_architecture(config)?;
    let mut text = serde_json::to_value(config)?;
    let object = text.as_object_mut().ok_or(Error::Invalid("Uzu Qwen text configuration"))?;
    let ple = object.remove("ple").ok_or(Error::Invalid("Uzu Qwen PLE configuration"))?;
    let quantization = object.remove("quantization").ok_or(Error::Invalid("Uzu Qwen quantization configuration"))?;
    // This is an internal adapter to the pinned architecture validator, built
    // solely from typed Uzu values. It is never an executable HF config file.
    let adapter = serde_json::json!({"model_type":"qwen4_exp", "text_config":text,
        "lily":{"format":"qwen4_exp-affine-v1", "quantization":quantization, "ple":ple}});
    stark_lily_core::qwen4exp::config::Qwen4ExpConfig::from_json(&serde_json::to_vec(&adapter)?)
        .map_err(|_| Error::Invalid("Uzu Qwen unsupported architecture"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) fn checkpoint(
    path: &Path,
    config: &crate::config::model::qwen38_language_model::Qwen38DecoderConfig,
) -> Result<stark_lily_core::safetensors::Checkpoint> {
    // Typed schema and the vendored storage admission share exactly the same
    // small binding record. No tensors or numeric parameters pass through JSON.
    let bindings = serde_json::from_value(serde_json::to_value(&config.tensor_bindings)?)?;
    stark_lily_core::safetensors::Checkpoint::open_uzu(path, &bindings)
        .map_err(|_| Error::Invalid("Uzu Qwen tensor storage admission failed"))
}
