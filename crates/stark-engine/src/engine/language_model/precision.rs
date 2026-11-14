//! Activation precision admission for ordinary text decoders.
//!
//! BF16 remains the established policy for every artifact. F16 activations are
//! selected only when the decoder stores F16 tensors and no BF16 tensor (see
//! `ParameterLoader::decoder_activation_data_type`), and are admitted only for
//! the plain dense attention/MLP graph used by Llama, Qwen2 and Mistral
//! artifacts. The F16 policy keeps weights and MLX affine scales/biases at
//! their F16 storage, runs F16 activations and KV cache, and accumulates
//! matrix products, normalization and attention in F32 like the BF16 path.
//! Other operators (query/key norms, sinks, soft caps, sliding windows, PLE,
//! image embeddings, MTP heads and speculators) have no qualified F16 kernels
//! and fail closed.

use crate::{
    config::{
        activation::AnyActivation, embedding::AnyEmbeddingConfig, mlp::AnyMLPConfig,
        model::language_model::LanguageModelConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig,
        token_mixer::AnyTokenMixerConfig,
    },
    data_type::DataType,
};

fn plain_norm(config: &NormalizationConfig) -> bool {
    !config.mlx_bf16_reduction
}

/// Whether the configured graph is inside the F16 dense activation profile.
pub(crate) fn f16_dense_admitted(config: &LanguageModelConfig) -> bool {
    let decoder = &config.decoder_config;
    if config.mtp_config.is_some()
        || decoder.gemma4_image_embeddings
        || decoder.scalar_image_embeddings
        || decoder.bidirectional_image_attention
        || decoder.separate_prefill_last_token
        || decoder.ple_model_config.is_some()
        || decoder.embedding_norm_config.is_some()
    {
        return false;
    }
    let embedding_plain = |input_scale: &Option<f32>,
                           soft_cap: &Option<f32>,
                           logit_scale: &Option<f32>,
                           materialize_dequant: bool,
                           materialize_soft_cap: bool| {
        input_scale.is_none()
            && soft_cap.is_none()
            && logit_scale.is_none()
            && !materialize_dequant
            && !materialize_soft_cap
    };
    let embedding = &decoder.embedding_config;
    if !matches!(embedding, AnyEmbeddingConfig::TiedEmbeddingConfig(_) | AnyEmbeddingConfig::UntiedEmbeddingConfig(_))
        || !embedding_plain(
            embedding.input_scale(),
            embedding.logit_soft_cap(),
            embedding.logit_scale(),
            *embedding.materialize_dequant(),
            *embedding.materialize_soft_cap(),
        )
    {
        return false;
    }
    let transformer = &decoder.transformer_config;
    if !plain_norm(&transformer.output_norm_config) || transformer.layer_configs.is_empty() {
        return false;
    }
    transformer.layer_configs.iter().all(|layer| {
        let AnyTokenMixerConfig::AttentionConfig(attention) = &layer.mixer_config else {
            return false;
        };
        let AnyMLPConfig::DenseMLPConfig(mlp) = &layer.mlp_config else {
            return false;
        };
        let attention_plain = attention.query_norm_config.is_none()
            && attention.key_norm_config.is_none()
            && !attention.qk_norm_full_width
            && attention.is_causal
            && attention.sliding_window_size.is_none()
            && attention.logit_soft_cap.is_none()
            && !attention.has_sinks
            && !attention.has_gate
            && !attention.normalize_values
            && !attention.is_kv_sharing
            && !attention.reference_bf16
            && !attention.materialize_rope_bf16
            && !attention.materialize_prefill_scores;
        let mlp_plain = matches!(&mlp.activation, AnyActivation::SiLU(silu) if silu.alpha == 1.0 && !silu.materialize_sigmoid)
            && mlp.gate_clipping.into_pair().is_none()
            && mlp.up_clipping.into_pair().is_none();
        let rope_plain = matches!(
            layer.rope_config,
            Some(AnyRoPEConfig::UnscaledRoPEConfig(_) | AnyRoPEConfig::LlamaRoPEConfig(_))
        );
        attention_plain
            && mlp_plain
            && rope_plain
            && !layer.post_norm
            && layer.pre_mixer_norm_config.as_ref().is_some_and(plain_norm)
            && plain_norm(&layer.pre_mlp_norm_config)
            && layer.mixer_conv_config.is_none()
            && layer.mlp_conv_config.is_none()
            && layer.post_mixer_norm_config.is_none()
            && layer.post_mlp_norm_config.is_none()
            && layer.ple_config.is_none()
            && !layer.has_post_layer_scalar
            && !layer.materialize_post_layer_scalar
            && layer.kv_source_layer_index.is_none()
    })
}

/// Admission result for a selected activation dtype. BF16 is never changed.
pub(crate) fn activation_precision_admitted(
    data_type: DataType,
    config: &LanguageModelConfig,
    has_speculator: bool,
) -> bool {
    match data_type {
        DataType::BF16 => true,
        DataType::F16 => !has_speculator && f16_dense_admitted(config),
        _ => false,
    }
}
