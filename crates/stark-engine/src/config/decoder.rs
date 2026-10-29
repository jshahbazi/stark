use stark_engine_macros::stark_config;

use crate::config::{
    embedding::AnyEmbeddingConfig, normalization::NormalizationConfig, per_layer_embedding::PLEModelConfig,
    transformer::TransformerConfig,
};

#[stark_config]
pub struct DecoderConfig {
    /// Admit Gemma 4 image embeddings with ordinary sequential text RoPE.
    /// Only a validated native Gemma image artifact sets this flag.
    #[serde(default)]
    pub gemma4_image_embeddings: bool,
    /// Use a distinct single-row projection for the final prompt token.
    #[serde(default)]
    pub separate_prefill_last_token: bool,
    pub embedding_config: AnyEmbeddingConfig,
    pub transformer_config: TransformerConfig,

    pub vocab_size: u32,
    pub ple_model_config: Option<PLEModelConfig>,
    pub embedding_norm_config: Option<NormalizationConfig>,
}
