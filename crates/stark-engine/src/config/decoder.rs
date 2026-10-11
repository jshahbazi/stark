use stark_engine_macros::stark_config;

use crate::config::{
    embedding::AnyEmbeddingConfig, normalization::NormalizationConfig, per_layer_embedding::PLEModelConfig,
    transformer::TransformerConfig,
};

#[stark_config]
pub struct DecoderConfig {
    pub embedding_config: AnyEmbeddingConfig,
    pub transformer_config: TransformerConfig,

    pub vocab_size: u32,
    pub ple_model_config: Option<PLEModelConfig>,
    pub embedding_norm_config: Option<NormalizationConfig>,
}
