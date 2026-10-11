use stark_engine_macros::stark_config;

use crate::config::{normalization::NormalizationConfig, transformer_layer::TransformerLayerConfig};

#[stark_config]
pub struct TransformerConfig {
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: u32,
    pub hidden_dim: u32,
}
