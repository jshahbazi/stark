use stark_engine_macros::stark_config;

use crate::config::{mlp::dense_mlp::DenseMLPConfig, normalization::NormalizationConfig};

#[stark_config(super::MLPConfig)]
pub struct Gemma4MoeConfig {
    pub expert_config: DenseMLPConfig,
    pub norm_config: NormalizationConfig,
    pub num_experts: u32,
    pub active_experts: u32,
    pub expert_hidden_dim: u32,
    pub router_root_scale: f32,
}
