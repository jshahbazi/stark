use stark_engine_macros::stark_config;

#[stark_config(super::RoPEConfig)]
pub struct LlamaRoPEConfig {
    pub scaling_factor: f32,
    pub original_context_length: u32,
    pub low_frequency_factor: f32,
    pub high_frequency_factor: f32,
}
