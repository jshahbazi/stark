use stark_engine_macros::stark_config;

#[stark_config(super::RoPEConfig)]
pub struct LinearScalingRoPEConfig {
    pub scaling_factor: f32,
}
