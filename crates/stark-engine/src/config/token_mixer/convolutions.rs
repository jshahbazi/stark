use stark_engine_macros::stark_config;

#[stark_config]
pub struct SeparableCausalConvConfig {
    pub has_biases: bool,
}
