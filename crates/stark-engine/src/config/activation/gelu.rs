use stark_engine_macros::stark_config;

#[stark_config(super::Activation)]
pub struct GELU {
    pub approximate: bool,
    /// Preserve the BF16 tensor primitives of the MLX approximate polynomial.
    #[serde(default)]
    pub materialize_polynomial: bool,
}
