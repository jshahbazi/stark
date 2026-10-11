use stark_engine_macros::stark_config;

#[stark_config]
#[serde(rename_all = "snake_case")]
pub enum UpcastMode {
    OnlyNormalization,
    FullLayer,
}

#[stark_config]
pub struct NormalizationConfig {
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    pub subtract_mean: bool,
    pub has_scale: bool,
    pub has_biases: bool,
}
