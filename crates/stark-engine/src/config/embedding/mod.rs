use stark_engine_macros::stark_config_abstract;

pub mod tied_embedding;
pub mod untied_embedding;

#[stark_config_abstract(tied_embedding::TiedEmbeddingConfig, untied_embedding::UntiedEmbeddingConfig)]
pub struct EmbeddingConfig {
    pub input_scale: Option<f32>,
    pub logit_soft_cap: Option<f32>,
    pub logit_scale: Option<f32>,
    /// Round dequantized values to the activation dtype before input scaling.
    #[serde(default)]
    pub materialize_dequant: bool,
    /// Materialize the activation dtype after soft-cap division and tanh.
    #[serde(default)]
    pub materialize_soft_cap: bool,
}
