use stark_engine_macros::stark_config_abstract;

pub mod tied_embedding;
pub mod untied_embedding;

#[stark_config_abstract(tied_embedding::TiedEmbeddingConfig, untied_embedding::UntiedEmbeddingConfig)]
pub struct EmbeddingConfig {
    pub input_scale: Option<f32>,
    pub logit_soft_cap: Option<f32>,
    pub logit_scale: Option<f32>,
}
