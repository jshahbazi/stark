use stark_engine_macros::stark_config_abstract;

use crate::config::token_codec::AnyTokenCodecConfig;

pub mod classifier_model;
pub mod generation;
pub mod language_model;
pub mod qwen38_language_model;

#[stark_config_abstract(
    language_model::LanguageModelConfig,
    classifier_model::ClassifierModelConfig,
    qwen38_language_model::Qwen38LanguageModelConfig
)]
pub struct ModelConfig {
    pub token_codec_config: AnyTokenCodecConfig,
}
