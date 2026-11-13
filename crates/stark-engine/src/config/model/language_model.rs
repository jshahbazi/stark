use stark_engine_macros::stark_config;

use crate::config::{decoder::DecoderConfig, model::generation::GenerationConfig};

#[stark_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    #[serde(default)]
    pub mtp_config: Option<crate::config::qwen_mtp::QwenMtpConfig>,
}
