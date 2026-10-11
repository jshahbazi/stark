use stark_engine_macros::stark_config;

use crate::config::classifier::ClassifierConfig;

#[stark_config(super::ModelConfig)]
pub struct ClassifierModelConfig {
    pub classifier_config: ClassifierConfig,
}
