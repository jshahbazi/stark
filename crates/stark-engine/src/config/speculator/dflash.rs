use stark_engine_macros::stark_config;

use crate::config::{dflash::DFlashDraftConfig, weaver::WeaverConfig};

#[stark_config(super::SpeculatorConfig)]
pub struct DFlashSpeculatorConfig {
    pub draft_config: DFlashDraftConfig,
    pub weaver_config: Option<WeaverConfig>,
}
