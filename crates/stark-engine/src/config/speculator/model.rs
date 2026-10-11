use stark_engine_macros::stark_config;

use crate::config::speculator::AnySpeculatorConfig;

#[stark_config(super::SpeculatorConfig)]
pub struct SpeculatorModelConfig {
    pub speculator_config: AnySpeculatorConfig,
}
