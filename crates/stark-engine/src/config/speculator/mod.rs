use stark_engine_macros::stark_config_abstract;

pub mod dflash;
pub mod model;

#[stark_config_abstract(dflash::DFlashSpeculatorConfig)]
pub struct SpeculatorConfig;
