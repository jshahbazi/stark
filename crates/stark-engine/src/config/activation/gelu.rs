use stark_engine_macros::stark_config;

#[stark_config(super::Activation)]
pub struct GELU {
    pub approximate: bool,
}
