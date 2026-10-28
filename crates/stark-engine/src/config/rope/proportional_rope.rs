use stark_engine_macros::stark_config;

/// Gemma 4 rotates a prefix of frequency pairs across the complete split-half
/// head. The frequency denominator remains the complete head width.
#[stark_config(super::RoPEConfig)]
pub struct ProportionalRoPEConfig {
    pub rotary_dim: u32,
}
