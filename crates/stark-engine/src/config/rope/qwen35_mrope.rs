use stark_engine_macros::stark_config;

use crate::backends::common::InvalidInput;

/// Qwen3.5 text-side interleaved temporal/height/width rotary coefficients.
/// `head_dim` is the rotated prefix width, which can be smaller than an attention head.
#[stark_config(super::RoPEConfig)]
pub struct Qwen35MRoPEConfig {
    pub mrope_section: [u32; 3],
    pub interleaved: bool,
}

impl Qwen35MRoPEConfig {
    pub fn validate(&self) -> Result<(), InvalidInput> {
        if !self.base.is_finite() || self.base <= 0.0 {
            return Err(InvalidInput("M-RoPE base must be positive and finite"));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) || self.max_sequence_length == 0 {
            return Err(InvalidInput("M-RoPE width must be positive and even with nonzero context capacity"));
        }
        if !self.interleaved {
            return Err(InvalidInput("native Qwen3.5 M-RoPE requires interleaved sections"));
        }
        let half_width = self.head_dim / 2;
        let section_sum = self
            .mrope_section
            .iter()
            .try_fold(0u32, |sum, section| sum.checked_add(*section))
            .ok_or(InvalidInput("M-RoPE section sum overflow"))?;
        if section_sum != half_width {
            return Err(InvalidInput("M-RoPE sections must sum to half the rotary width"));
        }
        // Python's slice(offset, section * 3, 3) truncates silently at the
        // table boundary. Require every declared H/W coefficient to fit.
        // The exclusive end may exceed half_width: [11,11,10]/64 ends H at 33
        // but its last selected channel is 31, which is valid.
        for axis in [1usize, 2] {
            if let Some(last_section) = self.mrope_section[axis].checked_sub(1) {
                let last_channel = last_section
                    .checked_mul(3)
                    .and_then(|channel| channel.checked_add(axis as u32))
                    .ok_or(InvalidInput("M-RoPE interleaved section overflow"))?;
                if last_channel >= half_width {
                    return Err(InvalidInput("M-RoPE interleaved section exceeds the rotary width"));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn config() -> Qwen35MRoPEConfig {
        serde_json::from_value(serde_json::json!({
            "type":"Qwen35MRoPEConfig", "base":10000000.0, "max_sequence_length":262144,
            "head_dim":64, "mrope_section":[11,11,10], "interleaved":true
        }))
        .unwrap()
    }

    #[stark_test]
    fn qwen35_mrope_config_round_trips_the_converter_contract() {
        let config = super::super::AnyRoPEConfig::Qwen35MRoPEConfig(config());
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["type"], "Qwen35MRoPEConfig");
        assert_eq!(value["mrope_section"], serde_json::json!([11, 11, 10]));
        assert_eq!(value["interleaved"], true);
        let decoded: super::super::AnyRoPEConfig = serde_json::from_value(value).unwrap();
        assert_eq!(config, decoded);
        assert!(decoded.validate().is_ok());
    }

    #[stark_test]
    fn qwen35_mrope_validates_base_width_capacity_and_interleaving() {
        assert!(config().validate().is_ok());
        for base in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut invalid = config();
            invalid.base = base;
            assert!(invalid.validate().is_err());
        }
        for width in [0, 1, 63, 65] {
            let mut invalid = config();
            invalid.head_dim = width;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = config();
        invalid.max_sequence_length = 0;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.interleaved = false;
        assert!(invalid.validate().is_err());
    }

    #[stark_test]
    fn qwen35_mrope_rejects_section_truncation_and_overflow() {
        for sections in [[10, 11, 10], [10, 12, 10], [10, 11, 11], [u32::MAX, 1, 1]] {
            let mut invalid = config();
            invalid.mrope_section = sections;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = config();
        invalid.head_dim = u32::MAX - 1;
        invalid.mrope_section = [0, (u32::MAX - 1) / 2, 0];
        assert!(invalid.validate().is_err());
        // Zero spatial sections deliberately leave all channels temporal.
        let mut temporal = config();
        temporal.mrope_section = [32, 0, 0];
        assert!(temporal.validate().is_ok());
    }
}
