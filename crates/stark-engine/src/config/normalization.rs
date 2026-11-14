use stark_engine_macros::stark_config;

#[stark_config]
#[serde(rename_all = "snake_case")]
pub enum UpcastMode {
    OnlyNormalization,
    FullLayer,
}

#[stark_config]
pub struct NormalizationConfig {
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    /// Explicit MLX BF16 row-reduction arithmetic; legacy and canonical defaults remain unchanged.
    #[serde(default)]
    pub mlx_bf16_reduction: bool,
    pub subtract_mean: bool,
    pub has_scale: bool,
    pub has_biases: bool,
}

impl NormalizationConfig {
    pub(crate) fn validate_mlx_bf16_reduction(
        &self,
        data_type: crate::data_type::DataType,
        element_count: u32,
        use_hadamard: bool,
    ) -> Result<(), crate::backends::common::InvalidInput> {
        if self.mlx_bf16_reduction
            && (data_type != crate::data_type::DataType::BF16
                || self.upcast_mode != UpcastMode::OnlyNormalization
                || self.subtract_mean
                || self.has_biases
                || self.scale_offset.is_some_and(|x| x != 0.0)
                || use_hadamard
                || element_count == 0
                || element_count > 4096)
        {
            return Err(crate::backends::common::InvalidInput("unsupported MLX BF16 normalization profile"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod source_tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::data_type::DataType;

    #[stark_test]
    fn mlx_bf16_row_policy_defaults_off_and_rejects_unqualified_arithmetic() {
        let value = serde_json::json!({"epsilon":1e-6,"scale_offset":null,"upcast_mode":"only_normalization",
            "subtract_mean":false,"has_scale":true,"has_biases":false});
        let mut config: NormalizationConfig = serde_json::from_value(value).unwrap();
        assert!(!config.mlx_bf16_reduction);
        assert!(config.validate_mlx_bf16_reduction(DataType::F32, 8192, true).is_ok());
        config.mlx_bf16_reduction = true;
        assert!(config.validate_mlx_bf16_reduction(DataType::BF16, 1536, false).is_ok());
        for (dtype, width, rht) in [
            (DataType::F32, 1536, false),
            (DataType::F16, 1536, false),
            (DataType::BF16, 4097, false),
            (DataType::BF16, 0, false),
            (DataType::BF16, 1536, true),
        ] {
            assert!(config.validate_mlx_bf16_reduction(dtype, width, rht).is_err());
        }
        for mutation in 0..4 {
            let mut bad = config.clone();
            match mutation {
                0 => bad.upcast_mode = UpcastMode::FullLayer,
                1 => bad.subtract_mean = true,
                2 => bad.has_biases = true,
                _ => bad.scale_offset = Some(1.0),
            }
            assert!(bad.validate_mlx_bf16_reduction(DataType::BF16, 1536, false).is_err());
        }
    }
}
