use stark_engine_macros::stark_config;

#[stark_config(super::Activation)]
pub struct SiLU {
    pub alpha: f32,
    /// Preserve source BF16 sigmoid boundaries on explicitly selected artifacts.
    #[serde(default)]
    pub materialize_sigmoid: bool,
}

impl Default for SiLU {
    fn default() -> Self {
        Self {
            ty: Default::default(),
            alpha: 1.0,
            materialize_sigmoid: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{backends::common::gpu_types::ActivationType, config::activation::AnyActivation};

    #[stark_test]
    fn silu_materialized_sigmoid_is_explicit_and_legacy_defaults_are_unchanged() {
        let legacy: AnyActivation = serde_json::from_value(serde_json::json!({"type":"SiLU","alpha":1.0})).unwrap();
        assert_eq!(legacy.act_type(), ActivationType::SILU);
        assert!(!SiLU::default().materialize_sigmoid);
        let materialized: AnyActivation =
            serde_json::from_value(serde_json::json!({"type":"SiLU","alpha":1.0,"materialize_sigmoid":true})).unwrap();
        assert_eq!(materialized.act_type(), ActivationType::SILUMaterializedSigmoid);
        assert_eq!(materialized.custom_alpha(), None);
    }
}
