use stark_engine_macros::stark_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
};

#[stark_config]
#[derive(Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageAttentionPolicy {
    /// Same-image visibility overrides the complete causal/window mask.
    #[default]
    SameImage,
    /// Image tokens follow the ordinary causal attention mask.
    Causal,
    /// Same-image visibility remains subject to the sliding lower bound.
    WindowedSameImage,
}

#[stark_config(super::TokenMixerConfig)]
pub struct AttentionConfig {
    pub qkvg_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,
    /// Normalize the complete Q/K projection before splitting into heads (OLMo 3).
    #[serde(default)]
    pub qk_norm_full_width: bool,

    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub is_causal: bool,
    #[serde(default)]
    pub image_attention_policy: ImageAttentionPolicy,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<u32>,
    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    pub has_qkvg_biases: bool,
    pub has_out_biases: bool,
    pub has_gate: bool,
    pub normalize_values: bool,
    pub is_kv_sharing: bool,
    /// Preserve the pinned BF16 coefficient and product boundaries in RoPE.
    #[serde(default)]
    pub reference_bf16: bool,
    /// Materialize BF16 rotary coefficients and both products without selecting
    /// a different attention-score implementation.
    #[serde(default)]
    pub materialize_rope_bf16: bool,
    /// Preserve source BF16 scores/probabilities for blocks of 9 through 1023
    /// queries and D512 decode with fewer than 1024 keys. Legacy routing stays the same.
    #[serde(default)]
    pub materialize_prefill_scores: bool,
}

impl AttentionConfig {
    pub fn value_norm_config(&self) -> Option<NormalizationConfig> {
        let source_key = self.key_norm_config.as_ref().filter(|config| config.mlx_bf16_reduction);
        let source_reduction = source_key.is_some();
        self.normalize_values.then_some(NormalizationConfig {
            epsilon: source_key.map_or(1e-6, |config| config.epsilon),
            scale_offset: None,
            upcast_mode: if source_reduction {
                UpcastMode::OnlyNormalization
            } else {
                UpcastMode::FullLayer
            },
            mlx_bf16_reduction: source_reduction,
            subtract_mean: false,
            has_scale: false,
            has_biases: false,
        })
    }
}

#[cfg(test)]
mod source_value_tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn weightless_value_rms_inherits_only_the_explicit_key_source_policy() {
        let mut attention: AttentionConfig = serde_json::from_value(serde_json::json!({
            "type":"AttentionConfig",
            "qkvg_projection_config":{},"out_projection_config":{},"query_norm_config":null,"key_norm_config":null,
            "num_heads":8,"num_groups":1,"head_dim":512,"is_causal":true,"scale":1.0,"sliding_window_size":null,
            "logit_soft_cap":null,"has_sinks":false,"has_qkvg_biases":false,"has_out_biases":false,
            "has_gate":false,"normalize_values":true,"is_kv_sharing":false
        }))
        .unwrap();
        assert_eq!(attention.image_attention_policy, ImageAttentionPolicy::SameImage);
        let legacy = attention.value_norm_config().unwrap();
        assert!(!legacy.mlx_bf16_reduction);
        assert_eq!(legacy.upcast_mode, UpcastMode::FullLayer);
        let mut key = legacy;
        key.upcast_mode = UpcastMode::OnlyNormalization;
        key.mlx_bf16_reduction = true;
        key.epsilon = 3e-6;
        attention.key_norm_config = Some(key);
        let source = attention.value_norm_config().unwrap();
        assert!(source.mlx_bf16_reduction);
        assert_eq!(source.upcast_mode, UpcastMode::OnlyNormalization);
        assert!(!source.has_scale && !source.has_biases && !source.subtract_mean);
        assert_eq!(source.epsilon, 3e-6);
        attention.normalize_values = false;
        assert!(attention.value_norm_config().is_none());
    }

    #[stark_test]
    fn explicit_image_mask_policy_requires_valid_window_and_known_semantics() {
        let mut value = serde_json::json!({
            "type":"AttentionConfig", "qkvg_projection_config":{}, "out_projection_config":{},
            "query_norm_config":null,"key_norm_config":null,"num_heads":2,"num_groups":1,"head_dim":64,
            "is_causal":true,"scale":1.0,"sliding_window_size":null,"logit_soft_cap":null,"has_sinks":false,
            "has_qkvg_biases":false,"has_out_biases":false,"has_gate":false,"normalize_values":false,
            "is_kv_sharing":false,"image_attention_policy":"causal"
        });
        let global: AttentionConfig = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(global.image_attention_policy, ImageAttentionPolicy::Causal);
        assert!(crate::config::validation::validate_attention(&global, None).is_ok());
        value["image_attention_policy"] = serde_json::json!("windowed_same_image");
        let local: AttentionConfig = serde_json::from_value(value.clone()).unwrap();
        assert!(crate::config::validation::validate_attention(&local, None).is_err());
        value["sliding_window_size"] = serde_json::json!(16);
        let local: AttentionConfig = serde_json::from_value(value.clone()).unwrap();
        assert!(crate::config::validation::validate_attention(&local, None).is_ok());
        value["image_attention_policy"] = serde_json::json!("unknown");
        assert!(serde_json::from_value::<AttentionConfig>(value).is_err());
    }
}
