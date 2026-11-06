//! Auxiliary Qwen3.5-family predictor; the target decoder remains authoritative.
use stark_engine_macros::stark_config;

use crate::{
    backends::common::InvalidInput,
    config::{
        normalization::NormalizationConfig, token_mixer::AnyTokenMixerConfig, transformer::TransformerConfig,
        transformer_layer::TransformerLayerConfig,
    },
};

#[stark_config]
pub struct QwenMtpConfig {
    pub model_dim: u32,
    pub target_layer: u32,
    pub input_norm_config: NormalizationConfig,
    pub transformer_config: TransformerConfig,
}

fn owned_full_attention(layer: &TransformerLayerConfig) -> bool {
    layer.kv_source_layer_index.is_none()
        && matches!(&layer.mixer_config, AnyTokenMixerConfig::AttentionConfig(attention)
            if attention.sliding_window_size.is_none() && !attention.is_kv_sharing && attention.is_causal)
}

impl QwenMtpConfig {
    /// This route retains only DeltaNet histories and append-only attention.
    /// Other state layouts require their own admission and restoration proof.
    pub(crate) fn validate_for_target(
        &self,
        target: &TransformerConfig,
    ) -> Result<(), InvalidInput> {
        self.validate_predictor()?;
        super::validation::validate_transformer(target)?;
        if self.model_dim != target.model_dim || self.target_layer as usize + 1 != target.layer_configs.len() {
            return Err(InvalidInput("dense MTP target dimensions or capture layer differ"));
        }
        if target.layer_configs.iter().any(|layer| {
            layer.kv_source_layer_index.is_some()
                || layer.mixer_conv_config.is_some()
                || layer.mlp_conv_config.is_some()
                || !(owned_full_attention(layer)
                    || matches!(layer.mixer_config, AnyTokenMixerConfig::DeltaNetConfig(_)))
        }) {
            return Err(InvalidInput("dense MTP target requires owned full attention or DeltaNet"));
        }
        Ok(())
    }

    pub(crate) fn validate_predictor(&self) -> Result<(), InvalidInput> {
        super::validation::validate_transformer(&self.transformer_config)?;
        if self.model_dim == 0
            || self.model_dim != self.transformer_config.model_dim
            || self.transformer_config.layer_configs.len() != 1
        {
            return Err(InvalidInput("dense MTP predictor dimensions or layer count differ"));
        }
        let layer = &self.transformer_config.layer_configs[0];
        if !owned_full_attention(layer)
            || layer.mixer_conv_config.is_some()
            || layer.mlp_conv_config.is_some()
            || layer.ple_config.is_some()
        {
            return Err(InvalidInput("dense MTP predictor requires one owned full attention layer"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn fixture() -> (TransformerConfig, QwenMtpConfig) {
        let norm = serde_json::json!({"epsilon":1e-6,"scale_offset":null,"upcast_mode":"only_normalization",
            "subtract_mean":false,"has_scale":true,"has_biases":false});
        let layer = serde_json::json!({
            "pre_mixer_norm_config":norm,"pre_mlp_norm_config":norm,"has_post_layer_scalar":false,
            "mixer_conv_config":null,"post_mixer_norm_config":null,"mlp_conv_config":null,
            "post_mlp_norm_config":null,"hidden_dim":null,"ple_config":null,
            "kv_source_layer_index":null,"rope_config":null,
            "mixer_config":{"type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},
                "num_heads":1,"num_groups":1,"head_dim":128,"is_causal":true,"has_sinks":false,
                "has_qkvg_biases":false,"has_out_biases":false,"has_gate":true,
                "normalize_values":false,"is_kv_sharing":false,"query_norm_config":null,"key_norm_config":null,
                "scale":null,"sliding_window_size":null,"logit_soft_cap":null},
            "mlp_config":{"type":"DenseMLPConfig","linear_config":{},"activation":{"type":"SiLU","alpha":1.0},
                "has_up_biases":false,"has_down_biases":false,"gate_clipping":null,"up_clipping":null}});
        let transformer: TransformerConfig = serde_json::from_value(serde_json::json!({
            "layer_configs":[layer],"output_norm_config":norm,"model_dim":128,"hidden_dim":256}))
        .unwrap();
        let head = QwenMtpConfig {
            model_dim: 128,
            target_layer: 0,
            input_norm_config: transformer.output_norm_config.clone(),
            transformer_config: transformer.clone(),
        };
        (transformer, head)
    }

    #[stark_test]
    fn dense_mtp_admission_accepts_full_attention_and_deltanet() {
        let (mut target, mut head) = fixture();
        assert!(head.validate_for_target(&target).is_ok());
        let mut recurrent = target.layer_configs[0].clone();
        recurrent.mixer_config = serde_json::from_value(serde_json::json!({"type":"DeltaNetConfig",
            "in_proj_config":{},"conv_config":{"has_biases":false},"out_proj_config":{},
            "norm_config":head.input_norm_config,"num_heads":1,"num_groups":1,"head_dim":128,
            "value_head_dim":128,"kernel_size":4}))
        .unwrap();
        target.layer_configs = vec![recurrent, target.layer_configs[0].clone()].into_boxed_slice();
        head.target_layer = 1;
        assert!(head.validate_for_target(&target).is_ok());
    }

    #[stark_test]
    fn dense_mtp_admission_preserves_legacy_config_without_head_field() {
        let (target, _) = fixture();
        let legacy = serde_json::json!({"type":"LanguageModelConfig", "token_codec_config":{"type":"RawTextCodecConfig"},
            "decoder_config":{"embedding_config":{"type":"UntiedEmbeddingConfig","input_scale":null,
                "logit_soft_cap":null,"logit_scale":null},"transformer_config":target,"vocab_size":256,
                "ple_model_config":null,"embedding_norm_config":null},
            "generation_config":{"stop_token_ids":[255],"temperature":null,"top_k":null,"top_p":null,
                "min_p":null,"banned_tokens":null,"repetition_penalty":null,"presence_penalty":null,
                "frequency_penalty":null,"suffix_repetition_length":null}});
        let parsed: super::super::model::language_model::LanguageModelConfig = serde_json::from_value(legacy).unwrap();
        assert!(parsed.mtp_config.is_none());
    }

    #[stark_test]
    fn dense_mtp_admission_rejects_target_ring_noncausal_and_shortconv() {
        for mutation in 0..3 {
            let (mut target, head) = fixture();
            let layer = &mut target.layer_configs[0];
            let AnyTokenMixerConfig::AttentionConfig(attention) = &mut layer.mixer_config else {
                unreachable!()
            };
            match mutation {
                0 => attention.sliding_window_size = Some(128),
                1 => attention.is_causal = false,
                _ => {
                    layer.mixer_config = serde_json::from_value(serde_json::json!({"type":"ShortConvConfig",
                    "in_projection_config":{},"conv_config":{"has_biases":false},
                    "out_projection_config":{},"kernel_size":4}))
                    .unwrap()
                },
            }
            assert!(head.validate_for_target(&target).is_err());
        }
    }

    #[stark_test]
    fn dense_mtp_admission_rejects_otherwise_valid_shared_target_cache() {
        let (mut target, mut head) = fixture();
        let owner = target.layer_configs[0].clone();
        let mut consumer = owner.clone();
        let AnyTokenMixerConfig::AttentionConfig(attention) = &mut consumer.mixer_config else {
            unreachable!()
        };
        attention.is_kv_sharing = true;
        consumer.kv_source_layer_index = Some(0);
        target.layer_configs = vec![owner, consumer].into_boxed_slice();
        head.target_layer = 1;
        assert!(super::super::validation::validate_transformer(&target).is_ok());
        assert!(head.validate_for_target(&target).is_err());
    }

    #[stark_test]
    fn dense_mtp_admission_rejects_predictor_ring_sharing_and_multiple_layers() {
        for mutation in 0..4 {
            let (target, mut head) = fixture();
            let layer = &mut head.transformer_config.layer_configs[0];
            let AnyTokenMixerConfig::AttentionConfig(attention) = &mut layer.mixer_config else {
                unreachable!()
            };
            match mutation {
                0 => attention.sliding_window_size = Some(128),
                1 => {
                    attention.is_kv_sharing = true;
                    layer.kv_source_layer_index = Some(0);
                },
                2 => attention.is_causal = false,
                _ => head.transformer_config.layer_configs = vec![layer.clone(), layer.clone()].into_boxed_slice(),
            }
            assert!(head.validate_for_target(&target).is_err());
        }
    }

    #[stark_test]
    fn dense_mtp_admission_rejects_mismatched_target_geometry() {
        for mutation in 0..3 {
            let (target, mut head) = fixture();
            match mutation {
                0 => head.model_dim = 256,
                1 => head.target_layer = 1,
                _ => head.transformer_config.model_dim = 256,
            }
            assert!(head.validate_for_target(&target).is_err());
        }
    }

    #[stark_test]
    fn dense_mtp_admission_rejects_extra_convolution_and_predictor_ple() {
        for mutation in 0..5 {
            let (mut target, mut head) = fixture();
            let convolution = serde_json::from_value(serde_json::json!({
                "conv_config":{"has_biases":false},"kernel_projection_config":{},
                "conv_kernel_size":4,"conv_group_size":1}))
            .unwrap();
            match mutation {
                0 => target.layer_configs[0].mixer_conv_config = Some(convolution),
                1 => target.layer_configs[0].mlp_conv_config = Some(convolution),
                2 => head.transformer_config.layer_configs[0].mixer_conv_config = Some(convolution),
                3 => head.transformer_config.layer_configs[0].mlp_conv_config = Some(convolution),
                _ => {
                    head.transformer_config.layer_configs[0].ple_config = Some(
                        serde_json::from_value(
                            serde_json::json!({"linear_config":{},"norm_config":head.input_norm_config,
                        "ple_dim":16,"activation":{"type":"SiLU","alpha":1.0}}),
                        )
                        .unwrap(),
                    )
                },
            }
            assert!(head.validate_for_target(&target).is_err());
        }
    }
}
