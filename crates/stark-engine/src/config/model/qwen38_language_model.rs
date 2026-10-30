//! Uzu-format text-only Qwen3.8 architecture and tensor binding contract.

use std::{collections::BTreeMap, fmt};

use serde::{
    Deserialize, Deserializer, Serialize,
    de::{MapAccess, Visitor},
};
use stark_engine_macros::stark_config;

use crate::{config::model::generation::GenerationConfig, utils::strict_serde::DeserializeStrict};

#[stark_config(super::ModelConfig)]
pub struct Qwen38LanguageModelConfig {
    pub decoder_config: Qwen38DecoderConfig,
    pub generation_config: GenerationConfig,
}

#[stark_config]
pub struct Qwen38DecoderConfig {
    pub architecture: Qwen38ArchitectureConfig,
    pub tensor_bindings: Qwen38TensorBindings,
}

#[stark_config]
#[serde(rename_all = "snake_case")]
pub enum Qwen38LayerType {
    LinearAttention,
    FullAttention,
}

#[stark_config]
pub struct Qwen38RopeConfig {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: Option<Box<[u32]>>,
    pub mrope_interleaved: Option<bool>,
    pub rope_type: Option<String>,
}

#[stark_config]
pub struct Qwen38QuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

#[stark_config]
pub struct Qwen38QuantizationConfig {
    pub default: Qwen38QuantConfig,
    pub ngram_embedding: Qwen38QuantConfig,
    pub q8_groups: Box<[String]>,
}

#[stark_config]
pub struct Qwen38PleConfig {
    pub layer_multipliers: Box<[u64]>,
    pub ngram_heads_vocab_sizes: Box<[u64]>,
    pub ngram_heads_offsets: Box<[u64]>,
    pub padded_vocab_size: u32,
}

#[stark_config]
pub struct Qwen38ArchitectureConfig {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub layer_types: Box<[Qwen38LayerType]>,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: u32,
    pub tie_word_embeddings: bool,
    pub eos_token_id: Box<[u32]>,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub norm_topk_prob: bool,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub rope_parameters: Qwen38RopeConfig,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,
    pub output_gate_type: Option<String>,
    pub hidden_act: String,
    pub hc_count: u32,
    pub hc_lowrank: u32,
    pub ple_layer_ids: Box<[u32]>,
    pub ple_embed_dim: Option<u32>,
    pub ple_conv_kernel_size: u32,
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub split_ngram_parts: u32,
    pub indexer_n_heads: u32,
    pub indexer_kv_heads: u32,
    pub indexer_head_dim: u32,
    pub indexer_budget: u32,
    pub indexer_compress_ratio: u32,
    pub ple: Qwen38PleConfig,
    pub quantization: Qwen38QuantizationConfig,
}

#[stark_config]
pub enum Qwen38TensorDType {
    BF16,
    F32,
    U32,
    I64,
    U64,
}

#[stark_config]
#[serde(rename_all = "snake_case")]
pub enum Qwen38TensorLayout {
    Direct,
    PackedU32,
    GroupOutput,
}

#[stark_config]
pub struct Qwen38TensorBinding {
    pub name: String,
    pub dtype: Qwen38TensorDType,
    pub shape: Box<[u32]>,
    pub layout: Qwen38TensorLayout,
}

/// Strict map: repeated logical names must not hide an earlier binding.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Qwen38TensorBindings(pub BTreeMap<String, Qwen38TensorBinding>);

impl<'de> Deserialize<'de> for Qwen38TensorBindings {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BindingsVisitor;
        impl<'de> Visitor<'de> for BindingsVisitor {
            type Value = Qwen38TensorBindings;
            fn expecting(
                &self,
                formatter: &mut fmt::Formatter,
            ) -> fmt::Result {
                formatter.write_str("a map of unique Qwen tensor bindings")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut bindings = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, Qwen38TensorBinding>()? {
                    if key.len() > 512 || bindings.len() >= 16_384 || bindings.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("invalid or duplicate Qwen tensor binding"));
                    }
                }
                Ok(Qwen38TensorBindings(bindings))
            }
        }
        deserializer.deserialize_map(BindingsVisitor)
    }
}

impl<'de> DeserializeStrict<'de> for Qwen38TensorBindings {}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn bindings_are_typed_and_reject_unknown_fields_and_duplicate_names() {
        let binding = r#"{"name":"language_model.lm_head.weights","dtype":"U32","shape":[3,16],"layout":"packed_u32"}"#;
        let valid = format!("{{\"lm_head.weight\":{binding}}}");
        let parsed: Qwen38TensorBindings = serde_json::from_str(&valid).unwrap();
        assert_eq!(parsed.0.len(), 1);
        let duplicate = format!("{{\"lm_head.weight\":{binding},\"lm_head.weight\":{binding}}}");
        assert!(serde_json::from_str::<Qwen38TensorBindings>(&duplicate).is_err());
        for invalid in [
            binding.replace("\"U32\"", "\"U8\""),
            binding.replace("\"packed_u32\"", "\"native\""),
            binding.replace("[3,16]", "[3,-1]"),
            binding.replace("\"layout\":", "\"source_path\":\"elsewhere\",\"layout\":"),
        ] {
            assert!(serde_json::from_str::<Qwen38TensorBinding>(&invalid).is_err());
        }
    }
}
