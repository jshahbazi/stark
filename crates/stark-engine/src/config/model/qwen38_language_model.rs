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

/// Validate the native text architecture without constructing another engine's
/// configuration. Backend-specific execution limits are checked by each block.
pub(crate) fn validate_qwen38_architecture(
    a: &Qwen38ArchitectureConfig
) -> Result<(), crate::backends::common::InvalidInput> {
    use crate::backends::common::InvalidInput;
    let dimensions = [
        a.hidden_size,
        a.num_hidden_layers,
        a.vocab_size,
        a.max_position_embeddings,
        a.num_experts,
        a.num_experts_per_tok,
        a.moe_intermediate_size,
        a.shared_expert_intermediate_size,
        a.num_attention_heads,
        a.num_key_value_heads,
        a.head_dim,
        a.linear_num_key_heads,
        a.linear_num_value_heads,
        a.linear_key_head_dim,
        a.linear_value_head_dim,
        a.linear_conv_kernel_dim,
        a.hc_count,
        a.hc_lowrank,
        a.ple_conv_kernel_size,
        a.ngram_size,
        a.heads_per_ngram,
        a.split_ngram_parts,
        a.indexer_n_heads,
        a.indexer_kv_heads,
        a.indexer_head_dim,
        a.indexer_budget,
        a.indexer_compress_ratio,
    ];
    if dimensions.iter().any(|&value| value == 0 || value > 1_048_576)
        || a.num_hidden_layers > 256
        || a.vocab_size > 1_000_000
        || a.layer_types.len() != a.num_hidden_layers as usize
        || a.tie_word_embeddings
        || a.hidden_act != "silu"
        || !a.rms_norm_eps.is_finite()
        || a.rms_norm_eps <= 0.0
        || a.num_experts_per_tok > a.num_experts
        || !a.num_attention_heads.is_multiple_of(a.num_key_value_heads)
        || !a.linear_num_value_heads.is_multiple_of(a.linear_num_key_heads)
        || a.linear_conv_kernel_dim < 2
        || a.ple_conv_kernel_size < 2
        || !(2..=16).contains(&a.ngram_size)
        || a.heads_per_ngram > 256
        || a.indexer_kv_heads != 1
        || !a.indexer_budget.is_multiple_of(a.indexer_compress_ratio)
        || !matches!(a.output_gate_type.as_deref().unwrap_or(&a.hidden_act), "silu" | "sigmoid")
    {
        return Err(InvalidInput("invalid Qwen3.8 architecture geometry"));
    }
    for factors in [
        [a.hidden_size, a.hc_count],
        [a.num_experts, a.hidden_size],
        [a.num_experts, a.moe_intermediate_size],
        [a.num_attention_heads, a.head_dim],
        [a.num_key_value_heads, a.head_dim],
        [a.linear_num_key_heads, a.linear_key_head_dim],
        [a.linear_num_value_heads, a.linear_value_head_dim],
    ] {
        if factors[0].checked_mul(factors[1]).is_none() {
            return Err(InvalidInput("Qwen3.8 dimension overflow"));
        }
    }
    let rope = &a.rope_parameters;
    let rotary = a.head_dim as f32 * rope.partial_rotary_factor;
    if !rope.rope_theta.is_finite()
        || rope.rope_theta <= 0.0
        || !rope.partial_rotary_factor.is_finite()
        || rope.partial_rotary_factor <= 0.0
        || rope.partial_rotary_factor > 1.0
        || rotary < 2.0
        || rotary.fract() != 0.0
        || !(rotary as u32).is_multiple_of(2)
        || rope.rope_type.as_deref().is_some_and(|name| name != "default")
        || rope.mrope_section.as_ref().is_some_and(|parts| parts.is_empty() || parts.len() > 32 || parts.contains(&0))
    {
        return Err(InvalidInput("invalid Qwen3.8 rotary geometry"));
    }
    let q = &a.quantization;
    let mut groups = std::collections::HashSet::new();
    if q.default.bits != 4
        || q.default.group_size != 64
        || q.ngram_embedding.bits != 4
        || q.ngram_embedding.group_size != 32
        || q.q8_groups.iter().any(|group| {
            !matches!(group.as_str(), "attn" | "gdn" | "shared" | "head" | "embed") || !groups.insert(group)
        })
    {
        return Err(InvalidInput("invalid Qwen3.8 affine storage policy"));
    }
    let mut eos = std::collections::HashSet::new();
    if a.eos_token_id.is_empty()
        || a.eos_token_id.len() > 32
        || a.eos_token_id.iter().any(|&id| id >= a.vocab_size || !eos.insert(id))
        || a.ple_layer_ids.len() > 1
        || a.ple_layer_ids.iter().any(|&layer| layer == 0 || layer > a.num_hidden_layers)
        || a.ple_embed_dim.is_some_and(|dim| dim == 0 || dim > 1_048_576)
    {
        return Err(InvalidInput("invalid Qwen3.8 token or PLE schedule"));
    }
    let ple = &a.ple;
    let heads = (a.ngram_size - 1) * a.heads_per_ngram;
    if a.ple_layer_ids.is_empty() {
        if !ple.layer_multipliers.is_empty()
            || !ple.ngram_heads_vocab_sizes.is_empty()
            || !ple.ngram_heads_offsets.is_empty()
            || ple.padded_vocab_size != 0
        {
            return Err(InvalidInput("unexpected inactive Qwen3.8 PLE configuration"));
        }
    } else if ple.layer_multipliers.len() != a.ngram_size as usize
        || ple.ngram_heads_vocab_sizes.len() != heads as usize
        || ple.ngram_heads_offsets.len() != heads as usize
        || ple.padded_vocab_size == 0
        || !ple.padded_vocab_size.is_multiple_of(a.split_ngram_parts)
        || !(a.ple_embed_dim.unwrap_or(a.hidden_size)).is_multiple_of(heads)
        || ple.ngram_heads_vocab_sizes.iter().zip(ple.ngram_heads_offsets.iter()).any(|(&size, &offset)| {
            size == 0 || offset.checked_add(size).is_none_or(|end| end > ple.padded_vocab_size as u64)
        })
    {
        return Err(InvalidInput("invalid Qwen3.8 PLE geometry"));
    }
    Ok(())
}

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
