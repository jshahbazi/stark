// SPDX-License-Identifier: Apache-2.0
//! Bind the pinned STQ trunk to the tensor names consumed by the native forward graph.

use std::collections::{BTreeMap, btree_map::Entry};

use serde::Deserialize;
use thiserror::Error;

use super::{
    stq_layout::{StqFormat, StqProjection},
    stq_source::{StqDtype, StqSource, StqTensorLocation},
};

#[derive(Debug, Error)]
pub enum StqTrunkError {
    #[error("invalid pinned STQ trunk geometry in config.json")]
    Config,
    #[error("invalid STQ binding for {alias} from {source_name}")]
    InvalidInput {
        alias: String,
        source_name: String,
    },
    #[error("duplicate STQ binding for {0}")]
    DuplicateAlias(String),
}

pub struct StqNamedTensor<'a> {
    pub name: String,
    pub dtype: StqDtype,
    /// Safetensors order, with the contiguous dimension last.
    pub shape: &'a [u32],
    pub location: StqTensorLocation<'a>,
}

pub struct StqNamedProjection<'a> {
    /// Base name of the three projection planes.
    pub name: String,
    pub projection: StqProjection<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StqRawTransform {
    AsStored,
    WidenToF32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StqProjectionUse {
    Linear,
    Embedding,
    /// `split_kv_b` produces BF16 key rows [heads, 512, 192].
    AbsorbedKey {
        heads: u32,
    },
    /// The same `split_kv_b` call produces BF16 value rows [heads, 256, 512].
    AbsorbedValue {
        heads: u32,
    },
}

pub struct StqEngramGateStream<'a> {
    pub key_norm: StqNamedTensor<'a>,
    pub query_norm: StqNamedTensor<'a>,
}

pub enum StqTrunkWeight<'a> {
    Tensor {
        source: StqNamedTensor<'a>,
        transform: StqRawTransform,
    },
    Projection {
        source: StqNamedProjection<'a>,
        use_as: StqProjectionUse,
    },
    Concatenate {
        /// Sources in destination order. Any widening precedes concatenation.
        sources: Vec<StqNamedTensor<'a>>,
        transform: StqRawTransform,
        /// Contiguous dimension last, as on the source tensors.
        shape: &'static [u32],
    },
    /// The GGUF alias addresses one precombined gate tensor. Sushi needs the
    /// two RMSNorm weights applied independently and preserves the operation
    /// order. Binding these inputs requires a format-specific gate dispatch.
    EngramGateNorms {
        streams: Vec<StqEngramGateStream<'a>>,
    },
}

pub struct StqTrunk<'a> {
    weights: BTreeMap<String, StqTrunkWeight<'a>>,
    routed: Vec<[StqProjection<'a>; 3]>,
}

impl<'a> StqTrunk<'a> {
    pub fn weight(
        &self,
        alias: &str,
    ) -> Option<&StqTrunkWeight<'a>> {
        self.weights.get(alias)
    }

    pub fn weights(&self) -> impl ExactSizeIterator<Item = (&str, &StqTrunkWeight<'a>)> {
        self.weights.iter().map(|(alias, binding)| (alias.as_str(), binding))
    }

    /// Weights outside `Model::find`, in native gate, up and down order.
    pub fn routed_experts(
        &self,
        block: u32,
    ) -> Option<&[StqProjection<'a>; 3]> {
        self.routed.get(block.checked_sub(1)? as usize)
    }
}

#[derive(Deserialize)]
struct Config {
    text_config: TextConfig,
}

#[derive(Deserialize)]
struct TextConfig {
    layer_types: Vec<Attention>,
    num_attention_heads: u32,
    swa_num_attention_heads: u32,
    engram_layer_ids: [usize; 4],
}

#[derive(Deserialize)]
enum Attention {
    #[serde(rename = "deepseek_sparse_attention")]
    Sparse,
    #[serde(rename = "sliding_attention")]
    Sliding,
}

impl StqSource {
    /// Resolve all trunk aliases against the validated headers and descriptors.
    /// Conversion and any weight payload reads belong to the execution layer.
    pub fn stq_trunk(&self) -> Result<StqTrunk<'_>> {
        let config: Config = serde_json::from_slice(self.config_bytes()).map_err(|_| StqTrunkError::Config)?;
        let text = config.text_config;
        if text.layer_types.len() != 42 || text.num_attention_heads != 32 || text.swa_num_attention_heads != 64 {
            return Err(StqTrunkError::Config);
        }
        let mut builder = Builder {
            source: self,
            trunk: StqTrunk {
                weights: BTreeMap::new(),
                routed: Vec::with_capacity(41),
            },
        };
        builder.bind_globals()?;
        for (block, attention) in text.layer_types.iter().enumerate() {
            let (heads, sparse) = match attention {
                Attention::Sparse => (text.num_attention_heads, true),
                Attention::Sliding => (text.swa_num_attention_heads, false),
            };
            builder.bind_block(block, heads, sparse, text.engram_layer_ids.contains(&(block + 1)))?;
        }
        Ok(builder.trunk)
    }
}

type Result<T> = std::result::Result<T, StqTrunkError>;
type RawSpec<'a> = (&'a str, &'a str, StqDtype, &'a [u32], StqRawTransform);
type ProjectionSpec<'a> = (&'a str, &'a str, u32, u32, StqProjectionUse);

struct Builder<'a> {
    source: &'a StqSource,
    trunk: StqTrunk<'a>,
}

impl<'a> Builder<'a> {
    fn bind_globals(&mut self) -> Result<()> {
        use StqDtype::{BF16, F32};
        use StqProjectionUse::{Embedding, Linear};
        use StqRawTransform::{AsStored, WidenToF32};

        self.raws(
            "",
            "model.",
            &[
                ("output_hc_fn.weight", "hc_head.hc_head_fn", F32, &[4, 12288], AsStored),
                ("output_hc_scale.weight", "hc_head.hc_head_scale", F32, &[1], AsStored),
                ("output_hc_base.weight", "hc_head.hc_head_base", F32, &[4], AsStored),
                ("output_norm.weight", "norm.weight", BF16, &[3072], WidenToF32),
            ],
        )?;
        self.projections(
            "",
            "",
            &[
                ("token_embd.weight", "model.embed_tokens", 3072, 120832, Embedding),
                ("output.weight", "lm_head", 3072, 120832, Linear),
            ],
        )
    }

    fn bind_block(
        &mut self,
        block: usize,
        heads: u32,
        sparse: bool,
        has_engram: bool,
    ) -> Result<()> {
        use StqDtype::{BF16, F32};
        use StqProjectionUse::{AbsorbedKey, AbsorbedValue, Linear};
        use StqRawTransform::{AsStored, WidenToF32};

        let alias = format!("blk.{block}.");
        let source = format!("model.layers.{block}.");
        self.raws(
            &alias,
            &source,
            &[
                ("hc_attn_fn.weight", "hc_attn_layer.hc_pre.hc_fn", F32, &[8, 12288], AsStored),
                ("hc_attn_scale.weight", "hc_attn_layer.hc_pre.hc_scale", F32, &[2], AsStored),
                ("hc_attn_base.weight", "hc_attn_layer.hc_pre.hc_base", F32, &[8], AsStored),
                ("hc_ffn_fn.weight", "hc_mlp_layer.hc_pre.hc_fn", F32, &[8, 12288], AsStored),
                ("hc_ffn_scale.weight", "hc_mlp_layer.hc_pre.hc_scale", F32, &[2], AsStored),
                ("hc_ffn_base.weight", "hc_mlp_layer.hc_pre.hc_base", F32, &[8], AsStored),
                ("attn_norm.weight", "input_layernorm.weight", BF16, &[3072], WidenToF32),
                ("ffn_norm.weight", "post_attention_layernorm.weight", BF16, &[3072], WidenToF32),
            ],
        )?;

        let attention = format!("{source}self_attn.");
        self.raws(
            &alias,
            &attention,
            &[
                ("attn_q_a_norm.weight", "q_a_layernorm.weight", BF16, &[1024], WidenToF32),
                ("attn_kv_a_mqa.weight", "kv_a_proj_with_mqa.weight", BF16, &[576, 3072], AsStored),
                ("attn_kv_a_norm.weight", "kv_a_layernorm.weight", BF16, &[512], WidenToF32),
                ("attn_sinks.weight", "learnable_sink_param", F32, &[heads], AsStored),
                ("post_attention_norm.weight", "post_proj_norm.weight", BF16, &[3072], WidenToF32),
            ],
        )?;
        self.projections(
            &alias,
            &attention,
            &[
                ("attn_q_a.weight", "q_a_proj", 3072, 1024, Linear),
                ("attn_q_b.weight", "q_b_proj", 1024, heads * 256, Linear),
                (
                    "attn_k_b.weight",
                    "kv_b_proj",
                    512,
                    heads * (192 + 256),
                    AbsorbedKey {
                        heads,
                    },
                ),
                (
                    "attn_v_b.weight",
                    "kv_b_proj",
                    512,
                    heads * (192 + 256),
                    AbsorbedValue {
                        heads,
                    },
                ),
                ("attn_gate.weight", "linear_gate", 3072, heads * 256, Linear),
                ("attn_output.weight", "o_proj", heads * 256, 3072, Linear),
            ],
        )?;

        let mlp = format!("{source}mlp.");
        if block == 0 {
            self.projections(
                &alias,
                &mlp,
                &[
                    ("ffn_gate.weight", "gate_proj", 3072, 13824, Linear),
                    ("ffn_up.weight", "up_proj", 3072, 13824, Linear),
                    ("ffn_down.weight", "down_proj", 13824, 3072, Linear),
                ],
            )?;
            self.raws(&alias, &mlp, &[("post_ffw_norm.weight", "post_fc2_norm.weight", BF16, &[3072], WidenToF32)])?;
        } else {
            self.projections(
                &alias,
                &mlp,
                &[
                    ("ffn_routed_down.weight", "routed_expert_down_proj", 3072, 1536, Linear),
                    ("ffn_routed_up.weight", "routed_expert_up_proj", 1536, 3072, Linear),
                    ("ffn_gate_shexp.weight", "shared_experts.gate_proj", 3072, 1536, Linear),
                    ("ffn_up_shexp.weight", "shared_experts.up_proj", 3072, 1536, Linear),
                    ("ffn_down_shexp.weight", "shared_experts.down_proj", 1536, 3072, Linear),
                ],
            )?;
            self.raws(
                &alias,
                &mlp,
                &[
                    ("post_ffw_norm.weight", "combine_output_norm.weight", BF16, &[3072], WidenToF32),
                    ("ffn_gate_inp.weight", "gate.weight", BF16, &[512, 3072], AsStored),
                    ("exp_probs_b.bias", "gate.e_score_correction_bias", F32, &[512], AsStored),
                ],
            )?;
            let experts = self
                .source
                .stq_routed_experts(block as u32)
                .filter(|banks| banks.iter().all(|bank| bank.format == StqFormat::Pair1K4))
                .ok_or_else(|| invalid_input(&format!("blk.{block} routed experts"), &format!("{mlp}switch_mlp")))?;
            self.trunk.routed.push(experts);
        }

        if sparse {
            self.raws(
                &alias,
                &format!("{attention}compressed_indexer."),
                &[
                    ("indexer.attn_q_b.weight", "query_projection.weight", BF16, &[2048, 1024], AsStored),
                    ("indexer.attn_k.weight", "token_key_projection.weight", BF16, &[128, 3072], AsStored),
                    ("indexer.k_norm.weight", "token_key_norm.weight", BF16, &[128], WidenToF32),
                    ("indexer.proj.weight", "head_weight_projection.weight", BF16, &[16, 3072], AsStored),
                    ("indexer_compressor_value.weight", "key_value_projection.weight", F32, &[128, 128], AsStored),
                    ("indexer_compressor_gate.weight", "key_gate_projection.weight", F32, &[128, 128], AsStored),
                    ("indexer_compressor_ape.weight", "intra_group_ape", F32, &[4, 128], AsStored),
                    ("indexer_compressor_norm.weight", "key_norm.weight", F32, &[128], AsStored),
                ],
            )?;
        }
        if has_engram {
            self.bind_engram(&alias, &format!("{source}engram_layer."))?;
        }
        Ok(())
    }

    fn bind_engram(
        &mut self,
        alias: &str,
        source: &str,
    ) -> Result<()> {
        use StqDtype::BF16;
        use StqRawTransform::{AsStored, WidenToF32};

        let head_norms = (0..16).map(|head| format!("lookup_per_head_norms.{head}.weight"));
        self.concatenate(
            format!("{alias}engram_lookup_norm.weight"),
            source,
            head_norms,
            &[96],
            &[16, 96],
            WidenToF32,
        )?;

        // The native projection's value rows follow all four streams' key rows.
        let kv_names = |field| {
            (0..4)
                .map(move |stream| format!("key_projs.{stream}.{field}"))
                .chain(std::iter::once(format!("value_proj.{field}")))
        };
        self.concatenate(
            format!("{alias}engram_kv.weight"),
            source,
            kv_names("weight"),
            &[3072, 1536],
            &[15360, 1536],
            AsStored,
        )?;
        self.concatenate(
            format!("{alias}engram_kv_b.weight"),
            source,
            kv_names("bias"),
            &[3072],
            &[15360],
            WidenToF32,
        )?;

        let gate_alias = format!("{alias}engram_gate.weight");
        let mut streams = Vec::with_capacity(4);
        for stream in 0..4 {
            streams.push(StqEngramGateStream {
                key_norm: self.tensor(&gate_alias, format!("{source}norm_key.{stream}.weight"), BF16, &[3072])?,
                query_norm: self.tensor(&gate_alias, format!("{source}norm_query.{stream}.weight"), BF16, &[3072])?,
            });
        }
        self.insert(
            gate_alias,
            StqTrunkWeight::EngramGateNorms {
                streams,
            },
        )?;

        let scales = (0..4).map(|stream| format!("unified_per_hc_scales.{stream}"));
        self.concatenate(format!("{alias}engram_hc_scale.weight"), source, scales, &[3072], &[4, 3072], WidenToF32)?;
        self.raws(alias, source, &[("engram_pre_gamma.weight", "unified_pre_gamma", BF16, &[3072], WidenToF32)])
    }

    fn tensor(
        &self,
        alias: &str,
        name: String,
        dtype: StqDtype,
        shape: &[u32],
    ) -> Result<StqNamedTensor<'a>> {
        let info = self
            .source
            .tensor(&name)
            .filter(|info| info.dtype == dtype && info.shape.as_slice() == shape)
            .ok_or_else(|| invalid_input(alias, &name))?;
        let location = self.source.tensor_location(&name).ok_or_else(|| invalid_input(alias, &name))?;
        Ok(StqNamedTensor {
            name,
            dtype,
            shape: &info.shape,
            location,
        })
    }

    fn raws(
        &mut self,
        alias_prefix: &str,
        source_prefix: &str,
        specs: &[RawSpec<'_>],
    ) -> Result<()> {
        for &(alias, source, dtype, shape, transform) in specs {
            let alias = format!("{alias_prefix}{alias}");
            let source = self.tensor(&alias, format!("{source_prefix}{source}"), dtype, shape)?;
            self.insert(
                alias,
                StqTrunkWeight::Tensor {
                    source,
                    transform,
                },
            )?;
        }
        Ok(())
    }

    fn projections(
        &mut self,
        alias_prefix: &str,
        source_prefix: &str,
        specs: &[ProjectionSpec<'_>],
    ) -> Result<()> {
        for &(alias, source, inputs, outputs, use_as) in specs {
            let alias = format!("{alias_prefix}{alias}");
            let name = format!("{source_prefix}{source}");
            let projection = self
                .source
                .stq_projection(&name)
                .filter(|p| {
                    p.format == StqFormat::Spec1K6 && p.banks == 1 && p.inputs == inputs && p.outputs == outputs
                })
                .ok_or_else(|| invalid_input(&alias, &name))?;
            self.insert(
                alias,
                StqTrunkWeight::Projection {
                    source: StqNamedProjection {
                        name,
                        projection,
                    },
                    use_as,
                },
            )?;
        }
        Ok(())
    }

    fn concatenate(
        &mut self,
        alias: String,
        source_prefix: &str,
        names: impl Iterator<Item = String>,
        source_shape: &[u32],
        shape: &'static [u32],
        transform: StqRawTransform,
    ) -> Result<()> {
        let sources = names
            .map(|name| self.tensor(&alias, format!("{source_prefix}{name}"), StqDtype::BF16, source_shape))
            .collect::<Result<Vec<_>>>()?;
        self.insert(
            alias,
            StqTrunkWeight::Concatenate {
                sources,
                transform,
                shape,
            },
        )
    }

    fn insert(
        &mut self,
        alias: String,
        weight: StqTrunkWeight<'a>,
    ) -> Result<()> {
        match self.trunk.weights.entry(alias) {
            Entry::Occupied(entry) => Err(StqTrunkError::DuplicateAlias(entry.key().clone())),
            Entry::Vacant(entry) => {
                entry.insert(weight);
                Ok(())
            },
        }
    }
}

fn invalid_input(
    alias: &str,
    source_name: &str,
) -> StqTrunkError {
    StqTrunkError::InvalidInput {
        alias: alias.to_owned(),
        source_name: source_name.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[stark_engine_macros::stark_test]
    #[ignore] // Requires STARK_HY4_STQ_TEST_PACK pointing to the pinned artifact.
    fn binds_pinned_trunk_headers() {
        let model = std::env::var_os("STARK_HY4_STQ_TEST_PACK").expect("set the pinned STQ pack path");
        let source = StqSource::open(Path::new(&model)).unwrap();
        let trunk = source.stq_trunk().unwrap();
        assert_eq!(trunk.weights().len(), 1256);
        assert!(trunk.routed_experts(0).is_none());
        assert!(trunk.routed_experts(42).is_none());

        for block in 1..42 {
            let banks = trunk.routed_experts(block).unwrap();
            assert!(banks.iter().all(|bank| bank.format == StqFormat::Pair1K4 && bank.banks == 512));
        }
        for (block, heads) in [(1, 32), (2, 64)] {
            let key = trunk.weight(&format!("blk.{block}.attn_k_b.weight")).unwrap();
            let value = trunk.weight(&format!("blk.{block}.attn_v_b.weight")).unwrap();
            let StqTrunkWeight::Projection {
                source: k,
                use_as: k_use,
            } = key
            else {
                panic!("missing split key")
            };
            let StqTrunkWeight::Projection {
                source: v,
                use_as: v_use,
            } = value
            else {
                panic!("missing split value")
            };
            assert_eq!(k.name, v.name);
            assert_eq!(k.projection.trellis.offset, v.projection.trellis.offset);
            assert_eq!(
                *k_use,
                StqProjectionUse::AbsorbedKey {
                    heads
                }
            );
            assert_eq!(
                *v_use,
                StqProjectionUse::AbsorbedValue {
                    heads
                }
            );
        }
        assert!(trunk.weight("blk.1.indexer.attn_q_b.weight").is_some());
        assert!(trunk.weight("blk.2.indexer.attn_q_b.weight").is_none());

        for block in [1, 11, 21, 31] {
            let StqTrunkWeight::Concatenate {
                sources,
                ..
            } = trunk.weight(&format!("blk.{block}.engram_kv.weight")).unwrap()
            else {
                panic!("invalid Engram projection list")
            };
            for (stream, key) in sources[..4].iter().enumerate() {
                assert!(key.name.ends_with(&format!("key_projs.{stream}.weight")));
            }
            assert_eq!(sources.len(), 5);
            assert!(sources[4].name.ends_with("value_proj.weight"));

            let StqTrunkWeight::EngramGateNorms {
                streams,
            } = trunk.weight(&format!("blk.{block}.engram_gate.weight")).unwrap()
            else {
                panic!("missing independent Engram gate inputs")
            };
            assert_eq!(streams.len(), 4);
            for (stream, norms) in streams.iter().enumerate() {
                assert!(norms.key_norm.name.ends_with(&format!("norm_key.{stream}.weight")));
                assert!(norms.query_norm.name.ends_with(&format!("norm_query.{stream}.weight")));
            }
        }
        source.verify_unchanged().unwrap();
    }
}
