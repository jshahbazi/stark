// SPDX-License-Identifier: Apache-2.0
//! Header-only bindings for the pinned Hy4 Sushi DFly2 drafter.

use std::collections::{BTreeMap, btree_map::Entry};

use serde::Deserialize;
use thiserror::Error;

use super::{
    stq_layout::StqFormat,
    stq_source::{StqDtype, StqSource},
    stq_trunk::{StqNamedProjection, StqNamedTensor, StqProjectionUse, StqRawTransform, StqTrunkWeight},
};

const PREFIX: &str = "model.dfly_layers.";
const SHARD: &str = "model-dfly.safetensors";
const LAYERS: usize = 5;
const HEADS: u32 = 32;
const HIDDEN: u32 = 3072;
const LATENT: u32 = 512;
const Q_RANK: u32 = 1024;
const HEAD_OUTPUT: u32 = HEADS * 256;
const INTERMEDIATE: u32 = 9216;
const TARGETS: [u32; LAYERS] = [25, 29, 33, 37, 41];
const MASK_TOKEN: u32 = 120_009;
const DRAFTS: u32 = 5;

#[derive(Debug, Error)]
pub enum StqDflyError {
    #[error("invalid pinned STQ DFly2 geometry in config.json")]
    Config,
    #[error("invalid STQ DFly2 tensor {0}")]
    Tensor(String),
    #[error("duplicate STQ DFly2 alias {0}")]
    Duplicate(String),
}

type Result<T> = std::result::Result<T, StqDflyError>;

pub struct StqDfly<'a> {
    pub targets: [u32; LAYERS],
    pub mask_token: u32,
    pub drafts: u32,
    weights: BTreeMap<String, StqTrunkWeight<'a>>,
}

impl<'a> StqDfly<'a> {
    pub fn weights(&self) -> impl ExactSizeIterator<Item = (&str, &StqTrunkWeight<'a>)> {
        self.weights.iter().map(|(alias, weight)| (alias.as_str(), weight))
    }
}

#[derive(Deserialize)]
struct Config {
    dfly_num_hidden_layers: usize,
    dfly_mask_token_id: u32,
    dfly_target_kv_layers: [u32; LAYERS],
    dfly_target_index_layer: [u32; LAYERS],
    dfly_target_index_topk: u32,
    dfly_context_local_window: Option<u32>,
    dfly_mtp_fusion: String,
    dfly_enable_hidden_correction: bool,
    speculative_config: Speculative,
}

#[derive(Deserialize)]
struct Speculative {
    method: String,
    num_speculative_tokens: u32,
}

impl Config {
    fn parse(source: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(source).map_err(|_| StqDflyError::Config)?;
        if config.dfly_num_hidden_layers != LAYERS
            || config.dfly_mask_token_id != MASK_TOKEN
            || config.dfly_target_kv_layers != TARGETS
            || config.dfly_target_index_layer != TARGETS
            || config.dfly_target_index_topk != 2048
            || config.dfly_context_local_window.is_some()
            || config.dfly_mtp_fusion != "slot0"
            || !config.dfly_enable_hidden_correction
            || config.speculative_config.method != "dfly"
            || config.speculative_config.num_speculative_tokens != DRAFTS
        {
            return Err(StqDflyError::Config);
        }
        Ok(config)
    }
}

impl StqSource {
    /// Resolve DFly2's source slices without reading their payloads. The token
    /// embedding, output head and target final norm come from the trunk.
    pub fn stq_dfly(&self) -> Result<StqDfly<'_>> {
        let config = Config::parse(self.config_bytes())?;
        let mut builder = Builder {
            source: self,
            weights: BTreeMap::new(),
        };
        for layer in 0..LAYERS {
            builder.layer(layer)?;
        }
        builder.global()?;
        Ok(StqDfly {
            targets: config.dfly_target_kv_layers,
            mask_token: config.dfly_mask_token_id,
            drafts: config.speculative_config.num_speculative_tokens,
            weights: builder.weights,
        })
    }
}

struct Builder<'a> {
    source: &'a StqSource,
    weights: BTreeMap<String, StqTrunkWeight<'a>>,
}

impl<'a> Builder<'a> {
    fn layer(
        &mut self,
        layer: usize,
    ) -> Result<()> {
        use StqProjectionUse::{AbsorbedKey, AbsorbedValue, Linear};
        use StqRawTransform::{AsStored, WidenToF32};

        let base = format!("{PREFIX}{layer}.");
        for (suffix, shape, transform) in [
            ("input_layernorm.weight", &[HIDDEN][..], WidenToF32),
            ("post_attention_layernorm.weight", &[HIDDEN], WidenToF32),
            ("self_attn.q_a_layernorm.weight", &[Q_RANK], WidenToF32),
            ("self_attn.kv_a_layernorm.weight", &[LATENT], WidenToF32),
            ("self_attn.kv_a_proj_with_mqa.weight", &[LATENT + 64, HIDDEN], AsStored),
            ("self_attn.learnable_sink_param", &[HEADS], WidenToF32),
            ("self_attn.post_proj_norm.weight", &[HIDDEN], WidenToF32),
        ] {
            self.raw(format!("{base}{suffix}"), shape, transform)?;
        }

        for (suffix, inputs, outputs) in [
            ("self_attn.q_a_proj", HIDDEN, Q_RANK),
            ("self_attn.q_b_proj", Q_RANK, HEAD_OUTPUT),
            ("self_attn.linear_gate", HIDDEN, HEAD_OUTPUT),
            ("self_attn.o_proj", HEAD_OUTPUT, HIDDEN),
            ("mlp.gate_proj", HIDDEN, INTERMEDIATE),
            ("mlp.up_proj", HIDDEN, INTERMEDIATE),
            ("mlp.down_proj", INTERMEDIATE, HIDDEN),
        ] {
            let name = format!("{base}{suffix}");
            self.projection(format!("{name}.weight"), name, inputs, outputs, Linear)?;
        }
        let name = format!("{base}self_attn.kv_b_proj");
        for (suffix, use_as) in [
            (
                "attn_k_b.weight",
                AbsorbedKey {
                    heads: HEADS,
                },
            ),
            (
                "attn_v_b.weight",
                AbsorbedValue {
                    heads: HEADS,
                },
            ),
        ] {
            self.projection(format!("{base}self_attn.{suffix}"), name.clone(), LATENT, HEADS * (192 + 256), use_as)?;
        }
        Ok(())
    }

    fn global(&mut self) -> Result<()> {
        use StqRawTransform::{AsStored, WidenToF32};

        for (suffix, shape, transform) in [
            ("mtp_fusion.enorm.weight", &[HIDDEN][..], WidenToF32),
            ("mtp_fusion.hnorm.weight", &[HIDDEN], WidenToF32),
            ("mtp_fusion.gate", &[HIDDEN], WidenToF32),
            ("hidden_correction.embed_norm.weight", &[HIDDEN], WidenToF32),
            ("hidden_correction.hidden_norm.weight", &[HIDDEN], WidenToF32),
            ("confidence_head.proj.weight", &[1, HIDDEN], AsStored),
            ("confidence_head.proj.bias", &[1], WidenToF32),
            ("final_norm.weight", &[HIDDEN], WidenToF32),
        ] {
            self.raw(format!("{PREFIX}{suffix}"), shape, transform)?;
        }
        for (suffix, inputs) in [
            ("mtp_fusion.eh_proj", 2 * HIDDEN),
            ("hidden_correction.gate_proj", 2 * HIDDEN),
            ("hidden_correction.up_proj", 2 * HIDDEN),
            ("hidden_correction.down_proj", HIDDEN),
        ] {
            let name = format!("{PREFIX}{suffix}");
            self.projection(format!("{name}.weight"), name, inputs, HIDDEN, StqProjectionUse::Linear)?;
        }
        Ok(())
    }

    fn raw(
        &mut self,
        name: String,
        shape: &[u32],
        transform: StqRawTransform,
    ) -> Result<()> {
        let invalid = || StqDflyError::Tensor(name.clone());
        let tensor = self.source.tensor(&name).ok_or_else(invalid)?;
        let location = self.source.tensor_location(&name).ok_or_else(invalid)?;
        if tensor.dtype != StqDtype::BF16 || tensor.shape != shape || location.shard_name != SHARD {
            return Err(invalid());
        }
        self.insert(
            name.clone(),
            StqTrunkWeight::Tensor {
                source: StqNamedTensor {
                    name,
                    dtype: tensor.dtype,
                    shape: &tensor.shape,
                    location,
                },
                transform,
            },
        )
    }

    fn projection(
        &mut self,
        alias: String,
        name: String,
        inputs: u32,
        outputs: u32,
        use_as: StqProjectionUse,
    ) -> Result<()> {
        let projection = self
            .source
            .stq_projection(&name)
            .filter(|p| {
                p.format == StqFormat::Spec1K4
                    && p.banks == 1
                    && p.inputs == inputs
                    && p.outputs == outputs
                    && [&p.trellis, &p.input_scale, &p.output_scale]
                        .into_iter()
                        .all(|span| span.shard_name == SHARD && span.offset.is_multiple_of(4))
            })
            .ok_or_else(|| StqDflyError::Tensor(name.clone()))?;
        self.insert(
            alias,
            StqTrunkWeight::Projection {
                source: StqNamedProjection {
                    name,
                    projection,
                },
                use_as,
            },
        )
    }

    fn insert(
        &mut self,
        alias: String,
        weight: StqTrunkWeight<'a>,
    ) -> Result<()> {
        match self.weights.entry(alias) {
            Entry::Vacant(entry) => {
                entry.insert(weight);
                Ok(())
            },
            Entry::Occupied(entry) => Err(StqDflyError::Duplicate(entry.key().clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, path::Path};

    use super::*;

    #[stark_engine_macros::stark_test]
    #[ignore] // Requires STARK_HY4_STQ_TEST_PACK pointing to the pinned artifact.
    fn binds_pinned_dfly_headers() {
        let path = std::env::var_os("STARK_HY4_STQ_TEST_PACK").expect("set the pinned STQ pack path");
        let source = StqSource::open(Path::new(&path)).unwrap();
        let draft = source.stq_dfly().unwrap();
        assert_eq!(draft.targets, [25, 29, 33, 37, 41]);
        assert_eq!(draft.mask_token, 120_009);
        assert_eq!(draft.drafts, 5);
        assert_eq!(draft.weights().len(), 92);

        let mut tensors = BTreeSet::new();
        let (mut packed_bytes, mut raw_bytes, mut widened) = (0, 0, 0);
        for (alias, binding) in draft.weights() {
            match binding {
                StqTrunkWeight::Tensor {
                    source,
                    transform,
                } => {
                    assert_eq!(alias, source.name);
                    assert!(tensors.insert(source.name.as_str()));
                    raw_bytes += source.location.byte_len;
                    widened += usize::from(*transform == StqRawTransform::WidenToF32);
                },
                StqTrunkWeight::Projection {
                    source,
                    use_as,
                } => {
                    let p = &source.projection;
                    assert_eq!(p.format, StqFormat::Spec1K4);
                    assert_eq!(p.trellis.byte_len, (p.inputs as usize / 16) * (p.outputs as usize / 16) * 128);
                    if tensors.insert(&source.name) {
                        packed_bytes += p.trellis.byte_len + p.input_scale.byte_len + p.output_scale.byte_len;
                    }
                    if alias.ends_with(".attn_v_b.weight") {
                        assert_eq!(
                            *use_as,
                            StqProjectionUse::AbsorbedValue {
                                heads: 32
                            }
                        );
                        let key = alias.replace(".attn_v_b.weight", ".attn_k_b.weight");
                        let StqTrunkWeight::Projection {
                            source: k,
                            use_as: k_use,
                        } = &draft.weights[&key]
                        else {
                            panic!("missing absorbed key alias");
                        };
                        assert_eq!(
                            *k_use,
                            StqProjectionUse::AbsorbedKey {
                                heads: 32
                            }
                        );
                        assert_eq!(k.projection.trellis.offset, p.trellis.offset);
                    }
                },
                _ => panic!("unexpected DFly2 binding {alias}"),
            }
        }
        assert_eq!(tensors.len(), 87);
        assert_eq!(widened, 37);
        assert_eq!(packed_bytes, 419_324_928);
        assert_eq!(raw_bytes, 17_845_570);
        assert_eq!(source.tensor_names().filter(|name| name.starts_with(PREFIX)).count(), 175);
        source.verify_unchanged().unwrap();
    }
}
