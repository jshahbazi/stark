// SPDX-License-Identifier: Apache-2.0
//! Qwen's F32 device graph. Host work supplies immutable embeddings. The
//! opt-in Edge0 path additionally reads bounded router-score rows for exact
//! host F64 routing; other activations remain on the device until completion.

#[path = "edge0_metal.rs"]
mod edge0_metal;

pub(super) const EDGE0_PREFILL_BATCH: usize = edge0_metal::MAX_BATCH;
// Request accounting conservatively charges three coexisting segment caps.
pub(super) const EDGE0_GRAPH_SCRATCH_BYTES: usize = edge0_metal::SEGMENT_SCRATCH;

use super::{Attention, Geometry, HyperConnection, Mixer, Ple, QwenModel};
use crate::engine::hybrid::{
    Error, Result,
    weight_metal::{DecodeGraph, DeviceVector, QwenGdnGeometry, QwenGdnState},
    weights::Weights,
};

// One compact copy of at most eight final four-stream residual rows. The
// 320 KiB peak fits the head leases existing 16 MiB bounded-output reserve.
fn edge0_wide_elements(
    count: usize,
    width: usize,
    readout: bool,
) -> Option<usize> {
    if count == 0 || count > EDGE0_PREFILL_BATCH || width == 0 || width > 10_240 || (readout && count != 1) {
        return None;
    }
    count.checked_mul(width)
}

struct HyperControls {
    norm: DeviceVector,
}

struct AttentionCache {
    capacity: usize,
    keys: DeviceVector,
    values: DeviceVector,
    index_keys: DeviceVector,
    pooled_index_keys: DeviceVector,
    index_queries: DeviceVector,
    query: DeviceVector,
    gates: DeviceVector,
    visible: DeviceVector,
    block_scores: DeviceVector,
    top_ids: DeviceVector,
    top_scores: DeviceVector,
}

struct ScalarAttentionWorkspace {
    capacity: usize,
    scores: DeviceVector,
    probabilities: DeviceVector,
}

struct AttentionControls {
    q_norm: DeviceVector,
    k_norm: DeviceVector,
    index_q_norm: DeviceVector,
    index_k_norm: DeviceVector,
    cache: Option<AttentionCache>,
    scalar: Option<ScalarAttentionWorkspace>,
}

impl AttentionControls {
    fn new(attention: &Attention) -> Result<Self> {
        Ok(Self {
            q_norm: DecodeGraph::upload(&attention.q_norm)?,
            k_norm: DecodeGraph::upload(&attention.k_norm)?,
            index_q_norm: DecodeGraph::upload(&attention.index_q_norm)?,
            index_k_norm: DecodeGraph::upload(&attention.index_k_norm)?,
            cache: None,
            scalar: None,
        })
    }

    /// Grow only for the visible request prefix. A checkpoint's advertised
    /// maximum context must never cause a maximum-sized allocation at load.
    fn ensure_cache(
        &mut self,
        graph: &mut DecodeGraph,
        g: &Geometry,
        position: usize,
    ) -> Result<()> {
        let count = position.checked_add(1).ok_or(Error::Invalid("Qwen graph attention position"))?;
        if self.cache.as_ref().is_some_and(|cache| cache.capacity >= count) {
            return Ok(());
        }
        let capacity = count
            .checked_next_power_of_two()
            .ok_or(Error::Invalid("Qwen graph attention capacity"))?
            .max(16)
            .min(g.context);
        let key_width = g.kv_heads * g.head_dim;
        let size = |width: usize| capacity.checked_mul(width).ok_or(Error::Invalid("Qwen graph cache size"));
        let cache = AttentionCache {
            capacity,
            keys: DecodeGraph::zeros(size(key_width)?)?,
            values: DecodeGraph::zeros(size(key_width)?)?,
            index_keys: DecodeGraph::zeros(size(g.index_dim)?)?,
            pooled_index_keys: DecodeGraph::zeros((capacity / g.ratio).max(1) * g.index_dim)?,
            index_queries: DecodeGraph::zeros(g.index_heads * g.index_dim)?,
            query: DecodeGraph::zeros(g.heads * g.head_dim)?,
            gates: DecodeGraph::zeros(g.heads * g.head_dim)?,
            visible: DecodeGraph::zeros(capacity)?,
            block_scores: DecodeGraph::zeros((capacity / g.ratio).max(1))?,
            top_ids: DecodeGraph::zeros(g.budget / g.ratio)?,
            top_scores: DecodeGraph::zeros(g.budget / g.ratio)?,
        };
        if let Some(previous) = &self.cache
            && position != 0
        {
            for (source, destination, width) in [
                (&previous.keys, &cache.keys, key_width),
                (&previous.values, &cache.values, key_width),
                (&previous.index_keys, &cache.index_keys, g.index_dim),
            ] {
                let used = position.checked_mul(width).ok_or(Error::Invalid("Qwen graph cache copy"))?;
                graph.copy_into(&source.slice(0, used)?, &destination.slice(0, used)?)?;
            }
            let pooled_used = (position / g.ratio) * g.index_dim;
            if pooled_used != 0 {
                graph.copy_into(
                    &previous.pooled_index_keys.slice(0, pooled_used)?,
                    &cache.pooled_index_keys.slice(0, pooled_used)?,
                )?;
            }
        }
        self.cache = Some(cache);
        Ok(())
    }

    /// Scalar attention needs its own row stride, independent of the KV
    /// cache's capacity. Normal decode stops using these arrays at 256 tokens;
    /// retain their small owners rather than doubling them with every KV grow.
    fn ensure_scalar_workspace(
        &mut self,
        g: &Geometry,
        count: usize,
    ) -> Result<()> {
        if self.scalar.as_ref().is_some_and(|scalar| scalar.capacity >= count) {
            return Ok(());
        }
        let capacity = count
            .checked_next_power_of_two()
            .ok_or(Error::Invalid("Qwen scalar attention capacity"))?
            .max(16)
            .min(g.context);
        let elements = capacity.checked_mul(g.heads).ok_or(Error::Invalid("Qwen scalar attention size"))?;
        self.scalar = Some(ScalarAttentionWorkspace {
            capacity,
            scores: DecodeGraph::zeros(elements)?,
            probabilities: DecodeGraph::zeros(elements)?,
        });
        Ok(())
    }

    fn encode(
        &mut self,
        graph: &mut DecodeGraph,
        weights: &Weights,
        attention: &Attention,
        input: &DeviceVector,
        g: &Geometry,
        position: usize,
    ) -> Result<DeviceVector> {
        self.encode_projected(graph, weights, attention, input, g, position, None)
    }

    #[expect(clippy::too_many_arguments, reason = "explicit causal position and optional preprojected token row")]
    fn encode_projected(
        &mut self,
        graph: &mut DecodeGraph,
        weights: &Weights,
        attention: &Attention,
        input: &DeviceVector,
        g: &Geometry,
        position: usize,
        projected: Option<&[DeviceVector; 4]>,
    ) -> Result<DeviceVector> {
        self.ensure_cache(graph, g, position)?;
        let count = position + 1;
        let partial_len = g.attention_partial_elements(count);
        let scalar_attention = count < 256 || partial_len.is_none() || super::forced_scalar_attention();
        if scalar_attention {
            self.ensure_scalar_workspace(g, count)?;
        }
        let cache = self.cache.as_ref().ok_or(Error::Invalid("Qwen graph attention cache"))?;
        let project = |graph: &mut DecodeGraph, name: &str| {
            weights.encode_matvec(graph, &format!("{}.{name}.weight", attention.prefix), input)
        };
        let [index, query_and_gate, keys, values] = if let Some(projections) = projected {
            std::array::from_fn(|index| projections[index].clone())
        } else {
            graph.begin_projection_level()?;
            let projections = [
                project(graph, "indexer.index_qk_proj")?,
                project(graph, "q_proj")?,
                project(graph, "k_proj")?,
                project(graph, "v_proj")?,
            ];
            graph.end_projection_level()?;
            projections
        };
        let scalar_prepare = std::env::var_os("STARK_QWEN38_SCALAR_QSA_PREPARE").is_some_and(|value| value == "1");
        graph.kernel(
            if scalar_prepare {
                "qwen_graph_qsa_prepare"
            } else {
                "qwen_graph_qsa_prepare_parallel"
            },
            &[
                &index,
                &query_and_gate,
                &keys,
                &values,
                &self.q_norm,
                &self.k_norm,
                &cache.index_keys,
                &cache.query,
                &cache.keys,
                &cache.values,
                &cache.gates,
            ],
            &[
                count as u32,
                g.index_heads as u32,
                g.index_dim as u32,
                g.heads as u32,
                g.head_dim as u32,
                g.kv_heads as u32,
                g.rotary_dim as u32,
                position as u32,
                g.eps.to_bits(),
                g.rope_theta.to_bits(),
            ],
            (g.heads + g.kv_heads + 1)
                * if scalar_prepare {
                    1
                } else {
                    256
                },
            if scalar_prepare {
                1
            } else {
                256
            },
        )?;
        let visibility_parameters = [
            count as u32,
            g.index_heads as u32,
            g.index_dim as u32,
            g.rotary_dim as u32,
            position as u32,
            g.eps.to_bits(),
            g.rope_theta.to_bits(),
            g.ratio as u32,
            g.budget as u32,
        ];
        if std::env::var_os("STARK_QWEN38_SCALAR_QSA_SELECTION").is_some_and(|value| value == "1") {
            graph.kernel(
                "qwen_graph_qsa_visible",
                &[
                    &index,
                    &cache.index_keys,
                    &self.index_q_norm,
                    &self.index_k_norm,
                    &cache.visible,
                    &cache.block_scores,
                    &cache.top_ids,
                    &cache.top_scores,
                ],
                &visibility_parameters,
                1,
                1,
            )?;
        } else {
            let completed_block = count.is_multiple_of(g.ratio);
            if count > g.budget || completed_block {
                let mut preparation_parameters = visibility_parameters.to_vec();
                preparation_parameters.extend([
                    if completed_block {
                        (count / g.ratio - 1) as u32
                    } else {
                        0
                    },
                    u32::from(completed_block),
                ]);
                graph.kernel(
                    "qwen_graph_qsa_index_prepare",
                    &[
                        &index,
                        &cache.index_keys,
                        &self.index_q_norm,
                        &self.index_k_norm,
                        &cache.index_queries,
                        &cache.pooled_index_keys,
                    ],
                    &preparation_parameters,
                    g.index_heads + usize::from(completed_block),
                    1,
                )?;
            }
            if count > g.budget {
                let blocks = count / g.ratio;
                graph.kernel(
                    "qwen_graph_qsa_index_cached_scores",
                    &[&cache.index_queries, &cache.pooled_index_keys, &cache.block_scores],
                    &[blocks as u32, g.index_heads as u32, g.index_dim as u32],
                    blocks,
                    64,
                )?;
                graph.kernel(
                    "qwen_graph_qsa_index_threshold",
                    &[&cache.block_scores, &cache.top_ids, &cache.top_scores],
                    &[blocks as u32, (g.budget / g.ratio).min(blocks) as u32],
                    256,
                    256,
                )?;
            }
            graph.kernel(
                "qwen_graph_qsa_index_mask",
                &[&cache.block_scores, &cache.top_ids, &cache.top_scores, &cache.visible],
                &[count as u32, g.ratio as u32, g.budget as u32],
                count,
                256,
            )?;
        }
        let output = graph.scratch(g.heads * g.head_dim)?;
        let splits = count.div_ceil(256);
        if scalar_attention {
            let scalar = self.scalar.as_ref().ok_or(Error::Invalid("Qwen scalar attention workspace"))?;
            let parameters =
                [count as u32, g.heads as u32, g.head_dim as u32, g.kv_heads as u32, scalar.capacity as u32];
            graph.kernel(
                "qwen_graph_qsa_scores",
                &[&cache.query, &cache.keys, &cache.visible, &scalar.scores],
                &parameters,
                g.heads * count,
                64,
            )?;
            graph.kernel(
                "qwen_graph_qsa_softmax",
                &[&scalar.scores, &cache.visible, &scalar.probabilities],
                &[count as u32, g.heads as u32, scalar.capacity as u32],
                g.heads,
                1,
            )?;
            graph.kernel(
                "qwen_graph_qsa_output",
                &[&scalar.probabilities, &cache.values, &cache.gates, &cache.visible, &output],
                &parameters,
                g.heads * g.head_dim,
                256,
            )?;
        } else {
            let partials = graph.scratch(partial_len.ok_or(Error::Invalid("Qwen split attention workspace"))?)?;
            graph.kernel(
                "qwen_graph_qsa_split_attention",
                &[&cache.query, &cache.keys, &cache.values, &cache.visible, &partials],
                &[count as u32, g.heads as u32, g.head_dim as u32, g.kv_heads as u32, splits as u32],
                g.heads * splits * 256,
                256,
            )?;
            graph.kernel(
                "qwen_graph_qsa_combine_attention",
                &[&partials, &cache.gates, &output],
                &[g.heads as u32, g.head_dim as u32, splits as u32],
                g.heads * g.head_dim,
                256,
            )?;
        }
        weights.encode_matvec(graph, &format!("{}.o_proj.weight", attention.prefix), &output)
    }
}

struct PleControls {
    key_norm: DeviceVector,
    query_norm: DeviceVector,
    conv_norm: DeviceVector,
    convolution: DeviceVector,
    history: DeviceVector,
    scratch: DeviceVector,
}

impl PleControls {
    fn new(
        ple: &Ple,
        g: &Geometry,
    ) -> Result<Self> {
        let width = g.streams * g.dim;
        Ok(Self {
            key_norm: DecodeGraph::upload(&ple.key_norm)?,
            query_norm: DecodeGraph::upload(&ple.query_norm)?,
            conv_norm: DecodeGraph::upload(&ple.conv_norm)?,
            convolution: DecodeGraph::upload(&ple.convolution)?,
            history: DecodeGraph::zeros(width * (g.ple_kernel - 1) * g.ngram_size)?,
            scratch: DecodeGraph::zeros(width)?,
        })
    }

    fn encode(
        &self,
        graph: &mut DecodeGraph,
        weights: &Weights,
        ple: &Ple,
        embedding: &DeviceVector,
        input: &DeviceVector,
        g: &Geometry,
    ) -> Result<DeviceVector> {
        graph.begin_projection_level()?;
        let key = weights.encode_matvec(graph, &format!("{}.key_proj.weight", ple.prefix), embedding)?;
        let value = weights.encode_matvec(graph, &format!("{}.value_proj.weight", ple.prefix), embedding)?;
        graph.end_projection_level()?;
        let output = graph.scratch(g.streams * g.dim)?;
        let scalar = std::env::var_os("STARK_QWEN38_SCALAR_PLE").is_some_and(|value| value == "1");
        graph.kernel(
            if scalar {
                "qwen_graph_ple"
            } else {
                "qwen_graph_ple_parallel"
            },
            &[
                &key,
                &value,
                input,
                &self.key_norm,
                &self.query_norm,
                &self.conv_norm,
                &self.convolution,
                &self.history,
                &output,
                &self.scratch,
            ],
            &[g.streams as u32, g.dim as u32, g.ple_kernel as u32, g.ngram_size as u32, g.eps.to_bits()],
            if scalar {
                g.streams
            } else {
                g.streams * 256
            },
            if scalar {
                1
            } else {
                256
            },
        )?;
        graph.kernel(
            "qwen_graph_ple_convolve",
            &[&self.convolution, &self.history, &output, &self.scratch],
            &[g.streams as u32, g.dim as u32, g.ple_kernel as u32, g.ngram_size as u32],
            g.streams * g.dim,
            256,
        )?;
        Ok(output)
    }
}

struct LayerControls {
    attention_hyper: HyperControls,
    mlp_hyper: HyperControls,
    attention: Option<AttentionControls>,
    ple: Option<PleControls>,
    edge0: Option<edge0_metal::Controls>,
}

pub(super) struct State {
    layers: Vec<LayerControls>,
    final_norm: DeviceVector,
}

/// Opaque, private host state. Current query/gate/selection buffers and scalar
/// attention workspaces are overwritten before reading and are not captured.
pub(super) struct Checkpoint {
    position: usize,
    geometry: [usize; 4],
    layers: Vec<LayerCheckpoint>,
}

struct LayerCheckpoint {
    attention: Option<AttentionCheckpoint>,
    ple_history: Option<Box<[f32]>>,
}

struct AttentionCheckpoint {
    capacity: usize,
    keys: Box<[f32]>,
    values: Box<[f32]>,
    // The raw prefix also retains the incomplete pooling block. Completed
    // pooled rows are captured separately; there is no other partial owner.
    index_keys: Box<[f32]>,
    pooled_index_keys: Box<[f32]>,
}

impl Checkpoint {
    pub(super) fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                let attention = layer.attention.as_ref().map_or(0, |value| {
                    value.keys.len() + value.values.len() + value.index_keys.len() + value.pooled_index_keys.len()
                });
                (attention + layer.ple_history.as_ref().map_or(0, |values| values.len())) * 4
            })
            .sum()
    }
}

fn checkpoint_geometry(g: &Geometry) -> [usize; 4] {
    [g.context, g.kv_heads * g.head_dim, g.index_dim, g.ratio]
}

fn checkpoint_attention_replacement_bytes(
    g: &Geometry,
    capacity: usize,
) -> Option<usize> {
    if g.ratio == 0 || capacity == 0 || capacity > g.context {
        return None;
    }
    // Full padded device inventory, including overwritten query/selection
    // scratch, plus the largest zero-padded host upload still in scope.
    g.graph_cache_bytes(capacity)?
        .checked_add(capacity.checked_mul((g.kv_heads * g.head_dim).max(g.index_dim))?.checked_mul(4)?)
}

fn checkpoint_attention(
    saved: &AttentionCheckpoint,
    g: &Geometry,
    position: usize,
) -> Result<AttentionCache> {
    let key_width = g.kv_heads * g.head_dim;
    if saved.capacity < position
        || saved.capacity > g.context
        || saved.capacity == 0
        || saved.keys.len() != position * key_width
        || saved.values.len() != position * key_width
        || saved.index_keys.len() != position * g.index_dim
        || saved.pooled_index_keys.len() != (position / g.ratio) * g.index_dim
    {
        return Err(Error::Invalid("QSA checkpoint geometry"));
    }
    let restore = |values: &[f32], len: usize| -> Result<DeviceVector> {
        if values.len() > len || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("QSA checkpoint state values"));
        }
        let mut storage = vec![0.0_f32; len];
        storage[..values.len()].copy_from_slice(values);
        DecodeGraph::upload(&storage)
    };
    let capacity = saved.capacity;
    Ok(AttentionCache {
        capacity,
        keys: restore(&saved.keys, capacity * key_width)?,
        values: restore(&saved.values, capacity * key_width)?,
        index_keys: restore(&saved.index_keys, capacity * g.index_dim)?,
        pooled_index_keys: restore(&saved.pooled_index_keys, (capacity / g.ratio).max(1) * g.index_dim)?,
        index_queries: DecodeGraph::zeros(g.index_heads * g.index_dim)?,
        query: DecodeGraph::zeros(g.heads * g.head_dim)?,
        gates: DecodeGraph::zeros(g.heads * g.head_dim)?,
        visible: DecodeGraph::zeros(capacity)?,
        block_scores: DecodeGraph::zeros((capacity / g.ratio).max(1))?,
        top_ids: DecodeGraph::zeros(g.budget / g.ratio)?,
        top_scores: DecodeGraph::zeros(g.budget / g.ratio)?,
    })
}

impl State {
    pub(super) fn checkpoint_replacement_bytes(
        &self,
        saved: &Checkpoint,
        g: &Geometry,
    ) -> Option<usize> {
        if saved.layers.len() != self.layers.len() || saved.geometry != checkpoint_geometry(g) {
            return None;
        }
        self.layers.iter().zip(&saved.layers).try_fold(0_usize, |bytes, (source, saved)| {
            let attention = match (&source.attention, &saved.attention) {
                (Some(_), Some(saved)) => checkpoint_attention_replacement_bytes(g, saved.capacity)?,
                (None, None) => 0,
                _ => return None,
            };
            let ple = match (&source.ple, &saved.ple_history) {
                (Some(source), Some(history)) if history.len() == source.history.len => {
                    source.history.len.checked_add(source.scratch.len)?.checked_mul(4)?
                },
                (None, None) => 0,
                _ => return None,
            };
            bytes.checked_add(attention)?.checked_add(ple)
        })
    }

    pub(super) fn capture_checkpoint(
        &self,
        g: &Geometry,
        position: usize,
    ) -> Result<Checkpoint> {
        if position > g.context || g.ratio == 0 {
            return Err(Error::Invalid("graph checkpoint position"));
        }
        let mut views = Vec::new();
        let mut append = |values: &DeviceVector, len: usize| -> Result<Option<usize>> {
            if len == 0 {
                return Ok(None);
            }
            let index = views.len();
            views.push(values.slice(0, len)?);
            Ok(Some(index))
        };
        let mut plans = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let attention = match &layer.attention {
                Some(control) => match &control.cache {
                    Some(cache) => {
                        if cache.capacity < position {
                            return Err(Error::Invalid("QSA checkpoint source capacity"));
                        }
                        let key_count = position * g.kv_heads * g.head_dim;
                        Some((
                            cache.capacity,
                            [
                                append(&cache.keys, key_count)?,
                                append(&cache.values, key_count)?,
                                append(&cache.index_keys, position * g.index_dim)?,
                                append(&cache.pooled_index_keys, (position / g.ratio) * g.index_dim)?,
                            ],
                        ))
                    },
                    None if position == 0 => None,
                    None => return Err(Error::Invalid("QSA checkpoint source missing")),
                },
                None => None,
            };
            let ple = layer.ple.as_ref().map(|ple| append(&ple.history, ple.history.len)).transpose()?.flatten();
            plans.push((attention, ple));
        }
        // Completion fences prior target work, validates all returned spans,
        // and copies only exact causal state into private RAM.
        let mut captured = DecodeGraph::new()?
            .finish(&views)?
            .into_iter()
            .map(|values| Some(values.into_boxed_slice()))
            .collect::<Vec<_>>();
        let mut take = |index: Option<usize>| -> Box<[f32]> {
            index.map_or_else(Box::<[f32]>::default, |index| captured[index].take().unwrap())
        };
        let layers = plans
            .into_iter()
            .map(|(attention, ple)| LayerCheckpoint {
                attention: attention.map(|(capacity, indices)| AttentionCheckpoint {
                    capacity,
                    keys: take(indices[0]),
                    values: take(indices[1]),
                    index_keys: take(indices[2]),
                    pooled_index_keys: take(indices[3]),
                }),
                ple_history: ple.map(|index| take(Some(index))),
            })
            .collect();
        Ok(Checkpoint {
            position,
            geometry: checkpoint_geometry(g),
            layers,
        })
    }

    /// Build an independent mutable state set. Only immutable norm/scale/
    /// convolution controls share owners with the current graph.
    pub(super) fn stage_checkpoint(
        &self,
        saved: &Checkpoint,
        g: &Geometry,
        position: usize,
    ) -> Result<Self> {
        if saved.position != position
            || saved.geometry != checkpoint_geometry(g)
            || saved.layers.len() != self.layers.len()
            || position > g.context
            || g.ratio == 0
        {
            return Err(Error::Invalid("graph checkpoint policy or layer inventory"));
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for (source, saved) in self.layers.iter().zip(&saved.layers) {
            let attention = match (&source.attention, &saved.attention) {
                (Some(source), saved) => Some(AttentionControls {
                    q_norm: source.q_norm.clone(),
                    k_norm: source.k_norm.clone(),
                    index_q_norm: source.index_q_norm.clone(),
                    index_k_norm: source.index_k_norm.clone(),
                    cache: match saved {
                        Some(value) => Some(checkpoint_attention(value, g, position)?),
                        None if position == 0 => None,
                        None => return Err(Error::Invalid("graph checkpoint attention missing")),
                    },
                    scalar: None,
                }),
                (None, None) => None,
                _ => return Err(Error::Invalid("graph checkpoint attention inventory")),
            };
            let ple = match (&source.ple, &saved.ple_history) {
                (Some(source), Some(history)) if history.len() == source.history.len => Some(PleControls {
                    key_norm: source.key_norm.clone(),
                    query_norm: source.query_norm.clone(),
                    conv_norm: source.conv_norm.clone(),
                    convolution: source.convolution.clone(),
                    history: DecodeGraph::upload(history)?,
                    scratch: DecodeGraph::zeros(source.scratch.len)?,
                }),
                (None, None) => None,
                _ => return Err(Error::Invalid("graph checkpoint PLE inventory")),
            };
            layers.push(LayerControls {
                attention_hyper: HyperControls {
                    norm: source.attention_hyper.norm.clone(),
                },
                mlp_hyper: HyperControls {
                    norm: source.mlp_hyper.norm.clone(),
                },
                attention,
                ple,
                edge0: source.edge0.clone(),
            });
        }
        Ok(Self {
            layers,
            final_norm: self.final_norm.clone(),
        })
    }

    fn new(model: &QwenModel) -> Result<Self> {
        if model.geometry.edge0 && !model.geometry.edge0_metal {
            return Err(Error::Unsupported("Edge0 reference requires scalar control"));
        }
        let mut layers = Vec::with_capacity(model.layers.len());
        for (index, layer) in model.layers.iter().enumerate() {
            layers.push(LayerControls {
                attention_hyper: HyperControls {
                    norm: DecodeGraph::upload(&layer.attn_hyper.norm)?,
                },
                mlp_hyper: HyperControls {
                    norm: DecodeGraph::upload(&layer.mlp_hyper.norm)?,
                },
                attention: match &layer.mixer {
                    Mixer::Attention(attention) => Some(AttentionControls::new(attention)?),
                    Mixer::Delta(_) => None,
                },
                ple: layer.ple.as_ref().map(|ple| PleControls::new(ple, &model.geometry)).transpose()?,
                edge0: model.geometry.edge0.then(|| edge0_metal::Controls::new(&model.weights, index)).transpose()?,
            });
        }
        Ok(Self {
            layers,
            final_norm: DecodeGraph::upload(&model.final_mix.norm)?,
        })
    }

    fn encode_step(
        &mut self,
        model: &mut QwenModel,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        if model.geometry.edge0 {
            return self.encode_edge0_batch(model, &[token], readout, None);
        }
        let g = &model.geometry;
        let embedding = model.weights.row(&model.embedding, token as usize)?;
        // Prepare every n-gram input before any device submission. These
        // lookups are independent of hidden activations and retain I64 hashes.
        let ple_embeddings = model
            .layers
            .iter()
            .map(|layer| {
                layer
                    .ple
                    .as_ref()
                    .map(|ple| {
                        let values = ple.gather(&model.weights, token, &model.history, g)?;
                        DecodeGraph::upload(&values)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut graph = if std::env::var_os("STARK_QWEN38_SINGLE_ENCODER").is_some_and(|value| value == "1") {
            DecodeGraph::new_parallel()?
        } else {
            DecodeGraph::new()?
        };
        let mut hidden = DecodeGraph::upload(&embedding.repeat(g.streams))?;
        for (index, layer) in model.layers.iter_mut().enumerate() {
            let controls = &mut self.layers[index];
            if let (Some(ple), Some(ple_controls), Some(embedding)) =
                (&layer.ple, &controls.ple, &ple_embeddings[index])
            {
                let extra = ple_controls.encode(&mut graph, &model.weights, ple, embedding, &hidden, g)?;
                hidden = graph.add(&hidden, &extra)?;
                graph.diagnostic_checkpoint("qwen.phase.ple")?;
            }
            let (input, gates) =
                mix(&mut graph, &model.weights, &layer.attn_hyper, &controls.attention_hyper.norm, &hidden, g)?;
            graph.diagnostic_checkpoint("qwen.phase.attention_mix")?;
            let output = match &mut layer.mixer {
                Mixer::Delta(mixer) => {
                    if mixer.metal.is_none() {
                        mixer.metal = Some(QwenGdnState::new(
                            QwenGdnGeometry {
                                key_heads: g.key_heads,
                                key_dim: g.key_dim,
                                value_heads: g.value_heads,
                                value_dim: g.value_dim,
                                conv_kernel: g.conv_kernel,
                                eps: g.eps,
                                sigmoid_output: g.sigmoid_output,
                            },
                            &mixer.convolution,
                            &mixer.a_log,
                            &mixer.dt_bias,
                            &mixer.norm,
                        )?);
                    }
                    graph.qwen_gdn(
                        &model.weights,
                        &mixer.prefix,
                        &input,
                        mixer.metal.as_mut().ok_or(Error::Invalid("Qwen graph GDN state"))?,
                    )?
                },
                Mixer::Attention(attention) => controls
                    .attention
                    .as_mut()
                    .ok_or(Error::Invalid("Qwen graph QSA controls"))?
                    .encode(&mut graph, &model.weights, attention, &input, g, model.position)?,
            };
            graph.diagnostic_checkpoint(match &layer.mixer {
                Mixer::Delta(_) => "qwen.phase.gdn",
                Mixer::Attention(_) => "qwen.phase.qsa",
            })?;
            hidden = inject(&mut graph, &hidden, &output, &gates, g)?;
            let (input, gates) =
                mix(&mut graph, &model.weights, &layer.mlp_hyper, &controls.mlp_hyper.norm, &hidden, g)?;
            graph.diagnostic_checkpoint("qwen.phase.ffn_mix")?;
            let output = moe(&mut graph, &model.weights, &format!("{}.mlp", layer.prefix), &input, g)?;
            graph.diagnostic_checkpoint("qwen.phase.moe")?;
            hidden = inject(&mut graph, &hidden, &output, &gates, g)?;
        }
        graph.check_finite(&hidden)?;
        let outputs = if readout {
            let (input, _) = mix(&mut graph, &model.weights, &model.final_mix, &self.final_norm, &hidden, g)?;
            let logits = model.weights.encode_matvec(&mut graph, &model.readout, &input)?;
            graph.finish(&[logits])?
        } else {
            graph.finish(&[])?
        };
        let logits = if readout {
            outputs.into_iter().next().ok_or(Error::Invalid("Qwen graph readout missing"))?
        } else {
            Vec::new()
        };
        if readout && (logits.len() != g.vocab || logits.iter().any(|value| !value.is_finite())) {
            return Err(Error::Invalid("Qwen graph readout values"));
        }
        model.position += 1;
        model.history.push(token);
        if model.history.len() >= g.ngram_size {
            model.history.remove(0);
        }
        Ok(logits)
    }

    /// Layer-major, bounded causal prefill. Trunk projection rows are batched
    /// before sequential GDN/QSA state updates; the latent stage then groups
    /// those token rows by selected expert. Neither causal state nor U's
    /// original route ordering is reassociated.
    fn encode_edge0_batch(
        &mut self,
        model: &mut QwenModel,
        tokens: &[u32],
        readout: bool,
        wide_output: Option<&mut Option<DeviceVector>>,
    ) -> Result<Vec<f32>> {
        let g = &model.geometry;
        let count = tokens.len();
        let start = model.position;
        if !g.edge0_metal
            || count == 0
            || count > edge0_metal::MAX_BATCH
            || tokens.iter().any(|&token| token as usize >= g.vocab)
            || start.checked_add(count).is_none_or(|end| end > g.context)
            || (readout && count != 1)
        {
            return Err(Error::Invalid("Edge0 graph token batch boundary"));
        }
        let wide_elements = if wide_output.is_some() {
            Some(
                edge0_wide_elements(
                    count,
                    g.streams.checked_mul(g.dim).ok_or(Error::Invalid("Edge0 wide row width"))?,
                    readout,
                )
                .ok_or(Error::Invalid("Edge0 wide batch geometry"))?,
            )
        } else {
            None
        };
        // Immutable gathers use each token's chronological history. Runtime
        // model history is published only after the entire batch succeeds.
        let mut history = model.history.clone();
        let mut histories = Vec::with_capacity(count);
        for &token in tokens {
            histories.push(history.clone());
            history.push(token);
            if history.len() >= g.ngram_size {
                history.remove(0);
            }
        }
        let mut hidden = tokens
            .iter()
            .map(|&token| {
                let embedding = model.weights.row(&model.embedding, token as usize)?;
                DecodeGraph::upload(&embedding.repeat(g.streams))
            })
            .collect::<Result<Vec<_>>>()?;
        let ple_embeddings = model
            .layers
            .iter()
            .map(|layer| {
                tokens
                    .iter()
                    .enumerate()
                    .map(|(index, &token)| {
                        layer
                            .ple
                            .as_ref()
                            .map(|ple| {
                                let values = ple.gather(&model.weights, token, &histories[index], g)?;
                                DecodeGraph::upload(&values)
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut graph = edge0_metal::new_graph()?;
        for (layer_index, layer) in model.layers.iter_mut().enumerate() {
            let controls = &mut self.layers[layer_index];
            let mut mixer_inputs = Vec::with_capacity(count);
            let mut attention_gates = Vec::with_capacity(count);
            for token in 0..count {
                if let (Some(ple), Some(ple_controls), Some(embedding)) =
                    (&layer.ple, &controls.ple, &ple_embeddings[layer_index][token])
                {
                    let extra = ple_controls.encode(&mut graph, &model.weights, ple, embedding, &hidden[token], g)?;
                    hidden[token] = graph.add(&hidden[token], &extra)?;
                }
                let (input, gates) = mix(
                    &mut graph,
                    &model.weights,
                    &layer.attn_hyper,
                    &controls.attention_hyper.norm,
                    &hidden[token],
                    g,
                )?;
                mixer_inputs.push(input);
                attention_gates.push(gates);
            }
            let packed_inputs = edge0_metal::concatenate(&mut graph, &mixer_inputs)?;
            let (prefix, names) = match &layer.mixer {
                Mixer::Delta(mixer) => (&mixer.prefix, ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]),
                Mixer::Attention(attention) => {
                    (&attention.prefix, ["indexer.index_qk_proj", "q_proj", "k_proj", "v_proj"])
                },
            };
            graph.begin_projection_level()?;
            let projected = names
                .iter()
                .map(|name| {
                    model.weights.encode_edge0_named_batch(
                        &mut graph,
                        &format!("{prefix}.{name}.weight"),
                        &packed_inputs,
                        count,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            graph.end_projection_level()?;
            let mut ffn_inputs = Vec::with_capacity(count);
            let mut ffn_gates = Vec::with_capacity(count);
            for token in 0..count {
                let projections: [DeviceVector; 4] = projected
                    .iter()
                    .map(|values| {
                        if !values.len.is_multiple_of(count) {
                            return Err(Error::Invalid("Edge0 graph projected row batch"));
                        }
                        let width = values.len / count;
                        values.slice(token * width, width)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| Error::Invalid("Edge0 graph mixer projection count"))?;
                let output = match &mut layer.mixer {
                    Mixer::Delta(mixer) => {
                        if mixer.metal.is_none() {
                            mixer.metal = Some(QwenGdnState::new_edge0(
                                QwenGdnGeometry {
                                    key_heads: g.key_heads,
                                    key_dim: g.key_dim,
                                    value_heads: g.value_heads,
                                    value_dim: g.value_dim,
                                    conv_kernel: g.conv_kernel,
                                    eps: g.eps,
                                    sigmoid_output: g.sigmoid_output,
                                },
                                &mixer.convolution,
                                &mixer.a_log,
                                &mixer.dt_bias,
                                &mixer.norm,
                            )?);
                        }
                        graph.qwen_gdn_projected(
                            &model.weights,
                            &mixer.prefix,
                            mixer.metal.as_mut().ok_or(Error::Invalid("Edge0 graph GDN state"))?,
                            &projections,
                        )?
                    },
                    Mixer::Attention(attention) => controls
                        .attention
                        .as_mut()
                        .ok_or(Error::Invalid("Edge0 graph attention controls"))?
                        .encode_projected(
                            &mut graph,
                            &model.weights,
                            attention,
                            &mixer_inputs[token],
                            g,
                            start + token,
                            Some(&projections),
                        )?,
                };
                hidden[token] = inject(&mut graph, &hidden[token], &output, &attention_gates[token], g)?;
                let (input, gates) =
                    mix(&mut graph, &model.weights, &layer.mlp_hyper, &controls.mlp_hyper.norm, &hidden[token], g)?;
                ffn_inputs.push(input);
                ffn_gates.push(gates);
            }
            let input = edge0_metal::concatenate(&mut graph, &ffn_inputs)?;
            let output = edge0_metal::encode(
                &mut graph,
                &model.weights,
                layer_index,
                &format!("{}.mlp", layer.prefix),
                &input,
                g,
                controls.edge0.as_ref().ok_or(Error::Invalid("Edge0 graph latent controls"))?,
                count,
            )?;
            for token in 0..count {
                hidden[token] =
                    inject(&mut graph, &hidden[token], &output.slice(token * g.dim, g.dim)?, &ffn_gates[token], g)?;
            }
        }
        for values in &hidden {
            graph.check_finite(values)?;
        }
        // The ordinary target path encodes no extra allocation or copy.
        // Copies follow all target arithmetic and preserve its batch schedule.
        let saved_wide = if let Some(elements) = wide_elements {
            let saved = DecodeGraph::zeros(elements)?;
            let width = g.streams * g.dim;
            for (row, values) in hidden.iter().enumerate() {
                graph.copy_into(values, &saved.slice(row * width, width)?)?;
            }
            Some(saved)
        } else {
            None
        };
        let outputs = if readout {
            let (input, _) = mix(&mut graph, &model.weights, &model.final_mix, &self.final_norm, &hidden[0], g)?;
            let logits = model.weights.encode_edge0_named_batch(&mut graph, &model.readout, &input, 1)?;
            model.weights.edge0_record_graph_scratch(graph.scratch_backing_bytes());
            graph.finish(&[logits])?
        } else {
            model.weights.edge0_record_graph_scratch(graph.scratch_backing_bytes());
            graph.finish(&[])?
        };
        let logits = if readout {
            outputs.into_iter().next().ok_or(Error::Invalid("Edge0 graph readout missing"))?
        } else {
            Vec::new()
        };
        if readout && (logits.len() != g.vocab || logits.iter().any(|v| !v.is_finite())) {
            return Err(Error::Invalid("Edge0 graph readout values"));
        }
        model.position += count;
        model.history = history;
        if let Some(output) = wide_output {
            *output = saved_wide;
        }
        Ok(logits)
    }
}

fn mix(
    graph: &mut DecodeGraph,
    weights: &Weights,
    hyper: &HyperConnection,
    norm: &DeviceVector,
    input: &DeviceVector,
    g: &Geometry,
) -> Result<(DeviceVector, DeviceVector)> {
    let inject = hyper.combine.then(|| format!("{}.block_inject_weight.weight", hyper.prefix));
    let normalized = graph.rms_norm(input, norm, g.dim, g.eps)?;
    graph.qwen_hyperconnection(
        weights,
        &format!("{}.input_mix_weight_down.weight", hyper.prefix),
        &format!("{}.input_mix_weight_up.weight", hyper.prefix),
        inject.as_deref(),
        &normalized,
        g.streams,
    )
}

fn inject(
    graph: &mut DecodeGraph,
    residual: &DeviceVector,
    output: &DeviceVector,
    gates: &DeviceVector,
    g: &Geometry,
) -> Result<DeviceVector> {
    let result = graph.scratch(g.streams * g.dim)?;
    graph.kernel(
        "qwen_graph_inject",
        &[residual, output, gates, &result],
        &[g.streams as u32, g.dim as u32],
        g.streams * g.dim,
        256,
    )?;
    Ok(result)
}

fn moe(
    graph: &mut DecodeGraph,
    weights: &Weights,
    prefix: &str,
    input: &DeviceVector,
    g: &Geometry,
) -> Result<DeviceVector> {
    graph.begin_projection_level()?;
    let scores = weights.encode_matvec(graph, &format!("{prefix}.gate.weight"), input)?;
    let shared_gate = weights.encode_matvec(graph, &format!("{prefix}.shared_expert_gate.weight"), input)?;
    let shared_values = weights.encode_matvec(graph, &format!("{prefix}.shared_expert.gate_proj.weight"), input)?;
    let shared_up = weights.encode_matvec(graph, &format!("{prefix}.shared_expert.up_proj.weight"), input)?;
    graph.end_projection_level()?;
    let indices = graph.scratch(g.active_experts)?;
    let probabilities = graph.scratch(g.active_experts)?;
    let scalar_router = std::env::var_os("STARK_QWEN38_SCALAR_ROUTER").is_some_and(|value| value == "1");
    graph.kernel(
        if scalar_router {
            "qwen_graph_router"
        } else {
            "qwen_graph_router_parallel"
        },
        &[&scores, &indices, &probabilities],
        &[g.experts as u32, g.active_experts as u32],
        if scalar_router {
            1
        } else {
            256
        },
        if scalar_router {
            1
        } else {
            256
        },
    )?;
    graph.begin_projection_level()?;
    let gate = weights.encode_indexed_matvec(
        graph,
        &format!("{prefix}.switch_mlp.gate_proj.weight"),
        &indices,
        g.active_experts,
        input,
        false,
    )?;
    let up = weights.encode_indexed_matvec(
        graph,
        &format!("{prefix}.switch_mlp.up_proj.weight"),
        &indices,
        g.active_experts,
        input,
        false,
    )?;
    graph.end_projection_level()?;
    let values = graph.swiglu(&gate, &up, None)?;
    let shared_values = graph.swiglu(&shared_values, &shared_up, None)?;
    graph.begin_projection_level()?;
    let down = weights.encode_indexed_matvec(
        graph,
        &format!("{prefix}.switch_mlp.down_proj.weight"),
        &indices,
        g.active_experts,
        &values,
        true,
    )?;
    let shared_down =
        weights.encode_matvec(graph, &format!("{prefix}.shared_expert.down_proj.weight"), &shared_values)?;
    graph.end_projection_level()?;
    let activated_gate = graph.scratch(1)?;
    graph.kernel("qwen_graph_sigmoid", &[&shared_gate, &activated_gate], &[1], 1, 1)?;
    graph.weighted_sum_device(&down, &probabilities, &shared_down, &activated_gate)
}

pub(super) fn step(
    model: &mut QwenModel,
    token: u32,
    readout: bool,
) -> Result<Vec<f32>> {
    let mut state = match model.graph.take() {
        Some(state) => state,
        None => State::new(model)?,
    };
    let result = state.encode_step(model, token, readout);
    model.graph = Some(state);
    result
}

/// Diagnostic-only target advance retaining a compact final wide residual.
pub(super) fn step_with_wide(
    model: &mut QwenModel,
    token: u32,
) -> Result<(Vec<f32>, DeviceVector)> {
    let mut state = match model.graph.take() {
        Some(state) => state,
        None => State::new(model)?,
    };
    let mut wide = None;
    let result = state.encode_edge0_batch(model, &[token], true, Some(&mut wide));
    model.graph = Some(state);
    let logits = result?;
    Ok((logits, wide.ok_or(Error::Invalid("private MTP target wide residual missing"))?))
}

/// One ordinary bounded no-readout prefill batch, with compact final-wide
/// copies for draft-front catch-up. This uses the identical target graph body.
pub(super) fn prefill_batch_with_wide(
    model: &mut QwenModel,
    tokens: &[u32],
) -> Result<DeviceVector> {
    edge0_wide_elements(
        tokens.len(),
        model.geometry.streams.checked_mul(model.geometry.dim).ok_or(Error::Invalid("Edge0 wide row width"))?,
        false,
    )
    .ok_or(Error::Invalid("Edge0 wide batch geometry"))?;
    let mut state = match model.graph.take() {
        Some(state) => state,
        None => State::new(model)?,
    };
    let mut wide = None;
    let result = state.encode_edge0_batch(model, tokens, false, Some(&mut wide));
    model.graph = Some(state);
    result?;
    wide.ok_or(Error::Invalid("Edge0 prefix wide residual missing"))
}

pub(super) fn prefill(
    model: &mut QwenModel,
    tokens: &[u32],
    before_token: &mut impl FnMut() -> Result<()>,
) -> Result<()> {
    if !model.geometry.edge0_metal {
        return Err(Error::Unsupported("batched Edge0 graph selector"));
    }
    if tokens.is_empty() {
        return Ok(());
    }
    let mut state = match model.graph.take() {
        Some(state) => state,
        None => State::new(model)?,
    };
    let result = (|| {
        for batch in tokens.chunks(edge0_metal::MAX_BATCH) {
            for _ in batch {
                before_token()?;
            }
            state.encode_edge0_batch(model, batch, false, None)?;
        }
        Ok(())
    })();
    if result.is_err() {
        model.reset();
    } else {
        model.graph = Some(state);
    }
    result
}

#[cfg(test)]
#[path = "qwen_graph_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "qwen_attention_tests.rs"]
mod attention_tests;

#[cfg(test)]
mod checkpoint_tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn edge0_prefill_wide_batch_shape_is_bounded_and_readout_stays_single_token() {
        for count in 1..=EDGE0_PREFILL_BATCH {
            assert_eq!(edge0_wide_elements(count, 10_240, false), Some(count * 10_240));
            assert_eq!(edge0_wide_elements(count, 10_240, true), (count == 1).then_some(10_240));
        }
        assert_eq!(edge0_wide_elements(8, 10_240, false).unwrap() * 4, 320 * 1024);
        for (count, width) in [(0, 10_240), (9, 10_240), (usize::MAX, 10_240), (1, 0), (1, 10_241), (8, usize::MAX)] {
            assert!(edge0_wide_elements(count, width, false).is_none());
        }
    }

    #[stark_test]
    fn private_graph_checkpoint_replacement_charges_capacity_and_overwritten_scratch() {
        let g = Geometry {
            context: 64,
            kv_heads: 1,
            head_dim: 8,
            index_dim: 4,
            ratio: 4,
            index_heads: 2,
            heads: 2,
            budget: 16,
            ..Geometry::default()
        };
        let one_token_snapshot = (2 * 8 + 4) * 4;
        let replacement = checkpoint_attention_replacement_bytes(&g, 32).unwrap();
        assert!(replacement == 4064 && replacement > one_token_snapshot);
        assert!(replacement == g.graph_cache_bytes(32).unwrap() + 32 * 8 * 4);
        assert!(checkpoint_attention_replacement_bytes(&g, 0).is_none());
        assert!(checkpoint_attention_replacement_bytes(&g, 65).is_none());
        let invalid = Geometry {
            ratio: 0,
            ..g
        };
        assert!(checkpoint_attention_replacement_bytes(&invalid, 32).is_none());
    }

    fn values(
        count: usize,
        salt: usize,
    ) -> Box<[f32]> {
        (0..count)
            .map(|index| match (index + salt) % 7 {
                0 => f32::from_bits(0x8000_0000),
                1 => f32::from_bits(1),
                _ => ((index + salt) % 13) as f32 * 0.125 - 0.75,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn equal(
        first: &Checkpoint,
        second: &Checkpoint,
    ) -> bool {
        let bits =
            |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
        first.position == second.position
            && first.geometry == second.geometry
            && first.layers.len() == second.layers.len()
            && first.layers.iter().zip(&second.layers).all(|(a, b)| {
                let attention = match (&a.attention, &b.attention) {
                    (Some(a), Some(b)) => {
                        a.capacity == b.capacity
                            && bits(&a.keys, &b.keys)
                            && bits(&a.values, &b.values)
                            && bits(&a.index_keys, &b.index_keys)
                            && bits(&a.pooled_index_keys, &b.pooled_index_keys)
                    },
                    (None, None) => true,
                    _ => false,
                };
                let ple = match (&a.ple_history, &b.ple_history) {
                    (Some(a), Some(b)) => bits(a, b),
                    (None, None) => true,
                    _ => false,
                };
                attention && ple
            })
    }

    #[stark_test]
    fn private_graph_checkpoint_restores_prefix_partial_pool_and_ple_without_aliasing() {
        let g = Geometry {
            context: 32,
            dim: 2,
            streams: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 1,
            index_dim: 2,
            ratio: 4,
            budget: 8,
            ple_kernel: 2,
            ngram_size: 3,
            ..Geometry::default()
        };
        let position = 6;
        let attention = AttentionCheckpoint {
            capacity: 16,
            keys: values(position * 2, 0),
            values: values(position * 2, 1),
            index_keys: values(position * 2, 2),
            pooled_index_keys: values(2, 3),
        };
        let norm = DecodeGraph::upload(&[1.0; 2]).unwrap();
        let source = State {
            final_norm: norm.clone(),
            layers: vec![LayerControls {
                attention_hyper: HyperControls {
                    norm: norm.clone(),
                },
                mlp_hyper: HyperControls {
                    norm: norm.clone(),
                },
                attention: Some(AttentionControls {
                    q_norm: norm.clone(),
                    k_norm: norm.clone(),
                    index_q_norm: norm.clone(),
                    index_k_norm: norm.clone(),
                    cache: Some(checkpoint_attention(&attention, &g, position).unwrap()),
                    scalar: None,
                }),
                ple: Some(PleControls {
                    key_norm: norm.clone(),
                    query_norm: norm.clone(),
                    conv_norm: norm.clone(),
                    convolution: norm,
                    history: DecodeGraph::upload(&values(6, 4)).unwrap(),
                    scratch: DecodeGraph::zeros(2).unwrap(),
                }),
                edge0: None,
            }],
        };
        let saved = source.capture_checkpoint(&g, position).unwrap();
        assert!(
            saved.layers[0].attention.as_ref().unwrap().index_keys.len() == 12,
            "partial index pool prefix missing"
        );
        assert!(saved.layers[0].attention.as_ref().unwrap().pooled_index_keys.len() == 2, "completed index pool count");
        let cache = source.layers[0].attention.as_ref().unwrap().cache.as_ref().unwrap();
        let history = &source.layers[0].ple.as_ref().unwrap().history;
        let mut mutation = DecodeGraph::new().unwrap();
        for view in [&cache.keys, &cache.values, &cache.index_keys, &cache.pooled_index_keys, history] {
            let overwrite = DecodeGraph::upload(&vec![42.0; view.len]).unwrap();
            mutation.copy_into(&overwrite, view).unwrap();
        }
        mutation.finish(&[]).unwrap();
        let restored = source.stage_checkpoint(&saved, &g, position).unwrap();
        assert!(
            equal(&saved, &restored.capture_checkpoint(&g, position).unwrap()),
            "graph checkpoint exact state roundtrip"
        );
        let restored_cache = restored.layers[0].attention.as_ref().unwrap().cache.as_ref().unwrap();
        assert!(!std::ptr::eq(&*cache.keys.buffer, &*restored_cache.keys.buffer), "QSA restore aliases mutable cache");
        assert!(
            !std::ptr::eq(&*history.buffer, &*restored.layers[0].ple.as_ref().unwrap().history.buffer),
            "PLE restore aliases mutable history"
        );
        assert!(
            std::ptr::eq(&*source.final_norm.buffer, &*restored.final_norm.buffer),
            "immutable norm ownership changed"
        );
        // Force a failure after replacement attention allocation. The current
        // source owners must remain untouched; model-level restore will reset
        // and invalidate handles when this staged operation fails.
        let current = source.capture_checkpoint(&g, position).unwrap();
        let mut bad = restored.capture_checkpoint(&g, position).unwrap();
        bad.layers[0].ple_history.as_mut().unwrap()[0] = f32::NAN;
        assert!(source.stage_checkpoint(&bad, &g, position).is_err());
        assert!(
            equal(&current, &source.capture_checkpoint(&g, position).unwrap()),
            "failed graph staging changed live state"
        );
        bad = restored.capture_checkpoint(&g, position).unwrap();
        bad.layers[0].attention.as_mut().unwrap().index_keys = Box::new([]);
        assert!(source.stage_checkpoint(&bad, &g, position).is_err());
        assert!(source.stage_checkpoint(&saved, &g, position + 1).is_err());
        let mut changed = g.clone();
        changed.ratio = 2;
        assert!(source.stage_checkpoint(&saved, &changed, position).is_err());
    }
}
