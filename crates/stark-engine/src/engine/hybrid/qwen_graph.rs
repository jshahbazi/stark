// SPDX-License-Identifier: Apache-2.0
//! Qwen's F32 device graph. Host work supplies immutable embeddings; all
//! dependent activations remain on the device until one token completes.

use super::{Attention, Geometry, HyperConnection, Mixer, Ple, QwenModel};
use crate::engine::hybrid::{
    Error, Result,
    weight_metal::{DecodeGraph, DeviceVector, QwenGdnGeometry, QwenGdnState},
    weights::Weights,
};

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
    scores: DeviceVector,
    probabilities: DeviceVector,
    block_scores: DeviceVector,
    top_ids: DeviceVector,
    top_scores: DeviceVector,
}

struct AttentionControls {
    q_norm: DeviceVector,
    k_norm: DeviceVector,
    index_q_norm: DeviceVector,
    index_k_norm: DeviceVector,
    cache: Option<AttentionCache>,
}

impl AttentionControls {
    fn new(attention: &Attention) -> Result<Self> {
        Ok(Self {
            q_norm: DecodeGraph::upload(&attention.q_norm)?,
            k_norm: DecodeGraph::upload(&attention.k_norm)?,
            index_q_norm: DecodeGraph::upload(&attention.index_q_norm)?,
            index_k_norm: DecodeGraph::upload(&attention.index_k_norm)?,
            cache: None,
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
            scores: DecodeGraph::zeros(size(g.heads)?)?,
            probabilities: DecodeGraph::zeros(size(g.heads)?)?,
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

    fn encode(
        &mut self,
        graph: &mut DecodeGraph,
        weights: &Weights,
        attention: &Attention,
        input: &DeviceVector,
        g: &Geometry,
        position: usize,
    ) -> Result<DeviceVector> {
        self.ensure_cache(graph, g, position)?;
        let cache = self.cache.as_ref().ok_or(Error::Invalid("Qwen graph attention cache"))?;
        let project = |graph: &mut DecodeGraph, name: &str| {
            weights.encode_matvec(graph, &format!("{}.{name}.weight", attention.prefix), input)
        };
        graph.begin_projection_level()?;
        let index = project(graph, "indexer.index_qk_proj")?;
        let query_and_gate = project(graph, "q_proj")?;
        let keys = project(graph, "k_proj")?;
        let values = project(graph, "v_proj")?;
        graph.end_projection_level()?;
        let count = position + 1;
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
        let partial_len =
            g.heads.checked_mul(splits).and_then(|n| n.checked_mul(g.head_dim + 2)).filter(|&n| n <= 8_000_000);
        let scalar_attention = count < 256
            || partial_len.is_none()
            || std::env::var_os("STARK_QWEN38_SCALAR_ATTENTION").is_some_and(|value| value == "1");
        if scalar_attention {
            let parameters =
                [count as u32, g.heads as u32, g.head_dim as u32, g.kv_heads as u32, cache.capacity as u32];
            graph.kernel(
                "qwen_graph_qsa_scores",
                &[&cache.query, &cache.keys, &cache.visible, &cache.scores],
                &parameters,
                g.heads * count,
                64,
            )?;
            graph.kernel(
                "qwen_graph_qsa_softmax",
                &[&cache.scores, &cache.visible, &cache.probabilities],
                &[count as u32, g.heads as u32, cache.capacity as u32],
                g.heads,
                1,
            )?;
            graph.kernel(
                "qwen_graph_qsa_output",
                &[&cache.probabilities, &cache.values, &cache.gates, &cache.visible, &output],
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
}

pub(super) struct State {
    layers: Vec<LayerControls>,
    final_norm: DeviceVector,
}

impl State {
    fn new(model: &QwenModel) -> Result<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
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

#[cfg(test)]
#[path = "qwen_graph_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "qwen_attention_tests.rs"]
mod attention_tests;
