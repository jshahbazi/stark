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
        let index = project(graph, "indexer.index_qk_proj")?;
        let query_and_gate = project(graph, "q_proj")?;
        let keys = project(graph, "k_proj")?;
        let values = project(graph, "v_proj")?;
        let count = position + 1;
        graph.kernel(
            "qwen_graph_qsa_prepare",
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
            g.heads + g.kv_heads + 1,
            1,
        )?;
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
            &[
                count as u32,
                g.index_heads as u32,
                g.index_dim as u32,
                g.rotary_dim as u32,
                position as u32,
                g.eps.to_bits(),
                g.rope_theta.to_bits(),
                g.ratio as u32,
                g.budget as u32,
            ],
            1,
            1,
        )?;
        let output = graph.scratch(g.heads * g.head_dim)?;
        let parameters = [count as u32, g.heads as u32, g.head_dim as u32, g.kv_heads as u32, cache.capacity as u32];
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
        let key = weights.encode_matvec(graph, &format!("{}.key_proj.weight", ple.prefix), embedding)?;
        let value = weights.encode_matvec(graph, &format!("{}.value_proj.weight", ple.prefix), embedding)?;
        let output = graph.scratch(g.streams * g.dim)?;
        graph.kernel(
            "qwen_graph_ple",
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
            g.streams,
            1,
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
        let mut graph = DecodeGraph::new()?;
        let mut hidden = DecodeGraph::upload(&embedding.repeat(g.streams))?;
        for (index, layer) in model.layers.iter_mut().enumerate() {
            let controls = &mut self.layers[index];
            if let (Some(ple), Some(ple_controls), Some(embedding)) =
                (&layer.ple, &controls.ple, &ple_embeddings[index])
            {
                let extra = ple_controls.encode(&mut graph, &model.weights, ple, embedding, &hidden, g)?;
                hidden = graph.add(&hidden, &extra)?;
            }
            let (input, gates) =
                mix(&mut graph, &model.weights, &layer.attn_hyper, &controls.attention_hyper.norm, &hidden, g)?;
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
            hidden = inject(&mut graph, &hidden, &output, &gates, g)?;
            let (input, gates) =
                mix(&mut graph, &model.weights, &layer.mlp_hyper, &controls.mlp_hyper.norm, &hidden, g)?;
            let output = moe(&mut graph, &model.weights, &format!("{}.mlp", layer.prefix), &input, g)?;
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
    let normalized = graph.rms_norm(input, norm, g.dim, g.eps)?;
    let inject = hyper.combine.then(|| format!("{}.block_inject_weight.weight", hyper.prefix));
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
    let scores = weights.encode_matvec(graph, &format!("{prefix}.gate.weight"), input)?;
    let shared_gate = weights.encode_matvec(graph, &format!("{prefix}.shared_expert_gate.weight"), input)?;
    let shared_values = weights.encode_matvec(graph, &format!("{prefix}.shared_expert.gate_proj.weight"), input)?;
    let shared_up = weights.encode_matvec(graph, &format!("{prefix}.shared_expert.up_proj.weight"), input)?;
    let indices = graph.scratch(g.active_experts)?;
    let probabilities = graph.scratch(g.active_experts)?;
    graph.kernel(
        "qwen_graph_router",
        &[&scores, &indices, &probabilities],
        &[g.experts as u32, g.active_experts as u32],
        1,
        1,
    )?;
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
    let values = graph.swiglu(&gate, &up, None)?;
    let down = weights.encode_indexed_matvec(
        graph,
        &format!("{prefix}.switch_mlp.down_proj.weight"),
        &indices,
        g.active_experts,
        &values,
        true,
    )?;
    let shared_values = graph.swiglu(&shared_values, &shared_up, None)?;
    let shared_down =
        weights.encode_matvec(graph, &format!("{prefix}.shared_expert.down_proj.weight"), &shared_values)?;
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
