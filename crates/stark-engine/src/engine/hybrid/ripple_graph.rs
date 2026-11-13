// SPDX-License-Identifier: Apache-2.0
//! Ripple's native F32 device graph. All dependent activations stay on Metal
//! through one token; only immutable embedding rows and final logits cross the
//! host boundary. The staged Rust forward remains a numerical reference.

use super::{Config, Error, Layer, MAX_CACHE_BYTES, Result, RippleModel};
use crate::engine::hybrid::{
    weight_metal::{DecodeGraph, DeviceVector},
    weights::Weights,
};

struct Cache {
    capacity: usize,
    keys: DeviceVector,
    values: DeviceVector,
    probabilities: DeviceVector,
}

fn cache_elements(
    c: &Config,
    capacity: usize,
) -> Result<usize> {
    capacity.checked_mul(c.kv_heads * c.dim * 2 + c.heads).ok_or(Error::Invalid("ripple_graph_cache_size"))
}

struct Controls {
    input_norm: DeviceVector,
    post_attention_norm: DeviceVector,
    pre_ffn_norm: DeviceVector,
    post_ffn_norm: DeviceVector,
    q_norm: DeviceVector,
    k_norm: DeviceVector,
    cache: Option<Cache>,
}

impl Controls {
    fn new(layer: &Layer) -> Result<Self> {
        Ok(Self {
            input_norm: DecodeGraph::upload(&layer.input_norm)?,
            post_attention_norm: DecodeGraph::upload(&layer.post_attention_norm)?,
            pre_ffn_norm: DecodeGraph::upload(&layer.pre_ffn_norm)?,
            post_ffn_norm: DecodeGraph::upload(&layer.post_ffn_norm)?,
            q_norm: DecodeGraph::upload(&layer.q_norm)?,
            k_norm: DecodeGraph::upload(&layer.k_norm)?,
            cache: None,
        })
    }

    fn grow_cache(
        &mut self,
        c: &Config,
        position: usize,
        capacity: usize,
    ) -> Result<()> {
        let width = c.kv_heads * c.dim;
        let cache = Cache {
            capacity,
            keys: DecodeGraph::zeros(capacity * width)?,
            values: DecodeGraph::zeros(capacity * width)?,
            probabilities: DecodeGraph::zeros(capacity * c.heads)?,
        };
        if let Some(previous) = &self.cache {
            let used = position.checked_mul(width).ok_or(Error::Invalid("ripple_graph_cache_copy"))?;
            if used != 0 {
                let mut graph = DecodeGraph::new()?;
                graph.copy_into(&previous.keys.slice(0, used)?, &cache.keys.slice(0, used)?)?;
                graph.copy_into(&previous.values.slice(0, used)?, &cache.values.slice(0, used)?)?;
                // Finish each growing layer before dropping its old buffers.
                // A single giant replacement command would transiently retain
                // both complete KV inventories near the request memory bound.
                graph.finish(&[])?;
            }
        }
        self.cache = Some(cache);
        Ok(())
    }

    fn attention(
        &self,
        graph: &mut DecodeGraph,
        weights: &Weights,
        layer: &Layer,
        c: &Config,
        input: &DeviceVector,
        seed: &DeviceVector,
        position: usize,
        sliding: bool,
        fast_controls: bool,
    ) -> Result<DeviceVector> {
        let cache = self.cache.as_ref().ok_or(Error::Invalid("ripple_graph_cache_missing"))?;
        let value_input = if let Some(blend) = layer.blend {
            let output = graph.scratch(c.hidden)?;
            graph.kernel(
                "ripple_graph_value_input",
                &[seed, input, &output],
                &[c.hidden as u32, blend.to_bits()],
                c.hidden,
                256,
            )?;
            output
        } else {
            input.clone()
        };
        graph.begin_projection_level()?;
        let queries = weights.encode_matvec(graph, &layer.q, input)?;
        let keys = weights.encode_matvec(graph, &layer.k, input)?;
        let values = weights.encode_matvec(graph, &layer.v, &value_input)?;
        let gates = layer.gate.as_ref().map(|name| weights.encode_matvec(graph, name, input)).transpose()?;
        graph.end_projection_level()?;
        graph.diagnostic_checkpoint("ripple.phase.attention_projections")?;

        let normalized_queries = graph.scratch(c.heads * c.dim)?;
        let count = if sliding {
            (position + 1).min(c.window)
        } else {
            position + 1
        };
        let slot = if sliding {
            position % cache.capacity
        } else {
            position
        };
        let start = if sliding {
            (position + 1 - count) % cache.capacity
        } else {
            0
        };
        let parameters = [
            c.heads as u32,
            c.kv_heads as u32,
            c.dim as u32,
            position as u32,
            slot as u32,
            u32::from(sliding),
            c.eps.to_bits(),
            c.theta.to_bits(),
        ];
        let parallel_prepare = fast_controls && c.dim == 128;
        graph.kernel(
            if parallel_prepare {
                "ripple_graph_attention_prepare_parallel"
            } else {
                "ripple_graph_attention_prepare"
            },
            &[&queries, &keys, &values, &self.q_norm, &self.k_norm, &normalized_queries, &cache.keys, &cache.values],
            &parameters,
            (c.heads + c.kv_heads)
                * if parallel_prepare {
                    32
                } else {
                    1
                },
            if parallel_prepare {
                32
            } else {
                1
            },
        )?;
        graph.diagnostic_checkpoint("ripple.phase.attention_prepare")?;
        let parameters =
            [count as u32, c.heads as u32, c.kv_heads as u32, c.dim as u32, cache.capacity as u32, start as u32];
        let parallel_scores = fast_controls && c.dim == 128 && count <= 64;
        graph.kernel(
            if parallel_scores {
                "ripple_graph_attention_scores_parallel"
            } else {
                "ripple_graph_attention_scores"
            },
            &[&normalized_queries, &cache.keys, &cache.probabilities],
            &parameters,
            c.heads
                * count
                * if parallel_scores {
                    32
                } else {
                    1
                },
            if parallel_scores {
                32
            } else {
                256
            },
        )?;
        graph.diagnostic_checkpoint("ripple.phase.attention_scores")?;
        let parallel_softmax = fast_controls && count >= 8;
        graph.kernel(
            if parallel_softmax {
                "ripple_graph_attention_softmax_parallel"
            } else {
                "ripple_graph_attention_softmax"
            },
            &[&cache.probabilities],
            &parameters,
            c.heads
                * if parallel_softmax {
                    32
                } else {
                    1
                },
            if parallel_softmax {
                32
            } else {
                1
            },
        )?;
        graph.diagnostic_checkpoint("ripple.phase.attention_softmax")?;
        let output = graph.scratch(c.heads * c.dim)?;
        let mut parameters = parameters.to_vec();
        parameters.push(u32::from(gates.is_some()));
        // The no-gate path binds a valid same-width vector but does not read it.
        // These shape bounds keep every layer temporary plus the existing
        // decode graph below its independent 512 MiB scratch ceiling.
        let partitioned = fast_controls
            && c.dim == 128
            && c.heads <= 80
            && c.kv_heads <= 8
            && c.hidden <= 5120
            && c.intermediate <= 40960
            && c.sliding.len() <= 80
            && (256..=4096).contains(&count);
        if partitioned {
            let tile = if count <= 1024 {
                64
            } else {
                128
            };
            parameters.push(tile as u32);
            let elements = c.heads * c.dim * count.div_ceil(tile);
            let partials = graph.scratch(elements)?;
            graph.kernel(
                "ripple_graph_attention_output_partial",
                &[&cache.probabilities, &cache.values, &partials],
                &parameters,
                elements,
                128,
            )?;
            graph.kernel(
                "ripple_graph_attention_output_reduce",
                &[
                    &partials,
                    &cache.probabilities,
                    &cache.values,
                    gates.as_ref().unwrap_or(&normalized_queries),
                    &output,
                ],
                &parameters,
                c.heads * c.dim,
                128,
            )?;
        } else {
            graph.kernel(
                "ripple_graph_attention_output",
                &[&cache.probabilities, &cache.values, gates.as_ref().unwrap_or(&normalized_queries), &output],
                &parameters,
                c.heads * c.dim,
                256,
            )?;
        }
        graph.diagnostic_checkpoint("ripple.phase.attention_output")?;
        let output = weights.encode_matvec(graph, &layer.o, &output)?;
        graph.diagnostic_checkpoint("ripple.phase.attention_o_projection")?;
        Ok(output)
    }
}

pub(super) struct State {
    layers: Vec<Controls>,
    norm: DeviceVector,
}

impl State {
    fn new(model: &RippleModel) -> Result<Self> {
        Ok(Self {
            layers: model.layers.iter().map(Controls::new).collect::<Result<Vec<_>>>()?,
            norm: DecodeGraph::upload(&model.norm)?,
        })
    }

    fn prepare_caches(
        &mut self,
        c: &Config,
        position: usize,
    ) -> Result<()> {
        let mut allocated = self.layers.iter().try_fold(0usize, |sum, layer| {
            let capacity = layer.cache.as_ref().map_or(0, |cache| cache.capacity);
            sum.checked_add(cache_elements(c, capacity)?).ok_or(Error::Invalid("ripple_graph_cache_size"))
        })?;
        for (layer, &sliding) in self.layers.iter_mut().zip(&c.sliding) {
            let limit = if sliding {
                c.window.min(c.context)
            } else {
                c.context
            };
            let count = (position + 1).min(limit);
            if layer.cache.as_ref().is_some_and(|cache| cache.capacity >= count) {
                continue;
            }
            // Geometric growth amortizes small-request copies. Beyond 16K,
            // 4K increments bound unused global-prefix allocation.
            let capacity = if count <= 16_384 {
                count.checked_next_power_of_two().ok_or(Error::Invalid("ripple_graph_cache_capacity"))?.max(16)
            } else {
                count.checked_add(4095).ok_or(Error::Invalid("ripple_graph_cache_capacity"))? / 4096 * 4096
            }
            .min(limit);
            let replacement = cache_elements(c, capacity)?;
            // Include the old layer's live buffers until its copy completes,
            // and persistent softmax scratch, in the same checked bound.
            allocated
                .checked_add(replacement)
                .and_then(|n| n.checked_mul(4))
                .filter(|&bytes| bytes <= MAX_CACHE_BYTES)
                .ok_or(Error::Invalid("ripple_graph_cache_bound"))?;
            let previous = layer.cache.as_ref().map_or(0, |cache| cache.capacity);
            layer.grow_cache(c, position, capacity)?;
            allocated = allocated - cache_elements(c, previous)? + replacement;
        }
        Ok(())
    }

    fn encode_step(
        &mut self,
        model: &mut RippleModel,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        let c = &model.config;
        self.prepare_caches(c, model.position)?;
        let mut embedding = model.weights.row(&model.embedding, token as usize)?;
        for value in &mut embedding {
            *value *= c.embed_scale;
        }
        #[cfg(not(test))]
        let mut graph = DecodeGraph::new()?;
        #[cfg(test)]
        let mut graph = if model.test_control_mode == 2 {
            DecodeGraph::new_parallel()?
        } else {
            DecodeGraph::new()?
        };
        #[cfg(not(test))]
        let fast_controls = true;
        #[cfg(test)]
        let fast_controls = model.test_control_mode != 0;
        graph.enable_ripple_dense_q8();
        graph.enable_ripple_dense_q4();
        let seed = graph.upload_into(&embedding)?;
        let mut hidden = seed.clone();
        for (index, (layer, controls)) in model.layers.iter().zip(&self.layers).enumerate() {
            let input = graph.rms_norm(&hidden, &controls.input_norm, c.hidden, c.eps)?;
            let update = controls.attention(
                &mut graph,
                &model.weights,
                layer,
                c,
                &input,
                &seed,
                model.position,
                c.sliding[index],
                fast_controls,
            )?;
            let update = graph.rms_norm(&update, &controls.post_attention_norm, c.hidden, c.eps)?;
            hidden = graph.add(&hidden, &update)?;
            let input = graph.rms_norm(&hidden, &controls.pre_ffn_norm, c.hidden, c.eps)?;
            graph.diagnostic_checkpoint("ripple.phase.ffn_norms")?;
            graph.begin_projection_level()?;
            let gate = model.weights.encode_matvec(&mut graph, &layer.ffn_gate, &input)?;
            let up = model.weights.encode_matvec(&mut graph, &layer.ffn_up, &input)?;
            graph.end_projection_level()?;
            graph.diagnostic_checkpoint("ripple.phase.ffn_gate_up")?;
            let activated = graph.scratch(c.intermediate)?;
            graph.kernel(
                "ripple_graph_swiglu",
                &[&gate, &up, &activated],
                &[c.intermediate as u32, c.clamp.unwrap_or(0.0).to_bits()],
                c.intermediate,
                256,
            )?;
            graph.diagnostic_checkpoint("ripple.phase.ffn_swiglu")?;
            let update = model.weights.encode_matvec(&mut graph, &layer.ffn_down, &activated)?;
            graph.diagnostic_checkpoint("ripple.phase.ffn_down")?;
            let update = graph.rms_norm(&update, &controls.post_ffn_norm, c.hidden, c.eps)?;
            hidden = graph.add(&hidden, &update)?;
            graph.diagnostic_checkpoint("ripple.phase.ffn_residual")?;
        }
        let outputs = if readout {
            let hidden = graph.rms_norm(&hidden, &self.norm, c.hidden, c.eps)?;
            let logits = model.weights.encode_matvec(&mut graph, &model.output, &hidden)?;
            graph.diagnostic_checkpoint("ripple.phase.readout")?;
            graph.finish(&[logits])?
        } else {
            graph.finish(&[])?
        };
        let logits = if readout {
            outputs.into_iter().next().ok_or(Error::Invalid("ripple_graph_missing_readout"))?
        } else {
            Vec::new()
        };
        if readout && (logits.len() != c.vocab || logits.iter().any(|v| !v.is_finite())) {
            return Err(Error::Invalid("ripple_graph_nonfinite_output"));
        }
        model.position += 1;
        Ok(logits)
    }
}

// Offline diagnostics only: this deliberately does not implement the product
// prefix-cache contract. Captures remain private CPU memory and never persist.
#[cfg(test)]
const TEST_SNAPSHOT_MAX_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(test)]
pub(super) struct TestCacheSnapshot {
    pub(super) capacity: usize,
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
}

#[cfg(test)]
pub(super) struct TestSnapshot {
    pub(super) position: usize,
    pub(super) payload_bytes: usize,
    pub(super) layers: Vec<TestCacheSnapshot>,
    kv_heads: usize,
    dim: usize,
    // Retention prevents a different State from reusing this identity address.
    owner_norm: DeviceVector,
}

#[cfg(test)]
impl State {
    pub(super) fn test_snapshot(
        &self,
        c: &Config,
        position: usize,
    ) -> Result<TestSnapshot> {
        if position == 0 || position > c.context || self.layers.len() != c.sliding.len() {
            return Err(Error::Invalid("ripple_test_snapshot_position"));
        }
        let width = c.kv_heads.checked_mul(c.dim).ok_or(Error::Invalid("ripple_test_snapshot_size"))?;
        let mut payload_bytes = 0usize;
        for layer in &self.layers {
            let cache = layer.cache.as_ref().ok_or(Error::Invalid("ripple_test_snapshot_missing_cache"))?;
            let elements = cache.capacity.checked_mul(width).ok_or(Error::Invalid("ripple_test_snapshot_size"))?;
            if cache.keys.len != elements || cache.values.len != elements {
                return Err(Error::Invalid("ripple_test_snapshot_geometry"));
            }
            payload_bytes = elements
                .checked_mul(8)
                .and_then(|n| payload_bytes.checked_add(n))
                .filter(|&n| n <= TEST_SNAPSHOT_MAX_BYTES)
                .ok_or(Error::Invalid("ripple_test_snapshot_cap"))?;
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let cache = layer.cache.as_ref().ok_or(Error::Invalid("ripple_test_snapshot_missing_cache"))?;
            let mut graph = DecodeGraph::new()?;
            graph.check_finite(&cache.keys)?;
            graph.check_finite(&cache.values)?;
            let mut values = graph.finish(&[cache.keys.clone(), cache.values.clone()])?;
            let value_cache = values.pop().ok_or(Error::Invalid("ripple_test_snapshot_readback"))?;
            let key_cache = values.pop().ok_or(Error::Invalid("ripple_test_snapshot_readback"))?;
            layers.push(TestCacheSnapshot {
                capacity: cache.capacity,
                keys: key_cache,
                values: value_cache,
            });
        }
        Ok(TestSnapshot {
            position,
            payload_bytes,
            layers,
            kv_heads: c.kv_heads,
            dim: c.dim,
            owner_norm: self.norm.clone(),
        })
    }

    pub(super) fn test_restore(
        &mut self,
        c: &Config,
        snapshot: &TestSnapshot,
    ) -> Result<usize> {
        if snapshot.position == 0
            || snapshot.position > c.context
            || snapshot.layers.len() != self.layers.len()
            || snapshot.layers.len() != c.sliding.len()
            || snapshot.kv_heads != c.kv_heads
            || snapshot.dim != c.dim
            || !std::ptr::eq(
                objc2::rc::Retained::as_ptr(&snapshot.owner_norm.buffer),
                objc2::rc::Retained::as_ptr(&self.norm.buffer),
            )
        {
            return Err(Error::Invalid("ripple_test_snapshot_identity"));
        }
        let width = c.kv_heads.checked_mul(c.dim).ok_or(Error::Invalid("ripple_test_snapshot_size"))?;
        let mut payload_bytes = 0usize;
        let mut replacement_elements = 0usize;
        for (layer, &sliding) in snapshot.layers.iter().zip(&c.sliding) {
            let limit = if sliding {
                c.window.min(c.context)
            } else {
                c.context
            };
            let required = snapshot.position.min(limit);
            if layer.capacity < required || layer.capacity > limit {
                return Err(Error::Invalid("ripple_test_snapshot_capacity"));
            }
            let elements = layer.capacity.checked_mul(width).ok_or(Error::Invalid("ripple_test_snapshot_size"))?;
            if layer.keys.len() != elements
                || layer.values.len() != elements
                || layer.keys.iter().chain(&layer.values).any(|value| !value.is_finite())
            {
                return Err(Error::Invalid("ripple_test_snapshot_geometry"));
            }
            payload_bytes = elements
                .checked_mul(8)
                .and_then(|n| payload_bytes.checked_add(n))
                .filter(|&n| n <= TEST_SNAPSHOT_MAX_BYTES)
                .ok_or(Error::Invalid("ripple_test_snapshot_cap"))?;
            replacement_elements = replacement_elements
                .checked_add(cache_elements(c, layer.capacity)?)
                .ok_or(Error::Invalid("ripple_test_snapshot_size"))?;
        }
        if payload_bytes != snapshot.payload_bytes {
            return Err(Error::Invalid("ripple_test_snapshot_geometry"));
        }
        let previous_elements = self.layers.iter().try_fold(0usize, |sum, layer| {
            let capacity = layer.cache.as_ref().map_or(0, |cache| cache.capacity);
            sum.checked_add(cache_elements(c, capacity)?).ok_or(Error::Invalid("ripple_test_snapshot_size"))
        })?;
        previous_elements
            .checked_add(replacement_elements)
            .and_then(|n| n.checked_mul(4))
            .filter(|&bytes| bytes <= MAX_CACHE_BYTES)
            .ok_or(Error::Invalid("ripple_test_snapshot_live_cap"))?;
        // Build all replacements before changing State. Allocation or upload
        // failure therefore leaves the previous State intact.
        let mut replacement = Vec::with_capacity(snapshot.layers.len());
        for layer in &snapshot.layers {
            replacement.push(Cache {
                capacity: layer.capacity,
                keys: DecodeGraph::upload(&layer.keys)?,
                values: DecodeGraph::upload(&layer.values)?,
                probabilities: DecodeGraph::zeros(
                    layer.capacity.checked_mul(c.heads).ok_or(Error::Invalid("ripple_test_snapshot_size"))?,
                )?,
            });
        }
        for (controls, cache) in self.layers.iter_mut().zip(replacement) {
            controls.cache = Some(cache);
        }
        Ok(snapshot.position)
    }
}

pub(super) fn step(
    model: &mut RippleModel,
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
