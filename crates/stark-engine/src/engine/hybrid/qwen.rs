// SPDX-License-Identifier: Apache-2.0
//! Native scalar-control forward for Qwen3.8-Flash-Next's `qwen4_exp` text model.
//!
//! Architecture and source tensor conventions follow PipeNetwork's Apache-2.0
//! MLX port, revision 0352280b927a731fd35276ac622ef2f28da5c936, and checkpoint
//! b2c422f3c643e36f04227a64d61796b44a4b1029. This Rust implementation computes
//! each visible prefix by recurrent steps, including the indexer's own trailing
//! partial block. Centered normalization weights in this MLX export already
//! contain the upstream +1; applying that shift again corrupts the model.

use serde_json::Value;

#[path = "edge0.rs"]
mod edge0;

#[cfg(feature = "metal")]
#[path = "qwen_graph.rs"]
mod decode_graph;

#[cfg(feature = "metal")]
#[path = "qwen_checkpoint.rs"]
// Private staging primitives remain unused until separately qualified MTP integration.
#[allow(dead_code)]
mod checkpoint;

#[cfg(feature = "metal")]
// Reserved for the private native-target adapter; no production caller is enabled.
#[allow(unused_imports)]
pub(super) use checkpoint::{Checkpoint, StagingPolicy};

use super::{
    Error, Result,
    math::{rms_norm, sigmoid, silu, softmax, topk},
    profile::Scope,
    weights::Weights,
};

#[derive(Clone)]
#[cfg_attr(test, derive(Default))]
struct Geometry {
    edge0: bool,
    edge0_metal: bool,
    dim: usize,
    vocab: usize,
    streams: usize,
    low_rank: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    experts: usize,
    active_experts: usize,
    expert_dim: usize,
    shared_dim: usize,
    index_heads: usize,
    index_dim: usize,
    budget: usize,
    ratio: usize,
    eps: f32,
    sigmoid_output: bool,
    context: usize,
    ngram_size: usize,
    ngram_heads: usize,
    ple_dim: usize,
    ple_kernel: usize,
    shards: usize,
    eos: u32,
}

fn integer(
    config: &Value,
    field: &str,
    limit: usize,
) -> Result<usize> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .filter(|&value| value > 0 && value <= limit as u64)
        .map(|value| value as usize)
        .ok_or(Error::Invalid("Qwen3.8 geometry"))
}

fn positive(
    config: &Value,
    field: &str,
) -> Result<f32> {
    config
        .get(field)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0 && *value <= f32::MAX as f64)
        .map(|value| value as f32)
        .ok_or(Error::Invalid("Qwen3.8 scalar configuration"))
}

fn shape(
    weights: &Weights,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    if weights.shape(name)? != expected {
        return Err(Error::Invalid("Qwen3.8 tensor geometry"));
    }
    Ok(())
}

fn matrix(
    weights: &Weights,
    prefix: &str,
    rows: usize,
    cols: usize,
) -> Result<()> {
    shape(weights, &format!("{prefix}.weight"), &[rows, cols])
}

fn vector(
    weights: &Weights,
    name: &str,
    length: usize,
) -> Result<Vec<f32>> {
    let result = weights.vector(name)?;
    if result.len() != length || result.iter().any(|value| !value.is_finite()) {
        return Err(Error::Invalid("Qwen3.8 control tensor"));
    }
    Ok(result)
}

fn project(
    weights: &Weights,
    prefix: &str,
    input: &[f32],
) -> Result<Vec<f32>> {
    weights.matvec(&format!("{prefix}.weight"), input)
}

/// Submit independent projections together without combining their arithmetic
/// or changing output order. Inputs may differ, as in hyperconnection gates.
fn projects<const N: usize>(
    weights: &Weights,
    prefix: &str,
    projections: [(&str, &[f32]); N],
) -> Result<[Vec<f32>; N]> {
    let names = projections.each_ref().map(|(name, _)| format!("{prefix}.{name}.weight"));
    let requests: [(&str, &[f32]); N] = std::array::from_fn(|index| (names[index].as_str(), projections[index].1));
    weights.matvec_many(&requests)?.try_into().map_err(|_| Error::Invalid("Qwen3.8 projection batch dimensions"))
}

fn grouped_norm(
    input: &[f32],
    weight: &[f32],
    group: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if group == 0 || input.len() != weight.len() || !input.len().is_multiple_of(group) {
        return Err(Error::Invalid("Qwen3.8 grouped normalization"));
    }
    let mut result = Vec::with_capacity(input.len());
    for (x, w) in input.chunks_exact(group).zip(weight.chunks_exact(group)) {
        result.extend(rms_norm(x, w, eps)?);
    }
    Ok(result)
}

fn rope(
    input: &mut [f32],
    rotary_dim: usize,
    position: usize,
    theta: f32,
) {
    let half = rotary_dim / 2;
    for index in 0..half {
        let phase = position as f32 * theta.powf(-2.0 * index as f32 / rotary_dim as f32);
        let (sin, cos) = phase.sin_cos();
        let left = input[index];
        let right = input[index + half];
        input[index] = left * cos - right * sin;
        input[index + half] = right * cos + left * sin;
    }
}

fn dot(
    left: &[f32],
    right: &[f32],
) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Channel-major `[channels, kernel, 1]` MLX depthwise weights and a chronological
/// history. Dilation belongs to the PLE convolution, not the DeltaNet one.
#[derive(Default)]
struct ConvState {
    history: Vec<f32>,
}

impl ConvState {
    fn clear(&mut self) {
        self.history.clear();
    }

    fn step(
        &mut self,
        input: &[f32],
        weight: &[f32],
        kernel: usize,
        dilation: usize,
    ) -> Result<Vec<f32>> {
        let channels = input.len();
        if channels == 0 || kernel < 2 || dilation == 0 || weight.len() != channels * kernel {
            return Err(Error::Invalid("Qwen3.8 convolution"));
        }
        let history_length = (kernel - 1) * dilation * channels;
        if self.history.is_empty() {
            self.history.resize(history_length, 0.0);
        }
        if self.history.len() != history_length {
            return Err(Error::Invalid("Qwen3.8 convolution state"));
        }
        let mut output = vec![0.0; channels];
        for channel in 0..channels {
            let mut sum = input[channel] * weight[channel * kernel + kernel - 1];
            for tap in 0..kernel - 1 {
                sum += self.history[tap * dilation * channels + channel] * weight[channel * kernel + tap];
            }
            output[channel] = silu(sum);
        }
        self.history.copy_within(channels.., 0);
        self.history[history_length - channels..].copy_from_slice(input);
        Ok(output)
    }
}

struct HyperConnection {
    prefix: String,
    norm: Vec<f32>,
    combine: bool,
}

impl HyperConnection {
    fn load(
        weights: &Weights,
        prefix: String,
        g: &Geometry,
        combine: bool,
    ) -> Result<Self> {
        let width = g.dim * g.streams;
        matrix(weights, &format!("{prefix}.input_mix_weight_down"), g.low_rank, width)?;
        matrix(weights, &format!("{prefix}.input_mix_weight_up"), width, g.low_rank)?;
        if combine {
            matrix(weights, &format!("{prefix}.block_inject_weight"), g.streams, width)?;
        }
        let norm = vector(weights, &format!("{prefix}.hc_norm.weight"), width)?;
        Ok(Self {
            prefix,
            norm,
            combine,
        })
    }

    fn mix(
        &self,
        weights: &Weights,
        input: &[f32],
        g: &Geometry,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let _profile = Scope::new("qwen.mhc");
        let normalized = grouped_norm(input, &self.norm, g.dim, g.eps)?;
        #[cfg(feature = "metal")]
        {
            let inject = self.combine.then(|| format!("{}.block_inject_weight.weight", self.prefix));
            weights.qwen_hyperconnection(
                &format!("{}.input_mix_weight_down.weight", self.prefix),
                &format!("{}.input_mix_weight_up.weight", self.prefix),
                inject.as_deref(),
                &normalized,
                g.streams,
            )
        }
        #[cfg(not(feature = "metal"))]
        {
            let mut reduced = project(weights, &format!("{}.input_mix_weight_down", self.prefix), &normalized)?;
            for value in &mut reduced {
                *value = silu(*value / g.streams as f32);
            }
            let (gates, inject) = if self.combine {
                let [gates, mut values] = projects(
                    weights,
                    &self.prefix,
                    [("input_mix_weight_up", &reduced), ("block_inject_weight", &normalized)],
                )?;
                if values.len() != g.streams {
                    return Err(Error::Invalid("Qwen3.8 write gate dimensions"));
                }
                values.iter_mut().for_each(|value| *value = 2.0 * sigmoid(*value / g.streams as f32));
                (gates, values)
            } else {
                (project(weights, &format!("{}.input_mix_weight_up", self.prefix), &reduced)?, Vec::new())
            };
            if gates.len() != input.len() {
                return Err(Error::Invalid("Qwen3.8 read gate dimensions"));
            }
            let mut mixed = vec![0.0; g.dim];
            for stream in 0..g.streams {
                for channel in 0..g.dim {
                    let index = stream * g.dim + channel;
                    mixed[channel] += sigmoid(gates[index]) * normalized[index] / g.streams as f32;
                }
            }
            Ok((mixed, inject))
        }
    }
}

fn inject(
    input: &mut [f32],
    output: &[f32],
    gates: &[f32],
    g: &Geometry,
) -> Result<()> {
    if input.len() != g.streams * g.dim || output.len() != g.dim || gates.len() != g.streams {
        return Err(Error::Invalid("Qwen3.8 residual injection"));
    }
    for stream in 0..g.streams {
        for channel in 0..g.dim {
            input[stream * g.dim + channel] += gates[stream] * output[channel];
        }
    }
    Ok(())
}

struct DeltaNet {
    prefix: String,
    convolution: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    #[cfg(feature = "metal")]
    metal: Option<super::weight_metal::QwenGdnState>,
    conv_state: ConvState,
    state: Vec<f32>,
}

impl DeltaNet {
    fn additional_memory_bytes(
        &self,
        g: &Geometry,
    ) -> Option<usize> {
        let width =
            g.key_heads.checked_mul(g.key_dim)?.checked_mul(2)?.checked_add(g.value_heads.checked_mul(g.value_dim)?)?;
        let recurrent = g.value_heads.checked_mul(g.value_dim)?.checked_mul(g.key_dim)?;
        if g.edge0 && !g.edge0_metal {
            return super::memory::sum(&[
                super::memory::vector_resize_bytes::<f32>(self.state.capacity(), recurrent)?,
                super::memory::vector_resize_bytes::<f32>(
                    self.conv_state.history.capacity(),
                    width.checked_mul(g.conv_kernel - 1)?,
                )?,
            ]);
        }
        #[cfg(feature = "metal")]
        {
            if self.metal.is_some() {
                // A completed constructor owns every persistent control,
                // recurrent and convolution buffer, including after reset.
                return Some(0);
            }
            let controls = width
                .checked_mul(g.conv_kernel)?
                .checked_add(g.value_heads.checked_mul(2)?)?
                .checked_add(g.value_dim)?;
            super::memory::sum(&[
                controls.checked_mul(2)?,
                width.checked_mul(g.conv_kernel - 1)?,
                recurrent,
                width,
                g.value_heads.checked_mul(g.value_dim)?,
            ])?
            .checked_mul(4)
        }
        #[cfg(not(feature = "metal"))]
        super::memory::sum(&[
            super::memory::vector_resize_bytes::<f32>(self.state.capacity(), recurrent)?,
            super::memory::vector_resize_bytes::<f32>(
                self.conv_state.history.capacity(),
                width.checked_mul(g.conv_kernel - 1)?,
            )?,
        ])
    }

    fn load(
        weights: &Weights,
        prefix: String,
        g: &Geometry,
    ) -> Result<Self> {
        let key_width = g.key_heads * g.key_dim;
        let value_width = g.value_heads * g.value_dim;
        let width = 2 * key_width + value_width;
        for (name, rows) in [
            ("in_proj_qkv", width),
            ("in_proj_z", value_width),
            ("in_proj_a", g.value_heads),
            ("in_proj_b", g.value_heads),
        ] {
            matrix(weights, &format!("{prefix}.{name}"), rows, g.dim)?;
        }
        matrix(weights, &format!("{prefix}.out_proj"), g.dim, value_width)?;
        shape(weights, &format!("{prefix}.conv1d.weight"), &[width, g.conv_kernel, 1])?;
        let a_log = vector(weights, &format!("{prefix}.A_log"), g.value_heads)?;
        if g.edge0 && a_log.iter().any(|value| *value > 0.0) {
            return Err(Error::Invalid("Edge0 recurrent stored negative A"));
        }
        Ok(Self {
            convolution: vector(weights, &format!("{prefix}.conv1d.weight"), width * g.conv_kernel)?,
            // GGUF already stores -exp(A_log); retain it directly for Edge0.
            a_log,
            dt_bias: vector(weights, &format!("{prefix}.dt_bias"), g.value_heads)?,
            norm: vector(weights, &format!("{prefix}.norm.weight"), g.value_dim)?,
            prefix,
            #[cfg(feature = "metal")]
            metal: None,
            conv_state: ConvState::default(),
            state: Vec::new(),
        })
    }

    fn reset(&mut self) {
        #[cfg(feature = "metal")]
        if let Some(state) = &mut self.metal {
            state.reset();
        }
        self.conv_state.clear();
        self.state.clear();
    }

    fn step(
        &mut self,
        weights: &Weights,
        input: &[f32],
        g: &Geometry,
    ) -> Result<Vec<f32>> {
        let _profile = Scope::new("qwen.gdn");
        #[cfg(feature = "metal")]
        if !g.edge0 || g.edge0_metal {
            if self.metal.is_none() {
                let constructor = if g.edge0 {
                    super::weight_metal::QwenGdnState::new_edge0
                } else {
                    super::weight_metal::QwenGdnState::new
                };
                self.metal = Some(constructor(
                    super::weight_metal::QwenGdnGeometry {
                        key_heads: g.key_heads,
                        key_dim: g.key_dim,
                        value_heads: g.value_heads,
                        value_dim: g.value_dim,
                        conv_kernel: g.conv_kernel,
                        eps: g.eps,
                        sigmoid_output: g.sigmoid_output,
                    },
                    &self.convolution,
                    &self.a_log,
                    &self.dt_bias,
                    &self.norm,
                )?);
            }
            return weights.qwen_gdn(
                &self.prefix,
                input,
                self.metal.as_mut().ok_or(Error::Invalid("Qwen GDN state unavailable"))?,
            );
        }
        {
            let [mixed, z, a, b] = projects(
                weights,
                &self.prefix,
                [("in_proj_qkv", input), ("in_proj_z", input), ("in_proj_a", input), ("in_proj_b", input)],
            )?;
            let convolved = self.conv_state.step(&mixed, &self.convolution, g.conv_kernel, 1)?;
            let result = self.recurrent_step(&convolved, &z, &a, &b, g)?;
            project(weights, &format!("{}.out_proj", self.prefix), &result)
        }
    }

    fn recurrent_step(
        &mut self,
        convolved: &[f32],
        z: &[f32],
        a: &[f32],
        b: &[f32],
        g: &Geometry,
    ) -> Result<Vec<f32>> {
        let _profile = Scope::new("qwen.gdn.recurrent");
        let key_width = g.key_heads * g.key_dim;
        let value_width = g.value_heads * g.value_dim;
        if convolved.len() != 2 * key_width + value_width
            || z.len() != value_width
            || a.len() != g.value_heads
            || b.len() != g.value_heads
        {
            return Err(Error::Invalid("Qwen recurrent projection geometry"));
        }
        let (q, remainder) = convolved.split_at(key_width);
        let (k, v) = remainder.split_at(key_width);
        let mut q = q.to_vec();
        let mut k = k.to_vec();
        normalize_l2(&mut q, g.key_dim, (g.key_dim as f32).sqrt().recip());
        normalize_l2(&mut k, g.key_dim, 1.0);
        let count = g.value_heads * g.value_dim * g.key_dim;
        if self.state.is_empty() {
            self.state.resize(count, 0.0);
        }
        if self.state.len() != count {
            return Err(Error::Invalid("Qwen recurrent state geometry"));
        }
        let mut result = vec![0.0; value_width];
        for head in 0..g.value_heads {
            // GGUF conversion tiles V heads to match ggml broadcast; the
            // MLX source keeps V heads grouped by their K head instead.
            let key_head = if g.edge0 {
                head % g.key_heads
            } else {
                head / (g.value_heads / g.key_heads)
            };
            let q = &q[key_head * g.key_dim..(key_head + 1) * g.key_dim];
            let k = &k[key_head * g.key_dim..(key_head + 1) * g.key_dim];
            let softplus_arg = a[head] + self.dt_bias[head];
            let softplus = softplus_arg.max(0.0) + (-softplus_arg.abs()).exp().ln_1p();
            let negative_a = if g.edge0 {
                self.a_log[head]
            } else {
                -self.a_log[head].exp()
            };
            let decay = (negative_a * softplus).exp();
            let beta = sigmoid(b[head]);
            for channel in 0..g.value_dim {
                let index = head * g.value_dim + channel;
                let state = &mut self.state[index * g.key_dim..(index + 1) * g.key_dim];
                result[index] = delta_row(state, q, k, v[index], decay, beta);
            }
            let normalized = rms_norm(&result[head * g.value_dim..(head + 1) * g.value_dim], &self.norm, g.eps)?;
            for (channel, value) in normalized.into_iter().enumerate() {
                let index = head * g.value_dim + channel;
                result[index] = value
                    * if g.sigmoid_output {
                        sigmoid(z[index])
                    } else {
                        silu(z[index])
                    };
            }
        }
        Ok(result)
    }
}

fn normalize_l2(
    input: &mut [f32],
    dim: usize,
    multiplier: f32,
) {
    for head in input.chunks_exact_mut(dim) {
        let scale = multiplier / (head.iter().map(|value| value * value).sum::<f32>() + 1e-6).sqrt();
        head.iter_mut().for_each(|value| *value *= scale);
    }
}

fn delta_row(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    value: f32,
    decay: f32,
    beta: f32,
) -> f32 {
    state.iter_mut().for_each(|entry| *entry *= decay);
    let delta = (value - dot(state, k)) * beta;
    for (entry, key) in state.iter_mut().zip(k) {
        *entry += key * delta;
    }
    dot(state, q)
}

struct Attention {
    prefix: String,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    index_q_norm: Vec<f32>,
    index_k_norm: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    index_keys: Vec<Vec<f32>>,
}

impl Attention {
    fn load(
        weights: &Weights,
        prefix: String,
        g: &Geometry,
    ) -> Result<Self> {
        for (name, rows) in [
            ("q_proj", 2 * g.heads * g.head_dim),
            ("k_proj", g.kv_heads * g.head_dim),
            ("v_proj", g.kv_heads * g.head_dim),
        ] {
            matrix(weights, &format!("{prefix}.{name}"), rows, g.dim)?;
        }
        matrix(weights, &format!("{prefix}.o_proj"), g.dim, g.heads * g.head_dim)?;
        matrix(weights, &format!("{prefix}.indexer.index_qk_proj"), (g.index_heads + 1) * g.index_dim, g.dim)?;
        Ok(Self {
            q_norm: vector(weights, &format!("{prefix}.q_norm.weight"), g.head_dim)?,
            k_norm: vector(weights, &format!("{prefix}.k_norm.weight"), g.head_dim)?,
            index_q_norm: vector(weights, &format!("{prefix}.indexer.q_layernorm.weight"), g.index_dim)?,
            index_k_norm: vector(weights, &format!("{prefix}.indexer.k_layernorm.weight"), g.index_dim)?,
            prefix,
            keys: Vec::new(),
            values: Vec::new(),
            index_keys: Vec::new(),
        })
    }

    fn reset(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.index_keys.clear();
    }

    fn visible(
        &self,
        index_query: &[f32],
        g: &Geometry,
    ) -> Result<Vec<usize>> {
        let _profile = Scope::new("qwen.qsa.visible");
        let count = self.index_keys.len();
        if count <= g.budget {
            return Ok((0..count).collect());
        }
        let position = count - 1;
        let mut queries = Vec::with_capacity(g.index_heads);
        for raw_query in index_query.chunks_exact(g.index_dim) {
            let mut query = rms_norm(raw_query, &self.index_q_norm, g.eps)?;
            rope(&mut query, g.rotary_dim, position, g.rope_theta);
            queries.push(query);
        }
        let blocks = count / g.ratio;
        let mut scores = Vec::with_capacity(blocks);
        for block in 0..blocks {
            let mut pooled = vec![0.0; g.index_dim];
            for key in &self.index_keys[block * g.ratio..(block + 1) * g.ratio] {
                for (value, key_value) in pooled.iter_mut().zip(key) {
                    *value += key_value / g.ratio as f32;
                }
            }
            let mut pooled = rms_norm(&pooled, &self.index_k_norm, g.eps)?;
            rope(&mut pooled, g.rotary_dim, block * g.ratio, g.rope_theta);
            let score =
                queries.iter().map(|query| dot(query, &pooled).max(0.0)).sum::<f32>() / (g.index_dim as f32).sqrt();
            scores.push(score);
        }
        selected_tokens(&scores, count, g.ratio, g.budget)
    }

    fn step(
        &mut self,
        weights: &Weights,
        input: &[f32],
        g: &Geometry,
    ) -> Result<Vec<f32>> {
        let _profile = Scope::new("qwen.qsa");
        let position = self.keys.len();
        let [index, query_and_gate, mut keys, values] = projects(
            weights,
            &self.prefix,
            [("indexer.index_qk_proj", input), ("q_proj", input), ("k_proj", input), ("v_proj", input)],
        )?;
        let query_width = g.index_heads * g.index_dim;
        self.index_keys.push(index[query_width..].to_vec());
        let visible = self.visible(&index[..query_width], g)?;
        let mut queries = Vec::with_capacity(g.heads * g.head_dim);
        let mut gates = Vec::with_capacity(g.heads * g.head_dim);
        for head in query_and_gate.chunks_exact(2 * g.head_dim) {
            let mut q = rms_norm(&head[..g.head_dim], &self.q_norm, g.eps)?;
            rope(&mut q, g.rotary_dim, position, g.rope_theta);
            queries.extend(q);
            gates.extend_from_slice(&head[g.head_dim..]);
        }
        for head in keys.chunks_exact_mut(g.head_dim) {
            let normalized = rms_norm(head, &self.k_norm, g.eps)?;
            head.copy_from_slice(&normalized);
            rope(head, g.rotary_dim, position, g.rope_theta);
        }
        self.keys.push(keys);
        self.values.push(values);
        let mut output = vec![0.0; g.heads * g.head_dim];
        for head in 0..g.heads {
            let query = &queries[head * g.head_dim..(head + 1) * g.head_dim];
            let kv_head = head / (g.heads / g.kv_heads);
            let start = kv_head * g.head_dim;
            let scores: Vec<f32> = visible
                .iter()
                .map(|&token| dot(query, &self.keys[token][start..start + g.head_dim]) / (g.head_dim as f32).sqrt())
                .collect();
            let probabilities = softmax(&scores)?;
            for (&token, probability) in visible.iter().zip(probabilities) {
                for channel in 0..g.head_dim {
                    output[head * g.head_dim + channel] += probability * self.values[token][start + channel];
                }
            }
        }
        for (value, gate) in output.iter_mut().zip(gates) {
            *value *= sigmoid(gate);
        }
        project(weights, &format!("{}.o_proj", self.prefix), &output)
    }
}

fn selected_tokens(
    scores: &[f32],
    count: usize,
    ratio: usize,
    budget: usize,
) -> Result<Vec<usize>> {
    if ratio == 0 || budget < ratio || scores.len() != count / ratio {
        return Err(Error::Invalid("Qwen3.8 sparse index geometry"));
    }
    let selected = topk(scores, (budget / ratio).min(scores.len()))?;
    let mut keep = vec![false; count];
    for block in selected {
        keep[block * ratio..(block + 1) * ratio].fill(true);
    }
    // The reference partitions the current query's visible prefix. Its partial
    // block is always visible; it is never scored with future tokens.
    keep[count / ratio * ratio..].fill(true);
    Ok(keep.iter().enumerate().filter_map(|(index, &keep)| keep.then_some(index)).collect())
}

struct Ple {
    prefix: String,
    multipliers: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    rows_per_shard: usize,
    key_norm: Vec<f32>,
    query_norm: Vec<f32>,
    conv_norm: Vec<f32>,
    convolution: Vec<f32>,
    state: ConvState,
}

impl Ple {
    fn load(
        weights: &Weights,
        prefix: String,
        g: &Geometry,
    ) -> Result<Self> {
        let embedding = format!("{prefix}.ple_embedding");
        let multipliers = weights.integers(&format!("{embedding}.layer_multipliers"))?;
        let sizes = weights.integers(&format!("{embedding}.ngram_heads_vocab_sizes"))?;
        let offsets = weights.integers(&format!("{embedding}.ngram_heads_offsets"))?;
        shape(weights, &format!("{embedding}.layer_multipliers"), &[g.ngram_size])?;
        shape(weights, &format!("{embedding}.ngram_heads_vocab_sizes"), &[g.ngram_heads])?;
        shape(weights, &format!("{embedding}.ngram_heads_offsets"), &[g.ngram_heads])?;
        if multipliers.len() != g.ngram_size
            || sizes.len() != g.ngram_heads
            || offsets.len() != g.ngram_heads
            || sizes.iter().any(|size| *size <= 0)
            || offsets.iter().any(|offset| *offset < 0)
        {
            return Err(Error::Invalid("Qwen3.8 n-gram hash buffers"));
        }
        let first = weights.shape(&format!("{embedding}.ngram_embedding.shard_0.weight"))?;
        if first.len() != 2 || first[1] != g.ple_dim / g.ngram_heads {
            return Err(Error::Invalid("Qwen3.8 n-gram shard geometry"));
        }
        let rows_per_shard = first[0];
        for shard in 0..g.shards {
            shape(weights, &format!("{embedding}.ngram_embedding.shard_{shard}.weight"), first)?;
        }
        let capacity = rows_per_shard.checked_mul(g.shards).ok_or(Error::Invalid("Qwen3.8 n-gram capacity"))?;
        if sizes
            .iter()
            .zip(&offsets)
            .any(|(&size, &offset)| offset.checked_add(size).is_none_or(|end| end as u64 > capacity as u64))
        {
            return Err(Error::Invalid("Qwen3.8 n-gram shard range"));
        }
        let width = g.dim * g.streams;
        matrix(weights, &format!("{prefix}.key_proj"), width, g.ple_dim)?;
        matrix(weights, &format!("{prefix}.value_proj"), g.dim, g.ple_dim)?;
        shape(weights, &format!("{prefix}.conv1d.weight"), &[width, g.ple_kernel, 1])?;
        Ok(Self {
            key_norm: vector(weights, &format!("{prefix}.norm_key.weight"), width)?,
            query_norm: vector(weights, &format!("{prefix}.norm_query.weight"), width)?,
            conv_norm: vector(weights, &format!("{prefix}.norm_conv.weight"), width)?,
            convolution: vector(weights, &format!("{prefix}.conv1d.weight"), width * g.ple_kernel)?,
            prefix,
            multipliers,
            sizes,
            offsets,
            rows_per_shard,
            state: ConvState::default(),
        })
    }

    /// Hashing and immutable n-gram lookup depend only on token history. Graph
    /// execution prepares these embeddings before submitting any device work.
    fn gather(
        &self,
        weights: &Weights,
        token: u32,
        history: &[u32],
        g: &Geometry,
    ) -> Result<Vec<f32>> {
        let shifted = shifted_tokens(token, history, g.ngram_size, g.eos);
        let mut embedding = Vec::with_capacity(g.ple_dim);
        let heads_per_size = g.ngram_heads / (g.ngram_size - 1);
        let mut hash = (shifted[0] as i64).wrapping_mul(self.multipliers[0]);
        for n in 2..=g.ngram_size {
            hash ^= (shifted[n - 1] as i64).wrapping_mul(self.multipliers[n - 1]);
            for head in (n - 2) * heads_per_size..(n - 1) * heads_per_size {
                let bucket = ple_bucket(hash, self.sizes[head], g.edge0);
                let global =
                    bucket.checked_add(self.offsets[head]).ok_or(Error::Invalid("Qwen3.8 n-gram index"))? as usize;
                let shard = global / self.rows_per_shard;
                let row = global % self.rows_per_shard;
                if shard >= g.shards {
                    return Err(Error::Invalid("Qwen3.8 n-gram index"));
                }
                embedding.extend(
                    weights.row(&format!("{}.ple_embedding.ngram_embedding.shard_{shard}.weight", self.prefix), row)?,
                );
            }
        }
        Ok(embedding)
    }

    fn step(
        &mut self,
        weights: &Weights,
        input: &[f32],
        token: u32,
        history: &[u32],
        g: &Geometry,
    ) -> Result<Vec<f32>> {
        let _profile = Scope::new("qwen.ple");
        let embedding = self.gather(weights, token, history, g)?;
        let [key, value] = projects(weights, &self.prefix, [("key_proj", &embedding), ("value_proj", &embedding)])?;
        let key = grouped_norm(&key, &self.key_norm, g.dim, g.eps)?;
        let query = grouped_norm(input, &self.query_norm, g.dim, g.eps)?;
        let mut gated = vec![0.0; g.streams * g.dim];
        for stream in 0..g.streams {
            let start = stream * g.dim;
            let score = dot(&key[start..start + g.dim], &query[start..start + g.dim]) / (g.dim as f32).sqrt();
            let sign = if score > 0.0 {
                1.0
            } else if score < 0.0 {
                -1.0
            } else {
                0.0
            };
            let gate = sigmoid(sign * score.abs().max(1e-6).sqrt());
            for channel in 0..g.dim {
                gated[start + channel] = gate * value[channel];
            }
        }
        let norm = grouped_norm(&gated, &self.conv_norm, g.dim, g.eps)?;
        let convolution = self.state.step(&norm, &self.convolution, g.ple_kernel, g.ngram_size)?;
        for (value, convolution) in gated.iter_mut().zip(convolution) {
            *value += convolution;
        }
        Ok(gated)
    }
}

fn ple_bucket(
    hash: i64,
    size: i64,
    edge0: bool,
) -> i64 {
    if edge0 {
        ((hash as u64) % size as u64) as i64
    } else {
        hash.rem_euclid(size)
    }
}

fn shifted_tokens(
    token: u32,
    history: &[u32],
    count: usize,
    eos: u32,
) -> Vec<u32> {
    let mut shifted = vec![token];
    let mut blocked = false;
    for position in 1..count {
        let previous = if blocked {
            eos
        } else {
            history.len().checked_sub(position).map_or(eos, |index| history[index])
        };
        shifted.push(previous);
        blocked |= previous == eos;
    }
    shifted
}

fn validate_moe(
    weights: &Weights,
    prefix: &str,
    g: &Geometry,
) -> Result<()> {
    matrix(weights, &format!("{prefix}.gate"), g.experts, g.dim)?;
    matrix(weights, &format!("{prefix}.shared_expert_gate"), 1, g.dim)?;
    for name in ["gate_proj", "up_proj"] {
        shape(weights, &format!("{prefix}.switch_mlp.{name}.weight"), &[g.experts, g.expert_dim, g.dim])?;
        matrix(weights, &format!("{prefix}.shared_expert.{name}"), g.shared_dim, g.dim)?;
    }
    shape(weights, &format!("{prefix}.switch_mlp.down_proj.weight"), &[g.experts, g.dim, g.expert_dim])?;
    matrix(weights, &format!("{prefix}.shared_expert.down_proj"), g.dim, g.shared_dim)
}

fn moe(
    weights: &Weights,
    prefix: &str,
    input: &[f32],
    g: &Geometry,
) -> Result<Vec<f32>> {
    let _profile = Scope::new("qwen.ffn");
    let [scores, shared_gate, shared_values, shared_up] = projects(
        weights,
        prefix,
        [
            ("gate", input),
            ("shared_expert_gate", input),
            ("shared_expert.gate_proj", input),
            ("shared_expert.up_proj", input),
        ],
    )?;
    let selected = topk(&scores, g.active_experts)?;
    let probabilities = softmax(&selected.iter().map(|&expert| scores[expert]).collect::<Vec<_>>())?;
    #[cfg(feature = "metal")]
    {
        weights.selected_ffn(
            &format!("{prefix}.switch_mlp.gate_proj.weight"),
            &format!("{prefix}.switch_mlp.up_proj.weight"),
            &format!("{prefix}.switch_mlp.down_proj.weight"),
            &format!("{prefix}.shared_expert.down_proj.weight"),
            &selected,
            input,
            &probabilities,
            sigmoid(shared_gate[0]),
            &shared_values,
            &shared_up,
            None,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        let repeated_input = input.repeat(selected.len());
        let mut gate =
            weights.selected_matvec(&format!("{prefix}.switch_mlp.gate_proj.weight"), &selected, &repeated_input)?;
        let up = weights.selected_matvec(&format!("{prefix}.switch_mlp.up_proj.weight"), &selected, &repeated_input)?;
        for (value, up) in gate.iter_mut().zip(up) {
            *value = silu(*value) * up;
        }
        let expert_outputs =
            weights.selected_matvec(&format!("{prefix}.switch_mlp.down_proj.weight"), &selected, &gate)?;
        let mut result = vec![0.0; g.dim];
        for (output, probability) in expert_outputs.chunks_exact(g.dim).zip(probabilities) {
            for (value, &expert) in result.iter_mut().zip(output) {
                *value += probability * expert;
            }
        }
        let shared_gate = sigmoid(shared_gate[0]);
        let mut shared_values = shared_values;
        for (value, up) in shared_values.iter_mut().zip(shared_up) {
            *value = silu(*value) * up;
        }
        let shared = project(weights, &format!("{prefix}.shared_expert.down_proj"), &shared_values)?;
        for (value, shared) in result.iter_mut().zip(shared) {
            *value += shared_gate * shared;
        }
        Ok(result)
    }
}

enum Mixer {
    Delta(DeltaNet),
    Attention(Attention),
}

struct Layer {
    edge0_layer: Option<usize>,
    prefix: String,
    attn_hyper: HyperConnection,
    mlp_hyper: HyperConnection,
    mixer: Mixer,
    ple: Option<Ple>,
}

pub struct QwenModel {
    weights: Weights,
    geometry: Geometry,
    layers: Vec<Layer>,
    final_mix: HyperConnection,
    embedding: String,
    readout: String,
    position: usize,
    history: Vec<u32>,
    #[cfg(feature = "metal")]
    graph: Option<decode_graph::State>,
    #[cfg(feature = "metal")]
    graph_enabled: bool,
    #[cfg(feature = "metal")]
    checkpoint_stamp: Option<checkpoint::Stamp>,
}

pub(super) fn forced_scalar_attention() -> bool {
    std::env::var_os("STARK_QWEN38_SCALAR_ATTENTION").is_some_and(|value| value == "1")
}

impl Geometry {
    fn attention_partial_elements(
        &self,
        count: usize,
    ) -> Option<usize> {
        self.heads
            .checked_mul(count.div_ceil(256))?
            .checked_mul(self.head_dim.checked_add(2)?)
            .filter(|&elements| elements <= 8_000_000)
    }

    fn scalar_attention_capacity(
        &self,
        count: u32,
        forced: bool,
    ) -> Option<usize> {
        // Every request passes through the short scalar prefix, even when its
        // final step uses split attention. Oversized split workspace and the
        // explicit scalar override can still need full request-sized arrays.
        let rows = if forced || self.attention_partial_elements(count as usize).is_none() {
            count
        } else {
            count.min(255)
        };
        super::memory::growth_capacity(rows, self.context, 16)
    }

    fn recurrent_inventory_bytes(
        &self,
        layers: usize,
    ) -> Option<usize> {
        super::memory::bytes(&[layers, self.value_heads, self.value_dim, self.key_dim], size_of::<f32>())
    }

    #[cfg(test)]
    fn request_memory_bytes(
        &self,
        layer_count: usize,
        attention_layers: usize,
        context_tokens: u32,
    ) -> Option<usize> {
        let _ = layer_count;
        self.execution_workspace(context_tokens, attention_layers, true)
    }

    fn graph_cache_bytes(
        &self,
        capacity: usize,
    ) -> Option<usize> {
        let g = self;
        let key_width = g.kv_heads.checked_mul(g.head_dim)?;
        let pooled = (capacity / g.ratio).max(1);
        super::memory::sum(&[
            super::memory::bytes(&[capacity, key_width.checked_mul(2)?.checked_add(g.index_dim)?], 4)?,
            super::memory::bytes(&[pooled, g.index_dim.checked_add(1)?], 4)?,
            super::memory::bytes(&[capacity], 4)?,
            super::memory::bytes(&[g.index_heads, g.index_dim], 4)?,
            super::memory::bytes(&[2, g.heads, g.head_dim], 4)?,
            super::memory::bytes(&[2, g.budget / g.ratio], 4)?,
        ])
    }

    fn execution_workspace(
        &self,
        context_tokens: u32,
        attention_layers: usize,
        graph: bool,
    ) -> Option<usize> {
        let g = self;
        let capacity = super::memory::growth_capacity(context_tokens, g.context, 16)?;
        let cache = if graph {
            // reset drops the graph and its attention owners. Each new graph
            // owns only device rows, not a duplicate scalar F32 host cache.
            let attention = super::memory::growing_owner_bytes(0, capacity, 16, |rows| g.graph_cache_bytes(rows))?;
            let scalar_capacity = g.scalar_attention_capacity(context_tokens, forced_scalar_attention())?;
            let scalar = super::memory::growing_owner_bytes(0, scalar_capacity, 16, |rows| {
                super::memory::bytes(&[2, rows, g.heads], 4)
            })?;
            attention.checked_add(scalar)?.checked_mul(attention_layers)?
        } else {
            // Scalar row payloads are dropped by reset. Outer descriptor Vecs
            // remain resident and are accounted separately from actual capacity.
            super::memory::bytes(
                &[
                    attention_layers,
                    context_tokens as usize,
                    g.kv_heads.checked_mul(g.head_dim)?.checked_mul(2)?.checked_add(g.index_dim)?,
                ],
                4,
            )?
        };
        let width = g
            .dim
            .checked_mul(g.streams)?
            .max(g.expert_dim.checked_mul(g.active_experts)?)
            .max(g.shared_dim)
            .max(g.heads.checked_mul(g.head_dim)?)
            .max(g.index_heads.checked_mul(g.index_dim)?)
            .max(g.dim.checked_mul(g.active_experts)?);
        let workspace = if graph {
            // This is the actual per-command arena allocation ceiling, not a
            // persistent model allocation. Do not also bill scalar activations.
            // Initial embedding upload is outside that arena; its host repeat
            // can coexist with the uploaded owner.
            let (arenas, batch) = if g.edge0_metal {
                // Router fences replace command segments while earlier device
                // owners can still be retained. Charge three bounded arenas.
                #[cfg(feature = "metal")]
                {
                    (decode_graph::EDGE0_GRAPH_SCRATCH_BYTES.checked_mul(3)?, decode_graph::EDGE0_PREFILL_BATCH)
                }
                #[cfg(not(feature = "metal"))]
                {
                    return None;
                }
            } else {
                (512 * 1024 * 1024, 1)
            };
            super::memory::sum(&[
                arenas,
                super::memory::bytes(&[batch, g.dim.checked_mul(g.streams)?.checked_mul(2)?.checked_add(g.dim)?], 4)?,
                // Router host scores, F64 sort/normalization scratch and
                // selected route descriptors coexist across a batch fence.
                if g.edge0_metal {
                    super::memory::bytes(&[batch, g.experts], 32)?
                        .checked_add(super::memory::bytes(&[batch, g.active_experts], 32)?)?
                } else {
                    0
                },
            ])?
        } else {
            // Scalar mapped kernels have layer-local host/device activation
            // copies, without a DecodeGraph arena.
            super::memory::bytes(&[width], 256)?
        };
        super::memory::sum(&[cache, workspace])
    }
}

impl QwenModel {
    pub(super) fn edge0_execution_metadata(&self) -> Option<Value> {
        if !self.geometry.edge0 {
            return None;
        }
        #[cfg(feature = "metal")]
        let graph = self.geometry.edge0_metal && self.graph_enabled;
        #[cfg(not(feature = "metal"))]
        let graph = false;
        Some(serde_json::json!({
            "backend": if graph { "edge0-metal-graph-v1" } else { "edge0-f32-reference" },
            "activation_precision": "f32",
            "accumulation_precision": "f32",
            "recurrent_state_precision": "f32",
            "attention_cache_precision": "f32",
            "prefill_max_batch": if graph { 8 } else { 1 },
            "prefix_reuse": false,
            "weights": self.weights.edge0_metal_metadata(),
        }))
    }

    pub(super) fn additional_scoring_device_readout(&self) -> bool {
        #[cfg(feature = "metal")]
        return !self.graph_enabled;
        #[cfg(not(feature = "metal"))]
        false
    }

    pub(super) fn prepare_request_memory(&mut self) -> Result<bool> {
        if self.position != 0 {
            return Err(Error::Invalid("Qwen request memory preparation requires reset state"));
        }
        #[cfg(feature = "metal")]
        return Ok(self.graph.take().is_some());
        #[cfg(not(feature = "metal"))]
        Ok(false)
    }
    pub(super) fn request_memory_bytes(
        &self,
        context_tokens: u32,
    ) -> Option<usize> {
        #[cfg(feature = "metal")]
        if self.graph_enabled {
            let g = &self.geometry;
            let capacity = super::memory::growth_capacity(context_tokens, g.context, 16)?;
            super::memory::gpu_buffer_fits(&[capacity, g.kv_heads, g.head_dim], 4)?;
            super::memory::gpu_buffer_fits(&[capacity, g.index_dim], 4)?;
            let scalar_capacity = g.scalar_attention_capacity(context_tokens, forced_scalar_attention())?;
            super::memory::gpu_buffer_fits(&[scalar_capacity, g.heads], 4)?;
        }
        let g = &self.geometry;
        #[cfg(feature = "metal")]
        let graph = self.graph_enabled;
        #[cfg(not(feature = "metal"))]
        let graph = false;
        let attention = self.layers.iter().filter(|layer| matches!(layer.mixer, Mixer::Attention(_))).count();
        let mut bytes = g.execution_workspace(context_tokens, attention, graph)?;
        #[cfg(feature = "metal")]
        let batch = if graph && g.edge0_metal {
            decode_graph::EDGE0_PREFILL_BATCH
        } else {
            1
        };
        #[cfg(not(feature = "metal"))]
        let batch = 1;
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Delta(state) => {
                    bytes = bytes.checked_add(state.additional_memory_bytes(g)?)?;
                },
                Mixer::Attention(state) => {
                    if graph {
                        // Graph controls upload these immutable host weights
                        // once per newly constructed graph state.
                        let norms = state
                            .q_norm
                            .len()
                            .checked_add(state.k_norm.len())?
                            .checked_add(state.index_q_norm.len())?
                            .checked_add(state.index_k_norm.len())?;
                        bytes = bytes.checked_add(norms.checked_mul(4)?)?;
                    } else {
                        for capacity in [state.keys.capacity(), state.values.capacity(), state.index_keys.capacity()] {
                            bytes = bytes.checked_add(super::memory::vector_growth_bytes::<Vec<f32>>(
                                capacity,
                                context_tokens as usize,
                            )?)?;
                        }
                    }
                },
            }
            if graph {
                bytes = bytes.checked_add(
                    layer.attn_hyper.norm.len().checked_add(layer.mlp_hyper.norm.len())?.checked_mul(4)?,
                )?;
                if let Some(index) = layer.edge0_layer {
                    // Each graph owns a device copy of the stored column scale
                    // vector; the existing source host vector remains resident.
                    bytes = bytes.checked_add(self.weights.edge0_column_scales(index).ok()?.len().checked_mul(4)?)?;
                }
            }
            if let Some(ple) = &layer.ple {
                let wide = g.dim.checked_mul(g.streams)?;
                let history = wide.checked_mul(g.ple_kernel - 1)?.checked_mul(g.ngram_size)?;
                if graph {
                    let controls = super::memory::sum(&[
                        ple.key_norm.len(),
                        ple.query_norm.len(),
                        ple.conv_norm.len(),
                        ple.convolution.len(),
                        history,
                        wide,
                    ])?;
                    bytes = bytes.checked_add(controls.checked_mul(4)?)?;
                    // Gathered PLE embeddings are separately uploaded for all
                    // layers before the graph begins. One host gather and its
                    // current row may overlap the last upload.
                    bytes = bytes.checked_add(super::memory::bytes(&[batch, g.ple_dim], 4)?)?;
                } else {
                    bytes = bytes.checked_add(super::memory::vector_resize_bytes::<f32>(
                        ple.state.history.capacity(),
                        history,
                    )?)?;
                }
            }
        }
        if graph {
            bytes = bytes.checked_add(self.final_mix.norm.len().checked_mul(4)?)?;
            if self.layers.iter().any(|layer| layer.ple.is_some()) {
                bytes = bytes.checked_add(super::memory::bytes(&[batch, g.ple_dim], 8)?)?;
            }
        }
        bytes = bytes.checked_add(super::memory::vector_growth_bytes::<u32>(self.history.capacity(), g.ngram_size)?)?;
        if g.edge0_metal {
            // Available memory already reflects wired owners. Admit only the
            // registry's remaining bank/static reservation once per request.
            bytes = bytes.checked_add(self.weights.edge0_additional_weight_bytes()?)?;
        }
        Some(bytes)
    }

    pub fn new(
        weights: Weights,
        config: &Value,
    ) -> Result<Self> {
        let edge0 = weights.is_edge0();
        if edge0 && !cfg!(feature = "metal") && std::env::var_os("STARK_EDGE0_METAL").is_some_and(|value| value == "1")
        {
            return Err(Error::Unsupported("Edge0 Metal requires a Metal-enabled build"));
        }
        if config.get("model_type").and_then(Value::as_str) != Some("qwen4_exp") {
            return Err(Error::Unsupported("Qwen3.8 architecture"));
        }
        let text = config.get("text_config").ok_or(Error::Invalid("Qwen3.8 text configuration"))?;
        if text.get("model_type").and_then(Value::as_str) != Some("qwen4_exp_text") {
            return Err(Error::Unsupported("Qwen3.8 text architecture"));
        }
        if text.get("hidden_act").and_then(Value::as_str).unwrap_or("silu") != "silu"
            || text.get("attention_bias").and_then(Value::as_bool).unwrap_or(false)
            || (!edge0 && config.get("quantization").and_then(Value::as_object).is_none())
        {
            return Err(Error::Unsupported("Qwen3.8 activation, biases or unconverted normalization weights"));
        }
        let head_dim = integer(text, "head_dim", 1024)?;
        let rope_config = text.get("rope_parameters").unwrap_or(text);
        if rope_config.get("rope_type").and_then(Value::as_str).unwrap_or("default") != "default" {
            return Err(Error::Unsupported("Qwen3.8 rotary scaling"));
        }
        let partial = positive(rope_config, "partial_rotary_factor")?;
        let rotary_width = head_dim as f32 * partial;
        if rotary_width.fract() != 0.0 {
            return Err(Error::Invalid("Qwen3.8 rotary dimensions must be integral"));
        }
        let rotary_dim = rotary_width as usize;
        let ngram_size = integer(text, "ngram_size", 8)?;
        if ngram_size < 2 {
            return Err(Error::Invalid("Qwen3.8 n-gram size"));
        }
        let eos = match text.get("eos_token_id") {
            Some(Value::Array(ids)) => ids.first().and_then(Value::as_u64),
            Some(id) => id.as_u64(),
            None => None,
        }
        .filter(|&id| id <= u32::MAX as u64)
        .ok_or(Error::Invalid("Qwen3.8 EOS"))? as u32;
        let output_gate = text.get("output_gate_type").and_then(Value::as_str).unwrap_or("sigmoid");
        if !matches!(output_gate, "sigmoid" | "silu") {
            return Err(Error::Unsupported("Qwen3.8 recurrent output activation"));
        }
        let g = Geometry {
            edge0,
            edge0_metal: edge0
                && cfg!(feature = "metal")
                && std::env::var_os("STARK_EDGE0_METAL").is_some_and(|value| value == "1"),
            dim: integer(text, "hidden_size", 16_384)?,
            vocab: integer(text, "vocab_size", 1_000_000)?,
            streams: integer(text, "hc_count", 16)?,
            low_rank: integer(text, "hc_lowrank", 16_384)?,
            heads: integer(text, "num_attention_heads", 256)?,
            kv_heads: integer(text, "num_key_value_heads", 256)?,
            head_dim,
            rotary_dim,
            rope_theta: positive(rope_config, "rope_theta")?,
            key_heads: integer(text, "linear_num_key_heads", 256)?,
            value_heads: integer(text, "linear_num_value_heads", 256)?,
            key_dim: integer(text, "linear_key_head_dim", 1024)?,
            value_dim: integer(text, "linear_value_head_dim", 1024)?,
            conv_kernel: integer(text, "linear_conv_kernel_dim", 32)?,
            experts: integer(text, "num_experts", 4096)?,
            active_experts: integer(text, "num_experts_per_tok", 256)?,
            expert_dim: integer(text, "moe_intermediate_size", 131_072)?,
            shared_dim: integer(text, "shared_expert_intermediate_size", 131_072)?,
            index_heads: integer(text, "indexer_n_heads", 256)?,
            index_dim: integer(text, "indexer_head_dim", 1024)?,
            budget: integer(text, "indexer_budget", 1_048_576)?,
            ratio: integer(text, "indexer_compress_ratio", 256)?,
            eps: positive(text, "rms_norm_eps")?,
            sigmoid_output: output_gate == "sigmoid",
            context: integer(text, "max_position_embeddings", 1_048_576)?,
            ngram_size,
            ngram_heads: (ngram_size - 1) * integer(text, "heads_per_ngram", 256)?,
            ple_dim: integer(text, "ple_embed_dim", 16_384)?,
            ple_kernel: integer(text, "ple_conv_kernel_size", 32)?,
            shards: integer(text, "split_ngram_parts", 4096)?,
            eos,
        };
        if g.edge0 && g.context > 512 {
            return Err(Error::Unsupported("Edge0 reference context qualification"));
        }
        if !g.heads.is_multiple_of(g.kv_heads)
            || !g.value_heads.is_multiple_of(g.key_heads)
            || g.active_experts > g.experts
            || g.rotary_dim == 0
            || !g.rotary_dim.is_multiple_of(2)
            || g.rotary_dim > g.head_dim
            || g.rotary_dim > g.index_dim
            || !g.ple_dim.is_multiple_of(g.ngram_heads)
            || g.conv_kernel < 2
            || g.ple_kernel < 2
            || g.budget < g.ratio
            || !g.budget.is_multiple_of(g.ratio)
            || g.eos as usize >= g.vocab
            || integer(text, "indexer_kv_heads", 256)? != 1
        {
            return Err(Error::Unsupported("Qwen3.8 geometry or indexer grouping"));
        }
        // Preserve the native per-buffer kernel geometry limit. Recurrent
        // owners are lazy and their aggregate bytes are admitted per request.
        if g.value_heads * g.value_dim * g.key_dim > 16_777_216 {
            return Err(Error::Invalid("Qwen3.8 recurrent state bounds"));
        }
        let count = integer(text, "num_hidden_layers", 256)?;
        let layer_types = text
            .get("layer_types")
            .and_then(Value::as_array)
            .filter(|types| types.len() == count)
            .ok_or(Error::Invalid("Qwen3.8 layer schedule"))?;
        let recurrent_layers = layer_types.iter().filter(|kind| kind.as_str() == Some("linear_attention")).count();
        g.recurrent_inventory_bytes(recurrent_layers)
            .ok_or(Error::Invalid("Qwen3.8 aggregate recurrent state overflow"))?;
        let ple_layers =
            text.get("ple_layer_ids").and_then(Value::as_array).ok_or(Error::Invalid("Qwen3.8 PLE schedule"))?;
        let mut ple_ids = Vec::new();
        for id in ple_layers {
            let id =
                id.as_u64().filter(|&id| id > 0 && id <= count as u64).ok_or(Error::Invalid("Qwen3.8 PLE layer"))?
                    as usize;
            if ple_ids.contains(&id) {
                return Err(Error::Invalid("duplicate Qwen3.8 PLE layer"));
            }
            ple_ids.push(id);
        }
        let base = if weights.has("language_model.model.embed_tokens.weight") {
            "language_model.model"
        } else if weights.has("model.embed_tokens.weight") {
            "model"
        } else {
            return Err(Error::Invalid("Qwen3.8 embedding"));
        };
        let embedding = format!("{base}.embed_tokens.weight");
        shape(&weights, &embedding, &[g.vocab, g.dim])?;
        let tied = text.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false);
        let readout = if tied {
            embedding.clone()
        } else if base.starts_with("language_model.") {
            "language_model.lm_head.weight".to_string()
        } else {
            "lm_head.weight".to_string()
        };
        shape(&weights, &readout, &[g.vocab, g.dim])?;
        let final_mix = HyperConnection::load(&weights, format!("{base}.hyper_connection_mixer"), &g, false)?;
        let mut layers = Vec::with_capacity(count);
        for (index, kind) in layer_types.iter().enumerate() {
            let prefix = format!("{base}.layers.{index}");
            let mixer = match kind.as_str() {
                Some("linear_attention") => {
                    Mixer::Delta(DeltaNet::load(&weights, format!("{prefix}.linear_attn"), &g)?)
                },
                Some("full_attention") => {
                    Mixer::Attention(Attention::load(&weights, format!("{prefix}.self_attn"), &g)?)
                },
                _ => return Err(Error::Unsupported("Qwen3.8 token mixer")),
            };
            if edge0 {
                edge0::validate(&weights, index, &g, &format!("{prefix}.mlp"))?;
            } else {
                validate_moe(&weights, &format!("{prefix}.mlp"), &g)?;
            }
            let ple = if ple_ids.contains(&(index + 1)) {
                Some(Ple::load(&weights, format!("{prefix}.ple"), &g)?)
            } else {
                None
            };
            layers.push(Layer {
                edge0_layer: edge0.then_some(index),
                attn_hyper: HyperConnection::load(&weights, format!("{prefix}.attn_hyper_connection"), &g, true)?,
                mlp_hyper: HyperConnection::load(&weights, format!("{prefix}.mlp_hyper_connection"), &g, true)?,
                prefix,
                mixer,
                ple,
            });
        }
        #[cfg(feature = "metal")]
        let graph_enabled = g.edge0_metal
            || (!g.edge0
                && g.active_experts <= 64
                && weights.indexed_banks_admitted()
                && std::env::var_os("STARK_QWEN38_DECODE_GRAPH").is_none_or(|value| value != "0"));
        let model = Self {
            weights,
            geometry: g,
            layers,
            final_mix,
            embedding,
            readout,
            position: 0,
            history: Vec::new(),
            #[cfg(feature = "metal")]
            graph: None,
            #[cfg(feature = "metal")]
            graph_enabled,
            #[cfg(feature = "metal")]
            checkpoint_stamp: None,
        };
        #[cfg(feature = "metal")]
        if model.geometry.edge0_metal {
            let dynamic = model
                .request_memory_bytes(model.geometry.context as u32)
                .ok_or(Error::Invalid("Edge0 Metal dynamic allocation inventory"))?;
            model.weights.prepare_edge0_metal(dynamic)?;
        }
        Ok(model)
    }

    pub fn reset(&mut self) {
        #[cfg(feature = "metal")]
        {
            if let Some(stamp) = &mut self.checkpoint_stamp {
                stamp.invalidate();
            }
            self.graph = None;
        }
        self.position = 0;
        self.history.clear();
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Delta(mixer) => mixer.reset(),
                Mixer::Attention(mixer) => mixer.reset(),
            }
            if let Some(ple) = &mut layer.ple {
                ple.state.clear();
            }
        }
    }

    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.advance(token, true)
    }

    /// Advance recurrent, attention and n-gram state for a prefix token without
    /// computing the stateless final mixer and vocabulary projection.
    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        self.advance(token, false).map(|_| ())
    }

    pub fn ingest_prefix(
        &mut self,
        tokens: &[u32],
        mut before_token: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let result = (|| {
            if tokens.iter().any(|&token| token as usize >= self.geometry.vocab)
                || self.position.checked_add(tokens.len()).is_none_or(|end| end > self.geometry.context)
            {
                return Err(Error::Invalid("Qwen3.8 token or context boundary"));
            }
            #[cfg(feature = "metal")]
            if self.geometry.edge0_metal && self.graph_enabled {
                return objc2::rc::autoreleasepool(|_| decode_graph::prefill(self, tokens, &mut before_token));
            }
            for &token in tokens {
                before_token()?;
                self.ingest(token)?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn advance(
        &mut self,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "metal")]
        {
            objc2::rc::autoreleasepool(|_| self.advance_inner(token, readout))
        }
        #[cfg(not(feature = "metal"))]
        {
            self.advance_inner(token, readout)
        }
    }

    fn advance_inner(
        &mut self,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        let result = self.step_inner(token, readout);
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn step_inner(
        &mut self,
        token: u32,
        readout: bool,
    ) -> Result<Vec<f32>> {
        let g = &self.geometry;
        if token as usize >= g.vocab || self.position >= g.context {
            return Err(Error::Invalid("Qwen3.8 token or context boundary"));
        }
        #[cfg(feature = "metal")]
        if self.graph_enabled {
            return decode_graph::step(self, token, readout);
        }
        let embedding = {
            let _profile = Scope::new("qwen.embedding");
            self.weights.row(&self.embedding, token as usize)?
        };
        let mut hidden = embedding.repeat(g.streams);
        for layer in &mut self.layers {
            if let Some(ple) = &mut layer.ple {
                let extra = ple.step(&self.weights, &hidden, token, &self.history, g)?;
                for (value, extra) in hidden.iter_mut().zip(extra) {
                    *value += extra;
                }
            }
            let (input, gates) = layer.attn_hyper.mix(&self.weights, &hidden, g)?;
            let output = match &mut layer.mixer {
                Mixer::Delta(mixer) => mixer.step(&self.weights, &input, g)?,
                Mixer::Attention(mixer) => mixer.step(&self.weights, &input, g)?,
            };
            inject(&mut hidden, &output, &gates, g)?;
            let (input, gates) = layer.mlp_hyper.mix(&self.weights, &hidden, g)?;
            let output = if let Some(index) = layer.edge0_layer {
                edge0::moe(&self.weights, index, &format!("{}.mlp", layer.prefix), &input, g)?
            } else {
                moe(&self.weights, &format!("{}.mlp", layer.prefix), &input, g)?
            };
            inject(&mut hidden, &output, &gates, g)?;
        }
        // The next projection normally rejects nonfinite residuals. Prefix
        // ingestion must preserve that failure boundary when it skips readout.
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("Qwen3.8 residual values"));
        }
        let logits = if readout {
            let (mixed, _) = self.final_mix.mix(&self.weights, &hidden, g)?;
            let logits = {
                let _profile = Scope::new("qwen.readout");
                self.weights.matvec(&self.readout, &mixed)?
            };
            if logits.len() != g.vocab || logits.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("Qwen3.8 readout values"));
            }
            logits
        } else {
            Vec::new()
        };
        self.position += 1;
        self.history.push(token);
        if self.history.len() >= g.ngram_size {
            self.history.remove(0);
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn qwen_request_inventory_covers_executing_graph_cache_growth_and_scores() {
        let mut g = Geometry {
            dim: 128,
            streams: 4,
            heads: 8,
            kv_heads: 2,
            head_dim: 64,
            key_heads: 2,
            value_heads: 4,
            key_dim: 64,
            value_dim: 64,
            conv_kernel: 4,
            active_experts: 2,
            expert_dim: 256,
            shared_dim: 256,
            index_heads: 4,
            index_dim: 32,
            ratio: 4,
            context: 4096,
            ple_kernel: 4,
            ngram_size: 3,
            ..Geometry::default()
        };
        let at_64 = g.request_memory_bytes(4, 2, 64).unwrap();
        assert_eq!(g.request_memory_bytes(4, 2, 63), Some(at_64));
        let at_128 = g.request_memory_bytes(4, 2, 65).unwrap();
        // Only the executing graph's cache is allocated. Its final growth
        // keeps the previous half-size owners until the command completes.
        let graph_row = (2 * 2 * 64 + 32 + 2 * 8 + 1) * 4;
        let growth = 2 * 96 * graph_row + 2 * 24 * (32 + 1) * 4;
        assert_eq!(at_128 - at_64, growth);
        assert_eq!(
            crate::encodable_block::memory::automatic_context(1, 4096, at_64, |context| {
                g.request_memory_bytes(4, 2, context).map(crate::encodable_block::memory::MemoryEstimate::state)
            }),
            Some(64)
        );
        assert!(g.request_memory_bytes(4, 2, 4097).is_none());
        g.heads = usize::MAX;
        assert!(g.request_memory_bytes(4, 2, 64).is_none());
    }

    #[cfg(feature = "metal")]
    #[stark_test]
    fn edge0_graph_inventory_bills_three_segments_batch_uploads_and_router_host_owners() {
        let mut g = Geometry {
            edge0: true,
            dim: 2560,
            streams: 4,
            heads: 16,
            kv_heads: 2,
            head_dim: 256,
            experts: 512,
            active_experts: 10,
            expert_dim: 1792,
            shared_dim: 2048,
            index_heads: 16,
            index_dim: 128,
            ratio: 4,
            context: 512,
            ..Geometry::default()
        };
        let reference = g.execution_workspace(128, 12, true).unwrap();
        g.edge0_metal = true;
        let candidate = g.execution_workspace(128, 12, true).unwrap();
        let embedding = (g.dim * g.streams * 2 + g.dim) * 4;
        let router = 8 * (g.experts + g.active_experts) * 32;
        assert_eq!(candidate + 512 * 1024 * 1024, reference + 3 * 64 * 1024 * 1024 + 7 * embedding + router);
        // Context/cache ownership remains bounded independently of grouping.
        assert!(g.execution_workspace(513, 12, true).is_none());
        g.experts = usize::MAX;
        assert!(g.execution_workspace(128, 12, true).is_none());
    }

    #[stark_test]
    fn qwen_scalar_attention_inventory_tracks_short_prefix_and_long_fallback() {
        let mut g = Geometry {
            heads: 8,
            head_dim: 128,
            context: 262_144,
            ..Geometry::default()
        };
        assert_eq!(g.scalar_attention_capacity(1, false), Some(16));
        assert_eq!(g.scalar_attention_capacity(255, false), Some(256));
        assert_eq!(g.scalar_attention_capacity(4096, false), Some(256));
        assert_eq!(g.scalar_attention_capacity(262_144, false), Some(256));
        assert_eq!(g.scalar_attention_capacity(4096, true), Some(4096));
        g.heads = 64;
        assert!(g.attention_partial_elements(262_144).is_none());
        assert_eq!(g.scalar_attention_capacity(262_144, false), Some(262_144));
        g.head_dim = usize::MAX;
        assert!(g.attention_partial_elements(256).is_none());
    }

    #[stark_test]
    fn qwen_lazy_recurrent_inventory_has_no_aggregate_memory_ceiling() {
        let mut g = Geometry {
            value_heads: 256,
            value_dim: 256,
            key_dim: 256,
            ..Geometry::default()
        };
        // Nine individually valid native state buffers exceed the former
        // aggregate 512 MiB load-time cap without allocating them at load.
        assert_eq!(g.recurrent_inventory_bytes(9), Some(9 * 64 * 1024 * 1024));
        g.value_heads = usize::MAX;
        assert_eq!(g.recurrent_inventory_bytes(9), None);
    }

    #[stark_test]
    fn qwen_recurrent_inventory_tracks_actual_retained_owners_after_reset() {
        let g = Geometry {
            key_heads: 1,
            key_dim: 2,
            value_heads: 2,
            value_dim: 2,
            conv_kernel: 3,
            eps: 1e-5,
            ..Geometry::default()
        };
        let width = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim;
        let mut state = DeltaNet {
            prefix: String::new(),
            convolution: vec![0.0; width * g.conv_kernel],
            a_log: vec![0.0; g.value_heads],
            dt_bias: vec![0.0; g.value_heads],
            norm: vec![1.0; g.value_dim],
            #[cfg(feature = "metal")]
            metal: None,
            conv_state: ConvState::default(),
            state: Vec::new(),
        };
        let cold = state.additional_memory_bytes(&g).unwrap();
        assert!(cold > 0);
        #[cfg(feature = "metal")]
        {
            state.metal = Some(
                super::super::weight_metal::QwenGdnState::new(
                    super::super::weight_metal::QwenGdnGeometry {
                        key_heads: g.key_heads,
                        key_dim: g.key_dim,
                        value_heads: g.value_heads,
                        value_dim: g.value_dim,
                        conv_kernel: g.conv_kernel,
                        eps: g.eps,
                        sigmoid_output: false,
                    },
                    &state.convolution,
                    &state.a_log,
                    &state.dt_bias,
                    &state.norm,
                )
                .unwrap(),
            );
        }
        #[cfg(not(feature = "metal"))]
        {
            state.state.resize(g.value_heads * g.value_dim * g.key_dim, 0.0);
            state.conv_state.history.resize(width * (g.conv_kernel - 1), 0.0);
        }
        assert_eq!(state.additional_memory_bytes(&g), Some(0));
        state.reset();
        assert_eq!(state.additional_memory_bytes(&g), Some(0));
        #[cfg(feature = "metal")]
        {
            state.metal = None;
        }
        #[cfg(not(feature = "metal"))]
        {
            state.state = Vec::new();
            state.conv_state = ConvState::default();
        }
        assert_eq!(state.additional_memory_bytes(&g), Some(cold));
    }

    #[stark_test]
    fn qwen_ngram_history_stops_at_eos_without_losing_current_eos_context() {
        assert_eq!(shifted_tokens(9, &[2, 3], 3, 0), [9, 3, 2]);
        assert_eq!(shifted_tokens(9, &[2, 0], 3, 0), [9, 0, 0]);
        assert_eq!(shifted_tokens(0, &[2, 3], 3, 0), [0, 3, 2]);
        assert_eq!(shifted_tokens(9, &[], 3, 0), [9, 0, 0]);
        // Hashing must never round the checkpoint's signed 64-bit multipliers.
        let multiplier = 9_007_199_254_740_993_i64;
        assert_ne!(multiplier.wrapping_mul(17), (multiplier as f32 as i64).wrapping_mul(17));
    }

    #[stark_test]
    fn edge0_ple_unsigned_modulo_keeps_source_semantics_outside_the_pinned_reachable_range() {
        // A synthetic overflow counterexample, not a token/artifact witness.
        assert_eq!(ple_bucket(i64::MIN, 11, true), 8);
        assert_eq!(ple_bucket(i64::MIN, 11, false), 3);
        for multiplier in [23_703_573_157_769_u64, 20_109_073_645_365, 8_052_911_324_071] {
            assert!(248_319_u64 * multiplier < (1_u64 << 63));
        }
        // Each valid pinned product has its sign bit clear, so their XOR does
        // too. The two policies coincide on this artifact's reachable hashes.
    }

    #[stark_test]
    fn edge0_gdn_tiled_negative_a_matches_independent_grouped_f64_recurrence() {
        // A post-convolution semantic unit, independent of the external latent
        // fixture. The reference stays in original grouped HF head order and
        // does not call the production normalization or delta-row helpers.
        let g = Geometry {
            edge0: true,
            key_heads: 2,
            key_dim: 2,
            value_heads: 4,
            value_dim: 2,
            eps: 1e-5,
            sigmoid_output: true,
            ..Geometry::default()
        };
        let order = [0, 2, 1, 3];
        let negative_a = [-0.5_f32, -1.0, -1.5, -2.0];
        let dt = [0.1_f32, -0.2, 0.3, -0.4];
        let norm = [0.9_f32, 1.1];
        let initial: Vec<f32> = (0..16).map(|i| (i as f32 - 7.0) * 0.025).collect();
        let mut reference_state: Vec<f64> = initial.iter().copied().map(f64::from).collect();
        let mut native = DeltaNet {
            prefix: String::new(),
            convolution: Vec::new(),
            a_log: order.iter().map(|&h| negative_a[h]).collect(),
            dt_bias: order.iter().map(|&h| dt[h]).collect(),
            norm: norm.to_vec(),
            #[cfg(feature = "metal")]
            metal: None,
            conv_state: ConvState::default(),
            state: order.iter().flat_map(|&h| initial[h * 4..h * 4 + 4].iter().copied()).collect(),
        };
        for step in 0..2 {
            let q = [0.4_f32 + step as f32 * 0.03, -0.8, 0.7, 0.2];
            let k = [0.6_f32, 0.1, -0.3, 0.9 - step as f32 * 0.02];
            let v: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 0.15 + step as f32 * 0.1).collect();
            let z: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.1).collect();
            let a = [-0.6_f32, 0.2, 0.8, -0.3];
            let b = [0.1_f32, -0.4, 0.7, -0.2];
            let mut convolved = q.to_vec();
            convolved.extend(k);
            convolved.extend(order.iter().flat_map(|&h| v[h * 2..h * 2 + 2].iter().copied()));
            let tiled_z: Vec<f32> = order.iter().flat_map(|&h| z[h * 2..h * 2 + 2].iter().copied()).collect();
            let tiled_a: Vec<f32> = order.iter().map(|&h| a[h]).collect();
            let tiled_b: Vec<f32> = order.iter().map(|&h| b[h]).collect();
            let actual = native.recurrent_step(&convolved, &tiled_z, &tiled_a, &tiled_b, &g).unwrap();
            let mut reference = [0.0_f64; 8];
            for head in 0..4 {
                let key = head / 2;
                let mut qn = [f64::from(q[key * 2]), f64::from(q[key * 2 + 1])];
                let mut kn = [f64::from(k[key * 2]), f64::from(k[key * 2 + 1])];
                let qs = 1.0 / ((qn[0] * qn[0] + qn[1] * qn[1] + 1e-6) * 2.0).sqrt();
                let ks = 1.0 / (kn[0] * kn[0] + kn[1] * kn[1] + 1e-6).sqrt();
                for c in 0..2 {
                    qn[c] *= qs;
                    kn[c] *= ks;
                }
                let arg = f64::from(a[head]) + f64::from(dt[head]);
                let sp = arg.max(0.0) + (-arg.abs()).exp().ln_1p();
                let decay = (f64::from(negative_a[head]) * sp).exp();
                let beta = 1.0 / (1.0 + (-f64::from(b[head])).exp());
                for channel in 0..2 {
                    let row = &mut reference_state[head * 4 + channel * 2..head * 4 + channel * 2 + 2];
                    row[0] *= decay;
                    row[1] *= decay;
                    let update = (f64::from(v[head * 2 + channel]) - row[0] * kn[0] - row[1] * kn[1]) * beta;
                    row[0] += kn[0] * update;
                    row[1] += kn[1] * update;
                    reference[head * 2 + channel] = row[0] * qn[0] + row[1] * qn[1];
                }
                let r = &reference[head * 2..head * 2 + 2];
                let inv = 1.0 / ((r[0] * r[0] + r[1] * r[1]) * 0.5 + f64::from(g.eps)).sqrt();
                for channel in 0..2 {
                    reference[head * 2 + channel] *=
                        inv * f64::from(norm[channel]) / (1.0 + (-f64::from(z[head * 2 + channel])).exp());
                }
            }
            for (tiled, &grouped) in order.iter().enumerate() {
                for channel in 0..2 {
                    assert!((f64::from(actual[tiled * 2 + channel]) - reference[grouped * 2 + channel]).abs() < 2e-5);
                }
                for state in 0..4 {
                    assert!(
                        (f64::from(native.state[tiled * 4 + state]) - reference_state[grouped * 4 + state]).abs()
                            < 2e-5
                    );
                }
            }
        }
    }

    #[stark_test]
    fn qwen_sparse_mask_keeps_only_selected_complete_blocks_and_own_tail() {
        assert_eq!(selected_tokens(&[10.0, 1.0], 10, 4, 4).unwrap(), [0, 1, 2, 3, 8, 9]);
        assert_eq!(selected_tokens(&[1.0, 10.0], 8, 4, 4).unwrap(), [4, 5, 6, 7]);
        assert!(selected_tokens(&[1.0], 10, 4, 4).is_err());
    }

    #[stark_test]
    fn qwen_recurrence_decays_before_delta_correction() {
        let mut state = [2.0, 4.0];
        let output = delta_row(&mut state, &[0.5, 0.25], &[1.0, 0.0], 5.0, 0.5, 0.25);
        assert_eq!(state, [2.0, 2.0]);
        assert_eq!(output, 1.5);
    }

    #[stark_test]
    fn qwen_depthwise_convolution_uses_channel_major_taps_and_dilation() {
        let mut state = ConvState::default();
        let weight = [2.0, 3.0, 5.0, 7.0];
        let first = state.step(&[1.0, 2.0], &weight, 2, 2).unwrap();
        assert!((first[0] - silu(3.0)).abs() < 1e-6);
        assert!((first[1] - silu(14.0)).abs() < 1e-6);
        state.step(&[10.0, 20.0], &weight, 2, 2).unwrap();
        let third = state.step(&[100.0, 200.0], &weight, 2, 2).unwrap();
        assert_eq!(third, [silu(302.0), silu(1410.0)]);
        state.clear();
        assert_eq!(state.step(&[1.0, 2.0], &weight, 2, 2).unwrap(), first);
    }

    #[stark_test]
    fn qwen_l2_epsilon_is_added_to_sum_and_rope_preserves_pass_through() {
        let mut tiny = [1e-4, 0.0];
        normalize_l2(&mut tiny, 2, 1.0);
        assert!((tiny[0] - 1e-4 / 1.01e-6_f32.sqrt()).abs() < 1e-6);
        let mut head = [1.0, 2.0, 3.0, 4.0];
        rope(&mut head, 2, 1, 10_000.0);
        assert_eq!(&head[2..], &[3.0, 4.0]);
        assert!((head[0] - (1.0_f32.cos() - 2.0 * 1.0_f32.sin())).abs() < 1e-6);
    }

    /// Uses a private complete numerical fixture, never a header-only model.
    /// Compare every visible prefix so recurrent state, sparse tails, EOS/PLE
    /// history, reset and the final output all exercise the ingest path.
    #[stark_test]
    #[ignore]
    fn qwen_prefix_ingest_preserves_full_readout_and_failure_recovery() {
        let path = std::env::var_os("STARK_QWEN38_NUMERICAL_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| panic!("private Qwen numerical fixture path is required"));
        assert!(!path.join("HEADER-ADMISSION-ONLY.json").exists());
        let read = |name| std::fs::read(path.join(name)).unwrap_or_else(|_| panic!("private Qwen fixture read failed"));
        let config: Value = serde_json::from_slice(&read("config.json"))
            .unwrap_or_else(|_| panic!("private Qwen fixture configuration is invalid"));
        let tokens: Vec<u32> = serde_json::from_slice(&read("tokens.json"))
            .unwrap_or_else(|_| panic!("private Qwen fixture sequence is invalid"));
        assert!(tokens.len() >= 2 && tokens.len() <= 4096);
        let load = || {
            let weights = Weights::open(&path).unwrap_or_else(|_| panic!("private Qwen fixture admission failed"));
            QwenModel::new(weights, &config).unwrap_or_else(|_| panic!("private Qwen fixture model admission failed"))
        };
        let mut complete = load();
        let expected: Vec<_> = tokens
            .iter()
            .map(|&token| complete.step(token).unwrap_or_else(|_| panic!("private Qwen reference step failed")))
            .collect();
        let mut prefix = load();
        for end in 0..tokens.len() {
            prefix.reset();
            for &token in &tokens[..end] {
                prefix.ingest(token).unwrap_or_else(|_| panic!("private Qwen prefix ingest failed"));
            }
            let actual = prefix.step(tokens[end]).unwrap_or_else(|_| panic!("private Qwen prefix step failed"));
            assert!(actual == expected[end], "private Qwen prefix output mismatch");
        }
        assert!(prefix.ingest(u32::MAX).is_err());
        let recovered = prefix.step(tokens[0]).unwrap_or_else(|_| panic!("private Qwen recovery step failed"));
        assert!(recovered == expected[0], "private Qwen failure recovery mismatch");
        prefix.reset();
        let reset = prefix.step(tokens[0]).unwrap_or_else(|_| panic!("private Qwen reset step failed"));
        assert!(reset == expected[0], "private Qwen reset output mismatch");
    }

    /// Keep the prior staged Metal path as an independent execution reference.
    /// Repeated fixture tokens cross the first device-cache growth boundary;
    /// failures and resets must return both paths to the first visible prefix.
    #[cfg(feature = "metal")]
    #[stark_test]
    #[ignore]
    fn qwen_decode_graph_matches_staged_forward_across_growth_and_reset() {
        let path = std::env::var_os("STARK_QWEN38_NUMERICAL_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| panic!("private Qwen numerical fixture path is required"));
        assert!(!path.join("HEADER-ADMISSION-ONLY.json").exists());
        let read = |name| std::fs::read(path.join(name)).unwrap_or_else(|_| panic!("private Qwen fixture read failed"));
        let config: Value = serde_json::from_slice(&read("config.json"))
            .unwrap_or_else(|_| panic!("private Qwen fixture configuration is invalid"));
        let tokens: Vec<u32> = serde_json::from_slice(&read("tokens.json"))
            .unwrap_or_else(|_| panic!("private Qwen fixture sequence is invalid"));
        assert!(!tokens.is_empty());
        let load = || {
            let weights = Weights::open(&path).unwrap_or_else(|_| panic!("private Qwen fixture admission failed"));
            QwenModel::new(weights, &config).unwrap_or_else(|_| panic!("private Qwen fixture model admission failed"))
        };
        let mut graph = load();
        assert!(graph.weights.indexed_banks_admitted());
        graph.graph_enabled = true;
        let mut staged = load();
        staged.graph_enabled = false;
        let count = 33.min(graph.geometry.context);
        assert!(count > 16, "private Qwen fixture must exercise graph cache growth");
        let mut first = Vec::new();
        for position in 0..count {
            let token = tokens[position % tokens.len()];
            let expected = staged.step(token).unwrap_or_else(|_| panic!("private Qwen staged step failed"));
            let actual = graph.step(token).unwrap_or_else(|_| panic!("private Qwen graph step failed"));
            assert_eq!(actual.len(), expected.len());
            let difference = actual.iter().zip(&expected).map(|(&a, &b)| (a - b).abs()).fold(0.0_f32, f32::max);
            assert!(difference <= 2e-5, "private Qwen graph differs from staged forward");
            if position == 0 {
                first = actual;
            }
        }
        assert!(graph.ingest(u32::MAX).is_err());
        assert!(graph.graph.is_none());
        let recovered = graph.step(tokens[0]).unwrap_or_else(|_| panic!("private Qwen graph recovery failed"));
        assert!(recovered == first, "private Qwen graph failure reset differs");
        graph.reset();
        assert!(graph.graph.is_none());
        let reset = graph.step(tokens[0]).unwrap_or_else(|_| panic!("private Qwen graph reset failed"));
        assert!(reset == first, "private Qwen graph explicit reset differs");
    }

    /// A synthetic complete fixture may use a longer context without changing
    /// its weights. This crosses the 255/256 scalar-to-split transition; it is
    /// component evidence and does not qualify a production checkpoint.
    #[cfg(feature = "metal")]
    #[stark_test]
    #[ignore]
    fn qwen_decode_graph_scalar_workspace_matches_staged_across_split_transition() {
        let path = std::env::var_os("STARK_QWEN38_NUMERICAL_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .expect("private Qwen numerical fixture path required");
        assert!(!path.join("HEADER-ADMISSION-ONLY.json").exists());
        let mut config: Value = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
        config["text_config"]["max_position_embeddings"] = serde_json::json!(512);
        let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(path.join("tokens.json")).unwrap()).unwrap();
        assert!(!tokens.is_empty());
        let load = || QwenModel::new(Weights::open(&path).unwrap(), &config).unwrap();
        let mut graph = load();
        assert!(graph.weights.indexed_banks_admitted());
        graph.graph_enabled = true;
        let mut staged = load();
        staged.graph_enabled = false;
        for position in 0..257 {
            let token = tokens[position % tokens.len()];
            let expected = staged.step(token).expect("private Qwen staged step failed");
            let actual = graph.step(token).expect("private Qwen graph step failed");
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual.iter().zip(expected).all(|(actual, expected)| (actual - expected).abs() <= 2e-5),
                "private Qwen attention workspace changed full readout"
            );
        }
    }

    /// The private fixture preserves complete production headers and only the
    /// original control/hash buffers. Matrix payloads are sparse holes. This
    /// validates admission and shape rejection; it must never run a forward.
    #[stark_test]
    #[ignore]
    fn qwen_production_header_admission_only() {
        let path = std::env::var_os("STARK_QWEN38_HEADER_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| panic!("Qwen production header fixture path is required"));
        let read = |name: &str| -> Value {
            let bytes =
                std::fs::read(path.join(name)).unwrap_or_else(|_| panic!("private Qwen header metadata read failed"));
            serde_json::from_slice(&bytes).unwrap_or_else(|_| panic!("private Qwen header metadata is invalid"))
        };
        let receipt = read("HEADER-ADMISSION-ONLY.json");
        assert!(receipt["header_only"] == true && receipt["forward_executed"] == false);
        assert!(receipt["source_revision"].as_str() == Some("b2c422f3c643e36f04227a64d61796b44a4b1029"));
        let config = read("config.json");
        let weights = Weights::open(&path).unwrap_or_else(|_| panic!("production Qwen header admission failed"));
        assert_eq!(weights.shape("vision_tower.patch_embed.proj.weight").unwrap(), &[1152, 3, 2, 16, 16]);
        let ngram = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
        assert_eq!(weights.shape(ngram).unwrap(), &[2_500_012, 160]);
        let model = QwenModel::new(weights, &config)
            .unwrap_or_else(|_| panic!("production Qwen model configuration admission failed"));
        assert_eq!(model.layers.len(), 48);
        assert_eq!(model.geometry.vocab, 248_320);
        assert_eq!(model.geometry.dim, 2560);
        assert_eq!(model.geometry.streams, 4);
        assert_eq!(model.geometry.context, 262_144);
        drop(model);
        for (name, value) in [("linear_num_value_heads", 47), ("ple_embed_dim", 1280), ("hc_count", 3)] {
            let mut changed = config.clone();
            changed["text_config"][name] = Value::from(value);
            let weights = Weights::open(&path).unwrap_or_else(|_| panic!("production Qwen header reopening failed"));
            assert!(QwenModel::new(weights, &changed).is_err(), "production Qwen tensor disagreement was accepted");
        }
        println!(
            "{}",
            serde_json::json!({
                "qwen_production_header_admission": true,
                "tensor_disagreements_rejected": 3,
                "shards": receipt["shards"],
                "tensors": receipt["tensors"],
                "forward_executed": false,
            })
        );
    }
}
