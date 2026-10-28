//! Persistent F32 GLM KDA decode state. No weight expansion or request caching.
//! Original Stark implementation of the scalar contract in `glm.rs`.

use std::{collections::HashMap, ptr::NonNull, sync::OnceLock};

use metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLLibraryExt,
    MTLResourceOptions, MTLSize,
};
#[cfg(test)]
use metal::{MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue};
use objc2::{rc::Retained, runtime::ProtocolObject};

#[cfg(test)]
use super::profile;
use super::{
    Error, Result,
    weight_metal::{DecodeGraph, DeviceVector},
};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(test)]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    graph_pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

static EXECUTOR: OnceLock<std::result::Result<Executor, &'static str>> = OnceLock::new();

fn executor() -> Result<&'static Executor> {
    EXECUTOR
        .get_or_init(|| {
            let device = <dyn MTLDevice>::system_default().ok_or("Metal device unavailable")?;
            #[cfg(test)]
            let queue = device.new_command_queue().ok_or("Metal KDA queue unavailable")?;
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = device
                .new_library_with_source(include_str!("glm_metal.metal"), Some(&options))
                .map_err(|_| "Metal KDA kernel compilation failed")?;
            let function = library.new_function_with_name("glm_kda_decode").ok_or("Metal KDA function unavailable")?;
            let pipeline =
                device.new_compute_pipeline_state_with_function(&function).map_err(|_| "Metal KDA pipeline failed")?;
            if pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("Metal KDA execution width is unsupported");
            }
            let mut graph_pipelines = HashMap::new();
            for name in [
                "glm_control_project",
                "glm_control_project_unprefetched",
                "glm_control_project_scalar",
                "glm_control_mix",
                "glm_control_mix_scalar",
                "glm_control_input",
                "glm_control_post",
                "glm_index_pool",
                "glm_index_scores",
                "glm_index_sort",
                "glm_index_mask",
                "glm_mla_scores",
                "glm_mla_softmax",
                "glm_mla_values",
                "glm_index_layer_norm",
                "glm_stream_mean",
                "glm_kda_graph_status",
            ] {
                let function = library.new_function_with_name(name).ok_or("Metal GLM graph function unavailable")?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| "Metal GLM graph pipeline failed")?;
                if pipeline.max_total_threads_per_threadgroup() < 256 {
                    return Err("Metal GLM graph execution width is unsupported");
                }
                if name == "glm_control_project" && pipeline.thread_execution_width() != 32 {
                    return Err("Metal GLM control SIMD width is unsupported");
                }
                graph_pipelines.insert(name, pipeline);
            }
            Ok(Executor {
                device,
                #[cfg(test)]
                queue,
                pipeline,
                graph_pipelines,
            })
        })
        .as_ref()
        .map_err(|message| Error::Unsupported(message))
}

pub(super) struct Kda {
    parameters: [u32; 6],
    input: Buffer,
    weights: Buffer,
    state: Buffer,
    output: Buffer,
    failure: Buffer,
    inner: usize,
    heads: usize,
    state_values: usize,
}

pub(super) struct Geometry {
    pub heads: usize,
    pub dim: usize,
    pub kernel: usize,
    pub rms_eps: f32,
    pub lower_bound: f32,
}

impl Kda {
    pub(super) fn new(
        geometry: Geometry,
        convolutions: [&[f32]; 3],
        dt: &[f32],
        norm: &[f32],
        a: &[f32],
    ) -> Result<Self> {
        let Geometry {
            heads,
            dim,
            kernel,
            rms_eps,
            lower_bound,
        } = geometry;
        let inner = heads.checked_mul(dim).ok_or(Error::Invalid("Metal KDA dimensions"))?;
        if heads == 0
            || heads > 256
            || dim == 0
            || dim > 256
            || !(2..=32).contains(&kernel)
            || !rms_eps.is_finite()
            || rms_eps <= 0.0
            || !lower_bound.is_finite()
            || lower_bound >= 0.0
            || convolutions.iter().any(|values| values.len() != inner * kernel)
            || dt.len() != inner
            || norm.len() != dim
            || a.len() != heads
        {
            return Err(Error::Invalid("Metal KDA dimensions"));
        }
        let all_weights: Vec<_> = convolutions.into_iter().flatten().chain(dt).chain(norm).chain(a).copied().collect();
        if all_weights.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("Metal KDA weights"));
        }
        let e = executor()?;
        let allocate = |length| {
            e.device
                .new_buffer(length, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal KDA allocation failed"))
        };
        let state_values = 3 * inner * (kernel - 1) + inner * dim;
        let state = allocate(state_values * 4)?;
        let weights = e
            .device
            .new_buffer_with_data(bytemuck::cast_slice(&all_weights), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(Error::Invalid("Metal KDA allocation failed"))?;
        let mut result = Self {
            parameters: [
                heads as u32,
                dim as u32,
                kernel as u32,
                inner as u32,
                rms_eps.to_bits(),
                lower_bound.to_bits(),
            ],
            input: allocate((5 * inner + heads) * 4)?,
            weights,
            state,
            output: allocate(inner * 4)?,
            failure: allocate(4)?,
            inner,
            heads,
            state_values,
        };
        result.reset();
        Ok(result)
    }

    pub(super) fn reset(&mut self) {
        // Every submitted KDA command is awaited before returning to its owner.
        // The model serializes decisions, so reset has exclusive buffer access.
        unsafe { self.state.contents().as_ptr().cast::<f32>().write_bytes(0, self.state_values) };
        unsafe { self.failure.contents().as_ptr().cast::<u32>().write(0) };
    }

    pub(super) fn inner(&self) -> usize {
        self.inner
    }
    pub(super) fn heads(&self) -> usize {
        self.heads
    }
    pub(super) fn dim(&self) -> usize {
        self.parameters[1] as usize
    }
    pub(super) fn input_buffer(&self) -> &Buffer {
        &self.input
    }
    pub(super) fn output_buffer(&self) -> &Buffer {
        &self.output
    }

    pub(super) fn encode_graph(
        &mut self,
        graph: &mut DecodeGraph,
    ) -> Result<()> {
        self.encode(graph.encoder())?;
        let status = DeviceVector {
            buffer: self.failure.clone(),
            offset: 0,
            len: 1,
        };
        dispatch(graph, "glm_kda_graph_status", &[&status], &[0], 1, false)
    }

    /// Append one state update to the caller's command. All projections must
    /// already be encoded into input_buffer in q/k/v/forget/gate/beta order.
    pub(super) fn encode(
        &mut self,
        encoding: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) -> Result<()> {
        // The model serializes steps and awaits each command before reusing
        // these buffers. This status is initialized before this command runs.
        unsafe { self.failure.contents().as_ptr().cast::<u32>().write(0) };
        encoding.set_compute_pipeline_state(&executor()?.pipeline);
        for (index, buffer) in [&self.input, &self.weights, &self.state, &self.output].into_iter().enumerate() {
            encoding.set_buffer(Some(buffer), 0, index);
        }
        encoding.set_bytes(NonNull::from(&self.parameters).cast(), std::mem::size_of_val(&self.parameters), 4);
        encoding.set_buffer(Some(&self.failure), 0, 5);
        encoding.dispatch_threadgroups(
            MTLSize {
                width: self.heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.dim(),
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Called only after the command containing encode has completed.
    pub(super) fn check_completed(&self) -> Result<()> {
        // GPU writes are visible after completion of the owning command.
        if unsafe { self.failure.contents().as_ptr().cast::<u32>().read() } != 0 {
            return Err(Error::Invalid("nonfinite Metal KDA intermediate"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn step(
        &mut self,
        projections: [&[f32]; 5],
        beta: &[f32],
    ) -> Result<Vec<f32>> {
        if projections.iter().any(|v| v.len() != self.inner)
            || beta.len() != self.heads
            || projections.iter().copied().flatten().chain(beta).any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("Metal KDA projection dimensions or values"));
        }
        let _scope = profile::Scope::new("glm.kda.metal");
        for (part, values) in projections.into_iter().chain(std::iter::once(beta)).enumerate() {
            // Checked disjoint input spans are fully initialized before submit.
            unsafe {
                self.input
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(part * self.inner)
                    .copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
        }
        objc2::rc::autoreleasepool(|_| {
            let e = executor()?;
            let command = e.queue.command_buffer().ok_or(Error::Invalid("Metal KDA command allocation failed"))?;
            let encoding =
                command.compute_command_encoder().ok_or(Error::Invalid("Metal KDA encoder allocation failed"))?;
            self.encode(&encoding)?;
            encoding.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Invalid("Metal KDA execution failed"));
            }
            self.check_completed()?;
            profile::record("glm.kda.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
            // Completion makes the initialized output visible to the CPU.
            let values =
                unsafe { std::slice::from_raw_parts(self.output.contents().as_ptr().cast::<f32>(), self.inner) }
                    .to_vec();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("nonfinite Metal KDA output"));
            }
            Ok(values)
        })
    }
}

fn dispatch(
    graph: &DecodeGraph,
    name: &'static str,
    buffers: &[&DeviceVector],
    parameters: &[u32],
    count: usize,
    one_group: bool,
) -> Result<()> {
    dispatch_with_threads(graph, name, buffers, parameters, count, one_group, 256)
}

fn dispatch_with_threads(
    graph: &DecodeGraph,
    name: &'static str,
    buffers: &[&DeviceVector],
    parameters: &[u32],
    count: usize,
    one_group: bool,
    threads: usize,
) -> Result<()> {
    if count == 0 || parameters.is_empty() {
        return Err(Error::Invalid("Metal GLM graph dispatch geometry"));
    }
    let encoding = graph.encoder();
    encoding.set_compute_pipeline_state(
        executor()?.graph_pipelines.get(name).ok_or(Error::Invalid("Metal GLM graph pipeline"))?,
    );
    for (index, value) in buffers.iter().enumerate() {
        encoding.set_buffer(Some(&value.buffer), value.offset, index);
    }
    encoding.set_bytes(
        NonNull::new(parameters.as_ptr().cast_mut().cast()).ok_or(Error::Invalid("Metal GLM graph parameters"))?,
        std::mem::size_of_val(parameters),
        buffers.len(),
    );
    encoding.set_buffer(Some(&graph.failure().buffer), graph.failure().offset, buffers.len() + 1);
    encoding.dispatch_threadgroups(
        MTLSize {
            width: if one_group {
                1
            } else {
                count.div_ceil(threads)
            },
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Immutable mHC controls and per-layer scratch. The caller owns the command
/// and keeps residual/output vectors on-device across both sublayer passes.
pub(super) struct Control {
    weights: DeviceVector,
    coefficients: DeviceVector,
    ones: DeviceVector,
    normalization: DeviceVector,
    projection: DeviceVector,
    mix: DeviceVector,
    input: DeviceVector,
    parameters: [u32; 5],
    rms_eps: f32,
}

pub(super) struct ControlGeometry {
    pub hidden: usize,
    pub streams: usize,
    pub iterations: usize,
    pub eps: f32,
    pub rms_eps: f32,
}

impl Control {
    pub(super) fn new(
        geometry: ControlGeometry,
        weights: &[f32],
        base: &[f32],
        scale: &[f32],
        normalization: &[f32],
    ) -> Result<Self> {
        let ControlGeometry {
            hidden,
            streams,
            iterations,
            eps,
            rms_eps,
        } = geometry;
        let columns = hidden.checked_mul(streams).ok_or(Error::Invalid("Metal GLM control dimensions"))?;
        let rows = (2 + streams) * streams;
        if hidden == 0
            || streams == 0
            || streams > 8
            || columns > 262_144
            || iterations == 0
            || iterations > 128
            || !eps.is_finite()
            || eps <= 0.0
            || !rms_eps.is_finite()
            || rms_eps <= 0.0
            || weights.len() != rows * columns
            || base.len() != rows
            || scale.len() != 3
            || normalization.len() != hidden
            || weights.iter().chain(base).chain(scale).chain(normalization).any(|value| !value.is_finite())
        {
            return Err(Error::Invalid("Metal GLM control dimensions or weights"));
        }
        let mut coefficients = base.to_vec();
        coefficients.extend(scale);
        Ok(Self {
            weights: DecodeGraph::upload(weights)?,
            coefficients: DecodeGraph::upload(&coefficients)?,
            ones: DecodeGraph::upload(&vec![1.0; columns])?,
            normalization: DecodeGraph::upload(normalization)?,
            projection: DecodeGraph::zeros(rows)?,
            mix: DecodeGraph::zeros(rows)?,
            input: DecodeGraph::zeros(hidden)?,
            parameters: [hidden as u32, streams as u32, iterations as u32, columns as u32, eps.to_bits()],
            rms_eps,
        })
    }

    pub(super) fn pre(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        if residual.len != self.parameters[3] as usize {
            return Err(Error::Invalid("Metal GLM control residual dimensions"));
        }
        let normalized = graph.rms_norm(residual, &self.ones, residual.len, self.rms_eps)?;
        dispatch_with_threads(
            graph,
            "glm_control_project",
            &[&normalized, &self.weights, &self.projection],
            &self.parameters,
            self.projection.len * 32,
            false,
            32,
        )?;
        dispatch_with_threads(
            graph,
            "glm_control_mix",
            &[&self.projection, &self.coefficients, &self.mix],
            &self.parameters,
            64,
            false,
            64,
        )?;
        dispatch(
            graph,
            "glm_control_input",
            &[residual, &self.mix, &self.input],
            &self.parameters,
            self.input.len,
            false,
        )?;
        graph.rms_norm(&self.input, &self.normalization, self.input.len, self.rms_eps)
    }

    pub(super) fn post(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        output: &DeviceVector,
    ) -> Result<DeviceVector> {
        if residual.len != self.parameters[3] as usize || output.len != self.input.len {
            return Err(Error::Invalid("Metal GLM control output dimensions"));
        }
        let next = graph.scratch(residual.len)?;
        dispatch(graph, "glm_control_post", &[residual, output, &self.mix, &next], &self.parameters, next.len, false)?;
        Ok(next)
    }
}

pub(super) fn stream_mean(
    graph: &mut DecodeGraph,
    streams: &DeviceVector,
    hidden: usize,
    count: usize,
) -> Result<DeviceVector> {
    if hidden == 0 || count == 0 || hidden.checked_mul(count) != Some(streams.len) {
        return Err(Error::Invalid("Metal GLM stream mean dimensions"));
    }
    let output = graph.scratch(hidden)?;
    dispatch(
        graph,
        "glm_stream_mean",
        &[streams, &output],
        &[hidden as u32, count as u32, 0, streams.len as u32, 0],
        hidden,
        false,
    )?;
    Ok(output)
}

pub(super) fn index_layer_norm(
    graph: &mut DecodeGraph,
    input: &DeviceVector,
    weight: &DeviceVector,
    bias: &DeviceVector,
) -> Result<DeviceVector> {
    if input.len == 0 || input.len != weight.len || input.len != bias.len {
        return Err(Error::Invalid("Metal GLM index norm dimensions"));
    }
    let output = graph.scratch(input.len)?;
    dispatch(graph, "glm_index_layer_norm", &[input, weight, bias, &output], &[input.len as u32], 256, true)?;
    Ok(output)
}

pub(super) struct AttentionGeometry {
    pub rank: usize,
    pub heads: usize,
    pub query_dim: usize,
    pub index_dim: usize,
    pub index_heads: usize,
    pub pool: usize,
    pub top_k: usize,
    pub context: usize,
}

/// Request-scoped MLA/indexer cache. Capacity grows geometrically; reset clears
/// the visible prefix and no subsequent dispatch can read the old request.
pub(super) struct Attention {
    geometry: AttentionGeometry,
    latents: DeviceVector,
    pool_keys: DeviceVector,
    tail_keys: DeviceVector,
    tail_gates: DeviceVector,
    ape: DeviceVector,
    position: usize,
    capacity: usize,
}

impl Attention {
    #[cfg(test)]
    pub(super) fn test_pool_keys(&self) -> DeviceVector {
        self.pool_keys.clone()
    }

    pub(super) fn new(
        geometry: AttentionGeometry,
        ape: &[f32],
    ) -> Result<Self> {
        let g = &geometry;
        if g.rank == 0
            || g.rank > 8192
            || g.heads == 0
            || g.heads > 256
            || g.query_dim == 0
            || g.query_dim > 512
            || g.index_dim == 0
            || g.index_dim > 512
            || g.index_heads == 0
            || g.index_heads > 128
            || !(2..=64).contains(&g.pool)
            || g.top_k == 0
            || !g.top_k.is_multiple_of(g.pool)
            || g.context == 0
            || g.context > 1_048_576
            || ape.len() != g.pool * g.index_dim
            || ape.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("Metal GLM attention geometry"));
        }
        let capacity = g.context.min(64);
        Ok(Self {
            latents: DecodeGraph::zeros(capacity * g.rank)?,
            pool_keys: DecodeGraph::zeros(capacity.div_ceil(g.pool) * g.index_dim)?,
            tail_keys: DecodeGraph::zeros(g.pool * g.index_dim)?,
            tail_gates: DecodeGraph::zeros(g.pool * g.index_dim)?,
            ape: DecodeGraph::upload(ape)?,
            geometry,
            position: 0,
            capacity,
        })
    }

    pub(super) fn reset(&mut self) {
        self.position = 0;
    }

    fn parameters(&self) -> [u32; 12] {
        let g = &self.geometry;
        let pools = self.position / g.pool;
        [
            self.position as u32,
            g.rank as u32,
            g.heads as u32,
            g.index_dim as u32,
            g.index_heads as u32,
            g.pool as u32,
            pools as u32,
            pools.min(g.top_k / g.pool) as u32,
            pools.max(1).next_power_of_two() as u32,
            0,
            0,
            (1.0 / (g.query_dim as f32).sqrt()).to_bits(),
        ]
    }

    pub(super) fn append(
        &mut self,
        graph: &mut DecodeGraph,
        latent: &DeviceVector,
        index_key: &DeviceVector,
        index_gate: &DeviceVector,
    ) -> Result<()> {
        let g = &self.geometry;
        if self.position >= g.context
            || latent.len != g.rank
            || index_key.len != g.index_dim
            || index_gate.len != g.index_dim
        {
            return Err(Error::Invalid("Metal GLM attention input dimensions"));
        }
        if self.position == self.capacity {
            let capacity = (self.capacity * 2).min(g.context);
            let latents = DecodeGraph::zeros(capacity * g.rank)?;
            graph.copy_into(&self.latents, &latents.slice(0, self.latents.len)?)?;
            let pools = DecodeGraph::zeros(capacity.div_ceil(g.pool) * g.index_dim)?;
            graph.copy_into(&self.pool_keys, &pools.slice(0, self.pool_keys.len)?)?;
            self.latents = latents;
            self.pool_keys = pools;
            self.capacity = capacity;
        }
        graph.copy_into(latent, &self.latents.slice(self.position * g.rank, g.rank)?)?;
        let tail = self.position % g.pool;
        graph.copy_into(index_key, &self.tail_keys.slice(tail * g.index_dim, g.index_dim)?)?;
        graph.copy_into(index_gate, &self.tail_gates.slice(tail * g.index_dim, g.index_dim)?)?;
        self.position += 1;
        if self.position.is_multiple_of(g.pool) {
            let pooled = self.pool_keys.slice((self.position / g.pool - 1) * g.index_dim, g.index_dim)?;
            dispatch(
                graph,
                "glm_index_pool",
                &[&self.tail_keys, &self.tail_gates, &self.ape, &pooled],
                &self.parameters(),
                g.index_dim,
                false,
            )?;
        }
        Ok(())
    }

    pub(super) fn attend(
        &self,
        graph: &mut DecodeGraph,
        queries: &DeviceVector,
        index_queries: &DeviceVector,
        index_weights: &DeviceVector,
    ) -> Result<DeviceVector> {
        let g = &self.geometry;
        if self.position == 0
            || queries.len != g.heads * g.rank
            || index_queries.len != g.index_heads * g.index_dim
            || index_weights.len != g.index_heads
        {
            return Err(Error::Invalid("Metal GLM attention query dimensions"));
        }
        let mut p = self.parameters();
        let mask = DecodeGraph::zeros((p[6] as usize).max(1))?;
        if p[7] < p[6] {
            let scores = graph.scratch(p[8] as usize)?;
            let indices = graph.scratch(p[8] as usize)?;
            dispatch(
                graph,
                "glm_index_scores",
                &[index_queries, index_weights, &self.pool_keys, &scores, &indices],
                &p,
                p[8] as usize,
                false,
            )?;
            let mut span = 2;
            while span <= p[8] {
                p[9] = span;
                let mut stride = span / 2;
                while stride > 0 {
                    p[10] = stride;
                    dispatch(graph, "glm_index_sort", &[&scores, &indices], &p, p[8] as usize, false)?;
                    stride /= 2;
                }
                span *= 2;
            }
            dispatch(graph, "glm_index_mask", &[&indices, &mask], &p, p[7] as usize, false)?;
        }
        let scores = if g.heads * self.position <= 8_000_000 {
            graph.scratch(g.heads * self.position)?
        } else {
            DecodeGraph::zeros(g.heads * self.position)?
        };
        let output = graph.scratch(g.heads * g.rank)?;
        dispatch(graph, "glm_mla_scores", &[queries, &self.latents, &mask, &scores], &p, scores.len, false)?;
        dispatch(graph, "glm_mla_softmax", &[&scores], &p, g.heads, false)?;
        dispatch(graph, "glm_mla_values", &[&scores, &self.latents, &mask, &output], &p, output.len, false)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn graph_parallel_sinkhorn_preserves_original_normalization_bits() {
        for streams in [2, 4, 8] {
            for iterations in [1, 2, 20, 128] {
                let rows = (2 + streams) * streams;
                let generate = |len: usize, offset: usize| -> Vec<f32> {
                    (0..len).map(|i| (((i * 17 + offset * 7) % 47) as f32 - 23.0) * 0.11).collect()
                };
                let mut graph = DecodeGraph::new().unwrap();
                let projection = DecodeGraph::upload(&generate(rows, 1)).unwrap();
                let mut coefficients = generate(rows, 2);
                coefficients.extend([0.5, -0.2, 0.4]);
                let coefficients = DecodeGraph::upload(&coefficients).unwrap();
                let scalar = graph.scratch(rows).unwrap();
                let parallel = graph.scratch(rows).unwrap();
                let parameters = [19, streams as u32, iterations, 19 * streams as u32, 1e-6f32.to_bits()];
                dispatch(
                    &graph,
                    "glm_control_mix_scalar",
                    &[&projection, &coefficients, &scalar],
                    &parameters,
                    1,
                    false,
                )
                .unwrap();
                dispatch_with_threads(
                    &graph,
                    "glm_control_mix",
                    &[&projection, &coefficients, &parallel],
                    &parameters,
                    64,
                    false,
                    64,
                )
                .unwrap();
                let results = graph.finish(&[scalar, parallel]).unwrap();
                assert!(
                    results[0].iter().zip(&results[1]).all(|(&a, &b)| a.to_bits() == b.to_bits()),
                    "parallel Sinkhorn changed its chronological normalization"
                );
            }
        }
    }

    /// Synthetic component timing, excluded from the normal validation suite.
    /// The only emitted data is geometry and aggregate GPU/host duration.
    #[stark_test]
    #[ignore]
    fn glm_control_graph_component_experiment() {
        use std::time::Instant;
        let hidden = 4096;
        let streams = 4;
        let columns = hidden * streams;
        let rows = (2 + streams) * streams;
        let count = 10;
        let controls: Vec<_> = (0..count)
            .map(|part| {
                let weights: Vec<_> =
                    (0..rows * columns).map(|i| (((i * 13 + part * 17) % 127) as f32 - 63.0) * 0.0001).collect();
                Control::new(
                    ControlGeometry {
                        hidden,
                        streams,
                        iterations: 20,
                        eps: 1e-6,
                        rms_eps: 1e-5,
                    },
                    &weights,
                    &vec![0.0; rows],
                    &[0.01, 0.01, 0.01],
                    &vec![1.0; hidden],
                )
                .unwrap()
            })
            .collect();
        let values: Vec<_> = (0..columns).map(|i| ((i % 127) as f32 - 63.0) * 0.001).collect();
        let input = DecodeGraph::upload(&values).unwrap();
        let branch = DecodeGraph::upload(&vec![0.001; hidden]).unwrap();
        let mut measurements = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        profile::enable(true);
        for replay in 0..4 {
            for (case, times) in measurements.iter_mut().enumerate() {
                profile::reset();
                let started = Instant::now();
                let mut graph = DecodeGraph::new().unwrap();
                let mut residual = input.clone();
                for control in &controls {
                    match case {
                        0 => {
                            let _normalized = control.pre(&mut graph, &residual).unwrap();
                            residual = control.post(&mut graph, &residual, &branch).unwrap();
                        },
                        1 => {
                            let normalized = graph.rms_norm(&input, &control.ones, columns, 1e-5).unwrap();
                            let _output = graph
                                .rms_norm(&normalized.slice(0, hidden).unwrap(), &control.normalization, hidden, 1e-5)
                                .unwrap();
                        },
                        2 => dispatch_with_threads(
                            &graph,
                            "glm_control_project",
                            &[&input, &control.weights, &control.projection],
                            &control.parameters,
                            rows * 32,
                            false,
                            32,
                        )
                        .unwrap(),
                        3 => dispatch_with_threads(
                            &graph,
                            "glm_control_project_unprefetched",
                            &[&input, &control.weights, &control.projection],
                            &control.parameters,
                            rows * 32,
                            false,
                            32,
                        )
                        .unwrap(),
                        4 => dispatch_with_threads(
                            &graph,
                            "glm_control_mix",
                            &[&control.projection, &control.coefficients, &control.mix],
                            &control.parameters,
                            64,
                            false,
                            64,
                        )
                        .unwrap(),
                        5 => dispatch(
                            &graph,
                            "glm_control_mix_scalar",
                            &[&control.projection, &control.coefficients, &control.mix],
                            &control.parameters,
                            1,
                            false,
                        )
                        .unwrap(),
                        _ => unreachable!(),
                    }
                }
                graph.finish(&[]).unwrap();
                let wall_seconds = started.elapsed().as_secs_f64();
                let gpu_seconds = profile::snapshot().get("native.metal.gpu").unwrap().seconds;
                if replay > 0 {
                    times.push((gpu_seconds, wall_seconds));
                }
            }
        }
        profile::enable(false);
        let cases: Vec<_> = [
            "control_pre_post",
            "two_configured_rms",
            "control_projection",
            "control_projection_unprefetched",
            "control_mix",
            "control_mix_scalar",
        ]
        .iter()
        .zip(measurements)
        .map(|(name, values)| {
            let mut gpu: Vec<_> = values.iter().map(|v| v.0).collect();
            let mut wall: Vec<_> = values.iter().map(|v| v.1).collect();
            gpu.sort_by(f64::total_cmp);
            wall.sort_by(f64::total_cmp);
            serde_json::json!({"case":name,"gpu_seconds":gpu,"wall_seconds":wall,
                    "gpu_median_seconds":gpu[1],"wall_median_seconds":wall[1]})
        })
        .collect();
        println!(
            "{}",
            serde_json::json!({"glm_synthetic_control_experiment":true,"hidden":hidden,
            "streams":streams,"controls_per_command":count,"measured_replays":3,
            "ordered_rms_requested":std::env::var_os("STARK_HYBRID_ORDERED_RMS").is_some_and(|v| v == "1"),"cases":cases})
        );
    }

    #[stark_test]
    fn graph_index_overflow_sanitizes_padding_and_next_command_recovers() {
        let mut parameters = [0u32; 12];
        parameters[3] = 1;
        parameters[4] = 1;
        parameters[6] = 3;
        parameters[7] = 1;
        parameters[8] = 4;
        let read = |value: &DeviceVector, index: usize| unsafe {
            value.buffer.contents().as_ptr().cast::<u8>().add(value.offset).cast::<u32>().add(index).read()
        };
        for overflow in [true, false] {
            let mut graph = DecodeGraph::new().unwrap();
            let q = DecodeGraph::upload(&[if overflow {
                f32::MAX
            } else {
                1.0
            }])
            .unwrap();
            let weights = DecodeGraph::upload(&[if overflow {
                0.0
            } else {
                1.0
            }])
            .unwrap();
            let keys = DecodeGraph::upload(
                &[if overflow {
                    f32::MAX
                } else {
                    1.0
                }; 3],
            )
            .unwrap();
            let scores = graph.scratch(4).unwrap();
            let indices = graph.scratch(4).unwrap();
            let allocation = DecodeGraph::upload(&[0.0, 0.0, 0.0, 7.0]).unwrap();
            let mask = allocation.slice(0, 3).unwrap();
            dispatch(&graph, "glm_index_scores", &[&q, &weights, &keys, &scores, &indices], &parameters, 4, false)
                .unwrap();
            for span in [2, 4] {
                parameters[9] = span;
                let mut stride = span / 2;
                while stride > 0 {
                    parameters[10] = stride;
                    dispatch(&graph, "glm_index_sort", &[&scores, &indices], &parameters, 4, false).unwrap();
                    stride /= 2;
                }
            }
            dispatch(&graph, "glm_index_mask", &[&indices, &mask], &parameters, 1, false).unwrap();
            let result = graph.finish(&[]);
            assert!(result.is_err() == overflow, "index score failure status or fresh-command recovery changed");
            assert!(read(&allocation, 3) == 7.0f32.to_bits(), "index score padding overwrote the mask canary");
            assert!(read(&indices, 0) == 0, "index score ties selected padding before an ordinary pool");
            if overflow {
                assert!(
                    (0..4).all(|i| read(&scores, i) == f32::NEG_INFINITY.to_bits()),
                    "failed index scores were not sanitized"
                );
            }
        }

        // An independently malformed/padded index must also be rejected by
        // the final scatter, before any out-of-bounds access can occur.
        let graph = DecodeGraph::new().unwrap();
        let indices = DecodeGraph::zeros(1).unwrap();
        unsafe {
            indices.buffer.contents().as_ptr().cast::<u32>().write(3);
        }
        let allocation = DecodeGraph::upload(&[0.0, 0.0, 0.0, 7.0]).unwrap();
        let mask = allocation.slice(0, 3).unwrap();
        dispatch(&graph, "glm_index_mask", &[&indices, &mask], &parameters, 1, false).unwrap();
        assert!(graph.finish(&[]).is_err(), "padded index was not rejected");
        assert!(read(&allocation, 3) == 7.0f32.to_bits(), "padded index overwrote the mask canary");
    }

    #[stark_test]
    fn graph_control_simd_preserves_original_projection_accumulator_order() {
        for (hidden, streams) in [(19, 2), (19, 4), (19, 8), (3072, 4), (4096, 4)] {
            let columns = hidden * streams;
            let rows = (2 + streams) * streams;
            let generate = |len: usize, offset: usize| -> Vec<f32> {
                (0..len).map(|i| (((i * 17 + offset * 7) % 47) as f32 - 23.0) * 0.011).collect()
            };
            let mut graph = DecodeGraph::new().unwrap();
            let input = DecodeGraph::upload(&generate(columns, 1)).unwrap();
            let weights = DecodeGraph::upload(&generate(rows * columns, 2)).unwrap();
            let scalar = graph.scratch(rows).unwrap();
            let simd = graph.scratch(rows).unwrap();
            let parameters = [hidden as u32, streams as u32, 20, columns as u32, 1e-6f32.to_bits()];
            dispatch(&graph, "glm_control_project_scalar", &[&input, &weights, &scalar], &parameters, rows, false)
                .unwrap();
            dispatch_with_threads(
                &graph,
                "glm_control_project",
                &[&input, &weights, &simd],
                &parameters,
                rows * 32,
                false,
                32,
            )
            .unwrap();
            let outputs = graph.finish(&[scalar, simd]).unwrap();
            assert!(
                outputs[0].iter().zip(&outputs[1]).all(|(&a, &b)| a.to_bits() == b.to_bits()),
                "SIMD control projection changed its accumulator order"
            );
        }
    }

    #[stark_test]
    fn graph_index_sort_preserves_adjacent_floats_signed_zero_and_exact_ties() {
        let values = [
            1.0f32,
            f32::from_bits(1.0f32.to_bits() + 1),
            0.0,
            -0.0,
            -0.5,
            1.0,
            f32::from_bits(1.0f32.to_bits() - 1),
            2.0,
        ];
        let mut expected: Vec<_> = (0..values.len()).collect();
        expected.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
        let graph = DecodeGraph::new().unwrap();
        let scores = DecodeGraph::upload(&values).unwrap();
        let indices = DecodeGraph::zeros(values.len()).unwrap();
        // The new owned buffer has no submitted writers or readers.
        for i in 0..values.len() {
            unsafe {
                indices.buffer.contents().as_ptr().cast::<u32>().add(i).write(i as u32);
            }
        }
        let mut parameters = [0u32; 12];
        parameters[8] = values.len() as u32;
        for span in [2, 4, 8] {
            parameters[9] = span;
            let mut stride = span / 2;
            while stride > 0 {
                parameters[10] = stride;
                dispatch(&graph, "glm_index_sort", &[&scores, &indices], &parameters, values.len(), false).unwrap();
                stride /= 2;
            }
        }
        let sorted = graph.finish(&[scores]).unwrap();
        // The complete owning command has finished writing these indices.
        let actual =
            unsafe { std::slice::from_raw_parts(indices.buffer.contents().as_ptr().cast::<u32>(), values.len()) };
        assert!(actual.iter().map(|&i| i as usize).eq(expected.iter().copied()), "GPU index ordering changed");
        assert!(
            sorted[0].iter().zip(expected).all(|(&a, i)| a.to_bits() == values[i].to_bits()),
            "GPU index scores changed"
        );
    }

    #[stark_test]
    fn gpu_kda_rejects_nonfinite_controls_that_gates_could_mask() {
        let mut state = Kda::new(
            Geometry {
                heads: 1,
                dim: 4,
                kernel: 2,
                rms_eps: 1e-5,
                lower_bound: -5.0,
            },
            [&[0.0; 8]; 3],
            &[0.0; 4],
            &[1.0; 4],
            &[-1.0],
        )
        .unwrap();
        for index in [3 * state.inner(), 4 * state.inner(), 5 * state.inner()] {
            state.reset();
            let mut input = vec![0.0; 5 * state.inner() + state.heads()];
            input[index] = f32::INFINITY;
            // Simulate a projection written by an earlier GPU dispatch. The
            // zero recurrent value would otherwise conceal these controls.
            unsafe {
                state.input.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(input.as_ptr(), input.len());
            }
            objc2::rc::autoreleasepool(|_| {
                let command = executor().unwrap().queue.command_buffer().unwrap();
                let encoding = command.compute_command_encoder().unwrap();
                state.encode(&encoding).unwrap();
                encoding.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(state.check_completed().is_err());
            });
        }
        state.reset();
        assert!(state.check_completed().is_ok());
        assert_eq!(state.step([&[0.0; 4]; 5], &[0.0]).unwrap(), [0.0; 4]);
    }
}
