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

#[path = "glm_control_prefill.rs"]
mod control_prefill;
#[path = "glm_kda_prefill.rs"]
mod kda_prefill;

#[cfg(test)]
#[path = "glm_prefix_metal_tests.rs"]
mod prefix_tests;

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(test)]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transposed_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    parallel_norm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
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
            let source = [
                include_str!("glm_metal.metal"),
                include_str!("glm_control_split.metal"),
                include_str!("glm_control_fused_parallel.metal"),
                include_str!("glm_control_prefill.metal"),
            ]
            .concat();
            let library = device
                .new_library_with_source(&source, Some(&options))
                .map_err(|_| "Metal KDA kernel compilation failed")?;
            let function = library.new_function_with_name("glm_kda_decode").ok_or("Metal KDA function unavailable")?;
            let pipeline =
                device.new_compute_pipeline_state_with_function(&function).map_err(|_| "Metal KDA pipeline failed")?;
            if pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("Metal KDA execution width is unsupported");
            }
            let fused = usize::from(std::env::var_os("STARK_GLM53_KDA_FUSED_STATE").is_some_and(|value| value == "1"));
            let transposed_source =
                format!("#define STARK_GLM_KDA_TRANSPOSED 1\n#define STARK_GLM_KDA_FUSED_STATE {fused}\n{source}");
            let transposed_library = device
                .new_library_with_source(&transposed_source, Some(&options))
                .map_err(|_| "Metal transposed KDA compilation failed")?;
            let transposed_function = transposed_library
                .new_function_with_name("glm_kda_decode")
                .ok_or("Metal transposed KDA function unavailable")?;
            let transposed_pipeline = device
                .new_compute_pipeline_state_with_function(&transposed_function)
                .map_err(|_| "Metal transposed KDA pipeline failed")?;
            if transposed_pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("Metal transposed KDA execution width is unsupported");
            }
            // The new norm tree is qualified only with the transposed/fused
            // state policy. Keep its Safe PSO distinct from both references.
            let parallel_norm_pipeline =
                if fused == 1 && std::env::var_os("STARK_GLM53_KDA_PARALLEL_NORM").is_some_and(|value| value == "1") {
                    let norm_source = format!("#define STARK_GLM_KDA_PARALLEL_NORM 1\n{transposed_source}");
                    let norm_library = device
                        .new_library_with_source(&norm_source, Some(&options))
                        .map_err(|_| "Metal parallel-norm KDA compilation failed")?;
                    let norm_function = norm_library
                        .new_function_with_name("glm_kda_decode")
                        .ok_or("Metal parallel-norm KDA function unavailable")?;
                    let candidate = device
                        .new_compute_pipeline_state_with_function(&norm_function)
                        .map_err(|_| "Metal parallel-norm KDA pipeline failed")?;
                    (candidate.thread_execution_width() == 32 && candidate.max_total_threads_per_threadgroup() >= 128)
                        .then_some(candidate)
                } else {
                    None
                };
            let mut graph_pipelines = HashMap::new();
            for name in [
                "glm_control_project",
                "glm_control_project_unprefetched",
                "glm_control_project_scalar",
                "glm_control_project_split",
                "glm_control_project_split16",
                "glm_control_project_split32",
                "glm_control_rms_project_split32",
                "glm_control_mix",
                "glm_control_mix_scalar",
                "glm_control_mix_simd",
                "glm_control_mix_split_simd",
                "glm_control_mix_split16_simd",
                "glm_control_mix_split32_simd",
                "glm_control_input",
                "glm_control_post",
                "glm_control_prefill_project",
                "glm_control_prefill_project_parts",
                "glm_control_prefill_rms_project_split32",
                "glm_control_prefill_mix",
                "glm_control_prefill_mix_simd",
                "glm_control_prefill_input",
                "glm_control_prefill_post",
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
                "glm_router_scores",
                "glm_router_select",
                "glm_router_parallel",
            ] {
                let function = library.new_function_with_name(name).ok_or("Metal GLM graph function unavailable")?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| "Metal GLM graph pipeline failed")?;
                if pipeline.max_total_threads_per_threadgroup() < 256 {
                    return Err("Metal GLM graph execution width is unsupported");
                }
                if matches!(
                    name,
                    "glm_control_project"
                        | "glm_router_parallel"
                        | "glm_control_prefill_project"
                        | "glm_control_prefill_project_parts"
                        | "glm_control_prefill_rms_project_split32"
                        | "glm_control_prefill_mix_simd"
                ) && pipeline.thread_execution_width() != 32
                {
                    return Err("Metal GLM control SIMD width is unsupported");
                }
                graph_pipelines.insert(name, pipeline);
            }
            // The optional 768-thread control schedule never rejects hardware
            // whose per-pipeline limit is smaller; that hardware uses the old route.
            if let Some(function) = library.new_function_with_name("glm_control_fused_parallel")
                && let Ok(candidate) = device.new_compute_pipeline_state_with_function(&function)
                && candidate.thread_execution_width() == 32
                && candidate.max_total_threads_per_threadgroup() >= 768
            {
                graph_pipelines.insert("glm_control_fused_parallel", candidate);
            }
            if let Some(function) = library.new_function_with_name("glm_control_prefill_fused_parallel")
                && let Ok(candidate) = device.new_compute_pipeline_state_with_function(&function)
                && candidate.thread_execution_width() == 32
                && candidate.max_total_threads_per_threadgroup() >= 768
            {
                graph_pipelines.insert("glm_control_prefill_fused_parallel", candidate);
            }
            Ok(Executor {
                device,
                #[cfg(test)]
                queue,
                pipeline,
                transposed_pipeline,
                parallel_norm_pipeline,
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
    transposed_state: bool,
}

pub(super) struct KdaSnapshot {
    parameters: [u32; 6],
    transposed_state: bool,
    values: Vec<f32>,
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
            transposed_state: std::env::var_os("STARK_GLM53_KDA_TRANSPOSED").is_some_and(|value| value == "1"),
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

    /// The owning model has drained every command before this exclusive read.
    pub(super) fn snapshot(&self) -> Result<KdaSnapshot> {
        self.check_completed()?;
        let values =
            unsafe { std::slice::from_raw_parts(self.state.contents().as_ptr().cast::<f32>(), self.state_values) }
                .to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("Metal GLM KDA snapshot values"));
        }
        Ok(KdaSnapshot {
            parameters: self.parameters,
            transposed_state: self.transposed_state,
            values,
        })
    }

    pub(super) fn restore(
        &mut self,
        snapshot: &KdaSnapshot,
    ) -> Result<()> {
        if self.parameters != snapshot.parameters
            || self.transposed_state != snapshot.transposed_state
            || self.state_values != snapshot.values.len()
        {
            return Err(Error::Invalid("Metal GLM KDA snapshot geometry"));
        }
        // All writers are fenced; preserve the exact stored F32 bits.
        unsafe {
            self.state
                .contents()
                .as_ptr()
                .cast::<f32>()
                .copy_from_nonoverlapping(snapshot.values.as_ptr(), self.state_values);
            self.failure.contents().as_ptr().cast::<u32>().write(0);
        }
        Ok(())
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
        // Graph status is sticky until model reset. Encoding a future command
        // may overlap an earlier GPU pass, so it must never clear shared status
        // from the CPU while that pass can still write it.
        for buffer in [&self.input, &self.weights, &self.state, &self.output, &self.failure] {
            graph.retain_external_buffer(buffer);
        }
        self.encode_inner(graph.encoder(), false)?;
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
        self.encode_inner(encoding, true)
    }

    fn parallel_norm_admitted(
        &self,
        executor: &Executor,
    ) -> bool {
        self.transposed_state && self.dim() == 128 && executor.parallel_norm_pipeline.is_some()
    }

    fn encode_inner(
        &mut self,
        encoding: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        clear_status: bool,
    ) -> Result<()> {
        // Standalone callers await each pass before this CPU write. Deferred
        // graphs retain the initialized sticky status until exclusive reset.
        if clear_status {
            unsafe { self.failure.contents().as_ptr().cast::<u32>().write(0) };
        }
        let executor = executor()?;
        encoding.set_compute_pipeline_state(if self.parallel_norm_admitted(executor) {
            executor.parallel_norm_pipeline.as_ref().ok_or(Error::Invalid("Metal KDA norm pipeline admission"))?
        } else if self.transposed_state {
            &executor.transposed_pipeline
        } else {
            &executor.pipeline
        });
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
    for value in buffers {
        graph.retain_external_buffer(&value.buffer);
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
    projection_chunks: usize,
    simd_mix: bool,
    rms_project: bool,
    fused_parallel: bool,
    ordered_rms: bool,
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
        let split =
            streams * streams <= 32 && std::env::var_os("STARK_GLM53_CONTROL_SPLIT").is_some_and(|value| value == "1");
        let simd_mix = streams * streams <= 32
            && (split || std::env::var_os("STARK_GLM53_CONTROL_SIMD").is_some_and(|value| value == "1"));
        let projection_chunks = if split {
            match std::env::var_os("STARK_GLM53_CONTROL_CHUNKS").as_deref().and_then(|v| v.to_str()) {
                Some("16") => 16,
                Some("32") => 32,
                Some("8") | None => 8,
                Some(_) => return Err(Error::Invalid("Metal GLM control projection chunks")),
            }
        } else {
            1
        };
        let ordered_rms = std::env::var_os("STARK_HYBRID_ORDERED_RMS").is_some_and(|value| value == "1");
        let rms_project = hidden == 4096
            && streams == 4
            && projection_chunks == 32
            && std::env::var_os("STARK_GLM53_CONTROL_RMS_PROJECT").is_some_and(|value| value == "1")
            && !ordered_rms;
        let fused_parallel =
            rms_project && std::env::var_os("STARK_GLM53_CONTROL_FUSED_PARALLEL").is_some_and(|value| value == "1");
        Ok(Self {
            weights: DecodeGraph::upload(weights)?,
            coefficients: DecodeGraph::upload(&coefficients)?,
            ones: DecodeGraph::upload(&vec![1.0; columns])?,
            normalization: DecodeGraph::upload(normalization)?,
            projection: DecodeGraph::zeros(rows * projection_chunks)?,
            mix: DecodeGraph::zeros(rows)?,
            input: DecodeGraph::zeros(hidden)?,
            parameters: [hidden as u32, streams as u32, iterations as u32, columns as u32, eps.to_bits()],
            rms_eps,
            projection_chunks,
            simd_mix,
            rms_project,
            fused_parallel,
            ordered_rms,
        })
    }

    pub(super) fn pre(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.pre_mode(graph, residual, self.rms_project)
    }

    /// Encode one row before its branch projection, preserving its mix until
    /// the matching post even when later rows reuse this Control's scratch.
    /// These dependent kernels must be outside an independent projection level.
    pub(super) fn pre_batch_row(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        normalized: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.validate_batch_view(residual, self.parameters[3] as usize)?;
        self.validate_batch_view(normalized, self.input.len)?;
        self.validate_batch_destination(normalized, &[residual])?;
        // Raw GLM dispatches bypass kernel()'s write hook. Discharge earlier
        // finite obligations before any persistent row scratch is overwritten.
        for destination in [&self.projection, &self.mix, &self.input, normalized] {
            graph.prepare_external_write(destination)?;
        }
        self.project_mode(graph, residual, self.rms_project)?;
        if self.fused_parallel && self.rms_project && self.fused_parallel_admitted() {
            let p = self.parameters;
            dispatch_with_threads(
                graph,
                "glm_control_fused_parallel",
                &[
                    &self.projection,
                    &self.coefficients,
                    residual,
                    &self.normalization,
                    &self.mix,
                    &self.input,
                    normalized,
                ],
                &[p[0], p[1], p[2], p[3], p[4], self.rms_eps.to_bits()],
                768,
                false,
                768,
            )?;
        } else {
            let value = self.pre_projected(graph, residual)?;
            graph.copy_into(&value, normalized)?;
        }
        let mix = graph.scratch(self.mix.len)?;
        graph.copy_into(&self.mix, &mix)?;
        Ok(mix)
    }

    /// Complete a row using its immutable pre snapshot and explicit output
    /// destination. Source overlap is rejected because streams read each other.
    pub(super) fn post_batch_row(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        output: &DeviceVector,
        mix: &DeviceVector,
        destination: &DeviceVector,
    ) -> Result<()> {
        self.validate_batch_view(residual, self.parameters[3] as usize)?;
        self.validate_batch_view(output, self.input.len)?;
        self.validate_batch_view(mix, self.mix.len)?;
        self.validate_batch_view(destination, residual.len)?;
        self.validate_batch_destination(destination, &[residual, output, mix])?;
        graph.prepare_external_write(destination)?;
        dispatch(
            graph,
            "glm_control_post",
            &[residual, output, mix, destination],
            &self.parameters,
            destination.len,
            false,
        )
    }

    fn validate_batch_view(
        &self,
        value: &DeviceVector,
        length: usize,
    ) -> Result<()> {
        if value.len != length {
            return Err(Error::Invalid("Metal GLM control batch row dimensions"));
        }
        // slice() validates the entire underlying span, including byte offset
        // alignment and checked allocation bounds, before any work is encoded.
        value.slice(0, length).map(|_| ())
    }

    fn validate_batch_destination(
        &self,
        destination: &DeviceVector,
        sources: &[&DeviceVector],
    ) -> Result<()> {
        let overlaps = |source: &DeviceVector| {
            std::ptr::eq(&*source.buffer, &*destination.buffer)
                && source.offset < destination.offset + destination.len * 4
                && destination.offset < source.offset + source.len * 4
        };
        // Neither input rows nor immutable or reusable Control buffers may be
        // silently overwritten by a destination supplied by the batch caller.
        if sources.iter().any(|source| overlaps(source))
            || [
                &self.weights,
                &self.coefficients,
                &self.ones,
                &self.normalization,
                &self.projection,
                &self.mix,
                &self.input,
            ]
            .into_iter()
            .any(overlaps)
        {
            return Err(Error::Invalid("Metal GLM control batch destination overlaps a source"));
        }
        Ok(())
    }

    fn pre_mode(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        rms_project: bool,
    ) -> Result<DeviceVector> {
        if residual.len != self.parameters[3] as usize {
            return Err(Error::Invalid("Metal GLM control residual dimensions"));
        }
        self.project_mode(graph, residual, rms_project)?;
        if self.fused_parallel && rms_project {
            self.pre_projected_fused_parallel(graph, residual)
        } else {
            self.pre_projected(graph, residual)
        }
    }

    fn project_mode(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        rms_project: bool,
    ) -> Result<()> {
        if rms_project && self.parameters[0] == 4096 && self.parameters[1] == 4 && self.projection_chunks == 32 {
            let partials = self.rms_partials(graph, residual)?;
            let p = self.parameters;
            dispatch_with_threads(
                graph,
                "glm_control_rms_project_split32",
                &[residual, &self.weights, &partials, &self.projection],
                &[p[0], p[1], p[2], p[3], p[4], self.rms_eps.to_bits()],
                self.projection.len * 32,
                false,
                32,
            )?;
        } else {
            let normalized = graph.rms_norm(residual, &self.ones, residual.len, self.rms_eps)?;
            dispatch_with_threads(
                graph,
                match self.projection_chunks {
                    8 => "glm_control_project_split",
                    16 => "glm_control_project_split16",
                    32 => "glm_control_project_split32",
                    _ => "glm_control_project",
                },
                &[&normalized, &self.weights, &self.projection],
                &self.parameters,
                self.projection.len * 32,
                false,
                32,
            )?;
        }
        Ok(())
    }

    fn rms_partials(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        let partials = graph.scratch(16)?;
        graph.kernel(
            "hybrid_graph_rms_partials",
            &[residual, &partials],
            &[1, 16384, self.rms_eps.to_bits(), 16384, 16],
            16 * 256,
            256,
        )?;
        Ok(partials)
    }

    #[cfg(test)]
    fn pre_parallel_reference(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.project_parallel_reference(graph, residual)?;
        self.pre_projected(graph, residual)
    }

    #[cfg(test)]
    fn project_parallel_reference(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<()> {
        let partials = self.rms_partials(graph, residual)?;
        let normalized = graph.scratch(16384)?;
        graph.kernel(
            "hybrid_graph_rms_apply",
            &[residual, &self.ones, &partials, &normalized],
            &[1, 16384, self.rms_eps.to_bits(), 16384, 16],
            64 * 256,
            256,
        )?;
        dispatch_with_threads(
            graph,
            "glm_control_project_split32",
            &[&normalized, &self.weights, &self.projection],
            &self.parameters,
            self.projection.len * 32,
            false,
            32,
        )?;
        Ok(())
    }

    fn pre_projected(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        dispatch_with_threads(
            graph,
            match self.projection_chunks {
                8 => "glm_control_mix_split_simd",
                16 => "glm_control_mix_split16_simd",
                32 => "glm_control_mix_split32_simd",
                _ if self.simd_mix => "glm_control_mix_simd",
                _ => "glm_control_mix",
            },
            &[&self.projection, &self.coefficients, &self.mix],
            &self.parameters,
            if self.simd_mix {
                32
            } else {
                64
            },
            false,
            if self.simd_mix {
                32
            } else {
                64
            },
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

    fn fused_parallel_admitted(&self) -> bool {
        self.parameters[0] == 4096
            && self.parameters[1] == 4
            && self.projection_chunks == 32
            && !self.ordered_rms
            && executor().is_ok_and(|e| e.graph_pipelines.contains_key("glm_control_fused_parallel"))
    }

    fn pre_projected_fused_parallel(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        if !self.fused_parallel_admitted() {
            return self.pre_projected(graph, residual);
        }
        let output = graph.scratch(4096)?;
        let p = self.parameters;
        dispatch_with_threads(
            graph,
            "glm_control_fused_parallel",
            &[&self.projection, &self.coefficients, residual, &self.normalization, &self.mix, &self.input, &output],
            &[p[0], p[1], p[2], p[3], p[4], self.rms_eps.to_bits()],
            768,
            false,
            768,
        )?;
        Ok(output)
    }

    #[cfg(test)]
    fn pre_unfused(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.project_mode(graph, residual, self.rms_project)?;
        self.pre_projected(graph, residual)
    }

    #[cfg(test)]
    fn pre_fused_parallel(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
    ) -> Result<DeviceVector> {
        if residual.len != self.parameters[3] as usize {
            return Err(Error::Invalid("Metal GLM fused control residual dimensions"));
        }
        if !self.fused_parallel_admitted() {
            return self.pre(graph, residual);
        }
        self.project_mode(graph, residual, self.rms_project)?;
        self.pre_projected_fused_parallel(graph, residual)
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

/// Correction-biased stable top-k stays on the GPU. Mixture coefficients use
/// the original sigmoid probabilities and selected-order F32 normalization.
pub(super) fn route(
    graph: &mut DecodeGraph,
    logits: &DeviceVector,
    bias: &DeviceVector,
    active: usize,
    scale: f32,
) -> Result<(DeviceVector, DeviceVector)> {
    static SERIAL_ROUTER: OnceLock<bool> = OnceLock::new();
    let serial =
        *SERIAL_ROUTER.get_or_init(|| std::env::var_os("STARK_GLM53_SERIAL_ROUTER").is_some_and(|value| value == "1"));
    route_mode(graph, logits, bias, active, scale, !serial)
}

fn route_mode(
    graph: &mut DecodeGraph,
    logits: &DeviceVector,
    bias: &DeviceVector,
    active: usize,
    scale: f32,
    parallel: bool,
) -> Result<(DeviceVector, DeviceVector)> {
    if logits.len == 0
        || logits.len > 2048
        || logits.len != bias.len
        || active == 0
        || active > 64
        || active > logits.len
        || !scale.is_finite()
    {
        return Err(Error::Invalid("Metal GLM router geometry"));
    }
    let indices = graph.scratch(active)?;
    let mixture = graph.scratch(active)?;
    let parameters = [logits.len as u32, active as u32, scale.to_bits()];
    if parallel {
        dispatch(graph, "glm_router_parallel", &[logits, bias, &indices, &mixture], &parameters, 1, true)?;
    } else {
        let probabilities = graph.scratch(logits.len)?;
        let choices = graph.scratch(logits.len)?;
        dispatch(
            graph,
            "glm_router_scores",
            &[logits, bias, &probabilities, &choices],
            &parameters,
            logits.len,
            false,
        )?;
        dispatch(graph, "glm_router_select", &[&probabilities, &choices, &indices, &mixture], &parameters, 1, true)?;
    }
    Ok((indices, mixture))
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

/// Completed pool and latent rows remain in the model's unchanged prefix.
/// The incomplete pool is overwritten by later requests, so copy its tail.
pub(super) struct AttentionSnapshot {
    position: usize,
    tail_keys: Vec<f32>,
    tail_gates: Vec<f32>,
}

impl Attention {
    /// Capacity is published only after both replacement owners exist. Failed
    /// growth therefore cannot grant speculative allocation credit.
    pub(super) fn resident_capacity(&self) -> usize {
        self.capacity
    }

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

    pub(super) fn snapshot(&self) -> Result<AttentionSnapshot> {
        let length = (self.position % self.geometry.pool) * self.geometry.index_dim;
        let read = |vector: &DeviceVector| {
            unsafe {
                std::slice::from_raw_parts(
                    vector.buffer.contents().as_ptr().byte_add(vector.offset).cast::<f32>(),
                    length,
                )
            }
            .to_vec()
        };
        let tail_keys = read(&self.tail_keys);
        let tail_gates = read(&self.tail_gates);
        if tail_keys.iter().chain(&tail_gates).any(|value| !value.is_finite()) {
            return Err(Error::Invalid("Metal GLM attention snapshot values"));
        }
        Ok(AttentionSnapshot {
            position: self.position,
            tail_keys,
            tail_gates,
        })
    }

    pub(super) fn restore(
        &mut self,
        snapshot: &AttentionSnapshot,
    ) -> Result<()> {
        let length = (snapshot.position % self.geometry.pool) * self.geometry.index_dim;
        if self.position != 0
            || snapshot.position == 0
            || snapshot.position > self.capacity
            || snapshot.position >= self.geometry.context
            || snapshot.tail_keys.len() != length
            || snapshot.tail_gates.len() != length
        {
            return Err(Error::Invalid("Metal GLM attention snapshot admission"));
        }
        for (vector, values) in [(&self.tail_keys, &snapshot.tail_keys), (&self.tail_gates, &snapshot.tail_gates)] {
            unsafe {
                vector
                    .buffer
                    .contents()
                    .as_ptr()
                    .byte_add(vector.offset)
                    .cast::<f32>()
                    .copy_from_nonoverlapping(values.as_ptr(), length);
            }
        }
        self.position = snapshot.position;
        Ok(())
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
    fn transposed_kda_state_preserves_output_and_logical_state_bits() {
        for dim in [7, 128] {
            let heads = 2;
            let inner = heads * dim;
            let kernel = 4;
            let values = |n: usize, seed: usize| -> Vec<f32> {
                (0..n).map(|i| (((i * 13 + seed * 7) % 43) as f32 - 21.0) * 0.013).collect()
            };
            let convolutions = [values(inner * kernel, 1), values(inner * kernel, 2), values(inner * kernel, 3)];
            let create = || {
                Kda::new(
                    Geometry {
                        heads,
                        dim,
                        kernel,
                        rms_eps: 1e-5,
                        lower_bound: -5.0,
                    },
                    [&convolutions[0], &convolutions[1], &convolutions[2]],
                    &values(inner, 4),
                    &vec![1.0; dim],
                    &vec![-1.25; heads],
                )
                .unwrap()
            };
            let mut legacy = create();
            legacy.transposed_state = false;
            let mut transposed = create();
            transposed.transposed_state = true;
            for _ in 0..2 {
                legacy.reset();
                transposed.reset();
                for step in 0..17 {
                    let inputs = [
                        values(inner, step + 5),
                        values(inner, step + 8),
                        values(inner, step + 11),
                        values(inner, step + 14),
                        values(inner, step + 17),
                    ];
                    let beta = values(heads, step + 20);
                    let first =
                        legacy.step([&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]], &beta).unwrap();
                    let second =
                        transposed.step([&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]], &beta).unwrap();
                    assert!(first.iter().zip(&second).all(|(a, b)| a.to_bits() == b.to_bits()));
                }
                let first = unsafe {
                    std::slice::from_raw_parts(legacy.state.contents().as_ptr().cast::<f32>(), legacy.state_values)
                };
                let second = unsafe {
                    std::slice::from_raw_parts(
                        transposed.state.contents().as_ptr().cast::<f32>(),
                        transposed.state_values,
                    )
                };
                let history = 3 * inner * (kernel - 1);
                assert!(first[..history].iter().zip(&second[..history]).all(|(a, b)| a.to_bits() == b.to_bits()));
                for head in 0..heads {
                    for value in 0..dim {
                        for key in 0..dim {
                            assert_eq!(
                                first[history + head * dim * dim + value * dim + key].to_bits(),
                                second[history + head * dim * dim + key * dim + value].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }

    #[stark_test]
    fn simd_control_mixing_preserves_chronological_sinkhorn_bits_and_failures() {
        for streams in 1..=5 {
            let rows = (2 + streams) * streams;
            for iterations in [1, 3, 20] {
                let parameters = [7, streams as u32, iterations, (7 * streams) as u32, 1e-6f32.to_bits()];
                let projection: Vec<f32> = (0..rows).map(|i| (i as f32 * 1.13).sin() * 7.0).collect();
                let mut coefficients: Vec<f32> = (0..rows).map(|i| (i as f32 * 0.37).cos() * 0.7).collect();
                coefficients.extend([0.4, 0.7, 1.2]);
                let projected = DecodeGraph::upload(&projection).unwrap();
                let weights = DecodeGraph::upload(&coefficients).unwrap();
                let legacy = DecodeGraph::zeros(rows).unwrap();
                let simd = DecodeGraph::zeros(rows).unwrap();
                let graph = DecodeGraph::new().unwrap();
                dispatch_with_threads(
                    &graph,
                    "glm_control_mix",
                    &[&projected, &weights, &legacy],
                    &parameters,
                    64,
                    false,
                    64,
                )
                .unwrap();
                dispatch_with_threads(
                    &graph,
                    "glm_control_mix_simd",
                    &[&projected, &weights, &simd],
                    &parameters,
                    32,
                    false,
                    32,
                )
                .unwrap();
                let values = graph.finish(&[legacy, simd]).unwrap();
                assert!(values[0].iter().zip(&values[1]).all(|(a, b)| a.to_bits() == b.to_bits()));

                // An infinite projection must remain a failure even when a
                // sigmoid or a later clipping operator produces finite values.
                unsafe { projected.buffer.contents().as_ptr().cast::<f32>().write(f32::INFINITY) };
                let graph = DecodeGraph::new().unwrap();
                let output = DecodeGraph::zeros(rows).unwrap();
                dispatch_with_threads(
                    &graph,
                    "glm_control_mix_simd",
                    &[&projected, &weights, &output],
                    &parameters,
                    32,
                    false,
                    32,
                )
                .unwrap();
                assert!(graph.finish(&[output]).is_err());
            }
        }
    }

    #[stark_test]
    fn split_control_projection_matches_wide_oracle_with_tails() {
        for (hidden, streams) in [(7, 4), (65, 4), (4096, 4), (13, 1), (13, 5)] {
            let columns = hidden * streams;
            let rows = (2 + streams) * streams;
            let input: Vec<f32> = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect();
            let weights: Vec<f32> = (0..rows * columns).map(|i| (((i * 17) % 97) as f32 - 48.0) * 0.0005).collect();
            let source = DecodeGraph::upload(&input).unwrap();
            let matrix = DecodeGraph::upload(&weights).unwrap();
            for (chunks, name) in [
                (8, "glm_control_project_split"),
                (16, "glm_control_project_split16"),
                (32, "glm_control_project_split32"),
            ] {
                let partials = DecodeGraph::zeros(rows * chunks).unwrap();
                let graph = DecodeGraph::new().unwrap();
                let parameters = [hidden as u32, streams as u32, 20, columns as u32, 1e-6f32.to_bits()];
                dispatch_with_threads(
                    &graph,
                    name,
                    &[&source, &matrix, &partials],
                    &parameters,
                    rows * chunks * 32,
                    false,
                    32,
                )
                .unwrap();
                let actual = graph.finish(&[partials]).unwrap().remove(0);
                for row in 0..rows {
                    let mut total = 0.0f32;
                    for value in &actual[row * chunks..(row + 1) * chunks] {
                        total += value;
                    }
                    let oracle: f64 = weights[row * columns..(row + 1) * columns]
                        .iter()
                        .zip(&input)
                        .map(|(&w, &x)| f64::from(w) * f64::from(x))
                        .sum();
                    let magnitude: f64 = weights[row * columns..(row + 1) * columns]
                        .iter()
                        .zip(&input)
                        .map(|(&w, &x)| (f64::from(w) * f64::from(x)).abs())
                        .sum();
                    assert!((f64::from(total) - oracle).abs() <= 5e-7 * magnitude.max(1.0));
                }
            }
        }
    }

    #[stark_test]
    fn graph_parallel_router_matches_serial_gpu_bits() {
        for (experts, active) in [(1, 1), (7, 3), (288, 8), (288, 64), (2048, 8), (2048, 64)] {
            for tied in [true, false] {
                let logits: Vec<_> = (0..experts)
                    .map(|i| {
                        if tied {
                            0.0
                        } else {
                            (i as f32 * 0.37).sin() * 9.0
                        }
                    })
                    .collect();
                let bias: Vec<_> = (0..experts)
                    .map(|i| {
                        if tied {
                            0.0
                        } else {
                            (i as f32 * 0.13).cos() * 0.2
                        }
                    })
                    .collect();
                let input = DecodeGraph::upload(&logits).unwrap();
                let correction = DecodeGraph::upload(&bias).unwrap();
                let mut graph = DecodeGraph::new().unwrap();
                let (indices, mixture) = route_mode(&mut graph, &input, &correction, active, 2.5, false).unwrap();
                let serial = graph.finish(&[indices, mixture]).unwrap();
                let mut graph = DecodeGraph::new().unwrap();
                let (indices, mixture) = route_mode(&mut graph, &input, &correction, active, 2.5, true).unwrap();
                let parallel = graph.finish(&[indices, mixture]).unwrap();
                assert!(
                    serial.iter().flatten().zip(parallel.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "parallel router changed selected IDs or coefficient bits"
                );
            }
        }
    }

    #[stark_test]
    fn graph_router_preserves_corrected_selection_and_probability_contract() {
        let cases = [
            (vec![0.0; 7], vec![0.0, 0.0, 0.0, f32::EPSILON, -f32::EPSILON, 0.0, 0.0], 3),
            (
                (0..288).map(|i| (i as f32 * 0.37).sin() * 9.0).collect(),
                (0..288).map(|i| (i as f32 * 0.13).cos() * 0.2).collect(),
                8,
            ),
            (vec![-1000.0, -80.0, 80.0, 1000.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1], 4),
        ];
        for (logits, bias, active) in cases {
            let probabilities: Vec<_> = logits.iter().map(|&v| super::super::math::sigmoid(v)).collect();
            let choices: Vec<_> = probabilities.iter().zip(&bias).map(|(&a, &b)| a + b).collect();
            let expected = super::super::math::topk(&choices, active).unwrap();
            let denominator = expected.iter().map(|&i| probabilities[i]).sum::<f32>() + 1e-20;
            let mut graph = DecodeGraph::new().unwrap();
            let input = DecodeGraph::upload(&logits).unwrap();
            let correction = DecodeGraph::upload(&bias).unwrap();
            let (indices, mixture) = route(&mut graph, &input, &correction, active, 2.5).unwrap();
            let output = graph.finish(&[indices, mixture]).unwrap();
            let actual: Vec<_> = output[0].iter().map(|v| v.to_bits() as usize).collect();
            assert!(actual == expected, "GPU router selection/order changed");
            assert!(
                output[1]
                    .iter()
                    .zip(&expected)
                    .all(|(&actual, &i)| (actual - probabilities[i] / denominator * 2.5).abs() <= 3e-7),
                "GPU router mixture coefficients exceeded the F32 comparison bound"
            );
        }
        let mut graph = DecodeGraph::new().unwrap();
        let probabilities = DecodeGraph::upload(&[0.5; 5]).unwrap();
        let choices = DecodeGraph::upload(&[-0.0, 0.0, 0.0, -1.0, 1.0]).unwrap();
        let indices = graph.scratch(4).unwrap();
        let mixture = graph.scratch(4).unwrap();
        dispatch(
            &graph,
            "glm_router_select",
            &[&probabilities, &choices, &indices, &mixture],
            &[5, 4, 1.0f32.to_bits()],
            1,
            true,
        )
        .unwrap();
        let output = graph.finish(&[indices]).unwrap();
        assert!(output[0].iter().map(|v| v.to_bits()).eq([4, 1, 2, 0]), "router IEEE tie order changed");

        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for parallel in [false, true] {
                let mut graph = DecodeGraph::new().unwrap();
                let logits = DecodeGraph::zeros(7).unwrap();
                unsafe {
                    logits.buffer.contents().as_ptr().cast::<f32>().add(3).write(poison);
                }
                let bias = DecodeGraph::upload(&[0.0; 7]).unwrap();
                let (indices, mixture) = route_mode(&mut graph, &logits, &bias, 3, 1.0, parallel).unwrap();
                assert!(graph.finish(&[indices, mixture]).is_err(), "router accepted nonfinite logits");
            }
        }
        let mut graph = DecodeGraph::new().unwrap();
        let logits = DecodeGraph::upload(&[0.0; 7]).unwrap();
        let bias = DecodeGraph::upload(&[0.0; 7]).unwrap();
        let (indices, mixture) = route(&mut graph, &logits, &bias, 3, 1.0).unwrap();
        assert!(graph.finish(&[indices, mixture]).is_ok(), "router failed to recover on a new graph");
    }

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

    fn synthetic_control_split32(part: usize) -> Control {
        let (hidden, streams) = (4096, 4);
        let rows = (2 + streams) * streams;
        let weights: Vec<_> =
            (0..rows * hidden * streams).map(|i| (((i * 13 + part * 17) % 127) as f32 - 63.0) * 0.0001).collect();
        let mut control = Control::new(
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
        .unwrap();
        control.projection_chunks = 32;
        control.projection = DecodeGraph::zeros(rows * 32).unwrap();
        control.simd_mix = true;
        control
    }

    fn batch_control(fused: bool) -> Control {
        let mut control = synthetic_control_split32(0);
        control.rms_project = fused;
        control.fused_parallel = fused;
        control
    }

    fn encode_control_batch_fixture(
        control: &Control,
        batched: bool,
        unretained: bool,
    ) -> (DecodeGraph, Vec<DeviceVector>) {
        let rows = 3;
        let hidden = control.input.len;
        let columns = control.parameters[3] as usize;
        let residual = DecodeGraph::upload(
            &(0..rows * columns)
                .map(|i| ((i % columns) as f32 * 0.017 + (i / columns) as f32 * 0.31).sin())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let branch = DecodeGraph::upload(
            &(0..rows * hidden)
                .map(|i| ((i % hidden) as f32 * 0.013 + (i / hidden) as f32 * 0.47).cos() * 0.1)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut graph = if unretained {
            DecodeGraph::new_unretained()
        } else {
            DecodeGraph::new()
        }
        .unwrap();
        let normalized = graph.scratch(rows * hidden).unwrap();
        let destination = graph.scratch(rows * columns).unwrap();
        let mut snapshots = Vec::new();
        for row in 0..rows {
            let input = residual.slice(row * columns, columns).unwrap();
            let norm = normalized.slice(row * hidden, hidden).unwrap();
            if batched {
                snapshots.push(control.pre_batch_row(&mut graph, &input, &norm).unwrap());
            } else {
                let value = control.pre(&mut graph, &input).unwrap();
                graph.copy_into(&value, &norm).unwrap();
                let mix = graph.scratch(control.mix.len).unwrap();
                graph.copy_into(&control.mix, &mix).unwrap();
                snapshots.push(mix);
                let output = branch.slice(row * hidden, hidden).unwrap();
                let post = control.post(&mut graph, &input, &output).unwrap();
                graph.copy_into(&post, &destination.slice(row * columns, columns).unwrap()).unwrap();
            }
        }
        if batched {
            // Every pre has overwritten persistent mix/input/projection. Read
            // snapshots in reverse order to catch use of the final row's mix.
            for row in (0..rows).rev() {
                control
                    .post_batch_row(
                        &mut graph,
                        &residual.slice(row * columns, columns).unwrap(),
                        &branch.slice(row * hidden, hidden).unwrap(),
                        &snapshots[row],
                        &destination.slice(row * columns, columns).unwrap(),
                    )
                    .unwrap();
            }
        }
        let mut outputs = Vec::new();
        for (row, mix) in snapshots.into_iter().enumerate() {
            outputs.extend([
                normalized.slice(row * hidden, hidden).unwrap(),
                mix,
                destination.slice(row * columns, columns).unwrap(),
            ]);
        }
        (graph, outputs)
    }

    #[stark_test]
    fn control_batch_rows_snapshot_mix_and_match_legacy_pre_post() {
        for fused in [false, true] {
            let control = batch_control(fused);
            let (reference, outputs) = encode_control_batch_fixture(&control, false, false);
            let expected = reference.finish(&outputs).unwrap();
            for unretained in [false, true] {
                let (graph, outputs) = encode_control_batch_fixture(&control, true, unretained);
                let actual = graph.finish(&outputs).unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| { a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()) }),
                    "Control batch snapshot changed pre/mix/post bits"
                );
            }
        }
    }

    #[stark_test]
    fn control_batch_queued_unretained_snapshots_retain_control_and_row_owners() {
        let control = batch_control(true);
        let (reference, outputs) = encode_control_batch_fixture(&control, false, false);
        let expected = reference.finish(&outputs).unwrap();
        drop(control);
        let pending = objc2::rc::autoreleasepool(|_| {
            let control = batch_control(true);
            let mut pending = Vec::new();
            for _ in 0..2 {
                let (graph, outputs) = encode_control_batch_fixture(&control, true, true);
                graph.validate_queued_submission().unwrap();
                pending.push(graph.submit(&outputs).unwrap());
            }
            // Control, residual/branch locals, row views, and temporary Metal
            // references are gone before either Pending is awaited below.
            pending
        });
        for command in pending {
            let actual = command.wait().unwrap();
            assert!(
                actual.iter().zip(&expected).all(|(a, b)| { a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()) }),
                "Control batch queued ownership changed row results"
            );
        }
    }

    #[stark_test]
    fn control_batch_rows_reject_invalid_spans_and_overlapping_destinations() {
        let control = batch_control(false);
        let residual = DecodeGraph::zeros(16384).unwrap();
        let output = DecodeGraph::zeros(4096).unwrap();
        let wrong = DecodeGraph::zeros(4095).unwrap();
        let invalid = DeviceVector {
            buffer: output.buffer.clone(),
            offset: output.buffer.length(),
            len: 4096,
        };
        let mut graph = DecodeGraph::new().unwrap();
        assert!(control.pre_batch_row(&mut graph, &residual, &wrong).is_err());
        assert!(control.pre_batch_row(&mut graph, &residual, &invalid).is_err());
        assert!(control.pre_batch_row(&mut graph, &residual, &residual.slice(0, 4096).unwrap()).is_err());
        assert!(control.post_batch_row(&mut graph, &residual, &output, &control.mix, &residual).is_err());
        graph.finish(&[]).unwrap();
    }

    #[stark_test]
    fn control_batch_rows_preserve_finite_obligations_before_scratch_reuse() {
        for fused in [false, true] {
            let control = batch_control(fused);
            let residual = DecodeGraph::zeros(16384).unwrap();
            let branch = DecodeGraph::zeros(4096).unwrap();
            for poisoned in [&control.projection, &control.mix, &control.input] {
                // A previous operator's pending obligation must execute before
                // this raw GLM writer replaces its nonfinite value with zero.
                unsafe { poisoned.buffer.contents().as_ptr().cast::<f32>().add(poisoned.offset / 4).write(f32::NAN) };
                let mut graph = DecodeGraph::new().unwrap();
                graph.check_finite(poisoned).unwrap();
                let normalized = graph.scratch(4096).unwrap();
                control.pre_batch_row(&mut graph, &residual, &normalized).unwrap();
                assert!(graph.finish(&[normalized]).is_err(), "Control batch reuse hid a pending nonfinite value");
            }
            let mut graph = DecodeGraph::new().unwrap();
            let normalized = graph.scratch(4096).unwrap();
            unsafe { normalized.buffer.contents().as_ptr().cast::<f32>().add(normalized.offset / 4).write(f32::NAN) };
            graph.check_finite(&normalized).unwrap();
            control.pre_batch_row(&mut graph, &residual, &normalized).unwrap();
            assert!(graph.finish(&[normalized]).is_err(), "Control batch normalized overwrite hid a finite obligation");
            let mut graph = DecodeGraph::new().unwrap();
            let normalized = graph.scratch(4096).unwrap();
            let mix = control.pre_batch_row(&mut graph, &residual, &normalized).unwrap();
            let destination = graph.scratch(16384).unwrap();
            unsafe {
                destination.buffer.contents().as_ptr().cast::<f32>().add(destination.offset / 4).write(f32::INFINITY)
            };
            graph.check_finite(&destination).unwrap();
            control.post_batch_row(&mut graph, &residual, &branch, &mix, &destination).unwrap();
            assert!(graph.finish(&[destination]).is_err(), "Control batch post overwrite hid a finite obligation");
            let mut graph = DecodeGraph::new().unwrap();
            let normalized = graph.scratch(4096).unwrap();
            let mix = control.pre_batch_row(&mut graph, &residual, &normalized).unwrap();
            let destination = graph.scratch(16384).unwrap();
            control.post_batch_row(&mut graph, &residual, &branch, &mix, &destination).unwrap();
            assert!(graph.finish(&[normalized, destination]).is_ok(), "Control batch fresh graph did not recover");
        }
    }

    #[stark_test]
    fn control_rms_project_matches_parallel_reference_bits_and_failures() {
        let control = synthetic_control_split32(0);
        let branch = DecodeGraph::upload(&vec![0.001; 4096]).unwrap();
        let run = |fused: bool, input: &DeviceVector| {
            let mut graph = DecodeGraph::new().unwrap();
            let normalized = if fused {
                control.pre_mode(&mut graph, input, true)
            } else {
                control.pre_parallel_reference(&mut graph, input)
            }
            .unwrap();
            let post = control.post(&mut graph, input, &branch).unwrap();
            graph.finish(&[normalized, control.projection.clone(), control.mix.clone(), control.input.clone(), post])
        };
        for magnitude in [0.0f32, 1.0, 1.0e15] {
            let values: Vec<_> = (0..16384).map(|i| (i as f32 * 0.017).sin() * magnitude).collect();
            let input = DecodeGraph::upload(&values).unwrap();
            let reference = run(false, &input).unwrap();
            let actual = run(true, &input).unwrap();
            assert!(
                reference.iter().zip(&actual).all(|(a, b)| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())),
                "control RMS/project fusion changed pre/project/mix/input/post bits"
            );
            if magnitude == 1.0 {
                for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let poisoned = DecodeGraph::upload(&values).unwrap();
                    unsafe {
                        poisoned
                            .buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(poisoned.offset)
                            .cast::<f32>()
                            .add(16383)
                            .write(invalid);
                    }
                    assert!(run(false, &poisoned).is_err());
                    assert!(run(true, &poisoned).is_err());
                    let recovered = run(true, &input).unwrap();
                    assert!(
                        reference
                            .iter()
                            .zip(&recovered)
                            .all(|(a, b)| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()))
                    );
                }
            }
        }
        // The reference permits finite MAX input to overflow its sum of
        // squares: scale becomes zero and the normalized projection is zero.
        let input = DecodeGraph::upload(&vec![f32::MAX; 16384]).unwrap();
        let mut expected = None;
        for fused in [false, true] {
            let mut graph = DecodeGraph::new().unwrap();
            if fused {
                control.project_mode(&mut graph, &input, true).unwrap();
            } else {
                control.project_parallel_reference(&mut graph, &input).unwrap();
            }
            let actual = graph.finish(&[control.projection.clone()]).unwrap().remove(0);
            assert!(actual.iter().all(|&value| value == 0.0));
            if let Some(reference) = &expected {
                assert!(actual.iter().zip(reference).all(|(a, b): (&f32, &f32)| a.to_bits() == b.to_bits()));
            } else {
                expected = Some(actual);
            }
        }
    }

    #[stark_test]
    #[ignore]
    fn control_rms_project_gpu_component_timings() {
        let _activity = super::super::activity::InferenceActivity::begin().unwrap();
        let controls: Vec<_> = (0..10).map(synthetic_control_split32).collect();
        let input =
            DecodeGraph::upload(&(0..16384).map(|i| ((i % 127) as f32 - 63.0) * 0.001).collect::<Vec<_>>()).unwrap();
        let branch = DecodeGraph::upload(&vec![0.001; 4096]).unwrap();
        profile::enable(true);
        let mut samples = [Vec::new(), Vec::new()];
        for replay in 0..7 {
            for fused in [false, true, true, false] {
                profile::reset();
                let mut graph = DecodeGraph::new().unwrap();
                let mut residual = input.clone();
                for control in &controls {
                    let _ = if fused {
                        control.pre_mode(&mut graph, &residual, true)
                    } else {
                        control.pre_parallel_reference(&mut graph, &residual)
                    }
                    .unwrap();
                    residual = control.post(&mut graph, &residual, &branch).unwrap();
                }
                graph.finish(&[]).unwrap();
                if replay > 0 {
                    samples[usize::from(fused)].push(profile::snapshot()["native.metal.gpu"].seconds / 10.0);
                }
            }
        }
        profile::enable(false);
        for (fused, mut values) in samples.into_iter().enumerate() {
            values.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"component":"control_rms_project_pre_post",
                "fused":fused == 1,"activity":true,"parallel_first_rms":true,"chunks":32,
                "hidden":4096,"streams":4,"controls_per_command":10,"warm_samples":values.len(),
                "median_gpu_microseconds_per_control":(values[5]+values[6])*0.5e6,
                "minimum_gpu_microseconds_per_control":values[0]*1e6,
                "maximum_gpu_microseconds_per_control":values[11]*1e6,"paired_abba":true})
            );
        }
    }

    #[stark_test]
    fn control_fused_parallel_preserves_source_tolerance_and_failures() {
        let control = synthetic_control_split32(0);
        let branch = DecodeGraph::upload(&vec![0.001; 4096]).unwrap();
        let run = |values: &[f32], fused: bool| -> Result<Vec<Vec<f32>>> {
            let input = DecodeGraph::upload(values)?;
            let mut graph = DecodeGraph::new()?;
            let normalized = if fused {
                control.pre_fused_parallel(&mut graph, &input)?
            } else {
                control.pre_unfused(&mut graph, &input)?
            };
            let post = control.post(&mut graph, &input, &branch)?;
            graph.finish(&[normalized, control.mix.clone(), control.input.clone(), post])
        };
        for magnitude in [0.0f32, 1.0, 1.0e15, 1.0e21] {
            let values = (0..16384).map(|i| ((i % 127) as f32 - 63.0) * 0.001 * magnitude).collect::<Vec<_>>();
            let reference = run(&values, false).unwrap();
            let candidate = run(&values, true).unwrap();
            for (a, b) in candidate.iter().zip(&reference) {
                let error: f64 = a.iter().zip(b).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                let power: f64 = b.iter().map(|&v| f64::from(v).powi(2)).sum();
                assert!((error / power.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "parallel fused control source RMS");
            }
            // Independent F64 normalization of the candidate's F32 weighted input.
            let square: f64 = candidate[2].iter().map(|&v| f64::from(v).powi(2)).sum();
            let scale = 1.0 / (square / 4096.0 + f64::from(control.rms_eps)).sqrt();
            if magnitude <= 1.0 {
                for (&actual, &weighted) in candidate[0].iter().zip(&candidate[2]) {
                    let expected = f64::from(weighted) * scale;
                    assert!(
                        (f64::from(actual) - expected).abs() <= 2e-6 * expected.abs().max(1.0),
                        "parallel fused RMS F64 bound"
                    );
                }
            }
        }
        // Poison after allocation because upload correctly rejects nonfinite data.
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let input = DecodeGraph::zeros(16384).unwrap();
            unsafe {
                *input.buffer.contents().as_ptr().cast::<f32>() = poison;
            }
            let mut graph = DecodeGraph::new().unwrap();
            let output = control.pre_fused_parallel(&mut graph, &input).unwrap();
            assert!(graph.finish(&[output]).is_err(), "parallel fused residual poison escaped");
            let projection = vec![0.0f32; 24 * 32];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    projection.as_ptr(),
                    control.projection.buffer.contents().as_ptr().cast::<f32>(),
                    projection.len(),
                );
                *control.projection.buffer.contents().as_ptr().cast::<f32>() = poison;
            }
            let mut graph = DecodeGraph::new().unwrap();
            let clean = DecodeGraph::zeros(16384).unwrap();
            let output = control.pre_projected_fused_parallel(&mut graph, &clean).unwrap();
            assert!(graph.finish(&[output]).is_err(), "parallel fused projection poison escaped");
            assert!(run(&vec![0.0; 16384], true).is_ok(), "parallel fused fresh-command recovery failed");
        }
        // Exercise both ordered fallbacks without the initial residual RMS.
        for magnitude in [1e16f32, f32::MAX] {
            let mut risk_input = vec![0.0f32; 16384];
            risk_input[..4096].fill(magnitude);
            let input = DecodeGraph::upload(&risk_input).unwrap();
            let mut parts = vec![0.0f32; 24 * 32];
            parts[0] = 1e21;
            parts[1] = -1e21;
            parts[2] = 1.0;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    parts.as_ptr(),
                    control.projection.buffer.contents().as_ptr().cast::<f32>(),
                    parts.len(),
                );
            }
            let mut baseline = DecodeGraph::new().unwrap();
            let _ = control.pre_projected(&mut baseline, &input).unwrap();
            let expected = baseline.scratch(4096).unwrap();
            baseline
                .kernel(
                    "hybrid_graph_rms",
                    &[&control.input, &control.normalization, &expected],
                    &[1, 4096, control.rms_eps.to_bits(), 4096],
                    256,
                    256,
                )
                .unwrap();
            let expected = baseline.finish(&[expected]).unwrap().remove(0);
            let mut graph = DecodeGraph::new().unwrap();
            let output = control.pre_projected_fused_parallel(&mut graph, &input).unwrap();
            let actual = graph.finish(&[output]).unwrap().remove(0);
            assert!(
                actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                "parallel fused risk fallback changed RMS bits"
            );
        }
        let mut other = synthetic_control_split32(1);
        other.projection_chunks = 16;
        assert!(!other.fused_parallel_admitted(), "unsupported chunks admitted");
    }

    #[stark_test]
    #[ignore]
    fn control_fused_parallel_production_flag_matches_qualified_schedule() {
        let control = synthetic_control_split32(0);
        assert!(
            control.fused_parallel,
            "fresh process requires CONTROL_FUSED_PARALLEL1, CONTROL_RMS_PROJECT1, CONTROL_SPLIT1, CONTROL_CHUNKS32 and ordered RMS disabled"
        );
        let branch = DecodeGraph::upload(&vec![0.001; 4096]).unwrap();
        for magnitude in [0.0f32, 1.0, 1e15, 1e21] {
            let input = DecodeGraph::upload(
                &(0..16384).map(|i| ((i % 127) as f32 - 63.0) * 0.001 * magnitude).collect::<Vec<_>>(),
            )
            .unwrap();
            let run = |production: bool| {
                let mut graph = DecodeGraph::new().unwrap();
                let normalized = if production {
                    control.pre(&mut graph, &input)
                } else if control.fused_parallel_admitted() {
                    control.pre_fused_parallel(&mut graph, &input)
                } else {
                    control.pre_unfused(&mut graph, &input)
                }
                .unwrap();
                let post = control.post(&mut graph, &input, &branch).unwrap();
                graph.finish(&[normalized, control.mix.clone(), control.input.clone(), post]).unwrap()
            };
            let reference = run(false);
            let actual = run(true);
            for (a, b) in actual.iter().zip(&reference) {
                assert!(
                    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "production fused control diverged from qualified schedule"
                );
            }
        }
        let mut unsupported = synthetic_control_split32(1);
        unsupported.projection_chunks = 16;
        unsupported.projection = DecodeGraph::zeros(24 * 16).unwrap();
        assert!(!unsupported.fused_parallel_admitted());
        let input = DecodeGraph::zeros(16384).unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        let output = unsupported.pre(&mut graph, &input).unwrap();
        assert!(graph.finish(&[output]).is_ok(), "unsupported geometry fallback failed");
        unsupported.projection_chunks = 32;
        unsupported.projection = DecodeGraph::zeros(24 * 32).unwrap();
        unsupported.ordered_rms = true;
        assert!(!unsupported.fused_parallel_admitted(), "ordered diagnostic admitted fused control");
    }

    #[stark_test]
    #[ignore]
    fn control_fused_parallel_serial_concurrent_gpu_component_timings() {
        let _activity = super::super::activity::InferenceActivity::begin().unwrap();
        let controls: Vec<_> = (0..10).map(synthetic_control_split32).collect();
        let input =
            DecodeGraph::upload(&(0..16384).map(|i| ((i % 127) as f32 - 63.0) * 0.001).collect::<Vec<_>>()).unwrap();
        let branch = DecodeGraph::upload(&vec![0.001; 4096]).unwrap();
        profile::enable(true);
        for concurrent in [false, true] {
            let mut samples = [Vec::new(), Vec::new()];
            for replay in 0..7 {
                for fused in [false, true, true, false] {
                    profile::reset();
                    let mut graph = if concurrent {
                        DecodeGraph::new_parallel()
                    } else {
                        DecodeGraph::new()
                    }
                    .unwrap();
                    let mut residual = input.clone();
                    for control in &controls {
                        let _ = if fused {
                            control.pre_fused_parallel(&mut graph, &residual)
                        } else {
                            control.pre_unfused(&mut graph, &residual)
                        }
                        .unwrap();
                        residual = control.post(&mut graph, &residual, &branch).unwrap();
                    }
                    graph.finish(&[]).unwrap();
                    if replay > 0 {
                        samples[usize::from(fused)].push(profile::snapshot()["native.metal.gpu"].seconds / 10.0);
                    }
                }
            }
            for (fused, mut times) in samples.into_iter().enumerate() {
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"component":"control_fused_parallel_pre_post","activity":true,
                    "fused":fused==1,"admitted":controls[0].fused_parallel_admitted(),"concurrent_encoder":concurrent,
                    "hidden":4096,"streams":4,"chunks":32,"threads":768,"controls_per_command":10,"paired_abba":true,
                    "warm_samples":times.len(),"median_gpu_microseconds_per_control":(times[5]+times[6])*0.5e6,
                    "minimum_gpu_microseconds_per_control":times[0]*1e6,"maximum_gpu_microseconds_per_control":times[11]*1e6})
                );
            }
        }
        profile::enable(false);
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

    /// Isolated commands expose the five pre stages and projection occupancy
    /// choices without enabling encoder sampling or loading model weights.
    #[stark_test]
    #[ignore]
    fn glm_control_stage_gpu_component_timings() {
        use std::time::Instant;
        let hidden = 4096;
        let streams = 4;
        let columns = hidden * streams;
        let rows = (2 + streams) * streams;
        let count = 10;
        let chunks = [8, 16, 32];
        let controls: Vec<Vec<_>> = chunks
            .iter()
            .map(|&parts| {
                (0..count)
                    .map(|part| {
                        let weights: Vec<_> = (0..rows * columns)
                            .map(|i| (((i * 13 + part * 17) % 127) as f32 - 63.0) * 0.0001)
                            .collect();
                        let mut control = Control::new(
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
                        .unwrap();
                        control.projection_chunks = parts;
                        control.projection = DecodeGraph::zeros(rows * parts).unwrap();
                        control.simd_mix = true;
                        control
                    })
                    .collect()
            })
            .collect();
        let input =
            DecodeGraph::upload(&(0..columns).map(|i| ((i % 127) as f32 - 63.0) * 0.001).collect::<Vec<_>>()).unwrap();
        let branch = DecodeGraph::upload(&vec![0.001; hidden]).unwrap();
        let cases = [
            ("rms_16384", 0, 0),
            ("projection_split8", 1, 0),
            ("projection_split16", 1, 1),
            ("projection_split32", 1, 2),
            ("mix_split8", 2, 0),
            ("mix_split16", 2, 1),
            ("mix_split32", 2, 2),
            ("weighted_input_4096", 3, 0),
            ("rms_4096", 4, 0),
            ("pre_post_split8", 5, 0),
            ("pre_post_split16", 5, 1),
            ("pre_post_split32", 5, 2),
        ];
        let mut measurements = vec![Vec::new(); cases.len()];
        let concurrent = std::env::var_os("STARK_GLM53_CONCURRENT_ENCODER").is_some_and(|v| v == "1");
        let new_graph = || {
            if concurrent {
                DecodeGraph::new_parallel().unwrap()
            } else {
                DecodeGraph::new().unwrap()
            }
        };
        // Prepare nonzero projection, mixing, and weighted-input intermediates.
        for group in &controls {
            let mut graph = new_graph();
            for control in group {
                let _ = control.pre(&mut graph, &input).unwrap();
            }
            graph.finish(&[]).unwrap();
        }
        profile::enable(true);
        for replay in 0..5 {
            for (case, &(_, stage, group)) in cases.iter().enumerate() {
                profile::reset();
                let started = Instant::now();
                let mut graph = new_graph();
                let mut residual = input.clone();
                for control in &controls[group] {
                    match stage {
                        0 => {
                            let _ = graph.rms_norm(&input, &control.ones, columns, 1e-5).unwrap();
                        },
                        1 => dispatch_with_threads(
                            &graph,
                            ["glm_control_project_split", "glm_control_project_split16", "glm_control_project_split32"]
                                [group],
                            &[&input, &control.weights, &control.projection],
                            &control.parameters,
                            rows * chunks[group] * 32,
                            false,
                            32,
                        )
                        .unwrap(),
                        2 => dispatch_with_threads(
                            &graph,
                            [
                                "glm_control_mix_split_simd",
                                "glm_control_mix_split16_simd",
                                "glm_control_mix_split32_simd",
                            ][group],
                            &[&control.projection, &control.coefficients, &control.mix],
                            &control.parameters,
                            32,
                            false,
                            32,
                        )
                        .unwrap(),
                        3 => dispatch(
                            &graph,
                            "glm_control_input",
                            &[&input, &control.mix, &control.input],
                            &control.parameters,
                            hidden,
                            false,
                        )
                        .unwrap(),
                        4 => {
                            let _ = graph.rms_norm(&control.input, &control.normalization, hidden, 1e-5).unwrap();
                        },
                        5 => {
                            let _ = control.pre(&mut graph, &residual).unwrap();
                            residual = control.post(&mut graph, &residual, &branch).unwrap();
                        },
                        _ => unreachable!(),
                    }
                }
                graph.finish(&[]).unwrap();
                let wall_seconds = started.elapsed().as_secs_f64();
                let gpu_seconds = profile::snapshot().get("native.metal.gpu").unwrap().seconds;
                if replay > 0 {
                    measurements[case].push((gpu_seconds, wall_seconds));
                }
            }
        }
        profile::enable(false);
        let results: Vec<_> = cases
            .iter()
            .zip(measurements)
            .map(|(&(name, _, _), values)| {
                let mut gpu: Vec<_> = values.iter().map(|v| v.0).collect();
                let mut wall: Vec<_> = values.iter().map(|v| v.1).collect();
                gpu.sort_by(f64::total_cmp);
                wall.sort_by(f64::total_cmp);
                let median = (gpu[1] + gpu[2]) * 0.5;
                serde_json::json!({"case":name,"gpu_seconds":gpu,"wall_seconds":wall,
                "gpu_microseconds_per_control":median * 1e6 / count as f64})
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"glm_synthetic_control_stages":true,
            "hidden":hidden,"streams":streams,"controls_per_command":count,
            "measured_replays":4,"concurrent_encoder":concurrent,"cases":results})
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

#[cfg(test)]
#[path = "glm_kda_parallel_norm_tests.rs"]
mod kda_parallel_norm_tests;
