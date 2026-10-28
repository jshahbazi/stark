//! Native packed-weight GEMV. Only the selected expert matrix is submitted;
//! weights are never expanded into a full-precision model in memory.

use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
};

use metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLDevice, MTLDeviceExt,
    MTLDispatchType, MTLFunctionConstantValues, MTLLibrary, MTLLibraryExt, MTLResourceOptions, MTLSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

pub(super) use super::qwen_metal::{Gdn as QwenGdnState, Geometry as QwenGdnGeometry};
use super::{Error, Result, profile, weights};

type PipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type PipelineCache = Mutex<HashMap<MatrixFormat, PipelineState>>;

#[path = "decode_graph.rs"]
mod decode_graph;
pub(super) use decode_graph::{DecodeGraph, DeviceVector};

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: PipelineCache,
    ffn_pipelines: Mutex<Option<(PipelineState, PipelineState)>>,
    hc_pipelines: Mutex<Option<(PipelineState, PipelineState)>>,
    gdn_pipelines: Mutex<Option<(PipelineState, PipelineState)>>,
    graph_pipelines: Mutex<HashMap<&'static str, PipelineState>>,
    indexed_pipelines: PipelineCache,
    ordered_rms: bool,
    #[cfg(test)]
    tiled_pipelines: PipelineCache,
    #[cfg(test)]
    scalar_pipelines: PipelineCache,
}

static EXECUTOR: OnceLock<std::result::Result<Executor, &'static str>> = OnceLock::new();

/// A byte span backed by a Metal buffer whose deallocator retains the source
/// allocation. Only Mapping constructs these views; arbitrary Vec storage is
/// never rounded up or exposed beyond its initialized allocation.
#[derive(Clone)]
pub(super) struct MappedSlice {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    length: usize,
    registered_length: usize,
}

impl MappedSlice {
    pub(super) fn registered_bytes(&self) -> usize {
        self.registered_length
    }

    pub(super) fn subslice(
        &self,
        start: usize,
        length: usize,
    ) -> Result<Self> {
        let offset = self.offset.checked_add(start).ok_or(Error::Invalid("mapped subview offset overflow"))?;
        if length == 0 || start.checked_add(length).is_none_or(|end| end > self.length) || offset > u32::MAX as usize {
            return Err(Error::Invalid("invalid mapped subview span"));
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset,
            length,
            registered_length: self.registered_length,
        })
    }
}

#[derive(Clone)]
pub(super) struct MappedMatrix {
    pub codes: MappedSlice,
    pub planes: Option<(MappedSlice, MappedSlice)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MatrixFormat {
    Packed(u32),
    Affine {
        scale_dtype: u32,
        bits: usize,
        group: usize,
    },
}

pub(super) struct MappedOperation<'a> {
    pub matrices: Vec<MappedMatrix>,
    pub inputs: &'a [f32],
    pub rows: usize,
    pub columns: usize,
    pub rows_per_batch: usize,
    pub format: MatrixFormat,
}

/// Construct a view over page-aligned source memory.
///
/// # Safety
/// `pointer..pointer+mapped_length` must be an initialized, page-aligned mapping
/// retained by `owner`, including any final partial-file page. The logical span
/// must lie within the immutable source file. GPU kernels only read that span.
pub(super) unsafe fn mapped_slice<T: Send + Sync + 'static>(
    pointer: NonNull<c_void>,
    mapped_length: usize,
    offset: usize,
    length: usize,
    owner: Arc<T>,
) -> Result<MappedSlice> {
    if length == 0 || offset > u32::MAX as usize || offset.checked_add(length).is_none_or(|end| end > mapped_length) {
        return Err(Error::Invalid("invalid mapped Metal span"));
    }
    let executor = executor()?;
    if mapped_length > executor.device.max_buffer_length() {
        return Err(Error::Invalid("mapped Metal span exceeds device limit"));
    }
    let _scope = profile::Scope::bytes("native.metal.map", mapped_length);
    let buffer = executor
        .device
        .new_buffer_with_bytes_no_copy_with_deallocator(
            pointer,
            mapped_length,
            MTLResourceOptions::STORAGE_MODE_SHARED,
            move |_, _| {
                let _retained_mapping = &owner;
            },
        )
        .ok_or(Error::Invalid("mapped Metal buffer allocation failed"))?;
    Ok(MappedSlice {
        buffer,
        offset,
        length,
        registered_length: mapped_length,
    })
}

impl MappedOperation<'_> {
    fn parameters(&self) -> Result<[u32; 8]> {
        if self.inputs.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("nonfinite mapped matrix input"));
        }
        self.parameters_for_input(self.inputs.len())
    }

    fn parameters_for_input(
        &self,
        input_len: usize,
    ) -> Result<[u32; 8]> {
        let rows = self.rows;
        let columns = self.columns;
        let batches = if self.rows_per_batch == 0 {
            self.matrices.len()
        } else {
            if self.matrices.len() != 1 || !rows.is_multiple_of(self.rows_per_batch) {
                return Err(Error::Invalid("invalid mapped matrix batches"));
            }
            rows / self.rows_per_batch
        };
        if rows == 0
            || columns == 0
            || rows > 8_000_000
            || columns > 8_000_000
            || self.matrices.is_empty()
            || self.matrices.len().checked_mul(rows).is_none_or(|n| n > 8_000_000)
            || batches.checked_mul(columns) != Some(input_len)
            || input_len > 8_000_000
        {
            return Err(Error::Invalid("invalid mapped matrix dimensions"));
        }
        let (parameters, matrix_bytes, plane_bytes) = match self.format {
            MatrixFormat::Packed(kind) => {
                let (block, bytes) = weights::block_layout(kind)?;
                if !matches!(kind, 0 | 1 | 30 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 23)
                    || !columns.is_multiple_of(block)
                {
                    return Err(Error::Invalid("invalid mapped packed format"));
                }
                let size = rows
                    .checked_mul(columns / block)
                    .and_then(|n| n.checked_mul(bytes))
                    .ok_or(Error::Invalid("mapped matrix size overflow"))?;
                (
                    [rows as u32, columns as u32, kind, block as u32, bytes as u32, 0, self.rows_per_batch as u32, 0],
                    size,
                    0,
                )
            },
            MatrixFormat::Affine {
                scale_dtype,
                bits,
                group,
            } => {
                let item = match scale_dtype {
                    0 => 4,
                    1 | 30 => 2,
                    _ => return Err(Error::Invalid("invalid mapped affine dtype")),
                };
                if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64 | 128) || !columns.is_multiple_of(group) {
                    return Err(Error::Invalid("invalid mapped affine format"));
                }
                let size = rows
                    .checked_mul(columns)
                    .and_then(|n| n.checked_mul(bits))
                    .map(|n| n / 8)
                    .ok_or(Error::Invalid("mapped matrix size overflow"))?;
                let planes = rows
                    .checked_mul(columns / group)
                    .and_then(|n| n.checked_mul(item))
                    .ok_or(Error::Invalid("mapped affine plane overflow"))?;
                (
                    [
                        rows as u32,
                        columns as u32,
                        0,
                        self.rows_per_batch as u32,
                        0,
                        bits as u32,
                        group as u32,
                        scale_dtype,
                    ],
                    size,
                    planes,
                )
            },
        };
        for matrix in &self.matrices {
            if matrix.codes.length != matrix_bytes
                || match &matrix.planes {
                    None => plane_bytes != 0,
                    Some((scales, biases)) => {
                        plane_bytes == 0 || scales.length != plane_bytes || biases.length != plane_bytes
                    },
                }
            {
                return Err(Error::Invalid("invalid mapped matrix byte span"));
            }
        }
        Ok(parameters)
    }
}

pub(super) fn mapped_many(operations: &[MappedOperation<'_>]) -> Result<Vec<Vec<f32>>> {
    if operations.is_empty() || operations.len() > 64 {
        return Err(Error::Invalid("invalid mapped operation count"));
    }
    let parameters = operations.iter().map(MappedOperation::parameters).collect::<Result<Vec<_>>>()?;
    objc2::rc::autoreleasepool(|_| executor()?.run_mapped(operations, &parameters))
}

/// Keep routed gate/up projections, activation, down projections and mixture
/// reduction on the GPU until one final readout. Routing remains on the CPU;
/// the expert order and accumulation order are unchanged. The down-operation
/// inputs describe geometry only and are never uploaded or used.
pub(super) fn mapped_ffn(
    operations: &[MappedOperation<'_>; 4],
    probabilities: &[f32],
    shared_gate: f32,
    shared_values: &[f32],
    shared_up: &[f32],
    clamp: Option<(f32, f32)>,
) -> Result<Vec<f32>> {
    let parameters = operations.iter().map(MappedOperation::parameters).collect::<Result<Vec<_>>>()?;
    let [gate, up, down, shared] = operations;
    let experts = probabilities.len();
    if experts == 0
        || experts > 256
        || gate.matrices.len() != experts
        || up.matrices.len() != experts
        || down.matrices.len() != experts
        || shared.matrices.len() != 1
        || operations.iter().any(|operation| operation.rows_per_batch != 0)
        || gate.rows != up.rows
        || gate.columns != up.columns
        || gate.rows != down.columns
        || gate.columns != down.rows
        || shared.rows != down.rows
        || shared.columns != shared_values.len()
        || shared_values.len() != shared_up.len()
        || gate.inputs != up.inputs
        || probabilities.iter().any(|value| !value.is_finite() || *value < 0.0)
        || !shared_gate.is_finite()
        || shared_values.iter().chain(shared_up).any(|value| !value.is_finite())
        || clamp.is_some_and(|(routed, shared)| {
            !routed.is_finite() || routed <= 0.0 || !shared.is_finite() || shared <= 0.0
        })
    {
        return Err(Error::Invalid("invalid mapped feed-forward dimensions"));
    }
    objc2::rc::autoreleasepool(|_| {
        executor()?.run_ffn(operations, &parameters, probabilities, shared_gate, shared_values, shared_up, clamp)
    })
}

/// Native Qwen hyperconnection's dependent low-rank projections and read/write
/// gates, encoded with one completion boundary. Normalization remains on the
/// CPU so the established grouped F32 sum order is preserved.
pub(super) fn mapped_qwen_hyperconnection(
    operations: &[MappedOperation<'_>],
    normalized: &[f32],
    streams: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if !(2..=3).contains(&operations.len())
        || streams == 0
        || streams > 16
        || normalized.is_empty()
        || !normalized.len().is_multiple_of(streams)
    {
        return Err(Error::Invalid("invalid Qwen hyperconnection geometry"));
    }
    let parameters = operations.iter().map(MappedOperation::parameters).collect::<Result<Vec<_>>>()?;
    let down = &operations[0];
    let up = &operations[1];
    if operations.iter().any(|operation| operation.matrices.len() != 1 || operation.rows_per_batch != 0)
        || down.columns != normalized.len()
        || down.inputs != normalized
        || up.columns != down.rows
        || up.rows != normalized.len()
        || operations.get(2).is_some_and(|inject| {
            inject.rows != streams || inject.columns != normalized.len() || inject.inputs != normalized
        })
    {
        return Err(Error::Invalid("invalid Qwen hyperconnection projections"));
    }
    objc2::rc::autoreleasepool(|_| executor()?.run_qwen_hyperconnection(operations, &parameters, normalized, streams))
}

pub(super) fn mapped_qwen_gdn(
    operations: &[MappedOperation<'_>; 5],
    state: &mut QwenGdnState,
) -> Result<Vec<f32>> {
    let parameters = operations.iter().map(MappedOperation::parameters).collect::<Result<Vec<_>>>()?;
    let g = &state.geometry;
    let value_width = g.value_heads * g.value_dim;
    let widths = [2 * g.key_heads * g.key_dim + value_width, value_width, g.value_heads, g.value_heads];
    let input = operations[0].inputs;
    if operations.iter().any(|operation| operation.matrices.len() != 1 || operation.rows_per_batch != 0)
        || operations[..4].iter().zip(widths).any(|(operation, width)| {
            operation.rows != width || operation.columns != input.len() || operation.inputs != input
        })
        || operations[4].rows != input.len()
        || operations[4].columns != value_width
    {
        return Err(Error::Invalid("invalid Qwen GDN projection geometry"));
    }
    let result = objc2::rc::autoreleasepool(|_| executor()?.run_qwen_gdn(operations, &parameters, state));
    if result.is_err() {
        state.reset();
    }
    result
}

pub(super) fn mapped_glm_kda(
    operations: &[MappedOperation<'_>; 9],
    state: &mut super::glm_metal::Kda,
) -> Result<Vec<f32>> {
    let parameters = operations.iter().map(MappedOperation::parameters).collect::<Result<Vec<_>>>()?;
    let input = operations[0].inputs;
    let inner = state.inner();
    let rank = state.dim();
    let heads = state.heads();
    let widths = [inner, inner, inner, rank, rank, heads];
    if operations.iter().any(|operation| operation.matrices.len() != 1 || operation.rows_per_batch != 0)
        || operations[..6].iter().zip(widths).any(|(operation, width)| {
            operation.rows != width || operation.columns != input.len() || operation.inputs != input
        })
        || operations[6..8].iter().any(|operation| operation.rows != inner || operation.columns != rank)
        || operations[8].rows != input.len()
        || operations[8].columns != inner
    {
        return Err(Error::Invalid("invalid GLM KDA projection geometry"));
    }
    let result = objc2::rc::autoreleasepool(|_| executor()?.run_glm_kda(operations, &parameters, state));
    if result.is_err() {
        state.reset();
    }
    result
}

pub(super) fn initialize() -> Result<()> {
    // Establish a usable reduction pipeline before model readiness. Other
    // storage-specialized pipelines compile lazily on their first use.
    executor()?.pipeline(MatrixFormat::Packed(0)).map(|_| ())
}

pub(super) fn recommended_working_set() -> Result<usize> {
    Ok(executor()?.device.recommended_max_working_set_size() as usize)
}

pub(super) fn max_buffer_length() -> Result<usize> {
    Ok(executor()?.device.max_buffer_length())
}

#[cfg(test)]
pub(super) fn allocated_bytes() -> Result<usize> {
    executor().map(|executor| executor.device.current_allocated_size())
}

fn executor() -> Result<&'static Executor> {
    EXECUTOR
        .get_or_init(|| {
            let device = <dyn MTLDevice>::system_default().ok_or("Metal device unavailable")?;
            let queue = device.new_command_queue().ok_or("Metal queue unavailable")?;
            let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
            for (name, table) in
                [("iq2xxs_grid", weights::IQ2XXS_GRID.as_slice()), ("iq2xs_grid", weights::IQ2XS_GRID.as_slice())]
            {
                source.push_str(&format!("constant ulong {name}[{}]={{", table.len()));
                for value in table {
                    source.push_str(&format!("0x{value:016x}ul,"));
                }
                source.push_str("};\n");
            }
            source.push_str("constant uchar ksigns[128]={");
            for value in weights::KSIGNS_IQ2XS {
                source.push_str(&format!("{value},"));
            }
            source.push_str("};\nconstant short iq4values[16]={");
            for value in weights::KVALUES_IQ4NL {
                source.push_str(&format!("{value},"));
            }
            source.push_str("};\n");
            source.push_str("constant uint iq3xxs_grid[256]={");
            for value in weights::IQ3XXS_GRID {
                source.push_str(&format!("0x{value:08x}u,"));
            }
            source.push_str("};\n");
            source.push_str(include_str!("packed_matvec.metal"));
            source.push_str(include_str!("packed_vector_matvec.metal"));
            source.push_str(include_str!("packed_block_matvec.metal"));
            source.push_str(include_str!("feed_forward.metal"));
            source.push_str(include_str!("qwen_gdn.metal"));
            source.push_str(include_str!("decode_graph.metal"));
            source.push_str(include_str!("qwen_graph.metal"));
            #[cfg(test)]
            source.push_str(include_str!("../../../unit/fixtures/hybrid_tiled_experiment.metal"));
            #[cfg(test)]
            source.push_str(include_str!("../../../unit/fixtures/hybrid_graph_reference.metal"));
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = device
                .new_library_with_source(&source, Some(&options))
                .map_err(|_| "Metal kernel compilation failed")?;
            Ok(Executor {
                device,
                queue,
                library,
                pipelines: Mutex::new(HashMap::new()),
                ffn_pipelines: Mutex::new(None),
                hc_pipelines: Mutex::new(None),
                gdn_pipelines: Mutex::new(None),
                graph_pipelines: Mutex::new(HashMap::new()),
                indexed_pipelines: Mutex::new(HashMap::new()),
                ordered_rms: std::env::var_os("STARK_HYBRID_ORDERED_RMS").is_some_and(|value| value == "1"),
                #[cfg(test)]
                tiled_pipelines: Mutex::new(HashMap::new()),
                #[cfg(test)]
                scalar_pipelines: Mutex::new(HashMap::new()),
            })
        })
        .as_ref()
        .map_err(|message| Error::Unsupported(message))
}

fn buffer(
    executor: &Executor,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    executor
        .device
        .new_buffer_with_data(bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(Error::Invalid("Metal matrix buffer allocation failed"))
}

/// Keep pipeline selection and grid geometry paired at every encoding site.
/// SIMD-row schedules use two or eight 32-lane groups per threadgroup.
fn matrix_groups(
    format: MatrixFormat,
    rows: usize,
) -> usize {
    match format {
        MatrixFormat::Packed(13 | 14) => rows.div_ceil(2),
        MatrixFormat::Packed(17 | 18)
        | MatrixFormat::Affine {
            ..
        } => rows.div_ceil(8),
        _ => rows,
    }
}

fn matrix_threads(format: MatrixFormat) -> usize {
    match format {
        MatrixFormat::Packed(13 | 14) => 64,
        _ => 256,
    }
}

impl Executor {
    fn run_glm_kda(
        &self,
        operations: &[MappedOperation<'_>; 9],
        parameters: &[[u32; 8]],
        state: &mut super::glm_metal::Kda,
    ) -> Result<Vec<f32>> {
        let pipelines =
            operations.iter().map(|operation| self.pipeline(operation.format)).collect::<Result<Vec<_>>>()?;
        let inner = state.inner();
        let rank = state.dim();
        let dim = operations[8].rows;
        let allocate = |elements: usize| {
            self.device
                .new_buffer(elements * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal KDA projection allocation failed"))
        };
        let upload = profile::Scope::bytes("native.metal.upload", operations[0].inputs.len() * 4);
        let input = buffer(self, bytemuck::cast_slice(operations[0].inputs))?;
        let f_a = allocate(rank)?;
        let g_a = allocate(rank)?;
        let output = allocate(dim)?;
        drop(upload);
        profile::record("native.metal.allocated_bytes", 0.0, self.device.current_allocated_size());
        let execute = profile::Scope::new("native.metal.execute");
        let encode = profile::Scope::new("native.metal.encode");
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal KDA graph command allocation failed"))?;
        let encoding =
            command.compute_command_encoder().ok_or(Error::Invalid("Metal KDA graph encoder allocation failed"))?;
        for (index, offset) in [(0, 0), (1, inner), (2, 2 * inner), (5, 5 * inner)] {
            Self::encode_matrix_at(
                &encoding,
                &operations[index],
                &parameters[index],
                &pipelines[index],
                (&input, 0),
                (state.input_buffer(), offset * 4),
            );
        }
        Self::encode_ffn_matrix(&encoding, &operations[3], &parameters[3], &pipelines[3], &input, &f_a);
        Self::encode_ffn_matrix(&encoding, &operations[4], &parameters[4], &pipelines[4], &input, &g_a);
        Self::encode_matrix_at(
            &encoding,
            &operations[6],
            &parameters[6],
            &pipelines[6],
            (&f_a, 0),
            (state.input_buffer(), 3 * inner * 4),
        );
        Self::encode_matrix_at(
            &encoding,
            &operations[7],
            &parameters[7],
            &pipelines[7],
            (&g_a, 0),
            (state.input_buffer(), 4 * inner * 4),
        );
        state.encode(&encoding)?;
        Self::encode_ffn_matrix(
            &encoding,
            &operations[8],
            &parameters[8],
            &pipelines[8],
            state.output_buffer(),
            &output,
        );
        encoding.end_encoding();
        drop(encode);
        let submit = profile::Scope::new("native.metal.submit");
        command.commit();
        drop(submit);
        let wait = profile::Scope::new("native.metal.wait");
        command.wait_until_completed();
        drop(wait);
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal GLM KDA graph execution failed"));
        }
        drop(execute);
        profile::record("native.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
        state.check_completed()?;
        let _readback = profile::Scope::bytes("native.metal.readback", dim * 4);
        // SAFETY: the dim-sized owned shared output's writer has completed.
        let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), dim) }.to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("nonfinite GLM KDA graph output"));
        }
        Ok(values)
    }

    fn gdn_pipelines(&self) -> Result<(PipelineState, PipelineState)> {
        let mut pipelines = self.gdn_pipelines.lock().map_err(|_| Error::Invalid("Metal GDN pipeline cache failed"))?;
        if let Some(pipelines) = &*pipelines {
            return Ok(pipelines.clone());
        }
        let make = |name| {
            let function =
                self.library.new_function_with_name(name).ok_or(Error::Invalid("Metal GDN function failed"))?;
            self.device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Error::Invalid("Metal GDN pipeline failed"))
        };
        let result = (make("hybrid_qwen_gdn_convolve")?, make("hybrid_qwen_gdn_recurrent")?);
        *pipelines = Some(result.clone());
        Ok(result)
    }

    fn run_qwen_gdn(
        &self,
        operations: &[MappedOperation<'_>; 5],
        parameters: &[[u32; 8]],
        state: &mut QwenGdnState,
    ) -> Result<Vec<f32>> {
        let pipelines =
            operations.iter().map(|operation| self.pipeline(operation.format)).collect::<Result<Vec<_>>>()?;
        let (conv_pipeline, recurrent_pipeline) = self.gdn_pipelines()?;
        let g = &state.geometry;
        let threads = g.key_dim.max(g.value_dim).next_power_of_two();
        if recurrent_pipeline.max_total_threads_per_threadgroup() < threads {
            return Err(Error::Unsupported("Metal Qwen GDN execution width"));
        }
        let dim = operations[4].rows;
        let allocate = |elements: usize| {
            self.device
                .new_buffer(elements * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal GDN projection allocation failed"))
        };
        let upload = profile::Scope::bytes("native.metal.upload", operations[0].inputs.len() * 4);
        let input = buffer(self, bytemuck::cast_slice(operations[0].inputs))?;
        let projections =
            operations[..4].iter().map(|operation| allocate(operation.rows)).collect::<Result<Vec<_>>>()?;
        let output = allocate(dim)?;
        let failure = buffer(self, bytemuck::bytes_of(&0u32))?;
        drop(upload);
        profile::record("native.metal.allocated_bytes", 0.0, self.device.current_allocated_size());
        let execute = profile::Scope::new("native.metal.execute");
        let encode = profile::Scope::new("native.metal.encode");
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal GDN command allocation failed"))?;
        let encoding =
            command.compute_command_encoder().ok_or(Error::Invalid("Metal GDN encoder allocation failed"))?;
        for index in 0..4 {
            Self::encode_ffn_matrix(
                &encoding,
                &operations[index],
                &parameters[index],
                &pipelines[index],
                &input,
                &projections[index],
            );
        }
        encoding.set_compute_pipeline_state(&conv_pipeline);
        for (index, buffer) in
            [&projections[0], &state.controls, &state.history, &state.convolved].into_iter().enumerate()
        {
            encoding.set_buffer(Some(buffer), 0, index);
        }
        encoding.set_bytes(NonNull::from(&state.parameters).cast(), std::mem::size_of_val(&state.parameters), 4);
        encoding.set_buffer(Some(&failure), 0, 6);
        encoding.dispatch_threads(
            MTLSize {
                width: state.parameters[7] as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoding.set_compute_pipeline_state(&recurrent_pipeline);
        for (index, buffer) in [
            &state.convolved,
            &projections[1],
            &projections[2],
            &projections[3],
            &state.controls,
            &state.recurrent,
            &failure,
            &state.result,
        ]
        .into_iter()
        .enumerate()
        {
            encoding.set_buffer(Some(buffer), 0, index);
        }
        encoding.set_bytes(NonNull::from(&state.parameters).cast(), std::mem::size_of_val(&state.parameters), 8);
        encoding.dispatch_threadgroups(
            MTLSize {
                width: g.value_heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Self::encode_ffn_matrix(&encoding, &operations[4], &parameters[4], &pipelines[4], &state.result, &output);
        encoding.end_encoding();
        drop(encode);
        let submit = profile::Scope::new("native.metal.submit");
        command.commit();
        drop(submit);
        let wait = profile::Scope::new("native.metal.wait");
        command.wait_until_completed();
        drop(wait);
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal Qwen GDN execution failed"));
        }
        drop(execute);
        profile::record("native.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
        let _readback = profile::Scope::bytes("native.metal.readback", dim * 4);
        // SAFETY: all GPU writes have completed and owned shared buffers have
        // the checked output/status lengths. Recurrent state never leaves GPU
        // buffers during inference and is zeroed on reset and failures.
        let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), dim) }.to_vec();
        if values.iter().any(|value| !value.is_finite()) || unsafe { *failure.contents().as_ptr().cast::<u32>() } != 0 {
            return Err(Error::Invalid("nonfinite Metal Qwen GDN result"));
        }
        Ok(values)
    }

    fn hc_pipelines(&self) -> Result<(PipelineState, PipelineState)> {
        let mut pipelines = self.hc_pipelines.lock().map_err(|_| Error::Invalid("Metal HC pipeline cache failed"))?;
        if let Some(pipelines) = &*pipelines {
            return Ok(pipelines.clone());
        }
        let make = |name| {
            let function =
                self.library.new_function_with_name(name).ok_or(Error::Invalid("Metal HC function failed"))?;
            self.device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Error::Invalid("Metal HC pipeline failed"))
        };
        let result = (make("hybrid_qwen_hc_activate")?, make("hybrid_qwen_hc_mix")?);
        *pipelines = Some(result.clone());
        Ok(result)
    }

    fn run_qwen_hyperconnection(
        &self,
        operations: &[MappedOperation<'_>],
        parameters: &[[u32; 8]],
        normalized: &[f32],
        streams: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let pipelines =
            operations.iter().map(|operation| self.pipeline(operation.format)).collect::<Result<Vec<_>>>()?;
        let (activation_pipeline, mix_pipeline) = self.hc_pipelines()?;
        let dim = normalized.len() / streams;
        let low_rank = operations[0].rows;
        let combine = operations.len() == 3;
        let allocate = |elements: usize| {
            self.device
                .new_buffer(elements * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal HC allocation failed"))
        };
        let upload = profile::Scope::bytes("native.metal.upload", normalized.len() * 4);
        let input = buffer(self, bytemuck::cast_slice(normalized))?;
        let reduced = allocate(low_rank)?;
        let gates = allocate(normalized.len())?;
        let inject = allocate(streams)?;
        let output = allocate(
            dim + if combine {
                streams
            } else {
                0
            },
        )?;
        let failure = buffer(self, bytemuck::bytes_of(&0u32))?;
        drop(upload);
        profile::record("native.metal.allocated_bytes", 0.0, self.device.current_allocated_size());
        let execute = profile::Scope::new("native.metal.execute");
        let encode = profile::Scope::new("native.metal.encode");
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal HC command allocation failed"))?;
        let encoding = command.compute_command_encoder().ok_or(Error::Invalid("Metal HC encoder allocation failed"))?;
        Self::encode_ffn_matrix(&encoding, &operations[0], &parameters[0], &pipelines[0], &input, &reduced);
        encoding.set_compute_pipeline_state(&activation_pipeline);
        encoding.set_buffer(Some(&reduced), 0, 0);
        let sizes = [low_rank as u32, streams as u32];
        encoding.set_bytes(NonNull::from(&sizes).cast(), std::mem::size_of_val(&sizes), 1);
        encoding.set_buffer(Some(&failure), 0, 6);
        encoding.dispatch_threads(
            MTLSize {
                width: low_rank,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Self::encode_ffn_matrix(&encoding, &operations[1], &parameters[1], &pipelines[1], &reduced, &gates);
        if combine {
            Self::encode_ffn_matrix(&encoding, &operations[2], &parameters[2], &pipelines[2], &input, &inject);
        }
        encoding.set_compute_pipeline_state(&mix_pipeline);
        for (index, value) in [&gates, &input, &inject, &output].into_iter().enumerate() {
            encoding.set_buffer(Some(value), 0, index);
        }
        let sizes = [dim as u32, streams as u32, u32::from(combine), 0];
        encoding.set_bytes(NonNull::from(&sizes).cast(), std::mem::size_of_val(&sizes), 4);
        encoding.set_buffer(Some(&failure), 0, 6);
        encoding.dispatch_threads(
            MTLSize {
                width: dim.max(streams),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoding.end_encoding();
        drop(encode);
        let submit = profile::Scope::new("native.metal.submit");
        command.commit();
        drop(submit);
        let wait = profile::Scope::new("native.metal.wait");
        command.wait_until_completed();
        drop(wait);
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal HC execution failed"));
        }
        drop(execute);
        profile::record("native.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
        let count = dim
            + if combine {
                streams
            } else {
                0
            };
        let _readback = profile::Scope::bytes("native.metal.readback", count * 4);
        // SAFETY: checked shared buffers remain owned through GPU completion.
        let mut values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count) }.to_vec();
        if values.iter().any(|value| !value.is_finite()) || unsafe { *failure.contents().as_ptr().cast::<u32>() } != 0 {
            return Err(Error::Invalid("nonfinite Metal HC output"));
        }
        let inject = values.split_off(dim);
        Ok((values, inject))
    }

    fn ffn_pipelines(&self) -> Result<(PipelineState, PipelineState)> {
        let mut pipelines = self.ffn_pipelines.lock().map_err(|_| Error::Invalid("Metal FFN pipeline cache failed"))?;
        if let Some(pipelines) = &*pipelines {
            return Ok(pipelines.clone());
        }
        let make = |name| {
            let function =
                self.library.new_function_with_name(name).ok_or(Error::Invalid("Metal FFN function failed"))?;
            self.device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Error::Invalid("Metal FFN pipeline failed"))
        };
        let result = (make("hybrid_ffn_activate")?, make("hybrid_ffn_reduce")?);
        *pipelines = Some(result.clone());
        Ok(result)
    }

    /// Encode the existing matrix kernel against resident intermediates. The
    /// same format-specific row geometry is used by standalone mapped GEMV.
    fn encode_ffn_matrix(
        encoding: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        operation: &MappedOperation<'_>,
        parameters: &[u32; 8],
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
    ) {
        Self::encode_matrix_at(encoding, operation, parameters, pipeline, (input, 0), (output, 0));
    }

    fn encode_matrix_at(
        encoding: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        operation: &MappedOperation<'_>,
        parameters: &[u32; 8],
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: (&ProtocolObject<dyn MTLBuffer>, usize),
        output: (&ProtocolObject<dyn MTLBuffer>, usize),
    ) {
        encoding.set_compute_pipeline_state(pipeline);
        encoding.set_bytes(NonNull::from(parameters).cast(), std::mem::size_of_val(parameters), 3);
        for (index, matrix) in operation.matrices.iter().enumerate() {
            encoding.set_buffer(Some(&matrix.codes.buffer), 0, 0);
            encoding.set_buffer(Some(input.0), input.1 + index * operation.columns * 4, 1);
            encoding.set_buffer(Some(output.0), output.1 + index * operation.rows * 4, 2);
            let mut offsets = [matrix.codes.offset as u32, 0, 0, 0];
            if let Some((scales, biases)) = &matrix.planes {
                encoding.set_buffer(Some(&scales.buffer), 0, 4);
                encoding.set_buffer(Some(&biases.buffer), 0, 5);
                offsets[1] = scales.offset as u32;
                offsets[2] = biases.offset as u32;
            }
            encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: matrix_groups(operation.format, operation.rows),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: matrix_threads(operation.format),
                    height: 1,
                    depth: 1,
                },
            );
        }
    }

    #[expect(clippy::too_many_arguments, reason = "source-bound FFN operands are separate checked buffers")]
    fn run_ffn(
        &self,
        operations: &[MappedOperation<'_>; 4],
        parameters: &[[u32; 8]],
        probabilities: &[f32],
        shared_gate: f32,
        shared_values: &[f32],
        shared_up: &[f32],
        clamp: Option<(f32, f32)>,
    ) -> Result<Vec<f32>> {
        let pipelines =
            operations.iter().map(|operation| self.pipeline(operation.format)).collect::<Result<Vec<_>>>()?;
        let (activation_pipeline, reduction_pipeline) = self.ffn_pipelines()?;
        let experts = probabilities.len();
        let hidden = operations[0].rows;
        let dim = operations[2].rows;
        let shared_dim = shared_values.len();
        let allocate = |elements: usize| {
            self.device
                .new_buffer(elements * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal FFN allocation failed"))
        };
        let _upload = profile::Scope::bytes(
            "native.metal.upload",
            (operations[0].inputs.len() + shared_values.len() + shared_up.len() + probabilities.len()) * 4,
        );
        let input = buffer(self, bytemuck::cast_slice(operations[0].inputs))?;
        let gate = allocate(experts * hidden)?;
        let up = allocate(experts * hidden)?;
        let down = allocate(experts * dim)?;
        let shared_values = buffer(self, bytemuck::cast_slice(shared_values))?;
        let shared_up = buffer(self, bytemuck::cast_slice(shared_up))?;
        let shared_down = allocate(dim)?;
        let probabilities_buffer = buffer(self, bytemuck::cast_slice(probabilities))?;
        let output = allocate(dim)?;
        let failure = buffer(self, bytemuck::bytes_of(&0u32))?;
        drop(_upload);
        profile::record("native.metal.allocated_bytes", 0.0, self.device.current_allocated_size());
        let execute = profile::Scope::new("native.metal.execute");
        let encode = profile::Scope::new("native.metal.encode");
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal FFN command allocation failed"))?;
        let concurrent = std::env::var_os("STARK_HYBRID_CONCURRENT_FFN").is_none_or(|value| value != "0");
        let encoding = command
            .compute_command_encoder_with_dispatch_type(if concurrent {
                MTLDispatchType::CONCURRENT
            } else {
                MTLDispatchType::SERIAL
            })
            .ok_or(Error::Invalid("Metal FFN encoder allocation failed"))?;
        if concurrent {
            profile::record("native.metal.ffn.concurrent", 0.0, 0);
        }
        Self::encode_ffn_matrix(&encoding, &operations[0], &parameters[0], &pipelines[0], &input, &gate);
        Self::encode_ffn_matrix(&encoding, &operations[1], &parameters[1], &pipelines[1], &input, &up);
        // Every projection reads only immutable weights/inputs and writes a
        // disjoint expert range. Activation consumes all gate/up outputs.
        if concurrent {
            encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        }
        encoding.set_compute_pipeline_state(&activation_pipeline);
        for (index, value) in [&gate, &up, &shared_values, &shared_up].into_iter().enumerate() {
            encoding.set_buffer(Some(value), 0, index);
        }
        let sizes = [(experts * hidden) as u32, shared_dim as u32, 0, 0];
        let limits = clamp.map_or([0.0, 0.0], |(routed, shared)| [routed, shared]);
        encoding.set_bytes(NonNull::from(&sizes).cast(), std::mem::size_of_val(&sizes), 4);
        encoding.set_bytes(NonNull::from(&limits).cast(), std::mem::size_of_val(&limits), 5);
        encoding.set_buffer(Some(&failure), 0, 6);
        encoding.dispatch_threads(
            MTLSize {
                width: (experts * hidden).max(shared_dim),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        // Activation writes gate/shared_values in place; both down paths read
        // those buffers. Expert and shared down projections are independent.
        if concurrent {
            encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        }
        Self::encode_ffn_matrix(&encoding, &operations[2], &parameters[2], &pipelines[2], &gate, &down);
        Self::encode_ffn_matrix(&encoding, &operations[3], &parameters[3], &pipelines[3], &shared_values, &shared_down);
        // The ordered mixture must see every routed/shared down result. The
        // status buffer uses atomics and is read by the host after completion.
        if concurrent {
            encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        }
        encoding.set_compute_pipeline_state(&reduction_pipeline);
        for (index, value) in [&down, &shared_down, &probabilities_buffer, &output].into_iter().enumerate() {
            encoding.set_buffer(Some(value), 0, index);
        }
        let sizes = [dim as u32, experts as u32];
        encoding.set_bytes(NonNull::from(&sizes).cast(), std::mem::size_of_val(&sizes), 4);
        encoding.set_bytes(NonNull::from(&shared_gate).cast(), std::mem::size_of_val(&shared_gate), 5);
        encoding.set_buffer(Some(&failure), 0, 6);
        encoding.dispatch_threads(
            MTLSize {
                width: dim,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoding.end_encoding();
        drop(encode);
        let submit = profile::Scope::new("native.metal.submit");
        command.commit();
        drop(submit);
        let wait = profile::Scope::new("native.metal.wait");
        command.wait_until_completed();
        drop(wait);
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal FFN execution failed"));
        }
        drop(execute);
        profile::record("native.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
        let _readback = profile::Scope::bytes("native.metal.readback", dim * 4);
        // SAFETY: the checked dim-sized shared output has completed its sole
        // writer. All intermediate and mapped buffers remain owned above.
        let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), dim) }.to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("nonfinite Metal FFN output"));
        }
        // SAFETY: this initialized atomic status is read only after all GPU
        // writers complete. Preserve intermediate failure checks even where
        // clipping could otherwise conceal an overflowed projection.
        if unsafe { *failure.contents().as_ptr().cast::<u32>() } != 0 {
            return Err(Error::Invalid("nonfinite Metal FFN intermediate"));
        }
        Ok(values)
    }

    fn pipeline(
        &self,
        format: MatrixFormat,
    ) -> Result<PipelineState> {
        let name = match format {
            MatrixFormat::Packed(13 | 14) => "hybrid_packed_block_matvec",
            MatrixFormat::Packed(17 | 18) => "hybrid_iq_matvec_vector8",
            MatrixFormat::Packed(_) => "hybrid_packed_matvec",
            MatrixFormat::Affine {
                ..
            } => "hybrid_affine_matvec_vector8",
        };
        self.cached_pipeline(format, name, &self.pipelines)
    }

    #[cfg(test)]
    fn pipeline_with_tiling(
        &self,
        format: MatrixFormat,
        tiled: bool,
    ) -> Result<PipelineState> {
        let (name, cache) = match (format, tiled) {
            (MatrixFormat::Packed(_), false) => ("hybrid_packed_matvec", &self.scalar_pipelines),
            (
                MatrixFormat::Affine {
                    ..
                },
                false,
            ) => ("hybrid_affine_matvec", &self.scalar_pipelines),
            (MatrixFormat::Packed(_), true) => ("hybrid_packed_matvec_tiled", &self.tiled_pipelines),
            (
                MatrixFormat::Affine {
                    ..
                },
                true,
            ) => ("hybrid_affine_matvec_tiled", &self.tiled_pipelines),
        };
        self.cached_pipeline(format, name, cache)
    }

    fn cached_pipeline(
        &self,
        format: MatrixFormat,
        name: &str,
        cache: &PipelineCache,
    ) -> Result<PipelineState> {
        // The bounded key space describes code only: 13 GGML kinds and 18 MLX
        // affine layouts. No model weights, inputs or decisions enter this
        // process-local shader cache. The tiled experiment is test-only.
        let mut pipelines = cache.lock().map_err(|_| Error::Invalid("Metal pipeline cache failed"))?;
        if let Some(pipeline) = pipelines.get(&format) {
            return Ok(pipeline.clone());
        }
        let _compile = profile::Scope::new("native.metal.pipeline");
        let values = MTLFunctionConstantValues::new();
        let set = |index, value: u32| {
            values.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
        };
        match format {
            MatrixFormat::Packed(kind) => {
                let (block, bytes) = weights::block_layout(kind)?;
                set(0, kind);
                set(4, block as u32);
                set(5, bytes as u32);
            },
            MatrixFormat::Affine {
                scale_dtype,
                bits,
                group,
            } => {
                set(1, bits as u32);
                set(2, group as u32);
                set(3, scale_dtype);
            },
        };
        let function = self
            .library
            .new_function_with_name_constant_values(name, &values)
            .map_err(|_| Error::Invalid("Metal specialized function failed"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| Error::Invalid("Metal specialized pipeline failed"))?;
        // Retain the common 256-thread capability admission; block/lane GEMV
        // submits 64 threads while the other schedules submit 256.
        if pipeline.thread_execution_width() != 32 || pipeline.max_total_threads_per_threadgroup() < 256 {
            return Err(Error::Unsupported("Metal matrix execution width is unsupported"));
        }
        pipelines.insert(format, pipeline.clone());
        Ok(pipeline)
    }

    fn run_mapped(
        &self,
        operations: &[MappedOperation<'_>],
        parameters: &[[u32; 8]],
    ) -> Result<Vec<Vec<f32>>> {
        let mut working = Vec::with_capacity(operations.len());
        {
            let input_bytes = operations.iter().map(|operation| operation.inputs.len() * 4).sum();
            let _upload = profile::Scope::bytes("native.metal.upload", input_bytes);
            for operation in operations {
                let input = buffer(self, bytemuck::cast_slice(operation.inputs))?;
                let output = self
                    .device
                    .new_buffer(operation.rows * operation.matrices.len() * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                    .ok_or(Error::Invalid("Metal readout allocation failed"))?;
                working.push((input, output));
            }
        }
        let pipelines =
            operations.iter().map(|operation| self.pipeline(operation.format)).collect::<Result<Vec<_>>>()?;
        profile::record("native.metal.allocated_bytes", 0.0, self.device.current_allocated_size());
        let execute = profile::Scope::new("native.metal.execute");
        let encode = profile::Scope::new("native.metal.encode");
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal command allocation failed"))?;
        let encoding = command.compute_command_encoder().ok_or(Error::Invalid("Metal encoder allocation failed"))?;
        for (index, operation) in operations.iter().enumerate() {
            let parameters = &parameters[index];
            let (input, output) = &working[index];
            encoding.set_compute_pipeline_state(&pipelines[index]);
            // Metal copies these 32 bytes while encoding; the command never
            // borrows stack parameter storage after this call.
            encoding.set_bytes(NonNull::from(parameters).cast(), std::mem::size_of_val(parameters), 3);
            for (matrix_index, matrix) in operation.matrices.iter().enumerate() {
                encoding.set_buffer(Some(&matrix.codes.buffer), 0, 0);
                encoding.set_buffer(Some(input), matrix_index * operation.columns * 4, 1);
                encoding.set_buffer(Some(output), matrix_index * operation.rows * 4, 2);
                let mut offsets = [matrix.codes.offset as u32, 0, 0, 0];
                if let Some((scales, biases)) = &matrix.planes {
                    encoding.set_buffer(Some(&scales.buffer), 0, 4);
                    encoding.set_buffer(Some(&biases.buffer), 0, 5);
                    offsets[1] = scales.offset as u32;
                    offsets[2] = biases.offset as u32;
                }
                encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
                encoding.dispatch_threadgroups(
                    MTLSize {
                        width: matrix_groups(operation.format, operation.rows),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: matrix_threads(operation.format),
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        encoding.end_encoding();
        drop(encode);
        let submit = profile::Scope::new("native.metal.submit");
        command.commit();
        drop(submit);
        let wait = profile::Scope::new("native.metal.wait");
        command.wait_until_completed();
        drop(wait);
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal matrix execution failed"));
        }
        drop(execute);
        profile::record("native.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
        let readback_bytes = operations.iter().map(|operation| operation.rows * operation.matrices.len() * 4).sum();
        let _readback = profile::Scope::bytes("native.metal.readback", readback_bytes);
        let mut results = Vec::with_capacity(operations.len());
        for (operation, (_, output)) in operations.iter().zip(&working) {
            // SAFETY: completed commands wrote exactly the checked output
            // length. Every mapped weight buffer retains its source mapping
            // through its deallocator until Metal releases the resource.
            let values = unsafe {
                std::slice::from_raw_parts(
                    output.contents().as_ptr().cast::<f32>(),
                    operation.rows * operation.matrices.len(),
                )
            }
            .to_vec();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("nonfinite Metal matrix output"));
            }
            results.push(values);
        }
        Ok(results)
    }

    #[cfg(test)]
    fn run(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.run_inner(codes, input, parameters, affine, false, true))
    }

    #[cfg(test)]
    fn run_with_tiling(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
        tiled: bool,
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.run_inner(codes, input, parameters, affine, tiled, false))
    }

    #[cfg(test)]
    fn run_inner(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
        tiled: bool,
        production: bool,
    ) -> Result<Vec<f32>> {
        // These bounded uploads retain compressed storage and are released only
        // after GPU completion. No unowned mmap page or rounded Rust allocation
        // is exposed to Metal through an unsafe no-copy buffer.
        let upload_bytes = codes.len()
            + input.len() * 4
            + parameters.len() * 4
            + affine.map_or(0, |(scales, biases)| scales.len() + biases.len());
        let (codes, input, params, output, planes) = {
            let _scope = profile::Scope::bytes("native.metal.upload", upload_bytes);
            let codes = buffer(self, codes)?;
            let input = buffer(self, bytemuck::cast_slice(input))?;
            let params = buffer(self, bytemuck::cast_slice(&parameters))?;
            let output = self
                .device
                .new_buffer(parameters[0] as usize * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal readout allocation failed"))?;
            let planes = affine
                .map(|(scales, biases)| Ok::<_, Error>((buffer(self, scales)?, buffer(self, biases)?)))
                .transpose()?;
            (codes, input, params, output, planes)
        };
        profile::record("native.metal.allocated_bytes", 0.0, self.device.current_allocated_size());
        let format = if planes.is_some() {
            MatrixFormat::Affine {
                scale_dtype: parameters[7],
                bits: parameters[5] as usize,
                group: parameters[6] as usize,
            }
        } else {
            MatrixFormat::Packed(parameters[2])
        };
        let pipeline = if production {
            self.pipeline(format)?
        } else {
            self.pipeline_with_tiling(format, tiled)?
        };
        let execute = profile::Scope::new("native.metal.execute");
        let command = self.queue.command_buffer().ok_or(Error::Invalid("Metal command allocation failed"))?;
        let encoding = command.compute_command_encoder().ok_or(Error::Invalid("Metal encoder allocation failed"))?;
        encoding.set_compute_pipeline_state(&pipeline);
        for (index, value) in [&codes, &input, &output, &params].into_iter().enumerate() {
            encoding.set_buffer(Some(value), 0, index);
        }
        if let Some((scales, biases)) = &planes {
            encoding.set_buffer(Some(scales), 0, 4);
            encoding.set_buffer(Some(biases), 0, 5);
        }
        let offsets = [0u32; 4];
        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
        encoding.dispatch_threadgroups(
            MTLSize {
                width: if production {
                    matrix_groups(format, parameters[0] as usize)
                } else if tiled {
                    (parameters[0] as usize).div_ceil(4)
                } else {
                    parameters[0] as usize
                },
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: if production {
                    matrix_threads(format)
                } else {
                    256
                },
                height: 1,
                depth: 1,
            },
        );
        encoding.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Metal matrix execution failed"));
        }
        drop(execute);
        profile::record("native.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
        let _readback = profile::Scope::bytes("native.metal.readback", parameters[0] as usize * 4);
        // SAFETY: the shared buffer has exactly rows f32 slots. The command has
        // completed and wrote each slot once before this immutable host read.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), parameters[0] as usize) }
                .to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("nonfinite Metal matrix output"));
        }
        Ok(values)
    }
}

#[cfg(test)]
fn dimensions(
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<()> {
    if rows == 0
        || columns == 0
        || rows > 8_000_000
        || columns > 8_000_000
        || input.len() != columns
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("invalid Metal matrix dimensions"));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn matvec(
    bytes: &[u8],
    kind: u32,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    dimensions(rows, columns, input)?;
    packed_matvec(bytes, kind, rows, columns, input, 0)
}

#[cfg(test)]
fn batch_dimensions(
    batches: usize,
    rows: usize,
    columns: usize,
    inputs: &[f32],
) -> Result<usize> {
    let total_rows = batches.checked_mul(rows).ok_or(Error::Invalid("batched Metal rows overflow"))?;
    if batches == 0
        || rows == 0
        || columns == 0
        || total_rows > 8_000_000
        || inputs.len() > 8_000_000
        || batches.checked_mul(columns) != Some(inputs.len())
        || inputs.iter().any(|value| !value.is_finite())
    {
        return Err(Error::Invalid("invalid batched Metal matrix dimensions"));
    }
    Ok(total_rows)
}

#[cfg(test)]
pub(super) fn batched_matvec(
    bytes: &[u8],
    kind: u32,
    batches: usize,
    rows: usize,
    columns: usize,
    inputs: &[f32],
) -> Result<Vec<f32>> {
    let total_rows = batch_dimensions(batches, rows, columns, inputs)?;
    packed_matvec(bytes, kind, total_rows, columns, inputs, rows)
}

#[cfg(test)]
fn packed_matvec(
    bytes: &[u8],
    kind: u32,
    rows: usize,
    columns: usize,
    input: &[f32],
    rows_per_batch: usize,
) -> Result<Vec<f32>> {
    let (block, block_bytes) = weights::block_layout(kind)?;
    if !matches!(kind, 0 | 1 | 30 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 23)
        || !columns.is_multiple_of(block)
        || rows.checked_mul(columns / block).and_then(|n| n.checked_mul(block_bytes)) != Some(bytes.len())
    {
        return Err(Error::Invalid("invalid packed Metal matrix size"));
    }
    executor()?.run(
        bytes,
        input,
        [rows as u32, columns as u32, kind, block as u32, block_bytes as u32, 0, rows_per_batch as u32, 0],
        None,
    )
}

#[cfg(test)]
pub(super) fn mlx_matvec(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
    scale_dtype: u32,
    bits: usize,
    group: usize,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    dimensions(rows, columns, input)?;
    affine_matvec(codes, scales, biases, scale_dtype, bits, group, rows, columns, input, 0)
}

#[cfg(test)]
pub(super) fn mlx_batched_matvec(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
    scale_dtype: u32,
    bits: usize,
    group: usize,
    batches: usize,
    rows: usize,
    columns: usize,
    inputs: &[f32],
) -> Result<Vec<f32>> {
    let total_rows = batch_dimensions(batches, rows, columns, inputs)?;
    affine_matvec(codes, scales, biases, scale_dtype, bits, group, total_rows, columns, inputs, rows)
}

#[cfg(test)]
fn affine_matvec(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
    scale_dtype: u32,
    bits: usize,
    group: usize,
    rows: usize,
    columns: usize,
    input: &[f32],
    rows_per_batch: usize,
) -> Result<Vec<f32>> {
    let item = match scale_dtype {
        0 => 4,
        1 | 30 => 2,
        _ => return Err(Error::Invalid("invalid affine scale dtype")),
    };
    if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64 | 128) {
        return Err(Error::Invalid("invalid affine Metal quantization"));
    }
    let plane_size = rows.checked_mul(columns / group).and_then(|n| n.checked_mul(item));
    if !columns.is_multiple_of(group)
        || rows.checked_mul(columns).and_then(|n| n.checked_mul(bits)).map(|n| n / 8) != Some(codes.len())
        || plane_size != Some(scales.len())
        || scales.len() != biases.len()
    {
        return Err(Error::Invalid("invalid affine Metal matrix size"));
    }
    executor()?.run(
        codes,
        input,
        [rows as u32, columns as u32, 0, rows_per_batch as u32, 0, bits as u32, group as u32, scale_dtype],
        Some((scales, biases)),
    )
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn ffn_test_matrix(
        rows: usize,
        columns: usize,
        seed: usize,
    ) -> MappedMatrix {
        let values: Vec<f32> =
            (0..rows * columns).map(|index| ((index * 17 + seed * 31) as f32 * 0.073).sin() * 0.5).collect();
        let bytes = bytemuck::cast_slice(&values);
        MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), bytes).unwrap(),
                offset: 0,
                length: bytes.len(),
                registered_length: bytes.len(),
            },
            planes: None,
        }
    }

    #[stark_test]
    fn native_ffn_keeps_projection_and_ordered_mixture_numerics() {
        let experts = 3;
        let dim = 64;
        let hidden = 32;
        let shared = 32;
        let input: Vec<f32> = (0..dim).map(|index| (index as f32 * 0.11).cos()).collect();
        let inputs = input.repeat(experts);
        let down_inputs = vec![0.0; experts * hidden];
        let shared_inputs = vec![0.0; shared];
        let shared_values: Vec<f32> = (0..shared).map(|index| (index as f32 * 0.3).sin() * 3.0).collect();
        let shared_up: Vec<f32> = (0..shared).map(|index| (index as f32 * 0.17).cos() * 2.0).collect();
        let make = |rows, columns, count: usize, seed, inputs| MappedOperation {
            matrices: (0..count).map(|expert| ffn_test_matrix(rows, columns, seed + expert)).collect(),
            inputs,
            rows,
            columns,
            rows_per_batch: 0,
            format: MatrixFormat::Packed(0),
        };
        let mut operations = [
            make(hidden, dim, experts, 0, inputs.as_slice()),
            make(hidden, dim, experts, 3, inputs.as_slice()),
            make(dim, hidden, experts, 6, down_inputs.as_slice()),
            make(dim, shared, 1, 9, shared_inputs.as_slice()),
        ];
        let probabilities = [0.7, 0.1, 0.2];
        for clamp in [None, Some((0.25f32, 0.4f32))] {
            let actual = mapped_ffn(&operations, &probabilities, 0.63, &shared_values, &shared_up, clamp).unwrap();
            let projections = mapped_many(&operations[..2]).unwrap();
            let activate = |gate: f32, up: f32, limit: Option<f32>| {
                let (gate, up) = limit.map_or((gate, up), |limit| (gate.min(limit), up.clamp(-limit, limit)));
                super::super::math::silu(gate) * up
            };
            let activated: Vec<f32> = projections[0]
                .iter()
                .zip(&projections[1])
                .map(|(&gate, &up)| activate(gate, up, clamp.map(|limits| limits.0)))
                .collect();
            let shared_activated: Vec<f32> = shared_values
                .iter()
                .zip(&shared_up)
                .map(|(&gate, &up)| activate(gate, up, clamp.map(|limits| limits.1)))
                .collect();
            let down_operation = MappedOperation {
                matrices: operations[2].matrices.clone(),
                inputs: &activated,
                rows: dim,
                columns: hidden,
                rows_per_batch: 0,
                format: MatrixFormat::Packed(0),
            };
            let shared_operation = MappedOperation {
                matrices: operations[3].matrices.clone(),
                inputs: &shared_activated,
                rows: dim,
                columns: shared,
                rows_per_batch: 0,
                format: MatrixFormat::Packed(0),
            };
            let projected = mapped_many(&[down_operation, shared_operation]).unwrap();
            for (channel, &value) in actual.iter().enumerate() {
                let mut expected = 0.0;
                for (expert, probability) in probabilities.iter().enumerate() {
                    expected += probability * projected[0][expert * dim + channel];
                }
                expected += 0.63 * projected[1][channel];
                assert!((value - expected).abs() < 2e-5, "native FFN exceeded numerical tolerance");
            }
        }
        assert!(mapped_ffn(&operations, &[0.5, 0.5], 0.63, &shared_values, &shared_up, None).is_err());
        assert!(mapped_ffn(&operations, &probabilities, f32::NAN, &shared_values, &shared_up, None).is_err());
        assert!(mapped_ffn(&operations, &probabilities, 0.63, &shared_values, &shared_up, Some((0.0, 1.0))).is_err());
        // Overflowed projections must fail even when clipping could otherwise
        // conceal their nonfinite raw gates before the final readout.
        let overflow = vec![f32::MAX; hidden * dim];
        operations[0].matrices[0].codes.buffer = buffer(executor().unwrap(), bytemuck::cast_slice(&overflow)).unwrap();
        assert!(mapped_ffn(&operations, &probabilities, 0.63, &shared_values, &shared_up, Some((0.25, 0.4))).is_err());
    }

    #[stark_test]
    fn native_qwen_hyperconnection_matches_separate_projections() {
        let (dim, streams, rank) = (16, 4, 8);
        let normalized: Vec<f32> = (0..dim * streams).map(|index| (index as f32 * 0.31).sin() * 2.0).collect();
        let placeholder = vec![0.0; rank];
        for combine in [false, true] {
            let mut operations = vec![
                MappedOperation {
                    matrices: vec![ffn_test_matrix(rank, dim * streams, 13)],
                    inputs: &normalized,
                    rows: rank,
                    columns: dim * streams,
                    rows_per_batch: 0,
                    format: MatrixFormat::Packed(0),
                },
                MappedOperation {
                    matrices: vec![ffn_test_matrix(dim * streams, rank, 19)],
                    inputs: &placeholder,
                    rows: dim * streams,
                    columns: rank,
                    rows_per_batch: 0,
                    format: MatrixFormat::Packed(0),
                },
            ];
            if combine {
                operations.push(MappedOperation {
                    matrices: vec![ffn_test_matrix(streams, dim * streams, 23)],
                    inputs: &normalized,
                    rows: streams,
                    columns: dim * streams,
                    rows_per_batch: 0,
                    format: MatrixFormat::Packed(0),
                });
            }
            let (actual, actual_inject) = mapped_qwen_hyperconnection(&operations, &normalized, streams).unwrap();
            let reduced: Vec<f32> = mapped_many(&operations[..1])
                .unwrap()
                .remove(0)
                .iter()
                .map(|value| super::super::math::silu(*value / streams as f32))
                .collect();
            let up = MappedOperation {
                matrices: operations[1].matrices.clone(),
                inputs: &reduced,
                rows: dim * streams,
                columns: rank,
                rows_per_batch: 0,
                format: MatrixFormat::Packed(0),
            };
            let gates = mapped_many(&[up]).unwrap().remove(0);
            for channel in 0..dim {
                let mut expected = 0.0;
                for stream in 0..streams {
                    let index = stream * dim + channel;
                    expected += super::super::math::sigmoid(gates[index]) * normalized[index] / streams as f32;
                }
                assert!(
                    (actual[channel] - expected).abs() < 2e-6,
                    "native hyperconnection exceeded numerical tolerance"
                );
            }
            if combine {
                let injected = mapped_many(&operations[2..]).unwrap().remove(0);
                let expected: Vec<f32> =
                    injected.iter().map(|value| 2.0 * super::super::math::sigmoid(*value / streams as f32)).collect();
                assert!(actual_inject.iter().zip(expected).all(|(actual, expected)| (actual - expected).abs() < 2e-6));
            } else {
                assert!(actual_inject.is_empty());
            }
            assert!(mapped_qwen_hyperconnection(&operations, &normalized, 0).is_err());
            assert!(mapped_qwen_hyperconnection(&operations, &normalized[..normalized.len() - 1], streams).is_err());
        }
    }

    #[stark_test]
    fn power_of_two_shader_addresses_preserve_wide_row_offsets() {
        // Exercise the address algebra independently of any allocation. Large
        // admitted row/column products exceed u32 even though each axis fits.
        for kind in [0, 1, 30, 8, 10, 11, 12, 13, 14, 16, 17, 18, 23] {
            let (block, bytes) = weights::block_layout(kind).unwrap();
            let shift = block.trailing_zeros();
            for row in [0u64, 65_536, 7_999_999] {
                for columns in [block as u64, block as u64 * 33, 8_000_000 / block as u64 * block as u64] {
                    for column in [0, columns / 2, columns - 1] {
                        let old = ((row * columns + column) / block as u64) * bytes as u64;
                        let shifted = row * (columns >> shift) * bytes as u64 + (column >> shift) * bytes as u64;
                        assert_eq!(old, shifted);
                        assert_eq!(column % block as u64, column & (block as u64 - 1));
                    }
                }
            }
        }
        for bits in [4u64, 8] {
            for group in [32u64, 64, 128] {
                let packed_shift = u32::from(bits == 4);
                let group_shift = group.trailing_zeros();
                for row in [0u64, 65_536, 7_999_999] {
                    for columns in [group, group * 33, 8_000_000 / group * group] {
                        for column in [0, columns / 2, columns - 1] {
                            let element = row * columns + column;
                            assert_eq!(
                                element * bits / 8,
                                ((row * columns) >> packed_shift) + (column >> packed_shift)
                            );
                            assert_eq!(element / group, ((row * columns) >> group_shift) + (column >> group_shift));
                            assert_eq!(
                                (column * bits) % 8,
                                if bits == 4 {
                                    (column & 1) * 4
                                } else {
                                    0
                                }
                            );
                        }
                    }
                }
            }
        }
    }

    #[stark_test]
    fn completed_matrix_submissions_release_their_uploaded_weights() {
        let rows = 1024;
        let columns = 1024;
        let bytes = vec![0u8; rows * columns / 32 * 34];
        let input = vec![1.0; columns];
        matvec(&bytes, 8, rows, columns, &input).unwrap();
        let device = &executor().unwrap().device;
        let before = device.current_allocated_size();
        for _ in 0..32 {
            assert_eq!(matvec(&bytes, 8, rows, columns, &input).unwrap().len(), rows);
        }
        let after = device.current_allocated_size();
        // Allow driver scratch/reuse, but reject retaining a new compressed
        // matrix for every completed command in an undrained autorelease pool.
        assert!(after <= before + 16 * 1024 * 1024);
    }

    #[stark_test]
    fn metal_matches_all_checkpoint_quantization_formats() {
        for kind in [0, 1, 30, 8, 10, 11, 12, 13, 14, 16, 17, 18, 23] {
            let (block, size) = weights::block_layout(kind).unwrap();
            let columns = block.max(256);
            let mut bytes = Vec::new();
            for row in 0..3 {
                for number in 0..columns / block {
                    let mut source: Vec<u8> =
                        (0..size).map(|i| ((i * 37 + number * 13 + row * 23) % 256) as u8).collect();
                    let half_offsets: &[usize] = match kind {
                        1 | 8 | 16 | 17 | 18 | 23 => &[0],
                        10 => &[80, 82],
                        11 => &[108],
                        12 | 13 => &[0, 2],
                        14 => &[208],
                        _ => &[],
                    };
                    for &at in half_offsets {
                        source[at..at + 2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
                    }
                    if kind == 0 {
                        source.copy_from_slice(&(0.01 * (number + row + 1) as f32).to_le_bytes());
                    }
                    if kind == 30 {
                        source.copy_from_slice(&half::bf16::from_f32(0.125).to_bits().to_le_bytes());
                    }
                    bytes.extend(source);
                }
            }
            let input: Vec<f32> = (0..columns).map(|i| (i as f32 * 0.17).sin()).collect();
            let actual = matvec(&bytes, kind, 3, columns, &input).unwrap();
            let tiled = executor()
                .unwrap()
                .run_with_tiling(
                    &bytes,
                    &input,
                    [3, columns as u32, kind, block as u32, size as u32, 0, 0, 0],
                    None,
                    true,
                )
                .unwrap();
            let scalar = executor()
                .unwrap()
                .run_with_tiling(
                    &bytes,
                    &input,
                    [3, columns as u32, kind, block as u32, size as u32, 0, 0, 0],
                    None,
                    false,
                )
                .unwrap();
            assert_eq!(tiled, scalar, "tiled packed tail kind={kind}");
            let factors = [1.0f32, -0.5, 2.0];
            let batch_inputs: Vec<f32> =
                factors.iter().flat_map(|factor| input.iter().map(move |x| factor * x)).collect();
            let batched = batched_matvec(&bytes, kind, 3, 1, columns, &batch_inputs).unwrap();
            let repeated_codes = bytes.repeat(3);
            let batched_nine = batched_matvec(&repeated_codes, kind, 3, 3, columns, &batch_inputs).unwrap();
            let tiled_nine = executor()
                .unwrap()
                .run_with_tiling(
                    &repeated_codes,
                    &batch_inputs,
                    [9, columns as u32, kind, block as u32, size as u32, 0, 3, 0],
                    None,
                    true,
                )
                .unwrap();
            let scalar_nine = executor()
                .unwrap()
                .run_with_tiling(
                    &repeated_codes,
                    &batch_inputs,
                    [9, columns as u32, kind, block as u32, size as u32, 0, 3, 0],
                    None,
                    false,
                )
                .unwrap();
            assert_eq!(tiled_nine, scalar_nine, "tiled packed batch crossing kind={kind}");
            let row_size = bytes.len() / 3;
            for row in 0..3 {
                let mut expected = 0.0f64;
                let mut absolute = 0.0f64;
                for (source, x) in
                    bytes[row * row_size..(row + 1) * row_size].chunks_exact(size).zip(input.chunks_exact(block))
                {
                    let mut values = vec![0.0; block];
                    weights::decode_block(kind, source, &mut values).unwrap();
                    for (value, input) in values.iter().zip(x) {
                        let p = *value as f64 * *input as f64;
                        expected += p;
                        absolute += p.abs();
                    }
                }
                assert!((actual[row] as f64 - expected).abs() <= 2e-5 * absolute.max(1.0), "format {kind}");
                assert!(
                    (f64::from(batched[row]) - expected * f64::from(factors[row])).abs()
                        <= 2e-5 * (absolute * f64::from(factors[row]).abs()).max(1.0),
                    "batched format {kind}"
                );
                for batch in 0..3 {
                    assert!(
                        (f64::from(batched_nine[batch * 3 + row]) - expected * f64::from(factors[batch])).abs()
                            <= 2e-5 * (absolute * f64::from(factors[batch]).abs()).max(1.0),
                        "crossing batch format {kind}"
                    );
                }
            }
        }
    }

    fn float_bytes(
        kind: u32,
        value: f32,
    ) -> Vec<u8> {
        match kind {
            0 => value.to_le_bytes().to_vec(),
            1 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
            30 => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
            _ => unreachable!(),
        }
    }

    fn float_roundtrip(
        kind: u32,
        value: f32,
    ) -> f32 {
        match kind {
            0 => value,
            1 => half::f16::from_f32(value).to_f32(),
            30 => half::bf16::from_f32(value).to_f32(),
            _ => unreachable!(),
        }
    }

    #[stark_test]
    #[ignore]
    fn row_tile_gpu_component_timings() {
        struct DisableProfile;
        impl Drop for DisableProfile {
            fn drop(&mut self) {
                profile::enable(false);
            }
        }
        let _disable = DisableProfile;
        let cases = [
            ("f32", MatrixFormat::Packed(0), 2048usize, 4096usize),
            ("bf16", MatrixFormat::Packed(30), 2048, 4096),
            ("q8_0", MatrixFormat::Packed(8), 2048, 4096),
            ("iq2_xs", MatrixFormat::Packed(17), 2048, 4096),
            ("iq3_xxs", MatrixFormat::Packed(18), 2048, 4096),
            (
                "affine4_bf16_g64",
                MatrixFormat::Affine {
                    scale_dtype: 30,
                    bits: 4,
                    group: 64,
                },
                2048,
                4096,
            ),
            (
                "affine8_bf16_g64",
                MatrixFormat::Affine {
                    scale_dtype: 30,
                    bits: 8,
                    group: 64,
                },
                2048,
                4096,
            ),
            (
                "qwen_gateup_affine4",
                MatrixFormat::Affine {
                    scale_dtype: 30,
                    bits: 4,
                    group: 64,
                },
                640,
                2560,
            ),
            (
                "qwen_down_affine4",
                MatrixFormat::Affine {
                    scale_dtype: 30,
                    bits: 4,
                    group: 64,
                },
                2560,
                640,
            ),
        ];
        for (name, format, rows, columns) in cases {
            let input: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.013).sin()).collect();
            let (codes, scales, biases, parameters) = match format {
                MatrixFormat::Packed(kind) => {
                    let (block, bytes) = weights::block_layout(kind).unwrap();
                    let mut codes = vec![0u8; rows * columns / block * bytes];
                    if matches!(kind, 8 | 17 | 18) {
                        for encoded in codes.chunks_exact_mut(bytes) {
                            encoded[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
                        }
                    }
                    (
                        codes,
                        Vec::new(),
                        Vec::new(),
                        [rows as u32, columns as u32, kind, block as u32, bytes as u32, 0, 0, 0],
                    )
                },
                MatrixFormat::Affine {
                    scale_dtype,
                    bits,
                    group,
                } => {
                    let codes = vec![0x21u8; rows * columns * bits / 8];
                    let scales = float_bytes(scale_dtype, 0.125).repeat(rows * columns / group);
                    let biases = float_bytes(scale_dtype, -0.0625).repeat(rows * columns / group);
                    (
                        codes,
                        scales,
                        biases,
                        [rows as u32, columns as u32, 0, 0, 0, bits as u32, group as u32, scale_dtype],
                    )
                },
            };
            let planes = (!scales.is_empty()).then_some((scales.as_slice(), biases.as_slice()));
            let run = |tiled| executor().unwrap().run_with_tiling(&codes, &input, parameters, planes, tiled).unwrap();
            let plain = run(false);
            assert_eq!(run(true), plain, "component tiled equality layout={name}");
            for (tiled, schedule) in [(false, "single"), (true, "mr4")] {
                profile::reset();
                profile::enable(true);
                let started = std::time::Instant::now();
                for _ in 0..8 {
                    assert_eq!(run(tiled), plain);
                }
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0 / 8.0;
                profile::enable(false);
                let counters = profile::snapshot();
                let gpu_ms = counters["native.metal.gpu"].seconds * 1000.0 / 8.0;
                eprintln!(
                    "native_hybrid_row_tile layout={name} schedule={schedule} rows={rows} columns={columns} gpu_ms={gpu_ms:.6} wall_ms={wall_ms:.6}"
                );
            }
        }
    }

    #[stark_test]
    fn metal_affine_four_and_eight_bit_planes_match_scalar() {
        for bits in [4, 8] {
            for group in [32, 64, 128] {
                for dtype in [0, 1, 30] {
                    let rows = 3;
                    let columns = 3 * group;
                    let mut codes = vec![0u8; rows * columns * bits / 8];
                    let mut scales = Vec::new();
                    let mut biases = Vec::new();
                    let mut decoded = vec![0.0f32; rows * columns];
                    for row in 0..rows {
                        for block in 0..3 {
                            let scale = 0.123 + 0.02 * (row * 3 + block) as f32;
                            let bias = -0.07 * (row * 3 + block + 1) as f32;
                            scales.extend(float_bytes(dtype, scale));
                            biases.extend(float_bytes(dtype, bias));
                            for column in block * group..(block + 1) * group {
                                let q = ((row * 31 + column * 7) % (1usize << bits)) as u8;
                                let element = row * columns + column;
                                codes[element * bits / 8] |= q << ((column * bits) % 8);
                                decoded[element] =
                                    float_roundtrip(dtype, scale) * f32::from(q) + float_roundtrip(dtype, bias);
                            }
                        }
                    }
                    let input: Vec<f32> = (0..columns).map(|i| (i as f32 * 0.13).cos()).collect();
                    let actual =
                        mlx_matvec(&codes, &scales, &biases, dtype, bits, group, rows, columns, &input).unwrap();
                    let tiled = executor()
                        .unwrap()
                        .run_with_tiling(
                            &codes,
                            &input,
                            [rows as u32, columns as u32, 0, 0, 0, bits as u32, group as u32, dtype],
                            Some((&scales, &biases)),
                            true,
                        )
                        .unwrap();
                    let scalar = executor()
                        .unwrap()
                        .run_with_tiling(
                            &codes,
                            &input,
                            [rows as u32, columns as u32, 0, 0, 0, bits as u32, group as u32, dtype],
                            Some((&scales, &biases)),
                            false,
                        )
                        .unwrap();
                    assert_eq!(tiled, scalar, "tiled affine tail bits={bits} group={group} dtype={dtype}");
                    let factors = [1.0f32, -0.5, 2.0];
                    let batch_inputs: Vec<f32> =
                        factors.iter().flat_map(|factor| input.iter().map(move |x| factor * x)).collect();
                    let batched =
                        mlx_batched_matvec(&codes, &scales, &biases, dtype, bits, group, 3, 1, columns, &batch_inputs)
                            .unwrap();
                    let repeated_codes = codes.repeat(3);
                    let repeated_scales = scales.repeat(3);
                    let repeated_biases = biases.repeat(3);
                    let batched_nine = mlx_batched_matvec(
                        &repeated_codes,
                        &repeated_scales,
                        &repeated_biases,
                        dtype,
                        bits,
                        group,
                        3,
                        3,
                        columns,
                        &batch_inputs,
                    )
                    .unwrap();
                    let tiled_nine = executor()
                        .unwrap()
                        .run_with_tiling(
                            &repeated_codes,
                            &batch_inputs,
                            [9, columns as u32, 0, 3, 0, bits as u32, group as u32, dtype],
                            Some((&repeated_scales, &repeated_biases)),
                            true,
                        )
                        .unwrap();
                    let scalar_nine = executor()
                        .unwrap()
                        .run_with_tiling(
                            &repeated_codes,
                            &batch_inputs,
                            [9, columns as u32, 0, 3, 0, bits as u32, group as u32, dtype],
                            Some((&repeated_scales, &repeated_biases)),
                            false,
                        )
                        .unwrap();
                    assert_eq!(
                        tiled_nine, scalar_nine,
                        "tiled affine batch crossing bits={bits} group={group} dtype={dtype}"
                    );
                    for row in 0..rows {
                        let products: Vec<f64> = decoded[row * columns..(row + 1) * columns]
                            .iter()
                            .zip(&input)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .collect();
                        let expected: f64 = products.iter().sum();
                        let absolute: f64 = products.iter().map(|x| x.abs()).sum();
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-5 * absolute.max(1.0),
                            "affine bits={bits} group={group} dtype={dtype}"
                        );
                        assert!(
                            (f64::from(batched[row]) - expected * f64::from(factors[row])).abs()
                                <= 2e-5 * (absolute * f64::from(factors[row]).abs()).max(1.0),
                            "batched affine bits={bits} group={group} dtype={dtype}"
                        );
                        for batch in 0..3 {
                            assert!(
                                (f64::from(batched_nine[batch * 3 + row]) - expected * f64::from(factors[batch])).abs()
                                    <= 2e-5 * (absolute * f64::from(factors[batch]).abs()).max(1.0),
                                "crossing batch affine bits={bits} group={group} dtype={dtype}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[stark_test]
    fn invalid_affine_geometry_is_rejected_before_gpu_submission() {
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 0, 1, 32, &[0.0; 32]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 0, 32, 1, 32, &[0.0; 32]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 64, 1, 32, &[0.0; 32]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 32, 1, 33, &[0.0; 33]).is_err());
        assert!(mlx_matvec(&[0; 16], &[0; 4], &[0; 4], 0, 4, 32, 1, 32, &[f32::NAN; 32]).is_err());
        assert!(batched_matvec(&[0; 68], 8, 0, 1, 32, &[]).is_err());
        assert!(batched_matvec(&[0; 68], 8, 2, 1, 32, &[0.0; 32]).is_err());
        assert!(batched_matvec(&[0; 68], 8, usize::MAX, 2, 32, &[]).is_err());
        assert!(mlx_batched_matvec(&[0; 32], &[0; 8], &[0; 8], 0, 4, 0, 2, 1, 32, &[0.0; 64]).is_err());
        for kind in [101, 102, 103] {
            assert!(matvec(&[0; 8], kind, 1, 1, &[0.0]).is_err());
            assert!(batched_matvec(&[0; 8], kind, 1, 1, 1, &[0.0]).is_err());
        }
    }

    #[stark_test]
    fn parsed_affine_expert_slices_preserve_independent_planes() {
        let directory = std::env::temp_dir().join(format!(
            "stark-metal-expert-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut payload = Vec::new();
        let codes = vec![0x32u8; 2 * 3 * 16];
        payload.extend(&codes);
        let scale_start = payload.len();
        let scales: Vec<f32> = (0..6).map(|i| 0.125 * (i + 1) as f32).collect();
        for scale in &scales {
            payload.extend(scale.to_le_bytes());
        }
        let bias_start = payload.len();
        let biases: Vec<f32> = (0..6).map(|i| -0.25 * i as f32).collect();
        for bias in &biases {
            payload.extend(bias.to_le_bytes());
        }
        let mut header = serde_json::to_vec(&serde_json::json!({
            "experts.weight":{"dtype":"U32","shape":[2,3,4],"data_offsets":[0,scale_start]},
            "experts.scales":{"dtype":"F32","shape":[2,3,1],"data_offsets":[scale_start,bias_start]},
            "experts.biases":{"dtype":"F32","shape":[2,3,1],"data_offsets":[bias_start,payload.len()]},
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(directory.join("model.safetensors"), file).unwrap();
        std::fs::write(directory.join("config.json"), b"{\"quantization\":{\"bits\":4,\"group_size\":32}}").unwrap();
        let weights = weights::Weights::open(&directory).unwrap();
        assert_eq!(weights.shape("experts.weight").unwrap(), &[2, 3, 32]);
        let input: Vec<f32> = (0..32).map(|i| 0.125 * (i as f32 - 16.0)).collect();
        for expert in 0..2 {
            let actual = weights.expert_matvec("experts.weight", expert, &input).unwrap();
            for row in 0..3 {
                let index = expert * 3 + row;
                let expected: f32 = input
                    .iter()
                    .enumerate()
                    .map(|(col, &x)| {
                        (scales[index]
                            * if col % 2 == 0 {
                                2.0
                            } else {
                                3.0
                            }
                            + biases[index])
                            * x
                    })
                    .sum();
                assert!((actual[row] - expected).abs() < 1e-5);
            }
        }
        let batch_inputs: Vec<f32> = input.iter().copied().chain(input.iter().map(|x| -0.5 * x)).collect();
        let batched = weights.batched_matvec("experts.weight", &batch_inputs).unwrap();
        assert_eq!(batched.len(), 6);
        for expert in 0..2 {
            let single =
                weights.expert_matvec("experts.weight", expert, &batch_inputs[expert * 32..(expert + 1) * 32]).unwrap();
            for row in 0..3 {
                assert_eq!(batched[expert * 3 + row], single[row]);
            }
        }
        drop(weights);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[stark_test]
    fn native_head_batching_matches_separate_dispatches() {
        let (batches, rows, columns) = (64, 512, 256);
        let row_bytes = columns / 32 * 34;
        let matrix_bytes = rows * row_bytes;
        let mut bytes = vec![0u8; batches * matrix_bytes];
        for (index, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
            for (i, code) in block[2..].iter_mut().enumerate() {
                *code = ((index * 7 + i * 3) % 255) as u8;
            }
        }
        let inputs: Vec<f32> = (0..batches * columns).map(|i| (i as f32 * 0.011).sin()).collect();
        // Warm compilation and allocation before measuring the two execution
        // shapes. Only aggregate timings are reported; fixtures are synthetic.
        let batched = batched_matvec(&bytes, 8, batches, rows, columns, &inputs).unwrap();
        let start = std::time::Instant::now();
        let mut separate = Vec::with_capacity(batches * rows);
        for batch in 0..batches {
            separate.extend(
                matvec(
                    &bytes[batch * matrix_bytes..(batch + 1) * matrix_bytes],
                    8,
                    rows,
                    columns,
                    &inputs[batch * columns..(batch + 1) * columns],
                )
                .unwrap(),
            );
        }
        let separate_elapsed = start.elapsed();
        assert_eq!(batched, separate);
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let result = batched_matvec(&bytes, 8, batches, rows, columns, &inputs).unwrap();
            assert_eq!(result, separate);
        }
        let batch_elapsed = start.elapsed() / 3;
        eprintln!(
            "native_hybrid_head_batching heads={batches} rows={rows} columns={columns} separate_ms={:.3} batched_ms={:.3}",
            separate_elapsed.as_secs_f64() * 1000.0,
            batch_elapsed.as_secs_f64() * 1000.0
        );
    }
}
