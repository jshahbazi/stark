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
pub(super) use decode_graph::{DecodeGraph, DeviceVector, PendingDecodeGraph, ScratchHeap};

#[cfg(test)]
#[path = "weight_metal_storage_tests.rs"]
mod storage_tests;

#[cfg(test)]
#[path = "weight_metal_q8_grouped_tests.rs"]
mod q8_grouped_tests;

#[cfg(test)]
#[path = "weight_metal_q4_grouped_tests.rs"]
mod q4_grouped_tests;

#[cfg(test)]
#[path = "weight_metal_kquant_mma_tests.rs"]
mod kquant_mma_tests;

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    packed_fast_library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: PipelineCache,
    q8_block_pipelines: PipelineCache,
    q8_grouped8_pipelines: PipelineCache,
    q8_segmented8_pipelines: PipelineCache,
    q6_grouped16_pipelines: PipelineCache,
    q5_grouped32_pipelines: PipelineCache,
    q4_grouped32_pipelines: PipelineCache,
    ffn_pipelines: Mutex<Option<(PipelineState, PipelineState)>>,
    hc_pipelines: Mutex<Option<(PipelineState, PipelineState)>>,
    gdn_pipelines: Mutex<Option<(PipelineState, PipelineState)>>,
    graph_pipelines: Mutex<HashMap<&'static str, PipelineState>>,
    indexed_pipelines: PipelineCache,
    iq_bound_indexed_pipelines: PipelineCache,
    iq_bound_typed_indexed_pipelines: PipelineCache,
    iq_bound_factor_indexed_pipelines: PipelineCache,
    checked_matrix_pipelines: PipelineCache,
    typed_matrix_pipelines: PipelineCache,
    typed_indexed_pipelines: PipelineCache,
    chunk32_matrix_pipelines: PipelineCache,
    chunk32_indexed_pipelines: PipelineCache,
    uint4_matrix_pipelines: PipelineCache,
    uint4_indexed_pipelines: PipelineCache,
    factored_matrix_pipelines: PipelineCache,
    factored_indexed_pipelines: PipelineCache,
    ordered_rms: bool,
    #[cfg(test)]
    tiled_pipelines: PipelineCache,
    #[cfg(test)]
    scalar_pipelines: PipelineCache,
    #[cfg(test)]
    block_test_pipelines: PipelineCache,
    #[cfg(test)]
    factored_test_pipelines: PipelineCache,
    #[cfg(test)]
    iq4_test_pipelines: PipelineCache,
}

static EXECUTOR: OnceLock<std::result::Result<Executor, &'static str>> = OnceLock::new();
static Q4K_BLOCK: OnceLock<bool> = OnceLock::new();
static Q4_GROUPED32: OnceLock<bool> = OnceLock::new();
static KQUANT_FACTORED: OnceLock<bool> = OnceLock::new();
static Q8_SHAPES: OnceLock<bool> = OnceLock::new();
static Q8_GROUPED8: OnceLock<bool> = OnceLock::new();
static IQ4_VECTOR: OnceLock<bool> = OnceLock::new();
static IQ_MULTIROW: OnceLock<bool> = OnceLock::new();
static IQ_GPU_BOUND: OnceLock<bool> = OnceLock::new();
static IQ_TYPED16: OnceLock<bool> = OnceLock::new();
static IQ_SIGNED_FACTOR: OnceLock<bool> = OnceLock::new();
static Q6_GROUPED16: OnceLock<bool> = OnceLock::new();
static Q5_GROUPED32: OnceLock<bool> = OnceLock::new();

pub(super) fn iq_bound_schedule(
    format: MatrixFormat,
    columns: usize,
) -> bool {
    matches!(format, MatrixFormat::Packed(17 | 18))
        && matches!(columns, 2048 | 4096)
        && *IQ_GPU_BOUND.get_or_init(|| std::env::var_os("STARK_GLM53_IQ_GPU_BOUND").is_some_and(|value| value == "1"))
}

pub(super) fn iq_bound_name(format: MatrixFormat) -> Result<&'static str> {
    match format {
        MatrixFormat::Packed(17) => Ok("hybrid_graph_iq_indexed_gpu_bound_iq2xs"),
        MatrixFormat::Packed(18) => Ok("hybrid_graph_iq_indexed_gpu_bound_iq3xxs"),
        _ => Err(Error::Invalid("GPU-bound IQ format")),
    }
}

pub(super) fn iq_typed16_schedule(
    format: MatrixFormat,
    columns: usize,
    offset: usize,
) -> bool {
    iq_bound_schedule(format, columns)
        && offset.is_multiple_of(2)
        && *IQ_TYPED16.get_or_init(|| std::env::var_os("STARK_GLM53_IQ_TYPED16").is_some_and(|value| value == "1"))
}

pub(super) fn iq_bound_typed_name(format: MatrixFormat) -> Result<&'static str> {
    match format {
        MatrixFormat::Packed(17) => Ok("hybrid_graph_iq_indexed_gpu_bound_typed16_iq2xs"),
        MatrixFormat::Packed(18) => Ok("hybrid_graph_iq_indexed_gpu_bound_typed16_iq3xxs"),
        _ => Err(Error::Invalid("typed GPU-bound IQ format")),
    }
}

pub(super) fn iq_signed_factor_schedule(
    format: MatrixFormat,
    columns: usize,
    offset: usize,
) -> bool {
    iq_typed16_schedule(format, columns, offset)
        && *IQ_SIGNED_FACTOR
            .get_or_init(|| std::env::var_os("STARK_GLM53_IQ_SIGNED_FACTOR").is_some_and(|value| value == "1"))
}

pub(super) fn iq_bound_factor_name(format: MatrixFormat) -> Result<&'static str> {
    match format {
        MatrixFormat::Packed(17) => Ok("hybrid_graph_iq_indexed_gpu_bound_factor_iq2xs"),
        MatrixFormat::Packed(18) => Ok("hybrid_graph_iq_indexed_gpu_bound_factor_iq3xxs"),
        _ => Err(Error::Invalid("signed-factor GPU-bound IQ format")),
    }
}

fn q4_grouped32_shape(
    format: MatrixFormat,
    rows: usize,
    columns: usize,
) -> bool {
    matches!(format, MatrixFormat::Packed(12))
        && rows == 154_880
        && columns == 4096
        && *Q4_GROUPED32.get_or_init(|| std::env::var_os("STARK_GLM53_Q4_GROUPED32").is_some_and(|value| value == "1"))
}

fn q5_grouped32_shape(
    format: MatrixFormat,
    columns: usize,
) -> bool {
    matches!(format, MatrixFormat::Packed(13))
        && columns != 0
        && *Q5_GROUPED32.get_or_init(|| std::env::var_os("STARK_GLM53_Q5_GROUPED32").is_some_and(|value| value == "1"))
}

fn q6_grouped16_shape(
    format: MatrixFormat,
    columns: usize,
) -> bool {
    // Indexed dispatch queries geometry with columns=0 and keeps its old row
    // helper. Only dense/selected-view matrices admit this guarded experiment.
    matches!(format, MatrixFormat::Packed(14))
        && columns != 0
        && *Q6_GROUPED16.get_or_init(|| std::env::var_os("STARK_GLM53_Q6_GROUPED16").is_some_and(|value| value == "1"))
}

fn iq_multirow_schedule(format: MatrixFormat) -> bool {
    matches!(format, MatrixFormat::Packed(17 | 18))
        && *IQ_MULTIROW.get_or_init(|| std::env::var_os("STARK_GLM53_IQ_MULTIROW").is_some_and(|value| value == "1"))
}

fn iq4_vector_schedule(format: MatrixFormat) -> bool {
    matches!(format, MatrixFormat::Packed(23))
        && *IQ4_VECTOR.get_or_init(|| std::env::var_os("STARK_GLM53_IQ4_VECTOR").is_some_and(|value| value == "1"))
}

// Only the two measured dense geometries admit the reassociated Safe schedule.
// Indexed dispatch asks with columns=0 and therefore retains its prior route.
fn q8_grouped8_shape(
    format: MatrixFormat,
    rows: usize,
    columns: usize,
) -> Option<bool> {
    if !matches!(format, MatrixFormat::Packed(8))
        || !*Q8_GROUPED8.get_or_init(|| std::env::var_os("STARK_GLM53_Q8_GROUPED8").is_some_and(|v| v == "1"))
    {
        return None;
    }
    match (rows, columns) {
        (16384, 1536) => Some(true),
        (8192, 128) => Some(false),
        _ => None,
    }
}

fn q8_block_shape(
    format: MatrixFormat,
    rows: usize,
    columns: usize,
) -> bool {
    matches!(format, MatrixFormat::Packed(8))
        && matches!((rows, columns), (16_384, 1536) | (8192, 128))
        && *Q8_SHAPES.get_or_init(|| std::env::var_os("STARK_GLM53_Q8_SHAPES").is_some_and(|value| value == "1"))
}

fn q4k_block_enabled() -> bool {
    // Freeze this diagnostic before the first packed dispatch so the
    // format-keyed pipeline caches and their grid geometry cannot disagree.
    *Q4K_BLOCK.get_or_init(|| std::env::var_os("STARK_GLM53_Q4K_BLOCK").is_some_and(|value| value == "1"))
}

fn block_schedule(format: MatrixFormat) -> bool {
    matches!(format, MatrixFormat::Packed(13 | 14))
        || matches!(format, MatrixFormat::Packed(12)) && (q4k_block_enabled() || factored_schedule(format))
}

fn factored_schedule(format: MatrixFormat) -> bool {
    matches!(format, MatrixFormat::Packed(12 | 13))
        && *KQUANT_FACTORED
            .get_or_init(|| std::env::var_os("STARK_GLM53_KQUANT_FACTORED").is_some_and(|value| value == "1"))
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum TestMatrixSchedule {
    Production,
    Scalar,
    Tiled,
    Block,
    Factored,
    IQ4Vector,
}

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
    pub(super) fn buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }

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

impl MappedMatrix {
    fn aligned_affine_loads(
        &self,
        format: MatrixFormat,
    ) -> bool {
        let MatrixFormat::Affine {
            scale_dtype,
            ..
        } = format
        else {
            return false;
        };
        let alignment = match scale_dtype {
            0 => 4,
            1 | 30 => 2,
            _ => return false,
        };
        self.codes.offset.is_multiple_of(4)
            && self.planes.as_ref().is_some_and(|(scales, biases)| {
                scales.offset.is_multiple_of(alignment) && biases.offset.is_multiple_of(alignment)
            })
    }
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
            source.push_str(include_str!("packed_iq_rows_matvec.metal"));
            source.push_str(include_str!("packed_iq_multirow_matvec.metal"));
            source.push_str(include_str!("decode_graph_indexed_batch.metal"));
            source.push_str(include_str!("packed_iq_staged_matvec.metal"));
            source.push_str(include_str!("packed_iq_grouped32_matvec.metal"));
            source.push_str(include_str!("packed_iq4_matvec.metal"));
            source.push_str(include_str!("packed_block_matvec.metal"));
            source.push_str(include_str!("packed_q8_grouped8_matvec.metal"));
            source.push_str(include_str!("packed_factored_matvec.metal"));
            source.push_str(include_str!("packed_q6_grouped16_matvec.metal"));
            source.push_str(include_str!("packed_q5_grouped32_matvec.metal"));
            source.push_str(include_str!("packed_kquant_mma_matvec.metal"));
            source.push_str(include_str!("feed_forward.metal"));
            source.push_str(include_str!("qwen_gdn.metal"));
            source.push_str(include_str!("decode_graph.metal"));
            source.push_str(include_str!("affine_uint_matvec.metal"));
            source.push_str(include_str!("affine_chunk32_matvec.metal"));
            source.push_str(include_str!("qwen_graph.metal"));
            source.push_str(include_str!("qwen_attention.metal"));
            #[cfg(test)]
            source.push_str(include_str!("../../../unit/fixtures/hybrid_tiled_experiment.metal"));
            #[cfg(test)]
            source.push_str(include_str!("../../../unit/fixtures/hybrid_graph_reference.metal"));
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = device
                .new_library_with_source(&source, Some(&options))
                .map_err(|_| "Metal kernel compilation failed")?;
            // Only low-bit projection functions use this experimental policy.
            // State, normalization, control, failure scans and float matrices
            // continue to resolve functions from the safe-math library.
            let packed_fast_library =
                if std::env::var_os("STARK_GLM53_PACKED_FAST_MATH").is_some_and(|value| value == "1") {
                    let fast_options = metal::MTLCompileOptions::new();
                    fast_options.set_math_mode(metal::MTLMathMode::Fast);
                    Some(
                        device
                            .new_library_with_source(&source, Some(&fast_options))
                            .map_err(|_| "Metal fast projection compilation failed")?,
                    )
                } else {
                    None
                };
            Ok(Executor {
                device,
                queue,
                library,
                packed_fast_library,
                pipelines: Mutex::new(HashMap::new()),
                q8_block_pipelines: Mutex::new(HashMap::new()),
                q8_grouped8_pipelines: Mutex::new(HashMap::new()),
                q8_segmented8_pipelines: Mutex::new(HashMap::new()),
                q6_grouped16_pipelines: Mutex::new(HashMap::new()),
                q5_grouped32_pipelines: Mutex::new(HashMap::new()),
                q4_grouped32_pipelines: Mutex::new(HashMap::new()),
                ffn_pipelines: Mutex::new(None),
                hc_pipelines: Mutex::new(None),
                gdn_pipelines: Mutex::new(None),
                graph_pipelines: Mutex::new(HashMap::new()),
                indexed_pipelines: Mutex::new(HashMap::new()),
                iq_bound_indexed_pipelines: Mutex::new(HashMap::new()),
                iq_bound_typed_indexed_pipelines: Mutex::new(HashMap::new()),
                iq_bound_factor_indexed_pipelines: Mutex::new(HashMap::new()),
                checked_matrix_pipelines: Mutex::new(HashMap::new()),
                typed_matrix_pipelines: Mutex::new(HashMap::new()),
                typed_indexed_pipelines: Mutex::new(HashMap::new()),
                chunk32_matrix_pipelines: Mutex::new(HashMap::new()),
                chunk32_indexed_pipelines: Mutex::new(HashMap::new()),
                uint4_matrix_pipelines: Mutex::new(HashMap::new()),
                uint4_indexed_pipelines: Mutex::new(HashMap::new()),
                factored_matrix_pipelines: Mutex::new(HashMap::new()),
                factored_indexed_pipelines: Mutex::new(HashMap::new()),
                ordered_rms: std::env::var_os("STARK_HYBRID_ORDERED_RMS").is_some_and(|value| value == "1"),
                #[cfg(test)]
                tiled_pipelines: Mutex::new(HashMap::new()),
                #[cfg(test)]
                scalar_pipelines: Mutex::new(HashMap::new()),
                #[cfg(test)]
                block_test_pipelines: Mutex::new(HashMap::new()),
                #[cfg(test)]
                factored_test_pipelines: Mutex::new(HashMap::new()),
                #[cfg(test)]
                iq4_test_pipelines: Mutex::new(HashMap::new()),
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
    columns: usize,
) -> usize {
    match format {
        format if q4_grouped32_shape(format, rows, columns) => rows.div_ceil(2),
        format if q8_grouped8_shape(format, rows, columns) == Some(true) => rows.div_ceil(4),
        format if q8_grouped8_shape(format, rows, columns) == Some(false) => rows.div_ceil(8),
        format if q5_grouped32_shape(format, columns) => rows.div_ceil(2),
        format if q6_grouped16_shape(format, columns) => rows.div_ceil(4),
        format if q8_block_shape(format, rows, columns) => rows.div_ceil(2),
        format if block_schedule(format) => rows.div_ceil(2),
        format if iq4_vector_schedule(format) => rows.div_ceil(8),
        MatrixFormat::Packed(17 | 18)
        | MatrixFormat::Affine {
            ..
        } => rows.div_ceil(8),
        _ => rows,
    }
}

fn matrix_threads(
    format: MatrixFormat,
    rows: usize,
    columns: usize,
) -> usize {
    match format {
        format if q4_grouped32_shape(format, rows, columns) => 64,
        format if q8_grouped8_shape(format, rows, columns) == Some(true) => 64,
        format if q8_grouped8_shape(format, rows, columns) == Some(false) => 64,
        format if q5_grouped32_shape(format, columns) => 64,
        format if q6_grouped16_shape(format, columns) => 64,
        format if q8_block_shape(format, rows, columns) => 64,
        format if block_schedule(format) => 64,
        _ => 256,
    }
}

/// Indexed banks preserve the selected-view matrix reduction. The opt-in IQ
/// multirow geometry shares input loads across independent row accumulators.
/// Affine typed dispatch is selected separately by DecodeGraph.
pub(super) fn indexed_name(format: MatrixFormat) -> &'static str {
    match format {
        format if factored_schedule(format) => "hybrid_graph_packed_indexed_factored",
        format if block_schedule(format) => "hybrid_graph_packed_indexed_block",
        format if iq4_vector_schedule(format) => "hybrid_graph_iq4_indexed_vector8",
        format if iq_multirow_schedule(format) => "hybrid_graph_iq_indexed_multirow",
        MatrixFormat::Packed(17 | 18) => "hybrid_graph_iq_indexed_vector8",
        MatrixFormat::Packed(_) => "hybrid_graph_packed_indexed",
        MatrixFormat::Affine {
            ..
        } => "hybrid_graph_affine_indexed",
    }
}

pub(super) fn indexed_groups(
    format: MatrixFormat,
    rows: usize,
) -> usize {
    // Q8 shape admission is for the two measured dense projections only.
    // Indexed IQ multirow still owns eight rows per group; only its thread
    // count changes. Other banks retain their established schedules.
    matrix_groups(format, rows, 0)
}

pub(super) fn indexed_threads(format: MatrixFormat) -> usize {
    if iq_multirow_schedule(format) {
        return 64;
    }
    matrix_threads(format, 0, 0)
}

impl Executor {
    fn pipeline_for_geometry(
        &self,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
    ) -> Result<PipelineState> {
        if q8_grouped8_shape(format, rows, columns) == Some(true) {
            self.cached_pipeline(format, "hybrid_q8_grouped8_r4g2_matvec", &self.q8_grouped8_pipelines)
        } else if q8_grouped8_shape(format, rows, columns) == Some(false) {
            self.cached_pipeline(format, "hybrid_q8_segmented8_safe_matvec", &self.q8_segmented8_pipelines)
        } else if q4_grouped32_shape(format, rows, columns) {
            self.cached_pipeline(format, "hybrid_q4_grouped32_matvec", &self.q4_grouped32_pipelines)
        } else if q5_grouped32_shape(format, columns) {
            self.cached_pipeline(format, "hybrid_q5_grouped32_matvec", &self.q5_grouped32_pipelines)
        } else if q6_grouped16_shape(format, columns) {
            self.cached_pipeline(format, "hybrid_q6_grouped16_matvec", &self.q6_grouped16_pipelines)
        } else if q8_block_shape(format, rows, columns) {
            // One format can select either shape schedule in the same model;
            // keep the block pipeline separate from the scalar format cache.
            self.cached_pipeline(format, "hybrid_packed_block_matvec", &self.q8_block_pipelines)
        } else {
            self.pipeline(format)
        }
    }

    fn run_glm_kda(
        &self,
        operations: &[MappedOperation<'_>; 9],
        parameters: &[[u32; 8]],
        state: &mut super::glm_metal::Kda,
    ) -> Result<Vec<f32>> {
        let pipelines = operations
            .iter()
            .map(|operation| self.pipeline_for_geometry(operation.format, operation.rows, operation.columns))
            .collect::<Result<Vec<_>>>()?;
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
        let recurrent = if std::env::var_os("STARK_QWEN38_STAGED_GDN").is_some_and(|value| value == "1") {
            "hybrid_qwen_gdn_recurrent"
        } else {
            "hybrid_qwen_gdn_recurrent_fused"
        };
        let result = (make("hybrid_qwen_gdn_convolve")?, make(recurrent)?);
        *pipelines = Some(result.clone());
        Ok(result)
    }

    fn run_qwen_gdn(
        &self,
        operations: &[MappedOperation<'_>; 5],
        parameters: &[[u32; 8]],
        state: &mut QwenGdnState,
    ) -> Result<Vec<f32>> {
        let pipelines = operations
            .iter()
            .map(|operation| self.pipeline_for_geometry(operation.format, operation.rows, operation.columns))
            .collect::<Result<Vec<_>>>()?;
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
        let pipelines = operations
            .iter()
            .map(|operation| self.pipeline_for_geometry(operation.format, operation.rows, operation.columns))
            .collect::<Result<Vec<_>>>()?;
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
                    width: matrix_groups(operation.format, operation.rows, operation.columns),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: matrix_threads(operation.format, operation.rows, operation.columns),
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
        let pipelines = operations
            .iter()
            .map(|operation| self.pipeline_for_geometry(operation.format, operation.rows, operation.columns))
            .collect::<Result<Vec<_>>>()?;
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
            format if factored_schedule(format) => "hybrid_packed_factored_matvec",
            format if block_schedule(format) => "hybrid_packed_block_matvec",
            format if iq4_vector_schedule(format) => "hybrid_iq4_matvec_vector8",
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
        if name == "hybrid_graph_iq_indexed_multirow" || name.starts_with("hybrid_graph_iq_indexed_gpu_bound_") {
            set(71, 4);
        }
        if name == "hybrid_q6_grouped16_matvec" {
            // Production always preserves the input-envelope fallback. The
            // bounded-only constructor exists exclusively in cfg(test).
            set(73, 0);
        }
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
        let library = if matches!(
            name,
            "hybrid_q6_grouped16_matvec"
                | "hybrid_q5_grouped32_matvec"
                | "hybrid_q4_grouped32_matvec"
                | "hybrid_q8_grouped8_safe_matvec"
                | "hybrid_q8_grouped8_r4g2_matvec"
                | "hybrid_q8_segmented8_safe_matvec"
        ) || name.starts_with("hybrid_graph_iq_indexed_gpu_bound_")
        {
            &self.library
        } else if matches!(format, MatrixFormat::Packed(kind) if !matches!(kind, 0 | 1 | 30)) {
            self.packed_fast_library.as_ref().unwrap_or(&self.library)
        } else {
            &self.library
        };
        let function = library
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
        let pipelines = operations
            .iter()
            .map(|operation| self.pipeline_for_geometry(operation.format, operation.rows, operation.columns))
            .collect::<Result<Vec<_>>>()?;
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
                        width: matrix_groups(operation.format, operation.rows, operation.columns),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: matrix_threads(operation.format, operation.rows, operation.columns),
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
        objc2::rc::autoreleasepool(|_| self.run_inner(codes, input, parameters, affine, TestMatrixSchedule::Production))
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
        let schedule = if tiled {
            TestMatrixSchedule::Tiled
        } else {
            TestMatrixSchedule::Scalar
        };
        objc2::rc::autoreleasepool(|_| self.run_inner(codes, input, parameters, affine, schedule))
    }

    #[cfg(test)]
    fn run_inner(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
        schedule: TestMatrixSchedule,
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
        let production = matches!(schedule, TestMatrixSchedule::Production);
        let tiled = matches!(schedule, TestMatrixSchedule::Tiled);
        let factored = matches!(schedule, TestMatrixSchedule::Factored);
        let block = matches!(schedule, TestMatrixSchedule::Block | TestMatrixSchedule::Factored);
        let iq4 = matches!(schedule, TestMatrixSchedule::IQ4Vector);
        let pipeline = if factored {
            self.cached_pipeline(format, "hybrid_packed_factored_matvec", &self.factored_test_pipelines)?
        } else if block {
            self.cached_pipeline(format, "hybrid_packed_block_matvec", &self.block_test_pipelines)?
        } else if iq4 {
            self.cached_pipeline(format, "hybrid_iq4_matvec_vector8", &self.iq4_test_pipelines)?
        } else if production {
            self.pipeline_for_geometry(format, parameters[0] as usize, parameters[1] as usize)?
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
                width: if block {
                    (parameters[0] as usize).div_ceil(2)
                } else if iq4 {
                    (parameters[0] as usize).div_ceil(8)
                } else if production {
                    matrix_groups(format, parameters[0] as usize, parameters[1] as usize)
                } else if tiled {
                    (parameters[0] as usize).div_ceil(4)
                } else {
                    parameters[0] as usize
                },
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: if block {
                    64
                } else if production {
                    matrix_threads(format, parameters[0] as usize, parameters[1] as usize)
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

    pub(super) fn finite_packed_codes(
        kind: u32,
        rows: usize,
        columns: usize,
        seed: usize,
    ) -> Vec<u8> {
        let (block, size) = weights::block_layout(kind).unwrap();
        let mut codes = Vec::with_capacity(rows * columns / block * size);
        for number in 0..rows * columns / block {
            let mut encoded: Vec<_> = (0..size).map(|i| ((i * 37 + number * 13 + seed) % 256) as u8).collect();
            let half_offsets: &[usize] = match kind {
                12 | 13 => &[0, 2],
                14 => &[208],
                8 | 17 | 18 | 23 => &[0],
                _ => unreachable!(),
            };
            for &at in half_offsets {
                encoded[at..at + 2].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
            }
            codes.extend(encoded);
        }
        codes
    }

    pub(super) fn packed_f64_oracle(
        kind: u32,
        codes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        rows_per_batch: usize,
    ) -> Vec<(f64, f64)> {
        let (block, bytes) = weights::block_layout(kind).unwrap();
        let row_bytes = columns / block * bytes;
        (0..rows)
            .map(|row| {
                let input_start = if rows_per_batch == 0 {
                    0
                } else {
                    row / rows_per_batch * columns
                };
                let mut expected = 0.0f64;
                let mut absolute = 0.0f64;
                for (encoded, input) in codes[row * row_bytes..(row + 1) * row_bytes]
                    .chunks_exact(bytes)
                    .zip(input[input_start..input_start + columns].chunks_exact(block))
                {
                    let mut decoded = vec![0.0; block];
                    weights::decode_block(kind, encoded, &mut decoded).unwrap();
                    for (weight, input) in decoded.iter().zip(input) {
                        let product = f64::from(*weight) * f64::from(*input);
                        expected += product;
                        absolute += product.abs();
                    }
                }
                (expected, absolute)
            })
            .collect()
    }

    #[stark_test]
    fn kquant_factoring_matches_f64_and_preserves_extreme_input_failures() {
        let (rows, batches) = (13, 3);
        for kind in [12, 13, 14] {
            let (block, bytes) = weights::block_layout(kind).unwrap();
            for columns in [256, 768, 4096, 8192] {
                let codes = finite_packed_codes(kind, rows * batches, columns, 41);
                let input = (0..columns * batches).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>();
                let parameters =
                    [(rows * batches) as u32, columns as u32, kind, block as u32, bytes as u32, 0, rows as u32, 0];
                let actual = objc2::rc::autoreleasepool(|_| {
                    executor()
                        .unwrap()
                        .run_inner(&codes, &input, parameters, None, TestMatrixSchedule::Factored)
                        .unwrap()
                });
                for (row, (expected, absolute)) in
                    packed_f64_oracle(kind, &codes, rows * batches, columns, &input, rows).into_iter().enumerate()
                {
                    assert!(
                        (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                        "factored F64 kind={kind} columns={columns} row={row}"
                    );
                }
            }
            let columns = 256;
            let codes = finite_packed_codes(kind, 3, columns, 41);
            let parameters = [3, columns as u32, kind, block as u32, bytes as u32, 0, 0, 0];
            let run = |codes: &[u8], input: &[f32], schedule| {
                objc2::rc::autoreleasepool(|_| executor().unwrap().run_inner(codes, input, parameters, None, schedule))
            };
            let large = (0..columns)
                .map(|i| {
                    if i % 2 == 0 {
                        1e30
                    } else {
                        -1e30
                    }
                })
                .collect::<Vec<_>>();
            let expected = run(&codes, &large, TestMatrixSchedule::Block).unwrap();
            assert_eq!(
                run(&codes, &large, TestMatrixSchedule::Factored).unwrap(),
                expected,
                "large finite reference fallback kind={kind}"
            );
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let invalid = vec![invalid; columns];
                assert!(run(&codes, &invalid, TestMatrixSchedule::Block).is_err());
                assert!(run(&codes, &invalid, TestMatrixSchedule::Factored).is_err());
            }
            // An overflowing sum of inputs is harmless when every decoded
            // weight is zero. The fallback must accept this old valid case.
            let mut zero = vec![0u8; bytes];
            if kind == 14 {
                zero[128..192].fill(0xaa);
                zero[192..208].fill(1);
                zero[208..210].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
            } else {
                zero[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
                zero[4..16].fill(1);
            }
            let zero = zero.repeat(3);
            let input = vec![f32::MAX; columns];
            assert_eq!(run(&zero, &input, TestMatrixSchedule::Block).unwrap(), vec![0.0; 3]);
            assert_eq!(run(&zero, &input, TestMatrixSchedule::Factored).unwrap(), vec![0.0; 3]);
        }
    }

    #[stark_test]
    fn q4k_block_schedule_matches_wide_f64_reference_and_batch_boundaries() {
        let (rows, batches) = (13, 3);
        let (block, bytes) = weights::block_layout(12).unwrap();
        for columns in [256, 768, 4096, 8192] {
            let codes = finite_packed_codes(12, rows * batches, columns, 29);
            let input = (0..columns * batches).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let parameters =
                [(rows * batches) as u32, columns as u32, 12, block as u32, bytes as u32, 0, rows as u32, 0];
            let actual = objc2::rc::autoreleasepool(|_| {
                executor().unwrap().run_inner(&codes, &input, parameters, None, TestMatrixSchedule::Block).unwrap()
            });
            let scalar = executor().unwrap().run_with_tiling(&codes, &input, parameters, None, false).unwrap();
            let row_bytes = columns / block * bytes;
            for row in 0..rows * batches {
                let mut expected = 0.0f64;
                let mut absolute = 0.0f64;
                let source_input = &input[row / rows * columns..(row / rows + 1) * columns];
                for (encoded, input) in codes[row * row_bytes..(row + 1) * row_bytes]
                    .chunks_exact(bytes)
                    .zip(source_input.chunks_exact(block))
                {
                    let mut decoded = vec![0.0; block];
                    weights::decode_block(12, encoded, &mut decoded).unwrap();
                    for (weight, input) in decoded.iter().zip(input) {
                        let product = f64::from(*weight) * f64::from(*input);
                        expected += product;
                        absolute += product.abs();
                    }
                }
                // Keep a wide F64 oracle independent of either GPU reduction.
                let bound = 2e-6 * absolute.max(1.0);
                assert!(
                    (f64::from(actual[row]) - expected).abs() <= bound,
                    "Q4_K block F64 columns={columns} row={row}"
                );
                assert!(
                    (f64::from(scalar[row]) - expected).abs() <= bound,
                    "Q4_K scalar F64 columns={columns} row={row}"
                );
            }
        }
    }

    fn run_kquant_matrix_test(
        pipeline: &PipelineState,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        rows_per_group: usize,
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| {
            let executor = executor()?;
            let count = parameters[0] as usize;
            let codes = buffer(executor, codes)?;
            let input = buffer(executor, bytemuck::cast_slice(input))?;
            let output = buffer(executor, bytemuck::cast_slice(&vec![7.25f32; count + 2]))?;
            let command = executor.queue.command_buffer().unwrap();
            let encoding = command.compute_command_encoder().unwrap();
            encoding.set_compute_pipeline_state(pipeline);
            for (index, value) in [&codes, &input, &output].into_iter().enumerate() {
                encoding.set_buffer(
                    Some(value),
                    if index == 2 {
                        4
                    } else {
                        0
                    },
                    index,
                );
            }
            encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
            let offsets = [0u32; 4];
            encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: count.div_ceil(rows_per_group),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            encoding.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            // SAFETY: the owned shared output has a completed GPU writer.
            let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count + 2) };
            assert_eq!(values[0], 7.25, "Kquant matrix prefix guard changed");
            assert_eq!(values[count + 1], 7.25, "Kquant matrix row tail guard changed");
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("nonfinite Kquant matrix test result"));
            }
            Ok(values[1..count + 1].to_vec())
        })
    }

    fn safe_q6_grouped16_test_pipeline(bounded_only: bool) -> PipelineState {
        let executor = executor().unwrap();
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in [(0, 14u32), (4, 256), (5, 210), (73, u32::from(bounded_only))] {
            constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
        }
        let function =
            executor.library.new_function_with_name_constant_values("hybrid_q6_grouped16_matvec", &constants).unwrap();
        executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
    }

    #[stark_test]
    fn q6_grouped16_production_admission_preserves_safe_bits_and_indexed_geometry() {
        let (rows, columns) = (13usize, 768usize);
        let format = MatrixFormat::Packed(14);
        let enabled = q6_grouped16_shape(format, columns);
        assert_eq!(indexed_name(format), "hybrid_graph_packed_indexed_block");
        assert_eq!(indexed_groups(format, rows), rows.div_ceil(2));
        assert_eq!(indexed_threads(format), 64);
        assert_eq!(
            matrix_groups(format, rows, columns),
            rows.div_ceil(if enabled {
                4
            } else {
                2
            })
        );
        assert_eq!(matrix_threads(format, rows, columns), 64);
        if enabled {
            let codes = finite_packed_codes(14, rows, columns, 41);
            let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let parameters = [rows as u32, columns as u32, 14, 256, 210, 0, 0, 0];
            let safe = safe_q6_grouped16_test_pipeline(false);
            let reference = run_kquant_matrix_test(&safe, &codes, &input, parameters, 4).unwrap();
            let actual = executor()
                .unwrap()
                .run_inner(&codes, &input, parameters, None, TestMatrixSchedule::Production)
                .unwrap();
            assert!(
                actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                "Q6 production changed Safe grouped16 bits"
            );
        }
    }

    #[stark_test]
    fn q6_grouped16_matches_f64_source_and_safe_fallback() {
        let (rows, batches) = (13usize, 3usize);
        let original = safe_iq_test_pipeline(14, "hybrid_packed_block_matvec", 4);
        let guarded = safe_q6_grouped16_test_pipeline(false);
        let bounded_only = safe_q6_grouped16_test_pipeline(true);
        for columns in [256usize, 768, 4096, 8192] {
            let mut codes = finite_packed_codes(14, rows * batches, columns, 41);
            for (i, block) in codes.chunks_exact_mut(210).enumerate() {
                let d = half::f16::from_f32(0.001 + ((i * 17) % 257) as f32 * 0.00003125);
                block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            let parameters = [(rows * batches) as u32, columns as u32, 14, 256, 210, 0, rows as u32, 0];
            let input = (0..columns * batches).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            // The no-guard specialization is a diagnostic only. Its caller
            // must verify this envelope before any dispatch using that code.
            assert!(input.iter().all(|v| v.is_finite() && v.abs() <= 1e20));
            let oracle = packed_f64_oracle(14, &codes, rows * batches, columns, &input, rows);
            let reference = run_kquant_matrix_test(&original, &codes, &input, parameters, 2).unwrap();
            let guarded_values = run_kquant_matrix_test(&guarded, &codes, &input, parameters, 4).unwrap();
            let bounded_values = run_kquant_matrix_test(&bounded_only, &codes, &input, parameters, 4).unwrap();
            assert!(
                guarded_values.iter().zip(&bounded_values).all(|(a, b)| a.to_bits() == b.to_bits()),
                "Q6 grouped16 guard removal changed Safe arithmetic"
            );
            let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
            for actual in [&reference, &guarded_values, &bounded_values] {
                for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                    assert!(
                        (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                        "Q6 grouped16 F64 columns={columns}"
                    );
                }
                let error: f64 =
                    actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                assert!((error / squared_reference.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "Q6 grouped16 source RMS");
            }
            let large = (0..columns * batches)
                .map(|i| {
                    if i % 2 == 0 {
                        1e30
                    } else {
                        -1e30
                    }
                })
                .collect::<Vec<_>>();
            let reference = run_kquant_matrix_test(&original, &codes, &large, parameters, 2).unwrap();
            let actual = run_kquant_matrix_test(&guarded, &codes, &large, parameters, 4).unwrap();
            assert!(
                actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                "Q6 grouped16 fallback changed bits"
            );
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let invalid = vec![poison; columns * batches];
                assert!(run_kquant_matrix_test(&original, &codes, &invalid, parameters, 2).is_err());
                assert!(run_kquant_matrix_test(&guarded, &codes, &invalid, parameters, 4).is_err());
            }
        }
        let mut zero = vec![0u8; 210];
        zero[128..192].fill(0xaa);
        zero[192..208].fill(1);
        zero[208..210].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        let parameters = [(rows * batches) as u32, 256, 14, 256, 210, 0, rows as u32, 0];
        assert_eq!(
            run_kquant_matrix_test(
                &guarded,
                &zero.repeat(rows * batches),
                &vec![f32::MAX; 256 * batches],
                parameters,
                4
            )
            .unwrap(),
            vec![0.0; rows * batches]
        );
    }

    #[stark_test]
    fn q5_grouped32_production_admission_preserves_safe_bits_and_legacy_indexed_geometry() {
        let (rows, columns) = (13usize, 768usize);
        let format = MatrixFormat::Packed(13);
        let enabled = q5_grouped32_shape(format, columns);
        assert_eq!(
            indexed_name(format),
            if factored_schedule(format) {
                "hybrid_graph_packed_indexed_factored"
            } else {
                "hybrid_graph_packed_indexed_block"
            }
        );
        assert_eq!(indexed_groups(format, rows), rows.div_ceil(2));
        assert_eq!(indexed_threads(format), 64);
        assert_eq!(matrix_groups(format, rows, columns), rows.div_ceil(2));
        assert_eq!(matrix_threads(format, rows, columns), 64);
        if enabled {
            let codes = finite_packed_codes(13, rows, columns, 41);
            let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let parameters = [rows as u32, columns as u32, 13, 256, 176, 0, 0, 0];
            let safe = safe_iq_test_pipeline(13, "hybrid_q5_grouped32_matvec", 4);
            let reference = run_kquant_matrix_test(&safe, &codes, &input, parameters, 2).unwrap();
            let actual = executor()
                .unwrap()
                .run_inner(&codes, &input, parameters, None, TestMatrixSchedule::Production)
                .unwrap();
            assert!(
                actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                "Q5 production changed Safe grouped32 bits"
            );
        }
    }

    #[stark_test]
    fn q5_grouped32_matches_f64_source_and_safe_fallback() {
        let (rows, batches) = (13usize, 3usize);
        let original = safe_iq_test_pipeline(13, "hybrid_packed_factored_matvec", 4);
        let candidate = safe_iq_test_pipeline(13, "hybrid_q5_grouped32_matvec", 4);
        for columns in [256usize, 768, 4096, 8192] {
            let mut codes = finite_packed_codes(13, rows * batches, columns, 41);
            for (i, block) in codes.chunks_exact_mut(176).enumerate() {
                let d = half::f16::from_f32(0.001 + ((i * 17) % 257) as f32 * 0.00003125);
                let dmin = half::f16::from_f32(0.0004 + ((i * 29) % 131) as f32 * 0.000015625);
                block[..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            let parameters = [(rows * batches) as u32, columns as u32, 13, 256, 176, 0, rows as u32, 0];
            let input = (0..columns * batches).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let oracle = packed_f64_oracle(13, &codes, rows * batches, columns, &input, rows);
            let reference = run_kquant_matrix_test(&original, &codes, &input, parameters, 2).unwrap();
            let actual = run_kquant_matrix_test(&candidate, &codes, &input, parameters, 2).unwrap();
            for actual in [&reference, &actual] {
                for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                    assert!(
                        (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                        "Q5 grouped32 F64 columns={columns}"
                    );
                }
                let error: f64 =
                    actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                let scale: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                assert!((error / scale.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "Q5 grouped32 source RMS");
            }
            let large = (0..columns * batches)
                .map(|i| {
                    if i % 2 == 0 {
                        1e30
                    } else {
                        -1e30
                    }
                })
                .collect::<Vec<_>>();
            let reference = run_kquant_matrix_test(&original, &codes, &large, parameters, 2).unwrap();
            let actual = run_kquant_matrix_test(&candidate, &codes, &large, parameters, 2).unwrap();
            assert!(
                actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                "Q5 grouped32 fallback changed bits"
            );
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let invalid = vec![poison; columns * batches];
                assert!(run_kquant_matrix_test(&original, &codes, &invalid, parameters, 2).is_err());
                assert!(run_kquant_matrix_test(&candidate, &codes, &invalid, parameters, 2).is_err());
            }
        }
        let mut zero = vec![0u8; 176];
        zero[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        zero[4..16].fill(1);
        let parameters = [(rows * batches) as u32, 256, 13, 256, 176, 0, rows as u32, 0];
        assert_eq!(
            run_kquant_matrix_test(
                &candidate,
                &zero.repeat(rows * batches),
                &vec![f32::MAX; 256 * batches],
                parameters,
                2
            )
            .unwrap(),
            vec![0.0; rows * batches]
        );
    }

    #[stark_test]
    fn q8_hot_shapes_match_f64_and_keep_small_projection_scalar() {
        let format = MatrixFormat::Packed(8);
        for (rows, columns) in [(16_384, 1536), (8192, 128), (128, 4096), (13, 128)] {
            let codes = finite_packed_codes(8, rows, columns, 41);
            let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let parameters = [rows as u32, columns as u32, 8, 32, 34, 0, 0, 0];
            let run = |schedule| {
                objc2::rc::autoreleasepool(|_| {
                    executor().unwrap().run_inner(&codes, &input, parameters, None, schedule).unwrap()
                })
            };
            let actual = run(TestMatrixSchedule::Production);
            let scalar = run(TestMatrixSchedule::Scalar);
            let blocked = run(TestMatrixSchedule::Block);
            if q8_block_shape(format, rows, columns) {
                assert_eq!(actual, blocked, "Q8 hot shape did not use the block schedule");
            } else {
                assert_eq!(actual, scalar, "Q8 non-admitted shape changed scalar bits");
            }
            let row_bytes = columns / 32 * 34;
            for row in [0, 1, rows / 2, rows - 1] {
                let (expected, absolute) =
                    packed_f64_oracle(8, &codes[row * row_bytes..(row + 1) * row_bytes], 1, columns, &input, 0)[0];
                let bound = 2e-6 * absolute.max(1.0);
                assert!(
                    (f64::from(actual[row]) - expected).abs() <= bound,
                    "Q8 production F64 rows={rows} columns={columns}"
                );
                assert!(
                    (f64::from(blocked[row]) - expected).abs() <= bound,
                    "Q8 block F64 rows={rows} columns={columns}"
                );
            }
        }
    }

    pub(super) fn safe_iq_test_pipeline(
        kind: u32,
        name: &str,
        rows_per_simd: u32,
    ) -> PipelineState {
        let executor = executor().unwrap();
        let (_, bytes) = weights::block_layout(kind).unwrap();
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in [(0, kind), (4, 256), (5, bytes as u32), (71, rows_per_simd)] {
            constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
        }
        let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
        executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
    }

    fn safe_iq_checked_hint_pipeline(
        kind: u32,
        factor: bool,
        hinted: bool,
    ) -> PipelineState {
        let format = MatrixFormat::Packed(kind);
        let name = if factor {
            iq_bound_factor_name(format).unwrap()
        } else {
            iq_bound_typed_name(format).unwrap()
        };
        if !hinted {
            return safe_iq_test_pipeline(kind, name, 4);
        }
        let executor = executor().unwrap();
        let (_, bytes) = weights::block_layout(kind).unwrap();
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in [(0, kind), (4, 256), (5, bytes as u32), (71, 4)] {
            constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
        }
        let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
        let descriptor = metal::MTLComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        descriptor.set_max_total_threads_per_threadgroup(64);
        descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);
        let (pipeline, _) = executor
            .device
            .new_compute_pipeline_state_with_descriptor(&descriptor, metal::MTLPipelineOption::NONE)
            .unwrap();
        assert_eq!(pipeline.thread_execution_width(), 32);
        assert_eq!(pipeline.max_total_threads_per_threadgroup(), 64);
        pipeline
    }

    fn safe_staged_iq_test_pipeline(kind: u32) -> PipelineState {
        safe_iq_test_pipeline(
            kind,
            match kind {
                17 => "hybrid_graph_iq_indexed_staged_iq2xs",
                18 => "hybrid_graph_iq_indexed_staged_iq3xxs",
                _ => unreachable!("staged IQ test kind"),
            },
            4,
        )
    }

    fn safe_grouped32_iq_test_pipeline(kind: u32) -> PipelineState {
        safe_iq_test_pipeline(
            kind,
            match kind {
                17 => "hybrid_graph_iq_indexed_grouped32_iq2xs",
                18 => "hybrid_graph_iq_indexed_grouped32_iq3xxs",
                _ => unreachable!("grouped32 IQ test kind"),
            },
            4,
        )
    }

    fn safe_bound_grouped32_iq_test_pipeline(
        kind: u32,
        input: &[f32],
    ) -> Result<PipelineState> {
        // This predicate is deliberately host-only and test-only. It does not
        // create a trusted DeviceVector or change any production precondition.
        if input.is_empty() || input.iter().any(|value| !value.is_finite() || value.abs() > 1.0e20) {
            return Err(Error::Invalid("bound IQ diagnostic requires finite bounded host input"));
        }
        Ok(safe_iq_test_pipeline(
            kind,
            match kind {
                17 => "hybrid_graph_iq_indexed_grouped32_bound_iq2xs",
                18 => "hybrid_graph_iq_indexed_grouped32_bound_iq3xxs",
                _ => return Err(Error::Invalid("bound IQ diagnostic kind")),
            },
            4,
        ))
    }

    fn safe_iq_sweep_pipeline(
        kind: u32,
        rows: usize,
        typed16: bool,
        input: &[f32],
        code_offset: usize,
    ) -> Result<PipelineState> {
        if !matches!(rows, 1 | 2 | 4 | 8)
            || !matches!(kind, 17 | 18)
            || (typed16 && !code_offset.is_multiple_of(2))
            || input.is_empty()
            || input.iter().any(|x| !x.is_finite() || x.abs() > 1.0e20)
        {
            return Err(Error::Invalid("IQ sweep alignment or host input is not admitted"));
        }
        let name = format!(
            "hybrid_iq_sweep_{}_r{rows}_{}",
            if kind == 17 {
                "iq2xs"
            } else {
                "iq3xxs"
            },
            if typed16 {
                "typed16"
            } else {
                "byte"
            }
        );
        Ok(safe_iq_test_pipeline(kind, &name, 4))
    }

    fn safe_iq_arithmetic_pipeline(
        kind: u32,
        mode: usize,
        input: &[f32],
        code_offset: usize,
    ) -> Result<PipelineState> {
        if !matches!(mode, 1 | 2)
            || !matches!(kind, 17 | 18)
            || !code_offset.is_multiple_of(2)
            || input.is_empty()
            || input.iter().any(|x| !x.is_finite() || x.abs() > 1.0e20)
        {
            return Err(Error::Invalid("IQ arithmetic diagnostic host input or alignment is not admitted"));
        }
        let name = format!(
            "hybrid_iq_arithmetic_{}_{}",
            if kind == 17 {
                "iq2xs"
            } else {
                "iq3xxs"
            },
            if mode == 1 {
                "factor"
            } else {
                "fma"
            }
        );
        Ok(safe_iq_test_pipeline(kind, &name, 4))
    }

    fn safe_gpu_bound_iq_test_pipeline(kind: u32) -> PipelineState {
        safe_iq_test_pipeline(kind, iq_bound_name(MatrixFormat::Packed(kind)).unwrap(), 4)
    }

    #[expect(clippy::too_many_arguments, reason = "test keeps indexed geometry and source buffers explicit")]
    fn run_iq_indexed_test(
        pipeline: &PipelineState,
        codes: &[u8],
        input: &[f32],
        indices: &[u32],
        parameters: [u32; 4],
        rows_per_group: usize,
        threads: usize,
    ) -> Result<Vec<f32>> {
        run_iq_indexed_test_with_bound(pipeline, codes, input, indices, parameters, rows_per_group, threads, false)
    }

    #[expect(clippy::too_many_arguments, reason = "test keeps indexed geometry and source buffers explicit")]
    pub(super) fn run_iq_indexed_test_with_bound(
        pipeline: &PipelineState,
        codes: &[u8],
        input: &[f32],
        indices: &[u32],
        parameters: [u32; 4],
        rows_per_group: usize,
        threads: usize,
        gpu_bound: bool,
    ) -> Result<Vec<f32>> {
        run_iq_indexed_test_with_offsets(
            pipeline,
            codes,
            input,
            indices,
            parameters,
            rows_per_group,
            threads,
            gpu_bound,
            if gpu_bound {
                16
            } else {
                0
            },
        )
    }

    #[expect(clippy::too_many_arguments, reason = "test keeps indexed geometry and source buffers explicit")]
    fn run_iq_indexed_test_with_offsets(
        pipeline: &PipelineState,
        codes: &[u8],
        input: &[f32],
        indices: &[u32],
        parameters: [u32; 4],
        rows_per_group: usize,
        threads: usize,
        gpu_bound: bool,
        code_offset: usize,
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| {
            let executor = executor()?;
            let vector_offset = if gpu_bound {
                4
            } else {
                0
            };
            let mut code_storage = vec![0x75u8; code_offset];
            code_storage.extend_from_slice(codes);
            let codes = buffer(executor, &code_storage)?;
            let mut input_storage = if gpu_bound {
                vec![7.25]
            } else {
                Vec::new()
            };
            input_storage.extend_from_slice(input);
            let input = buffer(executor, bytemuck::cast_slice(&input_storage))?;
            let mut index_storage = if gpu_bound {
                vec![u32::MAX]
            } else {
                Vec::new()
            };
            index_storage.extend_from_slice(indices);
            let indices_buffer = buffer(executor, bytemuck::cast_slice(&index_storage))?;
            let batches = if parameters[3] != 0 {
                indices.len()
            } else {
                1
            };
            let bounds = buffer(executor, bytemuck::cast_slice(&vec![12345u32; batches + 2]))?;
            let count = parameters[0] as usize * indices.len();
            let output = buffer(executor, bytemuck::cast_slice(&vec![7.25f32; count + 2]))?;
            let failure = buffer(executor, bytemuck::cast_slice(&[0u32]))?;
            let command = executor.queue.command_buffer().unwrap();
            let encoding = command.compute_command_encoder().unwrap();
            if gpu_bound {
                let function = executor.library.new_function_with_name("hybrid_graph_iq_input_bound").unwrap();
                let prepass = executor.device.new_compute_pipeline_state_with_function(&function).unwrap();
                encoding.set_compute_pipeline_state(&prepass);
                encoding.set_buffer(Some(&input), vector_offset, 0);
                encoding.set_buffer(Some(&bounds), 4, 1);
                encoding.set_buffer(Some(&failure), 0, 30);
                let bound_parameters = [parameters[1], batches as u32];
                encoding.set_bytes(
                    NonNull::from(&bound_parameters).cast(),
                    std::mem::size_of_val(&bound_parameters),
                    2,
                );
                encoding.dispatch_threadgroups(
                    MTLSize {
                        width: batches,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
            }
            encoding.set_compute_pipeline_state(pipeline);
            for (index, value) in [&codes, &input, &output].into_iter().enumerate() {
                encoding.set_buffer(
                    Some(value),
                    match index {
                        0 => code_offset,
                        1 => vector_offset,
                        _ => 4,
                    },
                    index,
                );
            }
            encoding.set_buffer(Some(&indices_buffer), vector_offset, 6);
            if gpu_bound {
                encoding.set_buffer(Some(&bounds), 4, 7);
            }
            encoding.set_buffer(Some(&failure), 0, 30);
            encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: (parameters[0] as usize).div_ceil(rows_per_group),
                    height: indices.len(),
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            encoding.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            // SAFETY: owned shared test buffers have completed GPU writers.
            let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count + 2) };
            assert_eq!(values[0], 7.25, "IQ indexed prefix guard changed");
            assert_eq!(values[count + 1], 7.25, "IQ indexed row tail guard changed");
            if gpu_bound {
                let flags =
                    unsafe { std::slice::from_raw_parts(bounds.contents().as_ptr().cast::<u32>(), batches + 2) };
                assert_eq!(flags[0], 12345, "IQ bound prefix guard changed");
                assert_eq!(flags[batches + 1], 12345, "IQ bound suffix guard changed");
                for batch in 0..batches {
                    let start = batch * parameters[1] as usize;
                    let end = start + parameters[1] as usize;
                    let expected = u32::from(
                        input_storage[usize::from(gpu_bound) + start..usize::from(gpu_bound) + end]
                            .iter()
                            .any(|x| !x.is_finite() || x.abs() > 1.0e20),
                    );
                    assert_eq!(flags[batch + 1], expected, "IQ GPU bound predicate or input offset changed");
                }
            }
            if unsafe { *failure.contents().as_ptr().cast::<u32>() } != 0 || values.iter().any(|x| !x.is_finite()) {
                return Err(Error::Invalid("invalid IQ indexed test result"));
            }
            Ok(values[1..count + 1].to_vec())
        })
    }

    fn run_safe_iq_selected_test(
        kind: u32,
        codes: &[u8],
        input: &[f32],
        rows: usize,
        columns: usize,
        batch_rows: usize,
    ) -> Vec<f32> {
        objc2::rc::autoreleasepool(|_| {
            let executor = executor().unwrap();
            let pipeline = safe_iq_test_pipeline(kind, "hybrid_iq_matvec_vector8", 4);
            let (_, bytes) = weights::block_layout(kind).unwrap();
            let parameters = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, batch_rows as u32, 0];
            let codes = buffer(executor, codes).unwrap();
            let input = buffer(executor, bytemuck::cast_slice(input)).unwrap();
            let output = executor.device.new_buffer(rows * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
            let command = executor.queue.command_buffer().unwrap();
            let encoding = command.compute_command_encoder().unwrap();
            encoding.set_compute_pipeline_state(&pipeline);
            for (index, value) in [&codes, &input, &output].into_iter().enumerate() {
                encoding.set_buffer(Some(value), 0, index);
            }
            encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
            let offsets = [0u32; 4];
            encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: rows.div_ceil(8),
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
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            // SAFETY: the owned shared output has a completed GPU writer.
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows) }.to_vec()
        })
    }

    #[stark_test]
    fn iq_indexed_multirow_preserves_safe_selected_bits_tails_and_failures() {
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17u32, 18] {
            let original = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_vector8", 4);
            let candidates = [
                ("multirow2", 2usize, safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 2)),
                ("multirow4", 4, safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4)),
                ("staged4", 4, safe_staged_iq_test_pipeline(kind)),
            ];
            for columns in [256usize, 2048, 4096] {
                let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                let matrix_bytes = codes.len() / experts;
                let selected_codes = selected
                    .iter()
                    .flat_map(|&e| codes[e as usize * matrix_bytes..(e as usize + 1) * matrix_bytes].iter().copied())
                    .collect::<Vec<_>>();
                for batched in [false, true] {
                    let input = (0..columns
                        * if batched {
                            selected.len()
                        } else {
                            1
                        })
                        .map(|i| (i as f32 * 0.019).sin())
                        .collect::<Vec<_>>();
                    let parameters = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                    let reference = run_safe_iq_selected_test(
                        kind,
                        &selected_codes,
                        &input,
                        rows * selected.len(),
                        columns,
                        if batched {
                            rows
                        } else {
                            0
                        },
                    );
                    let indexed =
                        run_iq_indexed_test(&original, &codes, &input, &selected, parameters, 8, 256).unwrap();
                    assert!(indexed.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
                    for (schedule, n, candidate) in &candidates {
                        let actual =
                            run_iq_indexed_test(candidate, &codes, &input, &selected, parameters, n * 2, 64).unwrap();
                        assert!(
                            actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "IQ changed Safe bits kind={kind} schedule={schedule} rows_per_simd={n} columns={columns} batched={batched}"
                        );
                        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                            let mut invalid = input.clone();
                            invalid[if batched {
                                columns + 17
                            } else {
                                17
                            }] = poison;
                            assert!(
                                run_iq_indexed_test(candidate, &codes, &invalid, &selected, parameters, n * 2, 64)
                                    .is_err()
                            );
                        }
                        let parameters = [rows as u32, columns as u32, experts as u32, 0];
                        assert!(
                            run_iq_indexed_test(
                                candidate,
                                &codes,
                                &input[..columns],
                                &[experts as u32],
                                parameters,
                                n * 2,
                                64
                            )
                            .is_err()
                        );
                    }
                }
            }
        }
    }

    #[stark_test]
    fn iq_grouped32_arithmetic_matches_f64_and_rejects_unproved_input() {
        let selected = [3u32, 1, 3];
        let (experts, rows) = (4usize, 13usize);
        for kind in [17, 18] {
            for mode in [1, 2] {
                // Source-level sign factoring is a numerical candidate: Safe
                // compilation may contract its shortened expression to FMA.
                let mut bit_matches = 0usize;
                let mut total_values = 0usize;
                let mut maximum_absolute_error = 0.0f64;
                let mut maximum_relative_rms = 0.0f64;
                let mut factor_fma_bit_matches = 0usize;
                let mut factor_fma_maximum_absolute_error = 0.0f64;
                for invalid in
                    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::from_bits(1.0e20f32.to_bits() + 1)]
                {
                    assert!(safe_iq_arithmetic_pipeline(kind, mode, &[invalid], 0).is_err());
                }
                assert!(safe_iq_arithmetic_pipeline(kind, mode, &[1.0], 1).is_err());
                for columns in [2048usize, 4096] {
                    for batched in [false, true] {
                        for magnitude in [1.0f32, 1.0e20] {
                            let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                            let input: Vec<_> = (0..columns
                                * if batched {
                                    3
                                } else {
                                    1
                                })
                                .map(|i| (i as f32 * 0.019).sin() * magnitude)
                                .collect();
                            let original = safe_iq_sweep_pipeline(kind, 4, true, &input, 2).unwrap();
                            let candidate = safe_iq_arithmetic_pipeline(kind, mode, &input, 2).unwrap();
                            let p = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                            let reference = run_iq_indexed_test_with_offsets(
                                &original, &codes, &input, &selected, p, 8, 64, false, 2,
                            )
                            .unwrap();
                            let actual = run_iq_indexed_test_with_offsets(
                                &candidate, &codes, &input, &selected, p, 8, 64, false, 2,
                            )
                            .unwrap();
                            bit_matches +=
                                actual.iter().zip(&reference).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                            total_values += actual.len();
                            maximum_absolute_error = maximum_absolute_error.max(
                                actual
                                    .iter()
                                    .zip(&reference)
                                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                                    .fold(0.0, f64::max),
                            );
                            if mode == 1 {
                                let fma = safe_iq_arithmetic_pipeline(kind, 2, &input, 2).unwrap();
                                let fused = run_iq_indexed_test_with_offsets(
                                    &fma, &codes, &input, &selected, p, 8, 64, false, 2,
                                )
                                .unwrap();
                                factor_fma_bit_matches +=
                                    actual.iter().zip(&fused).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                                factor_fma_maximum_absolute_error = factor_fma_maximum_absolute_error.max(
                                    actual
                                        .iter()
                                        .zip(&fused)
                                        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                                        .fold(0.0, f64::max),
                                );
                            }
                            let matrix_bytes = codes.len() / experts;
                            let selected_codes: Vec<_> = selected
                                .iter()
                                .flat_map(|&e| {
                                    codes[e as usize * matrix_bytes..(e as usize + 1) * matrix_bytes].iter().copied()
                                })
                                .collect();
                            let oracle = packed_f64_oracle(
                                kind,
                                &selected_codes,
                                rows * 3,
                                columns,
                                &input,
                                if batched {
                                    rows
                                } else {
                                    0
                                },
                            );
                            for (i, &(expected, absolute)) in oracle.iter().enumerate() {
                                assert!(
                                    (f64::from(actual[i]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                                    "IQ arithmetic F64 bound kind={kind} mode={mode}"
                                );
                            }
                            let error: f64 = actual
                                .iter()
                                .zip(&reference)
                                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                                .sum();
                            let norm: f64 = reference.iter().map(|&x| f64::from(x).powi(2)).sum();
                            let relative_rms = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
                            maximum_relative_rms = maximum_relative_rms.max(relative_rms);
                            assert!(relative_rms <= 2e-5, "IQ arithmetic relative RMS");
                        }
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"iq_arithmetic_numerical_trial":true,"kind":kind,"mode":mode,
                    "bit_matches_vs_original":bit_matches,"total_values":total_values,
                    "maximum_absolute_error_vs_original":maximum_absolute_error,"maximum_relative_rms_vs_original":maximum_relative_rms,
                    "factor_fma_bit_matches":if mode==1 {Some(factor_fma_bit_matches)}else{None},
                    "factor_fma_maximum_absolute_error":if mode==1 {Some(factor_fma_maximum_absolute_error)}else{None},
                    "matching_fma_is_not_compiler_proof":true})
                );
            }
        }
    }

    #[stark_test]
    fn iq_grouped32_rows_typed_preserve_bits_f64_and_alignment() {
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17, 18] {
            for columns in [2048usize, 4096] {
                let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                for batched in [false, true] {
                    let input: Vec<_> = (0..columns
                        * if batched {
                            3
                        } else {
                            1
                        })
                        .map(|i| (i as f32 * 0.019).sin())
                        .collect();
                    let p = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                    let baseline = safe_iq_sweep_pipeline(kind, 4, false, &input, 0).unwrap();
                    let reference = run_iq_indexed_test(&baseline, &codes, &input, &selected, p, 8, 64).unwrap();
                    let matrix_bytes = codes.len() / experts;
                    let selected_codes: Vec<_> = selected
                        .iter()
                        .flat_map(|&e| {
                            codes[e as usize * matrix_bytes..(e as usize + 1) * matrix_bytes].iter().copied()
                        })
                        .collect();
                    let oracle = packed_f64_oracle(
                        kind,
                        &selected_codes,
                        rows * 3,
                        columns,
                        &input,
                        if batched {
                            rows
                        } else {
                            0
                        },
                    );
                    for count in [1, 2, 4, 8] {
                        for typed in [false, true] {
                            assert!(safe_iq_sweep_pipeline(kind, count, true, &input, 1).is_err());
                            let pipeline = safe_iq_sweep_pipeline(kind, count, typed, &input, 2).unwrap();
                            let actual = run_iq_indexed_test_with_offsets(
                                &pipeline,
                                &codes,
                                &input,
                                &selected,
                                p,
                                count * 2,
                                64,
                                false,
                                2,
                            )
                            .unwrap();
                            assert!(
                                actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                                "IQ rows/typed changed grouped32 bits kind={kind} count={count} typed={typed}"
                            );
                            for (i, &(expected, absolute)) in oracle.iter().enumerate() {
                                assert!(
                                    (f64::from(actual[i]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                                    "IQ sweep F64 bound"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[stark_test]
    fn iq_grouped32_bound_diagnostic_checks_host_input_and_normal_bits() {
        for kind in [17, 18] {
            for invalid in
                [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::from_bits(1.0e20f32.to_bits() + 1)]
            {
                assert!(safe_bound_grouped32_iq_test_pipeline(kind, &[invalid]).is_err());
            }
            assert!(safe_bound_grouped32_iq_test_pipeline(kind, &[]).is_err());
            let guarded = safe_grouped32_iq_test_pipeline(kind);
            for columns in [256usize, 2048, 4096] {
                let (experts, rows) = (4usize, 13usize);
                let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                for batched in [false, true] {
                    let input: Vec<_> = (0..columns
                        * if batched {
                            3
                        } else {
                            1
                        })
                        .map(|i| (i as f32 * 0.019).sin())
                        .collect();
                    let bound = safe_bound_grouped32_iq_test_pipeline(kind, &input).unwrap();
                    let parameters = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                    let selected = [3u32, 1, 3];
                    let reference =
                        run_iq_indexed_test(&guarded, &codes, &input, &selected, parameters, 8, 64).unwrap();
                    let actual = run_iq_indexed_test(&bound, &codes, &input, &selected, parameters, 8, 64).unwrap();
                    assert!(
                        actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "bound IQ diagnostic changed normal grouped32 bits kind={kind} columns={columns} batched={batched}"
                    );
                    let squared_error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!((squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5);
                }
            }
        }
    }

    #[stark_test]
    fn iq_grouped32_matches_f64_tails_batches_and_extreme_fallback() {
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17u32, 18] {
            let original = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_vector8", 4);
            let multirow = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4);
            let grouped = safe_grouped32_iq_test_pipeline(kind);
            for columns in [256usize, 2048, 4096] {
                let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                let matrix_bytes = codes.len() / experts;
                let selected_codes: Vec<_> = selected
                    .iter()
                    .flat_map(|&e| codes[e as usize * matrix_bytes..(e as usize + 1) * matrix_bytes].iter().copied())
                    .collect();
                for batched in [false, true] {
                    let input: Vec<_> = (0..columns
                        * if batched {
                            selected.len()
                        } else {
                            1
                        })
                        .map(|i| (i as f32 * 0.019).sin())
                        .collect();
                    let parameters = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                    let reference =
                        run_iq_indexed_test(&original, &codes, &input, &selected, parameters, 8, 256).unwrap();
                    let current = run_iq_indexed_test(&multirow, &codes, &input, &selected, parameters, 8, 64).unwrap();
                    assert!(current.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
                    let actual = run_iq_indexed_test(&grouped, &codes, &input, &selected, parameters, 8, 64).unwrap();
                    let oracle = packed_f64_oracle(
                        kind,
                        &selected_codes,
                        rows * selected.len(),
                        columns,
                        &input,
                        if batched {
                            rows
                        } else {
                            0
                        },
                    );
                    for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "IQ grouped32 F64 bound kind={kind} columns={columns} batched={batched}"
                        );
                    }
                    let squared_error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!(
                        (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5,
                        "IQ grouped32 relative RMS kind={kind} columns={columns}"
                    );
                    let large: Vec<_> = input.iter().map(|value| value * 1.0e21f32).collect();
                    let large_reference =
                        run_iq_indexed_test(&original, &codes, &large, &selected, parameters, 8, 256).unwrap();
                    let large_actual =
                        run_iq_indexed_test(&grouped, &codes, &large, &selected, parameters, 8, 64).unwrap();
                    assert!(
                        large_actual.iter().zip(&large_reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "IQ grouped32 large finite fallback changed bits kind={kind}"
                    );
                    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut poisoned = input.clone();
                        poisoned[columns - 1] = invalid;
                        assert!(
                            run_iq_indexed_test(&grouped, &codes, &poisoned, &selected, parameters, 8, 64).is_err()
                        );
                    }
                    assert!(run_iq_indexed_test(&grouped, &codes, &input, &[u32::MAX], parameters, 8, 64).is_err());
                    assert!(run_iq_indexed_test(&grouped, &codes, &input, &selected, parameters, 8, 64).is_ok());
                }
                let mut zeros = codes.clone();
                for block in zeros.chunks_exact_mut(weights::block_layout(kind).unwrap().1) {
                    block[..2].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
                }
                let input = vec![f32::MAX; columns];
                let p = [rows as u32, columns as u32, experts as u32, 0];
                let reference = run_iq_indexed_test(&original, &zeros, &input, &selected, p, 8, 256).unwrap();
                let actual = run_iq_indexed_test(&grouped, &zeros, &input, &selected, p, 8, 64).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "IQ grouped32 zero-times-MAX fallback changed bits kind={kind}"
                );
            }
        }
    }

    #[stark_test]
    #[ignore]
    fn iq_typed16_graph_admits_even_offsets_and_keeps_odd_byte_cache() {
        assert!(
            iq_typed16_schedule(MatrixFormat::Packed(17), 2048, 2),
            "typed integration requires IQ_GPU_BOUND=1 and IQ_TYPED16=1 in a fresh process"
        );
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17, 18] {
            for columns in [2048usize, 4096] {
                let bytes = finite_packed_codes(kind, experts * rows, columns, 7);
                let normal: Vec<_> = (0..columns).map(|i| (i as f32 * 0.019).sin()).collect();
                let large: Vec<_> = normal.iter().map(|x| x * 1.0e21f32).collect();
                let p = [rows as u32, columns as u32, experts as u32, 0];
                let byte = safe_bound_grouped32_iq_test_pipeline(kind, &normal).unwrap();
                let legacy = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4);
                let expected = run_iq_indexed_test(&byte, &bytes, &normal, &selected, p, 8, 64).unwrap();
                let expected_large = run_iq_indexed_test(&legacy, &bytes, &large, &selected, p, 8, 64).unwrap();
                for offset in [2usize, 1, 4, 3, 2] {
                    assert_eq!(iq_typed16_schedule(MatrixFormat::Packed(kind), columns, offset), offset % 2 == 0);
                    let mut padded = vec![0x75u8; offset];
                    padded.extend_from_slice(&bytes);
                    let matrix = MappedMatrix {
                        codes: MappedSlice {
                            buffer: buffer(executor().unwrap(), &padded).unwrap(),
                            offset,
                            length: bytes.len(),
                            registered_length: padded.len(),
                        },
                        planes: None,
                    };
                    let mut graph = DecodeGraph::new().unwrap();
                    let input = DecodeGraph::upload(&normal).unwrap();
                    let large = DecodeGraph::upload(&large).unwrap();
                    let indices = DecodeGraph::upload(&selected.map(f32::from_bits)).unwrap();
                    let first = graph
                        .indexed(
                            &matrix,
                            MatrixFormat::Packed(kind),
                            [experts, rows, columns],
                            &indices,
                            selected.len(),
                            &input,
                            false,
                        )
                        .unwrap();
                    graph.copy_into(&large, &input).unwrap();
                    let second = graph
                        .indexed(
                            &matrix,
                            MatrixFormat::Packed(kind),
                            [experts, rows, columns],
                            &indices,
                            selected.len(),
                            &input,
                            false,
                        )
                        .unwrap();
                    let results = graph.finish(&[first, second]).unwrap();
                    assert!(
                        results[0].iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "typed/byte cache admission changed normal bits offset={offset}"
                    );
                    assert!(
                        results[1].iter().zip(&expected_large).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "typed alignment path changed legacy extreme fallback offset={offset}"
                    );
                }
                let executor = executor().unwrap();
                assert!(executor.iq_bound_indexed_pipelines.lock().unwrap().contains_key(&MatrixFormat::Packed(kind)));
                assert!(
                    executor.iq_bound_typed_indexed_pipelines.lock().unwrap().contains_key(&MatrixFormat::Packed(kind))
                );
            }
        }
    }

    #[stark_test]
    #[ignore]
    fn iq_signed_factor_graph_admits_even_offsets_and_keeps_odd_byte_cache() {
        assert!(
            iq_signed_factor_schedule(MatrixFormat::Packed(17), 2048, 2),
            "signed-factor integration requires IQ_GPU_BOUND=1, IQ_TYPED16=1, IQ_SIGNED_FACTOR=1 in a fresh process"
        );
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17, 18] {
            for columns in [2048usize, 4096] {
                let bytes = finite_packed_codes(kind, experts * rows, columns, 7);
                let normal: Vec<_> = (0..columns).map(|i| (i as f32 * 0.019).sin()).collect();
                let large: Vec<_> = normal.iter().map(|x| x * 1.0e21f32).collect();
                let p = [rows as u32, columns as u32, experts as u32, 0];
                let byte = safe_bound_grouped32_iq_test_pipeline(kind, &normal).unwrap();
                let legacy = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4);
                let expected = run_iq_indexed_test(&byte, &bytes, &normal, &selected, p, 8, 64).unwrap();
                let factor = safe_iq_arithmetic_pipeline(kind, 1, &normal, 2).unwrap();
                let factor_expected = run_iq_indexed_test(&factor, &bytes, &normal, &selected, p, 8, 64).unwrap();
                let expected_large = run_iq_indexed_test(&legacy, &bytes, &large, &selected, p, 8, 64).unwrap();
                for offset in [2usize, 1, 4, 3, 2] {
                    assert_eq!(iq_signed_factor_schedule(MatrixFormat::Packed(kind), columns, offset), offset % 2 == 0);
                    let mut padded = vec![0x75u8; offset];
                    padded.extend_from_slice(&bytes);
                    let matrix = MappedMatrix {
                        codes: MappedSlice {
                            buffer: buffer(executor().unwrap(), &padded).unwrap(),
                            offset,
                            length: bytes.len(),
                            registered_length: padded.len(),
                        },
                        planes: None,
                    };
                    let mut graph = DecodeGraph::new().unwrap();
                    let input = DecodeGraph::upload(&normal).unwrap();
                    let large = DecodeGraph::upload(&large).unwrap();
                    let indices = DecodeGraph::upload(&selected.map(f32::from_bits)).unwrap();
                    let first = graph
                        .indexed(
                            &matrix,
                            MatrixFormat::Packed(kind),
                            [experts, rows, columns],
                            &indices,
                            selected.len(),
                            &input,
                            false,
                        )
                        .unwrap();
                    graph.copy_into(&large, &input).unwrap();
                    let second = graph
                        .indexed(
                            &matrix,
                            MatrixFormat::Packed(kind),
                            [experts, rows, columns],
                            &indices,
                            selected.len(),
                            &input,
                            false,
                        )
                        .unwrap();
                    let results = graph.finish(&[first, second]).unwrap();
                    assert!(
                        results[0]
                            .iter()
                            .zip(if offset % 2 == 0 {
                                &factor_expected
                            } else {
                                &expected
                            })
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "typed/byte cache admission changed normal bits offset={offset}"
                    );
                    assert!(
                        results[1].iter().zip(&expected_large).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "typed alignment path changed legacy extreme fallback offset={offset}"
                    );
                }
                let executor = executor().unwrap();
                assert!(executor.iq_bound_indexed_pipelines.lock().unwrap().contains_key(&MatrixFormat::Packed(kind)));
                assert!(
                    executor
                        .iq_bound_factor_indexed_pipelines
                        .lock()
                        .unwrap()
                        .contains_key(&MatrixFormat::Packed(kind))
                );
            }
        }
    }

    #[stark_test]
    #[ignore]
    fn iq_gpu_bound_graph_rechecks_mutated_input_across_projection_levels() {
        assert!(
            iq_bound_schedule(MatrixFormat::Packed(17), 2048),
            "integration diagnostic requires STARK_GLM53_IQ_GPU_BOUND=1 in a fresh process"
        );
        let (experts, rows, columns) = (4usize, 13usize, 2048usize);
        let selected = [3u32, 1, 3];
        for kind in [17, 18] {
            let bytes = finite_packed_codes(kind, experts * rows, columns, 7);
            let mut padded = vec![0x75u8; 16];
            padded.extend_from_slice(&bytes);
            let codes = buffer(executor().unwrap(), &padded).unwrap();
            let matrix = MappedMatrix {
                codes: MappedSlice {
                    buffer: codes,
                    offset: 16,
                    length: bytes.len(),
                    registered_length: padded.len(),
                },
                planes: None,
            };
            let normal: Vec<_> = (0..columns).map(|i| (i as f32 * 0.019).sin()).collect();
            let large: Vec<_> = normal.iter().map(|x| x * 1.0e21f32).collect();
            let reference = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4);
            let p = [rows as u32, columns as u32, experts as u32, 0];
            let expected_normal = run_iq_indexed_test(&reference, &bytes, &normal, &selected, p, 8, 64).unwrap();
            let expected_large = run_iq_indexed_test(&reference, &bytes, &large, &selected, p, 8, 64).unwrap();
            for concurrent in [false, true] {
                let mut graph = if concurrent {
                    DecodeGraph::new_parallel()
                } else {
                    DecodeGraph::new()
                }
                .unwrap();
                let mut padded = vec![7.25];
                padded.extend_from_slice(&normal);
                let input = DecodeGraph::upload(&padded).unwrap().slice(1, columns).unwrap();
                let large = DecodeGraph::upload(&large).unwrap();
                let mut ids = vec![f32::from_bits(999)];
                ids.extend(selected.map(f32::from_bits));
                let indices = DecodeGraph::upload(&ids).unwrap().slice(1, selected.len()).unwrap();
                graph.begin_projection_level().unwrap();
                let first = graph
                    .indexed(
                        &matrix,
                        MatrixFormat::Packed(kind),
                        [experts, rows, columns],
                        &indices,
                        selected.len(),
                        &input,
                        false,
                    )
                    .unwrap();
                graph.end_projection_level().unwrap();
                graph.copy_into(&large, &input).unwrap();
                graph.begin_projection_level().unwrap();
                let second = graph
                    .indexed(
                        &matrix,
                        MatrixFormat::Packed(kind),
                        [experts, rows, columns],
                        &indices,
                        selected.len(),
                        &input,
                        false,
                    )
                    .unwrap();
                graph.end_projection_level().unwrap();
                let results = graph.finish(&[first, second]).unwrap();
                let error: f64 =
                    results[0].iter().zip(&expected_normal).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                let magnitude: f64 = expected_normal.iter().map(|&x| f64::from(x).powi(2)).sum();
                assert!((error / magnitude.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5);
                assert!(
                    results[1].iter().zip(&expected_large).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "GPU bounds reused after input overwrite or projection dependency failed"
                );
            }
        }
    }

    #[stark_test]
    fn iq_gpu_bound_prepass_preserves_f64_offsets_and_exact_extreme_fallback() {
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17u32, 18] {
            let original = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_vector8", 4);
            let multirow = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4);
            let grouped = safe_gpu_bound_iq_test_pipeline(kind);
            for columns in [256usize, 2048, 4096] {
                let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                let matrix_bytes = codes.len() / experts;
                let selected_codes: Vec<_> = selected
                    .iter()
                    .flat_map(|&e| codes[e as usize * matrix_bytes..(e as usize + 1) * matrix_bytes].iter().copied())
                    .collect();
                for batched in [false, true] {
                    let input: Vec<_> = (0..columns
                        * if batched {
                            selected.len()
                        } else {
                            1
                        })
                        .map(|i| (i as f32 * 0.019).sin())
                        .collect();
                    let parameters = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                    let reference =
                        run_iq_indexed_test(&original, &codes, &input, &selected, parameters, 8, 256).unwrap();
                    let current = run_iq_indexed_test(&multirow, &codes, &input, &selected, parameters, 8, 64).unwrap();
                    assert!(current.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
                    let actual =
                        run_iq_indexed_test_with_bound(&grouped, &codes, &input, &selected, parameters, 8, 64, true)
                            .unwrap();
                    let oracle = packed_f64_oracle(
                        kind,
                        &selected_codes,
                        rows * selected.len(),
                        columns,
                        &input,
                        if batched {
                            rows
                        } else {
                            0
                        },
                    );
                    for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "IQ GPU-bound grouped32 F64 bound kind={kind} columns={columns} batched={batched}"
                        );
                    }
                    let squared_error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!(
                        (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5,
                        "IQ GPU-bound grouped32 relative RMS kind={kind} columns={columns}"
                    );
                    let large: Vec<_> = input.iter().map(|value| value * 1.0e21f32).collect();
                    let large_reference =
                        run_iq_indexed_test(&original, &codes, &large, &selected, parameters, 8, 256).unwrap();
                    let large_actual =
                        run_iq_indexed_test_with_bound(&grouped, &codes, &large, &selected, parameters, 8, 64, true)
                            .unwrap();
                    assert!(
                        large_actual.iter().zip(&large_reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "IQ GPU-bound grouped32 large finite fallback changed bits kind={kind}"
                    );
                    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut poisoned = input.clone();
                        poisoned[columns - 1] = invalid;
                        assert!(
                            run_iq_indexed_test_with_bound(
                                &grouped, &codes, &poisoned, &selected, parameters, 8, 64, true
                            )
                            .is_err()
                        );
                    }
                    assert!(
                        run_iq_indexed_test_with_bound(&grouped, &codes, &input, &[u32::MAX], parameters, 8, 64, true)
                            .is_err()
                    );
                    assert!(
                        run_iq_indexed_test_with_bound(&grouped, &codes, &input, &selected, parameters, 8, 64, true)
                            .is_ok()
                    );
                }
                let mut zeros = codes.clone();
                for block in zeros.chunks_exact_mut(weights::block_layout(kind).unwrap().1) {
                    block[..2].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
                }
                let input = vec![f32::MAX; columns];
                let p = [rows as u32, columns as u32, experts as u32, 0];
                let reference = run_iq_indexed_test(&original, &zeros, &input, &selected, p, 8, 256).unwrap();
                let actual =
                    run_iq_indexed_test_with_bound(&grouped, &zeros, &input, &selected, p, 8, 64, true).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "IQ GPU-bound grouped32 zero-times-MAX fallback changed bits kind={kind}"
                );
            }
        }
    }

    #[stark_test]
    fn iq_checked_pipeline_hints_preserve_bits_and_failure() {
        for kind in [17, 18] {
            let (rows, columns, experts) = (13usize, 2048usize, 4usize);
            let codes = finite_packed_codes(kind, rows * experts, columns, 7);
            let input = (0..columns).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>();
            let p = [rows as u32, columns as u32, experts as u32, 0];
            for factor in [false, true] {
                let normal = safe_iq_checked_hint_pipeline(kind, factor, false);
                let hinted = safe_iq_checked_hint_pipeline(kind, factor, true);
                let reference =
                    run_iq_indexed_test_with_bound(&normal, &codes, &input, &[3, 1, 3], p, 8, 64, true).unwrap();
                let actual =
                    run_iq_indexed_test_with_bound(&hinted, &codes, &input, &[3, 1, 3], p, 8, 64, true).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "IQ hint changed corresponding arithmetic bits"
                );
                for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut invalid = input.clone();
                    invalid[17] = poison;
                    assert!(
                        run_iq_indexed_test_with_bound(&hinted, &codes, &invalid, &[3, 1, 3], p, 8, 64, true).is_err()
                    );
                }
                assert!(run_iq_indexed_test_with_bound(&hinted, &codes, &input, &[u32::MAX], p, 8, 64, true).is_err());
            }
        }
    }

    #[stark_test]
    fn iq_signed_factor_prepass_preserves_f64_offsets_and_exact_extreme_fallback() {
        let (experts, rows) = (4usize, 13usize);
        let selected = [3u32, 1, 3];
        for kind in [17u32, 18] {
            let original = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_vector8", 4);
            let multirow = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_multirow", 4);
            let grouped = safe_iq_checked_hint_pipeline(kind, true, false);
            for columns in [256usize, 2048, 4096] {
                let codes = finite_packed_codes(kind, experts * rows, columns, 7);
                let matrix_bytes = codes.len() / experts;
                let selected_codes: Vec<_> = selected
                    .iter()
                    .flat_map(|&e| codes[e as usize * matrix_bytes..(e as usize + 1) * matrix_bytes].iter().copied())
                    .collect();
                for batched in [false, true] {
                    let input: Vec<_> = (0..columns
                        * if batched {
                            selected.len()
                        } else {
                            1
                        })
                        .map(|i| (i as f32 * 0.019).sin())
                        .collect();
                    let parameters = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                    let reference =
                        run_iq_indexed_test(&original, &codes, &input, &selected, parameters, 8, 256).unwrap();
                    let current = run_iq_indexed_test(&multirow, &codes, &input, &selected, parameters, 8, 64).unwrap();
                    assert!(current.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
                    let actual =
                        run_iq_indexed_test_with_bound(&grouped, &codes, &input, &selected, parameters, 8, 64, true)
                            .unwrap();
                    let oracle = packed_f64_oracle(
                        kind,
                        &selected_codes,
                        rows * selected.len(),
                        columns,
                        &input,
                        if batched {
                            rows
                        } else {
                            0
                        },
                    );
                    for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "IQ GPU-bound grouped32 F64 bound kind={kind} columns={columns} batched={batched}"
                        );
                    }
                    let squared_error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!(
                        (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5,
                        "IQ GPU-bound grouped32 relative RMS kind={kind} columns={columns}"
                    );
                    let large: Vec<_> = input.iter().map(|value| value * 1.0e21f32).collect();
                    let large_reference =
                        run_iq_indexed_test(&original, &codes, &large, &selected, parameters, 8, 256).unwrap();
                    let large_actual =
                        run_iq_indexed_test_with_bound(&grouped, &codes, &large, &selected, parameters, 8, 64, true)
                            .unwrap();
                    assert!(
                        large_actual.iter().zip(&large_reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "IQ GPU-bound grouped32 large finite fallback changed bits kind={kind}"
                    );
                    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut poisoned = input.clone();
                        poisoned[columns - 1] = invalid;
                        assert!(
                            run_iq_indexed_test_with_bound(
                                &grouped, &codes, &poisoned, &selected, parameters, 8, 64, true
                            )
                            .is_err()
                        );
                    }
                    assert!(
                        run_iq_indexed_test_with_bound(&grouped, &codes, &input, &[u32::MAX], parameters, 8, 64, true)
                            .is_err()
                    );
                    assert!(
                        run_iq_indexed_test_with_bound(&grouped, &codes, &input, &selected, parameters, 8, 64, true)
                            .is_ok()
                    );
                }
                let mut zeros = codes.clone();
                for block in zeros.chunks_exact_mut(weights::block_layout(kind).unwrap().1) {
                    block[..2].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
                }
                let input = vec![f32::MAX; columns];
                let p = [rows as u32, columns as u32, experts as u32, 0];
                let reference = run_iq_indexed_test(&original, &zeros, &input, &selected, p, 8, 256).unwrap();
                let actual =
                    run_iq_indexed_test_with_bound(&grouped, &zeros, &input, &selected, p, 8, 64, true).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "IQ GPU-bound grouped32 zero-times-MAX fallback changed bits kind={kind}"
                );
            }
        }
    }

    #[stark_test]
    fn iq4_vector_matches_f64_and_rejects_nonfinite_inputs() {
        let (rows, batches) = (13usize, 3usize);
        for columns in [256usize, 768, 2048, 4096] {
            let codes = finite_packed_codes(23, rows * batches, columns, 41);
            let input = (0..columns * batches).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>();
            let parameters = [(rows * batches) as u32, columns as u32, 23, 256, 136, 0, rows as u32, 0];
            let run = |input: &[f32], schedule| {
                objc2::rc::autoreleasepool(|_| executor().unwrap().run_inner(&codes, input, parameters, None, schedule))
            };
            let scalar = run(&input, TestMatrixSchedule::Scalar).unwrap();
            let vector = run(&input, TestMatrixSchedule::IQ4Vector).unwrap();
            let oracle = packed_f64_oracle(23, &codes, rows * batches, columns, &input, rows);
            for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                let bound = 2e-6 * absolute.max(1.0);
                assert!((f64::from(scalar[row]) - expected).abs() <= bound, "IQ4 scalar F64 columns={columns}");
                assert!((f64::from(vector[row]) - expected).abs() <= bound, "IQ4 vector F64 columns={columns}");
            }
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = input.clone();
                invalid[columns + 17] = value;
                assert!(run(&invalid, TestMatrixSchedule::Scalar).is_err());
                assert!(run(&invalid, TestMatrixSchedule::IQ4Vector).is_err());
            }
        }
    }

    #[stark_test]
    fn packed_indexed_schedules_match_selected_views_and_reject_bad_experts() {
        let (experts, rows, columns) = (4usize, 13usize, 4096usize);
        let selected = [3usize, 1, 3];
        for kind in [12, 13, 14, 17, 18, 23] {
            let codes = finite_packed_codes(kind, experts * rows, columns, 7);
            let matrix = MappedMatrix {
                codes: MappedSlice {
                    buffer: buffer(executor().unwrap(), &codes).unwrap(),
                    offset: 0,
                    length: codes.len(),
                    registered_length: codes.len(),
                },
                planes: None,
            };
            for batched in [false, true] {
                let input = (0..columns
                    * if batched {
                        selected.len()
                    } else {
                        1
                    })
                    .map(|i| (i as f32 * 0.019).sin())
                    .collect::<Vec<_>>();
                let mut graph = DecodeGraph::new().unwrap();
                let indices = DecodeGraph::upload(&selected.map(|id| f32::from_bits(id as u32))).unwrap();
                let output = graph
                    .indexed(
                        &matrix,
                        MatrixFormat::Packed(kind),
                        [experts, rows, columns],
                        &indices,
                        selected.len(),
                        &DecodeGraph::upload(&input).unwrap(),
                        batched,
                    )
                    .unwrap();
                let actual = graph.finish(&[output]).unwrap().remove(0);
                let expert_bytes = codes.len() / experts;
                for (slot, expert) in selected.into_iter().enumerate() {
                    let input_start = if batched {
                        slot * columns
                    } else {
                        0
                    };
                    let expected = if kind == 13 {
                        // Indexed Q5 keeps the legacy selected-view helper,
                        // independently of the dense grouped32 experiment.
                        let executor = executor().unwrap();
                        let format = MatrixFormat::Packed(kind);
                        let factored = factored_schedule(format);
                        let original = executor
                            .cached_pipeline(
                                format,
                                if factored {
                                    "hybrid_packed_factored_matvec"
                                } else {
                                    "hybrid_packed_block_matvec"
                                },
                                if factored {
                                    &executor.factored_test_pipelines
                                } else {
                                    &executor.block_test_pipelines
                                },
                            )
                            .unwrap();
                        run_kquant_matrix_test(
                            &original,
                            &codes[expert * expert_bytes..(expert + 1) * expert_bytes],
                            &input[input_start..input_start + columns],
                            [rows as u32, columns as u32, kind, 256, 176, 0, 0, 0],
                            2,
                        )
                        .unwrap()
                    } else if kind == 14 {
                        // Indexed Q6 retains the original block chronology.
                        // Its selected-view production policy can independently
                        // choose the reassociated grouped16 schedule.
                        let original = executor()
                            .unwrap()
                            .cached_pipeline(
                                MatrixFormat::Packed(kind),
                                "hybrid_packed_block_matvec",
                                &executor().unwrap().block_test_pipelines,
                            )
                            .unwrap();
                        run_kquant_matrix_test(
                            &original,
                            &codes[expert * expert_bytes..(expert + 1) * expert_bytes],
                            &input[input_start..input_start + columns],
                            [rows as u32, columns as u32, kind, 256, 210, 0, 0, 0],
                            2,
                        )
                        .unwrap()
                    } else {
                        matvec(
                            &codes[expert * expert_bytes..(expert + 1) * expert_bytes],
                            kind,
                            rows,
                            columns,
                            &input[input_start..input_start + columns],
                        )
                        .unwrap()
                    };
                    assert!(
                        actual[slot * rows..(slot + 1) * rows]
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "indexed packed arithmetic changed kind={kind} batched={batched}"
                    );
                }
            }
            let mut invalid = DecodeGraph::new().unwrap();
            let indices = DecodeGraph::upload(&[f32::from_bits(experts as u32)]).unwrap();
            let input = DecodeGraph::upload(&vec![0.0; columns]).unwrap();
            let output = invalid
                .indexed(&matrix, MatrixFormat::Packed(kind), [experts, rows, columns], &indices, 1, &input, false)
                .unwrap();
            assert!(invalid.finish(&[output]).is_err(), "indexed packed out-of-range expert kind={kind}");
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
