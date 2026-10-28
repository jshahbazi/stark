//! Native packed-weight GEMV. Only the selected expert matrix is submitted;
//! weights are never expanded into a full-precision model in memory.

use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
};

use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLDevice, MTLDeviceExt, MTLFunctionConstantValues,
    MTLLibrary, MTLLibraryExt, MTLResourceOptions, MTLSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{Error, Result, profile, weights};

type PipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type PipelineCache = Mutex<HashMap<MatrixFormat, PipelineState>>;

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: PipelineCache,
    #[cfg(test)]
    tiled_pipelines: PipelineCache,
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
}

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
            || batches.checked_mul(columns) != Some(self.inputs.len())
            || self.inputs.len() > 8_000_000
            || self.inputs.iter().any(|value| !value.is_finite())
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

pub(super) fn initialize() -> Result<()> {
    // Establish a usable reduction pipeline before model readiness. Other
    // storage-specialized pipelines compile lazily on their first use.
    executor()?.pipeline(MatrixFormat::Packed(0)).map(|_| ())
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
            #[cfg(test)]
            source.push_str(include_str!("../../../unit/fixtures/hybrid_tiled_experiment.metal"));
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
                #[cfg(test)]
                tiled_pipelines: Mutex::new(HashMap::new()),
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

impl Executor {
    fn pipeline(
        &self,
        format: MatrixFormat,
    ) -> Result<PipelineState> {
        let name = match format {
            MatrixFormat::Packed(_) => "hybrid_packed_matvec",
            MatrixFormat::Affine {
                ..
            } => "hybrid_affine_matvec",
        };
        self.cached_pipeline(format, name, &self.pipelines)
    }

    #[cfg(test)]
    fn pipeline_with_tiling(
        &self,
        format: MatrixFormat,
        tiled: bool,
    ) -> Result<PipelineState> {
        if !tiled {
            return self.pipeline(format);
        }
        let name = match format {
            MatrixFormat::Packed(_) => "hybrid_packed_matvec_tiled",
            MatrixFormat::Affine {
                ..
            } => "hybrid_affine_matvec_tiled",
        };
        self.cached_pipeline(format, name, &self.tiled_pipelines)
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
        // The unchanged reduction requires eight 32-lane SIMD groups.
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
                        width: operation.rows,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
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
        self.run_with_tiling(codes, input, parameters, affine, false)
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
        objc2::rc::autoreleasepool(|_| self.run_inner(codes, input, parameters, affine, tiled))
    }

    #[cfg(test)]
    fn run_inner(
        &self,
        codes: &[u8],
        input: &[f32],
        parameters: [u32; 8],
        affine: Option<(&[u8], &[u8])>,
        tiled: bool,
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
        let pipeline = self.pipeline_with_tiling(format, tiled)?;
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
                width: if tiled {
                    (parameters[0] as usize).div_ceil(4)
                } else {
                    parameters[0] as usize
                },
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
            assert_eq!(tiled, actual, "tiled packed tail kind={kind}");
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
            assert_eq!(tiled_nine, batched_nine, "tiled packed batch crossing kind={kind}");
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
                    assert_eq!(tiled, actual, "tiled affine tail bits={bits} group={group} dtype={dtype}");
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
                    assert_eq!(
                        tiled_nine, batched_nine,
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
