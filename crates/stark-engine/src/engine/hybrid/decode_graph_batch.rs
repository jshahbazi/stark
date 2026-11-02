// SPDX-License-Identifier: Apache-2.0
//! Same immutable rank-two matrix applied to a small token-major F32 batch.
//! MMA scheduling derives from MIT llama.cpp dde879bc4f954b809892fd0d660515f42536394a;
//! see packed_kquant_mma_matvec.metal and LICENSE.ggml-hybrid. F32 activations,
//! fragments and accumulation are retained; the reduction association changes.

use super::*;

type MmaCache = Mutex<HashMap<usize, PipelineState>>;
static MMA_PIPELINES: OnceLock<MmaCache> = OnceLock::new();

fn admitted(
    operation: &MappedOperation<'_>,
    tokens: usize,
) -> Result<bool> {
    Ok(tokens == 4
        && operation.format == MatrixFormat::Packed(14)
        && matches!((operation.rows, operation.columns), (8192, 4096) | (4096, 8192))
        // thread_elements layout is implementation-defined in MSL. This is
        // the exact device on which basis/tail/source/F64 gates were run.
        && executor()?.device.name() == "Apple M5 Max")
}

fn pipeline(nsg: usize) -> Result<PipelineState> {
    let mut cache = MMA_PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| Error::Invalid("prefill MMA cache failed"))?;
    if let Some(pipeline) = cache.get(&nsg) {
        return Ok(pipeline.clone());
    }
    let executor = executor()?;
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, 14u32), (4, 256), (5, 210), (80, nsg as u32)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let function = executor
        .library
        .new_function_with_name_constant_values("hybrid_kquant_mma_nt4_rt1", &constants)
        .map_err(|_| Error::Unsupported("prefill MMA function unavailable"))?;
    let result = executor
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|_| Error::Unsupported("prefill MMA pipeline unavailable"))?;
    if result.thread_execution_width() != 32
        || result.max_total_threads_per_threadgroup() < nsg * 32
        || result.static_threadgroup_memory_length() + nsg * 4 * 256 > executor.device.max_threadgroup_memory_length()
    {
        return Err(Error::Unsupported("prefill MMA qualified geometry unavailable"));
    }
    cache.insert(nsg, result.clone());
    Ok(result)
}

impl DecodeGraph {
    pub fn mapped_same_weight(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
        tokens: usize,
        use_mma: bool,
    ) -> Result<DeviceVector> {
        input.validate()?;
        if !(1..=4).contains(&tokens)
            || operation.matrices.len() != 1
            || operation.rows_per_batch != 0
            || operation.columns.checked_mul(tokens) != Some(input.len)
            || operation.rows.checked_mul(tokens).is_none_or(|n| n > 8_000_000)
        {
            return Err(Error::Invalid("same-weight prefill shape"));
        }
        let parameters = operation.parameters_for_input(operation.columns)?;
        let output = self.scratch(operation.rows * tokens)?;
        if !use_mma || !admitted(operation, tokens)? {
            for token in 0..tokens {
                self.mapped_into(
                    operation,
                    &input.slice(token * operation.columns, operation.columns)?,
                    &output.slice(token * operation.rows, operation.rows)?,
                    &parameters,
                )?;
            }
            return Ok(output);
        }
        if self.projection_level || self.encoding_dispatch != MTLDispatchType::SERIAL {
            return Err(Error::Unsupported("prefill MMA requires serial encoding"));
        }
        let matrix = &operation.matrices[0];
        let bytes = operation.rows * (operation.columns / 256) * 210;
        if matrix.planes.is_some()
            || matrix.codes.length < bytes
            || matrix.codes.offset.checked_add(bytes).is_none_or(|end| end > matrix.codes.buffer.length())
            || matrix.codes.offset > u32::MAX as usize
            || input.offset / 4 > u32::MAX as usize
            || output.offset / 4 > u32::MAX as usize
        {
            return Err(Error::Invalid("prefill MMA checked weight span"));
        }
        let nsg = if operation.rows == 8192 {
            8
        } else {
            16
        };
        let pipeline = pipeline(nsg)?;
        self.check_finite(input)?;
        self.order_dispatch();
        self.encoding.set_compute_pipeline_state(&pipeline);
        for (index, buffer) in [&matrix.codes.buffer, &input.buffer, &output.buffer].into_iter().enumerate() {
            self.encoding.set_buffer(Some(buffer), 0, index);
        }
        let p = [
            operation.rows as u32,
            operation.columns as u32,
            tokens as u32,
            matrix.codes.offset as u32,
            (input.offset / 4) as u32,
            (output.offset / 4) as u32,
        ];
        self.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 4);
        self.encoding.set_threadgroup_memory_length(nsg * 4 * 256, 0);
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: operation.rows.div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: nsg * 32,
                height: 1,
                depth: 1,
            },
        );
        self.resources.extend([matrix.codes.buffer.clone(), input.buffer.clone()]);
        self.check_finite(&output)?;
        profile::record("native.glm.prefill.q6_mma", 0.0, bytes);
        Ok(output)
    }
}
