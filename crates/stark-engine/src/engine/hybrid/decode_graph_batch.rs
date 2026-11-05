// SPDX-License-Identifier: Apache-2.0
//! Same immutable rank-two matrix applied to a small token-major F32 batch.
//! MMA scheduling derives from MIT llama.cpp dde879bc4f954b809892fd0d660515f42536394a;
//! see packed_kquant_mma_matvec.metal and LICENSE.ggml-hybrid. F32 activations,
//! fragments and accumulation are retained; the reduction association changes.

use super::*;

#[path = "decode_graph_dense_wide.rs"]
mod dense_wide;

type MmaCache = Mutex<HashMap<(u32, usize), PipelineState>>;
static MMA_PIPELINES: OnceLock<MmaCache> = OnceLock::new();

fn admitted(
    operation: &MappedOperation<'_>,
    tokens: usize,
) -> Result<bool> {
    Ok((tokens == 4 && operation.format == MatrixFormat::Packed(14)
        || tokens == 8 && matches!(operation.format, MatrixFormat::Packed(13 | 14)))
        && matches!((operation.rows, operation.columns), (8192, 4096) | (4096, 8192))
        // thread_elements layout is implementation-defined in MSL. This is
        // the exact device on which basis/tail/source/F64 gates were run.
        && executor()?.device.name() == "Apple M5 Max")
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn kquant_mma_scratch_high_water_reports_total_graph_backing() {
        let previous = profile::enabled();
        profile::reset();
        profile::enable(true);
        let mut graph = DecodeGraph::new().unwrap();
        graph.scratch(300_000).unwrap();
        graph.scratch(400_000).unwrap();
        let total = graph.scratch_bytes;
        let largest_slab = graph.scratch.iter().map(|(buffer, _)| buffer.length()).max().unwrap();
        let counter = profile::snapshot().remove("native.metal.graph.scratch_high_water").unwrap();
        profile::enable(previous);
        assert_eq!(counter.max_bytes, total as u64);
        assert!(total > largest_slab && total <= 512 * 1024 * 1024);
        // Counter.bytes sums successive snapshots; it is not an allocated
        // peak and must never be used as the resident/scratch memory bound.
        assert!(counter.bytes > counter.max_bytes);
    }
}

fn pipeline(
    kind: u32,
    nsg: usize,
) -> Result<PipelineState> {
    let mut cache = MMA_PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| Error::Invalid("prefill MMA cache failed"))?;
    if let Some(pipeline) = cache.get(&(kind, nsg)) {
        return Ok(pipeline.clone());
    }
    let executor = executor()?;
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, weights::block_layout(kind)?.1 as u32), (80, nsg as u32)] {
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
    cache.insert((kind, nsg), result.clone());
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
        let len = operation
            .rows
            .checked_mul(tokens)
            .filter(|&len| len <= 8_000_000)
            .ok_or(Error::Invalid("same-weight prefill output shape"))?;
        let output = self.scratch(len)?;
        self.mapped_same_weight_into(operation, input, &output, tokens, use_mma)?;
        Ok(output)
    }

    /// Apply an existing qualified T4/T8 schedule to an explicitly disjoint
    /// destination. Wider token tiles concatenate output spans without an
    /// extra projection buffer or device copy for every small matrix batch.
    pub fn mapped_same_weight_into(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
        use_mma: bool,
    ) -> Result<()> {
        input.validate()?;
        output.validate()?;
        if !(1..=8).contains(&tokens)
            || operation.matrices.len() != 1
            || operation.rows_per_batch != 0
            || operation.columns.checked_mul(tokens) != Some(input.len)
            || operation.rows.checked_mul(tokens).is_none_or(|n| n > 8_000_000)
            || operation.rows.checked_mul(tokens) != Some(output.len)
            || (std::ptr::eq(&*input.buffer, &*output.buffer)
                && input.offset < output.offset + output.len * 4
                && output.offset < input.offset + input.len * 4)
        {
            return Err(Error::Invalid("same-weight prefill shape"));
        }
        self.prepare_external_write(output)?;
        self.resources.push(output.buffer.clone());
        let parameters = operation.parameters_for_input(operation.columns)?;
        if !use_mma || !admitted(operation, tokens)? {
            for token in 0..tokens {
                self.mapped_into(
                    operation,
                    &input.slice(token * operation.columns, operation.columns)?,
                    &output.slice(token * operation.rows, operation.rows)?,
                    &parameters,
                )?;
            }
            return Ok(());
        }
        if self.projection_level || self.encoding_dispatch != MTLDispatchType::SERIAL {
            return Err(Error::Unsupported("prefill MMA requires serial encoding"));
        }
        let matrix = &operation.matrices[0];
        let MatrixFormat::Packed(kind) = operation.format else {
            return Err(Error::Invalid("prefill MMA packed format"));
        };
        let bytes = operation.rows * (operation.columns / 256) * weights::block_layout(kind)?.1;
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
        let pipeline = pipeline(kind, nsg)?;
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
        self.check_finite(output)?;
        profile::record(
            if kind == 13 {
                "native.glm.prefill.q5_mma"
            } else {
                "native.glm.prefill.q6_mma"
            },
            0.0,
            bytes,
        );
        Ok(())
    }
}
