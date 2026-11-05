// SPDX-License-Identifier: Apache-2.0
//! Opt-in Q5/Q6 F32 matrix tiles with cross-token decoded-weight reuse.
//! Scheduling derives from the attributed packed_kquant_dense_wide.metal;
//! source-level storage is bounded, while actual register spills and device
//! performance remain separate measurement gates.

use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct WideGeometry {
    nsg: usize,
    nt: usize,
    rt: usize,
}

impl WideGeometry {
    fn shared_bytes(self) -> usize {
        self.nsg * self.nt * 256
    }

    fn accumulator_elements(self) -> usize {
        2 * self.nt * self.rt
    }
}

fn wide_geometry(
    rows: usize,
    columns: usize,
    tokens: usize,
) -> Option<WideGeometry> {
    if !matches!(tokens, 32 | 64 | 128)
        || !matches!((rows, columns), (8192, 4096) | (4096, 8192))
        || rows.checked_mul(tokens).is_none_or(|n| n > 8_000_000)
        || columns.checked_mul(tokens).is_none_or(|n| n > 8_000_000)
    {
        return None;
    }
    Some(WideGeometry {
        // Preserve each actual shape's qualified T8 K partition and split
        // reduction order, even as the independent input tile becomes wider.
        nsg: if rows == 8192 {
            8
        } else {
            16
        },
        nt: 128 / tokens,
        rt: tokens / 8,
    })
}

type WidePipelineKey = (u32, WideGeometry);
type WidePipelineCache = Mutex<HashMap<WidePipelineKey, Option<PipelineState>>>;
static WIDE_PIPELINES: OnceLock<WidePipelineCache> = OnceLock::new();

fn wide_pipeline(
    kind: u32,
    geometry: WideGeometry,
) -> Result<Option<PipelineState>> {
    let mut cache = WIDE_PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| Error::Invalid("dense wide prefill pipeline lock failed"))?;
    if let Some(value) = cache.get(&(kind, geometry)) {
        return Ok(value.clone());
    }
    let executor = executor()?;
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, weights::block_layout(kind)?.1 as u32), (80, geometry.nsg as u32)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let name = format!("hybrid_kquant_dense_wide_nt{}_rt{}", geometry.nt, geometry.rt);
    let function = executor.library.new_function_with_name_constant_values(&name, &constants).ok();
    let pipeline = function
        .and_then(|function| executor.device.new_compute_pipeline_state_with_function(&function).ok())
        .filter(|pipeline| {
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= geometry.nsg * 32
                && pipeline.static_threadgroup_memory_length() + geometry.shared_bytes()
                    <= executor.device.max_threadgroup_memory_length()
        });
    cache.insert((kind, geometry), pipeline.clone());
    Ok(pipeline)
}

impl DecodeGraph {
    /// Try one true wide packed matrix tile. False performs no graph encoding
    /// or resource mutation, leaving the caller's existing T8/T4 route intact.
    pub fn mapped_dense_wide_into(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
    ) -> Result<bool> {
        if !std::env::var_os("STARK_GLM53_PREFILL_DENSE_WIDE").is_some_and(|value| value == "1") {
            return Ok(false);
        }
        let MatrixFormat::Packed(kind @ (13 | 14)) = operation.format else {
            return Ok(false);
        };
        let Some(geometry) = wide_geometry(operation.rows, operation.columns, tokens) else {
            return Ok(false);
        };
        // The F32 thread_elements lane layout is implementation-defined. Keep
        // the same actual device boundary as the T8 reference route.
        if executor()?.device.name() != "Apple M5 Max" {
            return Ok(false);
        }
        self.encode_dense_wide_geometry(operation, input, output, tokens, kind, geometry)
    }

    fn encode_dense_wide_geometry(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
        kind: u32,
        geometry: WideGeometry,
    ) -> Result<bool> {
        input.validate()?;
        output.validate()?;
        if !matches!(kind, 13 | 14)
            || !matches!(geometry.nsg, 8 | 16)
            || !matches!((geometry.nt, geometry.rt), (4, 4) | (2, 8) | (1, 16))
            || operation.format != MatrixFormat::Packed(kind)
            || operation.matrices.len() != 1
            || operation.rows_per_batch != 0
            || operation.rows == 0
            || operation.columns == 0
            || !operation.columns.is_multiple_of(256)
            || tokens == 0
            || tokens > geometry.rt * 8
            || geometry.accumulator_elements() > 32
            || operation.columns.checked_mul(tokens) != Some(input.len)
            || operation.rows.checked_mul(tokens) != Some(output.len)
            || input.len > 8_000_000
            || output.len > 8_000_000
            || (std::ptr::eq(&*input.buffer, &*output.buffer)
                && input.offset < output.offset + output.len * 4
                && output.offset < input.offset + input.len * 4)
        {
            return Err(Error::Invalid("dense wide prefill checked geometry"));
        }
        if self.projection_level || self.encoding_dispatch != MTLDispatchType::SERIAL {
            return Ok(false);
        }
        // Retain the existing mapped-operation descriptor contract, including
        // exact packed span length, in addition to the wide activation views.
        operation.parameters_for_input(operation.columns)?;
        let matrix = &operation.matrices[0];
        let block_bytes = weights::block_layout(kind)?.1;
        let bytes = operation
            .rows
            .checked_mul(operation.columns / 256)
            .and_then(|n| n.checked_mul(block_bytes))
            .ok_or(Error::Invalid("dense wide prefill weight size overflow"))?;
        if matrix.planes.is_some()
            || matrix.codes.length < bytes
            || matrix.codes.offset.checked_add(bytes).is_none_or(|end| end > matrix.codes.buffer.length())
            || matrix.codes.offset > u32::MAX as usize
            || input.offset / 4 > u32::MAX as usize
            || output.offset / 4 > u32::MAX as usize
            || (std::ptr::eq(&*matrix.codes.buffer, &*output.buffer)
                && matrix.codes.offset < output.offset + output.len * 4
                && output.offset < matrix.codes.offset + bytes)
        {
            return Err(Error::Invalid("dense wide prefill checked weight span"));
        }
        let Some(pipeline) = wide_pipeline(kind, geometry)? else {
            return Ok(false);
        };
        // Every decline above is encoding-free. From here onward the graph
        // owns all bound buffers and carries the usual sticky finite status.
        self.prepare_external_write(output)?;
        self.check_finite(input)?;
        self.resources.extend([matrix.codes.buffer.clone(), input.buffer.clone(), output.buffer.clone()]);
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
        self.encoding.set_threadgroup_memory_length(geometry.shared_bytes(), 0);
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: operation.rows.div_ceil(8 * geometry.nt),
                height: tokens.div_ceil(8 * geometry.rt),
                depth: 1,
            },
            MTLSize {
                width: geometry.nsg * 32,
                height: 1,
                depth: 1,
            },
        );
        self.check_finite(output)?;
        profile::record(
            if kind == 13 {
                "native.glm.prefill.dense_wide.q5"
            } else {
                "native.glm.prefill.dense_wide.q6"
            },
            0.0,
            bytes,
        );
        profile::record("native.glm.prefill.dense_wide.tokens", 0.0, tokens);
        Ok(true)
    }
}

#[cfg(test)]
#[path = "decode_graph_dense_wide_tests.rs"]
mod tests;
