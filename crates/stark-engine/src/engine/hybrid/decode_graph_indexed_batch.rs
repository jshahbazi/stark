// SPDX-License-Identifier: Apache-2.0
//! Bounded multi-token expert projection with four-route decoded-weight reuse.
//! The GPU keeps stable route lists; no router readback or CPU synchronization
//! is introduced. Output remains token/selected-slot/output-row major.

use super::*;

type IndexedBatchPipelines = (PipelineState, PipelineState);
static PIPELINES: OnceLock<Mutex<HashMap<u32, IndexedBatchPipelines>>> = OnceLock::new();

fn pipelines(kind: u32) -> Result<IndexedBatchPipelines> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| Error::Invalid("indexed batch pipeline cache failed"))?;
    if let Some(pipelines) = cache.get(&kind) {
        return Ok(pipelines.clone());
    }
    let executor = executor()?;
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, weights::block_layout(kind)?.1 as u32)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let create = |name, threads| -> Result<PipelineState> {
        let function = executor
            .library
            .new_function_with_name_constant_values(name, &constants)
            .map_err(|_| Error::Unsupported("indexed batch function unavailable"))?;
        let pipeline = executor
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| Error::Unsupported("indexed batch pipeline unavailable"))?;
        if pipeline.thread_execution_width() != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
            return Err(Error::Unsupported("indexed batch execution width unavailable"));
        }
        Ok(pipeline)
    };
    let pipelines = (create("hybrid_graph_iq_batch_route_pack", 256)?, create("hybrid_graph_iq_indexed_batch", 64)?);
    cache.insert(kind, pipelines.clone());
    Ok(pipelines)
}

impl DecodeGraph {
    #[expect(clippy::too_many_arguments, reason = "explicit indexed token, route and immutable bank geometry")]
    pub fn indexed_batch(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        shape: [usize; 3],
        indices: &DeviceVector,
        selected_per_token: usize,
        input: &DeviceVector,
        tokens: usize,
        input_batched: bool,
    ) -> Result<DeviceVector> {
        let [experts, rows, columns] = shape;
        let routes = tokens.checked_mul(selected_per_token).ok_or(Error::Invalid("indexed batch route overflow"))?;
        let input_rows = if input_batched {
            routes
        } else {
            tokens
        };
        if !(1..=64).contains(&tokens)
            || experts == 0
            || rows == 0
            || columns == 0
            || selected_per_token == 0
            || selected_per_token > 64
            || selected_per_token > experts
            || !(1..=512).contains(&routes)
            || indices.len != routes
            || columns.checked_mul(input_rows) != Some(input.len)
            || rows.checked_mul(routes).is_none_or(|n| n > 8_000_000)
        {
            return Err(Error::Invalid("indexed batch matrix geometry"));
        }
        indices.validate()?;
        input.validate()?;
        let operation = MappedOperation {
            matrices: vec![matrix.clone()],
            inputs: &[],
            rows: rows.checked_mul(experts).ok_or(Error::Invalid("indexed batch bank overflow"))?,
            columns,
            rows_per_batch: 0,
            format,
        };
        operation.parameters_for_input(columns)?;
        let output = self.scratch(rows * routes)?;
        let MatrixFormat::Packed(kind @ (17 | 18)) = format else {
            for token in 0..tokens {
                let input_first = if input_batched {
                    token * selected_per_token
                } else {
                    token
                };
                let row = self.indexed(
                    matrix,
                    format,
                    shape,
                    &indices.slice(token * selected_per_token, selected_per_token)?,
                    selected_per_token,
                    &input.slice(
                        input_first * columns,
                        columns
                            * if input_batched {
                                selected_per_token
                            } else {
                                1
                            },
                    )?,
                    input_batched,
                )?;
                self.copy_into(&row, &output.slice(token * selected_per_token * rows, selected_per_token * rows)?)?;
            }
            return Ok(output);
        };
        if self.projection_level || self.encoding_dispatch != MTLDispatchType::SERIAL {
            return Err(Error::Unsupported("indexed batch requires serial dependent encoding"));
        }
        let map_len = experts
            .checked_mul(routes + 1)
            .filter(|&n| n <= 8_000_000)
            .ok_or(Error::Invalid("indexed batch route map overflow"))?;
        let maps = self.scratch(map_len)?;
        let descriptors = self.scratch(routes * 3)?;
        let count = self.scratch(1)?;
        let p = [
            rows as u32,
            columns as u32,
            experts as u32,
            selected_per_token as u32,
            tokens as u32,
            u32::from(input_batched),
            routes as u32,
        ];
        let (pack, project) = pipelines(kind)?;
        self.check_finite(input)?;
        self.order_dispatch();
        self.encoding.set_compute_pipeline_state(&pack);
        for (index, vector) in [indices, &maps, &descriptors, &count].into_iter().enumerate() {
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, index);
        }
        self.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 4);
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.order_dispatch();
        self.encoding.set_compute_pipeline_state(&project);
        self.encoding.set_buffer(Some(&matrix.codes.buffer), matrix.codes.offset, 0);
        self.encoding.set_buffer(Some(&input.buffer), input.offset, 1);
        self.encoding.set_buffer(Some(&output.buffer), output.offset, 2);
        self.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        for (index, vector) in [&maps, &descriptors, &count].into_iter().enumerate() {
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, index + 4);
        }
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: rows.div_ceil(4),
                height: routes,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        self.resources.extend([
            matrix.codes.buffer.clone(),
            input.buffer.clone(),
            output.buffer.clone(),
            indices.buffer.clone(),
            maps.buffer.clone(),
            descriptors.buffer.clone(),
            count.buffer.clone(),
        ]);
        self.check_finite(&output)?;
        profile::record("native.glm.prefill.indexed_batch", 0.0, routes);
        Ok(output)
    }
}

#[cfg(test)]
#[path = "decode_graph_indexed_batch_tests.rs"]
mod tests;
