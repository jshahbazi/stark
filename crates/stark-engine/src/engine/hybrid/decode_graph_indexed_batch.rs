// SPDX-License-Identifier: Apache-2.0
//! Bounded multi-token expert projection with decoded-weight reuse.
//! The GPU keeps stable route lists; no router readback or CPU synchronization
//! is introduced. Output remains token/selected-slot/output-row major.

use super::*;

type IndexedBatchPipelines = (PipelineState, PipelineState);
type IndexedBatchCache = Mutex<HashMap<(u32, BatchSchedule), IndexedBatchPipelines>>;
static PIPELINES: OnceLock<IndexedBatchCache> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum BatchSchedule {
    Configured,
    Vector,
    #[cfg(test)]
    Mma4,
    Mma8,
}

impl BatchSchedule {
    fn nsg(self) -> Option<usize> {
        match self {
            #[cfg(test)]
            Self::Mma4 => Some(4),
            Self::Mma8 => Some(8),
            _ => None,
        }
    }
}

fn mma_admitted(
    format: MatrixFormat,
    shape: [usize; 3],
    selected: usize,
    tokens: usize,
) -> Result<bool> {
    Ok(matches!(format, MatrixFormat::Packed(17 | 18))
        && matches!(shape, [288, 2048, 4096] | [288, 4096, 2048])
        && selected == 8
        && matches!(tokens, 32 | 64)
        && executor()?.device.name() == "Apple M5 Max")
}

fn pipelines(
    kind: u32,
    schedule: BatchSchedule,
) -> Result<IndexedBatchPipelines> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| Error::Invalid("indexed batch pipeline cache failed"))?;
    if let Some(pipelines) = cache.get(&(kind, schedule)) {
        return Ok(pipelines.clone());
    }
    let executor = executor()?;
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, weights::block_layout(kind)?.1 as u32)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    if let Some(nsg) = schedule.nsg() {
        let value = nsg as u32;
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, 80);
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
    let project = if let Some(nsg) = schedule.nsg() {
        let pipeline = create("hybrid_graph_iq_indexed_batch_mma", nsg * 32)?;
        if pipeline.static_threadgroup_memory_length() + nsg * 4 * 64 * 4
            > executor.device.max_threadgroup_memory_length()
        {
            return Err(Error::Unsupported("indexed batch MMA threadgroup memory unavailable"));
        }
        pipeline
    } else {
        create("hybrid_graph_iq_indexed_batch", 64)?
    };
    let pipelines = (create("hybrid_graph_iq_batch_route_pack", 256)?, project);
    cache.insert((kind, schedule), pipelines.clone());
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
        let schedule = if std::env::var_os("STARK_GLM53_PREFILL_EXPERT_MMA").is_some_and(|value| value == "1") {
            if mma_admitted(format, shape, selected_per_token, tokens)? {
                BatchSchedule::Mma8
            } else {
                BatchSchedule::Configured
            }
        } else {
            BatchSchedule::Vector
        };
        self.indexed_batch_schedule(
            matrix,
            format,
            shape,
            indices,
            selected_per_token,
            input,
            tokens,
            input_batched,
            schedule,
        )
    }

    #[expect(clippy::too_many_arguments, reason = "testable schedule plus explicit immutable bank and route geometry")]
    fn indexed_batch_schedule(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        shape: [usize; 3],
        indices: &DeviceVector,
        selected_per_token: usize,
        input: &DeviceVector,
        tokens: usize,
        input_batched: bool,
        schedule: BatchSchedule,
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
        let kind = if let MatrixFormat::Packed(kind @ (17 | 18)) = format {
            Some(kind)
        } else {
            None
        };
        if kind.is_none() || schedule == BatchSchedule::Configured {
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
        }
        let kind = kind.ok_or(Error::Invalid("indexed batch packed schedule"))?;
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
            if schedule.nsg().is_some() {
                8
            } else {
                4
            },
        ];
        let (pack, project) = pipelines(kind, schedule)?;
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
        if let Some(nsg) = schedule.nsg() {
            self.encoding.set_threadgroup_memory_length(nsg * 4 * 64 * 4, 0);
        }
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: rows.div_ceil(if schedule.nsg().is_some() {
                    32
                } else {
                    4
                }),
                height: routes,
                depth: 1,
            },
            MTLSize {
                width: schedule.nsg().map_or(64, |nsg| nsg * 32),
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
        profile::record(
            if schedule.nsg().is_some() {
                "native.glm.prefill.indexed_batch_mma"
            } else {
                "native.glm.prefill.indexed_batch"
            },
            0.0,
            routes,
        );
        Ok(output)
    }
}

#[cfg(test)]
#[path = "decode_graph_indexed_batch_tests.rs"]
mod tests;
