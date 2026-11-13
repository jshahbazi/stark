// SPDX-License-Identifier: Apache-2.0
//! F32 token-major projections with decoded-weight reuse. The per-input
//! arithmetic follows the selected scalar projection, including its fallback.

use super::*;

static TOKEN_GENERIC: OnceLock<PipelineCache> = OnceLock::new();
static TOKEN_Q8: OnceLock<PipelineCache> = OnceLock::new();
static TOKEN_Q5: OnceLock<PipelineCache> = OnceLock::new();
static TOKEN_Q6: OnceLock<PipelineCache> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenSchedule {
    Generic,
    Q8Grouped,
    Q5Grouped,
    Q6Grouped,
}

impl TokenSchedule {
    fn kernel(self) -> &'static str {
        match self {
            Self::Generic => "hybrid_token_projection_generic",
            Self::Q8Grouped => "hybrid_token_projection_q8",
            Self::Q5Grouped => "hybrid_token_projection_q5",
            Self::Q6Grouped => "hybrid_token_projection_q6",
        }
    }

    fn tile(self) -> (usize, usize, usize) {
        match self {
            Self::Generic => (1, 8, 256),
            Self::Q8Grouped => (4, 4, 64),
            Self::Q5Grouped => (2, 4, 64),
            Self::Q6Grouped => (4, 4, 64),
        }
    }

    fn cache(self) -> &'static PipelineCache {
        match self {
            Self::Generic => TOKEN_GENERIC.get_or_init(|| Mutex::new(HashMap::new())),
            Self::Q8Grouped => TOKEN_Q8.get_or_init(|| Mutex::new(HashMap::new())),
            Self::Q5Grouped => TOKEN_Q5.get_or_init(|| Mutex::new(HashMap::new())),
            Self::Q6Grouped => TOKEN_Q6.get_or_init(|| Mutex::new(HashMap::new())),
        }
    }
}

fn token_schedule(operation: &MappedOperation<'_>) -> Option<TokenSchedule> {
    match operation.format {
        MatrixFormat::Packed(0 | 1 | 30) => Some(TokenSchedule::Generic),
        MatrixFormat::Packed(8) => match q8_grouped8_shape(operation.format, operation.rows, operation.columns) {
            Some(true) => Some(TokenSchedule::Q8Grouped),
            Some(false) => None,
            None if q8_block_shape(operation.format, operation.rows, operation.columns) => None,
            None => Some(TokenSchedule::Generic),
        },
        MatrixFormat::Packed(13)
            if operation.rows_per_batch == 0
                && matches!((operation.rows, operation.columns), (1536, 4096) | (4096, 16384))
                && q5_grouped32_shape(operation.format, operation.columns) =>
        {
            Some(TokenSchedule::Q5Grouped)
        },
        MatrixFormat::Packed(14)
            if operation.rows_per_batch == 0
                && matches!((operation.rows, operation.columns), (1536, 4096) | (4096, 16384))
                && q6_grouped16_shape(operation.format, operation.columns) =>
        {
            Some(TokenSchedule::Q6Grouped)
        },
        _ => None,
    }
}

impl DecodeGraph {
    /// Applies one immutable rank-two or flattened head matrix to all tokens.
    /// The rank-three layout is `[token,head,channel]` and uses the existing
    /// rows_per_batch descriptor. Unsupported routes decline before encoding.
    pub fn mapped_mla_tokens_into(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
    ) -> Result<bool> {
        if !std::env::var_os("STARK_GLM53_MLA_PREFILL_BATCH").is_some_and(|value| value == "1") {
            return Ok(false);
        }
        let Some(schedule) = token_schedule(operation) else {
            return Ok(false);
        };
        self.encode_token_projection(operation, input, output, tokens, schedule)
    }

    fn encode_token_projection(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
        schedule: TokenSchedule,
    ) -> Result<bool> {
        input.validate()?;
        output.validate()?;
        let heads = if operation.rows_per_batch == 0 {
            1
        } else {
            if !operation.rows.is_multiple_of(operation.rows_per_batch) {
                return Err(Error::Invalid("token projection head geometry"));
            }
            operation.rows / operation.rows_per_batch
        };
        let stride = heads.checked_mul(operation.columns).ok_or(Error::Invalid("token projection input overflow"))?;
        if !(1..=128).contains(&tokens)
            || operation.matrices.len() != 1
            || operation.rows == 0
            || operation.columns == 0
            || stride.checked_mul(tokens) != Some(input.len)
            || operation.rows.checked_mul(tokens) != Some(output.len)
            || input.len > 8_000_000
            || output.len > 8_000_000
            || (std::ptr::eq(&*input.buffer, &*output.buffer)
                && input.offset < output.offset + output.len * 4
                && output.offset < input.offset + input.len * 4)
        {
            return Err(Error::Invalid("token projection checked geometry"));
        }
        if self.projection_level || self.encoding_dispatch != MTLDispatchType::SERIAL {
            return Ok(false);
        }
        let MatrixFormat::Packed(kind) = operation.format else {
            return Ok(false);
        };
        if !matches!(
            (schedule, kind),
            (TokenSchedule::Generic, 0 | 1 | 8 | 30)
                | (TokenSchedule::Q8Grouped, 8)
                | (TokenSchedule::Q5Grouped, 13)
                | (TokenSchedule::Q6Grouped, 14)
        ) || (matches!(schedule, TokenSchedule::Q5Grouped | TokenSchedule::Q6Grouped)
            && operation.rows_per_batch != 0)
        {
            return Err(Error::Invalid("token projection reduction policy"));
        }
        operation.parameters_for_input(stride)?;
        let matrix = &operation.matrices[0];
        let (block, block_bytes) = weights::block_layout(kind)?;
        let bytes = operation
            .rows
            .checked_mul(operation.columns / block)
            .and_then(|n| n.checked_mul(block_bytes))
            .ok_or(Error::Invalid("token projection weight overflow"))?;
        if matrix.planes.is_some()
            || matrix.codes.length < bytes
            || matrix.codes.offset.checked_add(matrix.codes.length).is_none_or(|n| n > matrix.codes.buffer.length())
            || matrix.codes.offset.checked_add(bytes).is_none_or(|n| n > matrix.codes.buffer.length())
            || matrix.codes.offset > u32::MAX as usize
            || input.offset / 4 > u32::MAX as usize
            || output.offset / 4 > u32::MAX as usize
            || (std::ptr::eq(&*matrix.codes.buffer, &*output.buffer)
                && matrix.codes.offset < output.offset + output.len * 4
                && output.offset < matrix.codes.offset + bytes)
        {
            return Err(Error::Invalid("token projection weight span"));
        }
        let pipeline = match executor()?.cached_pipeline(operation.format, schedule.kernel(), schedule.cache()) {
            Ok(pipeline) => pipeline,
            Err(_) => return Ok(false),
        };
        let (row_tile, token_tile, threads) = schedule.tile();
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < threads
            || pipeline.static_threadgroup_memory_length() > executor()?.device.max_threadgroup_memory_length()
        {
            return Ok(false);
        }
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
            kind,
            block as u32,
            block_bytes as u32,
            operation.rows_per_batch as u32,
            stride as u32,
            tokens as u32,
            matrix.codes.offset as u32,
            (input.offset / 4) as u32,
            (output.offset / 4) as u32,
        ];
        self.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 4);
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: operation.rows.div_ceil(row_tile),
                height: tokens.div_ceil(token_tile),
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        self.check_finite(output)?;
        profile::record("native.glm.prefill.token_projection.tokens", 0.0, tokens);
        profile::record("native.glm.prefill.token_projection.weight_bytes", 0.0, bytes);
        Ok(true)
    }

    pub fn mla_index_norm_batch(
        &mut self,
        input: &DeviceVector,
        weight: &DeviceVector,
        bias: &DeviceVector,
        width: usize,
        tokens: usize,
    ) -> Result<DeviceVector> {
        if !(1..=128).contains(&tokens)
            || width == 0
            || width.checked_mul(tokens) != Some(input.len)
            || weight.len != width
            || bias.len != width
            || input.len > 8_000_000
        {
            return Err(Error::Invalid("MLA batch index norm geometry"));
        }
        let output = self.scratch(input.len)?;
        self.kernel(
            "hybrid_mla_index_norm_batch",
            &[input, weight, bias, &output],
            &[width as u32, tokens as u32],
            tokens * 256,
            256,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "decode_graph_mla_tests.rs"]
mod tests;
