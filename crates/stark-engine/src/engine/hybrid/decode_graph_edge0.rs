//! Checked bindings for Edge0's opt-in packed batched projections.

use super::*;

#[cfg(test)]
#[path = "decode_graph_edge0_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "decode_graph_edge0_direct_tests.rs"]
mod direct_tests;

#[path = "decode_graph_edge0_tokens.rs"]
mod tokens;

fn specialize_single() -> bool {
    // Native workers fix their selectors before model loading. Cache once per
    // process rather than reading the environment for every selected expert.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::engine::hybrid::edge0_config::local_evaluation_enabled()
            || std::env::var_os("STARK_EDGE0_SPECIALIZE_SINGLE").is_some_and(|value| value == "1")
    })
}

/// Diagnostic only. 1 labels the existing Edge0 stages; 2 also splits each
/// selected expert and U job into gather, matvec and scatter encoders.
fn stage_level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("STARK_EDGE0_STAGE_TIMING").ok().and_then(|value| value.parse().ok()).unwrap_or(0)
    })
}

/// Opt-in Edge0 route schedule, fixed once per process like the singleton
/// selector. `STARK_EDGE0_DIRECT_ROUTES=1` selects one-token kernels that
/// write route rows in place. `verify` encodes both schedules and rejects
/// the request unless every compared F32 value has identical bits.
///
/// `token-routes` is the direct schedule with token rows of one route or
/// trunk matrix sharing a dispatch. `tokens` also shares the hyper-connection
/// and mixer output projections across a token batch. `verify-tokens`
/// encodes `tokens` beside the one-token forms and compares them likewise.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Edge0Schedule {
    Grouped,
    Direct,
    Verify,
    TokenRoutes,
    Tokens,
    VerifyTokens,
}

impl Edge0Schedule {
    /// Token rows of one route or trunk matrix share dispatches.
    pub(crate) fn token_rows(self) -> bool {
        matches!(self, Self::TokenRoutes | Self::Tokens | Self::VerifyTokens)
    }

    /// Hyper-connection and mixer output projections share dispatches too.
    pub(crate) fn token_stages(self) -> bool {
        matches!(self, Self::Tokens | Self::VerifyTokens)
    }

    /// The reference form is encoded as well and compared on the device.
    pub(crate) fn verify(self) -> bool {
        matches!(self, Self::Verify | Self::VerifyTokens)
    }
}

fn schedule() -> Edge0Schedule {
    static SCHEDULE: std::sync::OnceLock<Edge0Schedule> = std::sync::OnceLock::new();
    *SCHEDULE.get_or_init(|| match std::env::var("STARK_EDGE0_DIRECT_ROUTES").as_deref() {
        Ok("1") => Edge0Schedule::Direct,
        Ok("verify") => Edge0Schedule::Verify,
        Ok("token-routes") => Edge0Schedule::TokenRoutes,
        Ok("tokens") => Edge0Schedule::Tokens,
        Ok("verify-tokens") => Edge0Schedule::VerifyTokens,
        _ => Edge0Schedule::Grouped,
    })
}

/// Largest token batch any Edge0 schedule encodes in one graph segment.
pub(crate) const EDGE0_TOKEN_BATCH_LIMIT: usize = 32;

/// Ordinary prefill batch size, fixed once per process. The token schedule
/// admits `STARK_EDGE0_PREFILL_TOKENS` of 16, 24 or 32 so that more rows of a
/// batch select the same expert; every other schedule keeps eight, the limit
/// of its kernels.
fn prefill_tokens() -> usize {
    static TOKENS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TOKENS.get_or_init(|| {
        let requested = std::env::var("STARK_EDGE0_PREFILL_TOKENS").ok().and_then(|value| value.parse().ok());
        match requested {
            Some(tokens @ (16 | 24 | 32)) if schedule().token_stages() => tokens,
            _ => 8,
        }
    })
}

fn single_kernel(kind: u32) -> Option<&'static str> {
    match kind {
        8 => Some("edge0_graph_q8_single"),
        11 => Some("edge0_graph_q3_single"),
        20 => Some("edge0_graph_iq4_single"),
        _ => None,
    }
}

fn scaled_single_kernel(kind: u32) -> Option<&'static str> {
    match kind {
        8 => Some("edge0_direct_q8_scaled_single"),
        11 => Some("edge0_direct_q3_scaled_single"),
        20 => Some("edge0_direct_iq4_scaled_single"),
        _ => None,
    }
}

fn overlaps(
    left: &DeviceVector,
    right: &DeviceVector,
) -> bool {
    std::ptr::eq(&*left.buffer, &*right.buffer)
        && left.offset < right.offset + right.len * 4
        && right.offset < left.offset + left.len * 4
}

fn geometry(
    rows: usize,
    columns: usize,
    tokens: usize,
    input: &DeviceVector,
) -> Result<()> {
    input.validate()?;
    if rows == 0
        || rows > 1_000_000
        || columns == 0
        || columns > 65_536
        || !(1..=8).contains(&tokens)
        || columns.checked_mul(tokens) != Some(input.len)
        || rows.checked_mul(tokens).is_none_or(|count| count > 8_000_000)
    {
        return Err(Error::Invalid("Edge0 packed batch geometry"));
    }
    Ok(())
}

fn codes(
    matrix: &MappedMatrix,
    bytes: usize,
) -> Result<()> {
    if matrix.planes.is_some()
        || bytes == 0
        || bytes > matrix.codes.length
        || matrix.codes.offset.checked_add(bytes).is_none_or(|end| end > matrix.codes.buffer.length())
    {
        return Err(Error::Invalid("Edge0 packed batch weight span"));
    }
    Ok(())
}

impl DecodeGraph {
    /// Diagnostic only. Labels the encoders completed since the previous
    /// label when encoder timing is selected; otherwise encodes nothing.
    pub fn edge0_stage(
        &mut self,
        name: &'static str,
        level: u8,
    ) -> Result<()> {
        if self.encoder_timing.is_none() || stage_level() < level {
            return Ok(());
        }
        self.diagnostic_checkpoint(name)
    }

    fn edge0_dispatch(
        &mut self,
        name: &'static str,
        matrices: &[&MappedMatrix],
        vectors: &[&DeviceVector],
        parameters: &[u32],
        rows: usize,
        scan_input: bool,
    ) -> Result<()> {
        // Vectors are input, output and at most one further read-only operand.
        if self.unretained || matrices.is_empty() || matrices.len() > 2 || !(2..=3).contains(&vectors.len()) {
            return Err(Error::Unsupported("Edge0 packed batch transport"));
        }
        let (input, output) = (vectors[0], vectors[1]);
        if scan_input {
            self.check_finite(input)?;
        }
        self.flush_finite_before_write(output)?;
        let executor = executor()?;
        let pipeline = {
            let mut cache = executor.graph_pipelines.lock().map_err(|_| Error::Invalid("Edge0 pipeline cache"))?;
            if let Some(pipeline) = cache.get(name) {
                pipeline.clone()
            } else {
                let function =
                    executor.library.new_function_with_name(name).ok_or(Error::Invalid("Edge0 kernel unavailable"))?;
                let pipeline = executor
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| Error::Invalid("Edge0 kernel pipeline"))?;
                cache.insert(name, pipeline.clone());
                pipeline
            }
        };
        if pipeline.max_total_threads_per_threadgroup() < 256 || pipeline.thread_execution_width() != 32 {
            return Err(Error::Unsupported("Edge0 packed batch SIMD geometry"));
        }
        self.encoding.set_compute_pipeline_state(&pipeline);
        for (index, matrix) in matrices.iter().enumerate() {
            self.encoding.set_buffer(Some(&matrix.codes.buffer), matrix.codes.offset, index);
            self.resources.push(matrix.codes.buffer.clone());
        }
        for (index, vector) in vectors.iter().enumerate() {
            vector.validate()?;
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, matrices.len() + index);
            self.resources.push(vector.buffer.clone());
        }
        self.encoding.set_bytes(
            NonNull::new(parameters.as_ptr().cast_mut()).unwrap().cast(),
            std::mem::size_of_val(parameters),
            matrices.len() + vectors.len(),
        );
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        self.order_dispatch();
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    pub fn edge0_packed_batch(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        self.edge0_packed_batch_selected(matrix, format, rows, columns, input, tokens, specialize_single())
    }

    pub fn edge0_schedule() -> Edge0Schedule {
        schedule()
    }

    pub fn edge0_prefill_tokens() -> usize {
        prefill_tokens()
    }

    /// Diagnostic verification for the route schedule selector. Both views
    /// must hold identical F32 bits, or the command's sticky failure status
    /// rejects the request. A projection level defers the comparison to
    /// `edge0_flush_verification`.
    pub fn edge0_require_identical(
        &mut self,
        left: &DeviceVector,
        right: &DeviceVector,
    ) -> Result<()> {
        left.validate()?;
        right.validate()?;
        if left.len != right.len || left.len > 8_000_000 {
            return Err(Error::Invalid("Edge0 schedule verification shape"));
        }
        if self.projection_level {
            self.edge0_verification.push((left.clone(), right.clone()));
            return Ok(());
        }
        profile::record("native.edge0.schedule.compared_values", 0.0, left.len);
        self.kernel("edge0_direct_require_identical", &[left, right], &[left.len as u32], left.len, 256)
    }

    pub fn edge0_flush_verification(&mut self) -> Result<()> {
        if self.projection_level {
            return Err(Error::Invalid("Edge0 schedule verification inside projection level"));
        }
        for (left, right) in std::mem::take(&mut self.edge0_verification) {
            self.edge0_require_identical(&left, &right)?;
        }
        Ok(())
    }

    /// Direct-route schedule for a token batch: one single-token dispatch per
    /// token row, each with the batched kernel's per-token F32 reduction.
    /// Formats without a format-constant kernel keep the batched kernel.
    pub fn edge0_packed_split(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let MatrixFormat::Packed(8 | 11 | 20) = format else {
            return self.edge0_packed_batch_selected(matrix, format, rows, columns, input, tokens, true);
        };
        geometry(rows, columns, tokens, input)?;
        let output = self.scratch(rows * tokens)?;
        // One scan covers every token row of this caller-supplied batch.
        self.check_finite(input)?;
        for token in 0..tokens {
            self.edge0_packed_single_into(
                matrix,
                format,
                rows,
                columns,
                &input.slice(token * columns, columns)?,
                &output.slice(token * rows, rows)?,
                None,
                false,
            )?;
        }
        Ok(output)
    }

    /// One-token packed projection into a caller-owned row. A stored scale
    /// selects the fused column-scale quotient. Every written value is checked
    /// in the kernel, and a nonfinite input makes every row result nonfinite;
    /// callers pass `scan_input: false` for rows already scanned or written
    /// by such a kernel.
    #[expect(clippy::too_many_arguments, reason = "explicit checked operands for one route row")]
    pub fn edge0_packed_single_into(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        output: &DeviceVector,
        scale: Option<&DeviceVector>,
        scan_input: bool,
    ) -> Result<()> {
        geometry(rows, columns, 1, input)?;
        output.validate()?;
        if output.len != rows || overlaps(input, output) || scale.is_some_and(|scale| scale.len != rows) {
            return Err(Error::Invalid("Edge0 direct row shape"));
        }
        let MatrixFormat::Packed(kind @ (8 | 11 | 20)) = format else {
            return Err(Error::Unsupported("Edge0 direct row format"));
        };
        let (block, bytes) = weights::block_layout(kind)?;
        if !columns.is_multiple_of(block) {
            return Err(Error::Invalid("Edge0 packed batch block alignment"));
        }
        let length = rows
            .checked_mul(columns / block)
            .and_then(|n| n.checked_mul(bytes))
            .ok_or(Error::Invalid("Edge0 packed batch byte overflow"))?;
        codes(matrix, length)?;
        let parameters = [rows as u32, columns as u32, kind, block as u32, bytes as u32, 1];
        let unsupported = Error::Unsupported("Edge0 direct row format");
        if let Some(scale) = scale {
            scale.validate()?;
            self.edge0_dispatch(
                scaled_single_kernel(kind).ok_or(unsupported)?,
                &[matrix],
                &[input, output, scale],
                &parameters,
                rows,
                scan_input,
            )
        } else {
            self.edge0_dispatch(
                single_kernel(kind).ok_or(unsupported)?,
                &[matrix],
                &[input, output],
                &parameters,
                rows,
                scan_input,
            )
        }
    }

    /// One-token fused Q3_K gate/up row for a direct route.
    pub fn edge0_gate_up_swiglu_single(
        &mut self,
        gate: &MappedMatrix,
        up: &MappedMatrix,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        scan_input: bool,
    ) -> Result<DeviceVector> {
        geometry(rows, columns, 1, input)?;
        if !columns.is_multiple_of(256) {
            return Err(Error::Invalid("Edge0 fused Q3_K alignment"));
        }
        let bytes = rows
            .checked_mul(columns / 256)
            .and_then(|n| n.checked_mul(110))
            .ok_or(Error::Invalid("Edge0 fused Q3_K byte overflow"))?;
        codes(gate, bytes)?;
        codes(up, bytes)?;
        let output = self.scratch(rows)?;
        self.edge0_dispatch(
            "edge0_graph_gate_up_swiglu_single",
            &[gate, up],
            &[input, &output],
            &[rows as u32, columns as u32, 1],
            rows,
            scan_input,
        )?;
        Ok(output)
    }

    #[expect(clippy::too_many_arguments, reason = "explicit experimental selector for model-free parity tests")]
    fn edge0_packed_batch_selected(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
        specialize: bool,
    ) -> Result<DeviceVector> {
        geometry(rows, columns, tokens, input)?;
        let MatrixFormat::Packed(kind @ (0 | 1 | 8 | 11 | 20 | 30)) = format else {
            return Err(Error::Unsupported("Edge0 packed batch format"));
        };
        let (block, bytes) = weights::block_layout(kind)?;
        if !columns.is_multiple_of(block) {
            return Err(Error::Invalid("Edge0 packed batch block alignment"));
        }
        let length = rows
            .checked_mul(columns / block)
            .and_then(|n| n.checked_mul(bytes))
            .ok_or(Error::Invalid("Edge0 packed batch byte overflow"))?;
        codes(matrix, length)?;
        let output = self.scratch(rows * tokens)?;
        self.edge0_dispatch(
            match (specialize && tokens == 1, kind) {
                (true, 8) => "edge0_graph_q8_single",
                (true, 11) => "edge0_graph_q3_single",
                (true, 20) => "edge0_graph_iq4_single",
                _ => "edge0_graph_packed_batch",
            },
            &[matrix],
            &[input, &output],
            &[rows as u32, columns as u32, kind, block as u32, bytes as u32, tokens as u32],
            rows,
            true,
        )?;
        Ok(output)
    }

    pub fn edge0_gate_up_swiglu_batch(
        &mut self,
        gate: &MappedMatrix,
        up: &MappedMatrix,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        self.edge0_gate_up_swiglu_batch_selected(gate, up, rows, columns, input, tokens, specialize_single())
    }

    #[expect(clippy::too_many_arguments, reason = "explicit experimental selector for model-free parity tests")]
    fn edge0_gate_up_swiglu_batch_selected(
        &mut self,
        gate: &MappedMatrix,
        up: &MappedMatrix,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
        specialize: bool,
    ) -> Result<DeviceVector> {
        geometry(rows, columns, tokens, input)?;
        if !columns.is_multiple_of(256) {
            return Err(Error::Invalid("Edge0 fused Q3_K alignment"));
        }
        let bytes = rows
            .checked_mul(columns / 256)
            .and_then(|n| n.checked_mul(110))
            .ok_or(Error::Invalid("Edge0 fused Q3_K byte overflow"))?;
        codes(gate, bytes)?;
        codes(up, bytes)?;
        let output = self.scratch(rows * tokens)?;
        self.edge0_dispatch(
            if specialize && tokens == 1 {
                "edge0_graph_gate_up_swiglu_single"
            } else {
                "edge0_graph_gate_up_swiglu_batch"
            },
            &[gate, up],
            &[input, &output],
            &[rows as u32, columns as u32, tokens as u32],
            rows,
            true,
        )?;
        Ok(output)
    }

    pub fn edge0_column_scale(
        &mut self,
        input: &DeviceVector,
        scale: &DeviceVector,
        width: usize,
        tokens: usize,
    ) -> Result<DeviceVector> {
        geometry(width, width, tokens, input)?;
        if scale.len != width {
            return Err(Error::Invalid("Edge0 column scale shape"));
        }
        let output = self.scratch(input.len)?;
        self.kernel(
            "edge0_graph_column_scale",
            &[input, scale, &output],
            &[width as u32, tokens as u32],
            input.len,
            256,
        )?;
        Ok(output)
    }

    #[expect(clippy::too_many_arguments, reason = "separate checked route and shared operands")]
    pub fn edge0_weighted_sum(
        &mut self,
        experts: &DeviceVector,
        probabilities: &DeviceVector,
        shared: &DeviceVector,
        gate: &DeviceVector,
        width: usize,
        selected: usize,
        tokens: usize,
    ) -> Result<DeviceVector> {
        if width == 0
            || width > 65_536
            || selected == 0
            || selected > 512
            || !(1..=EDGE0_TOKEN_BATCH_LIMIT).contains(&tokens)
            || width.checked_mul(selected).and_then(|n| n.checked_mul(tokens)) != Some(experts.len)
            || selected * tokens != probabilities.len
            || width * tokens != shared.len
            || gate.len != tokens
        {
            return Err(Error::Invalid("Edge0 ordered reduction shape"));
        }
        let output = self.scratch(shared.len)?;
        self.kernel(
            "edge0_graph_weighted_sum",
            &[experts, probabilities, shared, gate, &output],
            &[width as u32, selected as u32, tokens as u32],
            shared.len,
            256,
        )?;
        Ok(output)
    }
}
