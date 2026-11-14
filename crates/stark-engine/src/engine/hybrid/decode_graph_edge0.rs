//! Checked bindings for Edge0's opt-in packed batched projections.

use super::*;

#[cfg(test)]
#[path = "decode_graph_edge0_tests.rs"]
mod tests;

fn specialize_single() -> bool {
    // Native workers fix their selectors before model loading. Cache once per
    // process rather than reading the environment for every selected expert.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("STARK_EDGE0_SPECIALIZE_SINGLE").is_some_and(|value| value == "1"))
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
    fn edge0_dispatch(
        &mut self,
        name: &'static str,
        matrices: &[&MappedMatrix],
        input: &DeviceVector,
        output: &DeviceVector,
        parameters: &[u32],
        rows: usize,
    ) -> Result<()> {
        if self.unretained || matrices.is_empty() || matrices.len() > 2 {
            return Err(Error::Unsupported("Edge0 packed batch transport"));
        }
        self.check_finite(input)?;
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
        for (index, vector) in [input, output].iter().enumerate() {
            vector.validate()?;
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, matrices.len() + index);
            self.resources.push(vector.buffer.clone());
        }
        self.encoding.set_bytes(
            NonNull::new(parameters.as_ptr().cast_mut()).unwrap().cast(),
            std::mem::size_of_val(parameters),
            matrices.len() + 2,
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
            input,
            &output,
            &[rows as u32, columns as u32, kind, block as u32, bytes as u32, tokens as u32],
            rows,
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
            input,
            &output,
            &[rows as u32, columns as u32, tokens as u32],
            rows,
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
            || !(1..=8).contains(&tokens)
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
