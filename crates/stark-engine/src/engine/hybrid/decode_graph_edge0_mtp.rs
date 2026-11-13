// SPDX-License-Identifier: Apache-2.0
//! Staged original-plane Edge0 MTP expert operators, with F32 arithmetic.
//! The transport is one retained raw expert record, never a GGML matrix.
//! No worker, model capability, or generation-cache admission is implied.

use super::*;

#[cfg(test)]
#[path = "decode_graph_edge0_mtp_tests.rs"]
mod tests;

struct ExpertGeometry {
    hidden: usize,
    intermediate: usize,
    gu_codes: usize,
    gu_scales_offset: usize,
    down_scales_offset: usize,
    record: usize,
}

impl ExpertGeometry {
    fn checked(
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        if hidden == 0
            || intermediate == 0
            || hidden > 2560
            || intermediate > 640
            || !hidden.is_multiple_of(64)
            || !intermediate.is_multiple_of(64)
        {
            return Err(Error::Invalid("Edge0 MTP Q2 geometry"));
        }
        let blocks = hidden
            .checked_mul(intermediate)
            .and_then(|n| n.checked_div(64))
            .ok_or(Error::Invalid("Edge0 MTP Q2 block overflow"))?;
        let gu_codes = blocks.checked_mul(32).ok_or(Error::Invalid("Edge0 MTP Q2 code overflow"))?;
        let gu_scales_offset = blocks.checked_mul(48).ok_or(Error::Invalid("Edge0 MTP Q2 plane overflow"))?;
        let down_scales_offset = blocks.checked_mul(52).ok_or(Error::Invalid("Edge0 MTP Q2 plane overflow"))?;
        let record = blocks.checked_mul(54).ok_or(Error::Invalid("Edge0 MTP Q2 record overflow"))?;
        Ok(Self {
            hidden,
            intermediate,
            gu_codes,
            gu_scales_offset,
            down_scales_offset,
            record,
        })
    }

    fn transport(
        &self,
        matrix: &MappedMatrix,
    ) -> Result<()> {
        let codes = &matrix.codes;
        if matrix.planes.is_some()
            || codes.length != self.record
            || codes.registered_length > codes.buffer.length()
            || codes.offset.checked_add(self.record).is_none_or(|end| end > codes.registered_length)
        {
            return Err(Error::Invalid("Edge0 MTP Q2 original record span"));
        }
        Ok(())
    }

    fn input(
        input: &DeviceVector,
        columns: usize,
        tokens: usize,
    ) -> Result<()> {
        input.validate()?;
        if !(1..=8).contains(&tokens) || columns.checked_mul(tokens) != Some(input.len) {
            return Err(Error::Invalid("Edge0 MTP Q2 input shape"));
        }
        Ok(())
    }
}

impl DecodeGraph {
    fn edge0_mtp_q2_dispatch(
        &mut self,
        name: &'static str,
        matrix: &MappedMatrix,
        input: &DeviceVector,
        output: &DeviceVector,
        geometry: &ExpertGeometry,
        tokens: usize,
    ) -> Result<()> {
        if self.unretained {
            return Err(Error::Unsupported("Edge0 MTP Q2 retained transport required"));
        }
        // An external dependent write cannot be hidden in a projection level.
        self.prepare_external_write(output)?;
        self.check_finite(input)?;
        let executor = executor()?;
        let pipeline = {
            let mut cache = executor.graph_pipelines.lock().map_err(|_| Error::Invalid("Edge0 MTP pipeline cache"))?;
            if let Some(pipeline) = cache.get(name) {
                pipeline.clone()
            } else {
                let function = executor
                    .library
                    .new_function_with_name(name)
                    .ok_or(Error::Invalid("Edge0 MTP Q2 kernel unavailable"))?;
                let pipeline = executor
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| Error::Invalid("Edge0 MTP Q2 kernel pipeline"))?;
                cache.insert(name, pipeline.clone());
                pipeline
            }
        };
        if pipeline.max_total_threads_per_threadgroup() < 256 || pipeline.thread_execution_width() != 32 {
            return Err(Error::Unsupported("Edge0 MTP Q2 SIMD geometry"));
        }
        let parameters = [
            geometry.hidden as u32,
            geometry.intermediate as u32,
            tokens as u32,
            geometry.gu_codes as u32,
            geometry.gu_scales_offset as u32,
            geometry.down_scales_offset as u32,
        ];
        self.encoding.set_compute_pipeline_state(&pipeline);
        self.encoding.set_buffer(Some(&matrix.codes.buffer), matrix.codes.offset, 0);
        self.resources.push(matrix.codes.buffer.clone());
        for (index, vector) in [input, output].iter().enumerate() {
            vector.validate()?;
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, index + 1);
            self.resources.push(vector.buffer.clone());
        }
        self.encoding.set_bytes(
            NonNull::new(parameters.as_ptr().cast_mut()).unwrap().cast(),
            std::mem::size_of_val(&parameters),
            3,
        );
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        self.order_dispatch();
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: output.len / tokens,
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

    /// Fused gate/up/SwiGLU over one unchanged original four-plane expert.
    /// Input/output are token-major F32; both matrix reductions use F32.
    pub fn edge0_mtp_q2_gate_up(
        &mut self,
        matrix: &MappedMatrix,
        hidden: usize,
        intermediate: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let geometry = ExpertGeometry::checked(hidden, intermediate)?;
        geometry.transport(matrix)?;
        ExpertGeometry::input(input, hidden, tokens)?;
        let output = self.scratch(intermediate * tokens)?;
        self.edge0_mtp_q2_dispatch("edge0_mtp_q2_gate_up", matrix, input, &output, &geometry, tokens)?;
        Ok(output)
    }

    /// Down projection from the original expert's separate down-code/scale planes.
    pub fn edge0_mtp_q2_down(
        &mut self,
        matrix: &MappedMatrix,
        hidden: usize,
        intermediate: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let geometry = ExpertGeometry::checked(hidden, intermediate)?;
        geometry.transport(matrix)?;
        ExpertGeometry::input(input, intermediate, tokens)?;
        let output = self.scratch(hidden * tokens)?;
        self.edge0_mtp_q2_dispatch("edge0_mtp_q2_down", matrix, input, &output, &geometry, tokens)?;
        Ok(output)
    }
}
