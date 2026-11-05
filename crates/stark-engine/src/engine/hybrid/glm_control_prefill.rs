// SPDX-License-Identifier: Apache-2.0
//! Token-major mHC prefill. Every token retains the selected row control's
//! F32 reduction tree, stream chronology, and failure checks.

use super::*;

#[derive(Clone, Copy)]
struct BatchShape {
    tokens: usize,
    columns: usize,
    hidden: usize,
    mix: usize,
    projection: usize,
}

impl Control {
    fn batch_shape(
        &self,
        tokens: usize,
    ) -> Result<BatchShape> {
        let multiply = |width: usize| {
            width
                .checked_mul(tokens)
                .filter(|&length| length <= u32::MAX as usize)
                .ok_or(Error::Invalid("Metal GLM control prefill dimensions"))
        };
        if !(1..=128).contains(&tokens) || !matches!(self.projection_chunks, 1 | 8 | 16 | 32) {
            return Err(Error::Invalid("Metal GLM control prefill dimensions"));
        }
        Ok(BatchShape {
            tokens,
            columns: multiply(self.parameters[3] as usize)?,
            hidden: multiply(self.input.len)?,
            mix: multiply(self.mix.len)?,
            projection: multiply(self.projection.len)?,
        })
    }

    fn batch_parameters(
        &self,
        tokens: usize,
    ) -> [u32; 8] {
        let p = self.parameters;
        [p[0], p[1], p[2], p[3], p[4], self.rms_eps.to_bits(), self.projection_chunks as u32, tokens as u32]
    }

    /// Encode all independent rows in a fixed number of dispatches. The mix
    /// is graph-owned immutable scratch, separate from this Control's row
    /// scratch, and remains valid until its matching post has completed.
    pub(in crate::engine::hybrid) fn pre_batch(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        normalized: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let shape = self.batch_shape(tokens)?;
        self.validate_batch_view(residual, shape.columns)?;
        self.validate_batch_view(normalized, shape.hidden)?;
        self.validate_batch_destination(normalized, &[residual])?;
        graph.prepare_external_write(normalized)?;
        // scratch() accounts for all allocations against the existing 512 MiB
        // graph limit. No separate batch allowance or persistent arena exists.
        let mix = graph.scratch(shape.mix)?;
        let fused = self.fused_parallel && self.rms_project && self.fused_parallel_admitted();
        if fused && !executor()?.graph_pipelines.contains_key("glm_control_prefill_fused_parallel") {
            // Optional PSOs may have different device resource limits. Retain
            // the selected scalar fusion's arithmetic if its batch PSO cannot
            // run, rather than silently selecting the unfused RMS tree.
            for token in 0..tokens {
                let row_mix = self.pre_batch_row(
                    graph,
                    &residual.slice(token * self.parameters[3] as usize, self.parameters[3] as usize)?,
                    &normalized.slice(token * self.input.len, self.input.len)?,
                )?;
                graph.copy_into(&row_mix, &mix.slice(token * self.mix.len, self.mix.len)?)?;
            }
            return Ok(mix);
        }
        let projection = graph.scratch(shape.projection)?;
        let parameters = self.batch_parameters(shape.tokens);
        if self.rms_project && self.parameters[0] == 4096 && self.parameters[1] == 4 && self.projection_chunks == 32 {
            let partials = graph.scratch(tokens * 16)?;
            graph.kernel(
                "hybrid_graph_rms_partials",
                &[residual, &partials],
                &[tokens as u32, 16384, self.rms_eps.to_bits(), 16384, 16],
                tokens * 16 * 256,
                256,
            )?;
            dispatch_with_threads(
                graph,
                "glm_control_prefill_rms_project_split32",
                &[residual, &self.weights, &partials, &projection],
                &parameters,
                shape.projection * 32,
                false,
                32,
            )?;
        } else {
            // rms_norm already treats each `group` as an independent row and
            // selects exactly the same ordered/prefetch/parallel policy.
            let input = graph.rms_norm(residual, &self.ones, self.parameters[3] as usize, self.rms_eps)?;
            dispatch_with_threads(
                graph,
                if self.projection_chunks == 1 {
                    "glm_control_prefill_project"
                } else {
                    "glm_control_prefill_project_parts"
                },
                &[&input, &self.weights, &projection],
                &parameters,
                shape.projection * 32,
                false,
                32,
            )?;
        }
        let input = graph.scratch(shape.hidden)?;
        if fused {
            dispatch_with_threads(
                graph,
                "glm_control_prefill_fused_parallel",
                &[&projection, &self.coefficients, residual, &self.normalization, &mix, &input, normalized],
                &parameters,
                tokens * 768,
                false,
                768,
            )?;
        } else {
            let threads = if self.simd_mix {
                32
            } else {
                64
            };
            dispatch_with_threads(
                graph,
                if self.simd_mix {
                    "glm_control_prefill_mix_simd"
                } else {
                    "glm_control_prefill_mix"
                },
                &[&projection, &self.coefficients, &mix],
                &parameters,
                tokens * threads,
                false,
                threads,
            )?;
            dispatch(graph, "glm_control_prefill_input", &[residual, &mix, &input], &parameters, shape.hidden, false)?;
            let value = graph.rms_norm(&input, &self.normalization, self.input.len, self.rms_eps)?;
            graph.copy_into(&value, normalized)?;
        }
        super::super::profile::record("glm.control.prefill.batch.tokens", 0.0, tokens);
        Ok(mix)
    }

    /// Complete token-major rows using the immutable mix produced by pre.
    pub(in crate::engine::hybrid) fn post_batch(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        output: &DeviceVector,
        mix: &DeviceVector,
        destination: &DeviceVector,
        tokens: usize,
    ) -> Result<()> {
        let shape = self.batch_shape(tokens)?;
        self.validate_batch_view(residual, shape.columns)?;
        self.validate_batch_view(output, shape.hidden)?;
        self.validate_batch_view(mix, shape.mix)?;
        self.validate_batch_view(destination, shape.columns)?;
        self.validate_batch_destination(destination, &[residual, output, mix])?;
        graph.prepare_external_write(destination)?;
        dispatch(
            graph,
            "glm_control_prefill_post",
            &[residual, output, mix, destination],
            &self.batch_parameters(tokens),
            shape.columns,
            false,
        )
    }
}

#[cfg(test)]
#[path = "glm_control_prefill_tests.rs"]
mod tests;
