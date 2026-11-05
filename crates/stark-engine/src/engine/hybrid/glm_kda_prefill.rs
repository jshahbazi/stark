// SPDX-License-Identifier: Apache-2.0
//! Opt-in F32 multi-token KDA. Convolution histories and the recurrent backing
//! layout are unchanged, so existing exact prefix snapshots remain valid.
//!
//! Algorithmic reference: mlx-stream ac3b772b61bce66450862bca697dbb742c35453f,
//! src/arch/glm5_next/kda_percore.zig and kernels/kda_percore.metal. That kernel
//! credits oMLX 0.7.0, Copyright oMLX contributors, Apache-2.0; mlx-stream's host
//! code is Copyright (c) 2026 David Tai, MIT. This is an original implementation
//! of the segmented register recurrence, without their generated kernel text,
//! MLX transport, or BF16 policy. See docs/validation/GLM_KDA_PREFILL.md.

use super::*;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

struct Pipelines {
    prepare: Pipeline,
    recur: Pipeline,
    finish: Pipeline,
}

static PIPELINES: OnceLock<Option<Pipelines>> = OnceLock::new();

fn pipelines() -> Option<&'static Pipelines> {
    PIPELINES
        .get_or_init(|| {
            let e = executor().ok()?;
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            // Reuse the exact existing norm helper, including its extreme
            // input fallback. The decode kernels in this library are unused.
            let source = format!(
                "#define STARK_GLM_KDA_PARALLEL_NORM 1\n{}\n{}",
                include_str!("glm_metal.metal"),
                include_str!("glm_kda_prefill.metal")
            );
            let library = e.device.new_library_with_source(&source, Some(&options)).ok()?;
            let create = |name| {
                let function = library.new_function_with_name(name)?;
                e.device.new_compute_pipeline_state_with_function(&function).ok()
            };
            let result = Pipelines {
                prepare: create("glm_kda_prefill_prepare")?,
                recur: create("glm_kda_prefill_recur")?,
                finish: create("glm_kda_prefill_finish")?,
            };
            (result.prepare.max_total_threads_per_threadgroup() >= 128
                && result.finish.max_total_threads_per_threadgroup() >= 128
                && result.prepare.thread_execution_width() == 32
                && result.finish.thread_execution_width() == 32
                && result.recur.thread_execution_width() == 32
                && result.recur.max_total_threads_per_threadgroup() >= 32)
                .then_some(result)
        })
        .as_ref()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    groups: usize,
    rows: usize,
}

/// Even row boundaries preserve complete eight-lane reduction subgroups.
/// At most two heads share a group because each group carries <=128 rows.
fn plan_for(
    heads: usize,
    cores: usize,
    thread_cap: usize,
) -> Option<Plan> {
    let total = heads.checked_mul(128)?;
    if total == 0 || cores == 0 || thread_cap < 32 {
        return None;
    }
    let maximum = (thread_cap / 8).min(128) / 4 * 4;
    let per_core = total.div_ceil(cores * maximum);
    // A constrained kernel still gets a valid conservative schedule. This is
    // a scheduling fallback, with the same arithmetic and state ownership.
    let groups = (cores * per_core).min(total / 4);
    let rows = 4 * (total / 4).div_ceil(groups);
    (rows <= maximum).then_some(Plan {
        groups,
        rows,
    })
}

fn bind(
    graph: &DecodeGraph,
    pipeline: &Pipeline,
    buffers: &[&DeviceVector],
    parameters: &[u32; 10],
    groups: usize,
    threads: usize,
) {
    for vector in buffers {
        graph.retain_external_buffer(&vector.buffer);
    }
    let encoder = graph.encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, vector) in buffers.iter().enumerate() {
        encoder.set_buffer(Some(&vector.buffer), vector.offset, index);
    }
    encoder.set_bytes(NonNull::from(parameters).cast(), std::mem::size_of_val(parameters), buffers.len());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

impl Kda {
    /// Returns false without encoding when this optional route is unavailable.
    /// All spans are row-major; the model still owns submission, cancellation,
    /// completion, reset, and exact prefix snapshot boundaries.
    #[expect(clippy::too_many_arguments, reason = "separate existing projection buffers avoid per-token copies")]
    pub(in crate::engine::hybrid) fn encode_prefill_register(
        &mut self,
        graph: &mut DecodeGraph,
        q: &DeviceVector,
        k: &DeviceVector,
        v: &DeviceVector,
        forget: &DeviceVector,
        gate: &DeviceVector,
        beta: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
    ) -> Result<bool> {
        if !std::env::var_os("STARK_GLM53_KDA_REGISTER_PREFILL").is_some_and(|value| value == "1") {
            return Ok(false);
        }
        self.encode_prefill_register_inner(graph, q, k, v, forget, gate, beta, output, tokens)
    }

    #[expect(clippy::too_many_arguments, reason = "same checked layout as opt-in public entry point")]
    fn encode_prefill_register_inner(
        &mut self,
        graph: &mut DecodeGraph,
        q: &DeviceVector,
        k: &DeviceVector,
        v: &DeviceVector,
        forget: &DeviceVector,
        gate: &DeviceVector,
        beta: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
    ) -> Result<bool> {
        if self.dim() != 128 || !(1..=128).contains(&tokens) {
            return Ok(false);
        }
        let parallel_norm = self.parallel_norm_admitted(executor()?);
        self.encode_prefill_register_policy(graph, q, k, v, forget, gate, beta, output, tokens, parallel_norm)
    }

    #[expect(clippy::too_many_arguments, reason = "explicit selected policy enables source-bound operator checks")]
    fn encode_prefill_register_policy(
        &mut self,
        graph: &mut DecodeGraph,
        q: &DeviceVector,
        k: &DeviceVector,
        v: &DeviceVector,
        forget: &DeviceVector,
        gate: &DeviceVector,
        beta: &DeviceVector,
        output: &DeviceVector,
        tokens: usize,
        parallel_norm: bool,
    ) -> Result<bool> {
        if self.dim() != 128 || !(1..=128).contains(&tokens) {
            return Ok(false);
        }
        if parallel_norm && !self.transposed_state {
            return Err(Error::Invalid("Metal KDA prefill norm policy"));
        }
        let count = tokens.checked_mul(self.inner).ok_or(Error::Invalid("Metal KDA prefill dimensions"))?;
        for vector in [q, k, v, forget, gate, output] {
            if vector.len != count {
                return Err(Error::Invalid("Metal KDA prefill dimensions"));
            }
            vector.slice(0, count)?;
        }
        if beta.len != tokens * self.heads {
            return Err(Error::Invalid("Metal KDA prefill beta dimensions"));
        }
        beta.slice(0, beta.len)?;
        let Some(pipelines) = pipelines() else {
            return Ok(false);
        };
        let e = executor()?;
        use crate::backends::metal::DeviceExt;
        let Some(plan) = plan_for(
            self.heads,
            e.device.gpu_core_count() as usize,
            pipelines.recur.max_total_threads_per_threadgroup(),
        ) else {
            return Ok(false);
        };
        // All allocation/validation and admission precede the first encoder
        // mutation. A caller can safely run the scalar fallback on false.
        let prepared = graph.scratch(tokens * (4 * self.inner + self.heads))?;
        let values = graph.scratch(count)?;
        let weights = DeviceVector {
            buffer: self.weights.clone(),
            offset: 0,
            len: self.weights.length() / 4,
        };
        let state = DeviceVector {
            buffer: self.state.clone(),
            offset: 0,
            len: self.state_values,
        };
        let failure = DeviceVector {
            buffer: self.failure.clone(),
            offset: 0,
            len: 1,
        };
        let mut parameters = [0; 10];
        parameters[..6].copy_from_slice(&self.parameters);
        parameters[6] = tokens as u32;
        parameters[7] = u32::from(self.transposed_state) | (u32::from(parallel_norm) << 1);
        parameters[8] = plan.groups as u32;
        parameters[9] = plan.rows as u32;
        graph.prepare_external_write(output)?;
        // Never clear status from the CPU while an earlier graph may write.
        bind(
            graph,
            &pipelines.prepare,
            &[q, k, v, forget, beta, &weights, &state, &prepared, &failure],
            &parameters,
            self.heads,
            128,
        );
        bind(graph, &pipelines.recur, &[&prepared, &state, &values, &failure], &parameters, plan.groups, plan.rows * 8);
        bind(
            graph,
            &pipelines.finish,
            &[&values, gate, &weights, output, &failure],
            &parameters,
            tokens * self.heads,
            128,
        );
        dispatch(graph, "glm_kda_graph_status", &[&failure], &[0], 1, false)?;
        super::super::profile::record("glm.kda.prefill.register.tokens", 0.0, tokens);
        super::super::profile::record("glm.kda.prefill.register.groups", 0.0, plan.groups);
        super::super::profile::record("glm.kda.prefill.register.rows", 0.0, plan.rows);
        Ok(true)
    }
}

#[cfg(test)]
#[path = "glm_kda_prefill_tests.rs"]
mod tests;
