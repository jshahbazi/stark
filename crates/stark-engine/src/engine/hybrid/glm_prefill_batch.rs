// SPDX-License-Identifier: Apache-2.0
//! Experimental layer-major chunks of four known prompt tokens. Only dense
//! projection scheduling changes: recurrent and MLA cache writers retain token
//! order, and each chunk is fenced before another graph is constructed.

use super::{
    super::{
        weight_metal::{DecodeGraph, DeviceVector},
        weights::BoundedBankGroup,
    },
    *,
};

impl GlmModel {
    pub(super) fn ingest_prefix_chunked(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
        use_mma: bool,
    ) -> Result<()> {
        let result = (|| {
            if self.cleanup_failed {
                return Err(Error::Invalid("GLM context requires reload after cleanup failure"));
            }
            if tokens.is_empty() {
                return Ok(());
            }
            let enabled = |name| std::env::var_os(name).is_some_and(|value| value == "1");
            if !self.supports_greedy_readout()
                || !enabled("STARK_GLM53_RESIDENT_BANKS")
                || !enabled("STARK_GLM53_WIDE_ENCODER")
                || enabled("STARK_GLM53_CONCURRENT_ENCODER")
                || enabled("STARK_GLM53_PARALLEL_PROJECTIONS")
            {
                return Err(Error::Unsupported("GLM chunk prefill requires serial resident wide-encoder overlap"));
            }
            if tokens.iter().any(|&token| token as usize >= self.config.vocab)
                || self.position.checked_add(tokens.len()).is_none_or(|end| end > self.config.max_context)
            {
                return Err(Error::Invalid("GLM chunk prefill token or context boundary"));
            }
            self.last_greedy = None;
            if self.decode.is_none() {
                self.decode = Some(MetalDecode::new(&self.weights, &self.config, self.hyper_connections.as_deref())?);
                #[cfg(not(test))]
                {
                    self.hyper_connections = None;
                }
            }
            let decode = self.decode.as_mut().ok_or(Error::Invalid("GLM chunk decode state"))?;
            let plan = decode.bounded_plan.clone().ok_or(Error::Invalid("GLM chunk bank plan"))?;
            if !decode.encoder_banks || plan.groups.len() != 1 || decode.parallel_projections {
                return Err(Error::Unsupported("GLM chunk prefill requires one admitted serial encoder group"));
            }
            for chunk in tokens.chunks(4) {
                objc2::rc::autoreleasepool(|_| -> Result<()> {
                    let _encode = super::super::profile::Scope::new("native.glm.prefill.chunk_encode_cpu");
                    let c = &self.config;
                    let mut values = Vec::with_capacity(chunk.len() * c.hidden * c.hc);
                    for &token in chunk {
                        before_token()?;
                        let embedding = self.weights.row("token_embd.weight", token as usize)?;
                        if embedding.len() != c.hidden || embedding.iter().any(|value| !value.is_finite()) {
                            return Err(Error::Invalid("GLM chunk embedding values"));
                        }
                        for _ in 0..c.hc {
                            values.extend_from_slice(&embedding);
                        }
                    }
                    let mut graph = new_encoder_graph(decode)?;
                    let mut streams = graph.upload_into(&values)?;
                    let group = &plan.groups[0];
                    let banks = self.weights.encoder_indexed_group(&group.names, plan.budget)?;
                    if banks.registered_bytes() > group.bytes {
                        return Err(Error::Invalid("GLM chunk encoder inventory changed"));
                    }
                    for layer in group.start..group.end {
                        streams = encode_layer(
                            &self.weights,
                            c,
                            decode,
                            layer,
                            &mut self.states[layer],
                            &mut graph,
                            &streams,
                            &banks,
                            chunk.len(),
                            use_mma,
                        )?;
                    }
                    graph.check_finite(&streams)?;
                    // A cancelled unsubmitted graph drops before the outer
                    // reset. No fallible callback occurs after submission.
                    before_token()?;
                    drop(banks);
                    drop(_encode);
                    graph.submit(&[])?.wait()?;
                    for state in &self.states {
                        if let LayerState::Kda(state) = state
                            && let Some(metal) = &state.metal
                        {
                            metal.check_completed()?;
                        }
                    }
                    self.position += chunk.len();
                    super::super::profile::record("native.glm.prefill.chunk_tokens", 0.0, chunk.len());
                    Ok(())
                })?;
            }
            Ok(())
        })();
        // The chunk is either unsubmitted and dropped, or explicitly drained
        // before any error can reach this reset. No GPU writer survives return.
        if result.is_err() {
            self.reset();
        }
        result
    }
}

fn project(
    banks: &BoundedBankGroup<'_>,
    graph: &mut DecodeGraph,
    prefix: &str,
    name: &str,
    input: &DeviceVector,
    tokens: usize,
    use_mma: bool,
) -> Result<DeviceVector> {
    banks.encode_same_weight_batch(graph, &format!("{prefix}.{name}"), input, tokens, use_mma)
}

fn controls_pre(
    control: &super::super::glm_metal::Control,
    graph: &mut DecodeGraph,
    residual: &DeviceVector,
    hidden: usize,
    streams: usize,
    tokens: usize,
) -> Result<(DeviceVector, Vec<DeviceVector>)> {
    let normalized = graph.scratch(tokens * hidden)?;
    let mut mixes = Vec::with_capacity(tokens);
    for token in 0..tokens {
        mixes.push(control.pre_batch_row(
            graph,
            &residual.slice(token * hidden * streams, hidden * streams)?,
            &normalized.slice(token * hidden, hidden)?,
        )?);
    }
    Ok((normalized, mixes))
}

fn controls_post(
    control: &super::super::glm_metal::Control,
    graph: &mut DecodeGraph,
    residual: &DeviceVector,
    output: &DeviceVector,
    mixes: &[DeviceVector],
    hidden: usize,
    streams: usize,
) -> Result<DeviceVector> {
    let next = graph.scratch(residual.len)?;
    for (token, mix) in mixes.iter().enumerate() {
        control.post_batch_row(
            graph,
            &residual.slice(token * hidden * streams, hidden * streams)?,
            &output.slice(token * hidden, hidden)?,
            mix,
            &next.slice(token * hidden * streams, hidden * streams)?,
        )?;
    }
    Ok(next)
}

#[expect(clippy::too_many_arguments, reason = "one admitted layer with explicit token/state ownership")]
fn encode_layer(
    w: &Weights,
    c: &Config,
    decode: &mut MetalDecode,
    layer: usize,
    state: &mut LayerState,
    graph: &mut DecodeGraph,
    streams: &DeviceVector,
    banks: &BoundedBankGroup<'_>,
    tokens: usize,
    use_mma: bool,
) -> Result<DeviceVector> {
    let p = format!("blk.{layer}");
    let (norm, mix) = controls_pre(&decode.controls[layer][0], graph, streams, c.hidden, c.hc, tokens)?;
    let attention = match state {
        LayerState::Kda(state) => encode_kda_batch(w, c, graph, &norm, &p, state, banks, tokens, use_mma)?,
        LayerState::Attention(_) => {
            let output = graph.scratch(tokens * c.hidden)?;
            let state = decode.attention[layer].as_mut().ok_or(Error::Invalid("GLM chunk MLA state"))?;
            // Append then attend each token before appending its successor.
            // Cache pooling/tail compression therefore keeps causal chronology.
            for token in 0..tokens {
                let row = encode_mla(w, c, graph, &norm.slice(token * c.hidden, c.hidden)?, &p, state, false)?;
                graph.copy_into(&row, &output.slice(token * c.hidden, c.hidden)?)?;
            }
            output
        },
    };
    let residual = controls_post(&decode.controls[layer][0], graph, streams, &attention, &mix, c.hidden, c.hc)?;
    let (norm, mix) = controls_pre(&decode.controls[layer][1], graph, &residual, c.hidden, c.hc, tokens)?;
    let ffn = if layer < c.dense_layers {
        let gate = project(banks, graph, &p, "ffn_gate.weight", &norm, tokens, use_mma)?;
        let up = project(banks, graph, &p, "ffn_up.weight", &norm, tokens, use_mma)?;
        let active = graph.swiglu(&gate, &up, Some(c.swiglu_limit))?;
        project(banks, graph, &p, "ffn_down.weight", &active, tokens, use_mma)?
    } else {
        let logits = project(banks, graph, &p, "ffn_gate_inp.weight", &norm, tokens, use_mma)?;
        let gate = project(banks, graph, &p, "ffn_gate_shexp.weight", &norm, tokens, use_mma)?;
        let up = project(banks, graph, &p, "ffn_up_shexp.weight", &norm, tokens, use_mma)?;
        let active = graph.swiglu(&gate, &up, Some(c.swiglu_limit))?;
        let shared = project(banks, graph, &p, "ffn_down_shexp.weight", &active, tokens, use_mma)?;
        let output = graph.scratch(tokens * c.hidden)?;
        let bias = decode.routing_biases[layer].as_ref().ok_or(Error::Invalid("GLM chunk router bias"))?;
        let scale = graph.upload_into(&[1.0])?;
        for token in 0..tokens {
            let norm = norm.slice(token * c.hidden, c.hidden)?;
            let (indices, probabilities) = super::super::glm_metal::route(
                graph,
                &logits.slice(token * c.experts, c.experts)?,
                bias,
                c.active_experts,
                c.routed_scale,
            )?;
            let gate_name = format!("{p}.ffn_gate_exps.weight");
            let up_name = format!("{p}.ffn_up_exps.weight");
            let active = if let Some(active) = banks.try_encode_iq2_gate_up(
                graph,
                &gate_name,
                &up_name,
                &indices,
                c.active_experts,
                &norm,
                c.swiglu_limit,
            )? {
                active
            } else {
                let gate = banks.encode_indexed_matvec(graph, &gate_name, &indices, c.active_experts, &norm, false)?;
                let up = banks.encode_indexed_matvec(graph, &up_name, &indices, c.active_experts, &norm, false)?;
                graph.swiglu(&gate, &up, Some(c.swiglu_limit))?
            };
            let down = banks.encode_indexed_matvec(
                graph,
                &format!("{p}.ffn_down_exps.weight"),
                &indices,
                c.active_experts,
                &active,
                true,
            )?;
            let row =
                graph.weighted_sum_device(&down, &probabilities, &shared.slice(token * c.hidden, c.hidden)?, &scale)?;
            graph.copy_into(&row, &output.slice(token * c.hidden, c.hidden)?)?;
        }
        output
    };
    controls_post(&decode.controls[layer][1], graph, &residual, &ffn, &mix, c.hidden, c.hc)
}

#[expect(clippy::too_many_arguments, reason = "chronological recurrence between two same-weight projection levels")]
fn encode_kda_batch(
    w: &Weights,
    c: &Config,
    graph: &mut DecodeGraph,
    input: &DeviceVector,
    p: &str,
    state: &mut KdaState,
    banks: &BoundedBankGroup<'_>,
    tokens: usize,
    use_mma: bool,
) -> Result<DeviceVector> {
    initialize_kda(w, c, p, state)?;
    let metal = state.metal.as_mut().ok_or(Error::Invalid("GLM chunk KDA state"))?;
    let inner = metal.inner();
    let projections =
        ["attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_beta.weight", "ssm_f_a.weight", "ssm_g_a.weight"]
            .into_iter()
            .map(|name| project(banks, graph, p, name, input, tokens, use_mma))
            .collect::<Result<Vec<_>>>()?;
    let forget = project(banks, graph, p, "ssm_f_b.weight", &projections[4], tokens, use_mma)?;
    let gate = project(banks, graph, p, "ssm_g_b.weight", &projections[5], tokens, use_mma)?;
    let persistent_input = DeviceVector {
        buffer: metal.input_buffer().clone(),
        offset: 0,
        len: 5 * inner + metal.heads(),
    };
    let persistent_output = DeviceVector {
        buffer: metal.output_buffer().clone(),
        offset: 0,
        len: inner,
    };
    let recurrent = graph.scratch(tokens * inner)?;
    for token in 0..tokens {
        for (source, offset, width) in [
            (&projections[0], 0, inner),
            (&projections[1], inner, inner),
            (&projections[2], 2 * inner, inner),
            (&projections[3], 5 * inner, metal.heads()),
            (&forget, 3 * inner, inner),
            (&gate, 4 * inner, inner),
        ] {
            graph.copy_into(&source.slice(token * width, width)?, &persistent_input.slice(offset, width)?)?;
        }
        graph.prepare_external_write(&persistent_output)?;
        metal.encode_graph(graph)?;
        // KDA output is mutable model scratch, not a per-token value. Stash it
        // before the next recurrence reuses the buffer.
        graph.copy_into(&persistent_output, &recurrent.slice(token * inner, inner)?)?;
    }
    project(banks, graph, p, "attn_output.weight", &recurrent, tokens, use_mma)
}

#[cfg(test)]
#[path = "glm_prefill_batch_tests.rs"]
mod tests;
