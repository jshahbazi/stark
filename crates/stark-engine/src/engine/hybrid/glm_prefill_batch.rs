// SPDX-License-Identifier: Apache-2.0
//! Experimental layer-major chunks of known prompt tokens. Dense projections
//! retain their qualified few-row tiles; optional expert batches share decoded
//! weights across routes. Recurrent and MLA cache writers retain token order,
//! and each chunk is fenced before another graph is constructed.

use super::{
    super::{
        weight_metal::{DecodeGraph, DeviceVector},
        weights::BoundedBankGroup,
    },
    *,
};

pub(super) fn chunk_limit(value: Option<&std::ffi::OsStr>) -> Result<usize> {
    match value.and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => Ok(4),
        Some("4") => Ok(4),
        Some("8") => Ok(8),
        Some("16") => Ok(16),
        Some("32") => Ok(32),
        Some("64") => Ok(64),
        _ => Err(Error::Invalid("GLM prefill chunk limit must be 4, 8, 16, 32 or 64")),
    }
}

fn chunk_sizes(
    mut tokens: usize,
    maximum: usize,
) -> Result<Vec<usize>> {
    if ![4, 8, 16, 32, 64].contains(&maximum) {
        return Err(Error::Invalid("GLM prefill chunk limit must be 4, 8, 16, 32 or 64"));
    }
    let mut sizes = Vec::new();
    for width in [64, 32, 16, 8, 4] {
        if width <= maximum {
            sizes.extend(std::iter::repeat_n(width, tokens / width));
            tokens %= width;
        }
    }
    if tokens != 0 {
        sizes.push(tokens);
    }
    Ok(sizes)
}

impl GlmModel {
    pub(super) fn ingest_prefix_chunked(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
        use_mma: bool,
        maximum_chunk: usize,
    ) -> Result<()> {
        let result = (|| {
            if self.cleanup_failed {
                return Err(Error::Invalid("GLM context requires reload after cleanup failure"));
            }
            let sizes = chunk_sizes(tokens.len(), maximum_chunk)?;
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
            let mut remaining = tokens;
            for size in sizes {
                let (chunk, tail) = remaining.split_at(size);
                remaining = tail;
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
                    // Only these two stream buffers survive a layer. Reuse
                    // temporary storage after all of that layer's readers and
                    // finite scans, within the same ordered serial encoder.
                    let mut next_streams = if maximum_chunk > 8 {
                        Some(graph.scratch(values.len())?)
                    } else {
                        None
                    };
                    let checkpoint = if next_streams.is_some() {
                        Some(graph.scratch_checkpoint()?)
                    } else {
                        None
                    };
                    let group = &plan.groups[0];
                    let banks = self.weights.encoder_indexed_group(&group.names, plan.budget)?;
                    if banks.registered_bytes() > group.bytes {
                        return Err(Error::Invalid("GLM chunk encoder inventory changed"));
                    }
                    for layer in group.start..group.end {
                        let next = encode_layer(
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
                        if let (Some(target), Some(checkpoint)) = (&mut next_streams, &checkpoint) {
                            graph.copy_into(&next, target)?;
                            graph.scratch_rewind(checkpoint)?;
                            std::mem::swap(&mut streams, target);
                        } else {
                            streams = next;
                        }
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
    let name = format!("{prefix}.{name}");
    if tokens <= 8 {
        return banks.encode_same_weight_batch(graph, &name, input, tokens, use_mma);
    }
    // Retain the qualified few-row F32 MMA tile and scalar tails. The larger
    // graph batch primarily exposes repeated expert weights across tokens.
    let columns = input.len / tokens;
    let mut output = None;
    let mut start = 0;
    for width in chunk_sizes(tokens, 8)? {
        let tile = banks.encode_same_weight_batch(
            graph,
            &name,
            &input.slice(start * columns, width * columns)?,
            width,
            use_mma,
        )?;
        let rows = tile.len / width;
        let target = match &output {
            Some(value) => value,
            None => output.insert(graph.scratch(rows * tokens)?),
        };
        graph.copy_into(&tile, &target.slice(start * rows, width * rows)?)?;
        start += width;
    }
    output.ok_or(Error::Invalid("GLM empty projection batch"))
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
            let state = decode.attention[layer].as_mut().ok_or(Error::Invalid("GLM chunk MLA state"))?;
            if std::env::var_os("STARK_GLM53_PREFILL_MLA_BATCH").is_some_and(|value| value == "1") {
                encode_mla_batch(w, c, graph, &norm, &p, state, banks, tokens, use_mma)?
            } else {
                let output = graph.scratch(tokens * c.hidden)?;
                // Append then attend each token before appending its successor.
                // Cache pooling/tail compression therefore keeps causal chronology.
                for token in 0..tokens {
                    let row = encode_mla(w, c, graph, &norm.slice(token * c.hidden, c.hidden)?, &p, state, false)?;
                    graph.copy_into(&row, &output.slice(token * c.hidden, c.hidden)?)?;
                }
                output
            }
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
        let scale = DecodeGraph::upload(&[1.0])?;
        if std::env::var_os("STARK_GLM53_PREFILL_EXPERT_BATCH").is_some_and(|value| value == "1") {
            let indices = graph.scratch(tokens * c.active_experts)?;
            let probabilities = graph.scratch(tokens * c.active_experts)?;
            for token in 0..tokens {
                let (ids, weights) = super::super::glm_metal::route(
                    graph,
                    &logits.slice(token * c.experts, c.experts)?,
                    bias,
                    c.active_experts,
                    c.routed_scale,
                )?;
                // Router choices are integer bit patterns, including F32
                // subnormal encodings. Keep this transfer integer-only.
                graph.kernel(
                    "hybrid_graph_route_copy",
                    &[&ids, &indices.slice(token * c.active_experts, c.active_experts)?],
                    &[c.active_experts as u32],
                    c.active_experts,
                    256,
                )?;
                graph.copy_into(&weights, &probabilities.slice(token * c.active_experts, c.active_experts)?)?;
            }
            let gate = banks.encode_indexed_batch(
                graph,
                &format!("{p}.ffn_gate_exps.weight"),
                &indices,
                c.active_experts,
                &norm,
                tokens,
                false,
            )?;
            let up = banks.encode_indexed_batch(
                graph,
                &format!("{p}.ffn_up_exps.weight"),
                &indices,
                c.active_experts,
                &norm,
                tokens,
                false,
            )?;
            let active = graph.swiglu(&gate, &up, Some(c.swiglu_limit))?;
            let down = banks.encode_indexed_batch(
                graph,
                &format!("{p}.ffn_down_exps.weight"),
                &indices,
                c.active_experts,
                &active,
                tokens,
                true,
            )?;
            for token in 0..tokens {
                let row = graph.weighted_sum_device(
                    &down.slice(token * c.active_experts * c.hidden, c.active_experts * c.hidden)?,
                    &probabilities.slice(token * c.active_experts, c.active_experts)?,
                    &shared.slice(token * c.hidden, c.hidden)?,
                    &scale,
                )?;
                graph.copy_into(&row, &output.slice(token * c.hidden, c.hidden)?)?;
            }
            return controls_post(&decode.controls[layer][1], graph, &residual, &output, &mix, c.hidden, c.hc);
        }
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
            let gate = banks.encode_indexed_matvec(graph, &gate_name, &indices, c.active_experts, &norm, false)?;
            let up = banks.encode_indexed_matvec(graph, &up_name, &indices, c.active_experts, &norm, false)?;
            let active = graph.swiglu(&gate, &up, Some(c.swiglu_limit))?;
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

#[expect(clippy::too_many_arguments, reason = "batched stateless projections around chronological MLA cache writers")]
fn encode_mla_batch(
    w: &Weights,
    c: &Config,
    graph: &mut DecodeGraph,
    input: &DeviceVector,
    p: &str,
    state: &mut super::super::glm_metal::Attention,
    banks: &BoundedBankGroup<'_>,
    tokens: usize,
    use_mma: bool,
) -> Result<DeviceVector> {
    // Every input-dependent rank-two projection can share weights across the
    // prompt chunk without observing or mutating attention state.
    let q = project(banks, graph, p, "attn_q_a.weight", input, tokens, use_mma)?;
    let latent = project(banks, graph, p, "attn_kv_a_mqa.weight", input, tokens, use_mma)?;
    let key = project(banks, graph, p, "indexer.attn_k.weight", input, tokens, use_mma)?;
    let gate = project(banks, graph, p, "indexer_compressor_gate.weight", input, tokens, use_mma)?;
    let weights = project(banks, graph, p, "indexer.proj.weight", input, tokens, use_mma)?;

    // These immutable, independently owned uploads stay outside the recyclable
    // graph arena. Host writes into an earlier scratch scope would race with
    // queued GPU reads when larger chunks rewind their per-layer workspace.
    let qnorm = DecodeGraph::upload(&w.vector(&format!("{p}.attn_q_a_norm.weight"))?)?;
    let kvnorm = DecodeGraph::upload(&w.vector(&format!("{p}.attn_kv_a_norm.weight"))?)?;
    let keynorm = DecodeGraph::upload(&w.vector(&format!("{p}.indexer.k_norm.weight"))?)?;
    let keybias = DecodeGraph::upload(&w.vector(&format!("{p}.indexer.k_norm.bias"))?)?;
    let normalized_q = graph.scratch(tokens * c.q_rank)?;
    let normalized_latent = graph.scratch(tokens * c.kv_rank)?;
    let normalized_key = graph.scratch(tokens * c.index_dim)?;
    for token in 0..tokens {
        let row = graph.rms_norm(&q.slice(token * c.q_rank, c.q_rank)?, &qnorm, c.q_rank, c.rms_eps)?;
        graph.copy_into(&row, &normalized_q.slice(token * c.q_rank, c.q_rank)?)?;
        let row = graph.rms_norm(&latent.slice(token * c.kv_rank, c.kv_rank)?, &kvnorm, c.kv_rank, c.rms_eps)?;
        graph.copy_into(&row, &normalized_latent.slice(token * c.kv_rank, c.kv_rank)?)?;
        let row = super::super::glm_metal::index_layer_norm(
            graph,
            &key.slice(token * c.index_dim, c.index_dim)?,
            &keynorm,
            &keybias,
        )?;
        graph.copy_into(&row, &normalized_key.slice(token * c.index_dim, c.index_dim)?)?;
    }
    let query = project(banks, graph, p, "attn_q_b.weight", &normalized_q, tokens, use_mma)?;
    let index_query = project(banks, graph, p, "indexer.attn_q_b.weight", &normalized_q, tokens, use_mma)?;

    // Only these operations consume or mutate chronological MLA/indexer state.
    // Attend to token t before appending token t+1, including each pool/tail
    // compression boundary. Rank-three K_B/V_B retain their existing kernels.
    let output_width = c.heads * c.value_dim;
    let output = graph.scratch(tokens * output_width)?;
    for token in 0..tokens {
        state.append(
            graph,
            &normalized_latent.slice(token * c.kv_rank, c.kv_rank)?,
            &normalized_key.slice(token * c.index_dim, c.index_dim)?,
            &gate.slice(token * c.index_dim, c.index_dim)?,
        )?;
        let query = w.encode_batched_matvec(
            graph,
            &format!("{p}.attn_k_b.weight"),
            &query.slice(token * c.heads * c.qk_dim, c.heads * c.qk_dim)?,
        )?;
        let attended = state.attend(
            graph,
            &query,
            &index_query.slice(token * c.index_heads * c.index_dim, c.index_heads * c.index_dim)?,
            &weights.slice(token * c.index_heads, c.index_heads)?,
        )?;
        let row = w.encode_batched_matvec(graph, &format!("{p}.attn_v_b.weight"), &attended)?;
        graph.copy_into(&row, &output.slice(token * output_width, output_width)?)?;
    }
    // The final wide projection now reaches the existing same-weight eight-row
    // tile admission. Its separately gated MMA policy controls reassociation;
    // batching alone leaves normalization and attention/cache math unchanged.
    let output = project(banks, graph, p, "attn_output.weight", &output, tokens, use_mma)?;
    super::super::profile::record("native.glm.prefill.mla_batch", 0.0, tokens);
    Ok(output)
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
