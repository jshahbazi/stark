// SPDX-License-Identifier: Apache-2.0
//! Experimental layer-major prompt prefill. The qualified T4/T8 projection
//! groups remain intact inside wider, memory-admitted token tiles. Recurrent
//! and MLA writers retain causal order across bounded command owners.

use metal::MTLBuffer;

use super::{
    super::{
        weight_metal::{DecodeGraph, DeviceVector, PendingDecodeGraph},
        weights::BoundedBankGroup,
    },
    *,
};

const MIB: usize = 1024 * 1024;
// The required greedy-overlap admission reserves two 512 MiB graph owners.
// Each wide owner uses at most 448 MiB plus one <=32 MiB source arena, leaving
// headroom without increasing the existing resident-bank allowance.
const WIDE_OWNER_BYTES: usize = 448 * MIB;
const WIDE_CARRY_BYTES: usize = 32 * MIB;
const PREFIX_GRID: usize = 1024;

fn wide_limit(value: Option<&std::ffi::OsStr>) -> Result<Option<usize>> {
    match value.and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => Ok(None),
        Some("16") => Ok(Some(16)),
        Some("32") => Ok(Some(32)),
        Some("64") => Ok(Some(64)),
        Some("128") => Ok(Some(128)),
        _ => Err(Error::Invalid("GLM wide prefill tile must be 16, 32, 64 or 128")),
    }
}

fn next_tile(
    position: usize,
    remaining: usize,
    maximum: usize,
) -> usize {
    let available = remaining.min(PREFIX_GRID - position % PREFIX_GRID).min(maximum);
    if available < 4 {
        available
    } else {
        1usize << available.ilog2()
    }
}

#[derive(Clone, Debug)]
struct WideGroup {
    start: usize,
    end: usize,
    scratch: usize,
    external: usize,
}

/// Conservative upper bounds, independent of environment-selected operator
/// schedules. Every allocation is charged 256 bytes of padding; arena backing
/// is bounded by twice the charged payload plus its initial one-MiB slab.
/// External MLA scores/masks and immutable per-row norm uploads are charged
/// separately. Cache growth is already in encoder_state_allowance's two-copy
/// context inventory, including intermediate doubling allocations.
fn layer_workspace(
    c: &Config,
    layer: usize,
    tokens: usize,
    end: usize,
) -> Result<(usize, usize)> {
    let (h, s, t) = (c.hidden as u128, c.hc as u128, tokens as u128);
    let inner = c.kda_heads as u128 * c.kda_dim as u128;
    // Two controls, both fallback row scratch and the fully batched controls.
    let controls = 4 * h * s + 6 * h + 80 * s * (s + 2);
    let attention = if c.layer_types[layer] {
        // Projection concatenation and its T8 temporary results, optional
        // register-prefill prep/finish and the recurrent output projection.
        18 * inner + 4 * h + 8 * c.kda_dim as u128 + 8 * c.kda_heads as u128
    } else {
        4 * c.q_rank as u128
            + 4 * c.kv_rank as u128
            + 8 * c.index_dim as u128
            + 4 * c.heads as u128 * (c.qk_dim + c.kv_rank + c.value_dim) as u128
            + 4 * c.index_heads as u128 * c.index_dim as u128
            + 4 * c.index_heads as u128
            + 2 * h
            + c.heads as u128 * end as u128
            + 4 * end.div_ceil(c.index_pool).next_power_of_two() as u128
    };
    let ffn = if layer < c.dense_layers {
        5 * c.intermediate as u128 + 2 * h
    } else {
        5 * c.shared_experts as u128 * c.expert_intermediate as u128
            + c.active_experts as u128 * (3 * c.expert_intermediate as u128 + h)
            + 4 * h
            + 5 * c.experts as u128
            + 4 * c.active_experts as u128
    };
    // Fewer than 128 graph scratch allocations per token in every schedule,
    // including T8 projection concatenation and legacy one-row controls.
    let payload = t * (controls + attention + ffn) * 4 + t * 128 * 256;
    let scratch = 2 * payload + MIB as u128;
    let external = if c.layer_types[layer] {
        0
    } else {
        // Scores are charged here even when they fit in graph scratch. This
        // bounds the >8M-element separately allocated attention score path.
        t * 4
            * (c.heads as u128 * end as u128
                + end.div_ceil(c.index_pool) as u128
                + c.q_rank as u128
                + c.kv_rank as u128
                + 2 * c.index_dim as u128)
    };
    Ok((
        usize::try_from(scratch).map_err(|_| Error::Invalid("GLM wide workspace overflow"))?,
        usize::try_from(external).map_err(|_| Error::Invalid("GLM wide workspace overflow"))?,
    ))
}

fn wide_groups(
    c: &Config,
    tokens: usize,
    end: usize,
    maximum_layers: usize,
) -> Result<Vec<WideGroup>> {
    let largest_batch_row = [
        c.hidden as u128 * c.hc as u128,
        c.kda_heads as u128 * c.kda_dim as u128,
        c.intermediate as u128,
        c.shared_experts as u128 * c.expert_intermediate as u128,
        c.experts as u128,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    if tokens == 0
        || tokens > 128
        || maximum_layers == 0
        || largest_batch_row * tokens as u128 > 8_000_000
        || end < tokens
        || end > c.max_context
        || tokens
            .checked_mul(c.hidden)
            .and_then(|n| n.checked_mul(c.hc))
            .is_none_or(|n| n > 8_000_000 || n * 4 > WIDE_CARRY_BYTES)
    {
        return Err(Error::Invalid("GLM wide tile geometry exceeds its carry allowance"));
    }
    let mut groups: Vec<WideGroup> = Vec::new();
    for layer in 0..c.layer_types.len() {
        let (scratch, external) = layer_workspace(c, layer, tokens, end)?;
        if scratch.checked_add(external).is_none_or(|n| n > WIDE_OWNER_BYTES) {
            return Err(Error::Unsupported("GLM wide layer exceeds its workspace allowance"));
        }
        if let Some(group) = groups.last_mut()
            && group.end - group.start < maximum_layers
            && group
                .scratch
                .checked_add(group.external)
                .and_then(|n| n.checked_add(scratch))
                .and_then(|n| n.checked_add(external))
                .is_some_and(|n| n <= WIDE_OWNER_BYTES)
        {
            group.end = layer + 1;
            group.scratch += scratch;
            group.external += external;
        } else {
            groups.push(WideGroup {
                start: layer,
                end: layer + 1,
                scratch,
                external,
            });
        }
    }
    Ok(groups)
}

pub(super) fn chunk_limit(value: Option<&std::ffi::OsStr>) -> Result<usize> {
    match value.and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => Ok(4),
        Some("4") => Ok(4),
        Some("8") => Ok(8),
        _ => Err(Error::Invalid("GLM prefill chunk limit must be four or eight")),
    }
}

fn eight_token_prefix(
    tokens: usize,
    maximum: usize,
) -> Result<usize> {
    match maximum {
        4 => Ok(0),
        8 => Ok(tokens / 8 * 8),
        _ => Err(Error::Invalid("GLM prefill chunk limit must be four or eight")),
    }
}

impl GlmModel {
    pub(super) fn ingest_prefix_chunked(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
        use_mma: bool,
        maximum_chunk: usize,
    ) -> Result<()> {
        self.ingest_prefix_chunked_grouped(tokens, before_token, use_mma, maximum_chunk, usize::MAX)
    }

    /// Production always uses the unrestricted layer count above. Private
    /// fixture tests can force ownership boundaries without a runtime flag or
    /// reducing the production workspace allowance.
    fn ingest_prefix_chunked_grouped(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
        use_mma: bool,
        maximum_chunk: usize,
        maximum_group_layers: usize,
    ) -> Result<()> {
        let result = (|| {
            if self.cleanup_failed {
                return Err(Error::Invalid("GLM context requires reload after cleanup failure"));
            }
            let eight_prefix = eight_token_prefix(tokens.len(), maximum_chunk)?;
            let wide = wide_limit(std::env::var_os("STARK_GLM53_PREFILL_WIDE").as_deref())?;
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
            if let Some(maximum) = wide {
                return self.ingest_prefix_wide(tokens, before_token, use_mma, maximum, maximum_group_layers);
            }
            let decode = self.decode.as_mut().ok_or(Error::Invalid("GLM chunk decode state"))?;
            let plan = decode.bounded_plan.clone().ok_or(Error::Invalid("GLM chunk bank plan"))?;
            if !decode.encoder_banks || plan.groups.len() != 1 || decode.parallel_projections {
                return Err(Error::Unsupported("GLM chunk prefill requires one admitted serial encoder group"));
            }
            // Prefer complete eights; retain the qualified Q6/T4 tail before
            // the final one-to-three legacy tokens (13 => 8 + 4 + 1).
            for chunk in tokens[..eight_prefix].chunks(8).chain(tokens[eight_prefix..].chunks(4)) {
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

    fn ingest_prefix_wide(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
        use_mma: bool,
        maximum: usize,
        maximum_group_layers: usize,
    ) -> Result<()> {
        // Keep the submitted owner outside the fallible encoding scope. On an
        // error, the future graph drops first, then every live writer drains.
        let mut pending: Option<PendingDecodeGraph> = None;
        let mut execution_failed = false;
        let result = (|| {
            let c = &self.config;
            let decode = self.decode.as_mut().ok_or(Error::Invalid("GLM wide decode state"))?;
            let plan = decode.bounded_plan.clone().ok_or(Error::Invalid("GLM wide bank plan"))?;
            if !decode.encoder_banks || plan.groups.len() != 1 || decode.parallel_projections {
                return Err(Error::Unsupported("GLM wide prefill requires one admitted serial encoder group"));
            }
            let bank_group = &plan.groups[0];
            let mut consumed = 0;
            while consumed < tokens.len() {
                let mut count = next_tile(self.position, tokens.len() - consumed, maximum);
                // Admission happens before embeddings, recurrent writers or
                // attention reservations. Smaller powers of two keep the
                // same absolute 1024-token prefix-capture grid.
                let groups = loop {
                    match wide_groups(c, count, self.position + count, maximum_group_layers) {
                        Ok(groups) => break groups,
                        Err(_) if count > 1 => {
                            count = if count > 4 {
                                count / 2
                            } else {
                                count - 1
                            }
                        },
                        Err(error) => return Err(error),
                    }
                };
                let mut values = Vec::with_capacity(count * c.hidden * c.hc);
                for &token in &tokens[consumed..consumed + count] {
                    before_token()?;
                    let embedding = self.weights.row("token_embd.weight", token as usize)?;
                    if embedding.len() != c.hidden || embedding.iter().any(|value| !value.is_finite()) {
                        return Err(Error::Invalid("GLM wide embedding values"));
                    }
                    for _ in 0..c.hc {
                        values.extend_from_slice(&embedding);
                    }
                }
                let mut streams = DecodeGraph::upload(&values)?;
                drop(values);
                for group in groups {
                    let (graph, next) = objc2::rc::autoreleasepool(|_| -> Result<_> {
                        let _scope = super::super::profile::Scope::new("native.glm.prefill.wide_encode_cpu");
                        before_token()?;
                        let mut graph = new_encoder_graph(decode)?;
                        graph.set_scratch_limit(group.scratch)?;
                        let banks = self.weights.encoder_indexed_group(&bank_group.names, plan.budget)?;
                        if banks.registered_bytes() > bank_group.bytes {
                            return Err(Error::Invalid("GLM wide encoder inventory changed"));
                        }
                        let mut next = streams.clone();
                        for layer in group.start..group.end {
                            next = encode_layer(
                                &self.weights,
                                c,
                                decode,
                                layer,
                                &mut self.states[layer],
                                &mut graph,
                                &next,
                                &banks,
                                count,
                                use_mma,
                            )?;
                        }
                        graph.check_finite(&next)?;
                        // Graph resources retain mapped buffers and the source
                        // arena; bank wrappers may leave this CPU scope safely.
                        // scratch() allocates max(request,1 MiB) slabs, with
                        // each request capped at 8M F32 elements. A stream view
                        // therefore retains <=32 MiB, not its whole graph.
                        if graph
                            .scratch_backing_bytes()
                            .checked_add(group.external)
                            .is_none_or(|n| n > WIDE_OWNER_BYTES)
                            || next.buffer.length() > WIDE_CARRY_BYTES
                        {
                            return Err(Error::Invalid("GLM wide workspace inventory changed"));
                        }
                        before_token()?;
                        Ok((graph, next))
                    })?;
                    // While this graph was encoded, the preceding group could
                    // execute. Completion releases its unused arenas before
                    // submission, leaving at most two admitted graph owners.
                    if let Some(previous) = pending.take() {
                        previous.wait().inspect_err(|_| execution_failed = true)?;
                    }
                    // Completion may block past cancellation or a deadline.
                    // Recheck after the wait before submitting another writer.
                    before_token()?;
                    pending = Some(graph.submit(&[])?);
                    streams = next;
                    super::super::profile::record("native.glm.prefill.wide_groups", 0.0, group.end - group.start);
                }
                if let Some(previous) = pending.take() {
                    previous.wait().inspect_err(|_| execution_failed = true)?;
                }
                for state in &self.states {
                    if let LayerState::Kda(state) = state
                        && let Some(metal) = &state.metal
                    {
                        metal.check_completed()?;
                    }
                }
                self.position += count;
                consumed += count;
                super::super::profile::record("native.glm.prefill.wide_tokens", 0.0, count);
            }
            Ok(())
        })();
        if result.is_err() {
            let mut failed = execution_failed;
            failed |= pending.take().is_some_and(|previous| previous.wait().is_err());
            for state in &self.states {
                if let LayerState::Kda(state) = state
                    && let Some(metal) = &state.metal
                {
                    failed |= metal.check_completed().is_err();
                }
            }
            if failed {
                self.cleanup_failed = true;
            }
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

enum ControlMix {
    Rows(Vec<DeviceVector>),
    Batched(DeviceVector, usize),
}

fn controls_pre(
    control: &super::super::glm_metal::Control,
    graph: &mut DecodeGraph,
    residual: &DeviceVector,
    hidden: usize,
    streams: usize,
    tokens: usize,
) -> Result<(DeviceVector, ControlMix)> {
    let normalized = graph.scratch(tokens * hidden)?;
    if std::env::var_os("STARK_GLM53_CONTROL_PREFILL_BATCH").is_some_and(|value| value == "1") {
        let mix = control.pre_batch(graph, residual, &normalized, tokens)?;
        return Ok((normalized, ControlMix::Batched(mix, tokens)));
    }
    let mut mixes = Vec::with_capacity(tokens);
    for token in 0..tokens {
        mixes.push(control.pre_batch_row(
            graph,
            &residual.slice(token * hidden * streams, hidden * streams)?,
            &normalized.slice(token * hidden, hidden)?,
        )?);
    }
    Ok((normalized, ControlMix::Rows(mixes)))
}

fn controls_post(
    control: &super::super::glm_metal::Control,
    graph: &mut DecodeGraph,
    residual: &DeviceVector,
    output: &DeviceVector,
    mixes: &ControlMix,
    hidden: usize,
    streams: usize,
) -> Result<DeviceVector> {
    let next = graph.scratch(residual.len)?;
    let mixes = match mixes {
        ControlMix::Batched(mix, tokens) => {
            control.post_batch(graph, residual, output, mix, &next, *tokens)?;
            return Ok(next);
        },
        ControlMix::Rows(mixes) => mixes,
    };
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
    if std::env::var_os("STARK_GLM53_KDA_REGISTER_PREFILL").is_some_and(|value| value == "1")
        && metal.encode_prefill_register(
            graph,
            &projections[0],
            &projections[1],
            &projections[2],
            &forget,
            &gate,
            &projections[3],
            &recurrent,
            tokens,
        )?
    {
        return project(banks, graph, p, "attn_output.weight", &recurrent, tokens, use_mma);
    }
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
