// SPDX-License-Identifier: Apache-2.0
//! Request-scoped CPU encoding overlap. Future work remains unsubmitted until
//! the preceding pass and its exact caller-visible greedy token are validated.

use metal::MTLBuffer;

use super::*;
use crate::engine::hybrid::weight_metal::{DecodeGraph, DeviceVector, PendingDecodeGraph};

fn greedy_token(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.len() > u32::MAX as usize || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::Invalid("GLM greedy readout values"));
    }
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    Ok(best as u32)
}

fn greedy_index(
    values: &[f32],
    vocab: usize,
) -> Result<u32> {
    if vocab == 0
        || vocab > 8_000_000
        || values.len() != 1
        || !values[0].is_finite()
        || values[0] < 0.0
        || values[0].fract() != 0.0
        || values[0] as f64 >= vocab as f64
    {
        return Err(Error::Invalid("GLM GPU greedy index"));
    }
    Ok(values[0] as u32)
}

impl GlmModel {
    pub fn is_fenced(&self) -> bool {
        self.cleanup_failed
    }

    /// Ingest known prefix tokens and fence all state writers before return.
    /// The callback is checked before reserving/committing known future tokens.
    /// Queued mode relies on tracked same-queue hazards between two writers;
    /// the control mode validates each predecessor before its successor commit.
    pub fn ingest_prefix(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let overlap = std::env::var_os("STARK_GLM53_PREFILL_OVERLAP").is_some_and(|value| value == "1");
        let queued = std::env::var_os("STARK_GLM53_PREFILL_QUEUED").is_some_and(|value| value == "1");
        self.ingest_prefix_policy(tokens, before_token, overlap, queued)
    }

    fn ingest_prefix_policy(
        &mut self,
        tokens: &[u32],
        before_token: &mut dyn FnMut() -> Result<()>,
        overlap: bool,
        queued: bool,
    ) -> Result<()> {
        // Keep submitted ownership outside the fallible encoder/control
        // scope, so an earlier error cannot hide a failed completion in Drop.
        let mut pending: Option<PendingDecodeGraph> = None;
        let mut queued_pending = std::collections::VecDeque::<PendingDecodeGraph>::with_capacity(if queued {
            2
        } else {
            0
        });
        let result = (|| {
            if !overlap && !queued {
                for &token in tokens {
                    before_token()?;
                    self.ingest(token)?;
                }
                return Ok(());
            }
            if self.cleanup_failed {
                return Err(Error::Invalid("GLM context requires reload after cleanup failure"));
            }
            if tokens.is_empty() {
                return Ok(());
            }
            let enabled = |name| std::env::var_os(name).is_some_and(|value| value == "1");
            // GREEDY_OVERLAP already reserves two complete graph workspaces.
            // Both modes keep at most two live graphs: one pending plus one
            // encoded, or two committed. No third scratch allowance is added.
            if !self.supports_greedy_readout()
                || !enabled("STARK_GLM53_RESIDENT_BANKS")
                || !enabled("STARK_GLM53_WIDE_ENCODER")
                || enabled("STARK_GLM53_CONCURRENT_ENCODER")
                || enabled("STARK_GLM53_PARALLEL_PROJECTIONS")
            {
                return Err(Error::Unsupported("GLM prefill overlap requires serial resident wide-encoder overlap"));
            }
            if queued && !enabled("STARK_GLM53_UNRETAINED_GRAPH") {
                return Err(Error::Unsupported("GLM queued prefill requires the audited unretained graph"));
            }
            if tokens.iter().any(|&token| token as usize >= self.config.vocab)
                || self.position.checked_add(tokens.len()).is_none_or(|end| end > self.config.max_context)
            {
                return Err(Error::Invalid("GLM prefill token or context boundary"));
            }
            self.last_greedy = None;
            before_token()?;
            // Bootstrap lazy Metal/control/state construction through the
            // unchanged synchronous path. Subsequent passes can encode ahead.
            self.ingest(tokens[0])?;
            if self
                .decode
                .as_ref()
                .and_then(|decode| decode.bounded_plan.as_ref())
                .is_none_or(|plan| plan.groups.len() != 1)
            {
                return Err(Error::Unsupported("GLM prefill overlap requires one admitted encoder bank group"));
            }
            for &token in &tokens[1..] {
                if queued {
                    objc2::rc::autoreleasepool(|_| -> Result<()> {
                        if queued_pending.len() == 2 {
                            // Release the oldest owner before constructing a
                            // third graph. The second writer is already queued.
                            queued_pending.pop_front().unwrap().wait()?;
                        }
                        before_token()?;
                        let next = self.encode_pass(None)?;
                        self.stage_pass(&next, token)?;
                        next.graph.validate_queued_submission()?;
                        before_token()?;
                        queued_pending.push_back(next.graph.submit(&[])?);
                        crate::engine::hybrid::profile::record(
                            "native.glm.prefill.queued_pending",
                            0.0,
                            queued_pending.len(),
                        );
                        Ok(())
                    })?;
                    continue;
                }
                let submitted = objc2::rc::autoreleasepool(|_| {
                    before_token()?;
                    let next = self.encode_pass(None)?;
                    self.stage_pass(&next, token)?;
                    if let Some(previous) = pending.take() {
                        previous.wait()?;
                        self.check_greedy_completed()?;
                        // Cancellation during the preceding GPU pass stops
                        // this still-unsubmitted future state writer.
                        before_token()?;
                    }
                    next.graph.submit(&[])
                })?;
                pending = Some(submitted);
            }
            if let Some(previous) = pending.take() {
                previous.wait()?;
                self.check_greedy_completed()?;
            }
            if queued {
                while let Some(previous) = queued_pending.pop_front() {
                    previous.wait()?;
                }
                // Shared KDA status is GPU-writable by every queued token.
                // Read it only after the tail completion fence, never when
                // merely recycling the oldest completed graph owner.
                self.check_greedy_completed()?;
            }
            crate::engine::hybrid::profile::record(
                if queued {
                    "native.glm.prefill.queued_tokens"
                } else {
                    "native.glm.prefill.overlap_tokens"
                },
                0.0,
                tokens.len(),
            );
            Ok(())
        })();
        // Closure locals, especially any unsubmitted future graph, are gone
        // before explicitly draining the preceding submitted state writer.
        if result.is_err() {
            if queued {
                self.drain_prefill_queued_error(queued_pending);
            } else {
                self.drain_prefill_error(pending.take());
            }
        }
        result
    }

    fn drain_prefill_queued_error(
        &mut self,
        mut pending: std::collections::VecDeque<PendingDecodeGraph>,
    ) {
        let mut failed = false;
        while let Some(previous) = pending.pop_front() {
            // Drain every queued writer even when an earlier completion
            // failed. No shared KDA status may be read during this loop.
            failed |= previous.wait().is_err();
        }
        failed |= self.check_greedy_completed().is_err();
        if failed {
            self.cleanup_failed = true;
        }
        self.reset();
    }

    fn drain_prefill_error(
        &mut self,
        pending: Option<PendingDecodeGraph>,
    ) {
        if let Some(previous) = pending
            && previous.wait().and_then(|_| self.check_greedy_completed()).is_err()
        {
            // Preserve a failure found only during cancellation/error
            // cleanup. Reset must never make this model reusable.
            self.cleanup_failed = true;
        }
        self.reset();
    }

    pub fn supports_greedy_readout(&self) -> bool {
        !self.cleanup_failed && self.decode_graph_enabled
            && std::env::var_os("STARK_GLM53_GREEDY_OVERLAP").is_some_and(|value| value == "1")
            && std::env::var_os("STARK_GLM53_ENCODER_BANKS").is_some_and(|value| value == "1")
            // Diagnostic checkpoint submission cannot encode a future pass.
            && std::env::var_os("STARK_QWEN38_PHASE_TIMING").is_none_or(|value| value != "1")
            && std::env::var_os("STARK_QWEN38_ENCODER_TIMING").is_none_or(|value| value != "1")
    }

    pub fn step_greedy(
        &mut self,
        token: u32,
    ) -> Result<u32> {
        if !self.supports_greedy_readout() {
            return Err(Error::Unsupported("GLM greedy overlap unavailable"));
        }
        let result = self.step(token).and_then(|logits| greedy_token(&logits));
        match result {
            Ok(token) => {
                self.last_greedy = Some((token, self.position));
                Ok(token)
            },
            Err(error) => {
                self.reset();
                Err(error)
            },
        }
    }

    pub fn greedy_decode(
        &mut self,
        first: u32,
        remaining: usize,
    ) -> Result<GreedyDecode<'_>> {
        // Capture the readout policy once for this serialized request. An
        // encoded output's shape never depends on later environment reads.
        let gpu_greedy = std::env::var_os("STARK_GLM53_GPU_GREEDY").is_some_and(|value| value == "1");
        if !self.supports_greedy_readout()
            || self.last_greedy != Some((first, self.position))
            || self.position.checked_add(remaining).is_none_or(|end| end > self.config.max_context)
            || (gpu_greedy && self.config.vocab > 8_000_000)
        {
            self.reset();
            return Err(Error::Invalid("GLM greedy continuation boundary"));
        }
        self.last_greedy = None;
        Ok(GreedyDecode {
            model: self,
            first,
            remaining,
            produced: 0,
            current: None,
            ahead: None,
            failed: false,
            gpu_greedy,
        })
    }

    fn encode_pass(
        &mut self,
        readout: Option<bool>,
    ) -> Result<EncodedGreedy> {
        let _encode = crate::engine::hybrid::profile::Scope::new(if readout.is_some() {
            "native.glm.greedy.encode_cpu"
        } else {
            "native.glm.prefill.encode_cpu"
        });
        let c = &self.config;
        if self.position >= c.max_context {
            return Err(Error::Invalid("GLM greedy context boundary"));
        }
        let decode = self.decode.as_mut().ok_or(Error::Invalid("GLM greedy decode state"))?;
        if !decode.encoder_banks {
            return Err(Error::Invalid("GLM greedy bank mode"));
        }
        let plan = decode.bounded_plan.clone().ok_or(Error::Invalid("GLM greedy bank plan"))?;
        // Each encoded command gets its own host staging allocation. It is
        // initialized after its predecessor completes, before its own commit.
        let embedding = DecodeGraph::zeros(c.hidden * c.hc)?;
        let mut streams = embedding.clone();
        let mut graph = new_encoder_graph(decode)?;
        for group in &plan.groups {
            let banks = self.weights.encoder_indexed_group(&group.names, plan.budget)?;
            if banks.registered_bytes() > group.bytes {
                return Err(Error::Invalid("GLM greedy bank accounting changed"));
            }
            for layer in group.start..group.end {
                (graph, streams) = encode_metal_layer(
                    &self.weights,
                    c,
                    decode,
                    layer,
                    &mut self.states[layer],
                    graph,
                    streams,
                    Some(&banks),
                )?;
            }
            drop(banks);
            if group.end < c.layer_types.len() {
                graph.resource_boundary()?;
            }
        }
        let output = if let Some(gpu_greedy) = readout {
            let mean = super::super::glm_metal::stream_mean(&mut graph, &streams, c.hidden, c.hc)?;
            let norm = graph.rms_norm(&mean, &decode.output_norm, c.hidden, c.rms_eps)?;
            let logits = self.weights.encode_matvec(&mut graph, "output.weight", &norm)?;
            if gpu_greedy {
                graph.greedy_argmax(&logits)?
            } else {
                logits
            }
        } else {
            graph.check_finite(&streams)?;
            streams
        };
        // Attention metadata reserves this encoded position. Discarding an
        // unsubmitted future therefore requires a drained model reset.
        self.position += 1;
        Ok(EncodedGreedy {
            graph,
            embedding,
            output,
        })
    }

    fn stage_pass(
        &self,
        next: &EncodedGreedy,
        token: u32,
    ) -> Result<()> {
        if token as usize >= self.config.vocab {
            return Err(Error::Invalid("GLM greedy token ID"));
        }
        let row = self.weights.row("token_embd.weight", token as usize)?;
        if row.len() != self.config.hidden || row.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("GLM greedy embedding values"));
        }
        let values = row.repeat(self.config.hc);
        if next.embedding.offset != 0
            || next.embedding.len != values.len()
            || next.embedding.buffer.length() < values.len() * 4
        {
            return Err(Error::Invalid("GLM greedy embedding span"));
        }
        // This allocation belongs exclusively to unsubmitted `next`; no
        // current command references it. All host writes precede commit.
        unsafe {
            next.embedding
                .buffer
                .contents()
                .as_ptr()
                .cast::<f32>()
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        Ok(())
    }

    fn check_greedy_completed(&self) -> Result<()> {
        for state in &self.states {
            if let LayerState::Kda(state) = state
                && let Some(metal) = &state.metal
            {
                metal.check_completed()?;
            }
        }
        Ok(())
    }
}

struct EncodedGreedy {
    graph: DecodeGraph,
    embedding: DeviceVector,
    output: DeviceVector,
}

pub struct GreedyDecode<'a> {
    model: &'a mut GlmModel,
    first: u32,
    remaining: usize,
    produced: usize,
    current: Option<PendingDecodeGraph>,
    ahead: Option<EncodedGreedy>,
    failed: bool,
    gpu_greedy: bool,
}

impl GreedyDecode<'_> {
    fn submit(
        &mut self,
        next: EncodedGreedy,
        token: u32,
    ) -> Result<()> {
        if self.current.is_some() {
            return Err(Error::Invalid("GLM greedy pending command"));
        }
        self.model.stage_pass(&next, token)?;
        self.current = Some(next.graph.submit(&[next.output])?);
        Ok(())
    }

    fn encode_ahead(&mut self) -> Result<()> {
        if self.produced + 1 < self.remaining {
            self.ahead = Some(self.model.encode_pass(Some(self.gpu_greedy))?);
        }
        Ok(())
    }

    fn finish_current(&mut self) -> Result<u32> {
        let current = self.current.take().ok_or(Error::Invalid("GLM greedy pending command missing"))?;
        let mut values = current.wait()?;
        self.model.check_greedy_completed()?;
        let logits = values.pop().ok_or(Error::Invalid("GLM greedy readout missing"))?;
        if self.gpu_greedy {
            return greedy_index(&logits, self.model.config.vocab);
        }
        if logits.len() != self.model.config.vocab {
            return Err(Error::Invalid("GLM greedy vocabulary shape"));
        }
        greedy_token(&logits)
    }

    fn next_draw(&mut self) -> Result<u32> {
        if self.current.is_none() {
            let next = self.model.encode_pass(Some(self.gpu_greedy))?;
            self.submit(next, self.first)?;
            self.encode_ahead()?;
        }
        let token = self.finish_current()?;
        self.produced += 1;
        if self.produced < self.remaining {
            let next = self.ahead.take().ok_or(Error::Invalid("GLM greedy future command missing"))?;
            self.submit(next, token)?;
            self.encode_ahead()?;
        }
        Ok(token)
    }

    fn drain(&mut self) {
        // Discard future encoding before waiting, then reset only after every
        // submitted state writer has completed. No guessed token executes.
        let discarded = self.ahead.take().is_some();
        let pending = self.current.is_some();
        if pending && self.finish_current().is_err() {
            self.failed = true;
            // The collector checks this after Drop, including EOS/callback/
            // cancellation exits that cannot return the cleanup error directly.
            self.model.cleanup_failed = true;
        }
        if discarded || pending || self.failed {
            self.model.reset();
        }
    }
}

impl Iterator for GreedyDecode<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.produced >= self.remaining {
            return None;
        }
        let result = objc2::rc::autoreleasepool(|_| self.next_draw());
        if result.is_err() {
            self.failed = true;
            self.drain();
        }
        Some(result)
    }
}

impl Drop for GreedyDecode<'_> {
    fn drop(&mut self) {
        self.drain();
    }
}

#[cfg(test)]
#[path = "glm_prefill_tests.rs"]
mod prefill_tests;

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn greedy_readout_keeps_first_index_ties_and_rejects_nonfinite() {
        assert_eq!(greedy_token(&[1.0, 2.0, 2.0]).unwrap(), 1);
        assert_eq!(greedy_token(&[-0.0, 0.0]).unwrap(), 0);
        assert_eq!(greedy_token(&[0.0, -0.0]).unwrap(), 0);
        assert!(greedy_token(&[]).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(greedy_token(&[0.0, value]).is_err());
            assert!(greedy_index(&[value], 3).is_err());
        }
        assert_eq!(greedy_index(&[2.0], 3).unwrap(), 2);
        for values in [vec![], vec![0.0, 1.0], vec![-1.0], vec![0.5], vec![3.0]] {
            assert!(greedy_index(&values, 3).is_err());
        }
    }

    #[stark_test]
    #[ignore]
    fn glm_greedy_overlap_matches_synchronous_and_drains_early_exits() {
        let path = std::env::var_os("STARK_GLM_REFERENCE_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .expect("GLM private fixture path required");
        let config: Value = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
        let gpu_greedy = std::env::var_os("STARK_GLM53_GPU_GREEDY").is_some_and(|value| value == "1");
        let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(path.join("tokens.json")).unwrap()).unwrap();
        let new_model = || {
            // Complete normal GGUF/source-config admission first. Only this
            // test then extends synthetic capacity before lazy MetalDecode
            // construction; both paths get matching attention geometry and
            // residency allowances. Files/source-oracle checks stay unchanged.
            let mut model = GlmModel::new(Weights::open(&path).unwrap(), &config).unwrap();
            assert!(model.decode.is_none());
            if gpu_greedy {
                model.config.max_context = 128;
            }
            model
        };
        let mut model = new_model();
        assert!(model.supports_greedy_readout(), "GLM overlap flags required");
        let prefix = &tokens[..3];
        let continuation = if gpu_greedy {
            let target_position = 67usize;
            assert!(target_position <= model.config.max_context, "synthetic greedy capacity is too small");
            target_position - prefix.len()
        } else {
            7
        };
        let pool_lengths = |model: &GlmModel| {
            model
                .decode
                .as_ref()
                .unwrap()
                .attention
                .iter()
                .flatten()
                .map(|attention| attention.test_pool_keys().len)
                .collect::<Vec<_>>()
        };
        let mut expected = Vec::new();
        for &token in &prefix[..2] {
            model.ingest(token).unwrap();
        }
        let initial_pool_lengths = pool_lengths(&model);
        assert!(!initial_pool_lengths.is_empty(), "greedy fixture requires an MLA layer");
        let mut token = greedy_token(&model.step(prefix[2]).unwrap()).unwrap();
        expected.push(token);
        for _ in 0..continuation {
            token = greedy_token(&model.step(token).unwrap()).unwrap();
            expected.push(token);
        }
        let expected_position = model.position;
        if gpu_greedy {
            assert!(expected_position >= 67);
            assert!(pool_lengths(&model).iter().zip(&initial_pool_lengths).all(|(after, before)| after > before));
        }
        // A fresh cache must exercise growth on the overlapped path too;
        // reset intentionally preserves an already-grown cache allocation.
        drop(model);
        let mut model = new_model();
        for &token in &prefix[..2] {
            model.ingest(token).unwrap();
        }
        assert_eq!(pool_lengths(&model), initial_pool_lengths);
        let first = model.step_greedy(prefix[2]).unwrap();
        let mut actual = vec![first];
        let stream = model.greedy_decode(first, continuation).unwrap();
        assert_eq!(stream.gpu_greedy, gpu_greedy, "greedy request did not capture its readout policy");
        actual.extend(stream.collect::<Result<Vec<_>>>().unwrap());
        assert!(actual == expected, "GLM overlap token agreement failed");
        assert_eq!(model.position, expected_position);
        if gpu_greedy {
            assert!(pool_lengths(&model).iter().zip(&initial_pool_lengths).all(|(after, before)| after > before));
        }

        model.reset();
        let fresh = model.step(prefix[0]).unwrap();
        for budget in [0, 1, 4] {
            model.reset();
            let first = model.step_greedy(prefix[0]).unwrap();
            let mut stream = model.greedy_decode(first, budget).unwrap();
            if budget > 0 {
                assert!(stream.next().unwrap().is_ok());
            }
            drop(stream);
            if budget > 1 {
                assert_eq!(model.position, 0, "GLM early exit did not reset reserved metadata");
            }
            model.reset();
            assert!(model.step(prefix[0]).unwrap() == fresh, "GLM drained replay changed");
        }
        model.reset();
        let first = model.step_greedy(prefix[0]).unwrap();
        assert!(model.greedy_decode(first, model.config.max_context).is_err());
        assert_eq!(model.position, 0);
        let first = model.step_greedy(prefix[0]).unwrap();
        assert!(model.greedy_decode((first + 1) % model.config.vocab as u32, 1).is_err());
        assert_eq!(model.position, 0);
        assert!(model.step(prefix[0]).unwrap() == fresh);

        // A failed submitted pass discovered only during Drop must remain
        // visible to the outer collector and require reloading this model.
        let mut graph = DecodeGraph::new().unwrap();
        let huge = DecodeGraph::upload(&[f32::MAX]).unwrap();
        let overflow = graph.add(&huge, &huge).unwrap();
        let pending = graph.submit(&[overflow]).unwrap();
        drop(GreedyDecode {
            model: &mut model,
            first: 0,
            remaining: 1,
            produced: 0,
            current: Some(pending),
            ahead: None,
            failed: false,
            gpu_greedy: false,
        });
        assert!(model.is_fenced(), "GLM drain failure was hidden");
        model.reset();
        assert!(model.is_fenced(), "GLM reset cleared cleanup failure");
        assert!(model.step(prefix[0]).is_err());
    }
}
