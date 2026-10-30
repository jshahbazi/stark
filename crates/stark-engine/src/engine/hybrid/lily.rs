// SPDX-License-Identifier: Apache-2.0
//! In-process Qwen execution through Stark's pinned Lily core. Activations use
//! BF16, attention K/V uses Q8, and logits, reductions and GDN state use F32.
//! Every request has fresh logical state. Pending prompt tokens exist only
//! until its next readout; no prefix/session persistence or speculative head.

use std::path::Path;

use stark_lily_core::{
    engine::{DecodeStateApi, Draw, ScratchApi, VisionMode},
    kernels::{attention::KvFormat, sample::SamplingParams},
    metal::{EncodedPass, MetalContext, PendingPass, stage_profile},
    qwen4exp::{DecodeState, NgramStorage, Qwen4ExpModel, Scratch},
};

use super::{Error, Result, profile};

pub const MAX_CONTEXT: usize = 16_384;

/// Only aggregate static phase categories leave the native counter module.
fn record_gpu_profile() {
    if !profile::enabled() {
        return;
    }
    for pass in stage_profile::take() {
        profile::record("qwen.mixed.gpu.command", pass.command_gpu_seconds, 0);
        profile::record("qwen.mixed.gpu.measured", pass.measured_seconds, 0);
        profile::record("qwen.mixed.profile_resolve", pass.resolve_seconds, 0);
        for sample in pass.samples {
            let name = match sample.phase {
                stage_profile::Phase::Embedding => "qwen.mixed.gpu.embedding",
                stage_profile::Phase::Ple => "qwen.mixed.gpu.ple",
                stage_profile::Phase::Hc => "qwen.mixed.gpu.hc",
                stage_profile::Phase::Gdn => "qwen.mixed.gpu.gdn",
                stage_profile::Phase::Qsa => "qwen.mixed.gpu.qsa",
                stage_profile::Phase::Moe => "qwen.mixed.gpu.moe",
                stage_profile::Phase::Head => "qwen.mixed.gpu.head",
                stage_profile::Phase::Draw => "qwen.mixed.gpu.draw",
            };
            profile::record(name, sample.gpu_seconds, 0);
        }
    }
}

fn admitted_position(
    committed: usize,
    pending: usize,
    token: u32,
    vocab: usize,
    limit: usize,
) -> Result<usize> {
    if token as usize >= vocab {
        return Err(Error::Invalid("mixed precision token ID"));
    }
    committed
        .checked_add(pending)
        .and_then(|position| position.checked_add(1))
        .filter(|&position| position <= limit)
        .ok_or(Error::Invalid("mixed precision context boundary"))
}

fn admitted_decode_end(
    position: usize,
    remaining: usize,
    limit: usize,
) -> Result<usize> {
    position
        .checked_add(remaining)
        .filter(|&end| end <= limit)
        .ok_or(Error::Invalid("mixed precision greedy context boundary"))
}

fn admitted_sampled_start(
    position: usize,
    pending: usize,
    readout_position: Option<usize>,
    first: u32,
    vocabulary: usize,
) -> Result<()> {
    if position == 0 || readout_position != Some(position) || pending != 0 || first as usize >= vocabulary {
        return Err(Error::Invalid("mixed precision sampled start"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct GreedyReadout {
    token: u32,
    slot: usize,
    position: usize,
}

pub struct LilyModel {
    model: Qwen4ExpModel,
    state: DecodeState,
    scratch: Scratch,
    pending: Vec<u32>,
    vocab_size: usize,
    max_context: usize,
    usable: bool,
    last_greedy: Option<GreedyReadout>,
    last_full_readout: Option<usize>,
    // Context ownership outlives every model/state/scratch allocation on drop.
    context: MetalContext,
}

impl LilyModel {
    pub fn load(path: &Path) -> Result<Self> {
        let context =
            MetalContext::new().map_err(|_| Error::Unsupported("mixed precision Metal context unavailable"))?;
        let mut model = Qwen4ExpModel::load_with(
            &context,
            path,
            NgramStorage::Paged,
            false,
            VisionMode::Off,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .map_err(|_| Error::Unsupported("mixed precision checkpoint admission failed"))?;
        let vocab_size = model.config.vocab_size;
        let max_context = model.config.max_position_embeddings.min(MAX_CONTEXT);
        if vocab_size == 0 || vocab_size > 1_000_000 || max_context == 0 {
            return Err(Error::Invalid("mixed precision model geometry"));
        }
        // Match the pinned upstream fast path explicitly; do not depend on
        // an upstream default when qualifying this mixed-precision policy.
        model.set_kv_format(KvFormat::Q8);
        let state = model
            .new_state(&context, max_context)
            .map_err(|_| Error::Invalid("mixed precision state allocation failed"))?;
        let scratch = model
            .new_scratch_with_capacity(&context, max_context)
            .map_err(|_| Error::Invalid("mixed precision scratch allocation failed"))?;
        scratch.begin_request();
        Ok(Self {
            model,
            state,
            scratch,
            pending: Vec::new(),
            vocab_size,
            max_context,
            usable: true,
            last_greedy: None,
            last_full_readout: None,
            context,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Explicit diagnostic mode for future passes; no payload is captured.
    pub fn set_stage_profiling(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        if self.is_fenced() {
            return Err(Error::Invalid("mixed precision context requires reload"));
        }
        let _ = stage_profile::take();
        self.context.set_stage_profiling(enabled);
        Ok(())
    }

    /// Includes failures discovered while draining a greedy iterator on drop.
    /// A fenced context requires a model reload, even after request reset.
    pub fn is_fenced(&self) -> bool {
        !self.usable || self.context.fault().is_some()
    }

    /// Includes admitted prompt tokens whose batched prefill has not run yet.
    pub fn position(&self) -> usize {
        if self.usable {
            self.state.pos + self.pending.len()
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        let _scope = profile::Scope::new("qwen.mixed.reset");
        self.pending.clear();
        self.last_greedy = None;
        self.last_full_readout = None;
        // Core waits drain submitted GPU work even on execution failure.
        // A faulted context stays unusable until the worker reloads the model.
        self.usable &= self.state.reset().is_ok();
        self.scratch.begin_request();
        self.scratch.next_token.zero_fill();
    }

    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        self.last_greedy = None;
        self.last_full_readout = None;
        let result = self.check_token(token).map(|_| self.pending.push(token));
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Full F32 readout for Stark's existing scorer and non-greedy sampler.
    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.last_greedy = None;
        self.last_full_readout = None;
        let result = self.forward(token).and_then(|_| {
            let logits = {
                let _scope = profile::Scope::bytes("qwen.mixed.logit_readback", self.vocab_size * 4);
                self.scratch.logits().to_f32().map_err(|_| Error::Invalid("mixed precision readout failed"))?
            };
            let _scope = profile::Scope::new("qwen.mixed.logit_validation");
            if logits.len() != self.vocab_size || !super::finite_guard::all_finite(&logits) {
                return Err(Error::Invalid("mixed precision readout values"));
            }
            self.last_full_readout = Some(self.state.pos);
            Ok(logits)
        });
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Same non-speculative forward pass, returning only its greedy token.
    /// Outer worker EOS handling remains authoritative.
    pub fn step_greedy(
        &mut self,
        token: u32,
    ) -> Result<u32> {
        self.last_greedy = None;
        self.last_full_readout = None;
        let result = self.forward(token).and_then(|slot| {
            let values = self
                .scratch
                .next_token
                .view(slot, &[1])
                .and_then(|token| token.to_u32())
                .map_err(|_| Error::Invalid("mixed precision greedy readout failed"))?;
            let token = *values.first().ok_or(Error::Invalid("mixed precision greedy readout missing"))?;
            if token as usize >= self.vocab_size {
                return Err(Error::Invalid("mixed precision greedy token ID"));
            }
            self.last_greedy = Some(GreedyReadout {
                token,
                slot,
                position: self.state.pos,
            });
            Ok(token)
        });
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Subsequent greedy draws after a successful `step_greedy`. This borrows
    /// the request until dropped; no pass is submitted before its first pull.
    /// The caller supplies the collector's exact remaining decode count and
    /// must inspect `is_fenced` after dropping the iterator, including on an
    /// EOS, cancellation or callback exit. Full-logit and sampled steps keep
    /// their synchronous caller-selected input behavior.
    pub fn greedy_decode(
        &mut self,
        first: u32,
        remaining_decode_count: usize,
    ) -> Result<GreedyDecode<'_>> {
        if !self.usable {
            return Err(Error::Invalid("mixed precision context requires reload"));
        }
        let start = match self.last_greedy {
            Some(start)
                if start.token == first
                    && start.position == self.state.pos
                    && start.slot < 2
                    && self.pending.is_empty() =>
            {
                start
            },
            _ => {
                self.reset();
                return Err(Error::Invalid("mixed precision greedy start"));
            },
        };
        if let Err(error) = admitted_decode_end(self.state.pos, remaining_decode_count, self.max_context) {
            self.reset();
            return Err(error);
        }
        self.last_greedy = None;
        self.last_full_readout = None;
        Ok(GreedyDecode {
            model: &self.model,
            context: &self.context,
            state: &mut self.state,
            scratch: &self.scratch,
            usable: &mut self.usable,
            vocab_size: self.vocab_size,
            max_context: self.max_context,
            start,
            remaining: remaining_decode_count,
            produced: 0,
            stopped: false,
            current: None,
            ahead: None,
        })
    }

    /// Continue from a caller-selected token after a successful full readout.
    /// The callback owns the existing sampler and RNG; it runs once per
    /// returned draw, after verified full-logit readback. Dropping the stream
    /// validates submitted work without calling the sampler or advancing RNG.
    pub fn sampled_decode<'a, F>(
        &'a mut self,
        first: u32,
        remaining: usize,
        sample: F,
    ) -> Result<SampledDecode<'a, F>>
    where
        F: FnMut(&[f32]) -> Result<u32> + 'a,
    {
        if self.is_fenced() {
            return Err(Error::Invalid("mixed precision context requires reload"));
        }
        let admission =
            admitted_sampled_start(self.state.pos, self.pending.len(), self.last_full_readout, first, self.vocab_size)
                .and_then(|_| admitted_decode_end(self.state.pos, remaining, self.max_context));
        if let Err(error) = admission {
            self.reset();
            return Err(error);
        }
        self.last_greedy = None;
        self.last_full_readout = None;
        Ok(SampledDecode {
            model: &self.model,
            context: &self.context,
            state: &mut self.state,
            scratch: &self.scratch,
            usable: &mut self.usable,
            vocab_size: self.vocab_size,
            max_context: self.max_context,
            first,
            remaining,
            produced: 0,
            stopped: false,
            current: None,
            ahead: None,
            sample,
        })
    }

    fn check_token(
        &self,
        token: u32,
    ) -> Result<usize> {
        if !self.usable {
            return Err(Error::Invalid("mixed precision context requires reload"));
        }
        admitted_position(self.state.pos, self.pending.len(), token, self.vocab_size, self.max_context)
    }

    /// Returns the GPU sampler slot written by this synchronous forward pass.
    fn forward(
        &mut self,
        token: u32,
    ) -> Result<usize> {
        let expected_position = self.check_token(token)?;
        let greedy = SamplingParams::greedy();
        let draw = Draw {
            params: &greedy,
            step: self.state.pos,
        };
        if !self.pending.is_empty() || self.state.pos == 0 {
            let _scope = profile::Scope::new("qwen.mixed.prefill");
            self.pending.push(token);
            self.usable = false;
            self.model
                .prefill(&self.context, &mut self.state, &mut self.scratch, &self.pending, Some(draw))
                .map_err(|_| Error::Invalid("mixed precision prefill failed"))?;
            self.usable = true;
            self.pending.clear();
            if self.state.pos != expected_position {
                return Err(Error::Invalid("mixed precision prefill position"));
            }
            // Prefill itself advances state/history and writes slot zero.
            Ok(0)
        } else {
            let _scope = profile::Scope::new("qwen.mixed.decode");
            self.usable = false;
            self.state
                .ensure_capacity(&self.context, expected_position)
                .map_err(|_| Error::Invalid("mixed precision context growth failed"))?;
            // The caller may have sampled a different token from our last
            // greedy draw. Always overwrite the input before staging/encoding.
            {
                let _scope = profile::Scope::new("qwen.mixed.input_stage");
                self.scratch
                    .next_token
                    .view(0, &[1])
                    .and_then(|slot| slot.write_bytes(&token.to_le_bytes()))
                    .map_err(|_| Error::Invalid("mixed precision input staging failed"))?;
            }
            {
                let _scope = profile::Scope::new("qwen.mixed.ngram_stage");
                self.model
                    .prepare_step_inputs(&mut self.state, &self.scratch, token)
                    .map_err(|_| Error::Invalid("mixed precision ngram staging failed"))?;
            }
            let encoded = {
                let _scope = profile::Scope::new("qwen.mixed.encode");
                self.model
                    .encode_decode_step(&self.context, &self.state, &self.scratch, 0, 1, draw, None)
                    .map_err(|_| Error::Invalid("mixed precision decode failed"))?
            };
            let pending = {
                let _scope = profile::Scope::new("qwen.mixed.submit");
                encoded.commit().map_err(|_| Error::Invalid("mixed precision decode failed"))?
            };
            {
                let _scope = profile::Scope::new("qwen.mixed.wait");
                pending.wait().map_err(|_| Error::Invalid("mixed precision decode failed"))?;
            }
            record_gpu_profile();
            self.state.advance(1);
            self.usable = true;
            if self.state.pos != expected_position {
                return Err(Error::Invalid("mixed precision decode position"));
            }
            Ok(1)
        }
    }
}

struct Submitted<'a> {
    pass: PendingPass<'a>,
    output_slot: usize,
}

struct EncodedNext<'a> {
    pass: EncodedPass<'a>,
    input_slot: usize,
    output_slot: usize,
    position: usize,
}

/// Request-scoped greedy decode with CPU encoding overlapped with GPU work.
/// At most one pass is submitted. The future pass remains unsubmitted until
/// its input ID is checked, so the finite-readout sentinel cannot reach the
/// embedding gather. All references point into the borrowed model, not into
/// this iterator; no self-referential command-buffer ownership is needed.
pub struct GreedyDecode<'a> {
    model: &'a Qwen4ExpModel,
    context: &'a MetalContext,
    state: &'a mut DecodeState,
    scratch: &'a Scratch,
    usable: &'a mut bool,
    vocab_size: usize,
    max_context: usize,
    start: GreedyReadout,
    remaining: usize,
    produced: usize,
    stopped: bool,
    current: Option<Submitted<'a>>,
    ahead: Option<EncodedNext<'a>>,
}

impl<'a> GreedyDecode<'a> {
    fn encode(
        &self,
        input_slot: usize,
    ) -> Result<EncodedNext<'a>> {
        let _scope = profile::Scope::new("qwen.mixed.encode");
        if input_slot >= 2 || self.state.pos >= self.max_context {
            return Err(Error::Invalid("mixed precision greedy encoding boundary"));
        }
        let greedy = SamplingParams::greedy();
        let position = self.state.pos;
        let output_slot = 1 - input_slot;
        let pass = self
            .model
            .encode_decode_step(
                self.context,
                self.state,
                self.scratch,
                input_slot,
                output_slot,
                Draw {
                    params: &greedy,
                    step: position,
                },
                None,
            )
            .map_err(|_| Error::Invalid("mixed precision greedy encoding failed"))?;
        Ok(EncodedNext {
            pass,
            input_slot,
            output_slot,
            position,
        })
    }

    fn submit(
        &mut self,
        next: EncodedNext<'a>,
        token: u32,
    ) -> Result<()> {
        if self.current.is_some()
            || next.position != self.state.pos
            || token as usize >= self.vocab_size
            || self.state.pos >= self.max_context
        {
            return Err(Error::Invalid("mixed precision greedy submission boundary"));
        }
        // The preceding pass has finished. It is now safe to replace the PLE
        // staging rows; this history advances exactly once per consumed ID.
        {
            let _scope = profile::Scope::new("qwen.mixed.ngram_stage");
            self.model
                .prepare_step_inputs(self.state, self.scratch, token)
                .map_err(|_| Error::Invalid("mixed precision greedy staging failed"))?;
        }
        let output_slot = next.output_slot;
        let pass = {
            let _scope = profile::Scope::new("qwen.mixed.submit");
            next.pass.commit().map_err(|_| Error::Invalid("mixed precision greedy submission failed"))?
        };
        self.state.advance(1);
        self.current = Some(Submitted {
            pass,
            output_slot,
        });
        Ok(())
    }

    fn encode_ahead(&mut self) -> Result<()> {
        // The submitted pass accounts for the next draw. Only encode another
        // one if the collector will request a draw beyond it.
        if self.produced + 1 < self.remaining {
            let input_slot =
                self.current.as_ref().ok_or(Error::Invalid("mixed precision greedy submission missing"))?.output_slot;
            self.ahead = Some(self.encode(input_slot)?);
        }
        Ok(())
    }

    fn finish_current(&mut self) -> Result<(u32, usize)> {
        let current = self.current.take().ok_or(Error::Invalid("mixed precision greedy submission missing"))?;
        // On error PendingPass also drains submitted work before returning.
        {
            let _scope = profile::Scope::new("qwen.mixed.wait");
            current.pass.wait_polled().map_err(|_| Error::Invalid("mixed precision greedy execution failed"))?;
        }
        record_gpu_profile();
        let values = {
            let _scope = profile::Scope::bytes("qwen.mixed.greedy_readback", 4);
            self.scratch
                .next_token
                .view(current.output_slot, &[1])
                .and_then(|slot| slot.to_u32())
                .map_err(|_| Error::Invalid("mixed precision greedy readout failed"))?
        };
        let token = *values.first().ok_or(Error::Invalid("mixed precision greedy readout missing"))?;
        if token as usize >= self.vocab_size {
            return Err(Error::Invalid("mixed precision greedy token ID"));
        }
        Ok((token, current.output_slot))
    }

    fn next_draw(&mut self) -> Result<u32> {
        if self.current.is_none() {
            let next = self.encode(self.start.slot)?;
            self.submit(next, self.start.token)?;
            self.encode_ahead()?;
        }
        let (token, output_slot) = self.finish_current()?;
        self.produced += 1;
        if self.produced < self.remaining {
            let next = self.ahead.take().ok_or(Error::Invalid("mixed precision greedy encoding missing"))?;
            if next.input_slot != output_slot {
                return Err(Error::Invalid("mixed precision greedy input slot"));
            }
            // Validation precedes submission. Encoding alone cannot execute
            // the future embedding gather on a poisoned or out-of-range ID.
            self.submit(next, token)?;
            self.encode_ahead()?;
        }
        Ok(token)
    }

    fn drain(&mut self) {
        // Unsubmitted work may reference the current output slot, whose ID
        // is not known yet. Discard it before awaiting the submitted pass.
        self.ahead = None;
        if self.current.is_some() && self.finish_current().is_err() {
            *self.usable = false;
        }
    }
}

impl Iterator for GreedyDecode<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped || self.produced >= self.remaining {
            return None;
        }
        let result = self.next_draw();
        if result.is_err() {
            self.stopped = true;
            *self.usable = false;
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

enum SampledFailure {
    Forward(Error),
    Sampler(Error),
}

impl From<Error> for SampledFailure {
    fn from(error: Error) -> Self {
        Self::Forward(error)
    }
}

/// Sampled decode with safe CPU encoding overlap. Fixed input/output slots
/// are reused only after the preceding GPU pass completed. The unsubmitted
/// future never reads a guessed token; its actual sampled ID is staged first.
pub struct SampledDecode<'a, F>
where
    F: FnMut(&[f32]) -> Result<u32>,
{
    model: &'a Qwen4ExpModel,
    context: &'a MetalContext,
    state: &'a mut DecodeState,
    scratch: &'a Scratch,
    usable: &'a mut bool,
    vocab_size: usize,
    max_context: usize,
    first: u32,
    remaining: usize,
    produced: usize,
    stopped: bool,
    current: Option<PendingPass<'a>>,
    ahead: Option<EncodedNext<'a>>,
    sample: F,
}

impl<'a, F> SampledDecode<'a, F>
where
    F: FnMut(&[f32]) -> Result<u32>,
{
    fn encode(&self) -> Result<EncodedNext<'a>> {
        let _scope = profile::Scope::new("qwen.mixed.encode");
        if self.state.pos >= self.max_context {
            return Err(Error::Invalid("mixed precision sampled encoding boundary"));
        }
        let greedy = SamplingParams::greedy();
        let position = self.state.pos;
        let pass = self
            .model
            .encode_decode_step(
                self.context,
                self.state,
                self.scratch,
                0,
                1,
                Draw {
                    params: &greedy,
                    step: position,
                },
                None,
            )
            .map_err(|_| Error::Invalid("mixed precision sampled encoding failed"))?;
        Ok(EncodedNext {
            pass,
            input_slot: 0,
            output_slot: 1,
            position,
        })
    }

    fn submit(
        &mut self,
        next: EncodedNext<'a>,
        token: u32,
    ) -> Result<()> {
        if self.current.is_some()
            || next.position != self.state.pos
            || next.input_slot != 0
            || next.output_slot != 1
            || token as usize >= self.vocab_size
            || self.state.pos >= self.max_context
        {
            return Err(Error::Invalid("mixed precision sampled submission boundary"));
        }
        // No submitted pass still reads slot zero or the PLE staging rows.
        // The future is encoded but cannot execute until this actual input
        // has replaced the previous step's input and advanced n-gram history.
        {
            let _scope = profile::Scope::new("qwen.mixed.input_stage");
            self.scratch
                .next_token
                .view(0, &[1])
                .and_then(|slot| slot.write_bytes(&token.to_le_bytes()))
                .map_err(|_| Error::Invalid("mixed precision sampled input staging failed"))?;
        }
        {
            let _scope = profile::Scope::new("qwen.mixed.ngram_stage");
            self.model
                .prepare_step_inputs(self.state, self.scratch, token)
                .map_err(|_| Error::Invalid("mixed precision sampled ngram staging failed"))?;
        }
        let pass = {
            let _scope = profile::Scope::new("qwen.mixed.submit");
            next.pass.commit().map_err(|_| Error::Invalid("mixed precision sampled submission failed"))?
        };
        self.state.advance(1);
        self.current = Some(pass);
        Ok(())
    }

    fn encode_ahead(&mut self) -> Result<()> {
        if self.produced + 1 < self.remaining {
            self.ahead = Some(self.encode()?);
        }
        Ok(())
    }

    fn finish_current(&mut self) -> Result<Vec<f32>> {
        let current = self.current.take().ok_or(Error::Invalid("mixed precision sampled submission missing"))?;
        {
            let _scope = profile::Scope::new("qwen.mixed.wait");
            current.wait_polled().map_err(|_| Error::Invalid("mixed precision sampled execution failed"))?;
        }
        record_gpu_profile();
        let logits = {
            let _scope = profile::Scope::bytes("qwen.mixed.logit_readback", self.vocab_size * 4);
            self.scratch.logits().to_f32().map_err(|_| Error::Invalid("mixed precision sampled readout failed"))?
        };
        let _scope = profile::Scope::new("qwen.mixed.logit_validation");
        if logits.len() != self.vocab_size || !super::finite_guard::all_finite(&logits) {
            return Err(Error::Invalid("mixed precision sampled readout values"));
        }
        Ok(logits)
    }

    fn next_draw(&mut self) -> std::result::Result<u32, SampledFailure> {
        if self.current.is_none() {
            let next = self.encode()?;
            self.submit(next, self.first)?;
            self.encode_ahead()?;
        }
        let logits = self.finish_current()?;
        let token = (self.sample)(&logits).map_err(SampledFailure::Sampler)?;
        if token as usize >= self.vocab_size {
            return Err(SampledFailure::Sampler(Error::Invalid("mixed precision sampled token ID")));
        }
        drop(logits);
        self.produced += 1;
        if self.produced < self.remaining {
            let next = self.ahead.take().ok_or(Error::Invalid("mixed precision sampled encoding missing"))?;
            self.submit(next, token)?;
            self.encode_ahead()?;
        }
        Ok(token)
    }

    fn drain(&mut self) {
        self.ahead = None;
        // Deliberately no sampler callback: discarded lookahead must not
        // consume randomness or invoke application code during cleanup.
        if self.current.is_some() && self.finish_current().is_err() {
            *self.usable = false;
        }
    }
}

impl<F> Iterator for SampledDecode<'_, F>
where
    F: FnMut(&[f32]) -> Result<u32>,
{
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped || self.produced >= self.remaining {
            return None;
        }
        match self.next_draw() {
            Ok(token) => Some(Ok(token)),
            Err(failure) => {
                self.stopped = true;
                let error = match failure {
                    SampledFailure::Forward(error) => {
                        *self.usable = false;
                        error
                    },
                    SampledFailure::Sampler(error) => error,
                };
                self.drain();
                Some(Err(error))
            },
        }
    }
}

impl<F> Drop for SampledDecode<'_, F>
where
    F: FnMut(&[f32]) -> Result<u32>,
{
    fn drop(&mut self) {
        self.drain();
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn pending_prompt_tokens_count_toward_the_same_context_limit() {
        assert_eq!(admitted_position(3, 2, 7, 8, 6).unwrap(), 6);
        assert!(admitted_position(3, 3, 7, 8, 6).is_err());
        assert!(admitted_position(3, 2, 8, 8, 6).is_err());
        assert!(admitted_position(usize::MAX, 1, 0, 8, MAX_CONTEXT).is_err());
        assert_eq!(admitted_position(0, 0, 7, 8, 6).unwrap(), 1);
    }

    #[stark_test]
    fn greedy_decode_budget_admits_only_consumed_tokens_without_overflow() {
        assert_eq!(admitted_decode_end(6, 0, 6).unwrap(), 6);
        assert_eq!(admitted_decode_end(3, 3, 6).unwrap(), 6);
        assert!(admitted_decode_end(6, 1, 6).is_err());
        assert!(admitted_decode_end(3, 4, 6).is_err());
        assert!(admitted_decode_end(usize::MAX, 1, usize::MAX).is_err());
    }

    #[stark_test]
    fn sampled_decode_requires_a_fresh_full_readout_and_valid_selected_token() {
        assert!(admitted_sampled_start(3, 0, Some(3), 7, 8).is_ok());
        assert!(admitted_sampled_start(3, 0, None, 7, 8).is_err());
        assert!(admitted_sampled_start(3, 0, Some(2), 7, 8).is_err());
        assert!(admitted_sampled_start(3, 1, Some(3), 7, 8).is_err());
        assert!(admitted_sampled_start(3, 0, Some(3), 8, 8).is_err());
        assert!(admitted_sampled_start(0, 0, Some(0), 0, 8).is_err());
        assert_eq!(admitted_decode_end(3, 0, 6).unwrap(), 3);
        assert_eq!(admitted_decode_end(3, 3, 6).unwrap(), 6);
        assert!(admitted_decode_end(3, 4, 6).is_err());
        assert!(admitted_decode_end(usize::MAX, 1, usize::MAX).is_err());
    }

    /// Full-logit parity under deliberately non-argmax choices. Cleanup must
    /// not invoke the caller's sampler, consume randomness, or hide faults.
    #[stark_test]
    #[ignore]
    fn private_checkpoint_sampled_encode_ahead_preserves_readouts_and_callback_count() {
        use std::cell::Cell;

        use sha2::{Digest, Sha256};

        fn append_logits(
            digest: &mut Sha256,
            logits: &[f32],
        ) {
            for value in logits {
                digest.update(value.to_bits().to_le_bytes());
            }
        }

        fn digest_logits(logits: &[f32]) -> [u8; 32] {
            let mut digest = Sha256::new();
            append_logits(&mut digest, logits);
            digest.finalize().into()
        }

        fn forced_non_argmax(
            logits: &[f32],
            index: usize,
        ) -> u32 {
            assert!(logits.len() > 1);
            let mut best = 0;
            for candidate in 1..logits.len() {
                if logits[candidate] > logits[best] {
                    best = candidate;
                }
            }
            let candidate = [0, logits.len() / 2, logits.len() - 1][index % 3];
            (if candidate == best {
                (candidate + 1) % logits.len()
            } else {
                candidate
            }) as u32
        }

        let path = std::env::var_os("STARK_LILY_TEST_MODEL_PATH").expect("private model fixture path");
        let path = Path::new(&path);
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
            .unwrap_or_else(|_| panic!("private model tokenizer failed"));
        let input = tokenizer
            .encode("Hello.", false)
            .unwrap_or_else(|_| panic!("private model tokenization failed"))
            .get_ids()
            .to_vec();
        assert!(!input.is_empty());
        let mut model = LilyModel::load(path).unwrap();
        let prefill = |model: &mut LilyModel| {
            model.reset();
            for &token in &input[..input.len() - 1] {
                model.ingest(token).unwrap();
            }
            model.step(*input.last().unwrap()).unwrap()
        };
        let initial = prefill(&mut model);
        let initial_hash = digest_logits(&initial);

        for remaining in [0, 1, 2, 7] {
            let initial = prefill(&mut model);
            let mut token = forced_non_argmax(&initial, 0);
            let mut expected = Vec::new();
            let mut expected_logits = Sha256::new();
            for index in 0..remaining {
                let values = model.step(token).unwrap();
                append_logits(&mut expected_logits, &values);
                token = forced_non_argmax(&values, index + 1);
                expected.push(token);
            }
            let replay = prefill(&mut model);
            assert!(digest_logits(&replay) == initial_hash);
            let first = forced_non_argmax(&replay, 0);
            let mut actual_logits = Sha256::new();
            let mut callback_count = 0;
            let actual = model
                .sampled_decode(first, remaining, |values| {
                    append_logits(&mut actual_logits, values);
                    callback_count += 1;
                    Ok(forced_non_argmax(values, callback_count))
                })
                .unwrap()
                .collect::<Result<Vec<_>>>()
                .unwrap();
            assert!(actual == expected);
            assert!(actual_logits.finalize() == expected_logits.finalize());
            assert_eq!(callback_count, remaining);
            assert_eq!(model.position(), input.len() + remaining);
            assert!(!model.is_fenced());
        }

        for pulled in [0, 1, 3] {
            let initial = prefill(&mut model);
            let callbacks = Cell::new(0);
            let first = forced_non_argmax(&initial, 0);
            let mut stream = model
                .sampled_decode(first, 7, |values| {
                    callbacks.set(callbacks.get() + 1);
                    Ok(forced_non_argmax(values, callbacks.get()))
                })
                .unwrap();
            assert_eq!(callbacks.get(), 0);
            for _ in 0..pulled {
                stream.next().unwrap().unwrap();
            }
            drop(stream);
            assert_eq!(callbacks.get(), pulled);
            assert!(!model.is_fenced());
            assert_eq!(
                model.position(),
                input.len()
                    + if pulled == 0 {
                        0
                    } else {
                        pulled + 1
                    }
            );
            assert!(digest_logits(&prefill(&mut model)) == initial_hash);
        }

        for invalid_id in [false, true] {
            let initial = prefill(&mut model);
            let first = forced_non_argmax(&initial, 0);
            let invalid = model.vocab_size() as u32;
            let callbacks = Cell::new(0);
            let mut stream = model
                .sampled_decode(first, 3, |_| {
                    callbacks.set(callbacks.get() + 1);
                    if invalid_id {
                        Ok(invalid)
                    } else {
                        Err(Error::Invalid("private sampler failure"))
                    }
                })
                .unwrap();
            assert!(stream.next().unwrap().is_err());
            assert!(stream.next().is_none());
            drop(stream);
            assert_eq!(callbacks.get(), 1);
            assert!(!model.is_fenced());
            assert!(digest_logits(&prefill(&mut model)) == initial_hash);
        }

        let initial = prefill(&mut model);
        model.ingest(forced_non_argmax(&initial, 0)).unwrap();
        assert!(model.sampled_decode(0, 1, |_| Ok(0)).is_err());
        assert_eq!(model.position(), 0);
        let initial = prefill(&mut model);
        model.step_greedy(forced_non_argmax(&initial, 0)).unwrap();
        assert!(model.sampled_decode(0, 1, |_| Ok(0)).is_err());
        assert_eq!(model.position(), 0);
        assert!(digest_logits(&prefill(&mut model)) == initial_hash);
    }

    /// Matches the synchronous greedy path, including lazy and early drops.
    /// The private fixture is never printed or persisted by this test.
    #[stark_test]
    #[ignore]
    fn private_checkpoint_greedy_encode_ahead_matches_sync_and_drains_early_exits() {
        let path = std::env::var_os("STARK_LILY_TEST_MODEL_PATH").expect("private model fixture path");
        let path = Path::new(&path);
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
            .unwrap_or_else(|_| panic!("private model tokenizer failed"));
        let input = tokenizer
            .encode("Hello.", false)
            .unwrap_or_else(|_| panic!("private model tokenization failed"))
            .get_ids()
            .to_vec();
        assert!(!input.is_empty());
        let mut model = LilyModel::load(path).unwrap();
        let prefill = |model: &mut LilyModel| {
            model.reset();
            for &token in &input[..input.len() - 1] {
                model.ingest(token).unwrap();
            }
            model.step_greedy(*input.last().unwrap()).unwrap()
        };
        for remaining in [0, 1, 2, 7] {
            let first = prefill(&mut model);
            let mut token = first;
            let mut expected = Vec::new();
            for _ in 0..remaining {
                token = model.step_greedy(token).unwrap();
                expected.push(token);
            }
            let replay = prefill(&mut model);
            assert!(first == replay);
            let actual = model.greedy_decode(replay, remaining).unwrap().collect::<Result<Vec<_>>>().unwrap();
            assert!(actual == expected);
            assert!(!model.is_fenced());
            assert_eq!(model.position(), input.len() + remaining);
        }
        for pulled in [0, 1, 3] {
            let first = prefill(&mut model);
            let mut stream = model.greedy_decode(first, 7).unwrap();
            for _ in 0..pulled {
                stream.next().unwrap().unwrap();
            }
            drop(stream);
            assert!(!model.is_fenced());
            // A drop before first pull has no work. Later drops drain at most
            // the one already-submitted pass, without submitting its future.
            assert_eq!(
                model.position(),
                input.len()
                    + if pulled == 0 {
                        0
                    } else {
                        pulled + 1
                    }
            );
            assert!(prefill(&mut model) == first);
        }
    }

    /// Uses a separately supplied private fixture; never prints inputs/logits.
    #[stark_test]
    #[ignore]
    fn private_checkpoint_reset_replays_batched_prefill_and_caller_selected_decode() {
        let path = std::env::var_os("STARK_LILY_TEST_MODEL_PATH").expect("private model fixture path");
        let path = Path::new(&path);
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
            .unwrap_or_else(|_| panic!("private model tokenizer failed"));
        let input = tokenizer
            .encode("Hello.", false)
            .unwrap_or_else(|_| panic!("private model tokenization failed"))
            .get_ids()
            .to_vec();
        assert!(!input.is_empty());
        let mut model = LilyModel::load(path).unwrap();
        let mut previous = None;
        for _ in 0..2 {
            model.reset();
            assert_eq!(model.position(), 0);
            for &token in &input[..input.len() - 1] {
                model.ingest(token).unwrap();
            }
            assert_eq!(model.position(), input.len() - 1);
            let prompt = model.step(*input.last().unwrap()).unwrap();
            assert_eq!(model.position(), input.len());
            let greedy = prompt.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            let caller_selected = ((greedy + 1) % model.vocab_size()) as u32;
            let next = model.step(caller_selected).unwrap();
            assert_eq!(model.position(), input.len() + 1);
            if let Some(previous) = &previous {
                let previous: &Vec<f32> = previous;
                assert!(next.iter().zip(previous).all(|(a, b)| a.to_bits() == b.to_bits()));
            } else {
                previous = Some(next);
            }
        }
    }
}
