// SPDX-License-Identifier: Apache-2.0
//! Bounded local Edge0 exact-sampling worker execution. This adapter deliberately
//! uses ordinary target single-token steps and the caller's existing
//! backend-independent sampler. The complete request holds one exclusive head lease.
//! All residuals, checkpoints, tokens and distributions remain private in RAM.

use super::{QwenModel, mtp};
use crate::engine::hybrid::{
    Error, MtpSampler, MtpStats, Result, mtp_controller::MtpBackend, weight_metal::DeviceVector,
};

/// One serialized target/head pair. A pending target-wide is the residual
/// immediately preceding the as-yet-unconsumed output token. Only drafting
/// retains a speculative head-wide, which is discarded on every rewind.
struct Backend<'a> {
    target: &'a mut QwenModel,
    head: mtp::Head,
    control: &'a mut dyn FnMut() -> Result<()>,
    completed: bool,
    pending: Option<mtp::WideResidual>,
    draft_wide: Option<DeviceVector>,
    draft_depth: usize,
    fenced: bool,
}

impl MtpBackend for Backend<'_> {
    type Checkpoint = mtp::CombinedCheckpoint;

    fn check_control(&mut self) -> Result<()> {
        (self.control)()
    }

    fn vocab_size(&self) -> usize {
        self.target.geometry.vocab
    }

    fn is_fenced(&self) -> bool {
        self.fenced || self.head.is_fenced()
    }

    fn checkpoint(&mut self) -> Result<Self::Checkpoint> {
        if self.is_fenced() {
            return Err(Error::Invalid("private MTP sampling fenced"));
        }
        (self.control)()?;
        let saved = mtp::CombinedCheckpoint::capture(
            self.target,
            &mut self.head,
            self.pending.as_ref().ok_or(Error::Invalid("private MTP sampling pending wide"))?,
        )?;
        (self.control)()?;
        Ok(saved)
    }

    fn draft_logits(
        &mut self,
        input: u32,
        depth: usize,
    ) -> Result<Vec<f32>> {
        if self.is_fenced() || depth != self.draft_depth || depth >= 3 {
            return Err(Error::Invalid("private MTP sampling draft order"));
        }
        (self.control)()?;
        let mut output = if depth == 0 {
            self.head.target_step(
                self.target,
                self.pending.as_ref().ok_or(Error::Invalid("private MTP sampling pending wide"))?,
                input,
                false,
            )?
        } else {
            self.head.step(
                &mtp::ActualWeights {
                    weights: &self.target.weights,
                    embedding: &self.target.embedding,
                    readout: &self.target.readout,
                },
                self.head.next_cell(),
                self.draft_wide.as_ref().ok_or(Error::Invalid("private MTP sampling draft wide"))?,
                input,
                false,
            )?
        };
        if !output.stages.is_empty() {
            return Err(Error::Invalid("private MTP sampling trace enabled"));
        }
        self.draft_wide = Some(output.wide.take().ok_or(Error::Invalid("private MTP sampling draft wide"))?);
        self.draft_depth += 1;
        (self.control)()?;
        output.logits.take().ok_or(Error::Invalid("private MTP sampling draft readout"))
    }

    fn target_logits(
        &mut self,
        input: u32,
    ) -> Result<Vec<f32>> {
        if self.is_fenced() {
            return Err(Error::Invalid("private MTP sampling fenced"));
        }
        (self.control)()?;
        let (logits, wide) = self.target.step_with_wide(input)?;
        self.pending = Some(wide);
        (self.control)()?;
        Ok(logits)
    }

    fn rewind(
        &mut self,
        saved: &Self::Checkpoint,
        inputs: &[u32],
    ) -> Result<()> {
        if self.is_fenced() {
            return Err(Error::Invalid("private MTP sampling fenced"));
        }
        self.pending = None;
        self.draft_wide = None;
        self.draft_depth = 0;
        (self.control)()?;
        self.pending = Some(saved.restore(self.target, &mut self.head)?);
        // Replay aligns the front of the draft cache using actual target
        // residuals. It never projects a vocabulary row in the head or calls
        // a sampler. Target replay uses the identical serial execution path.
        for &input in inputs {
            (self.control)()?;
            self.head.target_catch_up(
                self.target,
                self.pending.as_ref().ok_or(Error::Invalid("private MTP sampling replay wide"))?,
                input,
                false,
            )?;
            (self.control)()?;
            let (_, wide) = self.target.step_with_wide(input)?;
            self.pending = Some(wide);
        }
        (self.control)()?;
        Ok(())
    }

    fn fence(&mut self) {
        self.fenced = true;
        self.pending = None;
        self.draft_wide = None;
        self.head.fence_target(self.target);
    }
}

// Covers prompt setup, callback panics and abandoned requests as well as decode
// errors. Successful cleanup is explicit, so unwinding can never retain a live
// causal boundary from an incomplete worker request.
impl Drop for Backend<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.fence();
        }
    }
}

impl QwenModel {
    pub(in crate::engine::hybrid) fn evaluate_edge0_mtp(
        &mut self,
        tokens: &[u32],
        max_outputs: usize,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
        on_output: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<MtpStats> {
        self.run_edge0_mtp(tokens, max_outputs, drafts, sampler, on_output, &mut || Ok(()))
    }

    /// Complete synchronous request. Control errors cancel promptly at every
    /// prompt/head/target/restore/replay boundary; submitted work drains before
    /// the exclusive lease is released. Output false means a clean early stop.
    pub(in crate::engine::hybrid) fn run_edge0_mtp(
        &mut self,
        tokens: &[u32],
        max_outputs: usize,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
        on_output: &mut dyn FnMut(u32) -> Result<bool>,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<MtpStats> {
        self.reset();
        if !self.supports_edge0_mtp_decode()
            || !(1..=3).contains(&drafts)
            || tokens.is_empty()
            || tokens.iter().any(|&token| token as usize >= self.geometry.vocab)
            || tokens.len().checked_add(max_outputs.saturating_sub(1)).is_none_or(|end| end > self.geometry.context)
        {
            return Err(Error::Invalid("Edge0 MTP request admission"));
        }
        control()?;
        let mut stats = MtpStats::default();
        if max_outputs == 0 {
            return Ok(stats);
        }
        // Control precedes head allocation and any submitted prompt work.
        let head = mtp::Head::for_target(self)?;
        let mut backend = Backend {
            target: self,
            head,
            control,
            completed: false,
            pending: None,
            draft_wide: None,
            draft_depth: 0,
            fenced: false,
        };
        let mut initial = None;
        for &input in tokens {
            if let Some(wide) = backend.pending.as_ref() {
                (backend.control)()?;
                backend.head.target_catch_up(backend.target, wide, input, false)?;
            }
            initial = Some(backend.target_logits(input)?);
        }
        let initial = initial.ok_or(Error::Invalid("Edge0 MTP initial readout"))?;
        if initial.len() != backend.vocab_size() || initial.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("Edge0 MTP initial readout"));
        }
        (backend.control)()?;
        sampler.begin_round();
        let first = sampler.target(&initial, None)?;
        if first as usize >= backend.vocab_size() {
            return Err(Error::Invalid("Edge0 MTP initial ID"));
        }
        // The collector checks control before publishing even when a round
        // has already queued several accepted outputs.
        crate::engine::hybrid::mtp_controller::collect(
            &mut backend,
            sampler,
            &mut stats,
            first,
            max_outputs - 1,
            drafts,
            on_output,
        )?;
        (backend.control)()?;
        // Acknowledge the final queue drain while errors can still propagate.
        // The lease drop performs a second defensive drain before readmission.
        mtp::new_graph()?.finish(&[])?;
        (backend.control)()?;
        backend.completed = true;
        backend.target.edge0_mtp_stats = stats;
        Ok(stats)
    }
}
