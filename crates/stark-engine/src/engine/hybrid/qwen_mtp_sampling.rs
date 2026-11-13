// SPDX-License-Identifier: Apache-2.0
//! Private, bounded Edge0 exact-sampling evaluation. This adapter deliberately
//! uses ordinary target single-token steps and the caller's existing
//! backend-independent sampler. It does not enable worker MTP capabilities.
//! All residuals, checkpoints, tokens and distributions remain private in RAM.

use super::{QwenModel, mtp};
use crate::engine::hybrid::{
    Error, MtpSampler, MtpStats, Result,
    mtp_controller::{MtpBackend, MtpDecode},
    weight_metal::DeviceVector,
};

/// One serialized target/head pair. A pending target-wide is the residual
/// immediately preceding the as-yet-unconsumed output token. Only drafting
/// retains a speculative head-wide, which is discarded on every rewind.
struct Backend<'a> {
    target: &'a mut QwenModel,
    head: &'a mut mtp::Head,
    pending: Option<mtp::WideResidual>,
    draft_wide: Option<DeviceVector>,
    draft_depth: usize,
    fenced: bool,
}

impl MtpBackend for Backend<'_> {
    type Checkpoint = mtp::CombinedCheckpoint;

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
        mtp::CombinedCheckpoint::capture(
            self.target,
            self.head,
            self.pending.as_ref().ok_or(Error::Invalid("private MTP sampling pending wide"))?,
        )
    }

    fn draft_logits(
        &mut self,
        input: u32,
        depth: usize,
    ) -> Result<Vec<f32>> {
        if self.is_fenced() || depth != self.draft_depth || depth >= 3 {
            return Err(Error::Invalid("private MTP sampling draft order"));
        }
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
        output.logits.take().ok_or(Error::Invalid("private MTP sampling draft readout"))
    }

    fn target_logits(
        &mut self,
        input: u32,
    ) -> Result<Vec<f32>> {
        if self.is_fenced() {
            return Err(Error::Invalid("private MTP sampling fenced"));
        }
        let (logits, wide) = self.target.step_with_wide(input)?;
        self.pending = Some(wide);
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
        self.pending = Some(saved.restore(self.target, self.head)?);
        // Replay aligns the front of the draft cache using actual target
        // residuals. It never projects a vocabulary row in the head or calls
        // a sampler. Target replay uses the identical serial execution path.
        for &input in inputs {
            self.head.target_catch_up(
                self.target,
                self.pending.as_ref().ok_or(Error::Invalid("private MTP sampling replay wide"))?,
                input,
                false,
            )?;
            let (_, wide) = self.target.step_with_wide(input)?;
            self.pending = Some(wide);
        }
        Ok(())
    }

    fn fence(&mut self) {
        self.fenced = true;
        self.pending = None;
        self.draft_wide = None;
        self.head.fence_target(self.target);
    }
}

impl QwenModel {
    /// Evaluation only: serial prompt setup, first draw and a bounded MTP tail.
    /// `on_output` returning false finishes at the delivered boundary; an error
    /// cancels and fences the complete request. Nothing promotes capabilities.
    pub(in crate::engine::hybrid) fn evaluate_edge0_mtp(
        &mut self,
        tokens: &[u32],
        max_outputs: usize,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
        on_output: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<MtpStats> {
        if std::env::var("STARK_EDGE0_REFERENCE").ok().as_deref() != Some("1")
            || std::env::var("STARK_EDGE0_METAL").ok().as_deref() != Some("1")
            || std::env::var("STARK_EDGE0_RESIDENT").ok().as_deref() == Some("1")
            || !self.geometry.edge0
            || !self.geometry.edge0_metal
            || !self.graph_enabled
            || !(1..=3).contains(&drafts)
            || tokens.is_empty()
            || tokens.iter().any(|&token| token as usize >= self.geometry.vocab)
            || tokens.len().checked_add(max_outputs.saturating_sub(1)).is_none_or(|end| end > self.geometry.context)
        {
            self.reset();
            return Err(Error::Invalid("private MTP sampling evaluation admission"));
        }
        let mut stats = MtpStats::default();
        if max_outputs == 0 {
            return Ok(stats);
        }
        self.reset();
        let mut head = mtp::Head::for_target(self)?;
        let result = (|| {
            let mut pending = None;
            let mut initial = None;
            for &input in tokens {
                if let Some(wide) = pending.as_ref() {
                    head.target_catch_up(self, wide, input, false)?;
                }
                let (logits, wide) = self.step_with_wide(input)?;
                initial = Some(logits);
                pending = Some(wide);
            }
            let initial = initial.ok_or(Error::Invalid("private MTP sampling initial readout"))?;
            if initial.len() != self.geometry.vocab || initial.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("private MTP sampling initial readout"));
            }
            sampler.begin_round();
            let first = sampler.target(&initial, None)?;
            if first as usize >= self.geometry.vocab {
                return Err(Error::Invalid("private MTP sampling initial ID"));
            }
            if !on_output(first)? || max_outputs == 1 {
                return Ok(());
            }
            let mut backend = Backend {
                target: self,
                head: &mut head,
                pending,
                draft_wide: None,
                draft_depth: 0,
                fenced: false,
            };
            let mut stream = MtpDecode::new(&mut backend, sampler, &mut stats, first, max_outputs - 1, drafts)?;
            while let Some(output) = stream.next() {
                let output = output?;
                match on_output(output) {
                    Ok(true) => {},
                    Ok(false) => break,
                    Err(error) => {
                        stream.cancel()?;
                        return Err(error);
                    },
                }
            }
            stream.finish()
        })();
        if result.is_err() {
            head.fence_target(self);
        }
        result.map(|()| stats)
    }
}
