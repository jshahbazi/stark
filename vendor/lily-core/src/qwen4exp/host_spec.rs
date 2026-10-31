// SPDX-License-Identifier: Apache-2.0
//! Stark's host-sampled MTP bridge. The GPU computes complete F32 readouts;
//! the caller owns draft distributions, acceptance and all random draws.

use anyhow::{Result, ensure};
use std::slice::ChunksExact;

use super::model::{
    AttnPos, BatchMode, DecodeState, LayerState, MAX_DRAFTS, Qwen4ExpModel, Scratch,
};
use super::ngram::NgramHasher;
use crate::engine::DecodeStateApi;
use crate::kernels::{Rope, sample::SamplingParams};
use crate::metal::{BlitCopy, MetalContext};
use crate::moe_ffn::{prefix_rows, project_mat};

/// Readouts for a pending token and its proposed successors. Consumed by
/// `finish_host_verify` before the state can run another forward pass.
pub struct HostVerification {
    flat_logits: Vec<f32>,
    vocab_size: usize,
    pos_before: usize,
    hist_before: Option<[u32; 2]>,
    tokens: Vec<u32>,
}

impl HostVerification {
    /// Borrows complete vocabulary rows from the owned flat readout buffer.
    /// The verifier validates divisibility before constructing this value.
    pub fn logits(&self) -> ChunksExact<'_, f32> {
        self.flat_logits.chunks_exact(self.vocab_size)
    }
}

impl Qwen4ExpModel {
    /// Verifies the pending token and up to three proposals in one batched
    /// graph with the ordinary decode arithmetic at every active boundary.
    /// The strict single-token implementation remains a reference control.
    pub fn verify_host_batched(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        pending: u32,
        drafts: &[u32],
    ) -> Result<HostVerification> {
        ensure!(
            state.spec.is_none()
                && drafts.len() <= MAX_DRAFTS
                && self.weights.mtp.is_some()
                && s.spec.is_some(),
            "invalid candidate verification state"
        );
        let mut tokens = vec![pending];
        tokens.extend_from_slice(drafts);
        ensure!(
            tokens.iter().all(|&t| (t as usize) < self.config.vocab_size),
            "candidate input ID"
        );
        let m = tokens.len();
        state.ensure_capacity(ctx, state.pos + m)?;
        self.ensure_prefill_scratch(ctx, s, m)?;
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("candidate scratch"))?;
        let ps = capacity.chunk(&tokens)?;
        if let (Some(w), Some(p), Some(pst)) = (
            self.weights.layers.iter().find_map(|l| l.ple.as_deref()),
            &ps.ple,
            &state.ple,
        ) {
            self.stage_ngram(w, p, &tokens, pst.hist)?;
        }
        let pos_before = state.pos;
        let hist_before = state.ple.as_ref().map(|p| p.hist);
        self.encode_batch(ctx, state, s, &ps, BatchMode::VerifyHost)?
            .commit()?
            .wait()?;
        state.pos += m;
        state.conv_slot = 1 - state.conv_slot;
        if let Some(pst) = &mut state.ple {
            pst.hist = NgramHasher::advance(pst.hist, &tokens);
        }
        let sp =
            s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("candidate captures"))?;
        let values = prefix_rows(&sp.logits, m)?.to_f32()?;
        ensure!(values.len() == m * self.config.vocab_size, "candidate readout shape");
        ensure!(crate::finite_guard::all_finite(&values), "candidate readout values");
        Ok(HostVerification {
            flat_logits: values,
            vocab_size: self.config.vocab_size,
            pos_before,
            hist_before,
            tokens,
        })
    }

    /// Builds a short draft chain using the same complete-logit sampler as
    /// the caller. Head caches may contain speculative suffixes afterwards;
    /// later valid rows overwrite them before reading them.
    pub fn draft_host(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        first: u32,
        drafts: usize,
        sample: &mut dyn FnMut(&[f32]) -> Result<u32>,
    ) -> Result<Vec<u32>> {
        ensure!(state.spec.is_none() && state.pos > 0, "invalid host draft state");
        ensure!((1..=MAX_DRAFTS).contains(&drafts), "invalid host draft count");
        ensure!((first as usize) < self.config.vocab_size, "invalid host draft input");
        state.ensure_capacity(ctx, state.pos + drafts)?;
        self.ensure_prefill_scratch(ctx, s, 1)?;
        let s = &*s;
        let mtp =
            self.weights.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP head"))?;
        let mst = state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP scratch"))?;
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("no head scratch"))?;
        let ps = capacity.rows(1)?;
        let wide = self.config.hc_width();
        let ids = sp.mtp_ids.view(0, &[1])?;
        let logits = sp.logits.view(0, &[1, self.config.vocab_size])?;
        let mut input = first;
        let mut proposals = Vec::with_capacity(drafts);
        for i in 0..drafts {
            ids.write_bytes(&input.to_le_bytes())?;
            let hidden = if i == 0 {
                mst.hidden.view(0, &[1, wide])?
            } else {
                ps.mtp_hyper
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("no head residual"))?
                    .view(0, &[1, wide])?
            };
            let pass = ctx.begin_concurrent()?;
            pass.set_label("host MTP draft");
            self.mtp_block(
                ctx,
                &pass,
                mtp,
                mst,
                &hidden,
                &ids,
                AttnPos::host(state.pos - 1 + i),
                s,
                &ps,
                Rope::Delta(state.rope_delta),
                &[],
            )?;
            let hyper = ps
                .mtp_hyper
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no head residual"))?;
            self.hc_read_batched(ctx, &pass, &mtp.mixer, hyper, s, &ps)?;
            project_mat(
                ctx,
                &pass,
                &ps.hc.mixed,
                mtp.head(&self.weights.lm_head),
                &logits,
                &s.dequant,
            )?;
            pass.level_barrier(&[&logits])?;
            s.sync.signal_done(&pass)?;
            pass.end()?.commit()?.wait()?;
            let values = logits.to_f32()?;
            ensure!(
                values.len() == self.config.vocab_size
                    && crate::finite_guard::all_finite(&values),
                "invalid host draft readout"
            );
            input = sample(&values)?;
            ensure!(
                (input as usize) < self.config.vocab_size,
                "invalid host draft output"
            );
            proposals.push(input);
        }
        Ok(proposals)
    }

    /// Executes ordinary decode for every proposed row, retaining its exact
    /// F32 readouts and state. This conservative verifier does not claim a
    /// batched arithmetic or bandwidth speedup.
    pub fn verify_host(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        pending: u32,
        drafts: &[u32],
    ) -> Result<HostVerification> {
        ensure!(
            state.spec.is_none() && drafts.len() <= MAX_DRAFTS,
            "invalid host verify state"
        );
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(pending);
        tokens.extend_from_slice(drafts);
        ensure!(
            tokens.iter().all(|&t| (t as usize) < self.config.vocab_size),
            "invalid host verify input"
        );
        let m = tokens.len();
        state.ensure_capacity(ctx, state.pos + m)?;
        self.ensure_prefill_scratch(ctx, s, m)?;
        let pos_before = state.pos;
        let hist_before = state.ple.as_ref().map(|p| p.hist);
        // Decode updates one window in place. Preserve its original in the
        // other slot so rollback can replay only committed projected inputs.
        let slot = state.conv_slot;
        let mut copies = Vec::new();
        for layer in &state.layers {
            if let LayerState::Gdn { conv_windows, .. } = layer {
                copies.push(BlitCopy {
                    src: &conv_windows[slot],
                    src_offset: 0,
                    dst: &conv_windows[1 - slot],
                    dst_offset: 0,
                    len: conv_windows[slot].byte_len(),
                });
            }
        }
        if let Some(ple) = &state.ple {
            copies.push(BlitCopy {
                src: &ple.conv_windows[slot],
                src_offset: 0,
                dst: &ple.conv_windows[1 - slot],
                dst_offset: 0,
                len: ple.conv_windows[slot].byte_len(),
            });
        }
        ctx.blit_copy(&copies)?;
        state.conv_slot = 1 - state.conv_slot;
        for (row, &token) in tokens.iter().enumerate() {
            s.next_token.view(0, &[1])?.write_bytes(&token.to_le_bytes())?;
            self.prepare_step_inputs(state, s, token)?;
            self.encode_host_verify_step(ctx, state, s, row, m)?.commit()?.wait()?;
            state.pos += 1;
        }
        let sp =
            s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no verify captures"))?;
        let values = prefix_rows(&sp.logits, m)?.to_f32()?;
        ensure!(
            values.len() == m * self.config.vocab_size
                && crate::finite_guard::all_finite(&values),
            "invalid host verify readout"
        );
        Ok(HostVerification {
            flat_logits: values,
            vocab_size: self.config.vocab_size,
            pos_before,
            hist_before,
            tokens,
        })
    }

    /// Keeps `accepted` proposals plus the pending input, restores recurrent
    /// state and convolution windows, and catches the head up only through
    /// committed rows. The final sampled output is still pending input.
    pub fn finish_host_verify(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &Scratch,
        verified: &HostVerification,
        accepted: usize,
    ) -> Result<()> {
        let m = verified.tokens.len();
        ensure!(
            accepted < m
                && state.pos > verified.pos_before
                && state.pos <= verified.pos_before + m,
            "invalid host verify rollback"
        );
        let sp =
            s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no verify captures"))?;
        if accepted > 0 {
            sp.mtp_ids
                .view(0, &[accepted])?
                .write_bytes(bytemuck::cast_slice(&verified.tokens[1..=accepted]))?;
        }
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("no verify scratch"))?;
        let wide = self.config.hc_width();
        let hidden = capacity.hyper.view(0, &[accepted.max(1), wide])?;
        let keep = capacity.hyper.view(accepted * wide, &[wide])?;
        self.encode_draft(
            ctx,
            state,
            s,
            &hidden,
            accepted,
            0,
            verified.pos_before,
            Some(&keep),
            (accepted + 1 < m).then_some((accepted, m)),
            0,
            &SamplingParams::greedy(),
            0,
        )?
        .commit()?
        .wait()?;
        state.pos = verified.pos_before + accepted + 1;
        if let (Some(pst), Some(hist)) = (&mut state.ple, verified.hist_before) {
            pst.hist = NgramHasher::advance(hist, &verified.tokens[..=accepted]);
        }
        Ok(())
    }
}
