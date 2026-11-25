// SPDX-License-Identifier: Apache-2.0
//! Stark's host-sampled MTP bridge. The GPU computes complete F32 readouts;
//! the caller owns draft distributions, acceptance and all random draws.

use anyhow::{Result, ensure};
use std::slice::ChunksExact;

use super::model::{
    AttnPos, BatchMode, DecodeState, LayerState, MAX_DRAFTS, Qwen4ExpModel, Scratch,
};
use super::ngram::NgramHasher;
use super::spec::Rollback;
use crate::engine::DecodeStateApi;
use crate::kernels::{
    Rope,
    sample::{CHAIN_ARGMAX_TINY, SamplingParams, argmax_chain_f32, runner_up_f32},
};
use crate::metal::{BlitCopy, EncodedPass, MetalContext};
use crate::moe_ffn::{prefix_rows, project_mat};
use crate::tensor::Tensor;

/// Readouts for a pending token and its proposed successors. Consumed by
/// `finish_host_verify` before the state can run another forward pass.
///
/// Stark-local: rows are borrowed from the shared-storage readout buffer
/// instead of being copied, and each row is validated when it is read. A row
/// the caller never reads (one after the first rejection) is neither copied
/// nor scanned. The buffer is next written by the following round's first
/// draft or verify pass, which the caller encodes and commits only after it
/// has finished reading rows and has run `finish_host_verify`.
pub struct HostVerification {
    readout: Tensor,
    rows: usize,
    vocab_size: usize,
    pos_before: usize,
    hist_before: Option<[u32; 2]>,
    tokens: Vec<u32>,
    sibling_depth: Option<usize>,
}

impl HostVerification {
    fn values(&self) -> Result<&[f32]> {
        let values: &[f32] = bytemuck::try_cast_slice(self.readout.contents())
            .map_err(|_| anyhow::anyhow!("candidate readout layout"))?;
        ensure!(values.len() == self.rows * self.vocab_size, "candidate readout shape");
        Ok(values)
    }

    /// Verified rows in the batch: the pending token plus its proposals.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// One complete vocabulary row, borrowed. A nonfinite value fails closed.
    pub fn row(&self, row: usize) -> Result<&[f32]> {
        ensure!(row < self.rows, "candidate readout row");
        let values =
            &self.values()?[row * self.vocab_size..(row + 1) * self.vocab_size];
        ensure!(crate::finite_guard::all_finite(values), "candidate readout values");
        Ok(values)
    }

    /// Every complete vocabulary row, borrowed, after validating all of them.
    pub fn logits(&self) -> Result<ChunksExact<'_, f32>> {
        let values = self.values()?;
        ensure!(crate::finite_guard::all_finite(values), "candidate readout values");
        Ok(values.chunks_exact(self.vocab_size))
    }
}

/// A verify graph encoded ahead of its draft ids (Stark-local). The graph
/// reads its input ids and n-gram rows from scratch buffers the host fills
/// right before commit, so it can be encoded while the last draft pass runs.
/// Committing checks that nothing it was encoded against has moved.
pub struct EncodedVerification<'a> {
    pass: EncodedPass<'a>,
    rows: usize,
    pos: usize,
    conv_slot: usize,
    capacity: usize,
    scratch_ids: usize,
    /// Buffer identities of the GDN recurrent states the graph reads. A
    /// verify swaps them with the spec scratch's shadow (replay rollback),
    /// so an encoding is only valid against the states current at encode.
    gdn_states: Vec<usize>,
    sibling_depth: Option<usize>,
}

/// Identities of every GDN layer's current recurrent state buffer.
fn gdn_state_identities(state: &DecodeState) -> Vec<usize> {
    state
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Gdn { state, .. } => Some(state.identity()),
            LayerState::Attn { .. } => None,
        })
        .collect()
}

/// One round's proposals and its verify graph, encoded ahead (Stark-local).
pub struct HostDrafts<'a> {
    pub proposals: Vec<u32>,
    pub verify: EncodedVerification<'a>,
    /// The device selected the proposals as greedy argmaxes; the sampler saw
    /// no draft readout and must record each proposal as a point mass.
    pub device_selected: bool,
    /// Logical depth and the discarded primary tail for a packed fork.
    pub sibling: Option<(usize, Option<u32>)>,
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
        self.verify_host_packet(ctx, state, s, pending, drafts, None)
    }

    /// Verify externally supplied candidates using the same packet geometry
    /// as the greedy policy. The last input names the sibling when present.
    pub fn verify_host_packet(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        sibling_depth: Option<usize>,
    ) -> Result<HostVerification> {
        ensure!(
            sibling_depth.is_none() || self.supports_runner_up(state.pos, drafts.len()),
            "unsupported sibling packet"
        );
        self.prepare_host_verification(ctx, state, s, drafts.len() + 1)?;
        let encoded =
            self.encode_host_packet(ctx, state, s, drafts.len() + 1, sibling_depth)?;
        self.verify_host_encoded(state, s, pending, drafts, encoded)
    }

    /// Grows the state and scratch for a verify batch of `rows`. Encoded
    /// passes bind buffer addresses, so this must precede any early encode.
    fn prepare_host_verification(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            state.spec.is_none()
                && (1..=MAX_DRAFTS + 1).contains(&rows)
                && self.weights.mtp.is_some()
                && s.spec.is_some(),
            "invalid candidate verification state"
        );
        state.ensure_capacity(ctx, state.pos + rows)?;
        self.ensure_prefill_scratch(ctx, s, rows, state.capacity())
    }

    /// Encodes, without committing, the verify graph for `rows` rows at the
    /// current position. Encoding reads no token value.
    fn encode_host_verification<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        rows: usize,
    ) -> Result<EncodedVerification<'a>> {
        self.encode_host_packet(ctx, state, s, rows, None)
    }

    fn encode_host_packet<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        rows: usize,
        sibling_depth: Option<usize>,
    ) -> Result<EncodedVerification<'a>> {
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("candidate scratch"))?;
        ensure!(
            state.spec.is_none()
                && (1..=MAX_DRAFTS + 1).contains(&rows)
                && rows <= capacity.m
                && state.pos + rows <= state.capacity(),
            "invalid candidate verification state"
        );
        let ps = capacity.rows(rows)?;
        let mode = match sibling_depth {
            Some(depth) => {
                ensure!(depth > 0 && depth + 1 < rows, "invalid sibling depth");
                BatchMode::VerifyHostFork { depth }
            }
            None => BatchMode::VerifyHost,
        };
        let pass = self.encode_batch(ctx, state, s, &ps, mode)?;
        Ok(EncodedVerification {
            pass,
            rows,
            pos: state.pos,
            conv_slot: state.conv_slot,
            capacity: state.capacity(),
            scratch_ids: capacity.ids.identity(),
            gdn_states: gdn_state_identities(state),
            sibling_depth,
        })
    }

    /// Stages the batch's ids and n-gram rows, then commits a verify graph
    /// encoded ahead (see [`Self::draft_host_encoding_verify`]) and waits
    /// for it. The result equals [`Self::verify_host_batched`]'s.
    pub fn verify_host_encoded(
        &self,
        state: &mut DecodeState,
        s: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        encoded: EncodedVerification<'_>,
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
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("candidate scratch"))?;
        ensure!(
            encoded.rows == m
                && encoded.pos == state.pos
                && encoded.conv_slot == state.conv_slot
                && encoded.capacity == state.capacity()
                && encoded.scratch_ids == capacity.ids.identity()
                && encoded.gdn_states == gdn_state_identities(state),
            "candidate verification encoding changed"
        );
        let ps = capacity.chunk(&tokens)?;
        if let (Some(w), Some(p), Some(pst)) = (
            self.weights.layers.iter().find_map(|l| l.ple.as_deref()),
            &ps.ple,
            &state.ple,
        ) {
            self.stage_ngram_packet(w, p, &tokens, pst.hist, encoded.sibling_depth)?;
        }
        let pos_before = state.pos;
        let hist_before = state.ple.as_ref().map(|p| p.hist);
        encoded.pass.commit()?.wait_polled()?;
        // The pass wrote every GDN layer's final state into the shadow and
        // left the pre-verify state in place: make the final states current
        // and keep the pre-verify ones for replay rollback.
        let sp =
            s.spec.as_mut().ok_or_else(|| anyhow::anyhow!("candidate captures"))?;
        state.swap_gdn_shadow(sp)?;
        state.pos += m;
        state.conv_slot = 1 - state.conv_slot;
        if let Some(pst) = &mut state.ple {
            pst.hist = NgramHasher::advance(pst.hist, &tokens);
        }
        let sp =
            s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("candidate captures"))?;
        let verified = HostVerification {
            readout: prefix_rows(&sp.logits, m)?,
            rows: m,
            vocab_size: self.config.vocab_size,
            pos_before,
            hist_before,
            tokens,
            sibling_depth: encoded.sibling_depth,
        };
        verified.values()?;
        Ok(verified)
    }

    /// Encodes one host draft step without committing. Step `i` reads the
    /// input id the host writes to `mtp_ids[0]` before commit, and the head
    /// residual of step `i - 1`; encoding reads no token value.
    fn encode_host_draft_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        step: usize,
    ) -> Result<EncodedPass<'a>> {
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
        let hyper =
            ps.mtp_hyper.as_ref().ok_or_else(|| anyhow::anyhow!("no head residual"))?;
        let hidden = if step == 0 {
            mst.hidden.view(0, &[1, wide])?
        } else {
            hyper.view(0, &[1, wide])?
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
            AttnPos::host(state.pos - 1 + step),
            s,
            &ps,
            Rope::Delta(state.rope_delta),
            &[],
        )?;
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
        pass.end()
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
        self.ensure_prefill_scratch(ctx, s, 1, state.capacity())?;
        self.draft_host_steps(ctx, state, s, first, drafts, sample, false)
            .map(|(proposals, _)| proposals)
    }

    /// [`Self::draft_host`] with fewer host round trips (Stark-local); pass
    /// the returned encoding to [`Self::verify_host_encoded`] with the same
    /// `first` and proposals, or drop it to submit nothing.
    ///
    /// Each draft step after the first, and then the verify graph for
    /// `first` plus the proposals, is encoded while the preceding pass runs.
    /// Every pass still commits only after its predecessor completed and its
    /// input ids were written, so the arithmetic, the sampler calls and their
    /// order are those of `draft_host` then `verify_host_batched`.
    ///
    /// With `device_greedy` (a caller whose sampler drafts the first maximum
    /// of each finite row and draws nothing at random), the steps instead
    /// run as one device chain: the same step graphs, each proposal the
    /// device argmax of its readout feeding the next step with no host read
    /// in between, and only the proposal ids and per-step status words read
    /// back. `sample` is then not called and `device_selected` is set; the
    /// caller records each proposal as a point mass. A nonfinite readout
    /// fails closed at the same step the host path would. If a selected
    /// logit is a zero or subnormal, where device comparisons may differ from
    /// the host's, the chain's result is discarded and the host path reruns
    /// the whole round from the same state: the chain wrote only head cache
    /// rows at this round's positions, which the rerun rewrites before
    /// reading, and scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_host_encoding_verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        first: u32,
        drafts: usize,
        device_greedy: bool,
        sample: &mut dyn FnMut(&[f32]) -> Result<u32>,
    ) -> Result<HostDrafts<'a>> {
        ensure!(state.spec.is_none() && state.pos > 0, "invalid host draft state");
        ensure!((1..=MAX_DRAFTS).contains(&drafts), "invalid host draft count");
        ensure!((first as usize) < self.config.vocab_size, "invalid host draft input");
        // Grow first: the verify graph is encoded before its turn.
        self.prepare_host_verification(ctx, state, s, drafts + 1)?;
        if device_greedy
            && let Some(drafted) =
                self.draft_device_greedy(ctx, state, s, first, drafts, false)?
        {
            return Ok(drafted);
        }
        let (proposals, verify) =
            self.draft_host_steps(ctx, state, s, first, drafts, sample, true)?;
        let verify =
            verify.ok_or_else(|| anyhow::anyhow!("candidate verification missing"))?;
        Ok(HostDrafts { proposals, verify, device_selected: false, sibling: None })
    }

    /// Short-context greedy experiment. The last slot is assigned to the
    /// first retained position where the runner-up has at least half the
    /// primary's probability. Up to three slots are available, including an
    /// otherwise unused slot on an output-capped one- or two-primary round.
    pub fn draft_host_with_runner_up<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        first: u32,
        drafts: usize,
    ) -> Result<HostDrafts<'a>> {
        ensure!(
            state.pos > 0 && (first as usize) < self.config.vocab_size,
            "invalid host draft input"
        );
        ensure!(self.supports_runner_up(state.pos, drafts), "invalid runner-up round");
        self.prepare_host_verification(ctx, state, s, drafts.min(2) + 2)?;
        if let Some(drafted) =
            self.draft_device_greedy(ctx, state, s, first, drafts, true)?
        {
            return Ok(drafted);
        }
        // As on the ordinary greedy route, retry a flushed primary on the
        // host. This route also ranks the runner-up with exact IEEE comparisons.
        let mut runners = Vec::new();
        let mut select = |values: &[f32]| {
            let (mut first, mut second) = (0, 1);
            if values[second] > values[first] {
                std::mem::swap(&mut first, &mut second);
            }
            for id in 2..values.len() {
                if values[id] > values[first] {
                    second = first;
                    first = id;
                } else if values[id] > values[second] {
                    second = id;
                }
            }
            runners.push((second as u32, values[first] - values[second]));
            Ok(first as u32)
        };
        let (proposals, verify) =
            self.draft_host_steps(ctx, state, s, first, drafts, &mut select, true)?;
        self.pack_host_drafts(
            ctx,
            state,
            s,
            proposals,
            verify.unwrap(),
            &runners[..drafts.min(2)],
        )
    }

    pub fn supports_runner_up(&self, pos: usize, drafts: usize) -> bool {
        !self.config.official_mlx_rounding()
            && (1..=MAX_DRAFTS).contains(&drafts)
            && pos.saturating_add(drafts.min(2) + 2)
                <= self.config.indexer.dense_limit()
    }

    fn pack_host_drafts<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        mut proposals: Vec<u32>,
        mut verify: EncodedVerification<'a>,
        runners: &[(u32, f32)],
    ) -> Result<HostDrafts<'a>> {
        let mut sibling = None;
        for (i, &(runner, gap)) in runners.iter().enumerate() {
            ensure!(
                (runner as usize) < self.config.vocab_size
                    && runner != proposals[i]
                    && gap >= 0.0,
                "invalid runner-up output"
            );
            if gap <= std::f32::consts::LN_2 {
                let discarded =
                    if proposals.len() == MAX_DRAFTS { proposals.pop() } else { None };
                sibling = Some((i + 1, discarded));
                proposals.push(runner);
                drop(verify);
                verify = self.encode_host_packet(
                    ctx,
                    state,
                    s,
                    proposals.len() + 1,
                    Some(i + 1),
                )?;
                break;
            }
        }
        Ok(HostDrafts { proposals, verify, device_selected: true, sibling })
    }

    /// The device chain of [`Self::draft_host_encoding_verify`]. `None`
    /// when a selection must be redone on the host.
    fn draft_device_greedy<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        first: u32,
        drafts: usize,
        runner_up: bool,
    ) -> Result<Option<HostDrafts<'a>>> {
        let mtp =
            self.weights.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP head"))?;
        let mst = state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP scratch"))?;
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("no head scratch"))?;
        let ps = capacity.rows(1)?;
        let wide = self.config.hc_width();
        let vocab = self.config.vocab_size;
        let logits = sp.logits.view(0, &[1, vocab])?;
        let row = sp.logits.view(0, &[vocab])?;
        let hyper =
            ps.mtp_hyper.as_ref().ok_or_else(|| anyhow::anyhow!("no head residual"))?;
        // mtp_ids[0] carries the host input. Slots 1..=drafts hold the steps'
        // status words; they are rewritten only by this round's rollback,
        // after the host has read them. Proposals go to draft_tokens.
        let first_id = sp.mtp_ids.view(0, &[1])?;
        let status = sp.mtp_ids.view(1, &[drafts])?;
        let pass = ctx.begin_concurrent()?;
        pass.set_label("host MTP greedy draft chain");
        for i in 0..drafts {
            let input = if i == 0 {
                first_id.clone()
            } else {
                sp.draft_tokens.view(i - 1, &[1])?
            };
            let hidden = if i == 0 {
                mst.hidden.view(0, &[1, wide])?
            } else {
                hyper.view(0, &[1, wide])?
            };
            self.mtp_block(
                ctx,
                &pass,
                mtp,
                mst,
                &hidden,
                &input,
                AttnPos::host(state.pos - 1 + i),
                s,
                &ps,
                Rope::Delta(state.rope_delta),
                &[],
            )?;
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
            argmax_chain_f32(
                ctx,
                &pass,
                &row,
                &s.sampler,
                &sp.draft_tokens.view(i, &[1])?,
                &sp.mtp_ids.view(1 + i, &[1])?,
            )?;
            if runner_up && i < 2 {
                runner_up_f32(
                    ctx,
                    &pass,
                    &row,
                    &s.sampler,
                    &sp.draft_tokens.view(i, &[1])?,
                    &sp.verify_tokens.view(2 * i, &[2])?,
                )?;
            }
        }
        s.sync.signal_done(&pass)?;
        let pass = pass.end()?;
        first_id.write_bytes(&first.to_le_bytes())?;
        status.write_bytes(bytemuck::cast_slice(&vec![0u32; drafts]))?;
        let pending = pass.commit()?;
        // Dropping the committed chain on an encoding error waits for it.
        let encoded = self.encode_host_verification(ctx, state, s, drafts + 1)?;
        pending.wait_polled()?;
        // Steps before the first nonzero word selected exactly as the host
        // would, so that step saw the host's input: a nonfinite readout there
        // is the host path's failure too. Later words are moot.
        match status.to_u32()?.into_iter().find(|&word| word != 0) {
            None => {}
            Some(word) if word & CHAIN_ARGMAX_TINY != 0 && word != u32::MAX => {
                return Ok(None);
            }
            Some(_) => anyhow::bail!("invalid host draft readout"),
        }
        let proposals = sp.draft_tokens.view(0, &[drafts])?.to_u32()?;
        ensure!(
            proposals.len() == drafts
                && proposals.iter().all(|&t| (t as usize) < vocab),
            "invalid host draft output"
        );
        let mut runners = Vec::new();
        if runner_up {
            let words = sp.verify_tokens.view(0, &[2 * drafts.min(2)])?.to_u32()?;
            runners.extend(
                words.chunks_exact(2).map(|pair| (pair[0], f32::from_bits(pair[1]))),
            );
        }
        Ok(Some(self.pack_host_drafts(ctx, state, s, proposals, encoded, &runners)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_host_steps<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        first: u32,
        drafts: usize,
        sample: &mut dyn FnMut(&[f32]) -> Result<u32>,
        verify: bool,
    ) -> Result<(Vec<u32>, Option<EncodedVerification<'a>>)> {
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP scratch"))?;
        let ids = sp.mtp_ids.view(0, &[1])?;
        let logits = sp.logits.view(0, &[1, self.config.vocab_size])?;
        let mut input = first;
        let mut proposals = Vec::with_capacity(drafts);
        let mut next = Some(self.encode_host_draft_step(ctx, state, s, 0)?);
        let mut encoded_verify = None;
        for i in 0..drafts {
            // The previous pass has completed: the GPU reads neither id.
            ids.write_bytes(&input.to_le_bytes())?;
            let pass = next
                .take()
                .ok_or_else(|| anyhow::anyhow!("host draft step missing"))?
                .commit()?;
            // Encode the next pass while this one runs. If encoding fails,
            // dropping the committed pass waits for the GPU first.
            if i + 1 < drafts {
                next = Some(self.encode_host_draft_step(ctx, state, s, i + 1)?);
            } else if verify {
                encoded_verify =
                    Some(self.encode_host_verification(ctx, state, s, drafts + 1)?);
            }
            pass.wait_polled()?;
            // Borrowed: no later pass that writes this row is committed until
            // the sampler has returned.
            let values: &[f32] = bytemuck::try_cast_slice(logits.contents())
                .map_err(|_| anyhow::anyhow!("invalid host draft readout"))?;
            ensure!(
                values.len() == self.config.vocab_size
                    && crate::finite_guard::all_finite(values),
                "invalid host draft readout"
            );
            input = sample(values)?;
            ensure!(
                (input as usize) < self.config.vocab_size,
                "invalid host draft output"
            );
            proposals.push(input);
        }
        Ok((proposals, encoded_verify))
    }

    /// Stark-local (attempt 20261119-LILYDEPTH-01): writes the draft head's
    /// cache row for the pending input, the row the first `draft_host` step
    /// writes, without its attention read, vocabulary readout or sampling.
    /// A round whose proposals do not come from the head (prompt lookup, or
    /// zero drafts before the end of the stream) calls this before
    /// verification, so `finish_host_verify` leaves the head's caches as
    /// after a head round: every committed position written in this
    /// request, none left from an earlier one.
    pub fn advance_head_host(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        first: u32,
    ) -> Result<()> {
        ensure!(state.spec.is_none() && state.pos > 0, "invalid head advance state");
        ensure!(
            (first as usize) < self.config.vocab_size,
            "invalid head advance input"
        );
        state.ensure_capacity(ctx, state.pos + 1)?;
        self.ensure_prefill_scratch(ctx, s, 1, state.capacity())?;
        let s = &*s;
        let mtp =
            self.weights.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP head"))?;
        let mst = state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP scratch"))?;
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("no head scratch"))?;
        let ps = capacity.rows(1)?;
        let ids = sp.mtp_ids.view(0, &[1])?;
        ids.write_bytes(&first.to_le_bytes())?;
        let hidden = mst.hidden.view(0, &[1, self.config.hc_width()])?;
        let pass = ctx.begin_concurrent()?;
        pass.set_label("host MTP head advance");
        self.mtp_block_caches(
            ctx,
            &pass,
            mtp,
            mst,
            &hidden,
            &ids,
            state.pos - 1,
            s,
            &ps,
            Rope::Delta(state.rope_delta),
            &[],
        )?;
        s.sync.signal_done(&pass)?;
        pass.end()?.commit()?.wait()?;
        Ok(())
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
        self.ensure_prefill_scratch(ctx, s, m, state.capacity())?;
        let pos_before = state.pos;
        let hist_before = state.ple.as_ref().map(|p| p.hist);
        // Decode updates one window in place. Preserve its original in the
        // other slot so rollback can replay only committed projected inputs.
        // Likewise keep each GDN layer's pre-verify state in the spec
        // scratch's shadow: finish_host_verify replays the committed rows
        // from it.
        let slot = state.conv_slot;
        let shadows = &s
            .spec
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no verify captures"))?
            .gdn_shadow;
        let mut copies = Vec::new();
        let mut gdn_index = 0;
        for layer in &state.layers {
            if let LayerState::Gdn { state: recurrent, conv_windows, .. } = layer {
                copies.push(BlitCopy {
                    src: &conv_windows[slot],
                    src_offset: 0,
                    dst: &conv_windows[1 - slot],
                    dst_offset: 0,
                    len: conv_windows[slot].byte_len(),
                });
                let shadow = shadows
                    .get(gdn_index)
                    .ok_or_else(|| anyhow::anyhow!("GDN shadow count"))?;
                ensure!(
                    shadow.byte_len() == recurrent.byte_len(),
                    "GDN shadow geometry"
                );
                copies.push(BlitCopy {
                    src: recurrent,
                    src_offset: 0,
                    dst: shadow,
                    dst_offset: 0,
                    len: recurrent.byte_len(),
                });
                gdn_index += 1;
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
        let verified = HostVerification {
            readout: prefix_rows(&sp.logits, m)?,
            rows: m,
            vocab_size: self.config.vocab_size,
            pos_before,
            hist_before,
            tokens,
            sibling_depth: None,
        };
        // The serial reference keeps validating every row up front.
        let _ = verified.logits()?;
        Ok(verified)
    }

    /// Promote the fork leaf's capture rows to its logical depth. Every
    /// possible early-stop prefix is now contiguous for ordinary rollback.
    pub fn select_host_sibling(
        &self,
        ctx: &MetalContext,
        state: &DecodeState,
        s: &Scratch,
        verified: &mut HostVerification,
    ) -> Result<()> {
        let depth = verified
            .sibling_depth
            .ok_or_else(|| anyhow::anyhow!("missing sibling depth"))?;
        let last = verified.rows - 1;
        ensure!(
            depth > 0
                && depth < last
                && state.pos == verified.pos_before + verified.rows,
            "invalid sibling promotion"
        );
        let sp =
            s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no verify captures"))?;
        let capacity =
            s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("no verify scratch"))?;
        let mut captures = vec![capacity.hyper.clone()];
        captures.extend(sp.conv_in.iter().cloned());
        captures.extend(sp.ple_conv_in.iter().cloned());
        for layer in 0..sp.gdn_replay.len() {
            let (qkv, a, b) = sp.replay_rows(
                layer,
                verified.rows,
                self.config.gdn_conv_channels(),
                self.config.linear_num_value_heads,
            )?;
            captures.extend([qkv, a, b]);
        }
        let copies = captures
            .iter()
            .map(|rows| {
                let width = rows.shape()[1] * rows.dtype().size();
                BlitCopy {
                    src: rows,
                    src_offset: last * width,
                    dst: rows,
                    dst_offset: depth * width,
                    len: width,
                }
            })
            .collect::<Vec<_>>();
        ctx.blit_copy(&copies)?;
        self.relocate_sibling_kv(ctx, state, verified.pos_before, last, depth)?;
        verified.tokens[depth] = verified.tokens[last];
        Ok(())
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
                && (verified.sibling_depth.is_none() || accepted + 1 < m)
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
            (accepted + 1 < m).then_some(Rollback::Replay { accepted, m }),
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
