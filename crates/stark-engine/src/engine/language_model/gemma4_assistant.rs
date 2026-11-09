//! Source-bound single-position draft component, without serving capability.
//! Target verification, accepted-prefix rollback, head-aware prefix reuse and
//! end-to-end performance qualification must precede native worker integration.
use std::sync::Arc;

use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferCpuAccessible, BufferRef, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, gpu_types::trie::TrieNode as GpuTrieNode,
    },
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::DecoderError,
        embedding::EmbeddingError,
        gemma4_assistant::{Gemma4Assistant, Gemma4AssistantError, Gemma4AssistantProfile},
        mixer::attention::AssistantAttentionHandoff,
        transformer::{TransformerPrefix, TransformerState},
    },
    engine::language_model::LanguageModel,
};

fn flat(length: usize) -> Box<[GpuTrieNode]> {
    (0..length as u32)
        .map(|index| GpuTrieNode {
            trie_start: index,
            trie_end: length as u32 - 1,
            height: index,
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum Gemma4AssistantSessionError<B: Backend> {
    #[error("invalid Gemma assistant target frame")]
    Invalid,
    #[error("Gemma assistant target prefill cancelled")]
    Cancelled,
    #[error("Gemma assistant backend failed")]
    Backend(#[source] B::Error),
    #[error("Gemma assistant target decoder failed")]
    Decoder(#[from] DecoderError<B>),
    #[error("Gemma assistant target embedding failed")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("Gemma assistant draft step failed")]
    Assistant(#[from] Gemma4AssistantError<B>),
}

/// Request-owned text target state; no Debug, serialization or persistence.
pub struct Gemma4AssistantTargetFrame<B: Backend> {
    identity: Arc<()>,
    state: TransformerState<B>,
    normalized: B::GlobalBuffer,
    local_handoff: AssistantAttentionHandoff<B>,
    logits: Vec<f32>,
    model_dim: u32,
    capacity: u32,
    profile: Gemma4AssistantProfile,
    prefix: Option<TransformerPrefix<B>>,
}
impl<B: Backend> Gemma4AssistantTargetFrame<B> {
    pub fn position(&self) -> u32 {
        self.state.context_length()
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn prefix_position(&self) -> Option<usize> {
        self.prefix.as_ref().map(|prefix| prefix.position() as usize)
    }
    pub fn prefix_payload_bytes(&self) -> usize {
        self.prefix.as_ref().map_or(0, TransformerPrefix::payload_bytes)
    }
    pub fn assistant_handoff_bytes(&self) -> usize {
        self.normalized.size() + self.local_handoff.payload_bytes()
    }
}

/// No Debug or serialization: token data remains request-owned.
/// Only `committed_tokens` have been fed to the returned target frame.
/// `next_token` is the next target draw and must be processed by the next round.
pub struct Gemma4AssistantVerifiedRound<B: Backend> {
    pub frame: Gemma4AssistantTargetFrame<B>,
    pub committed_tokens: Vec<u32>,
    pub next_token: u32,
    pub accepted_drafts: usize,
    pub verified_rows: usize,
    pub repaired_rows: usize,
}

pub struct Gemma4AssistantGeneratedRound<B: Backend> {
    pub verified: Gemma4AssistantVerifiedRound<B>,
    pub proposed: usize,
    pub draft_seconds: f64,
    pub verify_seconds: f64,
}

impl<B: Backend> LanguageModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
    pub(crate) fn gemma4_assistant_generation_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        let base = self.generation_memory_bytes_with_text_prefix(input_tokens, context_tokens, max_new_tokens)?;
        let allocated = self.generation_allocation_context_length(input_tokens, context_tokens, max_new_tokens)?;
        let undo = self.decoder.memory_estimate(allocated, input_tokens.max(4))?.prefix_snapshot_bytes;
        // The persistent prefill snapshot is in `base`; the undo snapshot is
        // additional. Bound old/new local handoffs plus materialized assistant
        // scores/readout and four target-logit rows without counting weights.
        base.checked_add(undo)?.checked_add(2 * 513 * 2 * 512 * 2)?.checked_add(32 * 1024 * 1024)
    }
    /// Internal component API. The admitting image wrapper must verify the
    /// complete companion/target artifact pair before constructing a head.
    /// Earlier chunks retain Gemma's cache-only trailing-layer pruning; only
    /// the last prompt row materializes the normalized assistant activation.
    pub fn prefill_gemma4_assistant_target(
        &self,
        input: &[u32],
        capacity: u32,
        profile: Gemma4AssistantProfile,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        self.prefill_gemma4_assistant_target_impl(input, capacity, profile, 0, None, cancelled)
    }

    pub fn prefill_gemma4_assistant_target_with_prefix(
        &self,
        input: &[u32],
        capacity: u32,
        profile: Gemma4AssistantProfile,
        capture_position: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        self.prefill_gemma4_assistant_target_impl(input, capacity, profile, capture_position, None, cancelled)
    }

    /// Restore only the saved full prefill boundary, then recompute the suffix
    /// and its final normalized activation and source local-KV handoff.
    pub fn restore_and_prefill_gemma4_assistant_target(
        &self,
        frame: Gemma4AssistantTargetFrame<B>,
        suffix: &[u32],
        capture_position: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        let capacity = frame.capacity;
        let profile = frame.profile;
        self.prefill_gemma4_assistant_target_impl(suffix, capacity, profile, capture_position, Some(frame), cancelled)
    }

    fn prefill_gemma4_assistant_target_impl(
        &self,
        input: &[u32],
        capacity: u32,
        profile: Gemma4AssistantProfile,
        capture_position: usize,
        restored: Option<Gemma4AssistantTargetFrame<B>>,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        let invalid = Gemma4AssistantSessionError::Invalid;
        let start = restored.as_ref().and_then(Gemma4AssistantTargetFrame::prefix_position).unwrap_or(0);
        if !self.decoder.gemma4_multimodal_supported()
            || self.speculator.is_some()
            || self.decoder.embedding().model_dim() != profile.backbone_dim()
            || input.is_empty()
            || start.checked_add(input.len()).is_none_or(|end| end >= capacity as usize)
            || capture_position % super::stream::PREFILL_CHUNK_TOKENS != 0
            || capture_position >= start + input.len()
            || self.max_context_length().is_some_and(|n| capacity > n)
            || input.iter().any(|&t| t >= self.decoder.embedding().vocab_size())
        {
            return Err(invalid);
        }
        let backend = Gemma4AssistantSessionError::Backend;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        let (mut state, mut saved_prefix) = if let Some(mut frame) = restored {
            if !Arc::ptr_eq(&self.model_identity, &frame.identity)
                || frame.capacity != capacity
                || frame.model_dim != profile.backbone_dim()
            {
                return Err(Gemma4AssistantSessionError::Invalid);
            }
            let saved = frame.prefix.take().ok_or(Gemma4AssistantSessionError::Invalid)?;
            let mut command = self
                .engine
                .context
                .create_command_buffer(Some("Gemma assistant prefix restoration"), None)
                .map_err(backend)?;
            if !frame.state.restore_prefix(&saved, &mut command) {
                return Err(Gemma4AssistantSessionError::Invalid);
            }
            command.end_encoding().submit().wait_until_completed().map_err(backend)?;
            if cancelled() {
                return Err(Gemma4AssistantSessionError::Cancelled);
            }
            (frame.state, Some(saved))
        } else {
            (self.decoder.create_empty_state(Some(capacity), &self.engine.context).map_err(backend)?, None)
        };
        let prefix = &input[..input.len() - 1];
        for chunk in prefix.chunks(super::stream::PREFILL_CHUNK_TOKENS) {
            if cancelled() {
                return Err(Gemma4AssistantSessionError::Cancelled);
            }
            state.prepare(state.context_length(), chunk.len() as u32).map_err(backend)?;
            let mut command = self
                .engine
                .context
                .create_command_buffer(Some("Gemma assistant target prefill"), None)
                .map_err(backend)?;
            let ids = command.allocate_constant_from_slice(chunk).map_err(backend)?;
            let nodes = flat(chunk.len());
            let batch = BatchTopology::new(&nodes, true);
            self.decoder.encode(&ids, &batch, None, None, &mut state, &mut command)?;
            drop(ids);
            state.encode_accept(&(0..chunk.len() as u32).collect::<Vec<_>>(), &mut command).map_err(backend)?;
            command.end_encoding().submit().wait_until_completed().map_err(backend)?;
            if capture_position > 0 && state.context_length() as usize == capture_position {
                let mut capture = self
                    .engine
                    .context
                    .create_command_buffer(Some("Gemma assistant prefill boundary snapshot"), None)
                    .map_err(backend)?;
                saved_prefix = state.capture_prefix(&self.engine.context, &mut capture).map_err(backend)?;
                if saved_prefix.is_none() {
                    return Err(Gemma4AssistantSessionError::Invalid);
                }
                capture.end_encoding().submit().wait_until_completed().map_err(backend)?;
            }
        }
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        state.prepare(state.context_length(), 1).map_err(backend)?;
        let mut command = self
            .engine
            .context
            .create_command_buffer(Some("Gemma assistant target final row"), None)
            .map_err(backend)?;
        let ids = command.allocate_constant_from_slice(&input[input.len() - 1..]).map_err(backend)?;
        let nodes = flat(1);
        let batch = BatchTopology::new(&nodes, true);
        let output = self.decoder.encode(&ids, &batch, Some((0..1).into()), Some(&[]), &mut state, &mut command)?;
        drop(ids);
        let hidden = output.final_hidden.ok_or(Gemma4AssistantSessionError::Invalid)?;
        let logits = output.logits.ok_or(Gemma4AssistantSessionError::Invalid)?;
        let model_dim = profile.backbone_dim();
        if hidden.size() != model_dim as usize * 2 {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let mut normalized = self.engine.context.create_buffer(hidden.size()).map_err(backend)?;
        command.encode_copy(&hidden, &mut normalized);
        let local_handoff = state
            .capture_assistant_local_handoff(1, &self.engine.context, &mut command)
            .map_err(backend)?
            .ok_or(Gemma4AssistantSessionError::Invalid)?;
        state.encode_accept(&[0], &mut command).map_err(backend)?;
        command.end_encoding().submit().wait_until_completed().map_err(backend)?;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        let logits = logits.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>();
        if logits.iter().any(|n| !n.is_finite()) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        Ok(Gemma4AssistantTargetFrame {
            identity: self.model_identity.clone(),
            state,
            normalized,
            local_handoff,
            logits,
            model_dim,
            capacity,
            profile,
            prefix: saved_prefix,
        })
    }

    /// Real greedy proposals only; this does not verify or commit target rows.
    /// Keeping the immutable frame borrowed fixes KV and RoPE through the round.
    /// Positions beyond 512 require the unimplemented source 513-row handoff.
    pub fn draft_gemma4_assistant_greedy(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        first: u32,
        count: usize,
    ) -> Result<Vec<u32>, Gemma4AssistantSessionError<B>> {
        self.draft_gemma4_assistant_greedy_controlled(frame, head, first, count, &mut || false)
    }

    pub fn draft_gemma4_assistant_greedy_controlled(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        first: u32,
        count: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<u32>, Gemma4AssistantSessionError<B>> {
        if !Arc::ptr_eq(&self.model_identity, &frame.identity)
            || !head.matches_target(&self.model_identity)
            || frame.position() > 512
            || head.backbone_dim() != frame.model_dim
            || count > 3
            || first >= self.decoder.embedding().vocab_size()
            || frame.position().checked_add(count as u32 + 1).is_none_or(|n| n > frame.capacity)
        {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let local = frame
            .state
            .last_committed_attention_kv(true)
            .and_then(|view| view.with_assistant_handoff(&frame.local_handoff))
            .ok_or(Gemma4AssistantSessionError::Invalid)?;
        let global = frame.state.last_committed_attention_kv(false).ok_or(Gemma4AssistantSessionError::Invalid)?;
        let backend = Gemma4AssistantSessionError::Backend;
        let mut feedback = None;
        let mut root = first;
        let mut tokens = Vec::new();
        for _ in 0..count {
            if cancelled() {
                return Err(Gemma4AssistantSessionError::Cancelled);
            }
            let mut command = self
                .engine
                .context
                .create_command_buffer(Some("Gemma assistant single-position draft"), None)
                .map_err(backend)?;
            let ids = command.allocate_constant_from_slice(&[root]).map_err(backend)?;
            let embedding = self.decoder.embedding().encode_lookup(&ids, 1, &mut command)?;
            drop(ids);
            let step = head.encode_step(
                &embedding,
                feedback.as_ref().unwrap_or(&frame.normalized),
                &local,
                &global,
                &mut command,
            )?;
            let mut next = self.engine.context.create_buffer(step.feedback.size()).map_err(backend)?;
            command.encode_copy(&step.feedback, &mut next);
            command.end_encoding().submit().wait_until_completed().map_err(backend)?;
            if cancelled() {
                return Err(Gemma4AssistantSessionError::Cancelled);
            }
            let logits = head.mask_logits(&step.centroids.copyout::<bf16>(), &step.logits.copyout::<bf16>())?;
            root = logits
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .map(|(i, _)| i as u32)
                .ok_or(Gemma4AssistantSessionError::Invalid)?;
            tokens.push(root);
            feedback = Some(next);
        }
        Ok(tokens)
    }
    /// Correctness component only: verifies each row before feeding it back.
    /// Consuming the frame makes every error/cancellation discard advanced KV.
    /// This performs no partially accepted ring batch and establishes no speedup.
    pub fn verify_gemma4_assistant_greedy_serial(
        &self,
        mut frame: Gemma4AssistantTargetFrame<B>,
        drafts: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantVerifiedRound<B>, Gemma4AssistantSessionError<B>> {
        if !Arc::ptr_eq(&self.model_identity, &frame.identity)
            || drafts.len() > 3
            || drafts.iter().any(|&t| t >= self.decoder.embedding().vocab_size())
            || frame.position().checked_add(drafts.len() as u32 + 1).is_none_or(|n| n > frame.capacity)
        {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let draw = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .map(|(i, _)| i as u32)
                .ok_or(Gemma4AssistantSessionError::Invalid)
        };
        let mut emitted = vec![draw(&frame.logits)?];
        let mut accepted = 0;
        for index in 0..=drafts.len() {
            self.advance_gemma4_assistant_frame(&mut frame, *emitted.last().unwrap(), cancelled)?;
            let next = draw(&frame.logits)?;
            emitted.push(next);
            if drafts.get(index).is_none_or(|&proposal| proposal != next) {
                break;
            }
            accepted += 1;
        }
        let next_token = emitted.pop().ok_or(Gemma4AssistantSessionError::Invalid)?;
        Ok(Gemma4AssistantVerifiedRound {
            frame,
            committed_tokens: emitted,
            next_token,
            accepted_drafts: accepted,
            verified_rows: accepted + 1,
            repaired_rows: 0,
        })
    }

    /// Transactional full-accept target batch. Rejected speculation is first
    /// completely restored, then repaired through serial committed inputs.
    /// Ordinary partial-ring acceptance is never requested or relaxed.
    pub fn verify_gemma4_assistant_greedy_batch(
        &self,
        mut frame: Gemma4AssistantTargetFrame<B>,
        drafts: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantVerifiedRound<B>, Gemma4AssistantSessionError<B>> {
        if !Arc::ptr_eq(&self.model_identity, &frame.identity)
            || drafts.len() > 3
            || drafts.iter().any(|&token| token >= self.decoder.embedding().vocab_size())
            || frame.position().checked_add(drafts.len() as u32 + 1).is_none_or(|end| end > frame.capacity)
        {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        if drafts.is_empty() {
            return self.verify_gemma4_assistant_greedy_serial(frame, drafts, cancelled);
        }
        let draw = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .map(|(index, _)| index as u32)
                .ok_or(Gemma4AssistantSessionError::Invalid)
        };
        let inputs = std::iter::once(draw(&frame.logits)?).chain(drafts.iter().copied()).collect::<Vec<_>>();
        let rows = inputs.len();
        let backend = Gemma4AssistantSessionError::Backend;
        frame.state.prepare(frame.position(), rows as u32).map_err(backend)?;
        let mut command = self
            .engine
            .context
            .create_command_buffer(Some("Gemma assistant transactional full target batch"), None)
            .map_err(backend)?;
        let snapshot = frame
            .state
            .capture_prefix(&self.engine.context, &mut command)
            .map_err(backend)?
            .ok_or(Gemma4AssistantSessionError::Invalid)?;
        let ids = command.allocate_constant_from_slice(&inputs).map_err(backend)?;
        let nodes = flat(rows);
        let batch = BatchTopology::new(&nodes, true);
        let output = self.decoder.encode(
            &ids,
            &batch,
            Some((0..rows as u32).into()),
            Some(&[]),
            &mut frame.state,
            &mut command,
        )?;
        drop(ids);
        let hidden = output.final_hidden.ok_or(Gemma4AssistantSessionError::Invalid)?;
        let logits = output.logits.ok_or(Gemma4AssistantSessionError::Invalid)?;
        let hidden_row_bytes = frame.model_dim as usize * 2;
        let vocabulary = self.decoder.embedding().vocab_size() as usize;
        if hidden.size() != rows * hidden_row_bytes || logits.size() != rows * vocabulary * 2 {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let mut normalized = self.engine.context.create_buffer(hidden_row_bytes).map_err(backend)?;
        command.encode_copy(hidden.subrange((rows - 1) * hidden_row_bytes..), &mut normalized);
        let local_handoff = frame
            .state
            .capture_assistant_local_handoff(rows as u32, &self.engine.context, &mut command)
            .map_err(backend)?
            .ok_or(Gemma4AssistantSessionError::Invalid)?;
        frame.state.encode_accept(&(0..rows as u32).collect::<Vec<_>>(), &mut command).map_err(backend)?;
        command.end_encoding().submit().wait_until_completed().map_err(backend)?;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        let logits = logits.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>();
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let mut accepted = 0;
        while accepted < drafts.len()
            && draw(&logits[accepted * vocabulary..(accepted + 1) * vocabulary])? == drafts[accepted]
        {
            accepted += 1;
        }
        let next_token = draw(&logits[accepted * vocabulary..(accepted + 1) * vocabulary])?;
        if accepted == drafts.len() {
            frame.normalized = normalized;
            frame.local_handoff = local_handoff;
            frame.logits = logits[(rows - 1) * vocabulary..].to_vec();
            return Ok(Gemma4AssistantVerifiedRound {
                frame,
                committed_tokens: inputs,
                next_token,
                accepted_drafts: accepted,
                verified_rows: rows,
                repaired_rows: 0,
            });
        }
        let mut restore = self
            .engine
            .context
            .create_command_buffer(Some("Gemma assistant full target snapshot restoration"), None)
            .map_err(backend)?;
        if !frame.state.restore_prefix(&snapshot, &mut restore) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        restore.end_encoding().submit().wait_until_completed().map_err(backend)?;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        let mut repaired = self.verify_gemma4_assistant_greedy_serial(frame, &drafts[..accepted], cancelled)?;
        if repaired.accepted_drafts != accepted
            || repaired.committed_tokens != inputs[..accepted + 1]
            || repaired.next_token != next_token
        {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        repaired.repaired_rows = accepted + 1;
        repaired.verified_rows += rows;
        Ok(repaired)
    }

    fn advance_gemma4_assistant_frame(
        &self,
        frame: &mut Gemma4AssistantTargetFrame<B>,
        token: u32,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), Gemma4AssistantSessionError<B>> {
        let backend = Gemma4AssistantSessionError::Backend;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        if frame.position() >= frame.capacity || token >= self.decoder.embedding().vocab_size() {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        frame.state.prepare(frame.position(), 1).map_err(backend)?;
        let mut command = self
            .engine
            .context
            .create_command_buffer(Some("Gemma assistant serial target verification"), None)
            .map_err(backend)?;
        let ids = command.allocate_constant_from_slice(&[token]).map_err(backend)?;
        let nodes = flat(1);
        let batch = BatchTopology::new(&nodes, true);
        let output =
            self.decoder.encode(&ids, &batch, Some((0..1).into()), Some(&[]), &mut frame.state, &mut command)?;
        drop(ids);
        let hidden = output.final_hidden.ok_or(Gemma4AssistantSessionError::Invalid)?;
        let logits = output.logits.ok_or(Gemma4AssistantSessionError::Invalid)?;
        if hidden.size() != frame.model_dim as usize * 2 {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let mut normalized = self.engine.context.create_buffer(hidden.size()).map_err(backend)?;
        command.encode_copy(&hidden, &mut normalized);
        let local_handoff = frame
            .state
            .capture_assistant_local_handoff(1, &self.engine.context, &mut command)
            .map_err(backend)?
            .ok_or(Gemma4AssistantSessionError::Invalid)?;
        frame.state.encode_accept(&[0], &mut command).map_err(backend)?;
        command.end_encoding().submit().wait_until_completed().map_err(backend)?;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        let logits = logits.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>();
        if logits.iter().any(|n| !n.is_finite()) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        frame.normalized = normalized;
        frame.local_handoff = local_handoff;
        frame.logits = logits;
        Ok(())
    }

    /// Ephemeral test-only tensors passed directly to a local reference process.
    /// Callers must never persist or print this value.
    #[cfg(test)]
    pub(crate) fn assistant_validation_payload(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        root: u32,
    ) -> Result<serde_json::Value, Gemma4AssistantSessionError<B>> {
        self.assistant_validation_payload_inner(frame, head, root, None, false)
    }

    #[cfg(test)]
    pub(crate) fn assistant_validation_round_payload(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        root: u32,
    ) -> Result<serde_json::Value, Gemma4AssistantSessionError<B>> {
        self.assistant_validation_round_payload_inner(frame, head, root, false)
    }

    #[cfg(test)]
    pub(crate) fn assistant_validation_trace_payload(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        root: u32,
    ) -> Result<serde_json::Value, Gemma4AssistantSessionError<B>> {
        self.assistant_validation_round_payload_inner(frame, head, root, true)
    }

    #[cfg(test)]
    fn assistant_validation_round_payload_inner(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        mut root: u32,
        trace: bool,
    ) -> Result<serde_json::Value, Gemma4AssistantSessionError<B>> {
        let mut feedback: Option<Vec<bf16>> = None;
        let mut steps = Vec::new();
        for _ in 0..3 {
            let payload = self.assistant_validation_payload_inner(frame, head, root, feedback.as_deref(), trace)?;
            let logits = payload["logits"].as_array().ok_or(Gemma4AssistantSessionError::Invalid)?;
            root = logits
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.as_f64().unwrap().total_cmp(&y.as_f64().unwrap()).then_with(|| b.cmp(a)))
                .map(|(index, _)| index as u32)
                .ok_or(Gemma4AssistantSessionError::Invalid)?;
            feedback = Some(
                payload["feedback"]
                    .as_array()
                    .ok_or(Gemma4AssistantSessionError::Invalid)?
                    .iter()
                    .map(|value| bf16::from_f32(value.as_f64().unwrap() as f32))
                    .collect(),
            );
            steps.push(payload);
        }
        Ok(serde_json::json!({"steps":steps}))
    }

    #[cfg(test)]
    fn assistant_validation_payload_inner(
        &self,
        frame: &Gemma4AssistantTargetFrame<B>,
        head: &Gemma4Assistant<B>,
        root: u32,
        feedback: Option<&[bf16]>,
        trace: bool,
    ) -> Result<serde_json::Value, Gemma4AssistantSessionError<B>> {
        if !Arc::ptr_eq(&self.model_identity, &frame.identity) || !head.matches_target(&self.model_identity) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let backend = Gemma4AssistantSessionError::Backend;
        let local = frame
            .state
            .last_committed_attention_kv(true)
            .and_then(|view| view.with_assistant_handoff(&frame.local_handoff))
            .ok_or(Gemma4AssistantSessionError::Invalid)?;
        let global = frame.state.last_committed_attention_kv(false).ok_or(Gemma4AssistantSessionError::Invalid)?;
        let mut commands =
            self.engine.context.create_command_buffer(Some("private assistant oracle"), None).map_err(backend)?;
        let (lk, lv) = local.validation_copy(&mut commands).map_err(backend)?;
        let (gk, gv) = global.validation_copy(&mut commands).map_err(backend)?;
        let mut hidden = commands.allocate_scratch(frame.normalized.size()).map_err(backend)?;
        if let Some(feedback) = feedback {
            if feedback.len() * 2 != frame.normalized.size() {
                return Err(Gemma4AssistantSessionError::Invalid);
            }
            let source = commands.allocate_constant_from_slice(feedback).map_err(backend)?;
            commands.encode_copy(&source, &mut hidden);
            drop(source);
        } else {
            commands.encode_copy(&frame.normalized, &mut hidden);
        }
        let ids = commands.allocate_constant_from_slice(&[root]).map_err(backend)?;
        let embedding = self.decoder.embedding().encode_lookup(&ids, 1, &mut commands)?;
        drop(ids);
        let mut embedding_copy = commands.allocate_scratch(embedding.size()).map_err(backend)?;
        commands.encode_copy(&embedding, &mut embedding_copy);
        let step = head.encode_step(&embedding, &hidden, &local, &global, &mut commands)?;
        let traced = if trace {
            Some(head.encode_step_traced(&embedding, &hidden, &local, &global, &mut commands)?)
        } else {
            None
        };
        commands.end_encoding().submit().wait_until_completed().map_err(backend)?;
        let copy = |value: &B::ScratchBuffer| value.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>();
        let centroids = step.centroids.copyout::<bf16>();
        let logits = head.mask_logits(&centroids, &step.logits.copyout::<bf16>())?;
        let mut payload = serde_json::json!({"position":frame.position(),"root":root,
            "local_rows":local.retained_rows(),"global_rows":global.retained_rows(),
            "local_owner":local.owner_layer(),"global_owner":global.owner_layer(),
            "local_keys":copy(&lk),"local_values":copy(&lv),"global_keys":copy(&gk),"global_values":copy(&gv),
            "embedding":copy(&embedding_copy),"hidden":copy(&hidden),"feedback":copy(&step.feedback),
            "centroids":copy(&step.centroids),"logits":logits});
        if let Some((traced_step, stages)) = traced {
            let equal = step.feedback.copyout::<bf16>() == traced_step.feedback.copyout::<bf16>()
                && step.centroids.copyout::<bf16>() == traced_step.centroids.copyout::<bf16>()
                && step.logits.copyout::<bf16>() == traced_step.logits.copyout::<bf16>();
            let trace = stages
                .iter()
                .map(|(name, buffer)| (name.clone(), serde_json::json!(copy(buffer))))
                .collect::<serde_json::Map<_, _>>();
            payload["trace"] = serde_json::Value::Object(trace);
            payload["unmasked_logits"] = serde_json::json!(copy(&step.logits));
            payload["instrumented_output_equal"] = serde_json::json!(equal);
        }
        Ok(payload)
    }
}
