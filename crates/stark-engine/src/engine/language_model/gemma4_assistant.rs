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
        transformer::TransformerState,
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
    logits: Vec<f32>,
    model_dim: u32,
    capacity: u32,
}
impl<B: Backend> Gemma4AssistantTargetFrame<B> {
    pub fn position(&self) -> u32 {
        self.state.context_length()
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
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
}

impl<B: Backend> LanguageModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
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
        let invalid = Gemma4AssistantSessionError::Invalid;
        if !self.decoder.gemma4_multimodal_supported()
            || self.speculator.is_some()
            || self.decoder.embedding().model_dim() != profile.backbone_dim()
            || input.is_empty()
            || input.len() >= capacity as usize
            || self.max_context_length().is_some_and(|n| capacity > n)
            || input.iter().any(|&t| t >= self.decoder.embedding().vocab_size())
        {
            return Err(invalid);
        }
        let backend = Gemma4AssistantSessionError::Backend;
        if cancelled() {
            return Err(Gemma4AssistantSessionError::Cancelled);
        }
        let mut state = self.decoder.create_empty_state(Some(capacity), &self.engine.context).map_err(backend)?;
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
            logits,
            model_dim,
            capacity,
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
        let local = frame.state.last_committed_attention_kv(true).ok_or(Gemma4AssistantSessionError::Invalid)?;
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
        })
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
        if !Arc::ptr_eq(&self.model_identity, &frame.identity) || !head.matches_target(&self.model_identity) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let backend = Gemma4AssistantSessionError::Backend;
        let local = frame.state.last_committed_attention_kv(true).ok_or(Gemma4AssistantSessionError::Invalid)?;
        let global = frame.state.last_committed_attention_kv(false).ok_or(Gemma4AssistantSessionError::Invalid)?;
        let mut commands =
            self.engine.context.create_command_buffer(Some("private assistant oracle"), None).map_err(backend)?;
        let (lk, lv) = local.validation_copy(&mut commands).map_err(backend)?;
        let (gk, gv) = global.validation_copy(&mut commands).map_err(backend)?;
        let mut hidden = commands.allocate_scratch(frame.normalized.size()).map_err(backend)?;
        commands.encode_copy(&frame.normalized, &mut hidden);
        let ids = commands.allocate_constant_from_slice(&[root]).map_err(backend)?;
        let embedding = self.decoder.embedding().encode_lookup(&ids, 1, &mut commands)?;
        drop(ids);
        let mut embedding_copy = commands.allocate_scratch(embedding.size()).map_err(backend)?;
        commands.encode_copy(&embedding, &mut embedding_copy);
        let step = head.encode_step(&embedding, &frame.normalized, &local, &global, &mut commands)?;
        commands.end_encoding().submit().wait_until_completed().map_err(backend)?;
        let copy = |value: &B::ScratchBuffer| value.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>();
        let centroids = step.centroids.copyout::<bf16>();
        let logits = head.mask_logits(&centroids, &step.logits.copyout::<bf16>())?;
        Ok(serde_json::json!({"position":frame.position(),"root":root,
            "local_rows":local.retained_rows(),"global_rows":global.retained_rows(),
            "local_owner":local.owner_layer(),"global_owner":global.owner_layer(),
            "local_keys":copy(&lk),"local_values":copy(&lv),"global_keys":copy(&gk),"global_values":copy(&gv),
            "embedding":copy(&embedding_copy),"hidden":copy(&hidden),"feedback":copy(&step.feedback),
            "centroids":copy(&step.centroids),"logits":logits}))
    }
}
