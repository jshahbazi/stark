//! Dense Qwen MTP sessions. Only target draws commit output; draft KV rows are
//! rewound and rebuilt from output-normalized target residuals after every
//! round, matching the pinned oMLX driver's Qwen head-input policy.
use std::{sync::Arc, time::Instant};

use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferCpuAccessible, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted,
        CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode as GpuTrieNode,
    },
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::DecoderError,
        embedding::EmbeddingError,
        qwen_mtp::QwenMtpError,
        transformer::{TransformerPrefix, TransformerState},
    },
    engine::{hybrid::MtpSampler, language_model::LanguageModel},
};

#[cfg(all(test, backend = "metal"))]
#[path = "dense_mtp_head_stage_tests.rs"]
mod head_stage_tests;

#[derive(Debug, Error)]
pub enum DenseMtpError<B: Backend> {
    #[error("invalid dense MTP request or state")]
    Invalid,
    #[error("dense MTP request cancelled")]
    Cancelled,
    #[error("dense MTP backend failed: {0}")]
    Backend(#[source] B::Error),
    #[error("dense MTP target failed: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("dense MTP embedding failed: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("dense MTP predictor failed: {0}")]
    Predictor(#[from] QwenMtpError<B>),
    #[error("dense MTP sampling failed")]
    Sampling,
}

pub struct DenseMtpSession<B: Backend> {
    identity: Arc<()>,
    pub(crate) target: TransformerState<B>,
    pub(crate) predictor: TransformerState<B>,
    pub(crate) last_hidden: B::GlobalBuffer,
    logits: Vec<f32>,
    capacity: usize,
    failed: bool,
    prefix: Option<DenseMtpPrefix<B>>,
}

struct DenseMtpPrefix<B: Backend> {
    target: TransformerPrefix<B>,
    predictor: TransformerPrefix<B>,
    hidden: B::GlobalBuffer,
    logits: Vec<f32>,
    position: usize,
}

impl<B: Backend> DenseMtpSession<B> {
    pub fn context_capacity(&self) -> usize {
        self.capacity
    }

    pub fn resident_payload_bytes(&self) -> Option<usize> {
        self.target
            .resident_payload_bytes()?
            .checked_add(self.predictor.resident_payload_bytes()?)?
            .checked_add(self.last_hidden.size())?
            .checked_add(self.logits.len().checked_mul(std::mem::size_of::<f32>())?)?
            .checked_add(self.prefix_payload_bytes())
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn position(&self) -> usize {
        self.target.context_length() as usize
    }
    pub fn prefix_position(&self) -> Option<usize> {
        self.prefix.as_ref().map(|p| p.position)
    }
    pub fn is_fenced(&self) -> bool {
        self.failed
    }
    pub fn prefix_payload_bytes(&self) -> usize {
        self.prefix.as_ref().map_or(0, |p| {
            p.target.payload_bytes()
                + p.predictor.payload_bytes()
                + p.hidden.size()
                + p.logits.len() * std::mem::size_of::<f32>()
        })
    }
}

pub struct DenseMtpRound {
    pub accepted_drafts: Vec<u32>,
    pub next_token: u32,
    pub proposed: usize,
    pub accepted: usize,
    pub verified_rows: usize,
    pub draft_seconds: f64,
    pub verify_seconds: f64,
    pub rollback_seconds: f64,
    pub target_encode_seconds: f64,
    pub target_gpu_seconds: f64,
    pub target_readout_seconds: f64,
}

struct TargetPass<B: Backend> {
    logits: Vec<f32>,
    hidden: B::GlobalBuffer,
    encode_seconds: f64,
    gpu_seconds: f64,
    readout_seconds: f64,
}

struct PredictorPass<B: Backend> {
    logits: Vec<f32>,
    hidden: Option<B::GlobalBuffer>,
}

fn chain(
    length: usize,
    full_accept: bool,
) -> (Box<[GpuTrieNode]>, bool) {
    (
        (0..length as u32)
            .map(|index| GpuTrieNode {
                trie_start: index,
                trie_end: length as u32 - 1,
                height: index,
            })
            .collect(),
        full_accept,
    )
}

impl<B: Backend> LanguageModel<B> {
    pub fn has_dense_mtp(&self) -> bool {
        // A validated dense head can also verify text serially on an image
        // decoder. Multimodal positions and reference recurrent kernels never
        // enter the non-flat, partially accepted speculative target path.
        self.dense_mtp.is_some() && self.decoder.prefix_snapshot_supported()
    }

    pub fn generation_memory_bytes_reusing_dense_mtp(
        &self,
        session: &DenseMtpSession<B>,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
        reused_input_tokens: u32,
    ) -> Option<usize> {
        if session.failed
            || !Arc::ptr_eq(&self.model_identity, &session.identity)
            || session.capacity != context_tokens as usize
            || reused_input_tokens == 0
            || reused_input_tokens >= input_tokens
            || reused_input_tokens as usize > session.position()
            || !(reused_input_tokens as usize).is_multiple_of(super::stream::PREFILL_CHUNK_TOKENS)
            || session.prefix_position() != Some(reused_input_tokens as usize)
        {
            return None;
        }
        let mut estimate = self.generation_memory_estimate_for_prefill(
            input_tokens,
            context_tokens,
            max_new_tokens,
            true,
            input_tokens - reused_input_tokens,
            false,
        )?;
        let allocated = self.generation_allocation_context_length(input_tokens, context_tokens, max_new_tokens)?;
        let suffix = input_tokens - reused_input_tokens;
        let head = self.dense_mtp.as_ref()?;
        let target_limit = self.decoder.persistent_state_bytes(allocated, suffix.max(4), input_tokens.max(4), false)?;
        let predictor_limit = head.persistent_state_bytes(allocated, suffix, input_tokens)?;
        let mut credit = session
            .target
            .resident_payload_bytes()?
            .min(target_limit)
            .checked_add(session.predictor.resident_payload_bytes()?.min(predictor_limit))?;
        let row_bytes = (self.decoder.embedding().model_dim() as usize).checked_mul(2)?;
        credit = credit.checked_add(session.last_hidden.size().min(row_bytes))?;
        let mut logits = session.logits.len().checked_mul(std::mem::size_of::<f32>())?;
        if let Some(prefix) = &session.prefix {
            let snapshot_limit = self
                .decoder
                .memory_estimate(allocated, suffix)?
                .prefix_snapshot_bytes
                .checked_add(head.memory_estimate(allocated, suffix)?.prefix_snapshot_bytes)?;
            let snapshots = prefix.target.payload_bytes().checked_add(prefix.predictor.payload_bytes())?;
            let boundary = input_tokens.saturating_sub(1) as usize / super::stream::PREFILL_CHUNK_TOKENS
                * super::stream::PREFILL_CHUNK_TOKENS;
            if prefix.position != boundary {
                estimate.state_bytes = estimate.state_bytes.checked_add(snapshot_limit)?;
            }
            credit =
                credit.checked_add(snapshots.min(snapshot_limit))?.checked_add(prefix.hidden.size().min(row_bytes))?;
            logits = logits.checked_add(prefix.logits.len().checked_mul(std::mem::size_of::<f32>())?)?;
        }
        let logits_limit = (self.decoder.embedding().vocab_size() as usize).checked_mul(4)?.checked_mul(8)?;
        credit = credit.checked_add(logits.min(logits_limit))?;
        super::additional_generation_bytes(estimate, credit)
    }
}

impl<B: Backend> LanguageModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
    fn dense_target_pass(
        &self,
        ids: &[u32],
        state: &mut TransformerState<B>,
        full: bool,
    ) -> Result<TargetPass<B>, DenseMtpError<B>> {
        let encode_started = Instant::now();
        let head = self.dense_mtp.as_ref().ok_or(DenseMtpError::Invalid)?;
        state.prepare(state.context_length(), ids.len() as u32).map_err(DenseMtpError::Backend)?;
        let pool = self.engine.context.create_allocation_pool();
        let mut cmd = self
            .engine
            .context
            .create_command_buffer(Some("dense MTP target"), Some(pool))
            .map_err(DenseMtpError::Backend)?;
        // Match ordinary stream/scoring embedding gathers: device rows are u32
        // even though LanguageModelState keeps its host token history as u64.
        let tokens = cmd.allocate_constant_from_slice(ids).map_err(DenseMtpError::Backend)?;
        let (nodes, full) = chain(ids.len(), full);
        let batch = BatchTopology::new(&nodes, full);
        // Prefill needs every residual to prime the predictor, but only its
        // final vocabulary row can seed generation or a retained boundary.
        let readout_start = if full {
            ids.len() as u32 - 1
        } else {
            0
        };
        let encoded = self.decoder.encode(
            &tokens,
            &batch,
            Some((readout_start..ids.len() as u32).into()),
            Some(&[head.target_layer]),
            state,
            &mut cmd,
        )?;
        let hidden =
            encoded.hidden_features.ok_or(DenseMtpError::Invalid)?.into_vec().pop().ok_or(DenseMtpError::Invalid)?;
        // oMLX batch_generator._head_input uses the final trunk norm on Qwen
        // hidden rows (HEAD_HIDDEN_POST_NORM=True), for both priming and decode.
        // The vocabulary logits above retain their existing execution path.
        let hidden = self
            .decoder
            .normalize_captured_output(&hidden, ids.len() as u32, &mut cmd)
            .map_err(DenseMtpError::Backend)?;
        let mut saved = self.engine.context.create_buffer(hidden.size()).map_err(DenseMtpError::Backend)?;
        cmd.encode_copy(&hidden, &mut saved);
        if full {
            state
                .encode_accept(&(0..ids.len() as u32).collect::<Vec<_>>(), &mut cmd)
                .map_err(DenseMtpError::Backend)?;
        }
        let logits = encoded.logits.ok_or(DenseMtpError::Invalid)?;
        drop(tokens);
        let encode_seconds = encode_started.elapsed().as_secs_f64();
        let complete = cmd.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        let gpu_seconds = complete.gpu_execution_time().as_secs_f64();
        let readout_started = Instant::now();
        let values = logits.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>();
        let readout_seconds = readout_started.elapsed().as_secs_f64();
        drop(complete);
        if values.iter().any(|x| !x.is_finite()) {
            return Err(DenseMtpError::Invalid);
        }
        Ok(TargetPass {
            logits: values,
            hidden: saved,
            encode_seconds,
            gpu_seconds,
            readout_seconds,
        })
    }

    fn dense_predictor_pass(
        &self,
        ids: &[u32],
        hidden: impl BufferRef<Backend = B>,
        state: &mut TransformerState<B>,
        read_logits: bool,
    ) -> Result<PredictorPass<B>, DenseMtpError<B>> {
        let head = self.dense_mtp.as_ref().ok_or(DenseMtpError::Invalid)?;
        state.prepare(state.context_length(), ids.len() as u32).map_err(DenseMtpError::Backend)?;
        let pool = self.engine.context.create_allocation_pool();
        let mut cmd = self
            .engine
            .context
            .create_command_buffer(Some("dense MTP predictor"), Some(pool))
            .map_err(DenseMtpError::Backend)?;
        let tokens = cmd.allocate_constant_from_slice(ids).map_err(DenseMtpError::Backend)?;
        let embedded = self.decoder.embedding().encode_lookup(&tokens, ids.len() as u32, &mut cmd)?;
        let (nodes, full) = chain(ids.len(), true);
        let batch = BatchTopology::new(&nodes, full);
        let encoded = head.encode(&embedded, hidden, &batch, state, &mut cmd)?;
        state.encode_accept(&(0..ids.len() as u32).collect::<Vec<_>>(), &mut cmd).map_err(DenseMtpError::Backend)?;
        // oMLX mtp_forward returns MTPModule's post-output-norm hidden to the
        // next draft step. Priming/repair consumes only predictor state.
        let saved = if read_logits {
            let mut saved =
                self.engine.context.create_buffer(encoded.normalized.size()).map_err(DenseMtpError::Backend)?;
            cmd.encode_copy(&encoded.normalized, &mut saved);
            Some(saved)
        } else {
            None
        };
        let mut shortlist = None;
        let logits = if read_logits {
            if let Some(coarse) = &self.dense_draft_readout {
                if ids.len() != 1 {
                    return Err(DenseMtpError::Invalid);
                }
                let (candidates, exact) = coarse.encode(self.decoder.embedding(), &encoded.normalized, &mut cmd)?;
                shortlist = Some(candidates);
                Some(exact)
            } else {
                Some(self.decoder.embedding().encode_readout(
                    ids.len() as u32,
                    &encoded.normalized,
                    self.decoder.embedding().vocab_size(),
                    None::<&B::ScratchBuffer>,
                    true,
                    &mut cmd,
                )?)
            }
        } else {
            None
        };
        drop(tokens);
        let complete = cmd.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        let mut values =
            logits.map(|x| x.copyout::<bf16>().into_iter().map(f32::from).collect::<Vec<_>>()).unwrap_or_default();
        if let Some(shortlist) = shortlist {
            let candidates = shortlist.copyout::<u32>();
            if candidates.len() != 64 || values.len() != 64 {
                return Err(DenseMtpError::Invalid);
            }
            let vocab = self.decoder.embedding().vocab_size() as usize;
            let mut expanded = vec![f32::MIN; vocab];
            let mut seen = std::collections::HashSet::with_capacity(candidates.len());
            for (candidate, value) in candidates.into_iter().zip(values) {
                if candidate as usize >= vocab || !value.is_finite() || !seen.insert(candidate) {
                    return Err(DenseMtpError::Invalid);
                }
                expanded[candidate as usize] = value;
            }
            values = expanded;
        }
        drop(complete);
        if values.iter().any(|x| !x.is_finite()) {
            return Err(DenseMtpError::Invalid);
        }
        Ok(PredictorPass {
            logits: values,
            hidden: saved,
        })
    }

    fn encode_dense_predictor_cache_only(
        &self,
        ids: &[u32],
        target_hidden: impl BufferRef<Backend = B>,
        state: &mut TransformerState<B>,
        cmd: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), DenseMtpError<B>> {
        let head = self.dense_mtp.as_ref().ok_or(DenseMtpError::Invalid)?;
        let row_bytes = self.decoder.embedding().model_dim() as usize * std::mem::size_of::<bf16>();
        let hidden_bytes = ids.len().checked_mul(row_bytes).ok_or(DenseMtpError::Invalid)?;
        if ids.is_empty() || target_hidden.size() != hidden_bytes {
            return Err(DenseMtpError::Invalid);
        }
        let tokens = cmd.allocate_constant_from_slice(ids).map_err(DenseMtpError::Backend)?;
        let embedded = self.decoder.embedding().encode_lookup(&tokens, ids.len() as u32, cmd)?;
        let (nodes, full) = chain(ids.len(), true);
        let batch = BatchTopology::new(&nodes, full);
        head.encode_cache_only(&embedded, target_hidden, &batch, state, cmd)?;
        state.encode_accept(&(0..ids.len() as u32).collect::<Vec<_>>(), cmd).map_err(DenseMtpError::Backend)?;
        drop(tokens);
        Ok(())
    }

    fn dense_copy_hidden(
        &self,
        previous: Option<&B::GlobalBuffer>,
        current: &B::GlobalBuffer,
        rows: usize,
    ) -> Result<B::GlobalBuffer, DenseMtpError<B>> {
        let row = self.decoder.embedding().model_dim() as usize * 2;
        let mut saved = self.engine.context.create_buffer(rows * row).map_err(DenseMtpError::Backend)?;
        let mut cmd = self
            .engine
            .context
            .create_command_buffer(Some("dense MTP residual copy"), None)
            .map_err(DenseMtpError::Backend)?;
        let offset = usize::from(previous.is_some());
        if let Some(previous) = previous {
            cmd.encode_copy(previous, saved.subrange_mut(0..row));
        }
        if rows > offset {
            cmd.encode_copy(current.subrange(0..(rows - offset) * row), saved.subrange_mut(offset * row..rows * row));
        }
        let complete = cmd.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        drop(complete);
        Ok(saved)
    }

    fn dense_last_hidden(
        &self,
        current: &B::GlobalBuffer,
        row_index: usize,
    ) -> Result<B::GlobalBuffer, DenseMtpError<B>> {
        let mut cmd = self
            .engine
            .context
            .create_command_buffer(Some("dense MTP last residual"), None)
            .map_err(DenseMtpError::Backend)?;
        let saved = self.encode_dense_last_hidden(current, row_index, &mut cmd)?;
        let complete = cmd.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        drop(complete);
        Ok(saved)
    }

    fn encode_dense_last_hidden(
        &self,
        current: &B::GlobalBuffer,
        row_index: usize,
        cmd: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::GlobalBuffer, DenseMtpError<B>> {
        let row = self.decoder.embedding().model_dim() as usize * 2;
        let end = row_index.checked_add(1).and_then(|count| count.checked_mul(row)).ok_or(DenseMtpError::Invalid)?;
        if end > current.size() {
            return Err(DenseMtpError::Invalid);
        }
        let mut saved = self.engine.context.create_buffer(row).map_err(DenseMtpError::Backend)?;
        cmd.encode_copy(current.subrange(end - row..end), &mut saved);
        Ok(saved)
    }

    pub fn prefill_dense_mtp(
        &self,
        input: &[u32],
        capacity: u32,
    ) -> Result<DenseMtpSession<B>, DenseMtpError<B>> {
        self.prefill_dense_mtp_controlled(input, capacity, &mut || false)
    }

    pub fn prefill_dense_mtp_controlled(
        &self,
        input: &[u32],
        capacity: u32,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<DenseMtpSession<B>, DenseMtpError<B>> {
        self.prefill_dense_mtp_with_prefix_controlled(input, capacity, true, cancelled)
    }

    /// Skip snapshot copies when no host cache scope can retain the result.
    pub fn prefill_dense_mtp_with_prefix_controlled(
        &self,
        input: &[u32],
        capacity: u32,
        capture_prefix: bool,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<DenseMtpSession<B>, DenseMtpError<B>> {
        if !self.has_dense_mtp()
            || input.is_empty()
            || capacity == 0
            || input.len() >= capacity as usize
            || self.max_context_length().is_some_and(|n| capacity > n)
            || input.iter().any(|&x| x >= self.decoder.embedding().vocab_size())
        {
            return Err(DenseMtpError::Invalid);
        }
        let target =
            self.decoder.create_empty_state(Some(capacity), &self.engine.context).map_err(DenseMtpError::Backend)?;
        let predictor = self
            .dense_mtp
            .as_ref()
            .unwrap()
            .create_state(capacity, &self.engine.context)
            .map_err(DenseMtpError::Backend)?;
        let last_hidden = self
            .engine
            .context
            .create_buffer(self.decoder.embedding().model_dim() as usize * 2)
            .map_err(DenseMtpError::Backend)?;
        let mut session = DenseMtpSession {
            identity: self.model_identity.clone(),
            target,
            predictor,
            last_hidden,
            logits: Vec::new(),
            capacity: capacity as usize,
            failed: false,
            prefix: None,
        };
        let boundary = if capture_prefix {
            input.len().saturating_sub(1) / super::stream::PREFILL_CHUNK_TOKENS * super::stream::PREFILL_CHUNK_TOKENS
        } else {
            0
        };
        self.continue_prefill_dense_mtp_controlled(&mut session, input, boundary, cancelled)?;
        Ok(session)
    }

    pub fn continue_prefill_dense_mtp_controlled(
        &self,
        session: &mut DenseMtpSession<B>,
        input: &[u32],
        capture_position: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), DenseMtpError<B>> {
        let grid = super::stream::PREFILL_CHUNK_TOKENS;
        let start = session.position();
        if session.failed
            || !Arc::ptr_eq(&self.model_identity, &session.identity)
            || input.is_empty()
            || !start.is_multiple_of(grid)
            || start.checked_add(input.len()).is_none_or(|n| n >= session.capacity)
            || input.iter().any(|&x| x >= self.decoder.embedding().vocab_size())
            || (capture_position != 0
                && (!capture_position.is_multiple_of(grid)
                    || capture_position < start
                    || capture_position >= start + input.len()))
        {
            return Err(DenseMtpError::Invalid);
        }
        let result = (|| {
            for chunk in input.chunks(grid) {
                if cancelled() {
                    return Err(DenseMtpError::Cancelled);
                }
                let has_previous = session.position() > 0;
                let pass = self.dense_target_pass(chunk, &mut session.target, true)?;
                let pairs = chunk.len() - usize::from(!has_previous);
                if pairs > 0 {
                    let joined =
                        self.dense_copy_hidden(has_previous.then_some(&session.last_hidden), &pass.hidden, pairs)?;
                    let next_ids = if has_previous {
                        chunk
                    } else {
                        &chunk[1..]
                    };
                    let pool = self.engine.context.create_allocation_pool();
                    let mut command = self
                        .engine
                        .context
                        .create_command_buffer(Some("dense MTP predictor priming"), Some(pool))
                        .map_err(DenseMtpError::Backend)?;
                    self.encode_dense_predictor_cache_only(next_ids, &joined, &mut session.predictor, &mut command)?;
                    let complete =
                        command.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
                    drop(complete);
                }
                session.last_hidden = self.dense_last_hidden(&pass.hidden, chunk.len() - 1)?;
                session.logits = pass.logits;
                if capture_position > 0 && session.position() == capture_position {
                    self.capture_dense_mtp_prefix(session)?;
                }
            }
            if cancelled() {
                return Err(DenseMtpError::Cancelled);
            }
            Ok(())
        })();
        if result.is_err() {
            session.failed = true;
        }
        result
    }

    fn capture_dense_mtp_prefix(
        &self,
        session: &mut DenseMtpSession<B>,
    ) -> Result<(), DenseMtpError<B>> {
        if session.predictor.context_length() + 1 != session.target.context_length() {
            return Err(DenseMtpError::Invalid);
        }
        let mut command = self
            .engine
            .context
            .create_command_buffer(Some("dense MTP prefix snapshot"), None)
            .map_err(DenseMtpError::Backend)?;
        let target = session
            .target
            .capture_prefix(&self.engine.context, &mut command)
            .map_err(DenseMtpError::Backend)?
            .ok_or(DenseMtpError::Invalid)?;
        let predictor = session
            .predictor
            .capture_prefix(&self.engine.context, &mut command)
            .map_err(DenseMtpError::Backend)?
            .ok_or(DenseMtpError::Invalid)?;
        let mut hidden =
            self.engine.context.create_buffer(session.last_hidden.size()).map_err(DenseMtpError::Backend)?;
        command.encode_copy(&session.last_hidden, &mut hidden);
        let complete = command.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        drop(complete);
        session.prefix = Some(DenseMtpPrefix {
            target,
            predictor,
            hidden,
            logits: session.logits.clone(),
            position: session.position(),
        });
        Ok(())
    }

    pub fn restore_dense_mtp_prefix(
        &self,
        session: &mut DenseMtpSession<B>,
        position: usize,
    ) -> Result<(), DenseMtpError<B>> {
        if session.failed || !Arc::ptr_eq(&self.model_identity, &session.identity) {
            return Err(DenseMtpError::Invalid);
        }
        let prefix = session.prefix.as_ref().ok_or(DenseMtpError::Invalid)?;
        if prefix.position != position
            || !session.target.can_restore_prefix(&prefix.target)
            || !session.predictor.can_restore_prefix(&prefix.predictor)
        {
            return Err(DenseMtpError::Invalid);
        }
        let result = (|| {
            let mut cmd = self
                .engine
                .context
                .create_command_buffer(Some("dense MTP prefix restore"), None)
                .map_err(DenseMtpError::Backend)?;
            if !session.target.restore_prefix(&prefix.target, &mut cmd)
                || !session.predictor.restore_prefix(&prefix.predictor, &mut cmd)
            {
                return Err(DenseMtpError::Invalid);
            }
            cmd.encode_copy(&prefix.hidden, &mut session.last_hidden);
            let complete = cmd.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
            drop(complete);
            session.logits = prefix.logits.clone();
            Ok(())
        })();
        if result.is_err() {
            session.failed = true;
        }
        result
    }

    pub fn dense_mtp_round(
        &self,
        session: &mut DenseMtpSession<B>,
        first: u32,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
    ) -> Result<DenseMtpRound, DenseMtpError<B>> {
        if session.failed
            || !Arc::ptr_eq(&self.model_identity, &session.identity)
            || !self.has_dense_mtp()
            || drafts > 3
            || first >= self.decoder.embedding().vocab_size()
            || session.position().checked_add(drafts + 1).is_none_or(|n| n > session.capacity)
        {
            return Err(DenseMtpError::Invalid);
        }
        let result = if self.decoder.speculation_supported() {
            self.dense_mtp_round_inner(session, first, drafts, sampler)
        } else {
            self.dense_mtp_serial_text_round(session, first, drafts, sampler)
        };
        if result.is_err() {
            session.failed = true;
        }
        result
    }

    fn dense_mtp_serial_text_round(
        &self,
        session: &mut DenseMtpSession<B>,
        first: u32,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
    ) -> Result<DenseMtpRound, DenseMtpError<B>> {
        let started = Instant::now();
        let prefix = session.predictor.context_length();
        let mut inputs = vec![first];
        let mut hidden = None;
        sampler.begin_round();
        for _ in 0..drafts {
            let pass = self.dense_predictor_pass(
                &[*inputs.last().unwrap()],
                hidden.as_ref().unwrap_or(&session.last_hidden),
                &mut session.predictor,
                true,
            )?;
            let proposal = sampler.draft(&pass.logits).map_err(|_| DenseMtpError::Sampling)?;
            if proposal >= self.decoder.embedding().vocab_size() {
                return Err(DenseMtpError::Invalid);
            }
            inputs.push(proposal);
            hidden = pass.hidden;
        }
        let draft_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut accepted = 0;
        let mut passes = Vec::new();
        let next = loop {
            // Always a single, fully accepted text row. Reference DeltaNet and
            // MRoPE retain their ordinary serial semantics; no image positions
            // or speculative suffix state are introduced.
            let pass = self.dense_target_pass(&[inputs[accepted]], &mut session.target, true)?;
            let proposal = inputs.get(accepted + 1).copied();
            let sampled = sampler.target(&pass.logits, proposal).map_err(|_| DenseMtpError::Sampling)?;
            if sampled >= self.decoder.embedding().vocab_size() {
                return Err(DenseMtpError::Invalid);
            }
            passes.push(pass);
            if proposal == Some(sampled) {
                accepted += 1;
            } else {
                break sampled;
            }
        };
        let verify_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let repair_start = usize::from(drafts > 0);
        if !session.predictor.rewind(prefix.checked_add(repair_start as u32).ok_or(DenseMtpError::Invalid)?) {
            return Err(DenseMtpError::Invalid);
        }
        let pool = self.engine.context.create_allocation_pool();
        let mut command = self
            .engine
            .context
            .create_command_buffer(Some("dense text predictor repair"), Some(pool))
            .map_err(DenseMtpError::Backend)?;
        for index in repair_start..passes.len() {
            let previous = if index == 0 {
                &session.last_hidden
            } else {
                &passes[index - 1].hidden
            };
            self.encode_dense_predictor_cache_only(
                &inputs[index..index + 1],
                previous,
                &mut session.predictor,
                &mut command,
            )?;
        }
        let complete = command.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        drop(complete);
        let (mut encode, mut gpu, mut readout) = (0.0, 0.0, 0.0);
        for pass in &passes {
            encode += pass.encode_seconds;
            gpu += pass.gpu_seconds;
            readout += pass.readout_seconds;
        }
        let final_pass = passes.pop().ok_or(DenseMtpError::Invalid)?;
        session.last_hidden = final_pass.hidden;
        session.logits = final_pass.logits;
        Ok(DenseMtpRound {
            accepted_drafts: inputs[1..accepted + 1].to_vec(),
            next_token: next,
            proposed: drafts,
            accepted,
            verified_rows: accepted + 1,
            draft_seconds,
            verify_seconds,
            rollback_seconds: started.elapsed().as_secs_f64(),
            target_encode_seconds: encode,
            target_gpu_seconds: gpu,
            target_readout_seconds: readout,
        })
    }

    fn dense_mtp_round_inner(
        &self,
        session: &mut DenseMtpSession<B>,
        first: u32,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
    ) -> Result<DenseMtpRound, DenseMtpError<B>> {
        let started = Instant::now();
        let prefix = session.predictor.context_length();
        let mut inputs = vec![first];
        // The session already owns its final residual until the round commits.
        // Later draft residuals are owned here and replace it only as inputs.
        let mut hidden = None;
        sampler.begin_round();
        for _ in 0..drafts {
            let pass = self.dense_predictor_pass(
                &[*inputs.last().unwrap()],
                hidden.as_ref().unwrap_or(&session.last_hidden),
                &mut session.predictor,
                true,
            )?;
            let proposal = sampler.draft(&pass.logits).map_err(|_| DenseMtpError::Sampling)?;
            if proposal >= self.decoder.embedding().vocab_size() {
                return Err(DenseMtpError::Invalid);
            }
            inputs.push(proposal);
            hidden = pass.hidden;
        }
        let draft_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let pass = self.dense_target_pass(&inputs, &mut session.target, false)?;
        let vocab = self.decoder.embedding().vocab_size() as usize;
        let mut accepted = 0;
        let next = loop {
            let proposal = inputs.get(accepted + 1).copied();
            let sampled = sampler
                .target(&pass.logits[accepted * vocab..(accepted + 1) * vocab], proposal)
                .map_err(|_| DenseMtpError::Sampling)?;
            if sampled >= vocab as u32 {
                return Err(DenseMtpError::Invalid);
            }
            if proposal == Some(sampled) {
                accepted += 1;
            } else {
                break sampled;
            }
        };
        let count = accepted + 1;
        let verify_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let pool = self.engine.context.create_allocation_pool();
        let mut cmd = self
            .engine
            .context
            .create_command_buffer(Some("dense MTP commit and predictor repair"), Some(pool))
            .map_err(DenseMtpError::Backend)?;
        session
            .target
            .encode_accept(&(0..count as u32).collect::<Vec<_>>(), &mut cmd)
            .map_err(DenseMtpError::Backend)?;
        if drafts == 0 {
            self.encode_dense_predictor_cache_only(
                &inputs[..count],
                &session.last_hidden,
                &mut session.predictor,
                &mut cmd,
            )?;
        } else {
            if !session.predictor.rewind(prefix.checked_add(1).ok_or(DenseMtpError::Invalid)?) {
                return Err(DenseMtpError::Invalid);
            }
            if count > 1 {
                let accepted_proposals = count - 1;
                let row_bytes = self.decoder.embedding().model_dim() as usize * 2;
                self.encode_dense_predictor_cache_only(
                    &inputs[1..count],
                    pass.hidden.subrange(..accepted_proposals * row_bytes),
                    &mut session.predictor,
                    &mut cmd,
                )?;
            }
        }
        let last_hidden = self.encode_dense_last_hidden(&pass.hidden, count - 1, &mut cmd)?;
        let complete = cmd.end_encoding().submit().wait_until_completed().map_err(DenseMtpError::Backend)?;
        drop(complete);
        session.last_hidden = last_hidden;
        session.logits = pass.logits[(count - 1) * vocab..count * vocab].to_vec();
        Ok(DenseMtpRound {
            accepted_drafts: inputs[1..count].to_vec(),
            next_token: next,
            proposed: drafts,
            accepted,
            verified_rows: inputs.len(),
            draft_seconds,
            verify_seconds,
            rollback_seconds: started.elapsed().as_secs_f64(),
            target_encode_seconds: pass.encode_seconds,
            target_gpu_seconds: pass.gpu_seconds,
            target_readout_seconds: pass.readout_seconds,
        })
    }
}
