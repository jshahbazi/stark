//! Exact E2B companion admission and owned-operation lifecycle.
//! Default worker serving remains off pending staged protocol qualification.
use std::{io::Seek, path::Path, time::Instant};

use thiserror::Error;

use super::Gemma4VisionLanguageModel;
use crate::{
    backends::common::{
        Backend, BufferCpuAccessible, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    },
    encodable_block::gemma4_assistant::{Gemma4Assistant, Gemma4AssistantError, Gemma4AssistantProfile},
    engine::{
        hybrid::MtpSampler,
        language_model::gemma4_assistant::{
            Gemma4AssistantGeneratedRound, Gemma4AssistantRetainedPrefix, Gemma4AssistantSessionError,
            Gemma4AssistantTargetFrame,
        },
        vision::{file_digest, open_artifact_component},
    },
    parameters::{ParameterLoader, ParameterLoaderError},
};

pub(super) const TARGET_CONFIG: &str = "9acc540130bd576271650c3805467141136570ffc4111c186e317442cc3dc762";
pub(super) const TARGET_WEIGHTS: &str = "5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993";
// Schema-2 native export of the pinned official assistant. The full-file hash
// covers its exact source/target metadata as well as every tensor payload.
const COMPANION: &str = "8b81d6337f1d3e8a562c13d57af43ce0cbd30698683a56eccadaf39f7549bb9f";
const COMPANION_BYTES: u64 = 157_581_400;

#[derive(Debug, Error)]
pub enum Gemma4AssistantAdmissionError<B: Backend> {
    #[error("unsupported Gemma assistant/target artifact pair")]
    Invalid,
    #[error("Gemma assistant selected artifact I/O failed")]
    IO(#[from] std::io::Error),
    #[error("Gemma assistant parameter admission failed")]
    Parameters(#[from] ParameterLoaderError<B>),
    #[error("Gemma assistant construction failed")]
    Assistant(#[from] Gemma4AssistantError<B>),
}

impl<B: Backend> Gemma4VisionLanguageModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
    fn assistant_pair_ready(
        &self,
        head: &Gemma4Assistant<B>,
    ) -> bool {
        !self.failed
            && head.matches_target(&self.language_model.model_identity)
            && self.identity.config_sha256 == TARGET_CONFIG
            && self.identity.artifact_sha256 == TARGET_WEIGHTS
    }

    /// Every error consumes its request-owned state. Fence non-cancellation
    /// failures and join our queue before any caller can reuse this model.
    fn finish_assistant_operation<T>(
        &mut self,
        result: Result<T, Gemma4AssistantSessionError<B>>,
    ) -> Result<T, Gemma4AssistantSessionError<B>> {
        if result.is_err() {
            let cancelled = matches!(&result, Err(Gemma4AssistantSessionError::Cancelled));
            self.failed = true;
            let drained = self
                .engine
                .context
                .create_command_buffer(Some("Gemma assistant owned queue drain"), None)
                .and_then(|command| command.end_encoding().submit().wait_until_completed());
            if let Err(error) = drained {
                return Err(Gemma4AssistantSessionError::Backend(error));
            }
            if cancelled {
                self.failed = false;
            }
        }
        result
    }

    pub fn assistant_generation_memory_bytes(
        &self,
        head: &Gemma4Assistant<B>,
        input: u32,
        context: u32,
        output: u32,
    ) -> Option<usize> {
        if !self.assistant_pair_ready(head) || context > 2048 {
            return None;
        }
        self.language_model
            .gemma4_assistant_generation_memory_bytes(input, context, output)?
            .checked_add(self.resident_parameter_bytes())?
            .checked_add(self.assistant_resident_memory_bytes(head)?)
    }

    pub fn assistant_resident_memory_bytes(
        &self,
        head: &Gemma4Assistant<B>,
    ) -> Option<usize> {
        self.assistant_pair_ready(head).then(|| head.resident_storage_bytes().max(COMPANION_BYTES as usize))
    }

    pub fn assistant_artifact_identity(
        &self,
        head: &Gemma4Assistant<B>,
    ) -> Option<(&'static str, u64)> {
        self.assistant_pair_ready(head).then_some((COMPANION, COMPANION_BYTES))
    }

    pub fn prefill_assistant_target(
        &mut self,
        head: &Gemma4Assistant<B>,
        input: &[u32],
        capacity: u32,
        capture: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        if !self.assistant_pair_ready(head) || capacity > 2048 {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let result = self.language_model.prefill_gemma4_assistant_target_with_prefix(
            input,
            capacity,
            Gemma4AssistantProfile::E2B,
            capture,
            cancelled,
        );
        self.finish_assistant_operation(result)
    }

    pub fn restore_assistant_target(
        &mut self,
        head: &Gemma4Assistant<B>,
        frame: Gemma4AssistantTargetFrame<B>,
        suffix: &[u32],
        capture: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        if !self.assistant_pair_ready(head) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let result = self.language_model.restore_and_prefill_gemma4_assistant_target(frame, suffix, capture, cancelled);
        self.finish_assistant_operation(result)
    }

    pub fn retain_assistant_prompt_prefix(
        &mut self,
        head: &Gemma4Assistant<B>,
        frame: Gemma4AssistantTargetFrame<B>,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<Gemma4AssistantRetainedPrefix<B>>, Gemma4AssistantSessionError<B>> {
        if !self.assistant_pair_ready(head) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let result = self.language_model.retain_gemma4_assistant_prompt_prefix(frame, cancelled);
        self.finish_assistant_operation(result)
    }

    pub fn resume_assistant_prompt_prefix(
        &mut self,
        head: &Gemma4Assistant<B>,
        prefix: Gemma4AssistantRetainedPrefix<B>,
        suffix: &[u32],
        capture: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantTargetFrame<B>, Gemma4AssistantSessionError<B>> {
        if !self.assistant_pair_ready(head) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let result =
            self.language_model.prefill_gemma4_assistant_from_retained_prefix(prefix, suffix, capture, cancelled);
        self.finish_assistant_operation(result)
    }

    pub fn assistant_greedy_round(
        &mut self,
        head: &Gemma4Assistant<B>,
        frame: Gemma4AssistantTargetFrame<B>,
        drafts: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantGeneratedRound<B>, Gemma4AssistantSessionError<B>> {
        if !self.assistant_pair_ready(head) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let result = (|| {
            let first = frame
                .logits()
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .map(|(index, _)| index as u32)
                .ok_or(Gemma4AssistantSessionError::Invalid)?;
            let started = Instant::now();
            let proposals =
                self.language_model.draft_gemma4_assistant_greedy_controlled(&frame, head, first, drafts, cancelled)?;
            let draft_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            // Preserve ordinary one-row target arithmetic for staged serving.
            // The experimental batch verifier has a separate component gate.
            let verified = self.language_model.verify_gemma4_assistant_greedy_serial(frame, &proposals, cancelled)?;
            Ok(Gemma4AssistantGeneratedRound {
                verified,
                proposed: proposals.len(),
                draft_seconds,
                verify_seconds: started.elapsed().as_secs_f64(),
            })
        })();
        self.finish_assistant_operation(result)
    }

    pub fn assistant_sampled_round(
        &mut self,
        head: &Gemma4Assistant<B>,
        frame: Gemma4AssistantTargetFrame<B>,
        first: u32,
        drafts: usize,
        sampler: &mut dyn MtpSampler,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Gemma4AssistantGeneratedRound<B>, Gemma4AssistantSessionError<B>> {
        if !self.assistant_pair_ready(head) {
            return Err(Gemma4AssistantSessionError::Invalid);
        }
        let result = (|| {
            sampler.begin_round();
            let started = Instant::now();
            let proposals = self
                .language_model
                .draft_gemma4_assistant_sampled_controlled(&frame, head, first, drafts, sampler, cancelled)?;
            let draft_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let verified = self
                .language_model
                .verify_gemma4_assistant_sampled_serial(frame, first, &proposals, sampler, cancelled)?;
            Ok(Gemma4AssistantGeneratedRound {
                verified,
                proposed: proposals.len(),
                draft_seconds,
                verify_seconds: started.elapsed().as_secs_f64(),
            })
        })();
        self.finish_assistant_operation(result)
    }
}

impl<B: Backend> Gemma4VisionLanguageModel<B> {
    /// Explicit validation component, pending numerical/serving qualification.
    /// Reads only the selected native sidecar; an auxiliary, self-declared
    /// config cannot substitute a different source or target pair.
    pub fn load_assistant_component(
        &self,
        path: &Path,
    ) -> Result<Gemma4Assistant<B>, Gemma4AssistantAdmissionError<B>> {
        if self.failed
            || self.identity.config_sha256 != TARGET_CONFIG
            || self.identity.artifact_sha256 != TARGET_WEIGHTS
        {
            return Err(Gemma4AssistantAdmissionError::Invalid);
        }
        let mut selected = open_artifact_component(&path.join("model.safetensors"))?;
        if selected.metadata()?.len() != COMPANION_BYTES || file_digest(&mut selected)? != COMPANION {
            return Err(Gemma4AssistantAdmissionError::Invalid);
        }
        selected.rewind()?;
        let loader = ParameterLoader::new(&selected, &*self.engine.context)?;
        let head = Gemma4Assistant::new(
            Gemma4AssistantProfile::E2B,
            &loader.tree().subtree("assistant"),
            &*self.engine.context,
        )?;
        loader.tree().assert_all_tensors_validated()?;
        let storage = loader.storage_statistics();
        drop(loader);
        selected.rewind()?;
        if selected.metadata()?.len() != COMPANION_BYTES || file_digest(&mut selected)? != COMPANION {
            return Err(Gemma4AssistantAdmissionError::Invalid);
        }
        Ok(head.bind_target(self.language_model.model_identity.clone(), storage))
    }
}
