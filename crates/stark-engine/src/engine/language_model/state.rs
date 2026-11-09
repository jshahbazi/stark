use std::sync::Arc;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    encodable_block::{
        dflash::DFlashState,
        sampling::PRng,
        transformer::{TransformerPrefix, TransformerState},
    },
    engine::language_model::LanguageModel,
};

fn effective_context_limit(
    requested: Option<u32>,
    maximum: Option<u32>,
) -> Result<Option<u32>, crate::backends::common::InvalidInput> {
    match (requested, maximum) {
        (Some(0), _) => Err(crate::backends::common::InvalidInput("zero context capacity")),
        (Some(requested), Some(maximum)) if requested > maximum => {
            Err(crate::backends::common::InvalidInput("context exceeds model capacity"))
        },
        (requested, maximum) => Ok(requested.or(maximum)),
    }
}

pub struct LanguageModelState<B: Backend> {
    pub(super) model_identity: Arc<()>,
    pub(super) tokens: Vec<u64>,
    pub(super) last_output_token: Option<u64>, // TODO: this leaks previous LanguageModelStreamOptions
    pub(super) prng: PRng,
    pub(super) transformer_state: TransformerState<B>,
    pub(super) speculator_state: Option<DFlashState<B>>,
    pub(super) max_context_length: Option<u32>,
    pub(super) multimodal_failed: bool,
    /// Image request marker independent of Qwen's spatial rotary delta.
    pub(super) multimodal_prefilled: bool,
    pub(super) text_prefix_capture_position: Option<usize>,
    pub(super) text_prefix: Option<(usize, TransformerPrefix<B>)>,
    /// One-shot proof that this exact state was restored before a suffix stream.
    pub(super) text_prefix_restored_position: Option<usize>,
    #[cfg(test)]
    pub(crate) vision_prefill_logits_readback: Option<B::GlobalBuffer>,
    #[cfg(test)]
    pub(crate) vision_prefill_layer_readbacks: Option<(usize, Box<[B::GlobalBuffer]>)>,
}

impl<B: Backend> LanguageModelState<B> {
    pub(super) fn has_multimodal_context(&self) -> bool {
        self.multimodal_prefilled || self.transformer_state.rotary_position_delta().is_some()
    }

    pub fn tokens(&self) -> &[u64] {
        &self.tokens
    }

    /// A completed text-only prefill boundary retained for consecutive reuse.
    pub fn text_prefix_position(&self) -> Option<usize> {
        (!self.multimodal_failed && !self.has_multimodal_context())
            .then(|| self.text_prefix.as_ref().map(|(position, _)| *position))
            .flatten()
    }

    pub(crate) fn context_capacity(&self) -> Option<u32> {
        self.max_context_length
    }

    #[cfg(test)]
    pub(crate) fn vision_validation_state(&self) -> (u32, Option<i64>) {
        (self.transformer_state.context_length(), self.transformer_state.rotary_position_delta())
    }
}

#[derive(Debug, Error)]
#[error("State cannot be rewound to this prefix")]
pub struct LanguageModelRewindStateError;

#[derive(Debug, Error)]
pub enum LanguageModelCreateEmptyStateError<B: Backend> {
    #[error("Invalid model state context")]
    InvalidContext,
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
}

impl<B: Backend> LanguageModel<B> {
    /// Allocate an empty per-request state.
    ///
    /// `max_context_length` bounds prompt plus generated tokens and sizes the
    /// KV cache; `None` means the model's own maximum, which is only valid for
    /// models whose attention layers have one (full-attention layers require a
    /// finite capacity and models with a speculator always require `Some`).
    /// `sampling_seed` seeds the per-state random generator used by stochastic
    /// sampling; greedy decoding ignores it.
    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        sampling_seed: u64,
    ) -> Result<LanguageModelState<B>, LanguageModelCreateEmptyStateError<B>> {
        let max_context_length = effective_context_limit(max_context_length, self.max_context_length())
            .map_err(|_| LanguageModelCreateEmptyStateError::InvalidContext)?;
        if self.speculator.is_some() && max_context_length.is_none() {
            return Err(LanguageModelCreateEmptyStateError::InvalidContext);
        }
        let tokens = Vec::new();
        let last_output_token = None;

        let prng = PRng::new(sampling_seed);

        let transformer_state = self
            .decoder
            .create_empty_state(max_context_length, &self.engine.context)
            .map_err(LanguageModelCreateEmptyStateError::Backend)?;

        let speculator_state = self
            .speculator
            .as_ref()
            .map(|speculator| {
                speculator.empty_state(max_context_length.expect("speculator doesn't support unlimited state capacity"))
            })
            .transpose()
            .map_err(LanguageModelCreateEmptyStateError::Backend)?;

        Ok(LanguageModelState {
            model_identity: self.model_identity.clone(),
            tokens,
            last_output_token,
            prng,
            transformer_state,
            speculator_state,
            max_context_length,
            multimodal_failed: false,
            multimodal_prefilled: false,
            text_prefix_capture_position: None,
            text_prefix: None,
            text_prefix_restored_position: None,
            #[cfg(test)]
            vision_prefill_logits_readback: None,
            #[cfg(test)]
            vision_prefill_layer_readbacks: None,
        })
    }

    /// Whether a completed state can be rewound for the next request.
    pub fn prefix_reuse_supported(&self) -> bool {
        self.speculator.is_none() && self.decoder.prefix_reuse_supported()
    }

    pub(crate) fn text_prefix_snapshot_supported(&self) -> bool {
        self.speculator.is_none() && self.decoder.prefix_snapshot_supported()
    }

    /// Arm one non-final prefill chunk. The stream checks that the requested
    /// boundary is reached before sampling and publishes only completed copies.
    pub(crate) fn prepare_text_prefix_capture(
        &self,
        state: &mut LanguageModelState<B>,
        position: usize,
    ) -> Result<(), LanguageModelRewindStateError> {
        if !Arc::ptr_eq(&self.model_identity, &state.model_identity)
            || !self.text_prefix_snapshot_supported()
            || state.speculator_state.is_some()
            || state.multimodal_failed
            || state.has_multimodal_context()
        {
            return Err(LanguageModelRewindStateError);
        }
        if position == 0 {
            state.text_prefix_capture_position = None;
            state.text_prefix = None;
            return Ok(());
        }
        if !position.is_multiple_of(super::stream::PREFILL_CHUNK_TOKENS)
            || state.max_context_length.is_none_or(|capacity| position >= capacity as usize)
        {
            return Err(LanguageModelRewindStateError);
        }
        if position == state.tokens.len() && state.text_prefix_position() == Some(position) {
            state.text_prefix_capture_position = None;
            return Ok(());
        }
        if position <= state.tokens.len() {
            return Err(LanguageModelRewindStateError);
        }
        state.text_prefix_capture_position = Some(position);
        state.text_prefix = None;
        Ok(())
    }

    pub(crate) fn preflight_restored_text_stream(
        &self,
        input: &[u64],
        state: &LanguageModelState<B>,
        options: &super::stream::LanguageModelStreamOptions,
    ) -> Result<(), super::stream::LanguageModelStreamError<B>> {
        use super::stream::LanguageModelStreamError;
        if !Arc::ptr_eq(&self.model_identity, &state.model_identity) {
            return Err(LanguageModelStreamError::ForeignState);
        }
        if !self.text_prefix_snapshot_supported()
            || state.multimodal_failed
            || state.has_multimodal_context()
            || state.last_output_token.is_some()
            || state.tokens.is_empty()
            || state.text_prefix_restored_position != Some(state.tokens.len())
            || state.transformer_state.context_length() as usize != state.tokens.len()
        {
            return Err(LanguageModelStreamError::InvalidMultimodal);
        }
        if input.is_empty() {
            return Err(LanguageModelStreamError::NoSeedToken);
        }
        if input.iter().any(|&token| token >= u64::from(self.sampling.vocab_size())) {
            return Err(LanguageModelStreamError::InvalidToken);
        }
        options
            .sampling_method
            .validate(self.sampling.vocab_size())
            .map_err(|_| LanguageModelStreamError::InvalidSampling)?;
        if state.max_context_length.is_some_and(|capacity| state.tokens.len() + input.len() >= capacity as usize) {
            return Err(LanguageModelStreamError::ContextOverflow);
        }
        Ok(())
    }

    pub(crate) fn restore_text_prefix(
        &self,
        state: &mut LanguageModelState<B>,
        position: usize,
        capacity: Option<u32>,
        seed: u64,
    ) -> Result<(), LanguageModelRewindStateError> {
        if !Arc::ptr_eq(&self.model_identity, &state.model_identity)
            || !self.text_prefix_snapshot_supported()
            || state.speculator_state.is_some()
            || state.multimodal_failed
            || state.has_multimodal_context()
            || state.max_context_length != capacity
            || position == 0
            || !position.is_multiple_of(super::stream::PREFILL_CHUNK_TOKENS)
            || position > state.tokens.len()
        {
            return Err(LanguageModelRewindStateError);
        }
        let Some((saved_position, prefix)) = state.text_prefix.as_ref() else {
            return Err(LanguageModelRewindStateError);
        };
        if *saved_position != position || !state.transformer_state.can_restore_prefix(prefix) {
            return Err(LanguageModelRewindStateError);
        }
        let restored = (|| {
            let mut command = self
                .engine
                .context
                .create_command_buffer(Some("text prefix restore"), None)
                .map_err(|_| LanguageModelRewindStateError)?;
            if !state.transformer_state.restore_prefix(prefix, &mut command) {
                return Err(LanguageModelRewindStateError);
            }
            command.end_encoding().submit().wait_until_completed().map_err(|_| LanguageModelRewindStateError)?;
            Ok(())
        })();
        if restored.is_err() {
            state.multimodal_failed = true;
            state.text_prefix = None;
            return restored;
        }
        state.tokens.truncate(position);
        state.last_output_token = None;
        state.prng = PRng::new(seed);
        state.text_prefix_capture_position = None;
        state.text_prefix_restored_position = Some(position);
        Ok(())
    }

    /// Prepare a completed request state for a new prompt that shares its
    /// first `prefix_length` tokens.
    ///
    /// `prefix_length` must lie on the prefill chunk grid, so the rest of the
    /// prompt is prefilled with the same chunk boundaries as on a fresh state.
    /// The capacity must match the one the state was created with. Clears the
    /// pending output token and reseeds the sampler as `create_empty_state`
    /// would. After an error the state must be discarded.
    pub fn rewind_state(
        &self,
        state: &mut LanguageModelState<B>,
        prefix_length: usize,
        max_context_length: Option<u32>,
        sampling_seed: u64,
    ) -> Result<(), LanguageModelRewindStateError> {
        if !Arc::ptr_eq(&self.model_identity, &state.model_identity)
            || !self.prefix_reuse_supported()
            || state.speculator_state.is_some()
            || state.multimodal_failed
            || state.max_context_length != max_context_length
            || !prefix_length.is_multiple_of(super::stream::PREFILL_CHUNK_TOKENS)
            || prefix_length > state.tokens.len()
        {
            return Err(LanguageModelRewindStateError);
        }
        let rows = u32::try_from(prefix_length).map_err(|_| LanguageModelRewindStateError)?;
        if !state.transformer_state.rewind(rows) {
            return Err(LanguageModelRewindStateError);
        }
        state.tokens.truncate(prefix_length);
        state.last_output_token = None;
        state.prng = PRng::new(sampling_seed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::effective_context_limit;

    #[stark_test]
    fn unspecified_context_retains_finite_model_capacity() {
        assert_eq!(effective_context_limit(None, Some(4096)).unwrap(), Some(4096));
        assert_eq!(effective_context_limit(Some(1024), Some(4096)).unwrap(), Some(1024));
        assert_eq!(effective_context_limit(None, None).unwrap(), None);
        assert_eq!(effective_context_limit(Some(1024), None).unwrap(), Some(1024));
        assert!(effective_context_limit(Some(0), Some(4096)).is_err());
        assert!(effective_context_limit(Some(4097), Some(4096)).is_err());
    }
}
