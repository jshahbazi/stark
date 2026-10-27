use std::sync::Arc;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::{dflash::DFlashState, sampling::PRng, transformer::TransformerState},
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
    #[cfg(test)]
    pub(crate) vision_prefill_logits_readback: Option<B::GlobalBuffer>,
    #[cfg(test)]
    pub(crate) vision_prefill_layer_readbacks: Option<(usize, Box<[B::GlobalBuffer]>)>,
}

impl<B: Backend> LanguageModelState<B> {
    pub fn tokens(&self) -> &[u64] {
        &self.tokens
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
            #[cfg(test)]
            vision_prefill_logits_readback: None,
            #[cfg(test)]
            vision_prefill_layer_readbacks: None,
        })
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
