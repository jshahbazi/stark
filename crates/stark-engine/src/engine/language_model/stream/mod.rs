use derive_more::Debug;
use thiserror::Error;

#[cfg(grammar)]
use crate::engine::language_model::grammar::{Grammar, GrammarError};
use crate::{
    backends::common::Backend,
    encodable_block::decoder::DecoderError,
    engine::language_model::{LanguageModel, state::LanguageModelState},
    speculators::dflash_tfm::DFlashTreeError,
};
pub use crate::{
    encodable_block::sampling::SamplingMethod,
    engine::language_model::stream::stream::{LanguageModelStream, TokenStreamMetrics},
};

mod stream;

pub struct LanguageModelStreamOptions {
    pub sampling_method: SamplingMethod,
    #[cfg(grammar)]
    pub grammar: Option<Grammar>,
}

#[derive(Debug, Error)]
pub enum LanguageModelStreamError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[cfg(grammar)]
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
    #[error("Trie accept error: {0}")]
    TrieAccept(#[from] crate::trie::TrieAcceptError),
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashTreeError<B>),
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken,
    #[error("Context overflow")]
    ContextOverflow,
    #[error("Token ID is outside the model vocabulary")]
    InvalidToken,
    #[error("Invalid sampling options")]
    InvalidSampling,
    #[error("State belongs to a different model")]
    ForeignState,
    #[error("Invalid multimodal input or request state")]
    InvalidMultimodal,
}

impl<B: Backend> LanguageModel<B> {
    pub fn default_stream_options(&self) -> LanguageModelStreamOptions {
        LanguageModelStreamOptions {
            sampling_method: self.default_sampling_method(),
            #[cfg(grammar)]
            grammar: None,
        }
    }

    pub fn stream<'a>(
        &'a self,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, LanguageModelStreamError<B>> {
        LanguageModelStream::new(self, input, state, options)
    }

    /// Internal projected-feature path. Serving remains disabled until the
    /// processor, vision tower and exact artifact are qualified together.
    pub(crate) fn stream_multimodal<'a>(
        &'a self,
        input: &super::multimodal::MultimodalPrefill,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, LanguageModelStreamError<B>> {
        LanguageModelStream::new_multimodal(self, input, state, options)
    }
}
