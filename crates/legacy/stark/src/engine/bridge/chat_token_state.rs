use std::sync::Arc;

use parking_lot::Mutex;
use shoji::traits::State as LlmInstanceState;
use stark_engine::{backends::common::Backend, engine::language_model::state::LanguageModelState};

pub struct StarkChatTokenBackendInstanceState<B: Backend> {
    pub value: Arc<Mutex<LanguageModelState<B>>>,
}

impl<B: Backend> StarkChatTokenBackendInstanceState<B> {
    pub fn new(value: LanguageModelState<B>) -> Self {
        Self {
            value: Arc::new(Mutex::new(value)),
        }
    }
}

impl<B: Backend> LlmInstanceState for StarkChatTokenBackendInstanceState<B> {}
