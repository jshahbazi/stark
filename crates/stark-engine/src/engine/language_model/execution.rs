//! Architecture dispatch inside the standard language-model execution API.

use std::range::Range;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, InvalidInput},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::{Decoder, DecoderEncodeOutput, DecoderError, EmbeddingReplacement},
        embedding::Embedding,
        qwen38::{Qwen38Decoder, state::Qwen38State},
        transformer::TransformerState,
    },
};

pub(super) enum DecoderExecution<B: Backend> {
    Standard(Box<Decoder<B>>),
    Qwen38(Box<Qwen38Decoder<B>>),
}

pub(super) enum DecoderState<B: Backend> {
    Standard(Box<TransformerState<B>>),
    Qwen38(Box<Qwen38State<B>>),
}

impl<B: Backend> DecoderExecution<B> {
    pub(super) fn vocab_size(&self) -> u32 {
        match self {
            Self::Standard(decoder) => decoder.embedding().vocab_size(),
            Self::Qwen38(decoder) => decoder.vocab_size(),
        }
    }

    pub(super) fn model_dim(&self) -> u32 {
        match self {
            Self::Standard(decoder) => decoder.embedding().model_dim(),
            Self::Qwen38(decoder) => decoder.model_dim(),
        }
    }

    pub(super) fn standard_embedding(&self) -> Option<&Embedding<B>> {
        match self {
            Self::Standard(decoder) => Some(decoder.embedding()),
            Self::Qwen38(_) => None,
        }
    }

    pub(super) fn logits_data_type(&self) -> DataType {
        match self {
            Self::Standard(_) => DataType::BF16,
            Self::Qwen38(_) => DataType::F32,
        }
    }

    /// Paged n-gram lookup needs the actual token before encoding its pass.
    pub(super) fn requires_host_tokens(&self) -> bool {
        match self {
            Self::Standard(_) => false,
            Self::Qwen38(decoder) => decoder.requires_host_tokens(),
        }
    }

    pub(super) fn speculation_supported(&self) -> bool {
        match self {
            Self::Standard(decoder) => decoder.speculation_supported(),
            Self::Qwen38(decoder) => decoder.speculation_supported(),
        }
    }

    pub(super) fn multimodal_supported(&self) -> bool {
        match self {
            Self::Standard(decoder) => decoder.multimodal_supported(),
            Self::Qwen38(_) => false,
        }
    }

    pub(super) fn prefill_cache_skips_trailing_layers(&self) -> bool {
        match self {
            Self::Standard(decoder) => decoder.prefill_cache_skips_trailing_layers(),
            Self::Qwen38(_) => false,
        }
    }

    pub(super) fn prefill_chunk_size(&self) -> usize {
        match self {
            Self::Standard(_) => 1024,
            Self::Qwen38(decoder) => decoder.prefill_chunk_size() as usize,
        }
    }

    pub(super) fn max_context_length(&self) -> Option<u32> {
        match self {
            Self::Standard(decoder) => decoder.max_context_length(),
            Self::Qwen38(decoder) => decoder.max_context_length(),
        }
    }

    pub(super) fn create_empty_state(
        &self,
        maximum: Option<u32>,
        context: &B::Context,
    ) -> Result<DecoderState<B>, B::Error> {
        match self {
            Self::Standard(decoder) => {
                decoder.create_empty_state(maximum, context).map(|state| DecoderState::Standard(Box::new(state)))
            },
            Self::Qwen38(decoder) => {
                decoder.create_empty_state(maximum, context).map(|state| DecoderState::Qwen38(Box::new(state)))
            },
        }
    }

    pub(super) fn encode(
        &self,
        tokens: impl BufferRef<Backend = B>,
        host_tokens: Option<&[u32]>,
        batch: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_layers: Option<&[u32]>,
        state: &mut DecoderState<B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        match (self, state) {
            (Self::Standard(decoder), DecoderState::Standard(state)) => {
                decoder.encode(tokens, batch, output_range, hidden_layers, state, commands)
            },
            (Self::Qwen38(decoder), DecoderState::Qwen38(state)) => {
                let host_tokens = host_tokens.ok_or(InvalidInput("Qwen3.8 requires resolved host token IDs"))?;
                decoder
                    .encode(tokens, host_tokens, batch, output_range, hidden_layers, state, commands)
                    .map_err(DecoderError::Qwen38)
            },
            _ => Err(InvalidInput("decoder state architecture mismatch").into()),
        }
    }

    pub(super) fn encode_with_features(
        &self,
        tokens: impl BufferRef<Backend = B>,
        host_tokens: Option<&[u32]>,
        replacement: Option<EmbeddingReplacement<'_>>,
        batch: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_layers: Option<&[u32]>,
        state: &mut DecoderState<B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        match (self, state) {
            (Self::Standard(decoder), DecoderState::Standard(state)) => {
                decoder.encode_with_features(tokens, replacement, batch, output_range, hidden_layers, state, commands)
            },
            (Self::Qwen38(decoder), DecoderState::Qwen38(state)) => {
                if replacement.is_some() {
                    return Err(InvalidInput("Qwen3.8 text decoder does not accept image features").into());
                }
                let host_tokens = host_tokens.ok_or(InvalidInput("Qwen3.8 requires resolved host token IDs"))?;
                decoder
                    .encode(tokens, host_tokens, batch, output_range, hidden_layers, state, commands)
                    .map_err(DecoderError::Qwen38)
            },
            _ => Err(InvalidInput("decoder state architecture mismatch").into()),
        }
    }
}

impl<B: Backend> DecoderState<B> {
    pub(super) fn ensure_valid(&self) -> Result<(), InvalidInput> {
        if matches!(self, Self::Qwen38(state) if !state.is_valid()) {
            return Err(InvalidInput("Qwen3.8 request state is invalid"));
        }
        Ok(())
    }

    pub(super) fn context_length(&self) -> u32 {
        match self {
            Self::Standard(state) => state.context_length(),
            Self::Qwen38(state) => state.context_length(),
        }
    }

    pub(super) fn rotary_position_delta(&self) -> Option<i64> {
        match self {
            Self::Standard(state) => state.rotary_position_delta(),
            Self::Qwen38(_) => None,
        }
    }

    pub(super) fn set_rotary_position_delta(
        &mut self,
        delta: Option<i64>,
    ) -> Result<(), InvalidInput> {
        match self {
            Self::Standard(state) => state.set_rotary_position_delta(delta),
            Self::Qwen38(_) if delta.is_some() => return Err(InvalidInput("Qwen3.8 has no multimodal rotary offset")),
            Self::Qwen38(_) => {},
        }
        Ok(())
    }

    #[cfg(feature = "scoring-last-layer-pruning")]
    pub(super) fn enable_scoring_last_layer_pruning(&mut self) {
        if let Self::Standard(state) = self {
            state.enable_scoring_last_layer_pruning();
        }
    }

    pub(super) fn prepare(
        &mut self,
        start: u32,
        length: u32,
    ) -> Result<(), B::Error> {
        match self {
            Self::Standard(state) => state.prepare(start, length),
            Self::Qwen38(state) => state.prepare(start, length),
        }
    }

    pub(super) fn encode_accept(
        &mut self,
        indices: &[u32],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        match self {
            Self::Standard(state) => state.encode_accept(indices, commands),
            Self::Qwen38(state) => state.encode_accept(indices, commands),
        }
    }

    pub(super) fn invalidate(&mut self) {
        if let Self::Qwen38(state) = self {
            state.invalidate();
        }
    }
}
