//! Synchronized next-token readout without sampling or persistent prompt state.
use std::time::Instant;

use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferCpuAccessible, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, gpu_types::trie::TrieNode as GpuTrieNode,
    },
    encodable_block::{batch_topology::BatchTopology, decoder::DecoderError},
    engine::language_model::{LanguageModel, state::LanguageModelCreateEmptyStateError},
};

// Exact scoring always prefills a linear sequence. Its topology depends only on
// chunk length; neither token identities nor sampling seeds affect these nodes.
fn flat_scoring_topology(length: u32) -> Box<[GpuTrieNode]> {
    assert!(length > 0, "scoring chunk must be nonempty");
    (0..length)
        .map(|index| GpuTrieNode {
            trie_start: index,
            trie_end: length - 1,
            height: index,
        })
        .collect()
}

// Drain every submitted buffer even when an earlier completion failed. GPU work
// must stop using transformer state before that state is freed on any exit.
fn drain_submissions<T, E>(
    pending: &mut Vec<T>,
    mut wait: impl FnMut(T) -> Result<(), E>,
) -> Result<(), E> {
    let mut first_error = None;
    for item in pending.drain(..) {
        if let Err(error) = wait(item)
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

struct ScoringSubmissions<B: Backend> {
    pending: Vec<<B::CommandBuffer as CommandBuffer>::Pending>,
}

impl<B: Backend> ScoringSubmissions<B> {
    fn wait_all(&mut self) -> Result<(), B::Error> {
        drain_submissions(&mut self.pending, |pending| pending.wait_until_completed().map(|_| ()))
    }
}

impl<B: Backend> Drop for ScoringSubmissions<B> {
    fn drop(&mut self) {
        // Also runs on later encode/allocation failure or panic. The caller
        // discards this fresh state after an error; never reuse partial state.
        let _ = self.wait_all();
    }
}

/// Final prompt-position raw logits. No temperature, penalty, grammar, or sampler
/// has been applied. Every call creates and destroys its own transformer state.
pub struct LanguageModelNextTokenScores {
    pub logits: Vec<f32>,
    pub state_creation_seconds: f64,
    pub prefill_seconds: f64,
    pub readout_seconds: f64,
}

#[derive(Debug, Error)]
pub enum LanguageModelScoreError<B: Backend> {
    #[error("Invalid scoring input")]
    InvalidInput,
    #[error("Context overflow")]
    ContextOverflow,
    #[error("Speculative model not supported")]
    SpeculatorNotSupported,
    #[error("Backend scoring failure")]
    Backend(#[source] B::Error),
    #[error("Decoder scoring failure")]
    Decoder(#[from] DecoderError<B>),
    #[error("Scoring state creation failure")]
    State(#[from] LanguageModelCreateEmptyStateError<B>),
    #[error("Invalid scoring output")]
    InvalidOutput,
}

impl<B: Backend> LanguageModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
    /// Evaluate exact token IDs with a fresh state and read only the last prompt
    /// position. Matches ordinary stream prefill batching and trailing-layer
    /// splitting, but does not sample or schedule a generated-token forward pass.
    ///
    /// Contract:
    /// - `input` must be non-empty, every ID must be below the vocabulary size,
    ///   and `input.len()` must not exceed `context_tokens`, which must be at
    ///   least 1 and at most the model's maximum context. Violations return
    ///   `InvalidInput` or `ContextOverflow` before any GPU work.
    /// - `context_tokens` sizes the per-layer KV capacity of the temporary
    ///   state; nothing is cached between calls and the state is dropped on
    ///   every exit path, including errors.
    /// - Logits are read from the engine's BF16 readout and widened to `f32`;
    ///   `InvalidOutput` covers both a missing readout and non-finite values.
    /// - The timing fields are wall-clock seconds; `readout_seconds` includes
    ///   the full-vocabulary conversion.
    /// - `&self` is not a concurrency guarantee: callers must serialize calls
    ///   against one loaded model.
    pub fn score_next_token(
        &self,
        input: &[u32],
        context_tokens: u32,
    ) -> Result<LanguageModelNextTokenScores, LanguageModelScoreError<B>> {
        let vocab_size = self.decoder.embedding().vocab_size() as usize;
        if input.is_empty() || context_tokens == 0 || input.iter().any(|id| *id as usize >= vocab_size) {
            return Err(LanguageModelScoreError::InvalidInput);
        }
        if input.len() > context_tokens as usize
            || self.max_context_length().is_some_and(|maximum| context_tokens > maximum)
        {
            return Err(LanguageModelScoreError::ContextOverflow);
        }
        if self.speculator.is_some() {
            return Err(LanguageModelScoreError::SpeculatorNotSupported);
        }
        let started = Instant::now();
        let mut state = self.create_empty_state(Some(context_tokens), 0)?;
        #[cfg(feature = "scoring-last-layer-pruning")]
        state.transformer_state.enable_scoring_last_layer_pruning();
        let state_creation_seconds = started.elapsed().as_secs_f64();
        let prefill_started = Instant::now();
        // Ordinary stream prefill requires this batch size for attention's
        // hardcoded 1024-token suffix; preserve the same preparation here.
        let max_batch_size = 1024;
        let number_of_batches = input.len().div_ceil(max_batch_size);
        state
            .transformer_state
            .prepare(((number_of_batches - 1) * max_batch_size) as u32, usize::min(max_batch_size, input.len()) as u32)
            .map_err(LanguageModelScoreError::Backend)?;
        let mut final_logits = None;
        // Declared after state/readout so this guard drains queued GPU work
        // before either resource can be dropped during an error unwind.
        let mut submissions = ScoringSubmissions::<B> {
            pending: Vec::with_capacity(number_of_batches),
        };
        for (batch_idx, batch) in input.chunks(max_batch_size).enumerate() {
            // Independent pools prevent host encoding of the next batch from
            // recycling scratch storage still used by an in-flight batch.
            let allocation_pool = self.engine.context.create_allocation_pool();
            let mut command_buffer = self
                .engine
                .context
                .create_command_buffer(Some("score prefill batch"), Some(allocation_pool))
                .map_err(LanguageModelScoreError::Backend)?;
            let last_batch = batch_idx + 1 == number_of_batches;
            let split = last_batch && self.decoder.prefill_requires_separate_logits_row() && batch.len() > 1;
            let parts = if split {
                [Some((&batch[..batch.len() - 1], false)), Some((&batch[batch.len() - 1..], true))]
            } else {
                [Some((batch, last_batch)), None]
            };
            for (input_chunk, output_last) in parts.into_iter().flatten() {
                let token_ids = command_buffer
                    .allocate_constant_from_slice(input_chunk)
                    .map_err(LanguageModelScoreError::Backend)?;
                let nodes = flat_scoring_topology(input_chunk.len() as u32);
                let batch_dim = BatchTopology::new(&nodes, true);
                let output = self.decoder.encode(
                    &token_ids,
                    &batch_dim,
                    output_last.then(|| (input_chunk.len() as u32 - 1..input_chunk.len() as u32).into()),
                    None,
                    &mut state.transformer_state,
                    &mut command_buffer,
                )?;
                if output_last {
                    final_logits = output.logits;
                }
                // Only subsequent chunks need the accepted state. Final logits
                // are already encoded, and this fresh state is dropped below.
                if !output_last {
                    state
                        .transformer_state
                        .encode_accept(&(0..input_chunk.len() as u32).collect::<Box<[u32]>>(), &mut command_buffer)
                        .map_err(LanguageModelScoreError::Backend)?;
                }
            }
            // Preserve the complete 1024-token batch and final split geometry.
            // Metal's command-buffer queue barriers order state writes before
            // the next batch while CPU encoding overlaps earlier GPU execution.
            submissions.pending.push(command_buffer.end_encoding().submit());
        }
        submissions.wait_all().map_err(LanguageModelScoreError::Backend)?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        let readout_started = Instant::now();
        let final_logits = final_logits.ok_or(LanguageModelScoreError::InvalidOutput)?;
        // Engine::load_language_model fixes internal readout data type to BF16.
        let logits: Vec<f32> = (&final_logits).as_slice::<bf16>().iter().map(|value| value.to_f32()).collect();
        if logits.len() != vocab_size || logits.iter().any(|value| !value.is_finite()) {
            return Err(LanguageModelScoreError::InvalidOutput);
        }
        let readout_seconds = readout_started.elapsed().as_secs_f64();
        Ok(LanguageModelNextTokenScores {
            logits,
            state_creation_seconds,
            prefill_seconds,
            readout_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{encodable_block::sampling::PRng, trie::TrieNode};

    #[stark_test]
    fn submission_drain_waits_every_buffer_after_failures() {
        let mut pending = vec![0, 1, 2, 3];
        let mut visited = Vec::new();
        let result = drain_submissions(&mut pending, |index| {
            visited.push(index);
            if index == 1 || index == 2 {
                Err(index)
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err(1));
        assert_eq!(visited, vec![0, 1, 2, 3]);
        assert!(pending.is_empty());
        let second: Result<(), usize> = drain_submissions(&mut pending, |_| panic!("already drained"));
        assert_eq!(second, Ok(()));
    }

    #[stark_test]
    fn submission_drain_handles_success_and_empty_queues() {
        let mut pending = vec![0, 1];
        let mut completions = 0;
        let result: Result<(), usize> = drain_submissions(&mut pending, |_| {
            completions += 1;
            Ok(())
        });
        assert_eq!(result, Ok(()));
        assert_eq!(completions, 2);
        assert!(pending.is_empty());
    }

    #[stark_test]
    fn direct_scoring_topology_matches_legacy_flat_trie() {
        let prng = PRng::new(0);
        for length in [1, 2, 1023, 1024] {
            let tokens = vec![1_u64; length];
            for prefix_length in [0, 1024, 7168] {
                let trie = TrieNode::flat(prefix_length, &tokens, &prng);
                let flat = trie.linearize();
                let legacy: Box<[GpuTrieNode]> = flat.token_subtrie_ranges().collect();
                let direct = flat_scoring_topology(length as u32);
                assert_eq!(direct.len(), legacy.len());
                for (actual, expected) in direct.iter().zip(legacy.iter()) {
                    assert_eq!(actual.trie_start, expected.trie_start);
                    assert_eq!(actual.trie_end, expected.trie_end);
                    assert_eq!(actual.height, expected.height);
                }
                let actual = BatchTopology::new(&direct, true);
                let expected = BatchTopology::new(&legacy, true);
                assert_eq!(actual.size(), expected.size());
                assert_eq!(actual.is_flat(), expected.is_flat());
                assert_eq!(actual.full_accept(), expected.full_accept());
                assert!(actual.parents() == expected.parents());
            }
        }
    }
}
