use std::{
    iter::{once, repeat_n},
    mem::replace,
    range::Range,
    sync::Arc,
};

#[cfg(grammar)]
use crate::engine::language_model::grammar::Grammar;
use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, gpu_types::trie::TrieNode as GpuTrieNode, kernel::ContextRingUpdateKernel,
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::EmbeddingReplacement, sampling::SamplingMethod},
    engine::{
        capture::CaptureSpan,
        language_model::{
            LanguageModel,
            multimodal::MultimodalPrefill,
            state::LanguageModelState,
            stream::{LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
    trie::TrieNode,
};

enum ForwardPassChaining<B: Backend> {
    Constant {
        token: u64,
        output_norm: Option<B::ScratchBuffer>,
        completed_pool: Option<Arc<B::AllocationPool>>,
    },
    InFlight(DecodingStatePending<B>),
}

impl<B: Backend> ForwardPassChaining<B> {
    fn resolve<'a>(
        &'a mut self,
        tokens: &mut Vec<u64>,
        #[cfg(grammar)] grammar: Option<&mut Grammar>,
    ) -> Result<(u64, Option<&'a B::ScratchBuffer>), LanguageModelStreamError<B>> {
        match self {
            Self::Constant {
                token,
                output_norm,
                ..
            } => Ok((*token, output_norm.as_ref())),
            Self::InFlight(in_flight) => {
                assert!(in_flight.full_accept);
                let completed_pool = in_flight.pending.take().unwrap().wait(|pending| {
                    pending.wait_until_completed().map(|_| ()).map_err(LanguageModelStreamError::Backend)
                })?;
                let output_tokens = in_flight.output_tokens.as_slice::<u32>();
                assert_eq!(output_tokens.len(), 1);
                let token_id = output_tokens[0] as u64;
                let output_norm = in_flight.output_norm.take();
                *self = Self::Constant {
                    token: token_id,
                    output_norm,
                    completed_pool,
                };
                tokens.push(token_id);
                #[cfg(grammar)]
                if let Some(grammar) = grammar {
                    grammar.accept_token(token_id)?;
                }
                let Self::Constant {
                    output_norm,
                    ..
                } = self
                else {
                    unreachable!()
                };
                Ok((token_id, output_norm.as_ref()))
            },
        }
    }

    fn drop_output_norm(&mut self) {
        match self {
            Self::Constant {
                output_norm,
                ..
            } => drop(output_norm.take()),
            Self::InFlight(pending) => drop(pending.output_norm.take()),
        }
    }

    fn take_completed_pool(&mut self) -> Option<Arc<B::AllocationPool>> {
        match self {
            Self::Constant {
                completed_pool,
                ..
            } => completed_pool.take(),
            Self::InFlight(_) => None,
        }
    }
}

/// Keep a pool unavailable until every submission using its pages completes.
/// Dropping a CPU pending handle alone is not a GPU completion signal.
struct SubmittedScratchPool<P, S> {
    pending: Vec<P>,
    reusable_pool: Option<S>,
    wait_on_drop: Option<fn(P)>,
}

impl<P, S> SubmittedScratchPool<P, S> {
    fn new(
        pending: Box<[P]>,
        reusable_pool: Option<S>,
    ) -> Self {
        Self {
            pending: pending.into_vec(),
            reusable_pool,
            wait_on_drop: None,
        }
    }

    fn with_drop_wait(
        mut self,
        enabled: bool,
        wait: fn(P),
    ) -> Self {
        self.wait_on_drop = enabled.then_some(wait);
        self
    }

    fn push(
        &mut self,
        pending: P,
    ) {
        self.pending.push(pending);
    }

    fn wait<E>(
        mut self,
        mut wait: impl FnMut(P) -> Result<(), E>,
    ) -> Result<Option<S>, E> {
        let mut first_error = None;
        for pending in std::mem::take(&mut self.pending) {
            if let Err(error) = wait(pending)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        // Drain every submission on failure, but never publish its pool for reuse.
        match first_error {
            Some(error) => Err(error),
            None => Ok(self.reusable_pool.take()),
        }
    }
}

impl<P, S> Drop for SubmittedScratchPool<P, S> {
    fn drop(&mut self) {
        if let Some(wait) = self.wait_on_drop {
            // Error exits may abandon a local forward pass before it can be
            // stored on the stream. Complete it while its pool is still owned.
            for pending in std::mem::take(&mut self.pending) {
                wait(pending);
            }
        }
    }
}

struct DecodingStatePending<B: Backend> {
    input_trie: TrieNode,
    full_accept: bool,
    pending: Option<SubmittedScratchPool<<B::CommandBuffer as CommandBuffer>::Pending, Arc<B::AllocationPool>>>,
    capture_span: Option<CaptureSpan<B>>,
    hidden_features: Option<Box<[B::ScratchBuffer]>>,
    output_norm: Option<B::ScratchBuffer>,
    output_tokens: Arc<B::GlobalBuffer>,
    _input_token_source: Option<Arc<B::GlobalBuffer>>,
}

enum DecodingState<B: Backend> {
    Seeded {
        seed_token: u64,
    },
    ForwardPassPending(DecodingStatePending<B>),
    Accepting {
        full: Box<[(usize, u64, u64)]>,
        num_accepted: usize,
        hidden_features: Option<Box<[B::ScratchBuffer]>>,
        output_norm: Option<(B::ScratchBuffer, usize)>,
        capture_span: Option<CaptureSpan<B>>,
    },
    Halted,
    Invalid,
}

fn prefill_chunk_parts(
    input_chunk: &[u64],
    last_batch: bool,
    split_logits_row: bool,
) -> [Option<(&[u64], bool)>; 2] {
    if last_batch && split_logits_row && input_chunk.len() > 1 {
        let (prompt_chunk, sample_chunk) = input_chunk.split_at(input_chunk.len() - 1);
        [Some((prompt_chunk, false)), Some((sample_chunk, true))]
    } else {
        [Some((input_chunk, last_batch)), None]
    }
}

#[derive(Debug, Clone, Default)]
pub struct TokenStreamMetrics {
    pub num_prefill_forward_passes: usize,
    pub num_decode_forward_passes: usize,
    pub num_tokens_prefilled: usize,
    pub num_tokens_proposed: usize,
    pub num_tokens_accepted: usize,
    pub num_tokens_returned: usize,
}

pub struct LanguageModelStream<'a, B: Backend> {
    model: &'a LanguageModel<B>,
    model_state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions,
    allocation_pool: Arc<B::AllocationPool>,
    reusable_decode_pool: Option<Arc<B::AllocationPool>>,
    reuse_decode_pool: bool,
    context_ring: Option<B::GlobalBuffer>,
    decoding_state: DecodingState<B>,
    metrics: TokenStreamMetrics,
    failure_fence: Option<&'a mut bool>,
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub fn new(
        model: &'a LanguageModel<B>,
        input: &[u64],
        model_state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        Self::new_impl(model, input, None, model_state, options)
    }

    pub(crate) fn new_multimodal(
        model: &'a LanguageModel<B>,
        input: &MultimodalPrefill,
        model_state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        Self::new_impl(model, &input.tokens, Some(input), model_state, options)
    }

    fn new_impl(
        model: &'a LanguageModel<B>,
        input: &[u64],
        multimodal: Option<&MultimodalPrefill>,
        model_state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        if !Arc::ptr_eq(&model.model_identity, &model_state.model_identity) {
            return Err(LanguageModelStreamError::ForeignState);
        }
        if model_state.multimodal_failed {
            return Err(LanguageModelStreamError::InvalidMultimodal);
        }
        if let Some(multimodal) = multimodal {
            model.preflight_multimodal_stream(input, model_state, &options)?;
            if !model.decoder.multimodal_supported()
                || model.speculator.is_some()
                || !model_state.tokens.is_empty()
                || model_state.transformer_state.context_length() != 0
                || model_state.last_output_token.is_some()
                || model_state.has_multimodal_context()
                || !Arc::ptr_eq(&multimodal.image.model_identity, &model.model_identity)
                || multimodal.image.model_dim != model.decoder.embedding().model_dim()
                || multimodal.image.data_type != DataType::BF16
                || multimodal.positions.len() != input.len()
                || (multimodal.uses_spatial_positions() && !model.decoder.qwen35_multimodal_supported())
                || (!multimodal.uses_spatial_positions() && !model.decoder.gemma4_multimodal_supported())
                || (!multimodal.uses_spatial_positions()
                    && (multimodal.position_delta != 0
                        || multimodal.positions.iter().enumerate().any(|(index, position)| {
                            u32::try_from(index).map_or(true, |index| *position != [index; 3])
                        })))
            {
                return Err(LanguageModelStreamError::InvalidMultimodal);
            }
        }
        let vocab_size = model.sampling.vocab_size();
        if !valid_stream_tokens(
            vocab_size,
            input.iter().chain(&model_state.tokens).chain(model_state.last_output_token.iter()),
        ) {
            return Err(LanguageModelStreamError::InvalidToken);
        }
        options.sampling_method.validate(vocab_size).map_err(|_| LanguageModelStreamError::InvalidSampling)?;
        #[cfg(grammar)]
        let mut options = options;
        if input.is_empty() && model_state.last_output_token.is_none() {
            return Err(LanguageModelStreamError::NoSeedToken);
        };

        if model_state
            .max_context_length
            .is_some_and(|max_context_length| model_state.tokens.len() + input.len() > max_context_length as usize)
        {
            return Err(LanguageModelStreamError::ContextOverflow);
        }

        // After this point failures fence this per-request state. A fresh state
        // is required rather than reusing partially encoded image/cache rows.
        let multimodal_request = multimodal.is_some() || model_state.has_multimodal_context();
        if multimodal_request {
            model_state.multimodal_failed = true;
        }
        if let Some(multimodal) = multimodal {
            model_state.multimodal_prefilled = true;
            model_state
                .transformer_state
                .set_rotary_position_delta(multimodal.uses_spatial_positions().then_some(multimodal.position_delta));
        }

        let setup = (|| -> Result<_, LanguageModelStreamError<B>> {
            let capture_span = if !model_state.has_multimodal_context()
                && let Some(capture_manager) = &model.engine.capture_manager
                && let Some(capture_request) = capture_manager.maybe_capture_prefill_step()
            {
                Some(capture_request.start().map_err(LanguageModelStreamError::Backend)?)
            } else {
                None
            };

            let allocation_pool = model.engine.context.create_allocation_pool();

            let mut context_ring =
                if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                    let mut context_ring = model
                        .engine
                        .context
                        .create_buffer(size_for_shape(&[2 + suffix_repetition_length], DataType::U32))
                        .map_err(LanguageModelStreamError::Backend)?;

                    let state_tokens_range = Range::from(
                        model_state.tokens.len().saturating_sub(suffix_repetition_length as usize)
                            ..model_state.tokens.len(),
                    );

                    context_ring.copyin(
                        &once(0) // offset
                            .chain(once(state_tokens_range.iter().len() as u64)) // length
                            .chain(model_state.tokens[state_tokens_range].iter().copied()) // tokens
                            .chain(repeat_n(0, suffix_repetition_length as usize - state_tokens_range.iter().len())) // pad if not full
                            .map(|x| x as u32)
                            .collect::<Box<[_]>>(),
                    );

                    Some(context_ring)
                } else {
                    None
                };

            let mut metrics = TokenStreamMetrics::default();

            let decoding_state = if !input.is_empty() {
                model_state.last_output_token.take();

                // NOTE: this is required for attention correctness (hardcoded suffix 1024). This is really bad design, attention should be rewritten to allow on-demand suffix length
                let max_batch_size = 1024;
                let number_of_batches = input.len().div_ceil(max_batch_size);

                model_state
                    .transformer_state
                    .prepare(
                        model_state.transformer_state.context_length()
                            + ((number_of_batches - 1) * max_batch_size) as u32,
                        usize::min(max_batch_size, input.len()) as u32,
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                let mut command_buffer = model
                    .engine
                    .context
                    .create_command_buffer(Some("prefill"), Some(allocation_pool.clone()))
                    .map_err(LanguageModelStreamError::Backend)?;

                let mut output_tokens = None;
                let mut output_norm = None;
                let split_logits_row = model.decoder.prefill_requires_separate_logits_row();
                let hidden_feature_layer_indices =
                    model.speculator.as_ref().map(|speculator| speculator.hidden_feature_layer_indices());
                #[cfg(test)]
                let vision_layer_indices = model_state
                    .vision_prefill_layer_readbacks
                    .as_ref()
                    .map(|(_, buffers)| (0..buffers.len() as u32).collect::<Box<[_]>>());
                #[cfg(test)]
                let hidden_feature_layer_indices = vision_layer_indices.as_deref().or(hidden_feature_layer_indices);

                for (input_chunk, sample_last) in input
                    .chunks(max_batch_size)
                    .enumerate()
                    .flat_map(|(batch_idx, input_chunk)| {
                        prefill_chunk_parts(input_chunk, batch_idx == number_of_batches - 1, split_logits_row)
                    })
                    .flatten()
                {
                    let input_trie = TrieNode::flat(model_state.tokens.len(), input_chunk, &model_state.prng);
                    let input_flat_trie = input_trie.linearize();

                    let token_ids = command_buffer
                        .allocate_constant_from_slice(
                            &input_chunk.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>(),
                        )
                        .map_err(LanguageModelStreamError::Backend)?;

                    let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
                    let chunk_start = model_state.tokens.len();
                    let batch_dim = match multimodal {
                        Some(input) if input.uses_spatial_positions() => BatchTopology::with_rotary_positions(
                            &input_flat_trie_nodes,
                            true,
                            &input.positions[chunk_start..chunk_start + input_chunk.len()],
                        )
                        .map_err(|_| LanguageModelStreamError::InvalidMultimodal)?,
                        _ => BatchTopology::new(&input_flat_trie_nodes, true),
                    };
                    let replacement = multimodal
                        .map(|input| {
                            input.chunk_feature_range(chunk_start, input_chunk.len()).map(|range| {
                                range.map(|(tokens, features)| {
                                    let width = input.image.model_dim as usize;
                                    EmbeddingReplacement {
                                        token_start: tokens.start as u32,
                                        features: &input.image.features[features.start * width..features.end * width],
                                    }
                                })
                            })
                        })
                        .transpose()
                        .map_err(|_| LanguageModelStreamError::InvalidMultimodal)?
                        .flatten();

                    let decoder_output = model.decoder.encode_with_features(
                        &token_ids,
                        replacement,
                        &batch_dim,
                        sample_last.then(|| (input_chunk.len() as u32 - 1..input_chunk.len() as u32).into()),
                        hidden_feature_layer_indices,
                        &mut model_state.transformer_state,
                        &mut command_buffer,
                    )?;
                    // Test-only private layer traces reuse the existing
                    // residual-combined hidden-feature path. Rows are copied
                    // into request-owned readbacks across prefill chunks.
                    #[cfg(test)]
                    if let Some((row_bytes, readbacks)) = &mut model_state.vision_prefill_layer_readbacks {
                        let features = decoder_output.hidden_features.as_ref().unwrap();
                        if features.len() != readbacks.len() {
                            return Err(LanguageModelStreamError::InvalidMultimodal);
                        }
                        for (feature, readback) in features.iter().zip(readbacks.iter_mut()) {
                            let start = chunk_start * *row_bytes;
                            let end = start + input_chunk.len() * *row_bytes;
                            if end > readback.size() || feature.size() != end - start {
                                return Err(LanguageModelStreamError::InvalidMultimodal);
                            }
                            command_buffer.encode_copy(feature, readback.subrange_mut(start..end));
                        }
                    }
                    let logits = decoder_output.logits;

                    if sample_last {
                        let logits = logits.unwrap();

                        // Private validation observes the final prompt row
                        // before penalties or sampling. Ownership stays in the
                        // request state until stream Drop completes the pass.
                        #[cfg(test)]
                        if let Some(readback) = &mut model_state.vision_prefill_logits_readback {
                            if (&logits).size() != readback.size() {
                                return Err(LanguageModelStreamError::InvalidMultimodal);
                            }
                            command_buffer.encode_copy(&logits, readback);
                        }

                        let seeds = if matches!(options.sampling_method, SamplingMethod::Stochastic { .. }) {
                            Some(
                                command_buffer
                                    .allocate_constant_from_slice(&[model_state
                                        .prng
                                        .derive((model_state.tokens.len() + input_chunk.len() - 1) as u64)])
                                    .map_err(LanguageModelStreamError::Backend)?,
                            )
                        } else {
                            None
                        };

                        #[cfg(grammar)]
                        let bitmask = if let Some(grammar) = options.grammar.as_mut() {
                            let mut bitmask = command_buffer
                                .allocate_constant(
                                    model.vocab_size.div_ceil(DataType::U32.size_in_bits())
                                        * DataType::U32.size_in_bytes(),
                                )
                                .map_err(LanguageModelStreamError::Backend)?;

                            if grammar.next_bitmask(bitmask.as_slice_mut()) {
                                Some(bitmask)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        #[cfg(not(grammar))]
                        let bitmask = None::<B::ConstantBuffer>;

                        output_norm = decoder_output.final_hidden;
                        let sampled_row = batch_dim.size() - 1;
                        output_tokens = Some(
                            model
                                .sampling
                                .encode(
                                    &logits,
                                    seeds.as_ref(),
                                    bitmask.as_ref(),
                                    context_ring.as_ref(),
                                    Some(&token_ids),
                                    &options.sampling_method,
                                    &batch_dim,
                                    (sampled_row..sampled_row + 1).into(),
                                    &mut command_buffer,
                                )
                                .map_err(LanguageModelStreamError::Backend)?,
                        );
                    }

                    model_state
                        .transformer_state
                        .encode_accept(&(0..input_chunk.len() as u32).collect::<Box<[u32]>>(), &mut command_buffer)
                        .map_err(LanguageModelStreamError::Backend)?;

                    if let Some(speculator) = model.speculator.as_ref() {
                        let speculator_state = model_state.speculator_state.as_mut().unwrap();
                        speculator
                            .encode_accept(
                                speculator_state,
                                decoder_output.hidden_features.as_ref().unwrap().iter(),
                                &(0..input_chunk.len() as u32).collect::<Box<[u32]>>(),
                                &mut command_buffer,
                            )
                            .map_err(LanguageModelStreamError::Backend)?;
                    }

                    if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                        model.context_ring_update.encode(
                            &token_ids,
                            context_ring.as_mut().unwrap(),
                            suffix_repetition_length,
                            input_chunk.len() as u32,
                            &mut command_buffer,
                        );
                    }

                    model_state.tokens.extend(input_chunk);
                }

                let pending = Box::new([command_buffer.end_encoding().submit()]);

                metrics.num_prefill_forward_passes += 1;
                metrics.num_tokens_prefilled += input.len();
                metrics.num_tokens_proposed += 1;
                metrics.num_tokens_accepted += 1;

                DecodingState::ForwardPassPending(DecodingStatePending {
                    input_trie: TrieNode::new(0, 0, 0.0),
                    full_accept: true,
                    // Large prefill workspaces are released rather than retained
                    // for the stream's single-token decode pool rotation.
                    pending: Some(SubmittedScratchPool::new(pending, None).with_drop_wait(
                        multimodal_request,
                        |pending| {
                            let _ = pending.wait_until_completed();
                        },
                    )),
                    capture_span,
                    hidden_features: None,
                    output_norm,
                    output_tokens: Arc::new(output_tokens.unwrap()),
                    _input_token_source: None,
                })
            } else {
                // TODO: this leaks previous LanguageModelStreamOptions
                DecodingState::Seeded {
                    seed_token: model_state.last_output_token.take().unwrap(),
                }
            };

            // Speculation and grammar can submit extra work on the active pool.
            // Keep their existing pool lifecycle until those paths have their own
            // completion accounting. Ordinary decode has at most two passes live.
            let reuse_decode_pool = model.speculator.is_none();
            #[cfg(grammar)]
            let reuse_decode_pool = reuse_decode_pool && options.grammar.is_none();

            Ok((allocation_pool, context_ring, metrics, decoding_state, reuse_decode_pool))
        })();
        let (allocation_pool, context_ring, metrics, decoding_state, reuse_decode_pool) = match setup {
            Ok(setup) => setup,
            Err(error) => {
                if multimodal_request {
                    fence_multimodal_state(model_state);
                }
                return Err(error);
            },
        };
        model_state.multimodal_failed = false;
        Ok(LanguageModelStream {
            model,
            model_state,
            options,
            allocation_pool,
            reusable_decode_pool: None,
            reuse_decode_pool,
            context_ring,
            decoding_state,
            metrics,
            failure_fence: None,
        })
    }

    /// Tie a private vision stream's native failures to its loaded instance.
    /// The exclusive borrow remains held until Drop drains pending GPU work.
    pub(crate) fn with_failure_fence(
        mut self,
        fence: &'a mut bool,
    ) -> Self {
        self.failure_fence = Some(fence);
        self
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let (mut prev_output, mut command_buffer): (
            ForwardPassChaining<B>,
            Option<<B::CommandBuffer as CommandBuffer>::Encoding>,
        ) = match replace(&mut self.decoding_state, DecodingState::Invalid) {
            DecodingState::Seeded {
                seed_token,
            } => {
                self.model_state.tokens.push(seed_token);
                #[cfg(grammar)]
                if let Some(grammar) = self.options.grammar.as_mut() {
                    let _ = grammar.accept_token(seed_token); // TODO: this should not be ignored
                }
                self.metrics.num_tokens_returned += 1;
                (
                    ForwardPassChaining::Constant {
                        token: seed_token,
                        output_norm: None,
                        completed_pool: None,
                    },
                    None,
                )
            },
            DecodingState::ForwardPassPending(forward_pass_pending) => {
                if forward_pass_pending.full_accept {
                    self.metrics.num_tokens_returned += 1;
                    (ForwardPassChaining::InFlight(forward_pass_pending), None)
                } else {
                    forward_pass_pending.pending.unwrap().wait(|pending| {
                        pending.wait_until_completed().map(|_| ()).map_err(LanguageModelStreamError::Backend)
                    })?;
                    let sampled_tokens = forward_pass_pending
                        .output_tokens
                        .as_slice::<u32>()
                        .iter()
                        .map(|x| *x as u64)
                        .collect::<Box<[u64]>>();
                    let flat_trie = forward_pass_pending.input_trie.linearize();
                    let full = flat_trie.accept(
                        &sampled_tokens,
                        #[cfg(grammar)]
                        self.options.grammar.as_mut(),
                    )?;
                    let output_norm = forward_pass_pending.output_norm.map(|norm| {
                        let row_bytes = norm.size() / flat_trie.len();
                        (norm, row_bytes)
                    });
                    self.metrics.num_tokens_accepted += full.len();
                    self.decoding_state = DecodingState::Accepting {
                        full,
                        num_accepted: 0,
                        hidden_features: forward_pass_pending.hidden_features,
                        output_norm,
                        capture_span: forward_pass_pending.capture_span,
                    };
                    return self.generate();
                }
            },
            DecodingState::Accepting {
                full,
                num_accepted,
                hidden_features,
                output_norm,
                capture_span,
            } => {
                let output_token_id = full[num_accepted].2;

                self.metrics.num_tokens_returned += 1;

                if num_accepted < full.len() - 1 {
                    self.decoding_state = DecodingState::Accepting {
                        full,
                        num_accepted: num_accepted + 1,
                        hidden_features,
                        output_norm,
                        capture_span,
                    };
                    return Ok(Some(output_token_id));
                } else {
                    let accepted_token_indicies = full.iter().map(|(i, _, _)| *i as u32).collect::<Box<[u32]>>();
                    let accepted_input_token_ids = full.iter().map(|(_, t, _)| *t).collect::<Box<[u64]>>();
                    let accepted_output_token_ids = full.iter().map(|(_, _, t)| *t).collect::<Box<[u64]>>();
                    let mut command_buffer = self
                        .model
                        .engine
                        .context
                        .create_command_buffer(Some("decode"), Some(self.allocation_pool.clone()))
                        .map_err(LanguageModelStreamError::Backend)?;
                    self.model_state
                        .transformer_state
                        .encode_accept(&accepted_token_indicies, &mut command_buffer)
                        .map_err(LanguageModelStreamError::Backend)?;
                    if let Some(speculator) = self.model.speculator.as_ref() {
                        speculator
                            .encode_accept(
                                self.model_state.speculator_state.as_mut().unwrap(),
                                hidden_features.as_ref().unwrap().iter(),
                                &accepted_token_indicies,
                                &mut command_buffer,
                            )
                            .map_err(LanguageModelStreamError::Backend)?;
                    }
                    let output_norm = if let Some((final_hidden, row_bytes)) = output_norm {
                        let row = full.last().unwrap().0;
                        let mut norm =
                            command_buffer.allocate_scratch(row_bytes).map_err(LanguageModelStreamError::Backend)?;
                        command_buffer
                            .encode_copy(final_hidden.subrange(row * row_bytes..(row + 1) * row_bytes), &mut norm);
                        Some(norm)
                    } else {
                        None
                    };
                    if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
                        command_buffer.push_debug_group("update repetition penalty ring");
                        let accepted_input_token_ids_const = command_buffer
                            .allocate_constant_from_slice(
                                &accepted_input_token_ids
                                    .iter()
                                    .map(|token_id| *token_id as u32)
                                    .collect::<Box<[u32]>>(),
                            )
                            .map_err(LanguageModelStreamError::Backend)?;
                        self.model.context_ring_update.encode(
                            &accepted_input_token_ids_const,
                            self.context_ring.as_mut().unwrap(),
                            suffix_repetition_length,
                            full.len() as u32,
                            &mut command_buffer,
                        );
                        command_buffer.pop_debug_group();
                    }
                    if let Some(capture_span) = capture_span {
                        command_buffer
                            .end_encoding()
                            .submit()
                            .wait_until_completed()
                            .map_err(LanguageModelStreamError::Backend)?;

                        drop(capture_span);

                        command_buffer = self
                            .model
                            .engine
                            .context
                            .create_command_buffer(Some("decode"), Some(self.allocation_pool.clone()))
                            .map_err(LanguageModelStreamError::Backend)?;
                    }
                    self.model_state.tokens.extend(accepted_output_token_ids);
                    (
                        ForwardPassChaining::Constant {
                            token: output_token_id,
                            output_norm,
                            completed_pool: None,
                        },
                        Some(command_buffer),
                    )
                }
            },
            DecodingState::Halted => return Ok(None),
            DecodingState::Invalid => unreachable!(),
        };

        let context_length = self.model_state.transformer_state.context_length();

        if self.model_state.max_context_length.is_some_and(|max_context_length| context_length >= max_context_length) {
            self.decoding_state = DecodingState::Halted;
            return Ok(Some(
                prev_output
                    .resolve(
                        &mut self.model_state.tokens,
                        #[cfg(grammar)]
                        self.options.grammar.as_mut(),
                    )?
                    .0,
            ));
        }

        let capture_span = if !self.model_state.has_multimodal_context()
            && let Some(capture_manager) = &self.model.engine.capture_manager
            && let Some(capture_request) = capture_manager.maybe_capture_decode_step()
        {
            prev_output.resolve(
                &mut self.model_state.tokens,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
            )?;
            Some(capture_request.start().map_err(LanguageModelStreamError::Backend)?)
        } else {
            None
        };

        let mut pending = SubmittedScratchPool::new(Box::new([]), None::<Arc<B::AllocationPool>>).with_drop_wait(
            self.model_state.has_multimodal_context(),
            |pending: <B::CommandBuffer as CommandBuffer>::Pending| {
                let _ = pending.wait_until_completed();
            },
        );
        let (input_trie, chain_copy, full_accept) = if let Some(speculator) = &self.model.speculator
            && let Some(shape) = speculator.make_shape(
                self.model_state.max_context_length.map(|max_context_length| max_context_length - context_length),
                &self.options.sampling_method,
            )
            && let (root_token, Some(output_norm)) = prev_output.resolve(
                &mut self.model_state.tokens,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
            )? {
            if let Some(accept_command_buffer) = command_buffer.take() {
                pending.push(accept_command_buffer.end_encoding().submit());
            }
            let trie = speculator.propose_tree(
                self.model_state.speculator_state.as_mut().unwrap(),
                output_norm,
                root_token as u32,
                self.model.decoder.embedding(),
                shape,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
                &self.model_state.prng,
                self.allocation_pool.clone(),
            )?;
            prev_output.drop_output_norm();
            (trie, None, false)
        } else {
            prev_output.drop_output_norm();
            let (token, chain_copy) = match &prev_output {
                ForwardPassChaining::Constant {
                    token,
                    ..
                } => (*token, None),
                ForwardPassChaining::InFlight(pending) => (0, Some(pending.output_tokens.clone())),
            };
            (TrieNode::new(token, self.model_state.prng.derive(context_length as u64), 0.0), chain_copy, true)
        };
        let input_flat_trie = input_trie.linearize();

        if let Some(accept_command_buffer) = command_buffer.take() {
            pending.push(accept_command_buffer.end_encoding().submit());
        }

        // The previous pass is still in flight while this pass is encoded.
        // Reuse only a pool returned by a completed earlier pass, never the
        // active pool whose CPU scratch handles may already have been dropped.
        self.allocation_pool =
            self.reusable_decode_pool.take().unwrap_or_else(|| self.model.engine.context.create_allocation_pool());

        let mut command_buffer = self
            .model
            .engine
            .context
            .create_command_buffer(Some("decode"), Some(self.allocation_pool.clone()))
            .map_err(LanguageModelStreamError::Backend)?;

        let scratch_token_ids;
        let mut constant_token_ids = None;
        let token_ids: &dyn Buffer<Backend = B> = if let Some(chain_copy) = chain_copy.as_deref() {
            let mut token_ids = command_buffer
                .allocate_scratch(DataType::U32.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            command_buffer.encode_copy(chain_copy, &mut token_ids);
            scratch_token_ids = token_ids;
            &scratch_token_ids
        } else {
            constant_token_ids.insert(
                command_buffer
                    .allocate_constant_from_slice(
                        &input_flat_trie.token_ids().map(|token_id| token_id as u32).collect::<Box<[u32]>>(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?,
            )
        };

        let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_dim = BatchTopology::new(&input_flat_trie_nodes, full_accept);

        self.model_state
            .transformer_state
            .prepare(self.model_state.transformer_state.context_length(), batch_dim.size())
            .map_err(LanguageModelStreamError::Backend)?;

        let hidden_feature_layer_indices =
            self.model.speculator.as_ref().map(|speculator| speculator.hidden_feature_layer_indices());

        let decoder_output = self.model.decoder.encode(
            token_ids,
            &batch_dim,
            Some((0..batch_dim.size()).into()),
            hidden_feature_layer_indices,
            &mut self.model_state.transformer_state,
            &mut command_buffer,
        )?;
        let logits = decoder_output.logits.unwrap();

        #[cfg(grammar)]
        let (bitmask, mut command_buffer) = if let Some(grammar) = self.options.grammar.as_mut() {
            if chain_copy.is_some() {
                pending.push(command_buffer.end_encoding().submit());

                let mut command_buffer = self
                    .model
                    .engine
                    .context
                    .create_command_buffer(Some("decode"), Some(self.allocation_pool.clone()))
                    .map_err(LanguageModelStreamError::Backend)?;

                let mut bitmask = command_buffer
                    .allocate_constant(
                        self.model.vocab_size.div_ceil(DataType::U32.size_in_bits()) * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                prev_output.resolve(&mut self.model_state.tokens, Some(grammar))?;
                if grammar.next_bitmask(bitmask.as_slice_mut()) {
                    (Some(bitmask), command_buffer)
                } else {
                    (None, command_buffer)
                }
            } else {
                let mut bitmasks = command_buffer
                    .allocate_constant(
                        input_flat_trie.len()
                            * self.model.vocab_size.div_ceil(DataType::U32.size_in_bits())
                            * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                if input_flat_trie.fill_bitmasks(bitmasks.as_slice_mut(), self.model.vocab_size, grammar) {
                    (Some(bitmasks), command_buffer)
                } else {
                    (None, command_buffer)
                }
            }
        } else {
            (None, command_buffer)
        };
        #[cfg(not(grammar))]
        let bitmask = None::<B::ConstantBuffer>;

        let seeds = if matches!(self.options.sampling_method, SamplingMethod::Stochastic { .. }) {
            Some(
                command_buffer
                    .allocate_constant_from_slice(&input_flat_trie.token_seeds().collect::<Box<[u64]>>())
                    .map_err(LanguageModelStreamError::Backend)?,
            )
        } else {
            None
        };

        let output_tokens = self
            .model
            .sampling
            .encode(
                &logits,
                seeds.as_ref(),
                bitmask.as_ref(),
                self.context_ring.as_ref(),
                Some(token_ids),
                &self.options.sampling_method,
                &batch_dim,
                (0..batch_dim.size()).into(),
                &mut command_buffer,
            )
            .map_err(LanguageModelStreamError::Backend)?;

        drop(seeds);
        drop(bitmask);
        drop(logits);

        if full_accept {
            self.model_state
                .transformer_state
                .encode_accept(&(0..batch_dim.size()).collect::<Box<[u32]>>(), &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;

            if let Some(speculator) = self.model.speculator.as_ref() {
                let speculator_state = self.model_state.speculator_state.as_mut().unwrap();
                speculator
                    .encode_accept(
                        speculator_state,
                        decoder_output.hidden_features.as_ref().unwrap().iter(),
                        &(0..batch_dim.size()).collect::<Box<[u32]>>(),
                        &mut command_buffer,
                    )
                    .map_err(LanguageModelStreamError::Backend)?;
            }

            if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
                self.model.context_ring_update.encode(
                    token_ids,
                    self.context_ring.as_mut().unwrap(),
                    suffix_repetition_length,
                    batch_dim.size(),
                    &mut command_buffer,
                );
            }
        }

        drop(constant_token_ids);
        pending.push(command_buffer.end_encoding().submit());

        self.metrics.num_decode_forward_passes += 1;
        self.metrics.num_tokens_proposed += input_flat_trie.len();
        if full_accept {
            self.metrics.num_tokens_accepted += input_flat_trie.len();
        }

        pending.reusable_pool = self.reuse_decode_pool.then(|| self.allocation_pool.clone());
        self.decoding_state = DecodingState::ForwardPassPending(DecodingStatePending {
            input_trie,
            full_accept,
            pending: Some(pending),
            capture_span,
            hidden_features: if full_accept {
                None
            } else {
                decoder_output.hidden_features
            },
            output_norm: decoder_output.final_hidden,
            output_tokens: Arc::new(output_tokens),
            _input_token_source: chain_copy,
        });

        let token = prev_output
            .resolve(
                &mut self.model_state.tokens,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
            )?
            .0;
        self.reusable_decode_pool = prev_output.take_completed_pool();
        Ok(Some(token))
    }

    pub fn metrics(&self) -> &TokenStreamMetrics {
        &self.metrics
    }
}

fn fence_multimodal_state<B: Backend>(model_state: &mut LanguageModelState<B>) {
    model_state.multimodal_failed = true;
    model_state.transformer_state.set_rotary_position_delta(None);
    model_state.last_output_token = None;
}

fn valid_stream_tokens<'a>(
    vocab_size: u32,
    tokens: impl IntoIterator<Item = &'a u64>,
) -> bool {
    vocab_size != 0 && tokens.into_iter().all(|&token| token < u64::from(vocab_size))
}

impl<'a, B: Backend> Iterator for LanguageModelStream<'a, B> {
    type Item = Result<u64, LanguageModelStreamError<B>>;

    fn next(&mut self) -> Option<Result<u64, LanguageModelStreamError<B>>> {
        // A failed image request is terminal. Keep any submitted work owned by
        // decoding_state until Drop drains it, without entering Invalid again.
        if self.model_state.multimodal_failed {
            if let Some(fence) = &mut self.failure_fence {
                **fence = true;
            }
            return None;
        }
        // The ordinary stream's lookahead may already hold a sampled next
        // token. Image requests must not return it after their context budget
        // is consumed; Drop still drains the owned submission safely.
        if self.model_state.has_multimodal_context()
            && self
                .model_state
                .max_context_length
                .is_some_and(|capacity| self.model_state.tokens.len() >= capacity as usize)
        {
            return None;
        }
        let result = self.generate();
        if result.is_err()
            && let Some(fence) = &mut self.failure_fence
        {
            **fence = true;
        }
        if result.is_err() && self.model_state.has_multimodal_context() {
            fence_multimodal_state(self.model_state);
        }
        result.transpose()
    }
}

impl<'a, B: Backend> Drop for LanguageModelStream<'a, B> {
    fn drop(&mut self) {
        let last_output_token = match replace(&mut self.decoding_state, DecodingState::Invalid) {
            DecodingState::Seeded {
                seed_token,
            } => Some(seed_token),
            DecodingState::ForwardPassPending(in_flight) => {
                let completion = in_flight.pending.unwrap().wait(|pending| pending.wait_until_completed().map(|_| ()));
                if completion.is_err()
                    && (self.model_state.multimodal_failed
                        || self.model_state.has_multimodal_context()
                        || self.failure_fence.is_some())
                {
                    fence_multimodal_state(self.model_state);
                    if let Some(fence) = &mut self.failure_fence {
                        **fence = true;
                    }
                    return;
                }
                completion.unwrap();
                if self.model_state.multimodal_failed {
                    if let Some(fence) = &mut self.failure_fence {
                        **fence = true;
                    }
                    self.model_state.last_output_token = None;
                    return;
                }

                if !in_flight.full_accept {
                    let mut command_buffer = self
                        .model
                        .engine
                        .context
                        .create_command_buffer(Some("drop accept"), Some(self.allocation_pool.clone()))
                        .unwrap();
                    self.model_state.transformer_state.encode_accept(&[0], &mut command_buffer).unwrap();
                    if let Some(speculator) = self.model.speculator.as_ref() {
                        speculator
                            .encode_accept(
                                self.model_state.speculator_state.as_mut().unwrap(),
                                in_flight.hidden_features.as_ref().unwrap().iter(),
                                &[0],
                                &mut command_buffer,
                            )
                            .unwrap();
                    }
                    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
                }

                Some(in_flight.output_tokens.as_slice::<u32>()[0] as u64)
            },
            DecodingState::Accepting {
                full,
                num_accepted,
                hidden_features,
                output_norm: _,
                capture_span,
            } => {
                assert!(num_accepted > 0 && num_accepted < full.len());

                let mut command_buffer = self
                    .model
                    .engine
                    .context
                    .create_command_buffer(Some("drop accept"), Some(self.allocation_pool.clone()))
                    .unwrap();
                let accepted_token_indicies =
                    full.iter().take(num_accepted + 1).map(|(i, _, _)| *i as u32).collect::<Box<[u32]>>();
                self.model_state
                    .transformer_state
                    .encode_accept(&accepted_token_indicies, &mut command_buffer)
                    .unwrap();
                if let Some(speculator) = self.model.speculator.as_ref() {
                    speculator
                        .encode_accept(
                            self.model_state.speculator_state.as_mut().unwrap(),
                            hidden_features.as_ref().unwrap().iter(),
                            &accepted_token_indicies,
                            &mut command_buffer,
                        )
                        .unwrap();
                }
                command_buffer.end_encoding().submit().wait_until_completed().unwrap();

                drop(capture_span);

                self.model_state.tokens.extend(full.iter().take(num_accepted).map(|(_, _, t)| *t));

                Some(full[num_accepted].2)
            },
            DecodingState::Halted => None,
            DecodingState::Invalid => None, // TODO: proper error handling
        };

        self.model_state.last_output_token = if self.model_state.multimodal_failed {
            if let Some(fence) = &mut self.failure_fence {
                **fence = true;
            }
            None
        } else if self.model_state.has_multimodal_context()
            && self
                .model_state
                .max_context_length
                .is_some_and(|capacity| self.model_state.tokens.len() >= capacity as usize)
        {
            None
        } else {
            last_output_token
        };
    }
}

#[cfg(test)]
mod decode_pool_tests {
    use std::{cell::Cell, rc::Rc};

    use stark_engine_macros::stark_test;

    use super::{SubmittedScratchPool, valid_stream_tokens};

    #[stark_test]
    fn stream_tokens_reject_vocabulary_edges_and_narrowing() {
        for invalid in [64, 65, u32::MAX as u64, u64::from(u32::MAX) + 1, u64::MAX] {
            assert!(!valid_stream_tokens(64, [0, invalid, 63].iter()));
        }
        assert!(!valid_stream_tokens(0, [].iter()));
        assert!(valid_stream_tokens(64, [0, 1, 63].iter()));
        assert!(valid_stream_tokens(64, [].iter()));
        // The same guard covers retained state and an uncommitted output token.
        assert!(!valid_stream_tokens(64, [1].iter().chain([64].iter()).chain([2].iter())));
    }

    struct Pool {
        outstanding: Cell<usize>,
    }

    impl Pool {
        fn submit(
            pool: &Rc<Self>,
            count: usize,
        ) -> SubmittedScratchPool<Rc<Self>, Rc<Self>> {
            assert_eq!(pool.outstanding.get(), 0, "attempted to reuse an in-flight scratch pool");
            pool.outstanding.set(count);
            SubmittedScratchPool::new(vec![pool.clone(); count].into_boxed_slice(), Some(pool.clone()))
        }

        fn complete(pool: Rc<Self>) -> Result<(), ()> {
            pool.outstanding.set(pool.outstanding.get() - 1);
            Ok(())
        }
    }

    #[stark_test]
    fn decode_pool_rotation_keeps_in_flight_pages_separate() {
        let first = Rc::new(Pool {
            outstanding: Cell::new(0),
        });
        let second = Rc::new(Pool {
            outstanding: Cell::new(0),
        });

        // Encode and submit the next pass before resolving the current pass,
        // matching the stream's one-token lookahead scheduling.
        let mut current = Pool::submit(&first, 2);
        let mut spare = second;
        for _ in 0..8 {
            let next = Pool::submit(&spare, 1);
            let completed = current.wait(Pool::complete).unwrap().unwrap();
            assert_eq!(completed.outstanding.get(), 0);
            assert_eq!(spare.outstanding.get(), 1);
            assert!(!Rc::ptr_eq(&completed, &spare));
            spare = completed;
            current = next;
        }
        current.wait(Pool::complete).unwrap();
    }

    #[stark_test]
    fn decode_pool_wait_failure_never_returns_pool() {
        for failure_position in 1..=3 {
            let pool = Rc::new(Pool {
                outstanding: Cell::new(0),
            });
            let submitted = Pool::submit(&pool, 3);
            let mut waited = 0;
            let result = submitted.wait(|pending| {
                waited += 1;
                Pool::complete(pending)?;
                if waited == failure_position {
                    Err(())
                } else {
                    Ok(())
                }
            });
            assert!(result.is_err());
            assert_eq!(waited, 3);
            assert_eq!(pool.outstanding.get(), 0);
            assert_eq!(Rc::strong_count(&pool), 1);
        }
    }

    #[stark_test]
    fn abandoned_image_submissions_drain_before_releasing_the_pool() {
        let pool = Rc::new(Pool {
            outstanding: Cell::new(0),
        });
        let submitted = Pool::submit(&pool, 3).with_drop_wait(true, |pending| {
            let _ = Pool::complete(pending);
        });
        // A local pass can be abandoned by an allocation/encoding error before
        // the stream receives it. Dropping its owner still drains every pass.
        drop(submitted);
        assert_eq!(pool.outstanding.get(), 0);
        assert_eq!(Rc::strong_count(&pool), 1);

        // Explicit completion must disarm cleanup rather than complete twice.
        let submitted = Pool::submit(&pool, 2).with_drop_wait(true, |pending| {
            let _ = Pool::complete(pending);
        });
        let returned_pool = submitted.wait(Pool::complete).unwrap().unwrap();
        assert_eq!(returned_pool.outstanding.get(), 0);
        assert_eq!(Rc::strong_count(&pool), 2);
    }
}
