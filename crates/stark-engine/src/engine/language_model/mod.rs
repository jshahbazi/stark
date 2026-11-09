use std::{
    fs::File,
    io,
    io::BufReader,
    path::{Path, PathBuf},
    sync::Arc,
};

use derive_more::Debug;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    backends::common::{Backend, Context, DeviceCapabilities, Kernels, kernel::ContextRingUpdateKernel},
    config::{
        model::{generation::GenerationConfig, language_model::LanguageModelConfig},
        token_codec::AnyTokenCodecConfig,
    },
    data_type::DataType,
    encodable_block::{
        decoder::{Decoder, DecoderError},
        sampling::{Sampling, SamplingMethod},
    },
    engine::Engine,
    parameters::{ParameterLoader, ParameterLoaderError},
    speculators::dflash_tfm::{DFlashSpeculatorLoadError, DFlashTfmSpeculator},
};

pub mod dense_mtp;
#[cfg(all(test, backend = "metal"))]
mod dense_mtp_tests;
pub(crate) mod multimodal;
#[cfg(test)]
mod multimodal_tests;
pub mod scoring;
pub mod state;
pub mod stream;

#[derive(Clone, Copy)]
pub(crate) enum SpeculatorPolicy {
    Discover,
    Disabled,
}

fn select_speculator_path(
    model_path: &Path,
    policy: SpeculatorPolicy,
) -> Option<PathBuf> {
    match policy {
        SpeculatorPolicy::Discover => {
            let path = model_path.join("speculator");
            path.exists().then_some(path)
        },
        SpeculatorPolicy::Disabled => None,
    }
}

#[cfg(grammar)]
pub mod grammar;

pub struct LanguageModel<B: Backend> {
    pub(super) model_identity: Arc<()>,
    engine: Arc<Engine<B>>,
    decoder: Decoder<B>,
    dense_mtp: Option<crate::encodable_block::qwen_mtp::QwenMtp<B>>,
    dense_draft_readout: Option<crate::encodable_block::embedding::coarse::CoarseDraftReadout<B>>,
    speculator: Option<DFlashTfmSpeculator<B>>,
    sampling: Sampling<B>,
    context_ring_update: <B::Kernels as Kernels>::ContextRingUpdateKernel,
    generation_config: GenerationConfig,
    end_of_thinking_tag: Option<String>,
    tokenizer: Arc<Tokenizer>,
    #[cfg(grammar)]
    vocab_size: usize,
}

#[derive(Debug, Error)]
pub enum EngineLoadLanguageModelError<B: Backend> {
    #[error("Tokenizer vocabulary exceeds decoder vocabulary")]
    InvalidTokenizerVocabulary,
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashSpeculatorLoadError<B>),
    #[error("Qwen MTP error: {0}")]
    QwenMtp(#[from] crate::encodable_block::qwen_mtp::QwenMtpError<B>),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

impl<B: Backend> Engine<B> {
    pub fn load_language_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let config: LanguageModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*self.context)?;

        let model =
            self.language_model_from_config(model_path, config, &weight_loader, None, SpeculatorPolicy::Discover)?;
        weight_loader.tree().assert_all_tensors_validated()?;
        Ok(model)
    }

    /// Shared construction for the private vision loader. Its caller validates
    /// the complete decoder and vision tensor tree before publishing the model.
    pub(crate) fn language_model_from_config(
        self: &Arc<Self>,
        model_path: &Path,
        config: LanguageModelConfig,
        weight_loader: &ParameterLoader<B>,
        tokenizer: Option<Arc<Tokenizer>>,
        speculator_policy: SpeculatorPolicy,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        // Dense-MTP readiness promises both speculative verification and exact
        // retained-prefix restoration. Refuse other state layouts before any
        // decoder/head parameters are allocated on the device.
        if let Some(mtp) = &config.mtp_config {
            mtp.validate_for_target(&config.decoder_config.transformer_config)
                .map_err(|_| crate::encodable_block::qwen_mtp::QwenMtpError::<B>::Invalid)?;
        }
        // The guarded vision loader disables discovery even if a new directory
        // appears after its initial artifact admission check.
        let speculator_path = select_speculator_path(model_path, speculator_policy);

        let tokenizer = match tokenizer {
            Some(tokenizer) => tokenizer,
            None => Arc::new(Tokenizer::from_file(model_path.join("tokenizer.json"))?),
        };
        if config.decoder_config.vocab_size == 0
            || tokenizer.get_vocab(true).values().any(|&id| id >= config.decoder_config.vocab_size)
        {
            return Err(EngineLoadLanguageModelError::InvalidTokenizerVocabulary);
        }

        let data_type = DataType::BF16;

        let decoder = Decoder::new(
            self.context.as_ref(),
            &config.decoder_config,
            &weight_loader.tree().subtree("decoder"),
            data_type,
        )?;
        let dense_mtp = config
            .mtp_config
            .as_ref()
            .map(|mtp| {
                if mtp.model_dim != config.decoder_config.transformer_config.model_dim {
                    return Err(crate::encodable_block::qwen_mtp::QwenMtpError::Invalid);
                }
                let head_tree = weight_loader.tree().subtree("mtp");
                let head_tree = if mtp.source_bf16_geometry(&config.decoder_config.transformer_config)
                    && weight_loader.qwen_target_and_head_full_bf16()
                {
                    head_tree.with_numerical_profile(crate::parameters::NumericalProfile::QwenMtpSourceBF16)
                } else {
                    head_tree
                };
                crate::encodable_block::qwen_mtp::QwenMtp::new(
                    mtp,
                    config.decoder_config.transformer_config.layer_configs.len(),
                    &head_tree,
                    self.context.as_ref(),
                )
            })
            .transpose()?;

        let dense_draft_readout = if let Some(bits) = config.mtp_config.as_ref().and_then(|mtp| mtp.draft_readout_bits)
        {
            Some(
                decoder
                    .embedding()
                    .coarse_draft_readout(self.context.as_ref(), bits)
                    .map_err(DecoderError::EmbeddingError)?
                    .ok_or(crate::encodable_block::qwen_mtp::QwenMtpError::Invalid)?,
            )
        } else {
            None
        };

        assert!(
            speculator_path.is_none() || decoder.speculation_supported(),
            "attempted to load speculator for a model that doesn't support one"
        );

        let speculator = speculator_path
            .as_deref()
            .map(|speculator_path| DFlashTfmSpeculator::new(speculator_path, self.context.clone()))
            .transpose()?
            .flatten();

        let sampling = Sampling::new(data_type, config.decoder_config.vocab_size);

        let context_ring_update = <B::Kernels as Kernels>::ContextRingUpdateKernel::new(&self.context)
            .map_err(EngineLoadLanguageModelError::Backend)?;

        let generation_config = config.generation_config;
        let end_of_thinking_tag = match &config.token_codec_config {
            AnyTokenCodecConfig::ChatCodecConfig(token_codec_config) => token_codec_config.end_of_thinking_tag.clone(),
            AnyTokenCodecConfig::RawTextCodecConfig(_) => None,
        };

        #[cfg(grammar)]
        let vocab_size = config.decoder_config.vocab_size as usize;

        Ok(LanguageModel {
            model_identity: Arc::new(()),
            engine: self.clone(),
            decoder,
            dense_mtp,
            dense_draft_readout,
            speculator,
            sampling,
            context_ring_update,
            generation_config,
            end_of_thinking_tag,
            tokenizer,
            #[cfg(grammar)]
            vocab_size,
        })
    }
}

#[cfg(test)]
mod speculator_policy_tests {
    use stark_engine_macros::stark_test;

    use super::{SpeculatorPolicy, select_speculator_path};

    #[stark_test]
    fn disabled_speculator_policy_ignores_directory_created_after_admission() {
        let path = std::env::temp_dir().join(format!(
            "stark-speculator-policy-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path).unwrap();
        assert!(select_speculator_path(&path, SpeculatorPolicy::Disabled).is_none());
        assert!(select_speculator_path(&path, SpeculatorPolicy::Discover).is_none());
        std::fs::create_dir(path.join("speculator")).unwrap();
        assert!(select_speculator_path(&path, SpeculatorPolicy::Disabled).is_none());
        assert_eq!(select_speculator_path(&path, SpeculatorPolicy::Discover), Some(path.join("speculator")));
        std::fs::remove_dir_all(path).unwrap();
    }
}

impl<B: Backend> LanguageModel<B> {
    pub fn max_context_length(&self) -> Option<u32> {
        self.decoder.max_context_length()
    }

    fn context_memory_estimate(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        if self.speculator.is_some() {
            return None;
        }
        let head = self.dense_mtp.as_ref().filter(|_| self.has_dense_mtp());
        let rows = input_tokens.max(if head.is_some() {
            4
        } else {
            1
        });
        let target = self.decoder.memory_estimate(context_tokens, rows)?;
        match head {
            Some(head) => crate::encodable_block::memory::dense_mtp_estimate(
                target,
                head.memory_estimate(context_tokens, rows)?,
                rows,
                self.decoder.embedding().model_dim(),
                self.decoder.embedding().vocab_size(),
            ),
            None => Some(target),
        }
    }

    pub fn supports_automatic_context_planning(&self) -> bool {
        self.context_memory_estimate(2, 1).is_some()
    }

    /// Bound physical request growth without charging unused sparse address
    /// space from a finite model's configured context to a short request.
    pub fn generation_allocation_context_length(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<u32> {
        crate::encodable_block::memory::generation_allocation_context(
            input_tokens,
            context_tokens,
            max_new_tokens,
            self.max_context_length().is_some()
                && self.engine.context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS),
        )
    }

    /// Inventory decoder state and execution workspace for the complete
    /// requested input/output span. Vision wrappers use this for their text
    /// decoder; image-encoder execution has a separate admitted contract.
    pub fn generation_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        if self.max_context_length().is_some_and(|maximum| context_tokens > maximum) {
            return None;
        }
        let allocated = self.generation_allocation_context_length(input_tokens, context_tokens, max_new_tokens)?;
        let estimate = self.context_memory_estimate(allocated, input_tokens)?;
        let mut reserved = self.decoder.context_cache_buffer_bytes(context_tokens)?;
        if let Some(head) = self.dense_mtp.as_ref().filter(|_| self.has_dense_mtp()) {
            reserved = reserved.max(head.context_cache_buffer_bytes(context_tokens)?);
        }
        if self.engine.context.max_buffer_length().is_some_and(|limit| estimate.max_buffer_bytes.max(reserved) > limit)
        {
            return None;
        }
        estimate.total_bytes()
    }

    /// Inventory the ordinary target stream without allocating predictor state.
    /// Loaded predictor weights remain resident and are not request workspace.
    pub fn ordinary_text_generation_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        if self.speculator.is_some() || self.max_context_length().is_some_and(|maximum| context_tokens > maximum) {
            return None;
        }
        let allocated = self.generation_allocation_context_length(input_tokens, context_tokens, max_new_tokens)?;
        let estimate = self.decoder.memory_estimate(allocated, input_tokens.max(1))?;
        let reserved = self.decoder.context_cache_buffer_bytes(context_tokens)?;
        if self.engine.context.max_buffer_length().is_some_and(|limit| estimate.max_buffer_bytes.max(reserved) > limit)
        {
            return None;
        }
        estimate.total_bytes()
    }

    /// Include the recurrent-state snapshot retained by an ordinary text stream.
    pub fn generation_memory_bytes_with_text_prefix(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        let bytes = self.ordinary_text_generation_memory_bytes(input_tokens, context_tokens, max_new_tokens)?;
        if !self.text_prefix_snapshot_supported() {
            return Some(bytes);
        }
        let allocated = self.generation_allocation_context_length(input_tokens, context_tokens, max_new_tokens)?;
        let snapshot = self.decoder.memory_estimate(allocated, input_tokens.max(1))?.prefix_snapshot_bytes;
        bytes.checked_add(snapshot)
    }

    /// Admit exact scoring's fresh state and all independently retained prompt
    /// chunk pools. Scoring reads the final prompt position without reserving
    /// an output token, so input may exactly fill the selected context.
    pub fn scoring_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
    ) -> Option<usize> {
        if input_tokens == 0
            || input_tokens > context_tokens
            || self.speculator.is_some()
            || self.max_context_length().is_some_and(|maximum| context_tokens > maximum)
        {
            return None;
        }
        let allocated = if self.max_context_length().is_some()
            && self.engine.context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS)
        {
            input_tokens
        } else {
            context_tokens
        };
        // Dense MTP's predictor is not executed by score_next_token.
        let estimate = self.decoder.scoring_memory_estimate(allocated, input_tokens)?;
        let reserved = self.decoder.context_cache_buffer_bytes(context_tokens)?;
        if self.engine.context.max_buffer_length().is_some_and(|limit| estimate.max_buffer_bytes.max(reserved) > limit)
        {
            return None;
        }
        estimate.scoring_total_bytes(input_tokens, self.decoder.prefill_requires_separate_logits_row())
    }

    /// Plan a finite execution allocation from loaded geometry and a caller's
    /// memory headroom. This is not a trained model context-capacity claim.
    pub fn automatic_context_length(
        &self,
        input_tokens: u32,
        maximum_context_tokens: u32,
        memory_budget_bytes: usize,
    ) -> Option<u32> {
        let maximum =
            self.max_context_length().map_or(maximum_context_tokens, |limit| limit.min(maximum_context_tokens));
        let max_buffer_length = self.engine.context.max_buffer_length();
        crate::encodable_block::memory::automatic_context(input_tokens, maximum, memory_budget_bytes, |capacity| {
            self.context_memory_estimate(capacity, input_tokens)
                .filter(|estimate| max_buffer_length.is_none_or(|limit| estimate.max_buffer_bytes <= limit))
        })
    }

    pub fn recommended_context_length(&self) -> Option<u32> {
        let max_context_length = self.max_context_length();

        // TODO: This is not the correct way to do it, there should be a real memory model
        if self.engine.context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
            // We just assume that all mixers use sparse if it's available to make max context free until it's actually used
            // Currenlty true for all mixers in stark:
            // - full attention uses sparse if it's available to make max context free until it's actually used
            // - sliding window attention is bound, usually well below the recommended max context size on non-sparse (but can be made to use sparse if we care about it enough)
            // - short conv/mamba2/delta net are constant state size
            max_context_length
        } else if let Some(max_context_length) = max_context_length {
            // If sparse buffers aren't supported and model has finite maximum context length we assume that kv cache is expensive enough that we should probably clamp it to
            // something reasonable-ish for the platform. This is very primitive but works I guess...
            let platform_recommended_context_length = if cfg!(target_os = "ios") {
                8192
            } else {
                16384
            };

            Some(u32::min(max_context_length, platform_recommended_context_length))
        } else {
            // We just assume that unlimited context means constant state size on all mixers and is thus free
            None
        }
    }

    pub fn speculation_supported(&self) -> bool {
        self.decoder.speculation_supported()
    }

    pub fn default_sampling_method(&self) -> SamplingMethod {
        SamplingMethod::Stochastic {
            temperature: self.generation_config.temperature,
            top_k: self.generation_config.top_k,
            top_p: self.generation_config.top_p,
            min_p: self.generation_config.min_p,
            repetition_penalty: self.generation_config.repetition_penalty,
            suffix_repetition_length: self.generation_config.suffix_repetition_length,
        }
    }

    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }

    pub fn end_of_thinking_tag(&self) -> Option<&str> {
        self.end_of_thinking_tag.as_deref()
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
}
