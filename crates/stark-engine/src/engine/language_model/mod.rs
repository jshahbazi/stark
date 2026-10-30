use std::{
    fs::File,
    io,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use derive_more::Debug;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    backends::common::{Backend, Context, DeviceCapabilities, Kernels, kernel::ContextRingUpdateKernel},
    config::{
        model::{
            AnyModelConfig, generation::GenerationConfig, language_model::LanguageModelConfig,
            qwen38_language_model::Qwen38LanguageModelConfig,
        },
        token_codec::AnyTokenCodecConfig,
    },
    data_type::DataType,
    encodable_block::{
        decoder::{Decoder, DecoderError},
        qwen38::Qwen38Decoder,
        sampling::{Sampling, SamplingMethod},
    },
    engine::Engine,
    parameters::{ParameterLoader, ParameterLoaderError},
    speculators::dflash_tfm::{DFlashSpeculatorLoadError, DFlashTfmSpeculator},
};

mod execution;
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
    decoder: execution::DecoderExecution<B>,
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
    #[error("Unsupported or invalid language-model configuration")]
    InvalidModelConfig,
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
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

impl<B: Backend> Engine<B> {
    pub fn load_language_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        const CONFIG_LIMIT: u64 = 8 * 1024 * 1024;
        let config_path = model_path.join("config.json");
        let metadata = std::fs::metadata(&config_path)?;
        if !metadata.is_file() || metadata.len() > CONFIG_LIMIT {
            return Err(EngineLoadLanguageModelError::InvalidModelConfig);
        }
        let mut config_bytes = Vec::new();
        File::open(config_path)?.take(CONFIG_LIMIT + 1).read_to_end(&mut config_bytes)?;
        if config_bytes.len() as u64 > CONFIG_LIMIT {
            return Err(EngineLoadLanguageModelError::InvalidModelConfig);
        }
        let config: AnyModelConfig = serde_json::from_slice(&config_bytes)?;
        let qwen38 = matches!(&config, AnyModelConfig::Qwen38LanguageModelConfig(_));

        if qwen38 {
            for name in ["config.json", "model.safetensors", "tokenizer.json"] {
                let metadata = std::fs::symlink_metadata(model_path.join(name))?;
                if !metadata.is_file() || metadata.file_type().is_symlink() {
                    return Err(EngineLoadLanguageModelError::InvalidModelConfig);
                }
            }
        }
        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = if qwen38 {
            ParameterLoader::new_with_header_limit(&weights_file, &*self.context, CONFIG_LIMIT as usize)?
        } else {
            ParameterLoader::new(&weights_file, &*self.context)?
        };

        let model = match config {
            AnyModelConfig::LanguageModelConfig(config) => {
                self.language_model_from_config(model_path, config, &weight_loader, None, SpeculatorPolicy::Discover)?
            },
            AnyModelConfig::Qwen38LanguageModelConfig(config) => {
                self.qwen38_language_model_from_config(model_path, config, &weight_loader)?
            },
            _ => return Err(EngineLoadLanguageModelError::InvalidModelConfig),
        };
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
            decoder: execution::DecoderExecution::Standard(Box::new(decoder)),
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

    fn qwen38_language_model_from_config(
        self: &Arc<Self>,
        model_path: &Path,
        config: Qwen38LanguageModelConfig,
        weights: &ParameterLoader<B>,
    ) -> Result<LanguageModel<B>, EngineLoadLanguageModelError<B>> {
        let tokenizer = Arc::new(Tokenizer::from_file(model_path.join("tokenizer.json"))?);
        let vocabulary = config.decoder_config.architecture.vocab_size;
        let vocab = tokenizer.get_vocab(true);
        if vocabulary == 0 || vocab.is_empty() || vocab.values().any(|&id| id >= vocabulary) {
            return Err(EngineLoadLanguageModelError::InvalidTokenizerVocabulary);
        }
        let decoder =
            Qwen38Decoder::new(self.context.as_ref(), &config.decoder_config, weights).map_err(DecoderError::Qwen38)?;
        let decoder = execution::DecoderExecution::Qwen38(Box::new(decoder));
        let mut generation_config = config.generation_config;
        let mut stops = generation_config.stop_token_ids.to_vec();
        stops.extend(config.decoder_config.architecture.eos_token_id.iter().map(|&id| u64::from(id)));
        let end_of_thinking_tag = match &config.token_codec_config {
            AnyTokenCodecConfig::ChatCodecConfig(codec) => {
                if let Some(eos) = &codec.eos_token {
                    stops.push(u64::from(
                        tokenizer.token_to_id(eos).ok_or(EngineLoadLanguageModelError::InvalidTokenizerVocabulary)?,
                    ));
                }
                codec.end_of_thinking_tag.clone()
            },
            AnyTokenCodecConfig::RawTextCodecConfig(_) => None,
        };
        if stops.iter().any(|&id| id >= u64::from(vocabulary) || !vocab.values().any(|&value| u64::from(value) == id)) {
            return Err(EngineLoadLanguageModelError::InvalidTokenizerVocabulary);
        }
        stops.sort_unstable();
        stops.dedup();
        generation_config.stop_token_ids = stops.into_boxed_slice();
        let context_ring_update = <B::Kernels as Kernels>::ContextRingUpdateKernel::new(&self.context)
            .map_err(EngineLoadLanguageModelError::Backend)?;
        Ok(LanguageModel {
            model_identity: Arc::new(()),
            engine: self.clone(),
            decoder,
            speculator: None,
            sampling: Sampling::new_checked(DataType::F32, vocabulary),
            context_ring_update,
            generation_config,
            end_of_thinking_tag,
            tokenizer,
            #[cfg(grammar)]
            vocab_size: vocabulary as usize,
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
    /// The admitted decoder variant, independent of source-directory metadata.
    pub fn is_qwen38(&self) -> bool {
        matches!(&self.decoder, execution::DecoderExecution::Qwen38(_))
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.decoder.max_context_length()
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
