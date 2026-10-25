use std::{collections::HashSet, fs::File, path::Path, time::Instant};

use serde_json::Value;
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{LanguageModel, stream::SamplingMethod},
    },
};

use crate::{
    cancel::CancelSignal,
    generation::{
        GeneratedTokens, GenerationBackend, GenerationOptions, Sampling, TextCallback, collect_tokens_controlled,
    },
    text_stop::StopFilter,
};

pub(crate) struct NativeBackend {
    model: LanguageModel<Metal>,
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
    cancellation: Option<CancelSignal>,
}

impl NativeBackend {
    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        if path.join("speculator").exists() {
            return Err("speculator_not_supported");
        }
        let config: Value =
            serde_json::from_reader(File::open(path.join("config.json")).map_err(|_| "model_config_unavailable")?)
                .map_err(|_| "invalid_model_config")?;
        let vocab_size = config
            .get("decoder_config")
            .and_then(|decoder| decoder.get("vocab_size"))
            .and_then(Value::as_u64)
            .filter(|size| *size > 0 && *size <= u32::MAX as u64)
            .ok_or("invalid_model_config")? as usize;
        let engine = Engine::<Metal>::new().map_err(|_| "metal_initialization_failed")?;
        let model = engine.load_language_model(path).map_err(|_| "model_load_failed")?;
        let tokenizer_ids: HashSet<u32> = model.tokenizer().get_vocab(true).into_values().collect();
        if tokenizer_ids.iter().any(|id| *id as usize >= vocab_size) {
            return Err("invalid_model_config");
        }
        let eos_token_ids = crate::model_stops::model_stop_token_ids(
            &model.generation_config().stop_token_ids,
            config.get("token_codec_config").and_then(|codec| codec.get("eos_token")).and_then(Value::as_str),
            vocab_size,
            |token| model.tokenizer().token_to_id(token),
            |id| tokenizer_ids.contains(&id),
        )?;
        Ok(Self {
            model,
            vocab_size,
            eos_token_ids,
            cancellation,
        })
    }
}

impl GenerationBackend for NativeBackend {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn max_context_length(&self) -> Option<u32> {
        self.model.max_context_length()
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        self.model
            .tokenizer()
            .encode(text, special)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|_| "tokenization_failed")
    }
    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str> {
        self.model.tokenizer().decode(ids, skip_special).map_err(|_| "decoding_failed")
    }
    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        // Include native state allocation, stream setup and the first GPU wait.
        // The engine's iterator can encode one lookahead decode before returning
        // its first token, so this is a wall-clock prefill proxy, not GPU time.
        let started = Instant::now();
        let cancellation = &mut self.cancellation;
        let mut cancelled = || match cancellation.as_mut() {
            Some(signal) => signal.check(options.request_id),
            None => Ok(false),
        };
        if cancelled()? {
            return Err("generation_cancelled");
        }
        let mut state = self
            .model
            .create_empty_state(Some(options.context_tokens), options.seed)
            .map_err(|_| "native_generation_failed")?;
        let input: Vec<u64> = input.iter().map(|id| u64::from(*id)).collect();
        let mut stream_options = self.model.default_stream_options();
        stream_options.sampling_method = match &options.sampling {
            Sampling::Greedy {} => SamplingMethod::Greedy,
            Sampling::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
            } => SamplingMethod::Stochastic {
                temperature: Some(*temperature),
                top_k: *top_k,
                top_p: *top_p,
                min_p: *min_p,
                repetition_penalty: None,
                suffix_repetition_length: None,
            },
        };
        let stream = self.model.stream(&input, &mut state, stream_options).map_err(|_| "native_generation_failed")?;
        let tokens = stream.map(|token| token.map_err(|_| "native_generation_failed"));
        let mut decoder = self.model.tokenizer().decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let result = collect_tokens_controlled(
            tokens,
            input.len(),
            options,
            self.vocab_size,
            started,
            &mut |token| {
                if needs_text && let Some(text) = decoder.step(token).map_err(|_| "decoding_failed")? {
                    raw.push_str(&text);
                    let (text, stopped) = filter.push(&text);
                    if let Some(callback) = on_text.as_deref_mut() {
                        callback(&text)?;
                    }
                    return Ok(!stopped);
                }
                Ok(true)
            },
            &mut cancelled,
        );
        // collect_tokens has dropped and drained the stream, including its lookahead.
        drop(state);
        let mut result = result?;
        if needs_text {
            let full =
                self.model.tokenizer().decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
            let suffix = full.strip_prefix(&raw).ok_or("decoding_failed")?;
            let (tail, stopped) = filter.push(suffix);
            if stopped {
                result.finish_reason = crate::generation::FinishReason::StopSequence;
            }
            let tail = tail + &filter.finish();
            if let Some(callback) = on_text {
                callback(&tail)?;
            }
            result.text = Some(filter.text().to_owned());
        }
        Ok(result)
    }
}
