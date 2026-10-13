use std::{collections::HashSet, fs::File, path::Path};

use serde_json::Value;
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{LanguageModel, stream::SamplingMethod},
    },
};

use crate::generation::{GeneratedTokens, GenerationBackend, GenerationOptions, Sampling, collect_tokens};

pub(crate) struct NativeBackend {
    model: LanguageModel<Metal>,
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
}

impl NativeBackend {
    pub(crate) fn load(path: &Path) -> Result<Self, &'static str> {
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
        let mut eos_token_ids = Vec::new();
        let tokenizer_ids: HashSet<u32> = model.tokenizer().get_vocab(true).into_values().collect();
        if tokenizer_ids.iter().any(|id| *id as usize >= vocab_size) {
            return Err("invalid_model_config");
        }
        for &id in &model.generation_config().stop_token_ids {
            if id >= vocab_size as u64 || !tokenizer_ids.contains(&(id as u32)) {
                return Err("invalid_model_config");
            }
            if !eos_token_ids.contains(&(id as u32)) {
                eos_token_ids.push(id as u32);
            }
        }
        Ok(Self {
            model,
            vocab_size,
            eos_token_ids,
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
    ) -> Result<GeneratedTokens, &'static str> {
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
        let result = collect_tokens(
            stream.map(|token| token.map_err(|_| "native_generation_failed")),
            input.len(),
            options,
            self.vocab_size,
        );
        // collect_tokens has dropped and drained the stream, including its lookahead.
        drop(state);
        result
    }
}
