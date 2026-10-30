//! Architecture-specific execution for GLM5-Next and Recursive Flint.
//!
//! These models carry distinct recurrent state, sparse attention and residual
//! streams. Converted Qwen3.8 executes through the standard LanguageModel API.

pub mod activity;
pub mod finite_guard;
pub mod flint;
pub mod glm;
#[cfg(feature = "metal")]
mod glm_metal;
pub mod math;
pub mod profile;
#[cfg(test)]
mod qwen;
#[cfg(all(test, feature = "metal"))]
mod qwen_metal;
#[cfg(feature = "metal")]
mod weight_metal;
pub mod weights;

use std::{fs::File, io::Read, path::Path, sync::Arc};

use serde_json::Value;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid native hybrid model: {0}")]
    Invalid(&'static str),
    #[error("unsupported native hybrid model: {0}")]
    Unsupported(&'static str),
    #[error("native Qwen3.8 checkpoint requires Uzu conversion")]
    ConversionRequired,
    #[error("native hybrid model I/O failed")]
    Io(#[from] std::io::Error),
    #[error("native hybrid model metadata is invalid")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

fn json_file(
    path: &Path,
    limit: u64,
) -> Result<Value> {
    let file = File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > limit {
        return Err(Error::Invalid("configuration exceeds size limit"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(Error::Invalid("configuration exceeds size limit"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn config(path: &Path) -> Result<Value> {
    json_file(&path.join("config.json"), 8 * 1024 * 1024)
}

fn extend_eos(
    value: &Value,
    eos: &mut Vec<u32>,
    vocab_size: usize,
) -> Result<()> {
    let values: Vec<&Value> = match value {
        Value::Array(values) => values.iter().collect(),
        _ => vec![value],
    };
    for value in values {
        let id = value.as_u64().filter(|id| *id < vocab_size as u64).ok_or(Error::Invalid("EOS IDs"))? as u32;
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    Ok(())
}

/// Detect only the explicitly implemented architectures; never infer one
/// from a repository name or load repository-provided Python code.
pub fn recognizes(path: &Path) -> Result<bool> {
    Ok(native_format(&config(path)?)?.is_some())
}

/// Native hybrid dispatch never admits the converted Qwen architecture.
/// Every native Qwen3.8 export requires conversion; no source precision or
/// marker can select the former F32 executor around the standard Uzu decoder.
fn native_format(config: &Value) -> Result<Option<&str>> {
    if config.get("type").and_then(Value::as_str) == Some("Qwen38LanguageModelConfig") {
        return Ok(None);
    }
    let format = config.get("model_type").and_then(Value::as_str);
    if format == Some("qwen4_exp") {
        return Err(Error::ConversionRequired);
    }
    Ok(format.filter(|format| matches!(*format, "glm5_next" | "flint")))
}

enum Forward {
    Flint(flint::FlintModel),
    Glm(glm::GlmModel),
}

pub struct Model {
    forward: Forward,
    tokenizer: Arc<Tokenizer>,
    vocab_size: usize,
    max_context: u32,
    eos: Vec<u32>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let outer = config(path)?;
        let format = native_format(&outer)?.ok_or(Error::Unsupported("use the standard language-model loader"))?;
        let text = outer.get("text_config").unwrap_or(&outer);
        let vocab_size = text
            .get("vocab_size")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0 && *n <= 1_000_000)
            .ok_or(Error::Invalid("vocabulary size"))? as usize;
        let max_context = text
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0 && *n <= u32::MAX as u64)
            .ok_or(Error::Invalid("context size"))? as u32;
        let tokenizer =
            Arc::new(Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| Error::Invalid("tokenizer"))?);
        let vocab = tokenizer.get_vocab(true);
        if vocab.is_empty() || vocab.values().any(|&id| id as usize >= vocab_size) {
            return Err(Error::Invalid("tokenizer vocabulary"));
        }
        let mut eos = Vec::new();
        if let Some(value) = text.get("eos_token_id") {
            extend_eos(value, &mut eos, vocab_size)?;
        }
        let generation_path = path.join("generation_config.json");
        if generation_path.exists() {
            let generation = json_file(&generation_path, 1024 * 1024)?;
            if let Some(value) = generation.get("eos_token_id") {
                extend_eos(value, &mut eos, vocab_size)?;
            }
        }
        let tokenizer_path = path.join("tokenizer_config.json");
        if tokenizer_path.exists() {
            let metadata = json_file(&tokenizer_path, 2 * 1024 * 1024)?;
            if let Some(value) = metadata.get("eos_token").filter(|value| !value.is_null()) {
                let token = value
                    .as_str()
                    .or_else(|| value.get("content").and_then(Value::as_str))
                    .filter(|token| !token.is_empty())
                    .ok_or(Error::Invalid("EOS token"))?;
                let id = tokenizer.token_to_id(token).ok_or(Error::Invalid("EOS token"))?;
                extend_eos(&Value::from(id), &mut eos, vocab_size)?;
            }
        }
        if eos.iter().any(|id| !vocab.values().any(|value| value == id)) {
            return Err(Error::Invalid("EOS IDs"));
        }
        if eos.is_empty() {
            return Err(Error::Invalid("EOS IDs"));
        }
        #[cfg(feature = "metal")]
        weight_metal::initialize()?;
        let weights = weights::Weights::open(path)?;
        let forward = match format {
            "flint" => Forward::Flint(flint::FlintModel::new(weights, &outer)?),
            "glm5_next" => Forward::Glm(glm::GlmModel::new(weights, &outer)?),
            _ => return Err(Error::Unsupported("architecture")),
        };
        Ok(Self {
            forward,
            tokenizer,
            vocab_size,
            max_context,
            eos,
        })
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn max_context_length(&self) -> u32 {
        self.max_context
    }
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Each worker request starts with fresh state. No prompt or recurrent state
    /// is persisted, and failures invalidate the entire current sequence.
    pub fn reset(&mut self) {
        match &mut self.forward {
            Forward::Flint(model) => model.reset(),
            Forward::Glm(model) => model.reset(),
        }
    }

    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        let result = if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            match &mut self.forward {
                Forward::Flint(model) => model.step(token),
                Forward::Glm(model) => model.step(token),
            }
            .and_then(|logits| {
                if logits.len() != self.vocab_size || logits.iter().any(|x| !x.is_finite()) {
                    Err(Error::Invalid("nonfinite or invalid readout"))
                } else {
                    Ok(logits)
                }
            })
        };
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Advance a prefill prefix without projecting vocabulary logits that the
    /// caller will discard. The final input token must still use `step`.
    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        let result = if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            match &mut self.forward {
                Forward::Flint(model) => model.ingest(token),
                Forward::Glm(model) => model.ingest(token),
            }
        };
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Native hybrid families currently expose synchronous full readouts.
    pub fn supports_greedy_readout(&self) -> bool {
        false
    }

    pub fn set_bf16_stage_profiling(
        &mut self,
        _enabled: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("BF16 stage profiling unavailable"))
    }

    pub fn supports_sampled_decode(&self) -> bool {
        false
    }

    pub fn sampled_decode<'a>(
        &'a mut self,
        _first: u32,
        _remaining: usize,
        _sample: impl FnMut(&[f32]) -> Result<u32> + 'a,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + 'a>> {
        Err(Error::Unsupported("native sampled stream unavailable"))
    }

    pub fn step_greedy(
        &mut self,
        token: u32,
    ) -> Result<u32> {
        self.reset();
        if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            Err(Error::Unsupported("native greedy readout unavailable"))
        }
    }

    pub fn greedy_decode(
        &mut self,
        _first: u32,
        _remaining: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + '_>> {
        Err(Error::Unsupported("native greedy stream unavailable"))
    }

    pub fn is_fenced(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn dispatch_excludes_converted_qwen_and_rejects_all_native_qwen_exports() {
        assert!(native_format(&serde_json::json!({"type":"Qwen38LanguageModelConfig"})).unwrap().is_none());
        for marker in
            [serde_json::json!({"format":"qwen4_exp-affine-v1"}), serde_json::json!({"format":"unknown"}), Value::Null]
        {
            assert!(matches!(
                native_format(&serde_json::json!({"model_type":"qwen4_exp","lily":marker})),
                Err(Error::ConversionRequired)
            ));
        }
        assert!(matches!(
            native_format(&serde_json::json!({"model_type":"qwen4_exp"})),
            Err(Error::ConversionRequired)
        ));
        for architecture in ["glm5_next", "flint"] {
            assert_eq!(native_format(&serde_json::json!({"model_type":architecture})).unwrap(), Some(architecture));
        }
        assert!(native_format(&serde_json::json!({"model_type":"unimplemented"})).unwrap().is_none());
    }

    #[stark_test]
    fn native_qwen_conversion_requirement_precedes_tokenizer_and_weight_access() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "stark-native-qwen-conversion-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            br#"{"model_type":"qwen4_exp","lily":{"format":"qwen4_exp-affine-v1"}}"#,
        )
        .unwrap();
        assert!(matches!(recognizes(&directory), Err(Error::ConversionRequired)));
        assert!(matches!(Model::load(&directory), Err(Error::ConversionRequired)));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[stark_test]
    fn generation_eos_supplements_primary_end_of_text_without_duplicates() {
        let mut ids = Vec::new();
        extend_eos(&serde_json::json!(248044), &mut ids, 248320).unwrap();
        extend_eos(&serde_json::json!([248046, 248044]), &mut ids, 248320).unwrap();
        assert_eq!(ids, [248044, 248046]);
        assert!(extend_eos(&serde_json::json!(-1), &mut ids, 248320).is_err());
        assert!(extend_eos(&serde_json::json!([248320]), &mut ids, 248320).is_err());
    }
}
