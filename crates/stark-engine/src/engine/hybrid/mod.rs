//! Native, source-format execution for the GLM5-Next and Qwen4-Exp families.
//!
//! These models carry distinct recurrent state, sparse attention and residual
//! streams. They deliberately do not enter the dense decoder's artifact path.

pub mod glm;
pub mod math;
pub mod qwen;
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
    json_file(&path.join("config.json"), 1024 * 1024)
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

/// Detect only the two explicitly implemented architectures; never infer one
/// from a repository name or load repository-provided Python code.
pub fn recognizes(path: &Path) -> Result<bool> {
    let value = config(path)?;
    Ok(matches!(value.get("model_type").and_then(Value::as_str), Some("glm5_next" | "qwen4_exp")))
}

enum Forward {
    Glm(glm::GlmModel),
    Qwen(qwen::QwenModel),
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
        #[cfg(feature = "metal")]
        weight_metal::initialize()?;
        let outer = config(path)?;
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
        let weights = weights::Weights::open(path)?;
        let forward = match outer.get("model_type").and_then(Value::as_str) {
            Some("glm5_next") => Forward::Glm(glm::GlmModel::new(weights, &outer)?),
            Some("qwen4_exp") => Forward::Qwen(qwen::QwenModel::new(weights, &outer)?),
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
            Forward::Glm(model) => model.reset(),
            Forward::Qwen(model) => model.reset(),
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
                Forward::Glm(model) => model.step(token),
                Forward::Qwen(model) => model.step(token),
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
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

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
