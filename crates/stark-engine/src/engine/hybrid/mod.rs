//! Native, source-format execution for GLM5-Next, Qwen4-Exp and Recursive Flint.
//!
//! These models carry distinct recurrent state, sparse attention and residual
//! streams. They deliberately do not enter the dense decoder's artifact path.

pub mod activity;
pub mod flint;
pub mod glm;
#[cfg(feature = "metal")]
mod glm_metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod lily;
pub mod math;
pub mod profile;
pub mod qwen;
#[cfg(feature = "metal")]
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
    #[error("native hybrid model I/O failed")]
    Io(#[from] std::io::Error),
    #[error("native hybrid model metadata is invalid")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Caller-owned complete-vocabulary sampling. A round records draft
/// distributions before target verification consumes them in prefix order.
pub trait MtpSampler {
    fn begin_round(&mut self);
    fn draft(
        &mut self,
        logits: &[f32],
    ) -> Result<u32>;
    fn target(
        &mut self,
        logits: &[f32],
        proposal: Option<u32>,
    ) -> Result<u32>;
}

/// Aggregate request counters. No token, logit or probability data is retained.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MtpStats {
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub verified_rows: usize,
    pub draft_seconds: f64,
    pub verify_seconds: f64,
    pub rollback_seconds: f64,
}

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

/// Detect only the explicitly implemented architectures; never infer one
/// from a repository name or load repository-provided Python code.
pub fn recognizes(path: &Path) -> Result<bool> {
    let value = config(path)?;
    Ok(matches!(value.get("model_type").and_then(Value::as_str), Some("glm5_next" | "qwen4_exp" | "flint")))
}

enum Forward {
    Flint(flint::FlintModel),
    Glm(glm::GlmModel),
    Qwen(qwen::QwenModel),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Lily(Box<lily::LilyModel>),
}

fn mixed_precision_checkpoint(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("qwen4_exp")
        && config.pointer("/lily/format").and_then(Value::as_str) == Some("qwen4_exp-affine-v1")
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
        Self::load_with_mtp(path, false)
    }

    /// Loads the native head when the supported checkpoint declares one.
    /// Request-level execution remains disabled until explicitly selected.
    pub fn load_with_mtp(
        path: &Path,
        with_mtp: bool,
    ) -> Result<Self> {
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let _ = with_mtp;
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
        let (forward, max_context) = if mixed_precision_checkpoint(&outer) {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let model = lily::LilyModel::load_with_mtp(path, with_mtp)?;
                if model.vocab_size() != vocab_size || model.position() != 0 {
                    return Err(Error::Invalid("mixed precision metadata mismatch"));
                }
                let context = max_context.min(model.max_context() as u32);
                (Forward::Lily(Box::new(model)), context)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                return Err(Error::Unsupported("mixed precision checkpoint requires macOS Metal"));
            }
        } else {
            #[cfg(feature = "metal")]
            weight_metal::initialize()?;
            let weights = weights::Weights::open(path)?;
            let forward = match outer.get("model_type").and_then(Value::as_str) {
                Some("flint") => Forward::Flint(flint::FlintModel::new(weights, &outer)?),
                Some("glm5_next") => Forward::Glm(glm::GlmModel::new(weights, &outer)?),
                Some("qwen4_exp") => Forward::Qwen(qwen::QwenModel::new(weights, &outer)?),
                _ => return Err(Error::Unsupported("architecture")),
            };
            (forward, max_context)
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
            Forward::Qwen(model) => model.reset(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.reset(),
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
                Forward::Qwen(model) => model.step(token),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                Forward::Lily(model) => model.step(token),
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
                Forward::Qwen(model) => model.ingest(token),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                Forward::Lily(model) => model.ingest(token),
            }
        };
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// A native GPU greedy readout is available for the mixed-precision core.
    pub fn supports_greedy_readout(&self) -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if matches!(self.forward, Forward::Lily(_)) {
            return true;
        }
        false
    }

    /// Opt-in aggregate GPU stage timestamps for the BF16 backend. Profiling
    /// perturbs execution and is never enabled for production rate comparisons.
    pub fn set_bf16_stage_profiling(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.set_stage_profiling(enabled),
            _ => {
                let _ = enabled;
                Err(Error::Unsupported("BF16 stage profiling unavailable"))
            },
        }
    }

    /// The mixed-precision backend can overlap encoding around a host sampler.
    pub fn supports_sampled_decode(&self) -> bool {
        self.supports_greedy_readout()
    }

    pub fn supports_mtp(&self) -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.supports_mtp();
        }
        false
    }

    pub fn set_mtp_enabled(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.set_mtp_enabled(enabled),
            _ if !enabled => Ok(()),
            _ => Err(Error::Unsupported("MTP unavailable")),
        }
    }

    pub fn mtp_stats(&self) -> MtpStats {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.mtp_stats();
        }
        MtpStats::default()
    }

    pub fn mtp_decode<'a>(
        &'a mut self,
        first: u32,
        remaining: usize,
        drafts: usize,
        sampler: &'a mut dyn MtpSampler,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + 'a>> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => Ok(Box::new(model.mtp_decode(first, remaining, drafts, sampler)?)),
            _ => {
                let _ = (first, remaining, drafts, sampler);
                Err(Error::Unsupported("MTP unavailable"))
            },
        }
    }

    /// Continue a validated full-logit readout using the caller's sampler.
    /// The callback runs only for requested draws; cleanup never samples.
    pub fn sampled_decode<'a>(
        &'a mut self,
        first: u32,
        remaining: usize,
        sample: impl FnMut(&[f32]) -> Result<u32> + 'a,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + 'a>> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => Ok(Box::new(model.sampled_decode(first, remaining, sample)?)),
            _ => {
                let _ = (first, remaining, sample);
                Err(Error::Unsupported("native sampled stream unavailable"))
            },
        }
    }

    pub fn step_greedy(
        &mut self,
        token: u32,
    ) -> Result<u32> {
        let result = if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            match &mut self.forward {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                Forward::Lily(model) => model.step_greedy(token),
                _ => Err(Error::Unsupported("native greedy readout unavailable")),
            }
        };
        if result.is_err() {
            self.reset();
        }
        result
    }
    /// Continue an already validated greedy draw within one serialized request.
    /// Dropping the iterator drains submitted work before the model is reusable.
    pub fn greedy_decode(
        &mut self,
        first: u32,
        remaining: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + '_>> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => Ok(Box::new(model.greedy_decode(first, remaining)?)),
            _ => {
                let _ = (first, remaining);
                Err(Error::Unsupported("native greedy stream unavailable"))
            },
        }
    }

    /// Includes failures found while draining an early-dropped greedy stream.
    pub fn is_fenced(&self) -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.is_fenced();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn mixed_precision_selection_requires_explicit_architecture_and_format() {
        assert!(mixed_precision_checkpoint(&serde_json::json!({
            "model_type": "qwen4_exp", "lily": {"format": "qwen4_exp-affine-v1"}
        })));
        assert!(!mixed_precision_checkpoint(&serde_json::json!({"model_type": "qwen4_exp"})));
        assert!(!mixed_precision_checkpoint(&serde_json::json!({
            "model_type": "flint", "lily": {"format": "qwen4_exp-affine-v1"}
        })));
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
