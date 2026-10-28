//! Serialized text generation through Stark's native hybrid-family forward.

use std::{path::Path, time::Instant};

use stark_engine::engine::hybrid::{Model, math};

use crate::{
    cancel::CancelSignal,
    generation::{
        GeneratedTokens, GenerationBackend, GenerationOptions, Sampling, TextCallback, collect_tokens_controlled,
    },
    text_stop::StopFilter,
};

pub(crate) struct HybridBackend {
    model: Model,
    cancellation: Option<CancelSignal>,
}

impl HybridBackend {
    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        let model = Model::load(path).map_err(|_| "hybrid_model_load_failed")?;
        Ok(Self {
            model,
            cancellation,
        })
    }

    fn generate_inner(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let started = Instant::now();
        let tokenizer = self.model.tokenizer().clone();
        let vocab_size = self.model.vocab_size();
        let cancellation = &mut self.cancellation;
        let mut cancelled = || match cancellation.as_mut() {
            Some(signal) => signal.check(options.request_id),
            None => Ok(false),
        };
        if input.is_empty() {
            return Err("invalid_generation_request");
        }
        let mut logits = Vec::new();
        for (index, &token) in input.iter().enumerate() {
            if cancelled()? {
                return Err("generation_cancelled");
            }
            if index + 1 == input.len() {
                logits = self.model.step(token).map_err(|_| "native_generation_failed")?;
            } else {
                self.model.ingest(token).map_err(|_| "native_generation_failed")?;
            }
        }
        let model = &mut self.model;
        let mut previous = None;
        let mut rng = options.seed;
        let tokens = std::iter::from_fn(|| {
            if let Some(token) = previous {
                match model.step(token) {
                    Ok(next) => logits = next,
                    Err(_) => return Some(Err("native_generation_failed")),
                }
            }
            let token = match sample(&logits, &options.sampling, &mut rng) {
                Ok(token) => token,
                Err(error) => return Some(Err(error)),
            };
            previous = Some(token);
            Some(Ok(token as u64))
        });
        let mut decoder = tokenizer.decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let mut result = collect_tokens_controlled(
            tokens,
            input.len(),
            options,
            vocab_size,
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
        )?;
        if needs_text {
            let full = tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
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

impl GenerationBackend for HybridBackend {
    type PreparedImage = ();
    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
    fn max_context_length(&self) -> Option<u32> {
        Some(self.model.max_context_length())
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.model.eos_token_ids()
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
        on_text: Option<&mut TextCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.model.reset();
        let result = self.generate_inner(input, options, on_text);
        self.model.reset();
        result
    }
    fn cancelled(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        match self.cancellation.as_mut() {
            Some(signal) => signal.check(request_id),
            None => Ok(false),
        }
    }
    fn runtime_identity(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"backend":"stark_native_hybrid_metal","execution":"autoregressive"}))
    }
}

fn uniform(state: &mut u64) -> f64 {
    // SplitMix64 supplies reproducible local sampling. It is not used for
    // secrets, identifiers or any other security-sensitive value.
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
}

fn sample(
    logits: &[f32],
    sampling: &Sampling,
    rng: &mut u64,
) -> Result<u32, &'static str> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("native_generation_failed");
    }
    let Sampling::Stochastic {
        temperature,
        top_k,
        top_p,
        min_p,
    } = sampling
    else {
        let mut best = 0;
        for index in 1..logits.len() {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        return Ok(best as u32);
    };
    let mut indices = math::topk(logits, logits.len()).map_err(|_| "native_generation_failed")?;
    if let Some(k) = top_k {
        indices.truncate(*k as usize);
    }
    let max = logits[indices[0]];
    let probabilities: Vec<f64> = indices.iter().map(|&i| (((logits[i] - max) / temperature) as f64).exp()).collect();
    let total = probabilities.iter().sum::<f64>();
    let mut kept = 0;
    let mut cumulative = 0.0;
    for &p in &probabilities {
        if kept > 0
            && (min_p.is_some_and(|minimum| p < minimum as f64)
                || top_p.is_some_and(|limit| cumulative / total >= limit as f64))
        {
            break;
        }
        cumulative += p;
        kept += 1;
    }
    let target = uniform(rng) * cumulative;
    let mut mass = 0.0;
    for (&index, &p) in indices[..kept].iter().zip(&probabilities) {
        mass += p;
        if target < mass {
            return Ok(index as u32);
        }
    }
    Ok(indices[kept - 1] as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hybrid_sampling_preserves_greedy_ties_and_filters() {
        assert_eq!(sample(&[1.0, 2.0, 2.0], &Sampling::Greedy {}, &mut 0).unwrap(), 1);
        let options = Sampling::Stochastic {
            temperature: 1.0,
            top_k: Some(1),
            top_p: Some(0.5),
            min_p: None,
        };
        assert_eq!(sample(&[-1000.0, 1000.0], &options, &mut 99).unwrap(), 1);
        assert!(sample(&[f32::NAN], &Sampling::Greedy {}, &mut 0).is_err());
    }
    #[test]
    fn stochastic_sampling_is_seeded_and_min_p_is_relative() {
        let options = Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: Some(0.9),
        };
        for seed in 0..100 {
            assert_eq!(sample(&[0.0, -1.0], &options, &mut seed.clone()).unwrap(), 0);
        }
        let options = Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
        };
        assert_eq!(sample(&[0.0, 0.0], &options, &mut 12).unwrap(), sample(&[0.0, 0.0], &options, &mut 12).unwrap());
    }
}
