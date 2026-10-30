//! Serialized text generation through Stark's native hybrid-family forward.

use std::{path::Path, time::Instant};

use rayon::prelude::*;
use stark_engine::engine::hybrid::{Model, finite_guard, profile};

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
    probability_mapper: ProbabilityMapper,
}

const PARALLEL_PROBABILITY_MIN_CANDIDATES: usize = 65_536;

/// Request serialization permits one worker-owned pool. Its threads borrow
/// the current readout only during `install`, then park between maps. Small
/// candidate sets never start the pool, and failed startup stays serial.
#[derive(Default)]
struct ProbabilityMapper {
    pool: Option<rayon::ThreadPool>,
    attempted: bool,
}

impl ProbabilityMapper {
    fn map(
        &mut self,
        logits: &[f32],
        indices: &[usize],
        max: f32,
        temperature: f32,
    ) -> Vec<f64> {
        if indices.len() < PARALLEL_PROBABILITY_MIN_CANDIDATES {
            return probability_values(logits, indices, max, temperature, None);
        }
        if !self.attempted {
            self.attempted = true;
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
            if threads > 1 {
                self.pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok();
            }
        }
        probability_values(logits, indices, max, temperature, self.pool.as_ref())
    }
}

fn probability_values(
    logits: &[f32],
    indices: &[usize],
    max: f32,
    temperature: f32,
    pool: Option<&rayon::ThreadPool>,
) -> Vec<f64> {
    if let Some(pool) = pool {
        let mut probabilities = Vec::new();
        if probabilities.try_reserve_exact(indices.len()).is_ok() {
            probabilities.resize(indices.len(), 0.0);
            pool.install(|| {
                probabilities.par_iter_mut().zip(indices.par_iter()).for_each(|(p, &i)| {
                    *p = (((logits[i] - max) / temperature) as f64).exp();
                });
            });
            return probabilities;
        }
    }
    indices.iter().map(|&i| (((logits[i] - max) / temperature) as f64).exp()).collect()
}

impl HybridBackend {
    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        let model = Model::load(path).map_err(|error| match error {
            stark_engine::engine::hybrid::Error::ConversionRequired => "model_conversion_required",
            _ => "hybrid_model_load_failed",
        })?;
        Ok(Self {
            model,
            cancellation,
            probability_mapper: ProbabilityMapper::default(),
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
        let probability_mapper = &mut self.probability_mapper;
        let mut previous = None;
        let mut rng = options.seed;
        let tokens = std::iter::from_fn(|| {
            if let Some(token) = previous {
                match model.step(token) {
                    Ok(next) => logits = next,
                    Err(_) => return Some(Err("native_generation_failed")),
                }
            }
            let token = match sample_with_mapper(&logits, &options.sampling, &mut rng, Some(&mut *probability_mapper)) {
                Ok(token) => token,
                Err(error) => return Some(Err(error)),
            };
            previous = Some(token);
            Some(Ok(u64::from(token)))
        });
        let mut decoder = tokenizer.decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let result = collect_tokens_controlled(
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
        );
        let mut result = result?;
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
        let _activity = stark_engine::engine::hybrid::activity::InferenceActivity::begin()
            .map_err(|_| "inference_activity_unavailable")?;
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

#[cfg(test)]
fn sample(
    logits: &[f32],
    sampling: &Sampling,
    rng: &mut u64,
) -> Result<u32, &'static str> {
    sample_with_mapper(logits, sampling, rng, None)
}

fn sample_with_mapper(
    logits: &[f32],
    sampling: &Sampling,
    rng: &mut u64,
    mapper: Option<&mut ProbabilityMapper>,
) -> Result<u32, &'static str> {
    let _total_scope = profile::Scope::new("sample.total");
    {
        let _scope = profile::Scope::new("sample.finite");
        if logits.is_empty() || !finite_guard::all_finite(logits) {
            return Err("native_generation_failed");
        }
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
    let indices = {
        let _scope = profile::Scope::new("sample.select");
        sampling_indices(logits, *top_k)?
    };
    let (probabilities, kept, cumulative) = {
        let _scope = profile::Scope::new("sample.probability_filter");
        let max = logits[indices[0]];
        let probabilities = match mapper {
            Some(mapper) => mapper.map(logits, &indices, max, *temperature),
            None => probability_values(logits, &indices, max, *temperature, None),
        };
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
        (probabilities, kept, cumulative)
    };
    let _draw_scope = profile::Scope::new("sample.draw");
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

/// Only the requested candidate prefix needs sorting. The comparison is the
/// same total F32 order and ascending ID tie-break as the full-sort reference.
/// The caller has already rejected nonfinite logits. Full-vocabulary sampling
/// uses stable integer radix ordering without changing probability sum order.
fn sampling_indices(
    logits: &[f32],
    top_k: Option<u32>,
) -> Result<Vec<usize>, &'static str> {
    let k = top_k.map_or(logits.len(), |k| (k as usize).min(logits.len()));
    if k == 0 {
        return Err("native_generation_failed");
    }
    if k == logits.len() {
        return Ok(radix_sampling_indices(logits));
    }
    if k <= 64
        && let Some(indices) = bounded_sampling_indices(logits, k)
    {
        return Ok(indices);
    }
    let _scope = profile::Scope::new("sample.select.partial");
    let compare = |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b));
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k - 1, compare);
    indices.truncate(k);
    indices.sort_unstable_by(compare);
    Ok(indices)
}

/// Keep the small candidate prefix in order while scanning contiguous logits.
/// New IDs are always greater than existing IDs, so equal keys cannot displace
/// the last candidate. A bounded replacement budget retains linear selection
/// for adversarial input orders that repeatedly replace the complete prefix.
fn bounded_sampling_indices(
    logits: &[f32],
    k: usize,
) -> Option<Vec<usize>> {
    let _scope = profile::Scope::new("sample.select.bounded");
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    let mut worst_key = descending_sample_key(logits[indices[k - 1]]);
    let mut replacements = 0;
    for index in k..logits.len() {
        let key = descending_sample_key(logits[index]);
        if key >= worst_key {
            continue;
        }
        replacements += 1;
        if replacements > 1_024 {
            return None;
        }
        let position = indices.partition_point(|&candidate| descending_sample_key(logits[candidate]) <= key);
        indices.copy_within(position..k - 1, position + 1);
        indices[position] = index;
        worst_key = descending_sample_key(logits[indices[k - 1]]);
    }
    Some(indices)
}

/// An ascending integer key for descending F32 `total_cmp` order. Negative
/// representations invert all bits; nonnegative representations flip the sign
/// bit, then the final complement reverses their order. Signed zero remains
/// distinct, matching the former full-sort comparator exactly.
fn descending_sample_key(value: f32) -> u32 {
    let bits = value.to_bits();
    let mask = (((bits as i32) >> 31) as u32) | 0x8000_0000;
    !(bits ^ mask)
}

/// Three stable 11-bit passes preserve ascending input IDs for equal logit
/// keys. Packing keys with IDs avoids random logit gathers on later passes.
/// Logits, F64 probability arithmetic and RNG stay intact.
fn radix_sampling_indices(logits: &[f32]) -> Vec<usize> {
    let _scope = profile::Scope::new("sample.select.radix");
    // Model token IDs are u32. Keep the selector's prior usize contract for
    // callers with a larger slice, even though such IDs cannot be sampled.
    if logits.len() > u32::MAX as usize {
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
        return indices;
    }
    let mut histograms = [[0_usize; 2_048]; 3];
    let mut entries: Vec<u64> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| {
            let key = descending_sample_key(logit);
            for (pass, histogram) in histograms.iter_mut().enumerate() {
                histogram[((key >> (pass * 11)) & 0x7ff) as usize] += 1;
            }
            (u64::from(key) << 32) | index as u64
        })
        .collect();
    let mut scratch = vec![0; logits.len()];
    for (pass, histogram) in histograms.iter().enumerate() {
        let mut offsets = [0_usize; 2_048];
        let mut end = 0;
        for (offset, &count) in offsets.iter_mut().zip(histogram) {
            *offset = end;
            end += count;
        }
        debug_assert_eq!(end, logits.len());
        for &entry in &entries {
            let digit = ((entry >> (32 + pass * 11)) & 0x7ff) as usize;
            let destination = &mut offsets[digit];
            scratch[*destination] = entry;
            *destination += 1;
        }
        std::mem::swap(&mut entries, &mut scratch);
    }
    entries.into_iter().map(|entry| entry as u32 as usize).collect()
}

#[cfg(test)]
#[path = "hybrid_sampler_bench.rs"]
mod sampler_bench;

#[cfg(test)]
mod tests {
    use stark_engine::engine::hybrid::math;

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

    /// The former production sampler is retained as the differential oracle.
    /// It fully sorts before truncating and preserves the original F64 order.
    fn full_sort_sample_reference(
        logits: &[f32],
        sampling: &Sampling,
        rng: &mut u64,
    ) -> u32 {
        let Sampling::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
        } = sampling
        else {
            panic!("stochastic reference requires stochastic options");
        };
        let mut indices = math::topk(logits, logits.len()).unwrap();
        if let Some(k) = top_k {
            indices.truncate(*k as usize);
        }
        let max = logits[indices[0]];
        let probabilities: Vec<f64> =
            indices.iter().map(|&i| (((logits[i] - max) / temperature) as f64).exp()).collect();
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
                return index as u32;
            }
        }
        indices[kept - 1] as u32
    }

    fn sampler_fixture(count: usize) -> Vec<f32> {
        let mut state = 0x4e32_a6c7_7f11_9865_u64;
        (0..count)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 31 == 0 {
                    if index % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    ((state % 257) as f32 - 128.0) / 16.0
                }
            })
            .collect()
    }

    fn finite_ieee_fixture(count: usize) -> Vec<f32> {
        let mut state = 0x6ad3_e802_9c61_4b75_u64;
        (0..count)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match index % 17 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::MAX,
                    3 => -f32::MAX,
                    4 => f32::from_bits(1),
                    5 => -f32::from_bits(1),
                    _ => {
                        let mut bits = state as u32;
                        if bits & 0x7f80_0000 == 0x7f80_0000 {
                            bits ^= 0x0080_0000;
                        }
                        f32::from_bits(bits)
                    },
                }
            })
            .collect()
    }

    #[test]
    fn radix_order_matches_total_cmp_across_finite_ieee_exponents_and_edges() {
        let mut logits = vec![0.0, -0.0, f32::MAX, -f32::MAX, f32::from_bits(1), -f32::from_bits(1)];
        for exponent in 0_u32..255 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_ffff] {
                let bits = (exponent << 23) | fraction;
                logits.push(f32::from_bits(bits));
                logits.push(f32::from_bits(bits | 0x8000_0000));
            }
        }
        // Repeat the values so stable ordering of identical keys is observed.
        logits.extend_from_within(..);
        assert!(logits.iter().all(|value| value.is_finite()));
        let expected = math::topk(&logits, logits.len()).unwrap();
        assert!(radix_sampling_indices(&logits) == expected, "radix IEEE total order changed");
        for pair in logits.windows(2) {
            assert!(descending_sample_key(pair[0]).cmp(&descending_sample_key(pair[1])) == pair[1].total_cmp(&pair[0]));
        }
    }

    #[test]
    fn radix_full_vocabulary_order_and_seeded_draws_match_full_sort() {
        const VOCAB: usize = 248_320;
        for logits in [sampler_fixture(VOCAB), finite_ieee_fixture(VOCAB)] {
            assert!(logits.iter().all(|value| value.is_finite()));
            let expected = math::topk(&logits, logits.len()).unwrap();
            for top_k in [None, Some(VOCAB as u32)] {
                assert!(
                    sampling_indices(&logits, top_k).unwrap() == expected,
                    "production vocabulary ordering changed"
                );
            }
            for (top_k, temperature, top_p, min_p) in [
                (None, 1.0, Some(0.95), None),
                (None, 0.6, Some(0.8), Some(0.05)),
                (None, 1.0, None, None),
                (Some(VOCAB as u32), 100.0, Some(1.0), Some(0.0)),
            ] {
                let options = Sampling::Stochastic {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                };
                for seed in [0_u64, 1, 42, u64::MAX] {
                    let mut actual_rng = seed;
                    let mut reference_rng = seed;
                    let actual = sample(&logits, &options, &mut actual_rng).unwrap();
                    let reference = full_sort_sample_reference(&logits, &options, &mut reference_rng);
                    assert!(actual == reference, "full vocabulary seeded draw changed");
                    assert!(actual_rng == reference_rng, "full vocabulary RNG advance changed");
                }
            }
        }
    }

    #[test]
    fn partial_selection_preserves_full_sort_prefix_and_total_order() {
        let extremes = [
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            0.0,
            -0.0,
            1.0,
            1.0,
        ];
        for count in [1, 2, 3, 31, 32, 33, 127, 128, 129, 1_025] {
            for logits in [
                sampler_fixture(count),
                vec![1.0; count],
                (0..count).map(|index| extremes[index % extremes.len()]).collect(),
            ] {
                let sorted = math::topk(&logits, logits.len()).unwrap();
                assert!(sampling_indices(&logits, None).unwrap() == sorted, "unbounded sort order changed");
                for k in [1, 2, 3, 7, 20, 32, 64, count.saturating_sub(1), count] {
                    if k == 0 || k > count {
                        continue;
                    }
                    let actual = sampling_indices(&logits, Some(k as u32)).unwrap();
                    assert!(actual.as_slice() == &sorted[..k], "partial candidate prefix changed");
                }
            }
        }
        // Cover the production vocabulary size and a larger candidate prefix.
        let logits = sampler_fixture(248_320);
        let sorted = math::topk(&logits, logits.len()).unwrap();
        for k in [1, 20, 64, 1_024] {
            assert!(sampling_indices(&logits, Some(k as u32)).unwrap().as_slice() == &sorted[..k]);
        }
    }

    #[test]
    fn bounded_selection_preserves_adversarial_orders_and_fallback_boundary() {
        for k in 1..=64 {
            for replacements in [1_023, 1_024, 1_025, 1_026] {
                let count = k + replacements;
                let ascending: Vec<f32> = (0..count).map(|index| index as f32).collect();
                assert!(
                    bounded_sampling_indices(&ascending, k).is_some() == (replacements <= 1_024),
                    "bounded replacement budget changed"
                );
                let descending: Vec<f32> = ascending.iter().rev().copied().collect();
                let signed_zero: Vec<f32> = (0..count)
                    .map(|index| {
                        if index.is_multiple_of(2) {
                            -0.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                for logits in [&ascending, &descending, &signed_zero] {
                    let expected = math::topk(logits, k).unwrap();
                    assert!(
                        sampling_indices(logits, Some(k as u32)).unwrap() == expected,
                        "bounded or fallback candidate order changed"
                    );
                }
            }
        }
    }

    #[test]
    fn partial_selection_preserves_seeded_samples_and_rng_advance() {
        let extreme_values = [f32::MAX, -f32::MAX, 0.0, -0.0, f32::from_bits(1), -f32::from_bits(1), 1.0, -1.0];
        let fixtures = [
            sampler_fixture(33),
            sampler_fixture(257),
            vec![1.0; 129],
            (0..65).map(|index| extreme_values[index % extreme_values.len()]).collect(),
        ];
        for logits in fixtures {
            for top_k in [None, Some(1), Some(2), Some(7), Some(20), Some(logits.len() as u32)] {
                for temperature in [0.01, 0.6, 1.0, 100.0] {
                    for (top_p, min_p) in [
                        (None, None),
                        (Some(0.95), None),
                        (None, Some(0.05)),
                        (Some(0.8), Some(0.2)),
                        (Some(0.5), Some(0.0)),
                        (Some(1.0), Some(1.0)),
                    ] {
                        let options = Sampling::Stochastic {
                            temperature,
                            top_k,
                            top_p,
                            min_p,
                        };
                        for seed in [0_u64, 1, 12, 99, 1 << 63, u64::MAX] {
                            let mut actual_rng = seed;
                            let mut reference_rng = seed;
                            let actual = sample(&logits, &options, &mut actual_rng).unwrap();
                            let expected = full_sort_sample_reference(&logits, &options, &mut reference_rng);
                            assert!(actual == expected, "seeded draw changed");
                            assert!(actual_rng == reference_rng, "draw consumed a different RNG state");
                            assert!(actual_rng == seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unbounded_sampling_preserves_full_sort_draws_and_rejects_nonfinite() {
        let options = Sampling::Stochastic {
            temperature: 0.7,
            top_k: None,
            top_p: Some(0.9),
            min_p: Some(0.02),
        };
        let logits = sampler_fixture(1_025);
        for seed in 0..128 {
            let mut actual_rng = seed;
            let mut reference_rng = seed;
            assert!(
                sample(&logits, &options, &mut actual_rng).unwrap()
                    == full_sort_sample_reference(&logits, &options, &mut reference_rng)
            );
            assert!(actual_rng == reference_rng);
        }
        for bits in [0x7fc0_0000, 0x7f80_0001, 0xffc0_0000, 0xff80_0001, 0x7f80_0000, 0xff80_0000] {
            let bad = f32::from_bits(bits);
            for top_k in [None, Some(1), Some(2)] {
                let invalid = Sampling::Stochastic {
                    temperature: 0.7,
                    top_k,
                    top_p: Some(0.9),
                    min_p: Some(0.02),
                };
                assert!(sample(&[0.0, bad], &invalid, &mut 0).is_err());
            }
        }
        assert!(sample(&[], &options, &mut 0).is_err());
    }

    #[test]
    fn probability_pool_is_lazy_and_startup_failure_retains_serial_mapping() {
        let mut mapper = ProbabilityMapper::default();
        let logits = sampler_fixture(PARALLEL_PROBABILITY_MIN_CANDIDATES);
        let indices = sampling_indices(&logits, None).unwrap();
        let max = logits[indices[0]];
        let small = mapper.map(&logits, &indices[..20], max, 0.7);
        let expected = probability_values(&logits, &indices[..20], max, 0.7, None);
        assert!(small.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(!mapper.attempted && mapper.pool.is_none(), "small sets started the probability pool");
        let _ = mapper.map(&logits, &indices, max, 0.7);
        assert!(mapper.attempted, "large set did not attempt pool startup");
        if let Some(pool) = &mapper.pool {
            assert!((2..=4).contains(&pool.current_num_threads()));
        }
        let mut failed_mapper = ProbabilityMapper {
            pool: None,
            attempted: true,
        };
        let fallback = failed_mapper.map(&logits, &indices, max, 0.7);
        let expected = probability_values(&logits, &indices, max, 0.7, None);
        assert!(
            fallback.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
            "serial fallback changed probability bits"
        );
        assert!(failed_mapper.pool.is_none(), "failed pool startup was retried");
    }

    #[test]
    fn parallel_probability_mapping_preserves_bits_for_finite_edges() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let n = PARALLEL_PROBABILITY_MIN_CANDIDATES;
        let edges = [
            0.0,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            1.0,
            -1.0,
        ];
        for logits in [
            sampler_fixture(n),
            (0..n).map(|i| edges[i % edges.len()]).collect(),
            vec![0.0; n],
            (0..n).map(|i| f32::from_bits((i as u32).wrapping_mul(104729) & 0x807f_ffff)).collect(),
        ] {
            let indices = sampling_indices(&logits, None).unwrap();
            let max = logits[indices[0]];
            for temperature in [0.01, 0.7, 1.0, 32.0] {
                let reference = probability_values(&logits, &indices, max, temperature, None);
                let parallel = probability_values(&logits, &indices, max, temperature, Some(&pool));
                assert!(reference.len() == parallel.len());
                assert!(
                    reference.iter().zip(&parallel).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "parallel probability bits changed"
                );
                assert!(
                    reference.iter().sum::<f64>().to_bits() == parallel.iter().sum::<f64>().to_bits(),
                    "serial probability sum changed"
                );
            }
        }
    }

    #[test]
    fn parallel_sampler_preserves_serial_filter_draw_and_rng() {
        let logits = sampler_fixture(PARALLEL_PROBABILITY_MIN_CANDIDATES);
        let mut mapper = ProbabilityMapper::default();
        for top_k in [None, Some(20)] {
            for temperature in [0.01, 0.7, 1.0, 32.0] {
                for (top_p, min_p) in [(None, None), (Some(0.95), None), (Some(0.5), Some(0.000001))] {
                    let sampling = Sampling::Stochastic {
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                    };
                    for seed in [0, 42, u64::MAX] {
                        let mut reference_rng = seed;
                        let mut parallel_rng = seed;
                        let reference = sample(&logits, &sampling, &mut reference_rng).unwrap();
                        let parallel =
                            sample_with_mapper(&logits, &sampling, &mut parallel_rng, Some(&mut mapper)).unwrap();
                        assert!(reference == parallel, "parallel seeded draw changed");
                        assert!(reference_rng == parallel_rng, "parallel RNG advancement changed");
                    }
                }
            }
        }
    }
}
