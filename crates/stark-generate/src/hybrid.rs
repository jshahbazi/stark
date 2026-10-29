//! Serialized text generation through Stark's native hybrid-family forward.

use std::{collections::BTreeMap, fs::File, io::Write, path::Path, time::Instant};

use stark_engine::engine::hybrid::{Model, profile};

use crate::{
    cancel::CancelSignal,
    generation::{
        GeneratedTokens, GenerationBackend, GenerationOptions, Sampling, TextCallback, collect_tokens_controlled,
        native_result_after_cleanup,
    },
    text_stop::StopFilter,
};

pub(crate) struct HybridBackend {
    model: Model,
    cancellation: Option<CancelSignal>,
    profile_output: Option<ProfileOutput>,
}

#[derive(serde::Serialize)]
struct AggregateProfile {
    schema_version: u8,
    scope: &'static str,
    success: bool,
    input_tokens: usize,
    sampled_tokens: Option<usize>,
    output_tokens: Option<usize>,
    counters: BTreeMap<&'static str, profile::Counter>,
}

/// Explicit diagnostic output contains only aggregate numeric counters. A new
/// private file is required; an existing file or symlink is never overwritten.
struct ProfileOutput {
    file: File,
    failed: bool,
}

impl ProfileOutput {
    fn open(path: &Path) -> Result<Self, &'static str> {
        #[cfg(unix)]
        {
            use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};

            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)
                .map_err(|_| "native_profile_output_unavailable")?;
            Ok(Self {
                file,
                failed: false,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Err("native_profile_output_unavailable")
        }
    }

    fn write(
        &mut self,
        record: &AggregateProfile,
    ) -> Result<(), &'static str> {
        if self.failed {
            return Err("native_profile_output_failed");
        }
        let result = (|| {
            let mut bytes = serde_json::to_vec(record).map_err(|_| "native_profile_output_failed")?;
            bytes.push(b'\n');
            self.file.write_all(&bytes).and_then(|()| self.file.flush()).map_err(|_| "native_profile_output_failed")
        })();
        self.failed |= result.is_err();
        result
    }
}

impl Drop for HybridBackend {
    fn drop(&mut self) {
        if self.profile_output.is_some() {
            profile::enable(false);
        }
    }
}

impl HybridBackend {
    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        let profile_output = std::env::var_os("STARK_QWEN38_PROFILE_OUTPUT")
            .map(|path| ProfileOutput::open(Path::new(&path)))
            .transpose()?;
        let mut model = Model::load(path).map_err(|_| "hybrid_model_load_failed")?;
        if profile_output.is_some() {
            model.set_bf16_stage_profiling(true).map_err(|_| "native_profile_initialization_failed")?;
            profile::enable(true);
        }
        Ok(Self {
            model,
            cancellation,
            profile_output,
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
        let gpu_greedy = matches!(options.sampling, Sampling::Greedy {}) && self.model.supports_greedy_readout();
        let sampled_decode =
            matches!(options.sampling, Sampling::Stochastic { .. }) && self.model.supports_sampled_decode();
        let cancellation = &mut self.cancellation;
        let mut cancelled = || match cancellation.as_mut() {
            Some(signal) => signal.check(options.request_id),
            None => Ok(false),
        };
        if input.is_empty() {
            return Err("invalid_generation_request");
        }
        let mut logits = Vec::new();
        let mut greedy_token = None;
        for (index, &token) in input.iter().enumerate() {
            if cancelled()? {
                return Err("generation_cancelled");
            }
            if index + 1 == input.len() {
                if gpu_greedy {
                    greedy_token = Some(self.model.step_greedy(token).map_err(|_| "native_generation_failed")?);
                } else {
                    logits = self.model.step(token).map_err(|_| "native_generation_failed")?;
                }
            } else {
                self.model.ingest(token).map_err(|_| "native_generation_failed")?;
            }
        }
        let model = &mut self.model;
        let mut previous = None;
        let mut rng = options.seed;
        let tokens: Box<dyn Iterator<Item = Result<u64, &'static str>> + '_> = if gpu_greedy {
            let first = greedy_token.take().ok_or("native_generation_failed")?;
            let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
            let tail = model
                .greedy_decode(first, budget.saturating_sub(1))
                .map_err(|_| "native_generation_failed")?
                .map(|token| token.map(u64::from).map_err(|_| "native_generation_failed"));
            Box::new(std::iter::once(Ok(u64::from(first))).chain(tail))
        } else if sampled_decode {
            let first = sample(&logits, &options.sampling, &mut rng)?;
            let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
            let tail = model
                .sampled_decode(first, budget.saturating_sub(1), |readout| {
                    sample(readout, &options.sampling, &mut rng)
                        .map_err(|_| stark_engine::engine::hybrid::Error::Invalid("sampled token selection failed"))
                })
                .map_err(|_| "native_generation_failed")?
                .map(|token| token.map(u64::from).map_err(|_| "native_generation_failed"));
            Box::new(std::iter::once(Ok(u64::from(first))).chain(tail))
        } else {
            Box::new(std::iter::from_fn(|| {
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
                Some(Ok(u64::from(token)))
            }))
        };
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
        // The collector has dropped its iterator and drained any submitted
        // lookahead, including EOS, cancellation and callback-error exits.
        let mut result = native_result_after_cleanup(result, self.model.is_fenced())?;
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
        if self.profile_output.is_some() {
            profile::reset();
        }
        self.model.reset();
        if self.profile_output.as_ref().is_some_and(|output| output.failed) {
            return Err("native_profile_output_failed");
        }
        let result = self.generate_inner(input, options, on_text);
        self.model.reset();
        if let Some(output) = self.profile_output.as_mut() {
            output.write(&AggregateProfile {
                schema_version: 1,
                scope: "qwen38_bf16_aggregate_profile",
                success: result.is_ok(),
                input_tokens: input.len(),
                sampled_tokens: result.as_ref().ok().map(|tokens| tokens.sampled_tokens),
                output_tokens: result.as_ref().ok().map(|tokens| tokens.generated_token_ids.len()),
                counters: profile::snapshot(),
            })?;
        }
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
    let _total_scope = profile::Scope::new("sample.total");
    {
        let _scope = profile::Scope::new("sample.finite");
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
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

    struct ProfileDirectory(std::path::PathBuf);

    impl ProfileDirectory {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};

            static SEQUENCE: AtomicU64 = AtomicU64::new(0);
            loop {
                let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
                let path =
                    std::env::temp_dir().join(format!("stark-sampler-profile-{}-{sequence}", std::process::id()));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
                    Err(_) => panic!("private profile test directory unavailable"),
                }
            }
        }
    }

    impl Drop for ProfileDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn aggregate_profile_fixture(success: bool) -> AggregateProfile {
        AggregateProfile {
            schema_version: 1,
            scope: "qwen38_bf16_aggregate_profile",
            success,
            input_tokens: 2,
            sampled_tokens: success.then_some(3),
            output_tokens: success.then_some(3),
            counters: BTreeMap::from([(
                "sample.select",
                profile::Counter {
                    calls: 3,
                    seconds: 0.002,
                    bytes: 0,
                    max_bytes: 0,
                },
            )]),
        }
    }

    #[cfg(unix)]
    #[test]
    fn profile_output_is_private_jsonl_with_only_aggregate_fields() {
        use std::os::unix::fs::PermissionsExt;

        let directory = ProfileDirectory::new();
        let path = directory.0.join("profile.jsonl");
        let mut output = ProfileOutput::open(&path).unwrap();
        output.write(&aggregate_profile_fixture(true)).unwrap();
        output.write(&aggregate_profile_fixture(false)).unwrap();
        assert!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o077 == 0);
        let contents = std::fs::read_to_string(&path).unwrap();
        let records =
            contents.lines().map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()).collect::<Vec<_>>();
        assert!(records.len() == 2);
        for record in &records {
            let fields = record.as_object().unwrap();
            let allowed =
                ["schema_version", "scope", "success", "input_tokens", "sampled_tokens", "output_tokens", "counters"];
            assert!(fields.len() == allowed.len() && fields.keys().all(|key| allowed.contains(&key.as_str())));
            assert!(record["schema_version"] == 1);
            assert!(record["counters"]["sample.select"]["calls"] == 3);
        }
        assert!(records[0]["success"] == true && records[1]["success"] == false);
        assert!(records[1]["sampled_tokens"].is_null() && records[1]["output_tokens"].is_null());
        assert!(contents.ends_with('\n'));
    }

    #[cfg(unix)]
    #[test]
    fn profile_output_does_not_replace_files_or_follow_leaf_symlinks() {
        let directory = ProfileDirectory::new();
        let existing = directory.0.join("existing.jsonl");
        std::fs::write(&existing, b"existing metadata").unwrap();
        assert!(ProfileOutput::open(&existing).is_err());
        let link = directory.0.join("link.jsonl");
        std::os::unix::fs::symlink(&existing, &link).unwrap();
        assert!(ProfileOutput::open(&link).is_err());
        assert!(std::fs::read(&existing).unwrap() == b"existing metadata");
    }

    #[test]
    fn profile_output_write_failure_is_sticky_and_authored() {
        let directory = ProfileDirectory::new();
        let path = directory.0.join("read-only.jsonl");
        std::fs::write(&path, b"existing metadata").unwrap();
        let mut output = ProfileOutput {
            file: File::open(&path).unwrap(),
            failed: false,
        };
        assert!(output.write(&aggregate_profile_fixture(true)) == Err("native_profile_output_failed"));
        assert!(output.failed);
        assert!(output.write(&aggregate_profile_fixture(false)) == Err("native_profile_output_failed"));
        assert!(std::fs::read(&path).unwrap() == b"existing metadata");
    }

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
}
