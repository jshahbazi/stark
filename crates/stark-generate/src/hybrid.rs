//! Serialized text generation through Stark's native hybrid-family forward.

use std::{collections::BTreeMap, fs::File, io::Write, path::Path, time::Instant};

use rayon::prelude::*;
use stark_engine::engine::hybrid::{Model, NativeSampling, PrefixSnapshot, PreparedHybridImage, finite_guard, profile};

use crate::{
    cancel::CancelSignal,
    generation::{
        ContextPlanningRequest, GeneratedTokens, GenerationBackend, GenerationOptions, ImageArtifactIdentity,
        MtpCapabilities, PreparedImageMetadata, ProgressCallback, Sampling, TextCallback, admit_request_memory,
        collect_tokens_controlled, native_result_after_cleanup,
    },
    prefix_cache::{self, BoundaryEntry},
    text_stop::StopFilter,
};

pub(crate) mod mtp;
#[path = "hybrid_nucleus.rs"]
mod nucleus;

/// Recurrent snapshots also depend on whether the draft head ran in prefill.
const PREFIX_CACHE_KEY: &str = "model-instance+context-tokens+cache-scope+mtp-enabled";

fn greedy_host_workspace(
    sampling: &Sampling,
    mtp_drafts: u32,
) -> bool {
    mtp_drafts == 0 && matches!(sampling, Sampling::Greedy {})
}

fn native_sampling(options: &GenerationOptions) -> Option<NativeSampling> {
    let sampling = match options.sampling {
        Sampling::Greedy {} => Some(NativeSampling::Greedy {
            seed: options.seed,
        }),
        Sampling::Stochastic {
            temperature,
            top_k: None,
            top_p,
            min_p: None,
        } if temperature >= 0.01 => Some(NativeSampling::Stochastic {
            temperature,
            top_p: top_p.unwrap_or(1.0),
            seed: options.seed,
        }),
        _ => None,
    };
    sampling.filter(|sampling| sampling.is_supported())
}

pub(crate) struct HybridBackend {
    model: Model,
    cancellation: Option<CancelSignal>,
    profile_output: Option<ProfileOutput>,
    profile_mode: Option<AggregateProfileMode>,
    native_prefill_snapshot: Option<BTreeMap<&'static str, profile::Counter>>,
    probability_mapper: ProbabilityMapper,
    nucleus_sampler: nucleus::NucleusSampler,
    last_mtp: Option<serde_json::Value>,
    /// Published snapshot of the previous request's last full prefill chunk
    /// boundary. The model's own caches hold the rows below it.
    prefix_cache: Option<BoundaryEntry<PrefixSnapshot>>,
    /// The last successful request's snapshot, until its reply is written.
    pending_prefix: Option<BoundaryEntry<PrefixSnapshot>>,
    image_runtime_identity: Option<stark_engine::engine::MetalDeviceIdentity>,
}

pub(crate) struct HybridPreparedRequest {
    image: PreparedHybridImage,
    tokens: Box<[u32]>,
}

const PARALLEL_PROBABILITY_MIN_CANDIDATES: usize = 65_536;

/// Request serialization permits one worker-owned pool. Its threads borrow
/// the current readout only during `install`, then park between maps. Small
/// candidate sets never start the pool, and failed startup stays serial.
#[derive(Default)]
pub(crate) struct ProbabilityMapper {
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

// Reference and nucleus paths share one scalar exp implementation. Preserve
// F32 subtraction/division before crossing its F32 argument boundary.
#[inline(never)]
fn probability_exp_f32(value: f32) -> f64 {
    (value as f64).exp()
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
                    *p = probability_exp_f32((logits[i] - max) / temperature);
                });
            });
            return probabilities;
        }
    }
    indices.iter().map(|&i| probability_exp_f32((logits[i] - max) / temperature)).collect()
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
    #[serde(skip_serializing_if = "Option::is_none")]
    native_phase_counters: Option<NativePhaseCounters>,
}

#[derive(serde::Serialize)]
/// Native diagnostic categories are inclusive and must not be summed. Decode
/// ends at the final request snapshot, including iterator drain and cleanup.
struct NativePhaseCounters {
    boundary: &'static str,
    counter_semantics: &'static str,
    prefill: BTreeMap<&'static str, profile::Counter>,
    decode: BTreeMap<&'static str, ProfileCounterDelta>,
}

#[derive(serde::Serialize)]
struct ProfileCounterDelta {
    calls: u64,
    seconds: f64,
    bytes: u64,
    // A cumulative snapshot cannot recover a phase-local maximum when the
    // earlier phase set a higher mark. This value is deliberately not a delta.
    request_max_bytes_at_end: u64,
}

fn profile_counter_difference(
    start: &BTreeMap<&'static str, profile::Counter>,
    end: &BTreeMap<&'static str, profile::Counter>,
) -> BTreeMap<&'static str, ProfileCounterDelta> {
    let zero = profile::Counter::default();
    end.iter()
        .filter_map(|(&name, counter)| {
            let prior = start.get(name).unwrap_or(&zero);
            let seconds = counter.seconds - prior.seconds;
            let delta = ProfileCounterDelta {
                calls: counter.calls.saturating_sub(prior.calls),
                seconds: if seconds.is_finite() && seconds >= 0.0 {
                    seconds
                } else {
                    0.0
                },
                bytes: counter.bytes.saturating_sub(prior.bytes),
                request_max_bytes_at_end: counter.max_bytes,
            };
            (delta.calls != 0 || delta.seconds != 0.0 || delta.bytes != 0).then_some((name, delta))
        })
        .collect()
}

fn native_phase_counters(
    mode: Option<AggregateProfileMode>,
    prefill: Option<BTreeMap<&'static str, profile::Counter>>,
    total: &BTreeMap<&'static str, profile::Counter>,
) -> Option<NativePhaseCounters> {
    if mode != Some(AggregateProfileMode::NativeHybrid) {
        return None;
    }
    prefill.map(|prefill| NativePhaseCounters {
        boundary: "after_final_prompt_model_readout_before_initial_sampling",
        counter_semantics: "inclusive categories; prefill max_bytes is request-so-far; decode request_max_bytes_at_end is request-wide, not phase-local",
        decode: profile_counter_difference(&prefill, total),
        prefill,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AggregateProfileMode {
    Qwen38Bf16,
    NativeHybrid,
}

impl AggregateProfileMode {
    fn scope(self) -> &'static str {
        match self {
            Self::Qwen38Bf16 => "qwen38_bf16_aggregate_profile",
            Self::NativeHybrid => "native_hybrid_aggregate_profile",
        }
    }

    fn enables_bf16_stage_profile(self) -> bool {
        matches!(self, Self::Qwen38Bf16)
    }
}

fn aggregate_profile_mode(
    qwen_configured: bool,
    native_configured: bool,
) -> Result<Option<AggregateProfileMode>, &'static str> {
    match (qwen_configured, native_configured) {
        (true, true) => Err("native_profile_configuration_conflict"),
        (true, false) => Ok(Some(AggregateProfileMode::Qwen38Bf16)),
        (false, true) => Ok(Some(AggregateProfileMode::NativeHybrid)),
        (false, false) => Ok(None),
    }
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
    fn record_mtp_metadata(
        &mut self,
        drafts: u32,
    ) -> Result<(), &'static str> {
        self.last_mtp = None;
        if let Some(capabilities) = self.model.mtp_capabilities() {
            let stats = if capabilities.sampling == "native-target-draw-match-v1" {
                serde_json::to_value(self.model.native_decode_stats())
            } else {
                serde_json::to_value(self.model.mtp_stats())
            }
            .map_err(|_| "native_generation_failed")?;
            self.last_mtp = Some(serde_json::json!({
                "enabled":drafts > 0,
                "drafts":drafts,
                "sampling":capabilities.sampling,
                "verification":capabilities.verification,
                "stats":stats,
            }));
        }
        Ok(())
    }

    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, false)
    }

    pub(crate) fn load_vision(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, true)
    }

    fn load_model(
        path: &Path,
        cancellation: Option<CancelSignal>,
        images: bool,
    ) -> Result<Self, &'static str> {
        let qwen_profile_path = std::env::var_os("STARK_QWEN38_PROFILE_OUTPUT");
        let native_profile_path = std::env::var_os("STARK_NATIVE_PROFILE_OUTPUT");
        let profile_mode = aggregate_profile_mode(qwen_profile_path.is_some(), native_profile_path.is_some())?;
        let profile_path = qwen_profile_path.or(native_profile_path);
        let profile_output = profile_path.map(|path| ProfileOutput::open(Path::new(&path))).transpose()?;
        let mut model = if images {
            Model::load_with_image_support(path, true)
        } else {
            Model::load_with_native_decode(path, true)
        }
        .map_err(|_| "hybrid_model_load_failed")?;
        let image_runtime_identity = if images {
            if !model.supports_images() {
                return Err("images_not_supported");
            }
            Some(
                stark_engine::engine::Engine::<stark_engine::backends::metal::Metal>::new()
                    .map_err(|_| "metal_initialization_failed")?
                    .metal_device_identity(),
            )
        } else {
            None
        };
        if profile_mode.is_some_and(AggregateProfileMode::enables_bf16_stage_profile) {
            model.set_bf16_stage_profiling(true).map_err(|_| "native_profile_initialization_failed")?;
        }
        if profile_output.is_some() {
            profile::enable(true);
        }
        Ok(Self {
            model,
            cancellation,
            profile_output,
            profile_mode,
            native_prefill_snapshot: None,
            probability_mapper: ProbabilityMapper::default(),
            nucleus_sampler: nucleus::NucleusSampler::default(),
            last_mtp: None,
            prefix_cache: None,
            pending_prefix: None,
            image_runtime_identity,
        })
    }

    /// Where a scoped request captures its snapshot: the start of its final
    /// prefill chunk.
    fn prefix_boundary(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> Option<usize> {
        options.cache_scope?;
        let chunk = self.model.prefix_chunk_tokens()?;
        Some(prefix_cache::chunk_boundary(input.len(), chunk)).filter(|&boundary| boundary > 0)
    }

    /// Resumes `reused` prompt tokens from `retained` and captures the state
    /// at `capture` before ingesting the rest. Restore time counts toward the
    /// first-token stage.
    #[allow(clippy::too_many_arguments)]
    fn generate_inner(
        &mut self,
        input: &[u32],
        reused: usize,
        retained: Option<&PrefixSnapshot>,
        capture: Option<usize>,
        image: Option<PreparedHybridImage>,
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let started = Instant::now();
        let is_image = image.is_some();
        if reused > 0 {
            let snapshot =
                retained.filter(|snapshot| snapshot.position() == reused).ok_or("native_generation_failed")?;
            self.model.restore_prefix(snapshot).map_err(|_| "native_generation_failed")?;
        }
        if let Some(position) = capture.filter(|&position| position > reused) {
            self.model.capture_prefix_at(position).map_err(|_| "native_generation_failed")?;
        }
        let tokenizer = self.model.tokenizer().clone();
        let vocab_size = self.model.vocab_size();
        let native_policy = if is_image {
            None
        } else {
            self.model
                .mtp_capabilities()
                .filter(|caps| caps.sampling == "native-target-draw-match-v1")
                .and_then(|_| native_sampling(options))
        };
        let native_decode = native_policy.is_some();
        let gpu_greedy = !is_image
            && options.mtp_drafts == 0
            && matches!(options.sampling, Sampling::Greedy {})
            && self.model.supports_greedy_readout();
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
        let mut native_first = None;
        let mut prefill_control_error = None;
        if let Some(image) = image {
            let result = self.model.prefill_image_controlled(input, image, || match cancelled() {
                Ok(cancelled) => cancelled,
                Err(error) => {
                    prefill_control_error = Some(error);
                    true
                },
            });
            logits = match result {
                Ok(logits) => logits,
                Err(stark_engine::engine::hybrid::Error::Cancelled) => {
                    return Err(prefill_control_error.unwrap_or("generation_cancelled"));
                },
                Err(_) => return Err("native_generation_failed"),
            };
            if cancelled()? {
                return Err("generation_cancelled");
            }
        } else {
            let prefix = input.get(reused..input.len() - 1).ok_or("native_generation_failed")?;
            let prefill = self.model.ingest_prefix(prefix, || {
                let control = match cancelled() {
                    Ok(false) => return Ok(()),
                    Ok(true) => "generation_cancelled",
                    Err(error) => error,
                };
                prefill_control_error = Some(control);
                Err(stark_engine::engine::hybrid::Error::Invalid("prefill control stopped"))
            });
            if prefill.is_err() {
                return Err(prefill_control_error.unwrap_or("native_generation_failed"));
            }
            if cancelled()? {
                return Err("generation_cancelled");
            }
            let token = input[input.len() - 1];
            if let Some(policy) = native_policy {
                native_first =
                    Some(self.model.step_native_initial(token, policy).map_err(|_| "native_generation_failed")?);
            } else if gpu_greedy {
                greedy_token = Some(self.model.step_greedy(token).map_err(|_| "native_generation_failed")?);
            } else {
                logits = self.model.step(token).map_err(|_| "native_generation_failed")?;
            }
        }
        // Native-only diagnostics add one request boundary. Qwen's existing
        // BF16 diagnostic schema/stage hooks and profiling-off work stay intact.
        if self.profile_mode == Some(AggregateProfileMode::NativeHybrid) {
            self.native_prefill_snapshot = Some(profile::snapshot());
        }
        let model = &mut self.model;
        let probability_mapper = &mut self.probability_mapper;
        let nucleus_sampler = &mut self.nucleus_sampler;
        let mut previous = None;
        let mut rng = options.seed;
        let mut mtp_sampler;
        let tokens: Box<dyn Iterator<Item = Result<u64, &'static str>> + '_> = if native_decode {
            let first = native_first.take().ok_or("native_generation_failed")?;
            if options.eos_token_ids.contains(&first) || options.extra_stop_token_ids.contains(&first) {
                Box::new(std::iter::once(Ok(u64::from(first))))
            } else {
                let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
                let tail = model
                    .native_decode(first, budget, options.mtp_drafts as usize)
                    .map_err(|_| "native_generation_failed")?
                    .map(|token| token.map(u64::from).map_err(|_| "native_generation_failed"));
                Box::new(std::iter::once(Ok(u64::from(first))).chain(tail))
            }
        } else if options.mtp_drafts > 0 {
            // The first target draw comes from prefill. The MTP sampler owns
            // separate draft randomness and exact target rejection sampling.
            mtp_sampler = mtp::ExactMtpSampler::new(&options.sampling, &mut rng, probability_mapper);
            let first = if gpu_greedy {
                greedy_token.take().ok_or("native_generation_failed")?
            } else {
                mtp_sampler.sample_initial(&logits)?
            };
            let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
            let tail = model
                .mtp_decode(first, budget.saturating_sub(1), options.mtp_drafts as usize, &mut mtp_sampler)
                .map_err(|_| "native_generation_failed")?
                .map(|token| token.map(u64::from).map_err(|_| "native_generation_failed"));
            Box::new(std::iter::once(Ok(u64::from(first))).chain(tail))
        } else if gpu_greedy {
            let first = greedy_token.take().ok_or("native_generation_failed")?;
            let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
            let tail = model
                .greedy_decode(first, budget.saturating_sub(1))
                .map_err(|_| "native_generation_failed")?
                .map(|token| token.map(u64::from).map_err(|_| "native_generation_failed"));
            Box::new(std::iter::once(Ok(u64::from(first))).chain(tail))
        } else if sampled_decode {
            let first = nucleus_sampler.sample(&logits, &options.sampling, &mut rng, probability_mapper)?;
            let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
            let tail = model
                .sampled_decode(first, budget.saturating_sub(1), |readout| {
                    nucleus_sampler
                        .sample(readout, &options.sampling, &mut rng, probability_mapper)
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
                let token = match sample_with_mapper(&logits, &options.sampling, &mut rng, Some(probability_mapper)) {
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
            on_progress,
            &mut cancelled,
        );
        // The collector has dropped its iterator and drained any submitted
        // lookahead, including EOS, cancellation and callback-error exits.
        let native_cleanup_failed = native_decode && self.model.finish_native_request().is_err();
        let mut result = native_result_after_cleanup(result, self.model.is_fenced() || native_cleanup_failed)?;
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
    type PreparedImage = HybridPreparedRequest;
    fn idle_maintenance_interval(&self) -> Option<std::time::Duration> {
        self.model.idle_maintenance_interval()
    }
    fn maintain_idle_queue(&mut self) -> Result<(), &'static str> {
        self.model.maintain_idle_queue().map_err(|_| "native_idle_maintenance_failed")
    }

    fn supports_images(&self) -> bool {
        self.image_runtime_identity.is_some() && self.model.supports_images()
    }

    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        let identity = self.model.image_artifact_identity()?;
        Some(ImageArtifactIdentity {
            schema_version: identity.schema_version,
            artifact_sha256: &identity.artifact_sha256,
            config_sha256: &identity.config_sha256,
            tokenizer_sha256: &identity.tokenizer_sha256,
            processor_sha256: &identity.processor_sha256,
            template_sha256: &identity.template_sha256,
        })
    }

    fn image_family_profile(&self) -> Option<(&str, u32)> {
        self.model.image_family_profile()
    }
    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        self.model.image_source_identity()
    }

    fn prepare_image(
        &mut self,
        input: Vec<u32>,
        encoded: &[u8],
        context: u32,
        request_id: u64,
    ) -> Result<(HybridPreparedRequest, PreparedImageMetadata), &'static str> {
        self.clear_prefix_cache();
        if !self.supports_images() {
            return Err("images_not_supported");
        }
        if self.cancelled(request_id)? {
            return Err("generation_cancelled");
        }
        let image = self.model.prepare_image(encoded).map_err(|_| "invalid_image_request")?;
        let tokens = self.model.prepare_image_prompt(&input, &image, context).map_err(|_| "invalid_image_request")?;
        if self.cancelled(request_id)? {
            return Err("generation_cancelled");
        }
        let metadata = PreparedImageMetadata {
            input_tokens: tokens.len(),
            image_tokens: image.image_token_count(),
            grid: image.grid(),
            processed_dimensions: image.processed_dimensions(),
        };
        Ok((
            HybridPreparedRequest {
                image,
                tokens,
            },
            metadata,
        ))
    }

    fn generate_image(
        &mut self,
        prepared: HybridPreparedRequest,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let input_tokens = u32::try_from(prepared.tokens.len()).map_err(|_| "context_overflow")?;
        let end = input_tokens.checked_add(options.max_new_tokens).ok_or("context_overflow")?;
        if end > options.context_tokens {
            return Err("context_overflow");
        }
        // Prepared image tokens contribute to the same growing text caches.
        // The separately bounded encoder is not an automatic-context planner.
        let available = stark_resource_budget::available_memory_bytes().unwrap_or(0);
        admit_request_memory(
            self.model.generation_memory_bytes_for_request(
                end,
                input_tokens,
                false,
                greedy_host_workspace(&options.sampling, options.mtp_drafts),
            ),
            available,
        )?;
        self.clear_prefix_cache();
        self.last_mtp = None;
        self.nucleus_sampler.clear();
        if options.mtp_drafts != 0 {
            return Err("mtp_not_supported");
        }
        let _activity = stark_engine::engine::hybrid::activity::InferenceActivity::begin()
            .map_err(|_| "inference_activity_unavailable")?;
        if self.profile_output.is_some() {
            profile::reset();
            self.native_prefill_snapshot = None;
        }
        self.model.reset();
        self.model.set_mtp_enabled(false).map_err(|_| "mtp_not_supported")?;
        if self.profile_output.as_ref().is_some_and(|output| output.failed) {
            return Err("native_profile_output_failed");
        }
        let input_tokens = prepared.tokens.len();
        let result =
            self.generate_inner(&prepared.tokens, 0, None, None, Some(prepared.image), options, on_text, on_progress);
        // MTP remains advertised for text on this loaded instance. Image
        // replies must still acknowledge zero drafts with current counters.
        self.record_mtp_metadata(0)?;
        self.model.reset();
        let result = native_result_after_cleanup(result, self.model.is_fenced());
        self.nucleus_sampler.clear();
        if let Some(output) = self.profile_output.as_mut() {
            let counters = profile::snapshot();
            let native_phase_counters =
                native_phase_counters(self.profile_mode, self.native_prefill_snapshot.take(), &counters);
            output.write(&AggregateProfile {
                schema_version: 1,
                scope: self.profile_mode.expect("profile output has mode").scope(),
                success: result.is_ok(),
                input_tokens,
                sampled_tokens: result.as_ref().ok().map(|tokens| tokens.sampled_tokens),
                output_tokens: result.as_ref().ok().map(|tokens| tokens.generated_token_ids.len()),
                counters,
                native_phase_counters,
            })?;
        }
        result
    }
    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
    fn max_context_length(&self) -> Option<u32> {
        Some(self.model.max_context_length())
    }
    fn automatic_context_length(
        &self,
        input_tokens: u32,
        maximum: u32,
        memory_budget_bytes: usize,
    ) -> Option<u32> {
        self.model.automatic_context_length(input_tokens, maximum, memory_budget_bytes)
    }
    fn supports_automatic_context_planning(&self) -> bool {
        self.model.supports_automatic_context_planning()
    }
    fn plan_context_length(
        &mut self,
        request: &ContextPlanningRequest<'_>,
        memory_budget_bytes: usize,
    ) -> Result<Option<u32>, &'static str> {
        if self.model.is_fenced() {
            return Err("native_generation_failed");
        }
        let released = self.model.prepare_request_memory().map_err(|_| "native_generation_failed")?;
        let memory_budget_bytes = if released {
            stark_resource_budget::inference_memory_budget(stark_resource_budget::available_memory_bytes().unwrap_or(0))
        } else {
            memory_budget_bytes
        };
        let Some(input) = request.input_token_ids else {
            return Ok(self.model.automatic_context_length(request.input_tokens, request.maximum, memory_budget_bytes));
        };
        let capture = self.prefix_boundary(input, request.options).is_some();
        let greedy = greedy_host_workspace(&request.options.sampling, request.options.mtp_drafts);
        let ordinary = self.model.automatic_context_length_for_request(
            request.input_tokens,
            request.maximum,
            memory_budget_bytes,
            capture,
            greedy,
        );
        // A matching retained boundary may avoid allocating a new snapshot at
        // exactly its old context. That isolated cheaper point is not monotonic
        // and must be checked outside the ordinary binary search.
        let retained = self
            .prefix_cache
            .as_ref()
            .map(BoundaryEntry::context_tokens)
            .filter(|&context| context > request.input_tokens && context <= request.maximum)
            .filter(|&context| {
                let mut options = request.options.clone();
                options.context_tokens = context;
                options.max_new_tokens = context - request.input_tokens;
                let reused = self.prefix_cache_plan(input, &options);
                reused > 0
                    && self.prefix_boundary(input, &options).is_some_and(|boundary| boundary <= reused)
                    && self
                        .model
                        .generation_memory_bytes_for_request(context, request.input_tokens, false, greedy)
                        .is_some_and(|bytes| bytes <= memory_budget_bytes)
            });
        Ok(ordinary.into_iter().chain(retained).max())
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
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let input_tokens = u32::try_from(input.len()).map_err(|_| "context_overflow")?;
        let end = input_tokens.checked_add(options.max_new_tokens).ok_or("context_overflow")?;
        if end > options.context_tokens {
            return Err("context_overflow");
        }
        self.model.prepare_request_memory().map_err(|_| "native_generation_failed")?;
        let available = stark_resource_budget::available_memory_bytes().unwrap_or(0);
        let reused = self.prefix_cache_plan(input, options);
        let capture = self.prefix_boundary(input, options).is_some_and(|boundary| boundary > reused);
        admit_request_memory(
            self.model.generation_memory_bytes_for_request(
                end,
                input_tokens,
                capture,
                greedy_host_workspace(&options.sampling, options.mtp_drafts),
            ),
            available,
        )?;
        self.nucleus_sampler.clear();
        // A miss drops the snapshot; the next prefill overwrites the rows.
        let retained = self.prefix_cache.take().map(|entry| entry.state);
        self.pending_prefix = None;
        let _activity = stark_engine::engine::hybrid::activity::InferenceActivity::begin()
            .map_err(|_| "inference_activity_unavailable")?;
        if self.profile_output.is_some() {
            profile::reset();
            self.native_prefill_snapshot = None;
        }
        self.model.reset();
        self.last_mtp = None;
        self.model.set_mtp_enabled(options.mtp_drafts > 0).map_err(|_| "mtp_not_supported")?;
        if self.profile_output.as_ref().is_some_and(|output| output.failed) {
            return Err("native_profile_output_failed");
        }
        let boundary = self.prefix_boundary(input, options);
        let mut result =
            self.generate_inner(input, reused, retained.as_ref(), boundary, None, options, on_text, on_progress);
        // Host cancellation, callbacks and text decoding can fail after a
        // native iterator has closed successfully. Invalidate its native epoch
        // as well as the worker's retained entry before another request.
        if result.is_err()
            && self.model.mtp_capabilities().is_some_and(|caps| caps.sampling == "native-target-draw-match-v1")
            && self.model.abort_native_request().is_err()
        {
            result = Err("native_generation_failed");
        }
        if let Ok(generated) = result.as_mut() {
            generated.reused_input_tokens = reused;
            // A request that ended on the retained boundary keeps its snapshot.
            let snapshot = self
                .model
                .take_prefix_snapshot()
                .or_else(|| retained.filter(|_| boundary == Some(reused)))
                .filter(|snapshot| Some(snapshot.position()) == boundary);
            if let (Some(snapshot), Some(scope), Some(boundary)) = (snapshot, options.cache_scope, boundary) {
                self.pending_prefix =
                    Some(BoundaryEntry::new(snapshot, scope, options.context_tokens, &input[..boundary]));
            }
        }
        self.record_mtp_metadata(options.mtp_drafts)?;
        self.model.reset();
        result = native_result_after_cleanup(result, self.model.is_fenced());
        if result.is_err() {
            self.pending_prefix = None;
        }
        self.nucleus_sampler.clear();
        if let Some(output) = self.profile_output.as_mut() {
            let counters = profile::snapshot();
            let native_phase_counters =
                native_phase_counters(self.profile_mode, self.native_prefill_snapshot.take(), &counters);
            output.write(&AggregateProfile {
                schema_version: 1,
                scope: self.profile_mode.expect("profile output has mode").scope(),
                success: result.is_ok(),
                input_tokens: input.len(),
                sampled_tokens: result.as_ref().ok().map(|tokens| tokens.sampled_tokens),
                output_tokens: result.as_ref().ok().map(|tokens| tokens.generated_token_ids.len()),
                counters,
                native_phase_counters,
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
        if let Some(identity) = &self.image_runtime_identity {
            return Some(serde_json::json!({"backend":identity.backend,"device_name":identity.device_name,
                "apple_gpu_family":identity.apple_gpu_family,"gpu_core_count":identity.gpu_core_count,
                "supports_mxu":identity.supports_mxu}));
        }
        Some(serde_json::json!({"backend":"stark_native_hybrid_metal","execution":"autoregressive"}))
    }
    fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        self.model.mtp_capabilities().map(|capabilities| MtpCapabilities {
            max_drafts: capabilities.max_drafts,
            recommended_drafts: capabilities.recommended_drafts,
            sampling: capabilities.sampling,
            verification: capabilities.verification,
        })
    }
    fn mtp_metadata(&self) -> Option<serde_json::Value> {
        self.last_mtp.clone()
    }
    fn default_mtp_drafts(&self) -> u32 {
        self.model.default_mtp_drafts()
    }
    fn effective_mtp_drafts(
        &self,
        options: &GenerationOptions,
    ) -> Result<u32, &'static str> {
        if options.mtp_drafts == 0
            || !self.model.mtp_capabilities().is_some_and(|caps| caps.sampling == "native-target-draw-match-v1")
        {
            return Ok(options.mtp_drafts);
        }
        Ok(if native_sampling(options).is_some() {
            options.mtp_drafts
        } else {
            0
        })
    }
    fn prefix_cache_supported(&self) -> bool {
        self.model.prefix_chunk_tokens().is_some_and(|chunk| {
            chunk >= prefix_cache::PREFILL_CHUNK_TOKENS && chunk.is_multiple_of(prefix_cache::PREFILL_CHUNK_TOKENS)
        })
    }
    fn prefix_cache_chunk_tokens(&self) -> usize {
        self.model.prefix_chunk_tokens().unwrap_or(prefix_cache::PREFILL_CHUNK_TOKENS)
    }
    fn prefix_cache_key(&self) -> &'static str {
        PREFIX_CACHE_KEY
    }
    fn prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        match &self.prefix_cache {
            Some(entry) if !self.model.is_fenced() && entry.state.mtp_enabled() == (options.mtp_drafts > 0) => {
                entry.reusable(input, options.context_tokens, options.cache_scope.as_ref())
            },
            _ => 0,
        }
    }
    fn publish_prefix_cache(&mut self) {
        if let Some(entry) = self.pending_prefix.take() {
            self.prefix_cache = Some(entry);
        }
    }
    fn clear_prefix_cache(&mut self) {
        self.prefix_cache = None;
        self.pending_prefix = None;
    }
    fn prefix_cache_retained(&self) -> bool {
        self.prefix_cache.is_some() || self.pending_prefix.is_some()
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
    let distribution = sampling_distribution(logits, sampling, mapper)?;
    if matches!(sampling, Sampling::Greedy {}) {
        return Ok(distribution.indices[0] as u32);
    }
    let _draw_scope = profile::Scope::new("sample.draw");
    let target = uniform(rng) * distribution.total;
    let mut mass = 0.0;
    for (&index, &p) in distribution.indices.iter().zip(&distribution.weights) {
        mass += p;
        if target < mass {
            return Ok(index as u32);
        }
    }
    Ok(*distribution.indices.last().ok_or("native_generation_failed")? as u32)
}

struct Distribution {
    indices: Vec<usize>,
    weights: Vec<f64>,
    total: f64,
}

/// Shared by ordinary and speculative sampling: filtering and accumulation
/// retain the qualified F32 temperature and ordered F64 probability policy.
fn sampling_distribution(
    logits: &[f32],
    sampling: &Sampling,
    mapper: Option<&mut ProbabilityMapper>,
) -> Result<Distribution, &'static str> {
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
        return Ok(Distribution {
            indices: vec![best],
            weights: vec![1.0],
            total: 1.0,
        });
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
    Ok(Distribution {
        indices: indices.into_iter().take(kept).collect(),
        weights: probabilities.into_iter().take(kept).collect(),
        total: cumulative,
    })
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
    fn greedy_memory_mode_requires_greedy_sampling_and_disabled_mtp() {
        assert!(greedy_host_workspace(&Sampling::Greedy {}, 0));
        assert!(!greedy_host_workspace(&Sampling::Greedy {}, 1));
        assert!(!greedy_host_workspace(
            &Sampling::Stochastic {
                temperature: 1.0,
                top_k: None,
                top_p: Some(0.9),
                min_p: None,
            },
            0
        ));
    }

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
            native_phase_counters: None,
        }
    }

    fn counter(
        calls: u64,
        seconds: f64,
        bytes: u64,
        max_bytes: u64,
    ) -> profile::Counter {
        profile::Counter {
            calls,
            seconds,
            bytes,
            max_bytes,
        }
    }

    #[test]
    fn native_phase_difference_subtracts_additive_counters_without_inventing_maxima() {
        let start = BTreeMap::from([
            ("native.metal.gpu", counter(2, 0.5, 100, 90)),
            ("native.metal.map", counter(1, 0.25, 64, 64)),
            ("native.removed", counter(1, 0.125, 8, 8)),
        ]);
        let end = BTreeMap::from([
            ("native.metal.gpu", counter(5, 1.25, 140, 90)),
            ("native.metal.map", counter(1, 0.25, 64, 64)),
            ("native.sample", counter(1, 0.125, 7, 7)),
        ]);
        let delta = profile_counter_difference(&start, &end);
        assert_eq!(delta.len(), 2);
        assert_eq!(delta["native.metal.gpu"].calls, 3);
        assert_eq!(delta["native.metal.gpu"].seconds, 0.75);
        assert_eq!(delta["native.metal.gpu"].bytes, 40);
        // The earlier phase's 90-byte mark remains the request high-water
        // value; subtracting it or labeling it decode-local would be wrong.
        assert_eq!(delta["native.metal.gpu"].request_max_bytes_at_end, 90);
        assert_eq!(delta["native.sample"].calls, 1);
        assert_eq!(delta["native.sample"].seconds, 0.125);
        assert_eq!(delta["native.sample"].bytes, 7);
        assert!(!delta.contains_key("native.metal.map"));
        assert!(!delta.contains_key("native.removed"));
    }

    #[test]
    fn native_phase_difference_clamps_regressions_and_nonfinite_seconds() {
        let start = BTreeMap::from([("native.reset", counter(5, 3.0, 90, 90))]);
        let end = BTreeMap::from([
            ("native.reset", counter(1, 1.0, 20, 20)),
            ("native.nan", counter(1, f64::NAN, 1, 1)),
            ("native.infinity", counter(1, f64::INFINITY, 1, 1)),
        ]);
        let delta = profile_counter_difference(&start, &end);
        assert!(!delta.contains_key("native.reset"));
        assert_eq!(delta["native.nan"].seconds, 0.0);
        assert_eq!(delta["native.infinity"].seconds, 0.0);
        assert_eq!(delta["native.nan"].calls, 1);
        assert_eq!(delta["native.infinity"].bytes, 1);
        assert!(profile_counter_difference(&BTreeMap::new(), &BTreeMap::new()).is_empty());
    }

    #[test]
    fn native_phase_output_is_optional_mode_scoped_and_aggregate_only() {
        let prefill = BTreeMap::from([("native.metal.gpu", counter(2, 0.5, 100, 90))]);
        let total = BTreeMap::from([("native.metal.gpu", counter(5, 1.25, 140, 90))]);
        // Failed/cancelled requests that never reached prompt completion have
        // no boundary; later failures preserve the completed prefill snapshot.
        assert!(native_phase_counters(Some(AggregateProfileMode::NativeHybrid), None, &total).is_none());
        assert!(native_phase_counters(None, Some(prefill.clone()), &total).is_none());
        assert!(native_phase_counters(Some(AggregateProfileMode::Qwen38Bf16), Some(prefill.clone()), &total).is_none());
        let mut record = aggregate_profile_fixture(false);
        let qwen = serde_json::to_value(&record).unwrap();
        assert!(qwen.get("native_phase_counters").is_none());
        record.scope = AggregateProfileMode::NativeHybrid.scope();
        record.counters = total.clone();
        record.native_phase_counters =
            native_phase_counters(Some(AggregateProfileMode::NativeHybrid), Some(prefill), &total);
        let value = serde_json::to_value(&record).unwrap();
        let allowed = [
            "schema_version",
            "scope",
            "success",
            "input_tokens",
            "sampled_tokens",
            "output_tokens",
            "counters",
            "native_phase_counters",
        ];
        let fields = value.as_object().unwrap();
        assert_eq!(fields.len(), allowed.len());
        assert!(fields.keys().all(|key| allowed.contains(&key.as_str())));
        assert_eq!(value["success"], false);
        assert!(value["sampled_tokens"].is_null());
        let phases = &value["native_phase_counters"];
        let allowed = ["boundary", "counter_semantics", "prefill", "decode"];
        let fields = phases.as_object().unwrap();
        assert_eq!(fields.len(), allowed.len());
        assert!(fields.keys().all(|key| allowed.contains(&key.as_str())));
        assert_eq!(phases["boundary"], "after_final_prompt_model_readout_before_initial_sampling");
        assert!(phases["counter_semantics"].as_str().unwrap().contains("not phase-local"));
        for (phase, maximum) in [("prefill", "max_bytes"), ("decode", "request_max_bytes_at_end")] {
            let counters = phases[phase].as_object().unwrap();
            assert_eq!(counters.len(), 1);
            for fields in counters.values() {
                let fields = fields.as_object().unwrap();
                let allowed = ["calls", "seconds", "bytes", maximum];
                assert_eq!(fields.len(), allowed.len());
                assert!(fields.keys().all(|key| allowed.contains(&key.as_str())));
                assert!(fields.values().all(serde_json::Value::is_number));
            }
        }
        assert_eq!(phases["decode"]["native.metal.gpu"]["calls"], 3);
    }

    #[test]
    fn aggregate_profile_mode_keeps_scopes_distinct_and_rejects_conflict() {
        assert_eq!(aggregate_profile_mode(false, false), Ok(None));
        assert_eq!(aggregate_profile_mode(true, false), Ok(Some(AggregateProfileMode::Qwen38Bf16)));
        assert_eq!(aggregate_profile_mode(false, true), Ok(Some(AggregateProfileMode::NativeHybrid)));
        assert_eq!(aggregate_profile_mode(true, true), Err("native_profile_configuration_conflict"));
        assert_eq!(AggregateProfileMode::Qwen38Bf16.scope(), "qwen38_bf16_aggregate_profile");
        assert_eq!(AggregateProfileMode::NativeHybrid.scope(), "native_hybrid_aggregate_profile");
        assert!(AggregateProfileMode::Qwen38Bf16.enables_bf16_stage_profile());
        assert!(!AggregateProfileMode::NativeHybrid.enables_bf16_stage_profile());
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
