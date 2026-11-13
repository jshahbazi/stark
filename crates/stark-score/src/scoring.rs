//! Isolated exact-token scoring protocol. Request content is never diagnostic data.
#[cfg(test)]
use std::io::Cursor;
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    os::{fd::FromRawFd, unix::fs::OpenOptionsExt},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine::{
    backends::metal::Metal,
    engine::{Engine, language_model::LanguageModel},
};
use stark_resource_budget::Budget;

use crate::cache::{CacheEntry, LastScoreCache};

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum ScoreRequest {
    ResetCache {},
    Tokenize {
        text: String,
        #[serde(default)]
        add_special_tokens: bool,
    },
    Score {
        input_token_ids: Vec<u32>,
        candidate_token_ids: Vec<u32>,
        context_tokens: u32,
        temperature: f64,
    },
}

#[derive(Serialize)]
struct ScoreMetrics {
    state_creation_seconds: f64,
    prefill_seconds: f64,
    readout_seconds: f64,
    score_seconds: f64,
    total_seconds: f64,
}

#[derive(Serialize)]
struct ScoreResult {
    input_tokens: usize,
    prompt_tokens_sha256: String,
    candidate_token_ids: Vec<u32>,
    logits: Vec<f64>,
    log_probabilities: Vec<f64>,
    candidate_probabilities: Vec<f64>,
    temperature: f64,
    cache_hit: bool,
    truncated: bool,
    metrics: ScoreMetrics,
    peak_memory_bytes: Option<usize>,
    cache: CacheTelemetry,
}

struct NativeBackend {
    engine: Option<Arc<Engine<Metal>>>,
    model: NativeModel,
    vocab_size: usize,
    mma_probe: Option<MmaScoreProbe<File>>,
}

/// Private qualification output, separate from the response protocol. The
/// software counter records encoded dispatches, not hardware utilization.
struct MmaScoreProbe<W: Write> {
    output: W,
    failed: bool,
    previous_profile: Option<bool>,
}

#[derive(Serialize)]
struct MmaScoreProbeRow {
    success: bool,
    // Totals retain their existing names; current records include Q5 + Q6.
    encoded_dispatch_calls: u64,
    encoded_weight_bytes: u64,
    encoded_q5_dispatch_calls: u64,
    encoded_q5_weight_bytes: u64,
}

impl MmaScoreProbe<File> {
    fn open(path: &Path) -> Result<Self, &'static str> {
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|_| "mma_score_probe_open_failed")?;
        Ok(Self {
            output,
            failed: false,
            previous_profile: None,
        })
    }

    fn from_env() -> Result<Option<Self>, &'static str> {
        std::env::var_os("STARK_GLM53_MMA_SCORE_PROBE").map(|path| Self::open(Path::new(&path))).transpose()
    }
}

impl<W: Write> MmaScoreProbe<W> {
    fn begin(&mut self) -> Result<(), &'static str> {
        if self.failed || self.previous_profile.is_some() {
            return Err("mma_score_probe_failed");
        }
        use stark_engine::engine::hybrid::profile;
        self.previous_profile = Some(profile::enabled());
        profile::reset();
        profile::enable(true);
        Ok(())
    }

    /// Called only after the serialized hybrid operation and model reset have
    /// fenced every submitted GPU writer, including its error cleanup.
    fn complete(
        &mut self,
        success: bool,
    ) -> Result<(), &'static str> {
        use stark_engine::engine::hybrid::profile;
        let previous = self.previous_profile.take().ok_or("mma_score_probe_failed")?;
        let counters = profile::snapshot();
        profile::enable(previous);
        let q6 = counters.get("native.glm.prefill.q6_mma");
        let q5 = counters.get("native.glm.prefill.q5_mma");
        let q5_calls = q5.map_or(0, |counter| counter.calls);
        let q5_bytes = q5.map_or(0, |counter| counter.bytes);
        let Some((calls, bytes)) = q6
            .map_or(0, |counter| counter.calls)
            .checked_add(q5_calls)
            .zip(q6.map_or(0, |counter| counter.bytes).checked_add(q5_bytes))
        else {
            self.failed = true;
            return Err("mma_score_probe_failed");
        };
        self.record(MmaScoreProbeRow {
            success,
            encoded_dispatch_calls: calls,
            encoded_weight_bytes: bytes,
            encoded_q5_dispatch_calls: q5_calls,
            encoded_q5_weight_bytes: q5_bytes,
        })
    }

    fn record(
        &mut self,
        row: MmaScoreProbeRow,
    ) -> Result<(), &'static str> {
        if self.failed {
            return Err("mma_score_probe_failed");
        }
        if write_reply(&mut self.output, &row).is_err() {
            self.failed = true;
            return Err("mma_score_probe_failed");
        }
        Ok(())
    }
}

impl<W: Write> Drop for MmaScoreProbe<W> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous_profile.take() {
            stark_engine::engine::hybrid::profile::enable(previous);
        }
    }
}

enum NativeModel {
    Dense(Box<LanguageModel<Metal>>),
    Hybrid(Box<stark_engine::engine::hybrid::Model>),
}

struct NativeScores {
    logits: Vec<f32>,
    state_creation_seconds: f64,
    prefill_seconds: f64,
    readout_seconds: f64,
}

fn admit_scoring_memory(
    request_bytes: Option<usize>,
    available: usize,
) -> Result<(), &'static str> {
    stark_resource_budget::admit_inference_memory(request_bytes, available)
}

fn admit_scoring_memory_with_reclaim(
    request_bytes: Option<usize>,
    mut available: impl FnMut() -> usize,
    reclaim: impl FnOnce() -> Result<usize, &'static str>,
) -> Result<(), &'static str> {
    let request_bytes = request_bytes.ok_or("resource_budget_exceeded")?;
    if admit_scoring_memory(Some(request_bytes), available()).is_ok() {
        return Ok(());
    }
    reclaim()?;
    // Released allocator pages are not a reservation. Check the actual host
    // headroom again rather than adding a hypothetical freed-byte allowance.
    admit_scoring_memory(Some(request_bytes), available())
}

trait ScoringBackend {
    fn idle_maintenance_interval(&self) -> Option<Duration> {
        None
    }
    fn maintain_idle(&mut self) -> Result<(), &'static str> {
        Ok(())
    }
    fn vocab_size(&self) -> usize;
    fn max_context_length(&self) -> Option<u32>;
    fn peak_memory_usage(&self) -> Option<usize>;
    fn tokenize(
        &mut self,
        text: String,
        special: bool,
    ) -> Result<Vec<u32>, &'static str>;
    fn score(
        &mut self,
        input: &[u32],
        context: u32,
    ) -> Result<NativeScores, &'static str>;
}

struct ScoreWorker<B> {
    backend: B,
    cache: LastScoreCache,
    digest_mode: bool,
    hits: u64,
    native_score_calls: u64,
    budget: Budget,
}

#[derive(Serialize)]
struct CacheTelemetry {
    enabled: bool,
    hits: u64,
    native_score_calls: u64,
}

struct PendingReply {
    value: Value,
    // None means a successful tokenize helper retains the committed entry.
    next_entry: Option<CacheEntry>,
    hit: bool,
}

fn write_reply(
    writer: &mut impl Write,
    reply: &impl Serialize,
) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, reply)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn token_hash(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn validate_score(
    input: &[u32],
    candidates: &[u32],
    context: u32,
    temperature: f64,
    vocab_size: usize,
) -> Result<(), &'static str> {
    if input.is_empty()
        || !(2..=16).contains(&candidates.len())
        || context == 0
        || !temperature.is_finite()
        || temperature <= 0.0
        || input.iter().chain(candidates).any(|id| *id as usize >= vocab_size)
        || candidates.iter().collect::<HashSet<_>>().len() != candidates.len()
    {
        return Err("invalid_scoring_request");
    }
    if input.len() > context as usize {
        return Err("context_overflow");
    }
    Ok(())
}

/// The common vocabulary normalization constant cancels from Jev's candidate
/// normalization. Apply its temperature directly to raw candidate logits in f64.
fn calibrated_probabilities(
    logits: &[f64],
    temperature: f64,
) -> Result<Vec<f64>, &'static str> {
    if logits.is_empty()
        || logits.iter().any(|value| !value.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
    {
        return Err("invalid_score_values");
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits.iter().map(|value| ((value - maximum) / temperature).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    Ok(exponentials.iter().map(|value| value / total).collect())
}

type CandidateReadout = (Vec<f64>, Vec<f64>, Vec<f64>);

fn candidate_readout(
    vocabulary_logits: &[f32],
    candidates: &[u32],
    temperature: f64,
) -> Result<CandidateReadout, &'static str> {
    if vocabulary_logits.is_empty()
        || vocabulary_logits.iter().any(|value| !value.is_finite())
        || candidates.iter().any(|id| *id as usize >= vocabulary_logits.len())
    {
        return Err("invalid_score_values");
    }
    let maximum = vocabulary_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_partition =
        maximum + vocabulary_logits.iter().map(|value| (f64::from(*value) - maximum).exp()).sum::<f64>().ln();
    let logits: Vec<f64> = candidates.iter().map(|id| f64::from(vocabulary_logits[*id as usize])).collect();
    let log_probabilities = logits.iter().map(|value| value - log_partition).collect();
    let candidate_probabilities = calibrated_probabilities(&logits, temperature)?;
    Ok((logits, log_probabilities, candidate_probabilities))
}

impl NativeBackend {
    fn load(path: &Path) -> Result<Self, &'static str> {
        if path.join("speculator").exists() {
            return Err("speculator_not_supported");
        }
        let hybrid = stark_engine::engine::hybrid::recognizes(path).map_err(|error| match error {
            stark_engine::engine::hybrid::Error::Io(_) => "model_config_unavailable",
            _ => "invalid_model_config",
        })?;
        if hybrid {
            let mma_probe = MmaScoreProbe::from_env()?;
            let model = stark_engine::engine::hybrid::Model::load(path).map_err(|_| "hybrid_model_load_failed")?;
            let vocab_size = model.vocab_size();
            return Ok(Self {
                engine: None,
                model: NativeModel::Hybrid(Box::new(model)),
                vocab_size,
                mma_probe,
            });
        }
        if std::env::var_os("STARK_GLM53_MMA_SCORE_PROBE").is_some() {
            return Err("mma_score_probe_requires_hybrid");
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
        Ok(Self {
            engine: Some(engine),
            model: NativeModel::Dense(Box::new(model)),
            vocab_size,
            mma_probe: None,
        })
    }
}

impl ScoringBackend for NativeBackend {
    fn idle_maintenance_interval(&self) -> Option<Duration> {
        match &self.model {
            NativeModel::Hybrid(model) => model.idle_maintenance_interval(),
            NativeModel::Dense(_) => None,
        }
    }
    fn maintain_idle(&mut self) -> Result<(), &'static str> {
        match &self.model {
            NativeModel::Hybrid(model) => model.maintain_idle_queue().map_err(|_| "native_idle_maintenance_failed"),
            NativeModel::Dense(_) => Ok(()),
        }
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn max_context_length(&self) -> Option<u32> {
        match &self.model {
            NativeModel::Dense(model) => model.max_context_length(),
            NativeModel::Hybrid(model) => Some(model.max_context_length()),
        }
    }
    fn peak_memory_usage(&self) -> Option<usize> {
        self.engine.as_ref().and_then(|engine| engine.peak_memory_usage())
    }
    fn tokenize(
        &mut self,
        text: String,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        let tokenizer = match &self.model {
            NativeModel::Dense(model) => model.tokenizer(),
            NativeModel::Hybrid(model) => model.tokenizer(),
        };
        tokenizer.encode(text, special).map(|encoding| encoding.get_ids().to_vec()).map_err(|_| "tokenization_failed")
    }
    fn score(
        &mut self,
        input: &[u32],
        context: u32,
    ) -> Result<NativeScores, &'static str> {
        let input_tokens = u32::try_from(input.len()).map_err(|_| "context_overflow")?;
        if let NativeModel::Hybrid(model) = &mut self.model {
            model.prepare_request_memory().map_err(|_| "native_scoring_failed")?;
        }
        let request_bytes = match &self.model {
            NativeModel::Dense(model) => model.scoring_memory_bytes(input_tokens, context),
            // Hybrid caches grow with committed prompt rows; unused selected
            // capacity does not allocate additional cache storage here.
            NativeModel::Hybrid(model) => model.scoring_memory_bytes(input_tokens),
        };
        match &self.model {
            NativeModel::Dense(model) => admit_scoring_memory_with_reclaim(
                request_bytes,
                || stark_resource_budget::available_memory_bytes().unwrap_or(0),
                || model.reclaim_cached_memory().map_err(|_| "native_scoring_failed"),
            )?,
            NativeModel::Hybrid(_) => {
                let available = stark_resource_budget::available_memory_bytes().unwrap_or(0);
                admit_scoring_memory(request_bytes, available)?;
            },
        }
        let NativeModel::Dense(model) = &self.model else {
            let NativeModel::Hybrid(model) = &mut self.model else {
                unreachable!()
            };
            let _activity = stark_engine::engine::hybrid::activity::InferenceActivity::begin()
                .map_err(|_| "inference_activity_unavailable")?;
            if let Some(probe) = &mut self.mma_probe {
                probe.begin()?;
            }
            let started = Instant::now();
            model.reset();
            let state_creation_seconds = started.elapsed().as_secs_f64();
            let prefill = Instant::now();
            let result = (|| {
                if input.is_empty() || input.len() > context as usize || context > model.max_context_length() {
                    return Err("context_overflow");
                }
                let mut logits = Vec::new();
                if std::env::var_os("STARK_GLM53_PREFILL_MMA").is_some_and(|value| value == "1") {
                    // Exercise the same known-prefix path as generation when
                    // qualifying the opt-in projection scheduling policy.
                    // The ordinary scalar scorer remains the reference.
                    let (&last, prefix) = input.split_last().ok_or("context_overflow")?;
                    model.ingest_prefix(prefix, || Ok(())).map_err(|_| "native_scoring_failed")?;
                    logits = model.step(last).map_err(|_| "native_scoring_failed")?;
                } else {
                    for (index, &token) in input.iter().enumerate() {
                        if index + 1 == input.len() {
                            logits = model.step(token).map_err(|_| "native_scoring_failed")?;
                        } else {
                            model.ingest(token).map_err(|_| "native_scoring_failed")?;
                        }
                    }
                }
                Ok(NativeScores {
                    logits,
                    state_creation_seconds,
                    prefill_seconds: prefill.elapsed().as_secs_f64(),
                    // The hybrid step includes its vocabulary projection.
                    readout_seconds: 0.0,
                })
            })();
            model.reset();
            if let Some(probe) = &mut self.mma_probe {
                probe.complete(result.is_ok())?;
            }
            return result;
        };
        model
            .score_next_token(input, context)
            .map(|scores| NativeScores {
                logits: scores.logits,
                state_creation_seconds: scores.state_creation_seconds,
                prefill_seconds: scores.prefill_seconds,
                readout_seconds: scores.readout_seconds,
            })
            .map_err(|_| "native_scoring_failed")
    }
}

impl<B: ScoringBackend> ScoreWorker<B> {
    #[cfg(test)]
    fn new(
        backend: B,
        enabled: bool,
        digest_mode: bool,
    ) -> Self {
        Self::new_with_budget(
            backend,
            enabled,
            digest_mode,
            Budget::from_environment().expect("valid test resource budget"),
        )
    }

    fn new_with_budget(
        backend: B,
        enabled: bool,
        digest_mode: bool,
        budget: Budget,
    ) -> Self {
        Self {
            backend,
            cache: LastScoreCache::new(enabled),
            digest_mode,
            hits: 0,
            native_score_calls: 0,
            budget,
        }
    }

    fn telemetry(
        &self,
        pending_hit: bool,
    ) -> CacheTelemetry {
        CacheTelemetry {
            enabled: self.cache.enabled(),
            hits: self.hits + u64::from(pending_hit),
            native_score_calls: self.native_score_calls,
        }
    }

    fn handle(
        &mut self,
        request: ScoreRequest,
    ) -> Result<PendingReply, &'static str> {
        let result = self.handle_inner(request);
        if result.is_err() {
            self.cache.clear();
        }
        result
    }

    fn handle_inner(
        &mut self,
        request: ScoreRequest,
    ) -> Result<PendingReply, &'static str> {
        match request {
            ScoreRequest::ResetCache {} => {
                self.cache.clear();
                Ok(PendingReply {
                    value: json!({"cache_reset":true,"cache":self.telemetry(false)}),
                    next_entry: None,
                    hit: false,
                })
            },
            ScoreRequest::Tokenize {
                text,
                add_special_tokens,
            } => {
                if text.is_empty() {
                    return Err("invalid_tokenization_request");
                }
                if text.len() > self.budget.input_bytes {
                    return Err("resource_budget_exceeded");
                }
                let started = Instant::now();
                let ids = self.backend.tokenize(text, add_special_tokens)?;
                if ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>() {
                    return Err("resource_budget_exceeded");
                }
                Ok(PendingReply {
                    value: json!({"token_ids":ids,"token_count":ids.len(),"token_ids_sha256":token_hash(&ids),
                    "vocab_size":self.backend.vocab_size(),"tokenization_seconds":started.elapsed().as_secs_f64(),
                    "cache":self.telemetry(false)}),
                    next_entry: None,
                    hit: false,
                })
            },
            ScoreRequest::Score {
                input_token_ids,
                candidate_token_ids,
                context_tokens,
                temperature,
            } => {
                if self.backend.vocab_size() > self.budget.input_bytes / std::mem::size_of::<f32>() {
                    return Err("resource_budget_exceeded");
                }
                if input_token_ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>()
                    || candidate_token_ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>()
                {
                    return Err("resource_budget_exceeded");
                }
                let started = Instant::now();
                let previous = self.cache.take_matching(&input_token_ids, context_tokens);
                validate_score(
                    &input_token_ids,
                    &candidate_token_ids,
                    context_tokens,
                    temperature,
                    self.backend.vocab_size(),
                )?;
                if self.backend.max_context_length().is_some_and(|maximum| context_tokens > maximum) {
                    return Err("context_overflow");
                }
                let prompt_tokens_sha256 = token_hash(&input_token_ids);
                let scoring_started = Instant::now();
                let cache_hit = previous.is_some();
                let scores = match previous {
                    Some(entry) => NativeScores {
                        logits: entry.logits,
                        state_creation_seconds: 0.0,
                        prefill_seconds: 0.0,
                        readout_seconds: 0.0,
                    },
                    None => {
                        // NativeBackend::score invokes score_next_token directly.
                        // This counts API dispatches, including failed attempts.
                        self.native_score_calls += 1;
                        self.backend.score(&input_token_ids, context_tokens)?
                    },
                };
                if scores.logits.len() != self.backend.vocab_size()
                    || scores.logits.iter().any(|value| !value.is_finite())
                {
                    return Err("invalid_score_values");
                }
                if scores.logits.len() > self.budget.input_bytes / std::mem::size_of::<f32>() {
                    return Err("resource_budget_exceeded");
                }
                let readout_started = Instant::now();
                let (logits, log_probabilities, candidate_probabilities) =
                    candidate_readout(&scores.logits, &candidate_token_ids, temperature)?;
                let readout_seconds = scores.readout_seconds + readout_started.elapsed().as_secs_f64();
                let result = ScoreResult {
                    input_tokens: input_token_ids.len(),
                    prompt_tokens_sha256,
                    candidate_token_ids,
                    logits,
                    log_probabilities,
                    candidate_probabilities,
                    temperature,
                    cache_hit,
                    truncated: false,
                    metrics: ScoreMetrics {
                        state_creation_seconds: scores.state_creation_seconds,
                        prefill_seconds: scores.prefill_seconds,
                        readout_seconds,
                        score_seconds: scoring_started.elapsed().as_secs_f64(),
                        total_seconds: started.elapsed().as_secs_f64(),
                    },
                    peak_memory_bytes: self.backend.peak_memory_usage(),
                    cache: self.telemetry(cache_hit),
                };
                let value = if self.digest_mode {
                    let mut digest = Sha256::new();
                    for logit in &scores.logits {
                        digest.update(logit.to_bits().to_le_bytes());
                    }
                    json!({"vocab_logits_sha256":format!("{:x}", digest.finalize()),
                        "vocab_size":self.backend.vocab_size(),"prompt_tokens_sha256":result.prompt_tokens_sha256,
                        "input_tokens":result.input_tokens,"cache_hit":cache_hit,"truncated":false,
                        "cache":self.telemetry(cache_hit)})
                } else {
                    serde_json::to_value(result).map_err(|_| "protocol_serialization_failed")?
                };
                Ok(PendingReply {
                    value,
                    next_entry: self.cache.enabled().then_some(CacheEntry {
                        input: input_token_ids,
                        context: context_tokens,
                        logits: scores.logits,
                    }),
                    hit: cache_hit,
                })
            },
        }
    }

    fn process_line(
        &mut self,
        bytes: &[u8],
        output: &mut impl Write,
    ) -> Result<(), &'static str> {
        let result = stark_resource_budget::check_json_structure(bytes, self.budget)
            .and_then(|()| serde_json::from_slice::<ScoreRequest>(bytes).map_err(|_| "invalid_request"))
            .and_then(|request| self.handle(request));
        match result {
            Ok(pending) => {
                if write_reply(output, &pending.value).is_err() {
                    self.cache.clear();
                    return Err("protocol_write_failed");
                }
                // The only publication site follows serialization, write, newline and flush.
                if let Some(entry) = pending.next_entry {
                    self.cache.publish(entry);
                }
                self.hits += u64::from(pending.hit);
                Ok(())
            },
            Err(code) => {
                self.cache.clear();
                write_reply(output, &json!({"error":code,"cache":self.telemetry(false)}))
                    .map_err(|_| "protocol_write_failed")?;
                if matches!(code, "native_scoring_failed" | "invalid_score_values" | "mma_score_probe_failed") {
                    Err("native_operation_failed")
                } else {
                    Ok(())
                }
            },
        }
    }
}

pub(super) fn run_lines(
    path: &Path,
    enabled: bool,
    digest_mode: bool,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    let budget = Budget::from_environment()?;
    let started = Instant::now();
    let mut worker = ScoreWorker::new_with_budget(NativeBackend::load(path)?, enabled, digest_mode, budget);
    write_reply(
        output,
        &json!({"ready":true,"engine":"stark","version":"0.1.0",
        "mode":if digest_mode { "exact-token-digest-cache-v1" } else { "exact-token-scoring-cache-v1" },
        "upstream_version":"0.6.1",
        "upstream_revision":"8d01581474760bd1fc7f4a547ff1d02f7dada437",
        "scoring_patch_sha256":option_env!("STARK_SCORING_PATCH_SHA256").unwrap_or("unverified"),
        "cache_policy":worker.cache.policy(),"cache_enabled":worker.cache.enabled(),
        "cache_key":"exact-token-ids+context-tokens+model-instance",
        "cache":worker.telemetry(false),
        "load_seconds":started.elapsed().as_secs_f64(),"vocab_size":worker.backend.vocab_size(),
        "max_context_tokens":worker.backend.max_context_length(),
        "max_request_bytes":worker.budget.wire_bytes,"request_budget_bytes":worker.budget.input_bytes,
        "peak_memory_bytes":worker.backend.peak_memory_usage()}),
    )
    .map_err(|_| "protocol_write_failed")?;
    run_request_loop(&mut worker, &mut StandardInput::reader(), output)
}

trait ScoreInput: BufRead {
    fn wait_readable(
        &mut self,
        timeout: Duration,
    ) -> io::Result<bool>;
}

#[cfg(test)]
impl<T: AsRef<[u8]>> ScoreInput for Cursor<T> {
    fn wait_readable(
        &mut self,
        _timeout: Duration,
    ) -> io::Result<bool> {
        Ok(true)
    }
}

/// stdin wrapper that does not close descriptor 0 and whose read-ahead is
/// visible to the timed wait below.
struct StandardInput(std::mem::ManuallyDrop<File>);

impl StandardInput {
    fn reader() -> BufReader<Self> {
        // SAFETY: descriptor 0 remains open for the process lifetime and this
        // ManuallyDrop wrapper never closes it.
        let file = unsafe { File::from_raw_fd(libc::STDIN_FILENO) };
        BufReader::with_capacity(64 * 1024, Self(std::mem::ManuallyDrop::new(file)))
    }
}

impl Read for StandardInput {
    fn read(
        &mut self,
        buffer: &mut [u8],
    ) -> io::Result<usize> {
        self.0.read(buffer)
    }
}

impl ScoreInput for BufReader<StandardInput> {
    fn wait_readable(
        &mut self,
        timeout: Duration,
    ) -> io::Result<bool> {
        if !self.buffer().is_empty() {
            return Ok(true);
        }
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        let milliseconds = timeout.as_millis().min(libc::c_int::MAX as u128) as libc::c_int;
        loop {
            match unsafe { libc::poll(&mut descriptor, 1, milliseconds) } {
                0 => return Ok(false),
                count if count > 0 => return Ok(true),
                _ => {
                    let error = io::Error::last_os_error();
                    if error.kind() != io::ErrorKind::Interrupted {
                        return Err(error);
                    }
                },
            }
        }
    }
}

fn run_request_loop<B: ScoringBackend, I: ScoreInput>(
    worker: &mut ScoreWorker<B>,
    input: &mut I,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    let interval = worker.backend.idle_maintenance_interval().filter(|interval| !interval.is_zero());
    loop {
        if let Some(interval) = interval {
            match input.wait_readable(interval) {
                Ok(true) => {},
                Ok(false) => {
                    if worker.backend.maintain_idle().is_err() {
                        worker.cache.clear();
                        return Err("native_operation_failed");
                    }
                    continue;
                },
                Err(_) => {
                    worker.cache.clear();
                    return Err("protocol_read_failed");
                },
            }
        }
        let mut bytes = Vec::new();
        let count = match Read::by_ref(input).take(worker.budget.wire_bytes as u64 + 1).read_until(b'\n', &mut bytes) {
            Ok(count) => count,
            Err(_) => {
                worker.cache.clear();
                return Err("protocol_read_failed");
            },
        };
        if count == 0 {
            worker.cache.clear();
            return Ok(());
        }
        if bytes.len() > worker.budget.wire_bytes {
            worker.cache.clear();
            write_reply(output, &json!({"error":"request_too_large","cache":worker.telemetry(false)}))
                .map_err(|_| "protocol_write_failed")?;
            return Ok(());
        }
        worker.process_line(&bytes, output)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converted_qwen_and_native_hybrid_types_dispatch_before_dense_loading() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-score-dispatch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        struct Directory(std::path::PathBuf);
        impl Drop for Directory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let directory = Directory(path);
        // Deliberately incomplete configurations fail before any Metal
        // allocation. The error identifies which real loader was selected.
        for config in [
            json!({"type":"Qwen38LanguageModelConfig"}),
            json!({"model_type":"glm5_next"}),
            json!({"model_type":"qwen4_exp"}),
            json!({"model_type":"flint"}),
            json!({"model_type":"ricursive"}),
        ] {
            std::fs::write(directory.0.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
            assert!(matches!(NativeBackend::load(&directory.0), Err("hybrid_model_load_failed")));
        }
        std::fs::write(directory.0.join("config.json"), br#"{"type":"LanguageModelConfig"}"#).unwrap();
        assert!(matches!(NativeBackend::load(&directory.0), Err("invalid_model_config")));
        std::fs::write(directory.0.join("config.json"), b"{").unwrap();
        assert!(matches!(NativeBackend::load(&directory.0), Err("invalid_model_config")));
    }

    struct FakeBackend {
        calls: usize,
        idle_interval: Option<Duration>,
        idle_calls: usize,
        fail_idle: bool,
        fail_score: bool,
        fail_resource: bool,
        fail_probe: bool,
        invalid_output: bool,
        fail_tokenize: bool,
        max_context: Option<u32>,
    }
    impl ScoringBackend for FakeBackend {
        fn idle_maintenance_interval(&self) -> Option<Duration> {
            self.idle_interval
        }
        fn maintain_idle(&mut self) -> Result<(), &'static str> {
            self.idle_calls += 1;
            if self.fail_idle {
                Err("private_idle_error")
            } else {
                Ok(())
            }
        }
        fn vocab_size(&self) -> usize {
            6
        }
        fn max_context_length(&self) -> Option<u32> {
            self.max_context
        }
        fn peak_memory_usage(&self) -> Option<usize> {
            None
        }
        fn tokenize(
            &mut self,
            _: String,
            _: bool,
        ) -> Result<Vec<u32>, &'static str> {
            if self.fail_tokenize {
                Err("tokenization_failed")
            } else {
                Ok(vec![1, 2])
            }
        }
        fn score(
            &mut self,
            _: &[u32],
            _: u32,
        ) -> Result<NativeScores, &'static str> {
            self.calls += 1;
            if self.fail_probe {
                return Err("mma_score_probe_failed");
            }
            if self.fail_score {
                return Err("native_scoring_failed");
            }
            if self.fail_resource {
                return Err("resource_budget_exceeded");
            }
            Ok(NativeScores {
                logits: if self.invalid_output {
                    vec![f32::NAN; 6]
                } else {
                    vec![-0.0, -1.0, 0.0, 1.0, 2.0, 3.0]
                },
                state_creation_seconds: 0.1,
                prefill_seconds: 0.2,
                readout_seconds: 0.3,
            })
        }
    }
    fn worker(
        enabled: bool,
        digest: bool,
    ) -> ScoreWorker<FakeBackend> {
        ScoreWorker::new(
            FakeBackend {
                calls: 0,
                idle_interval: None,
                idle_calls: 0,
                fail_idle: false,
                fail_score: false,
                fail_resource: false,
                fail_probe: false,
                invalid_output: false,
                fail_tokenize: false,
                max_context: Some(262_144),
            },
            enabled,
            digest,
        )
    }
    fn request(
        input: &[u32],
        candidates: &[u32],
        context: u32,
        temperature: f64,
    ) -> Value {
        json!({"operation":"score", "input_token_ids":input, "candidate_token_ids":candidates,
            "context_tokens":context,"temperature":temperature})
    }
    fn post(
        worker: &mut ScoreWorker<FakeBackend>,
        request: Value,
    ) -> Value {
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        serde_json::from_slice(&output).unwrap()
    }
    fn standard() -> Value {
        request(&[1, 2], &[2, 3], 8192, 1.22)
    }

    enum InputEvent {
        Timeout,
        Data(Vec<u8>),
        Eof,
        Error,
    }

    struct ScriptInput {
        current: Vec<u8>,
        at: usize,
        events: std::collections::VecDeque<InputEvent>,
    }

    impl ScriptInput {
        fn new(events: impl IntoIterator<Item = InputEvent>) -> Self {
            Self {
                current: Vec::new(),
                at: 0,
                events: events.into_iter().collect(),
            }
        }
    }

    impl Read for ScriptInput {
        fn read(
            &mut self,
            output: &mut [u8],
        ) -> io::Result<usize> {
            let available = self.fill_buf()?;
            let count = available.len().min(output.len());
            output[..count].copy_from_slice(&available[..count]);
            self.consume(count);
            Ok(count)
        }
    }

    impl BufRead for ScriptInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Ok(&self.current[self.at..])
        }
        fn consume(
            &mut self,
            amount: usize,
        ) {
            self.at = (self.at + amount).min(self.current.len());
        }
    }

    impl ScoreInput for ScriptInput {
        fn wait_readable(
            &mut self,
            _timeout: Duration,
        ) -> io::Result<bool> {
            if !self.fill_buf()?.is_empty() {
                return Ok(true);
            }
            self.current.clear();
            self.at = 0;
            match self.events.pop_front().unwrap_or(InputEvent::Eof) {
                InputEvent::Timeout => Ok(false),
                InputEvent::Data(bytes) => {
                    self.current = bytes;
                    Ok(true)
                },
                InputEvent::Eof => Ok(true),
                InputEvent::Error => Err(io::Error::other("private input error")),
            }
        }
    }

    fn input_line(request: Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(&request).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn run_script(
        worker: &mut ScoreWorker<FakeBackend>,
        events: impl IntoIterator<Item = InputEvent>,
    ) -> Result<Vec<Value>, &'static str> {
        let mut input = ScriptInput::new(events);
        let mut output = Vec::new();
        let result = run_request_loop(worker, &mut input, &mut output);
        let replies = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        result.map(|()| replies)
    }

    #[test]
    fn idle_tick_preserves_consecutive_cache_and_hit_is_bit_exact() {
        let mut worker = worker(true, false);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let line = input_line(standard());
        let replies = run_script(
            &mut worker,
            [InputEvent::Data(line.clone()), InputEvent::Timeout, InputEvent::Data(line), InputEvent::Eof],
        )
        .unwrap();
        assert_eq!(worker.backend.idle_calls, 1);
        assert_eq!(worker.backend.calls, 1);
        assert_eq!(replies[0]["cache_hit"], false);
        assert_eq!(replies[1]["cache_hit"], true);
        for field in ["logits", "log_probabilities", "candidate_probabilities"] {
            let fresh = replies[0][field].as_array().unwrap();
            let hit = replies[1][field].as_array().unwrap();
            assert_eq!(fresh.len(), hit.len());
            for (fresh, hit) in fresh.iter().zip(hit) {
                assert_eq!(fresh.as_f64().unwrap().to_bits(), hit.as_f64().unwrap().to_bits());
            }
        }
        assert_eq!(worker.hits, 1);
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
    }

    #[test]
    fn buffered_request_bypasses_idle_wait_and_eof_clears_cache() {
        let mut worker = worker(true, false);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let mut both = input_line(standard());
        both.extend(input_line(standard()));
        let replies = run_script(&mut worker, [InputEvent::Data(both), InputEvent::Eof]).unwrap();
        assert_eq!(worker.backend.idle_calls, 0);
        assert_eq!(replies[1]["cache_hit"], true);
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
    }

    #[test]
    fn idle_maintenance_failure_clears_cache_and_returns_static_fatal_error() {
        let mut worker = worker(true, false);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        worker.backend.fail_idle = true;
        let result = run_script(&mut worker, [InputEvent::Data(input_line(standard())), InputEvent::Timeout]);
        assert_eq!(result, Err("native_operation_failed"));
        assert_eq!(worker.backend.calls, 1);
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
    }

    #[test]
    fn input_wait_failure_clears_cache_and_returns_static_error() {
        let mut worker = worker(true, false);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let result = run_script(&mut worker, [InputEvent::Data(input_line(standard())), InputEvent::Error]);
        assert_eq!(result, Err("protocol_read_failed"));
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
    }

    #[test]
    fn oversized_line_clears_cache_and_returns_existing_protocol_error() {
        let mut worker = worker(true, false);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let mut bytes = vec![b'x'; worker.budget.wire_bytes + 1];
        bytes.push(b'\n');
        let replies = run_script(&mut worker, [InputEvent::Data(bytes)]).unwrap();
        assert_eq!(replies[0]["error"], "request_too_large");
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
    }

    #[test]
    fn disabled_policy_always_scores_and_holds_no_distribution() {
        let mut worker = worker(false, false);
        assert_eq!(worker.cache.policy(), "fresh-state");
        for _ in 0..3 {
            assert_eq!(post(&mut worker, standard())["cache_hit"], false);
        }
        assert_eq!(worker.backend.calls, 3);
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
    }

    #[test]
    fn identical_consecutive_requests_skip_native_execution_and_preserve_every_scalar_bit() {
        let mut worker = worker(true, false);
        let fresh = post(&mut worker, standard());
        let hit = post(&mut worker, standard());
        assert_eq!(worker.backend.calls, 1);
        assert_eq!(fresh["cache_hit"], false);
        assert_eq!(hit["cache_hit"], true);
        for field in ["logits", "log_probabilities", "candidate_probabilities"] {
            for (actual, expected) in hit[field].as_array().unwrap().iter().zip(fresh[field].as_array().unwrap()) {
                assert_eq!(actual.as_f64().unwrap().to_bits(), expected.as_f64().unwrap().to_bits());
            }
        }
        assert_eq!(hit["metrics"]["state_creation_seconds"], 0.0);
        assert_eq!(hit["metrics"]["prefill_seconds"], 0.0);
        assert!(hit["metrics"]["readout_seconds"].as_f64().unwrap() < 0.3);
    }

    #[test]
    fn exact_identity_context_and_single_entry_fence_a_b_a() {
        let mut worker = worker(true, false);
        for (input, context) in [(&[1, 2][..], 8192), (&[2, 1][..], 8192), (&[1, 2][..], 8192), (&[1, 2][..], 8193)] {
            assert_eq!(post(&mut worker, request(input, &[2, 3], context, 1.22))["cache_hit"], false);
        }
        assert_eq!(worker.backend.calls, 4);
    }

    #[test]
    fn memory_admission_reports_recoverable_resource_errors() {
        const MIB: usize = 1024 * 1024;
        let available = 1024 * MIB;
        let budget = stark_resource_budget::inference_memory_budget(available);
        assert_eq!(admit_scoring_memory(None, available), Err("resource_budget_exceeded"));
        assert_eq!(admit_scoring_memory(Some(512 * MIB), available), Ok(()));
        assert_eq!(admit_scoring_memory(Some(budget), available), Ok(()));
        assert_eq!(admit_scoring_memory(Some(budget + 1), available), Err("resource_budget_exceeded"));
        assert_eq!(admit_scoring_memory(Some(1), 0), Err("resource_budget_exceeded"));
        assert_eq!(admit_scoring_memory(Some(usize::MAX), usize::MAX), Err("resource_budget_exceeded"));
    }

    #[test]
    fn scoring_memory_rejection_clears_logits_and_accepts_same_worker_retry() {
        let mut worker = worker(true, false);
        post(&mut worker, standard());
        assert_eq!(post(&mut worker, standard())["cache_hit"], true);
        worker.backend.fail_resource = true;
        let full_context = request(&[2, 1], &[2, 3], 2, 1.0);
        assert_eq!(post(&mut worker, full_context.clone())["error"], "resource_budget_exceeded");
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
        worker.backend.fail_resource = false;
        let retry = post(&mut worker, full_context.clone());
        assert_eq!(retry["input_tokens"], 2);
        assert_eq!(retry["cache_hit"], false);
        assert_eq!(post(&mut worker, full_context)["cache_hit"], true);
        assert_eq!(worker.backend.calls, 3);
    }

    #[test]
    fn cached_score_does_not_require_memory_for_a_new_native_state() {
        let mut worker = worker(true, false);
        let fresh = post(&mut worker, standard());
        worker.backend.fail_resource = true;
        let cached = post(&mut worker, standard());
        assert_eq!(cached["cache_hit"], true);
        assert_eq!(worker.backend.calls, 1);
        assert_eq!(cached["logits"], fresh["logits"]);
        // A changed input still needs a fresh native allocation and must fail.
        let changed = request(&[2, 1], &[2, 3], 8192, 1.0);
        assert_eq!(post(&mut worker, changed)["error"], "resource_budget_exceeded");
        assert_eq!(worker.backend.calls, 2);
    }

    #[test]
    fn scoring_reclaims_only_on_pressure_and_rechecks_observed_headroom() {
        use std::cell::Cell;

        const MIB: usize = 1024 * 1024;
        let reclaimed = Cell::new(false);
        assert_eq!(
            admit_scoring_memory_with_reclaim(
                Some(512 * MIB),
                || if reclaimed.get() {
                    1024 * MIB
                } else {
                    512 * MIB
                },
                || {
                    reclaimed.set(true);
                    Ok(512 * MIB)
                },
            ),
            Ok(())
        );
        assert!(reclaimed.get());
        assert_eq!(
            admit_scoring_memory_with_reclaim(Some(512 * MIB), || 512 * MIB, || Ok(usize::MAX),),
            Err("resource_budget_exceeded")
        );
        assert_eq!(admit_scoring_memory_with_reclaim(Some(1), || 1024 * MIB, || panic!("no pressure"),), Ok(()));
        assert_eq!(
            admit_scoring_memory_with_reclaim(None, || panic!("invalid estimate"), || panic!("invalid estimate"),),
            Err("resource_budget_exceeded")
        );
        assert_eq!(
            admit_scoring_memory_with_reclaim(Some(512 * MIB), || 512 * MIB, || Err("native_scoring_failed"),),
            Err("native_scoring_failed")
        );
    }

    #[test]
    fn model_context_capacity_controls_large_requests_and_failures_clear_cache() {
        let mut worker = worker(true, false);
        let input = vec![1; 16_385];
        let large = request(&input, &[2, 3], 262_144, 1.0);
        let fresh = post(&mut worker, large.clone());
        assert_eq!(fresh["input_tokens"], 16_385);
        assert_eq!(fresh["truncated"], false);
        assert_eq!(post(&mut worker, large.clone())["cache_hit"], true);
        assert_eq!(worker.backend.calls, 1);
        assert_eq!(post(&mut worker, request(&input, &[2, 3], 262_145, 1.0))["error"], "context_overflow");
        assert_eq!(worker.backend.calls, 1);
        assert_eq!(post(&mut worker, large)["cache_hit"], false);
        worker.backend.max_context = None;
        assert_eq!(post(&mut worker, request(&input, &[2, 3], u32::MAX, 1.0))["truncated"], false);
    }

    #[test]
    fn changed_candidates_order_and_temperature_recompute_readout_from_full_vocabulary() {
        let mut worker = worker(true, false);
        post(&mut worker, standard());
        let hit = post(&mut worker, request(&[1, 2], &[5, 2, 0], 8192, 0.93));
        assert_eq!(worker.backend.calls, 1);
        assert_eq!(hit["cache_hit"], true);
        // Compare complete protocol round trips. serde_json's default parser
        // can round a decimal differently from its in-memory f64 input.
        let control = post(&mut self::worker(false, false), request(&[1, 2], &[5, 2, 0], 8192, 0.93));
        for field in ["logits", "log_probabilities", "candidate_probabilities"] {
            for (actual, expected) in hit[field].as_array().unwrap().iter().zip(control[field].as_array().unwrap()) {
                assert_eq!(actual.as_f64().unwrap().to_bits(), expected.as_f64().unwrap().to_bits());
            }
        }
    }

    #[test]
    fn digest_probe_uses_same_hit_path_and_preserves_all_f32_bits() {
        let mut worker = worker(true, true);
        let fresh = post(&mut worker, standard());
        let hit = post(&mut worker, standard());
        let mut hash = Sha256::new();
        for value in [-0.0_f32, -1.0, 0.0, 1.0, 2.0, 3.0] {
            hash.update(value.to_bits().to_le_bytes());
        }
        assert_eq!(hit["vocab_logits_sha256"], format!("{:x}", hash.finalize()));
        assert_eq!(fresh["vocab_logits_sha256"], hit["vocab_logits_sha256"]);
        assert_eq!(hit["cache_hit"], true);
        assert_eq!(worker.backend.calls, 1);
    }

    #[test]
    fn successful_tokenize_helpers_preserve_scoring_entry_failed_helpers_clear_it() {
        let mut worker = worker(true, false);
        post(&mut worker, standard());
        post(&mut worker, json!({"operation":"tokenize","text":"synthetic helper"}));
        assert_eq!(post(&mut worker, standard())["cache_hit"], true);
        worker.backend.fail_tokenize = true;
        assert_eq!(
            post(&mut worker, json!({"operation":"tokenize","text":"synthetic helper"}))["error"],
            "tokenization_failed"
        );
        assert_eq!(post(&mut worker, standard())["cache_hit"], false);
    }

    #[test]
    fn every_parse_and_validation_failure_clears_previous_success() {
        let mut worker = worker(true, false);
        for invalid in [
            b"{".to_vec(),
            serde_json::to_vec(&request(&[], &[2, 3], 8192, 1.22)).unwrap(),
            serde_json::to_vec(&request(&[1, 2], &[2, 2], 8192, 1.22)).unwrap(),
            serde_json::to_vec(&request(&[1, 2], &[2, 3], 1, 1.22)).unwrap(),
            serde_json::to_vec(&json!({"operation":"tokenize","text":""})).unwrap(),
        ] {
            post(&mut worker, standard());
            let mut output = Vec::new();
            worker.process_line(&invalid, &mut output).unwrap();
            assert!(serde_json::from_slice::<Value>(&output).unwrap().get("error").is_some());
            assert_eq!(post(&mut worker, standard())["cache_hit"], false);
        }
    }

    #[test]
    fn native_and_readout_faults_fence_worker_and_leave_no_reusable_entry() {
        for invalid_output in [false, true] {
            let mut worker = worker(true, false);
            post(&mut worker, standard());
            worker.backend.fail_score = !invalid_output;
            worker.backend.invalid_output = invalid_output;
            let mut output = Vec::new();
            let bytes = serde_json::to_vec(&request(&[2, 1], &[2, 3], 8192, 1.22)).unwrap();
            assert_eq!(worker.process_line(&bytes, &mut output), Err("native_operation_failed"));
            assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
            worker.backend.fail_score = false;
            worker.backend.invalid_output = false;
            assert_eq!(post(&mut worker, standard())["cache_hit"], false);
        }
    }

    #[test]
    fn mma_probe_failure_is_fatal_and_clears_the_consecutive_cache() {
        let mut worker = worker(true, false);
        post(&mut worker, standard());
        worker.backend.fail_probe = true;
        let request = serde_json::to_vec(&request(&[2, 1], &[2, 3], 8192, 1.22)).unwrap();
        assert_eq!(worker.process_line(&request, &mut Vec::new()), Err("native_operation_failed"));
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
        // An earlier successful distribution cannot bypass the sticky probe
        // error after the failed request cleared the consecutive cache.
        assert_eq!(
            worker.process_line(&serde_json::to_vec(&standard()).unwrap(), &mut Vec::new()),
            Err("native_operation_failed")
        );
    }

    #[test]
    fn mma_probe_file_is_fresh_private_and_contains_only_aggregate_scalars() {
        use std::{
            os::unix::fs::PermissionsExt,
            sync::atomic::{AtomicU64, Ordering},
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-score-mma-probe-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut probe = MmaScoreProbe::open(&path).unwrap();
        probe
            .record(MmaScoreProbeRow {
                success: true,
                encoded_dispatch_calls: 2,
                encoded_weight_bytes: 4096,
                encoded_q5_dispatch_calls: 1,
                encoded_q5_weight_bytes: 1024,
            })
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        let before = std::fs::read(&path).unwrap();
        assert!(MmaScoreProbe::open(&path).is_err(), "probe replaced existing evidence");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let row: Value = serde_json::from_slice(&before).unwrap();
        assert_eq!(
            row,
            json!({"success":true,"encoded_dispatch_calls":2,"encoded_weight_bytes":4096,
            "encoded_q5_dispatch_calls":1,"encoded_q5_weight_bytes":1024})
        );
        drop(probe);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mma_probe_write_and_flush_failures_remain_sticky() {
        for fail_flush in [false, true] {
            let mut probe = MmaScoreProbe {
                output: FailingWriter {
                    fail_flush,
                },
                failed: false,
                previous_profile: None,
            };
            let row = || MmaScoreProbeRow {
                success: true,
                encoded_dispatch_calls: 2,
                encoded_weight_bytes: 4096,
                encoded_q5_dispatch_calls: 1,
                encoded_q5_weight_bytes: 1024,
            };
            assert_eq!(probe.record(row()), Err("mma_score_probe_failed"));
            assert!(probe.failed);
            assert_eq!(probe.begin(), Err("mma_score_probe_failed"));
            assert_eq!(probe.record(row()), Err("mma_score_probe_failed"));
        }
    }

    #[test]
    fn mma_probe_counters_reset_per_uncached_request_and_filter_other_categories() {
        use stark_engine::engine::hybrid::profile;
        let previous = profile::enabled();
        let mut probe = MmaScoreProbe {
            output: Vec::new(),
            failed: false,
            previous_profile: None,
        };
        probe.begin().unwrap();
        profile::record("native.glm.prefill.q6_mma", 123.0, 4096);
        profile::record("native.glm.prefill.q5_mma", 234.0, 2048);
        profile::record("native.glm.prefill.q5_mma", 345.0, 1024);
        profile::record("synthetic.unrelated", 321.0, 999);
        probe.complete(true).unwrap();
        assert_eq!(profile::enabled(), previous);
        probe.begin().unwrap();
        probe.complete(false).unwrap();
        assert_eq!(profile::enabled(), previous);
        let rows = std::str::from_utf8(&probe.output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![
                json!({"success":true,"encoded_dispatch_calls":3,"encoded_weight_bytes":7168,
                    "encoded_q5_dispatch_calls":2,"encoded_q5_weight_bytes":3072}),
                json!({"success":false,"encoded_dispatch_calls":0,"encoded_weight_bytes":0,
                    "encoded_q5_dispatch_calls":0,"encoded_q5_weight_bytes":0})
            ]
        );
    }

    struct FailingWriter {
        fail_flush: bool,
    }
    impl Write for FailingWriter {
        fn write(
            &mut self,
            bytes: &[u8],
        ) -> io::Result<usize> {
            if self.fail_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("synthetic write failure"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("synthetic flush failure"))
        }
    }

    #[test]
    fn publication_waits_for_serialization_write_and_flush_on_miss_and_hit() {
        for fail_flush in [false, true] {
            for hit in [false, true] {
                let mut worker = worker(true, false);
                if hit {
                    post(&mut worker, standard());
                }
                assert_eq!(
                    worker.process_line(
                        &serde_json::to_vec(&standard()).unwrap(),
                        &mut FailingWriter {
                            fail_flush
                        }
                    ),
                    Err("protocol_write_failed")
                );
                assert_eq!(post(&mut worker, standard())["cache_hit"], false);
            }
        }
    }

    #[test]
    fn a_pending_reply_never_establishes_a_cache_hit_before_publication() {
        let mut worker = worker(true, false);
        let request: ScoreRequest = serde_json::from_value(standard()).unwrap();
        let pending = worker.handle(request).unwrap();
        assert!(pending.next_entry.is_some());
        assert!(worker.cache.take_matching(&[1, 2], 8192).is_none());
        drop(pending);
        assert_eq!(post(&mut worker, standard())["cache_hit"], false);
        assert_eq!(worker.backend.calls, 2);
    }

    #[test]
    fn reset_clears_only_distribution_and_keeps_cumulative_accurate_counters() {
        let mut worker = worker(true, false);
        assert_eq!(post(&mut worker, standard())["cache"]["native_score_calls"], 1);
        let hit = post(&mut worker, standard());
        assert_eq!(hit["cache"]["native_score_calls"], 1);
        assert_eq!(hit["cache"]["hits"], 1);
        let reset = post(&mut worker, json!({"operation":"reset_cache"}));
        assert_eq!(reset["cache_reset"], true);
        assert_eq!(reset["cache"]["hits"], 1);
        assert_eq!(reset["cache"]["native_score_calls"], 1);
        let fresh = post(&mut worker, standard());
        assert_eq!(fresh["cache_hit"], false);
        assert_eq!(fresh["cache"]["native_score_calls"], 2);
    }

    #[test]
    fn failed_reset_flush_leaves_cache_cleared_and_failed_hit_never_counts_success() {
        let mut worker = worker(true, false);
        post(&mut worker, standard());
        let bytes = serde_json::to_vec(&json!({"operation":"reset_cache"})).unwrap();
        assert_eq!(
            worker.process_line(
                &bytes,
                &mut FailingWriter {
                    fail_flush: true
                }
            ),
            Err("protocol_write_failed")
        );
        let fresh = post(&mut worker, standard());
        assert_eq!(fresh["cache_hit"], false);
        assert_eq!(fresh["cache"]["hits"], 0);
        assert_eq!(
            worker.process_line(
                &serde_json::to_vec(&standard()).unwrap(),
                &mut FailingWriter {
                    fail_flush: false
                }
            ),
            Err("protocol_write_failed")
        );
        assert_eq!(worker.hits, 0);
    }

    #[test]
    fn calibration_matches_official_probability_power_formula() {
        let logits = [-3.0, 1.25, -0.5, 2.0];
        let temperature = 1.22;
        let uncalibrated = calibrated_probabilities(&logits, 1.0).unwrap();
        let weights: Vec<f64> = uncalibrated.iter().map(|p| p.powf(1.0 / temperature)).collect();
        let sum: f64 = weights.iter().sum();
        let calibrated = calibrated_probabilities(&logits, temperature).unwrap();
        for (actual, weight) in calibrated.iter().zip(weights) {
            assert!((actual - weight / sum).abs() < 1e-14);
        }
    }

    #[test]
    fn calibration_is_shift_invariant_and_stable_at_large_logits() {
        let expected = calibrated_probabilities(&[-2.0, 0.0, 1.0], 0.93).unwrap();
        let actual = calibrated_probabilities(&[9998.0, 10000.0, 10001.0], 0.93).unwrap();
        assert_eq!(expected, actual);
        assert!((actual.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert_eq!(calibrated_probabilities(&[0.0, 0.0], 1.22).unwrap(), vec![0.5, 0.5]);
        assert_eq!(calibrated_probabilities(&[f64::MAX, -f64::MAX], 1.22).unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn readout_uses_full_vocabulary_log_partition_and_ordered_candidates() {
        let vocabulary = [-1.0, 0.0, 1.0, 2.0, -3.0];
        let (logits, log_probs, probs) = candidate_readout(&vocabulary, &[3, 1], 1.22).unwrap();
        assert_eq!(logits, [2.0, 0.0]);
        let full_mass: f64 = vocabulary.iter().map(|value| f64::from(*value).exp()).sum();
        assert!((log_probs[0] - (2.0_f64.exp() / full_mass).ln()).abs() < 1e-14);
        assert!((log_probs[1] - (1.0 / full_mass).ln()).abs() < 1e-14);
        let from_log_probs = calibrated_probabilities(&log_probs, 1.22).unwrap();
        for (direct, normalized) in probs.iter().zip(from_log_probs) {
            assert!((direct - normalized).abs() < 1e-14);
        }
    }

    #[test]
    fn invalid_math_and_tokens_are_rejected() {
        assert!(calibrated_probabilities(&[], 1.22).is_err());
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(calibrated_probabilities(&[0.0, 1.0], temperature).is_err());
            assert!(validate_score(&[1], &[2, 3], 8192, temperature, 100).is_err());
        }
        assert!(calibrated_probabilities(&[f64::NAN, 0.0], 1.22).is_err());
        assert!(candidate_readout(&[f32::INFINITY, 0.0], &[0, 1], 1.22).is_err());
        assert!(candidate_readout(&[0.0, 0.0], &[0, 2], 1.22).is_err());
        assert!(validate_score(&[], &[2, 3], 8192, 1.22, 100).is_err());
        assert!(validate_score(&[1], &[2, 2], 8192, 1.22, 100).is_err());
        assert!(validate_score(&[100], &[2, 3], 8192, 1.22, 100).is_err());
        assert!(validate_score(&[1], &[2], 8192, 1.22, 100).is_err());
        assert!(validate_score(&[1], &(0..17).collect::<Vec<_>>(), 8192, 1.22, 100).is_err());
        assert_eq!(validate_score(&[1, 2], &[2, 3], 1, 1.22, 100), Err("context_overflow"));
        assert_eq!(validate_score(&[1, 2], &[2, 3], 2, 1.22, 100), Ok(()));
        assert_eq!(validate_score(&[1], &[2, 3], 16385, 1.22, 100), Ok(()));
    }

    #[test]
    fn protocol_rejects_unknown_fields_and_wrong_token_types() {
        assert!(
            serde_json::from_value::<ScoreRequest>(json!({"operation":"score","input_token_ids":[1],
            "candidate_token_ids":[2,3],"context_tokens":8192,"temperature":1.22,"cache_prompt":true}))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ScoreRequest>(json!({"operation":"score","input_token_ids":[-1],
            "candidate_token_ids":[2,3],"context_tokens":8192,"temperature":1.22}))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ScoreRequest>(json!({"operation":"tokenize","text":"test","private":true}))
                .is_err()
        );
        assert!(matches!(
            serde_json::from_value::<ScoreRequest>(json!({"operation":"tokenize","text":"test"})).unwrap(),
            ScoreRequest::Tokenize {
                add_special_tokens: false,
                ..
            }
        ));
    }

    #[test]
    fn token_identity_matches_little_endian_u32_protocol() {
        assert_eq!(token_hash(&[1, 2]), format!("{:x}", Sha256::digest([1, 0, 0, 0, 2, 0, 0, 0])));
        assert_ne!(token_hash(&[1, 2]), token_hash(&[2, 1]));
    }
}
