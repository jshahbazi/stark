// SPDX-License-Identifier: Apache-2.0
//! Original Inkling 2 Mini target through the serialized JSONL worker.

mod thinking_budget;

use std::{
    path::Path,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        hybrid::activity::InferenceActivity,
        inkling_slice::{
            DepthPlanner, INKLING_IMAGE_TOKEN as IMAGE_TOKEN, ImageEmbeddings, MAX_VERIFY_ROWS, Session, Snapshot,
            VOCABULARY as VOCAB, Verification,
        },
    },
};
use tokenizers::Tokenizer;

use self::thinking_budget::{SamplingControl, THINKING, ThinkingBudget};
use crate::{
    cancel::CancelSignal,
    generation::{
        FinishReason, GeneratedTokens, GenerationBackend, GenerationOptions, GenerationProgress, ImageArtifactIdentity,
        MtpCapabilities, PreparedImageMetadata, ProgressCallback, Sampling, TextCallback, admit_request_memory,
        collect_tokens_controlled,
    },
    hybrid::{ProbabilityMapper, sample_certified_prefix, sample_with_mapper},
    inkling_artifact::{self, Artifact},
    prefix_cache::{self, BoundaryEntry},
    text_stop::StopFilter,
};

const EOS: &[u32] = &[200_006];
const IMAGE_CONTENT: u32 = 200_005;
const END_MESSAGE: u32 = 200_010;
const CHUNK: usize = prefix_cache::PREFILL_CHUNK_TOKENS;
/// Mechanical minimum of the source schedule. Below five tokens the bulk is
/// empty and the tail shrinks (`saturating_sub`); bulk commands under
/// `Session::SOURCE_MIN_BULK_ROWS` run the lane or serial kernels inside the
/// source scope, and every longer command runs the packed source kernels.
const SOURCE_PREFILL_MIN_INPUT: usize = 1;
const SOURCE_PROMPT_SUFFIX: usize = 4;
/// Default proposals through the multi-row source verifier (measured best).
const GREEDY_DEFAULT_DRAFTS: u32 = 3;
const SAMPLED_DEFAULT_DRAFTS: u32 = 2;
const INITIAL_CONTEXT: usize = 4352;
const SHORT_MTP_VERIFICATION: &str = "inkling-dspark-short-target-draw-v1";

#[derive(Default, serde::Serialize)]
struct DraftStats {
    proposed: usize,
    accepted: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    speculative: Option<SpeculationStats>,
}

#[derive(Default, serde::Serialize)]
struct SpeculationStats {
    rounds: [usize; MAX_VERIFY_ROWS],
    #[serde(skip_serializing_if = "Option::is_none")]
    overlapped_rounds: Option<usize>,
    draft_seconds: f64,
    verify_seconds: f64,
    finish_seconds: f64,
    /// Wall seconds of complete drafting rounds (round start to commit end)
    /// and their count, by accepted draft count.
    round_seconds_by_accepted: [f64; MAX_VERIFY_ROWS],
    rounds_by_accepted: [usize; MAX_VERIFY_ROWS],
    /// Host time inside rounds between the verifier result and the commit
    /// (sampling and the consumer's per-token work).
    in_round_host_seconds: f64,
    /// Time from a commit to the start of the next round.
    between_rounds_seconds: f64,
}

struct PendingDraft<'a> {
    transaction: Verification<'a>,
    proposals: Vec<u32>,
    consumed: usize,
}

pub(crate) struct InklingBackend {
    tokenizer: Tokenizer,
    config_sha256: String,
    /// Original or affine8 artifact, as verified at load.
    layout: inkling_artifact::Layout,
    target: Session,
    cancellation: Option<CancelSignal>,
    sampler: ProbabilityMapper,
    retained: Option<BoundaryEntry<((bool, bool), Snapshot)>>,
    pending: Option<BoundaryEntry<((bool, bool), Snapshot)>>,
    source_verifier: bool,
    /// The packed source prefill is loaded, so reuse boundaries sit on its grid.
    source_grid: bool,
    draft_stats: DraftStats,
    runtime_identity: stark_engine::engine::MetalDeviceIdentity,
}

/// Default draft count for a request that names no MTP setting.
fn default_drafts_for(sampling: &Sampling) -> u32 {
    if matches!(sampling, Sampling::Greedy {}) {
        GREEDY_DEFAULT_DRAFTS
    } else {
        SAMPLED_DEFAULT_DRAFTS
    }
}

/// Source prefill command width chosen at load: the widest of
/// `Session::SOURCE_COMMAND_WIDTHS` the host admits. Default prefill output
/// does not depend on the width and prefix snapshots stay 2,048-aligned, so
/// this only trades workspace for fewer, larger commands. The base width keeps
/// its original admission rule; a wider one must also leave room for the
/// verifier, the proposal head and a maximum-context request, so it never
/// costs a request that the base width would have served.
fn source_command_rows(
    fixed: usize,
    available: usize,
) -> Option<usize> {
    let limit = Session::source_context_limit(true);
    let headroom = Session::context_memory_bytes(limit, true).zip(Session::dspark_memory_bytes(limit, true)).map(
        |(target, draft)| {
            target
                + draft
                + Session::source_global_extra_bytes(true, limit)
                + Session::SOURCE_VERIFIER_RESERVE_BYTES
                + Session::DSPARK_INT8_HEAD_BYTES
        },
    );
    Session::SOURCE_COMMAND_WIDTHS.iter().rev().copied().find(|&rows| {
        let reserve = Session::batched_prefill_memory_bytes_at(true, rows);
        let needed = if rows == Session::SOURCE_PREFILL_ROWS {
            Some(fixed + reserve)
        } else {
            headroom.map(|headroom| fixed + reserve + headroom)
        };
        admit_request_memory(needed, available).is_ok()
    })
}

/// The worker's reuse grid and the prompt tail a request always prefills, as
/// readiness advertises them. With the packed source prefill loaded that is
/// its 2,048-row bulk alignment and four-token suffix.
fn reuse_grid(source_grid: bool) -> (usize, usize) {
    if source_grid {
        (Session::SOURCE_PREFILL_ROWS, SOURCE_PROMPT_SUFFIX)
    } else {
        (CHUNK, 2)
    }
}

/// Prefix-snapshot boundary on the worker's reuse grid. Every request of a
/// worker uses the one grid, whichever prefill path it takes, so reuse counts
/// are multiples of the advertised chunk; a short input has no boundary.
fn prefix_boundary(
    input_len: usize,
    scoped: bool,
    source_grid: bool,
) -> usize {
    let (chunk, tail) = reuse_grid(source_grid);
    if scoped {
        input_len.saturating_sub(tail) / chunk * chunk
    } else {
        0
    }
}

/// Where the remaining source bulk of a request splits before its final
/// commands: the snapshot boundary a scoped request would use, when the
/// request has not already been split there or beyond it.
fn unscoped_source_split(
    input_len: usize,
    start: usize,
    source_prefill: bool,
) -> usize {
    if source_prefill {
        prefix_boundary(input_len, true, true).max(start)
    } else {
        start
    }
}

pub(crate) struct PreparedInklingImage {
    image: ImageEmbeddings,
    input: Vec<u32>,
    slot: usize,
}

fn check_cancelled(
    signal: &mut Option<CancelSignal>,
    request: u64,
) -> Result<bool, &'static str> {
    match signal {
        Some(signal) => signal.check(request),
        None => Ok(false),
    }
}

fn require_active(
    signal: &mut Option<CancelSignal>,
    request: u64,
) -> Result<(), &'static str> {
    if check_cancelled(signal, request)? {
        Err("generation_cancelled")
    } else {
        Ok(())
    }
}

fn admit(bytes: Option<usize>) -> Result<(), &'static str> {
    admit_request_memory(bytes, stark_resource_budget::available_memory_bytes().unwrap_or(0))
}

struct PrefillProgress<'state, 'callback, 'environment> {
    cancellation: &'state mut Option<CancelSignal>,
    request: u64,
    started: Instant,
    last_progress: Duration,
    batch_rows: usize,
    callback: &'state mut Option<&'callback mut ProgressCallback<'environment>>,
}

impl PrefillProgress<'_, '_, '_> {
    fn run(
        &mut self,
        rows: usize,
        mut encode: impl FnMut(std::ops::Range<usize>) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        for start in (0..rows).step_by(self.batch_rows) {
            require_active(self.cancellation, self.request)?;
            encode(start..rows.min(start + self.batch_rows)).map_err(|_| "native_generation_failed")?;
            let now = self.started.elapsed();
            if now.saturating_sub(self.last_progress) >= Duration::from_secs(1) {
                if let Some(callback) = self.callback.as_deref_mut() {
                    callback(GenerationProgress {
                        sampled_tokens: 0,
                        elapsed_seconds: now.as_secs_f64(),
                        prefill_seconds: None,
                    })?;
                }
                self.last_progress = now;
            }
        }
        require_active(self.cancellation, self.request)
    }

    fn text(
        &mut self,
        target: &mut Session,
        tokens: &[u32],
    ) -> Result<(), &'static str> {
        self.run(tokens.len(), |rows| target.ingest_many(&tokens[rows]))
    }
}

impl InklingBackend {
    pub(crate) fn load(
        root: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        let runtime_identity = Engine::<Metal>::new().map_err(|_| "inkling_load_failed")?.metal_device_identity();
        let m5_max = runtime_identity.device_name == "Apple M5 Max" && runtime_identity.supports_mxu;
        let available = stark_resource_budget::available_memory_bytes().unwrap_or(0);
        let fixed = inkling_artifact::TARGET_BYTES
            + inkling_artifact::DRAFT_BYTES
            + 1024 * 1024 * 1024
            + inkling_artifact::Layout::of(root).extra_resident_bytes();
        let source_rows = if m5_max {
            source_command_rows(fixed, available)
        } else {
            None
        };
        let source_allowed = source_rows.is_some();
        let command_rows = source_rows.unwrap_or(Session::SOURCE_PREFILL_ROWS);
        let prefill_reserve = if m5_max {
            Session::batched_prefill_memory_bytes_at(source_allowed, command_rows)
        } else {
            128 * 1024 * 1024
        };
        admit_request_memory(Some(fixed + prefill_reserve), available)?;
        let artifact = Artifact::load(root)?;
        let mut target =
            Session::load_file(artifact.target.descriptor()?, INITIAL_CONTEXT).map_err(|_| "inkling_load_failed")?;
        artifact.target.verify_unchanged()?;
        if m5_max {
            target.load_dspark_file(artifact.draft.descriptor()?).map_err(|_| "inkling_draft_load_failed")?;
            artifact.draft.verify_unchanged()?;
        }
        if target.supports_batched_prefill() {
            target
                .set_batched_prefill_width(true, source_allowed, command_rows)
                .map_err(|_| "inkling_prefill_load_failed")?;
        }
        let source_verifier = source_allowed
            && admit_request_memory(Some(fixed + prefill_reserve + Session::SOURCE_VERIFIER_RESERVE_BYTES), available)
                .is_ok()
            && target.set_source_verifier(true).is_ok();
        if source_verifier {
            // The INT8 proposal head stays resident once prepared; `context_bytes`
            // counts it through `dspark_int8_head_resident_bytes`.
            if admit_request_memory(
                Some(
                    fixed + prefill_reserve + Session::SOURCE_VERIFIER_RESERVE_BYTES + Session::DSPARK_INT8_HEAD_BYTES,
                ),
                available,
            )
            .is_ok()
            {
                target.prepare_dspark_int8_head().map_err(|_| "inkling_draft_load_failed")?;
            }
            target.set_source_verifier(false).map_err(|_| "inkling_prefill_load_failed")?;
        }
        let source_grid = target.supports_source_text_prefill(INITIAL_CONTEXT);
        Ok(Self {
            tokenizer: artifact.tokenizer,
            config_sha256: artifact.config_sha256,
            layout: artifact.layout,
            target,
            cancellation,
            sampler: ProbabilityMapper::default(),
            retained: None,
            pending: None,
            source_verifier,
            source_grid,
            draft_stats: DraftStats::default(),
            runtime_identity,
        })
    }

    fn source_text_prefill(
        &self,
        input_len: usize,
        options: &GenerationOptions,
    ) -> bool {
        input_len >= SOURCE_PREFILL_MIN_INPUT
            && (options.mtp_drafts == 0 || self.source_verifier)
            && self.target.supports_source_text_prefill(options.context_tokens as usize)
    }

    fn boundary(
        &self,
        input_len: usize,
        options: &GenerationOptions,
    ) -> usize {
        prefix_boundary(input_len, options.cache_scope.is_some(), self.source_grid)
    }

    fn generate_inner(
        &mut self,
        input: &[u32],
        image: Option<(&ImageEmbeddings, usize)>,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        mut on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let expanded =
            input.len().checked_add(image.map_or(0, |(image, _)| image.token_count() - 1)).ok_or("context_overflow")?;
        if input.is_empty() || expanded >= options.context_tokens as usize {
            return Err("context_overflow");
        }
        self.draft_stats = DraftStats::default();
        let use_draft = options.mtp_drafts > 0;
        let source_prefill = image.is_none() && self.source_text_prefill(input.len(), options);
        let source_verify = self.source_verifier && use_draft && source_prefill;
        let supported_drafts = if source_verify {
            MAX_VERIFY_ROWS - 1
        } else {
            1
        };
        if use_draft && (options.mtp_drafts as usize > supported_drafts || image.is_some() || !self.target.has_dspark())
        {
            return Err("mtp_not_supported");
        }
        let verify_depth = if source_verify {
            options.mtp_drafts as usize
        } else {
            0
        };
        let boundary = if image.is_none() {
            self.boundary(input.len(), options)
        } else {
            0
        };
        let reused = if image.is_none() {
            self.prefix_cache_plan(input, options)
        } else {
            0
        };
        let retained = self.retained.take();
        self.pending = None;
        admit(self.context_bytes(options.context_tokens as usize, boundary > reused, source_prefill))?;
        let _activity = InferenceActivity::begin().map_err(|_| "inference_activity_unavailable")?;
        let started = Instant::now();
        require_active(&mut self.cancellation, options.request_id)?;
        // Select the scope first so `set_context` sizes the source global
        // workspace only for a request that was admitted with it.
        self.target.reset();
        self.target.set_text_prefill_scope(source_prefill);
        self.target.set_context(options.context_tokens as usize).map_err(|_| "native_generation_failed")?;
        self.target.set_source_verifier(source_verify).map_err(|_| "mtp_not_supported")?;
        self.target.set_dspark_enabled(use_draft).map_err(|_| "mtp_not_supported")?;
        let mut retained = retained.filter(|_| reused > 0);
        if let Some(entry) = retained.as_ref() {
            self.target.restore(&entry.state.1).map_err(|_| "native_generation_failed")?;
        }
        let mut prefill = PrefillProgress {
            cancellation: &mut self.cancellation,
            request: options.request_id,
            started,
            last_progress: Duration::ZERO,
            batch_rows: if image.is_some() {
                self.target.image_prefill_batch_rows()
            } else {
                self.target.prefill_batch_rows()
            },
            callback: &mut on_progress,
        };
        let last = input.len() - 1;
        let mut snapshot = None;
        let logits = if let Some((image, slot)) = image {
            prefill.text(&mut self.target, &input[..slot])?;
            if slot == last {
                prefill.run(image.token_count() - 1, |rows| self.target.ingest_image_rows(image, rows))?;
                self.target.step_image_row(image, image.token_count() - 1)
            } else {
                prefill.run(image.token_count(), |rows| self.target.ingest_image_rows(image, rows))?;
                prefill.text(&mut self.target, &input[slot + 1..last])?;
                self.target.step(input[last])
            }
        } else {
            if boundary > reused {
                drop(retained.take());
                prefill.text(&mut self.target, &input[reused..boundary])?;
                snapshot = Some(self.target.snapshot().map_err(|_| "native_generation_failed")?);
            } else if boundary > 0 && boundary == reused {
                snapshot = retained.take().map(|entry| entry.state.1);
            }
            let bulk_end = if source_prefill {
                input.len().saturating_sub(SOURCE_PROMPT_SUFFIX)
            } else {
                last
            };
            let mut start = boundary.max(reused);
            // A request without a cache scope takes no snapshot, but its bulk
            // still splits where a scoped request's would. The rows after that
            // split then run in a command of the same width class either way,
            // so the output does not depend on whether the prefix cache is on,
            // at any source command width.
            let split = unscoped_source_split(input.len(), start, source_prefill);
            if split > start {
                prefill.text(&mut self.target, &input[start..split])?;
                start = split;
            }
            prefill.text(&mut self.target, &input[start..bulk_end])?;
            if source_prefill {
                prefill.text(&mut self.target, &input[bulk_end..last])?;
            }
            self.target.step(input[last])
        }
        .map_err(|_| "native_generation_failed")?;
        let result = collect_output(
            &mut self.target,
            &self.tokenizer,
            &mut self.sampler,
            &mut self.cancellation,
            &mut self.draft_stats,
            verify_depth,
            logits,
            expanded,
            input.last().copied() == Some(THINKING),
            options,
            started,
            on_text,
            on_progress,
        );
        let mut result = result?;
        result.reused_input_tokens = reused;
        if let (Some(state), Some(scope)) = (snapshot, options.cache_scope) {
            self.pending = Some(BoundaryEntry::new(
                ((use_draft, source_prefill), state),
                scope,
                options.context_tokens,
                &input[..boundary],
            ));
        }
        Ok(result)
    }

    /// Automatic planning covers text requests, which take the source path
    /// whenever the loaded worker supports it at that context.
    fn plans_source(
        &self,
        context: usize,
    ) -> bool {
        self.target.supports_source_text_prefill(context)
    }

    fn context_bytes(
        &self,
        context: usize,
        retain: bool,
        source: bool,
    ) -> Option<usize> {
        // Only source-path requests grow the global workspace beyond the
        // 4,352 keys that the load-time reserve covers.
        let workspace = if source {
            self.target.source_context_workspace_bytes(context)
        } else {
            0
        };
        let target = Session::context_memory_bytes(context, retain)?.checked_add(workspace)?;
        if self.target.has_dspark() {
            target
                .checked_add(Session::dspark_memory_bytes(context, retain)?)?
                .checked_add(self.target.dspark_int8_head_resident_bytes())
        } else {
            Some(target)
        }
    }
}

fn sample_with_thinking_budget(
    scores: &[f32],
    control: SamplingControl,
    sampling: &Sampling,
    rng: &mut u64,
    mapper: &mut ProbabilityMapper,
    scratch: &mut Vec<f32>,
) -> Result<u32, &'static str> {
    let scores = match control {
        SamplingControl::Natural => scores,
        SamplingControl::Close => return Ok(END_MESSAGE),
        SamplingControl::BiasClose(bias) => {
            scratch.clear();
            scratch.extend_from_slice(scores);
            scratch[END_MESSAGE as usize] += bias;
            scratch
        },
        SamplingControl::BlockThinking => {
            // The normal sampler rejects nonfinite scores. Remove the opening
            // marker from its candidates and preserve the other IDs' ordering.
            if !scores[THINKING as usize].is_finite() {
                return Err("native_generation_failed");
            }
            scratch.clear();
            scratch.extend_from_slice(&scores[..THINKING as usize]);
            scratch.extend_from_slice(&scores[THINKING as usize + 1..]);
            scratch
        },
    };
    let token = match sample_certified_prefix(scores, sampling, rng) {
        Some(token) => token,
        None => sample_with_mapper(scores, sampling, rng, Some(mapper))?,
    };
    Ok(token + u32::from(matches!(control, SamplingControl::BlockThinking) && token >= THINKING))
}

#[allow(clippy::too_many_arguments)]
fn collect_output(
    target: &mut Session,
    tokenizer: &Tokenizer,
    mapper: &mut ProbabilityMapper,
    cancellation: &mut Option<CancelSignal>,
    draft_stats: &mut DraftStats,
    verify_depth: usize,
    mut logits: Vec<f32>,
    input_tokens: usize,
    starts_thinking: bool,
    options: &GenerationOptions,
    started: Instant,
    mut on_text: Option<&mut TextCallback<'_>>,
    on_progress: Option<&mut ProgressCallback<'_>>,
) -> Result<GeneratedTokens, &'static str> {
    let mut previous = None;
    let mut rng = options.seed;
    let mut modified_scores = Vec::new();
    let mut sample = |scores: &[f32], control| {
        sample_with_thinking_budget(scores, control, &options.sampling, &mut rng, mapper, &mut modified_scores)
    };
    let budget = (options.context_tokens as usize - input_tokens).min(options.max_new_tokens as usize);
    let mut thinking = options.thinking_budget_tokens.map(|limit| ThinkingBudget::new(limit, budget, starts_thinking));
    let mut delivered = 0;
    let mut session = Some(target);
    let mut pending: Option<PendingDraft<'_>> = None;
    let mut speculative = (verify_depth > 0).then(SpeculationStats::default);
    let mut round_started: Option<Instant> = None;
    let mut verified_at: Option<Instant> = None;
    let mut committed_at: Option<Instant> = None;
    // Per-round draft depth, up to the requested count, from running
    // acceptance and measured round costs. Output does not depend on it.
    let mut planner = DepthPlanner::new();
    let mut round_depth = 0;
    let tokens = std::iter::from_fn(|| {
        Some((|| {
            let mut legacy_proposal = None;
            if let Some(anchor) = previous
                && pending.is_none()
            {
                let target = session.take().ok_or("native_generation_failed")?;
                let limit = verify_depth.min(budget.saturating_sub(delivered + 1));
                let depth = if verify_depth > 0 {
                    planner.choose(limit)
                } else {
                    0
                };
                round_depth = depth;
                if let Some(stats) = speculative.as_mut() {
                    stats.rounds[depth] += 1;
                }
                if depth == 0 {
                    if verify_depth == 0 {
                        legacy_proposal = target.draft_propose(anchor).map_err(|_| "native_generation_failed")?;
                    }
                    let started = Instant::now();
                    logits = target.step(anchor).map_err(|_| "native_generation_failed")?;
                    if verify_depth > 0 {
                        planner.record(0, 0, 1000.0 * started.elapsed().as_secs_f64());
                    }
                    session = Some(target);
                } else {
                    let stats = speculative.as_mut().unwrap();
                    let now = Instant::now();
                    if let Some(at) = committed_at.take() {
                        stats.between_rounds_seconds += (now - at).as_secs_f64();
                    }
                    round_started = Some(now);
                    let (proposals, transaction) = if target.overlaps_verify_for(depth) {
                        let (proposals, transaction, before_verify_ms, verify_wall_ms) =
                            target.begin_overlapped_verify(anchor, depth).map_err(|_| "native_generation_failed")?;
                        *stats.overlapped_rounds.get_or_insert(0) += 1;
                        stats.draft_seconds += before_verify_ms * 0.001;
                        stats.verify_seconds += verify_wall_ms * 0.001;
                        (proposals, transaction)
                    } else {
                        let start = Instant::now();
                        let proposals = target.propose_verify(anchor, depth).map_err(|_| "native_generation_failed")?;
                        stats.draft_seconds += start.elapsed().as_secs_f64();
                        let input = std::iter::once(anchor).chain(proposals.iter().copied()).collect::<Vec<_>>();
                        let start = Instant::now();
                        let transaction = target.begin_verify(&input).map_err(|_| "native_generation_failed")?;
                        stats.verify_seconds += start.elapsed().as_secs_f64();
                        (proposals, transaction)
                    };
                    draft_stats.proposed += proposals.len();
                    verified_at = Some(Instant::now());
                    pending = Some(PendingDraft {
                        transaction,
                        proposals,
                        consumed: 0,
                    });
                }
            }
            let control = thinking
                .as_ref()
                .map_or(SamplingControl::Natural, |thinking| thinking.sampling_control(budget - delivered));
            let token = if let Some(mut draft) = pending.take() {
                let token =
                    sample(draft.transaction.logits(draft.consumed).map_err(|_| "native_generation_failed")?, control)?;
                let matched = draft.proposals.get(draft.consumed) == Some(&token);
                draft.consumed += 1;
                if matched {
                    draft_stats.accepted += 1;
                    pending = Some(draft);
                } else {
                    let start = Instant::now();
                    session = Some(draft.transaction.commit(draft.consumed).map_err(|_| "native_generation_failed")?);
                    let stats = speculative.as_mut().unwrap();
                    stats.finish_seconds += start.elapsed().as_secs_f64();
                    let done = Instant::now();
                    if let (Some(begun), Some(verified)) = (round_started.take(), verified_at.take()) {
                        let accepted = (draft.consumed - 1).min(MAX_VERIFY_ROWS - 1);
                        stats.round_seconds_by_accepted[accepted] += (done - begun).as_secs_f64();
                        stats.rounds_by_accepted[accepted] += 1;
                        stats.in_round_host_seconds += (start - verified).as_secs_f64();
                        planner.record(round_depth, draft.consumed - 1, 1000.0 * (done - begun).as_secs_f64());
                    }
                    committed_at = Some(done);
                }
                token
            } else {
                let token = sample(&logits, control)?;
                if let Some(proposal) = legacy_proposal {
                    draft_stats.proposed += 1;
                    draft_stats.accepted += usize::from(proposal == token);
                }
                token
            };
            if let Some(thinking) = thinking.as_mut() {
                thinking.observe(token, || tokenizer.id_to_token(token));
            }
            previous = Some(token);
            delivered += 1;
            Ok(u64::from(token))
        })())
    });
    let mut decoder = tokenizer.decode_stream(false);
    let mut filter = StopFilter::new(options.stop.clone());
    let mut raw = String::new();
    let needs_text = on_text.is_some() || !options.stop.is_empty();
    let result = collect_tokens_controlled(
        tokens,
        input_tokens,
        options,
        VOCAB,
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
        &mut || check_cancelled(cancellation, options.request_id),
    );
    let mut result = result?;
    if let Some(draft) = pending {
        // The consumer stopped mid-round. Its final sampled token remains
        // pending, just as it does after a serial step.
        let start = Instant::now();
        draft.transaction.commit(draft.consumed).map_err(|_| "native_generation_failed")?;
        speculative.as_mut().unwrap().finish_seconds += start.elapsed().as_secs_f64();
    }
    draft_stats.speculative = speculative;
    if needs_text {
        let complete = tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
        let (tail, stopped) = filter.push(complete.strip_prefix(&raw).ok_or("decoding_failed")?);
        if stopped {
            result.finish_reason = FinishReason::StopSequence;
        }
        if let Some(callback) = on_text {
            callback(&(tail + &filter.finish()))?;
        } else {
            filter.finish();
        }
        result.text = Some(filter.text().to_owned());
    }
    Ok(result)
}

impl GenerationBackend for InklingBackend {
    type PreparedImage = PreparedInklingImage;

    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn max_context_length(&self) -> Option<u32> {
        u32::try_from(self.target.max_context_length()).ok()
    }
    fn eos_token_ids(&self) -> &[u32] {
        EOS
    }

    fn supports_automatic_context_planning(&self) -> bool {
        true
    }
    fn supports_thinking_budget(&self) -> bool {
        true
    }
    fn automatic_context_length(
        &self,
        input: u32,
        maximum: u32,
        memory: usize,
    ) -> Option<u32> {
        let minimum = input.checked_add(1)?;
        let mut max = maximum.min(self.max_context_length()?);
        if minimum > max || self.context_bytes(minimum as usize, true, self.plans_source(minimum as usize))? > memory {
            return None;
        }
        let mut min = minimum;
        while min < max {
            let midpoint = min + (max - min).div_ceil(2);
            if self
                .context_bytes(midpoint as usize, true, self.plans_source(midpoint as usize))
                .is_some_and(|bytes| bytes <= memory)
            {
                min = midpoint;
            } else {
                max = midpoint - 1;
            }
        }
        Some(min)
    }

    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        self.tokenizer.encode(text, special).map(|x| x.get_ids().to_vec()).map_err(|_| "tokenization_failed")
    }

    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str> {
        self.tokenizer.decode(ids, skip_special).map_err(|_| "decoding_failed")
    }

    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        if input.contains(&IMAGE_TOKEN) {
            self.clear_prefix_cache();
            return Err("image_placeholder_requires_media");
        }
        let result = self.generate_inner(input, None, options, on_text, on_progress);
        self.target.reset();
        if result.is_err() {
            self.clear_prefix_cache();
        }
        result
    }

    fn cancelled(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        check_cancelled(&mut self.cancellation, request_id)
    }

    fn prefix_cache_supported(&self) -> bool {
        true
    }
    fn prefix_cache_chunk_tokens(&self) -> usize {
        reuse_grid(self.source_grid).0
    }
    fn prefix_cache_tail_tokens(&self) -> usize {
        reuse_grid(self.source_grid).1
    }
    fn prefix_cache_key(&self) -> &'static str {
        if self.source_verifier {
            "model-instance+context-tokens+cache-scope+mtp-enabled+source-precision"
        } else {
            "model-instance+context-tokens+cache-scope+mtp-enabled"
        }
    }
    fn prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        let source_prefill = self.source_text_prefill(input.len(), options);
        let boundary = self.boundary(input.len(), options);
        let reused = self
            .retained
            .as_ref()
            .filter(|entry| entry.state.0 == (options.mtp_drafts > 0, source_prefill))
            .map_or(0, |entry| entry.reusable(input, options.context_tokens, options.cache_scope.as_ref()));
        if reused <= boundary {
            reused
        } else {
            0
        }
    }
    fn publish_prefix_cache(&mut self) {
        self.retained = self.pending.take();
    }
    fn clear_prefix_cache(&mut self) {
        self.retained = None;
        self.pending = None;
    }
    fn prefix_cache_retained(&self) -> bool {
        self.retained.is_some() || self.pending.is_some()
    }

    fn supports_images(&self) -> bool {
        true
    }
    /// MTP is advertised only with the multi-row source verifier. The legacy
    /// one-draft route measured slower than serial decoding, so a worker
    /// without the verifier decodes serially and advertises no MTP.
    fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        (self.source_verifier && self.target.has_dspark()).then_some(MtpCapabilities {
            max_drafts: MAX_VERIFY_ROWS - 1,
            recommended_drafts: GREEDY_DEFAULT_DRAFTS as usize,
            sampling: "native-target-draw-match-v1",
            verification: SHORT_MTP_VERIFICATION,
        })
    }
    fn recommended_mtp_drafts_for(
        &self,
        sampling: &Sampling,
    ) -> Option<u32> {
        (self.default_mtp_drafts() > 0).then(|| default_drafts_for(sampling))
    }
    /// Text requests that name no MTP setting draft through the multi-row
    /// source verifier. Image requests are sent with zero drafts.
    fn default_mtp_drafts(&self) -> u32 {
        if self.source_verifier && self.target.has_dspark() {
            GREEDY_DEFAULT_DRAFTS.max(SAMPLED_DEFAULT_DRAFTS)
        } else {
            0
        }
    }
    fn default_mtp_drafts_for(
        &self,
        sampling: &Sampling,
    ) -> u32 {
        self.default_mtp_drafts().min(default_drafts_for(sampling))
    }
    /// Every draft depth, including one, runs through the source verifier. If
    /// it is unavailable at this context, decode serially rather than take
    /// the slower legacy one-draft route.
    fn effective_mtp_drafts(
        &self,
        options: &GenerationOptions,
    ) -> Result<u32, &'static str> {
        Ok(
            if options.mtp_drafts > 0
                && !(self.source_verifier && self.target.supports_source_text_prefill(options.context_tokens as usize))
            {
                0
            } else {
                options.mtp_drafts
            },
        )
    }
    fn mtp_metadata(&self) -> Option<Value> {
        self.target.has_dspark().then(|| json!({"stats":self.draft_stats}))
    }
    fn image_geometry(&self) -> (u32, u32) {
        (20, 2)
    }
    fn image_family_profile(&self) -> Option<(&str, u32)> {
        Some(("inkling2-mini-native-v1", 1))
    }
    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        Some((inkling_artifact::SOURCE, Some(inkling_artifact::REVISION)))
    }

    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        Some(ImageArtifactIdentity {
            schema_version: 7,
            artifact_sha256: self.layout.target_sha256(),
            config_sha256: &self.config_sha256,
            tokenizer_sha256: inkling_artifact::TOKENIZER_SHA,
            processor_sha256: inkling_artifact::PROCESSOR_SHA,
            template_sha256: inkling_artifact::TEMPLATE_SHA,
        })
    }

    fn runtime_identity(&self) -> Option<Value> {
        let identity = &self.runtime_identity;
        Some(json!({"backend":identity.backend, "device_name":identity.device_name,
            "apple_gpu_family":identity.apple_gpu_family, "gpu_core_count":identity.gpu_core_count, "supports_mxu":identity.supports_mxu}))
    }

    fn prepare_image(
        &mut self,
        input: Vec<u32>,
        encoded: &[u8],
        context: u32,
        request: u64,
    ) -> Result<(PreparedInklingImage, PreparedImageMetadata), &'static str> {
        self.clear_prefix_cache();
        require_active(&mut self.cancellation, request)?;
        self.target.reset();
        self.target.set_dspark_enabled(false).map_err(|_| "native_generation_failed")?;
        let mut slots = input.iter().enumerate().filter_map(|(i, &id)| (id == IMAGE_TOKEN).then_some(i));
        let slot = slots.next().ok_or("invalid_image_request")?;
        if slots.next().is_some()
            || slot == 0
            || input.get(slot - 1) != Some(&IMAGE_CONTENT)
            || input.get(slot + 1) != Some(&END_MESSAGE)
        {
            return Err("invalid_image_request");
        }
        let remaining = (context as usize).checked_sub(input.len()).ok_or("context_overflow")?;
        let image = self.target.encode_image(encoded, remaining).map_err(|_| "invalid_image_request")?;
        require_active(&mut self.cancellation, request)?;
        let grid = image.grid();
        let image_tokens = u32::try_from(image.token_count()).map_err(|_| "invalid_image_request")?;
        let metadata = PreparedImageMetadata {
            input_tokens: input.len() + image.token_count() - 1,
            image_tokens,
            grid,
            processed_dimensions: [grid[2] * 20, grid[1] * 20],
        };
        Ok((
            PreparedInklingImage {
                image,
                input,
                slot,
            },
            metadata,
        ))
    }

    fn generate_image(
        &mut self,
        prepared: PreparedInklingImage,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.clear_prefix_cache();
        let result =
            self.generate_inner(&prepared.input, Some((&prepared.image, prepared.slot)), options, on_text, on_progress);
        self.target.reset();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    #[test]
    fn thinking_budget_changes_only_the_selected_marker_in_target_sampling() {
        let alternate = THINKING + 1;
        let mut scores = vec![-1.0; VOCAB];
        scores[THINKING as usize] = 7.0;
        scores[alternate as usize] = 6.0;
        scores[END_MESSAGE as usize] = 5.0;
        let mut mapper = ProbabilityMapper::default();
        let mut scratch = Vec::new();

        for sampling in [
            Sampling::Greedy {},
            Sampling::Stochastic {
                temperature: 1.0,
                top_k: Some(1),
                top_p: None,
                min_p: None,
            },
        ] {
            let mut rng = 42;
            for (control, expected) in [
                (SamplingControl::Natural, THINKING),
                (SamplingControl::BlockThinking, alternate),
                (SamplingControl::BiasClose(3.0), END_MESSAGE),
                (SamplingControl::Close, END_MESSAGE),
            ] {
                assert_eq!(
                    sample_with_thinking_budget(&scores, control, &sampling, &mut rng, &mut mapper, &mut scratch),
                    Ok(expected),
                );
            }
        }
        assert_eq!(scores[THINKING as usize], 7.0);
        assert_eq!(scores[END_MESSAGE as usize], 5.0);
    }

    #[test]
    fn source_prefill_admits_every_nonempty_prompt() {
        // Empty input is rejected before admission; one token is the seed alone.
        assert_eq!(SOURCE_PREFILL_MIN_INPUT, 1);
        assert_eq!(Session::SOURCE_MIN_BULK_ROWS, 16);
        const { assert!(SOURCE_PREFILL_MIN_INPUT <= SOURCE_PROMPT_SUFFIX) };
    }

    #[test]
    fn text_requests_draft_through_the_multi_row_verifier_by_default() {
        assert_eq!(default_drafts_for(&Sampling::Greedy {}), 3);
        let sampled = Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1.0),
            min_p: None,
        };
        assert_eq!(default_drafts_for(&sampled), 2);
        assert_eq!(GREEDY_DEFAULT_DRAFTS as usize, MAX_VERIFY_ROWS - 1);
    }

    #[test]
    fn short_source_inputs_have_no_prefix_boundary() {
        for input in [0, 1, 3, 4, 5, 68, 69, 73, 511, 512, 2051] {
            assert_eq!(prefix_boundary(input, true, true), 0, "{input}");
        }
        assert_eq!(prefix_boundary(2052, true, true), 2048);
        assert_eq!(prefix_boundary(4096, true, true), 2048);
        assert_eq!(prefix_boundary(8800, true, true), 8192);
        assert_eq!(prefix_boundary(16_388, true, true), 16_384);
        assert_eq!(prefix_boundary(4096, false, true), 0);
        assert_eq!(prefix_boundary(73, true, false), 0);
        assert_eq!(prefix_boundary(CHUNK + 2, true, false), CHUNK);
    }

    #[test]
    fn reuse_boundaries_stay_on_the_advertised_grid() {
        assert_eq!(reuse_grid(true), (2048, 4));
        assert_eq!(reuse_grid(false), (CHUNK, 2));
        for source_grid in [false, true] {
            let (chunk, tail) = reuse_grid(source_grid);
            assert!(chunk.is_multiple_of(CHUNK) && tail >= 1);
            for input in [0, 1, tail, chunk, chunk + tail - 1, chunk + tail, 3 * chunk + 1, 9 * chunk + tail + 7] {
                let boundary = prefix_boundary(input, true, source_grid);
                assert!(boundary.is_multiple_of(chunk), "{input}");
                assert_eq!(boundary, input.saturating_sub(tail) / chunk * chunk, "{input}");
                assert_eq!(prefix_boundary(input, false, source_grid), 0, "{input}");
            }
            // A snapshot is reusable from `tail` remaining prompt tokens.
            assert_eq!(prefix_boundary(chunk + tail - 1, true, source_grid), 0);
            assert_eq!(prefix_boundary(chunk + tail, true, source_grid), chunk);
        }
    }

    #[test]
    fn unscoped_source_requests_split_where_scoped_ones_do() {
        // An unscoped request starts at 0 and splits at the scoped boundary.
        for input in [2052, 4096, 8800, 18_974, 20_584] {
            assert_eq!(unscoped_source_split(input, 0, true), prefix_boundary(input, true, true), "{input}");
        }
        assert_eq!(unscoped_source_split(18_974, 0, true), 18_432);
        // A scoped request has already run to its boundary, so nothing more is split.
        assert_eq!(unscoped_source_split(18_974, 18_432, true), 18_432);
        // Short inputs and the non-source path are left as one bulk.
        assert_eq!(unscoped_source_split(73, 0, true), 0);
        assert_eq!(unscoped_source_split(8800, 0, false), 0);
        assert_eq!(unscoped_source_split(8800, 1024, false), 1024);
    }

    #[test]
    fn source_command_rows_follow_admitted_memory() {
        const GIB: usize = 1024 * MIB;
        let fixed = 20 * GIB;
        // The resource budget keeps this much of the available memory free.
        let slack = 256 * MIB;
        let base = Session::batched_prefill_memory_bytes_at(true, Session::SOURCE_PREFILL_ROWS);
        assert_eq!(base, Session::batched_prefill_memory_bytes(true));
        // Below the base reserve the source path is not admitted at all.
        assert_eq!(source_command_rows(fixed, fixed + slack + base - 1), None);
        // The base width keeps its original rule: its reserve alone.
        assert_eq!(source_command_rows(fixed, fixed + slack + base), Some(2048));
        // A wider width must also leave room for the verifier, the proposal
        // head and a maximum-context request.
        let limit = Session::source_context_limit(true);
        let headroom = Session::context_memory_bytes(limit, true).unwrap()
            + Session::dspark_memory_bytes(limit, true).unwrap()
            + Session::source_global_extra_bytes(true, limit)
            + Session::SOURCE_VERIFIER_RESERVE_BYTES
            + Session::DSPARK_INT8_HEAD_BYTES;
        for (rows, narrower) in [(4096, 2048), (8192, 4096)] {
            let needed = fixed + slack + Session::batched_prefill_memory_bytes_at(true, rows) + headroom;
            assert_eq!(source_command_rows(fixed, needed - 1), Some(narrower), "{rows}");
            assert_eq!(source_command_rows(fixed, needed), Some(rows), "{rows}");
        }
        assert_eq!(source_command_rows(fixed, usize::MAX / 2), Some(8192));
        assert_eq!(Session::SOURCE_COMMAND_WIDTHS, [2048, 4096, 8192]);
    }

    #[test]
    fn source_context_limit_and_capacity() {
        assert_eq!(Session::source_context_limit(false), INITIAL_CONTEXT);
        assert_eq!(Session::source_context_limit(true), 32_768);
        for (context, capacity) in
            [(1, 4352), (73, 4352), (4352, 4352), (4353, 4353), (9216, 9216), (16_384, 16_384), (32_768, 32_768)]
        {
            assert_eq!(Session::source_global_capacity(context), capacity, "{context}");
        }
        assert_eq!(Session::source_global_capacity(40_000), 32_768);
    }

    #[test]
    fn source_global_workspace_bytes() {
        for (keys, mask, total) in [
            (4352, 133_275_648, 35_651_584),
            (9216, 282_230_784, 75_497_472),
            (16_384, 501_743_616, 134_217_728),
            (32_768, 1_003_487_232, 268_435_456),
        ] {
            assert_eq!(Session::source_global_mask_bytes(keys), mask, "{keys}");
            assert_eq!(Session::source_global_workspace_bytes(keys), total, "{keys}");
        }
        // The default path computes the relative bias in the attention kernel,
        // so the workspace is the key and value staging alone. Only the exact
        // diagnostic reference allocates the explicit mask (127 MiB at 4,352 keys).
        assert_eq!(Session::source_global_mask_bytes(4352) / MIB, 127);
        assert_eq!(Session::source_global_workspace_bytes(32_768) / MIB, 256);
    }

    #[test]
    fn source_global_extra_bytes_follow_the_context() {
        for context in [1, 73, 4096, 4352, 9216, 32_768] {
            assert_eq!(Session::source_global_extra_bytes(false, context), 0, "{context}");
        }
        for context in [1, 73, 4096, 4352] {
            assert_eq!(Session::source_global_extra_bytes(true, context), 0, "{context}");
        }
        let base = Session::source_global_workspace_bytes(4352);
        for keys in [4353, 9216, 16_384, 32_768] {
            assert_eq!(
                Session::source_global_extra_bytes(true, keys),
                Session::source_global_workspace_bytes(keys) - base,
                "{keys}"
            );
        }
        assert_eq!(Session::source_global_extra_bytes(true, 9216), 39_845_888);
        assert_eq!(Session::source_global_extra_bytes(true, 32_768), 232_783_872);
        // Contexts beyond the session maximum are rejected before allocation.
        assert_eq!(Session::source_global_extra_bytes(true, 32_769), 0);
        let mut previous = 0;
        for context in (4352..=32_768).step_by(512) {
            let extra = Session::source_global_extra_bytes(true, context);
            assert!(extra >= previous);
            previous = extra;
        }
    }
}
