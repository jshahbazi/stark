// SPDX-License-Identifier: Apache-2.0
//! Hy4 Metal backend through Stark's serialized generation worker.

mod source;
mod vision;

use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        hy4::{
            DraftStats, ExpertStorage, ImageInput, MAX_DRAFTS, MAX_NEXTN_DRAFTS, Options, PREFILL_CHUNK,
            RECOMMENDED_DRAFTS, SHARD, Session, VOCAB, stq_source,
        },
        hybrid::{
            activity::InferenceActivity,
            mtp_lookup::{NewlineTokens, PromptLookup},
        },
    },
};
use tokenizers::Tokenizer;

use self::vision::Hy4PreparedImage;
use crate::{
    cancel::CancelSignal,
    generation::{
        FinishReason, GeneratedTokens, GenerationBackend, GenerationOptions, HY4_STQ_VISION_PROFILE,
        HY4_VISION_PROFILE, ImageArtifactIdentity, MtpCapabilities, PreparedImageMetadata, ProgressCallback, Sampling,
        TextCallback, collect_tokens_controlled,
    },
    hybrid::{ProbabilityMapper, report_prefill_progress, sample_with_mapper},
    prefix_cache::{self, BoundaryEntry, PrefixEntry},
    text_stop::StopFilter,
};

const EOS: &[u32] = &[120_025];
const TOKENIZER_SHA: &str = "2e3f3d22065c2be5765ce99ab2410411ae338b32ce809ea20dcdb61d00198e83";
const CONFIG_SHA: &str = "1f56ac1a3063f926e91ec43842bfdc0fd6d4401ef91bcdecd1fc27dd506ea968";
const PROCESSOR_SHA: &str = "cc5cc63da33e0ebf53b8ac035879982ba18c47965f82a5f341bc82d98759dc1d";
const TEMPLATE_SHA: &str = "11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35";
const ENGRAM_MAP_SHA: &str = "99ed7bbd800d9d9c087aa2cbbf0fcc497839c519045f6f3bf9aca177cc1769b5";
const STQ_CONFIG_SHA: &str = "c1c5af8449a8e7db0dc5c5e8aa768c80673c94604997cfe13c6ee6aa98e9a3e3";
const STQ_TEMPLATE_SHA: &str = "63ddd5de294df36f10c8f2c8e5f7b2a9dc2011c8481335422f04554256aee4ad";
// SHA-256 of the pinned sorted file-name, size and content-digest inventory.
const STQ_ARTIFACT_SHA: &str = "1ed7e181a27f78ffb1e93f9bef2eb00be3c9c5933cf5d7f50494320b45e9a04f";

fn stq_validation(path: &Path) -> bool {
    std::env::var("STARK_HY4_STQ_VALIDATION").as_deref() == Ok("1") && path.join("model-dfly.safetensors").is_file()
}

pub(crate) fn recognizes(path: &Path) -> bool {
    path.join(SHARD).is_file() || stq_validation(path)
}

fn verify(
    path: &Path,
    expected: &str,
) -> Result<(), &'static str> {
    let artifact = File::open(path).map_err(|_| "incomplete_hy4_artifact")?;
    let mut digest = Sha256::new();
    io::copy(&mut BufReader::with_capacity(1024 * 1024, artifact), &mut digest).map_err(|_| "invalid_hy4_artifact")?;
    (format!("{:x}", digest.finalize()) == expected).then_some(()).ok_or("invalid_hy4_artifact")
}

fn check_cancelled(
    signal: &mut Option<CancelSignal>,
    request: u64,
) -> Result<bool, &'static str> {
    signal.as_mut().map_or(Ok(false), |signal| signal.check(request))
}

fn verify_switch(
    name: &str,
    error: &'static str,
    default: bool,
) -> Result<bool, &'static str> {
    match std::env::var(name).as_deref() {
        Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        Err(std::env::VarError::NotPresent) => Ok(default),
        _ => Err(error),
    }
}

fn optional_bound(
    name: &str,
    bounds: std::ops::RangeInclusive<u32>,
    error: &'static str,
) -> Result<Option<u32>, &'static str> {
    match std::env::var(name) {
        Ok(value) => value.parse().ok().filter(|value| bounds.contains(value)).map(Some).ok_or(error),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(error),
    }
}

pub(crate) struct Hy4Backend {
    native: Session,
    tokenizer: Tokenizer,
    cancellation: Option<CancelSignal>,
    mapper: ProbabilityMapper,
    retained: Option<PrefixEntry<(u32, bool)>>,
    pending: Option<PrefixEntry<(u32, bool)>>,
    tail_retained: Option<BoundaryEntry<(u32, bool)>>,
    tail_pending: Option<BoundaryEntry<(u32, bool)>>,
    stats: DraftStats,
    lookup_stats: LookupStats,
    lookup_newline: Option<Arc<NewlineTokens>>,
    tail_prefix_enabled: bool,
    runtime: Value,
    chunk: usize,
    stq: bool,
    recommended_drafts: usize,
}

#[derive(Default, serde::Serialize)]
struct LookupStats {
    proposed: usize,
    accepted: usize,
    rounds: usize,
}

const STREAMED_PREFILL_ARENA_GB: u32 = 40;

fn resolve_load_profiles(
    setting: Result<&str, &std::env::VarError>,
    supports_metal4: bool,
    storage_override: bool,
) -> Result<&'static [&'static str], &'static str> {
    match setting {
        Ok("default") => Ok(&["default"]),
        Ok("streamed-prefill") => Ok(&["streamed-prefill"]),
        Ok("resident-decode") => Ok(&["resident-decode"]),
        Err(std::env::VarError::NotPresent) if supports_metal4 && !storage_override => {
            Ok(&["resident-decode", "streamed-prefill", "default"])
        },
        Err(std::env::VarError::NotPresent) if supports_metal4 => Ok(&["streamed-prefill", "default"]),
        Err(std::env::VarError::NotPresent) => Ok(&["default"]),
        _ => Err("invalid_hy4_load_profile"),
    }
}

fn load_memory_failure(result: &Result<Session, &'static str>) -> bool {
    matches!(
        result,
        Err("hy4_mapped_experts_exceed_budget"
            | "hy4_larger_arena_exceeds_budget"
            | "hy4_expert_arena_load_failed"
            | "resource_budget_exceeded")
    )
}

impl Hy4Backend {
    fn load_options(
        load_profile: &str,
        lookup_enabled: bool,
    ) -> Result<(Options, bool), &'static str> {
        let streamed_prefill = load_profile == "streamed-prefill";
        let resident_decode = load_profile == "resident-decode";
        let tuned = streamed_prefill || resident_decode;
        let storage = match std::env::var("STARK_HY4_EXPERT_STORAGE").as_deref() {
            Ok("streamed") => ExpertStorage::Streamed,
            Ok("mapped") => ExpertStorage::Mapped,
            Ok("mapped-resident") => ExpertStorage::MappedResident,
            Ok("auto") => ExpertStorage::Auto,
            Err(std::env::VarError::NotPresent) if resident_decode => ExpertStorage::MappedResident,
            Err(std::env::VarError::NotPresent) => ExpertStorage::Streamed,
            _ => return Err("invalid_hy4_expert_storage"),
        };
        let grouped = verify_switch("STARK_HY4_GROUPED_PREFILL", "invalid_hy4_grouped_prefill", true)?;
        let gpu_plan = verify_switch("STARK_HY4_GPU_PLAN", "invalid_hy4_gpu_plan", true)?;
        let chunk = match std::env::var("STARK_HY4_PREFILL_CHUNK").as_deref() {
            Ok("1024") => PREFILL_CHUNK,
            Ok("4096") => 4096,
            Err(std::env::VarError::NotPresent) if tuned => 4096,
            Err(std::env::VarError::NotPresent) => PREFILL_CHUNK,
            _ => return Err("invalid_hy4_prefill_chunk"),
        };
        let profile_prefill = verify_switch("STARK_HY4_PROFILE_PREFILL", "invalid_hy4_prefill_profile", false)?;
        let larger_arena = verify_switch("STARK_HY4_STREAM_ARENA_36GB", "invalid_hy4_stream_arena", false)?;
        let arena_gb = optional_bound("STARK_HY4_STREAM_ARENA_GB", 1..=128, "invalid_hy4_stream_arena")?;
        if larger_arena && arena_gb.is_some() {
            return Err("invalid_hy4_stream_arena");
        }
        let prefill_arena = streamed_prefill && matches!(storage, ExpertStorage::Streamed);
        let arena_bytes_requested = u64::from(arena_gb.unwrap_or(if larger_arena {
            36
        } else if prefill_arena {
            STREAMED_PREFILL_ARENA_GB
        } else {
            0
        })) * 1_000_000_000;
        let context_limit = optional_bound("STARK_HY4_CONTEXT_LIMIT", 2..=u32::MAX, "invalid_hy4_context_limit")?;
        let demand_retention =
            verify_switch("STARK_HY4_STREAM_DEMAND_RETENTION", "invalid_hy4_stream_demand_retention", prefill_arena)?;
        let draft_limit = if lookup_enabled {
            MAX_DRAFTS
        } else {
            MAX_NEXTN_DRAFTS
        };
        let max_drafts = optional_bound(
            "STARK_HY4_MTP_MAX_DRAFTS",
            RECOMMENDED_DRAFTS as u32..=draft_limit as u32,
            "invalid_hy4_mtp_max_drafts",
        )?
        .unwrap_or(if resident_decode {
            3
        } else {
            RECOMMENDED_DRAFTS as u32
        }) as usize;
        let routed_half =
            verify_switch("STARK_HY4_PREFILL_ROUTED_HALF", "invalid_hy4_prefill_routed_half", tuned && grouped)?;
        let routed_fragments = verify_switch(
            "STARK_HY4_PREFILL_ROUTED_FRAGMENTS",
            "invalid_hy4_prefill_routed_fragments",
            tuned && routed_half && grouped,
        )?;
        let dense_packed = verify_switch("STARK_HY4_PREFILL_DENSE_PACKED", "invalid_hy4_prefill_dense_packed", tuned)?;
        let dense_device_views = verify_switch(
            "STARK_HY4_PREFILL_DENSE_DEVICE_VIEWS",
            "invalid_hy4_prefill_dense_device_views",
            tuned && dense_packed,
        )?;
        let dense_ffn_device_views = verify_switch(
            "STARK_HY4_PREFILL_DENSE_FFN_DEVICE_VIEWS",
            "invalid_hy4_prefill_dense_ffn_device_views",
            prefill_arena && dense_device_views,
        )?;
        let tensor_mla = verify_switch("STARK_HY4_PREFILL_TENSOR_MLA", "invalid_hy4_prefill_tensor_mla", tuned)?;
        let compensated_sparse_mla = verify_switch(
            "STARK_HY4_PREFILL_COMPENSATED_SPARSE_MLA",
            "invalid_hy4_prefill_compensated_sparse_mla",
            false,
        )?;
        let parallel_router = verify_switch("STARK_HY4_PARALLEL_ROUTER", "invalid_hy4_parallel_router", false)?;
        let resident_token_embeddings =
            verify_switch("STARK_HY4_EMBEDDINGS_RESIDENT", "invalid_hy4_embeddings_resident", tuned)?;
        let resident_draft_experts =
            verify_switch("STARK_HY4_MTP_RESIDENT_EXPERTS", "invalid_hy4_mtp_resident_experts", resident_decode)?;
        let draft_head_q5 = verify_switch("STARK_HY4_MTP_HEAD_Q5", "invalid_hy4_mtp_head_q5", tuned)?;
        if routed_half && !grouped {
            return Err("invalid_hy4_prefill_routed_half");
        }
        if dense_device_views && !dense_packed {
            return Err("invalid_hy4_prefill_dense_device_views");
        }
        if dense_ffn_device_views && !dense_device_views {
            return Err("invalid_hy4_prefill_dense_ffn_device_views");
        }
        if routed_fragments && !(routed_half && grouped) {
            return Err("invalid_hy4_prefill_routed_fragments");
        }
        if demand_retention && !matches!(storage, ExpertStorage::Streamed) {
            return Err("invalid_hy4_stream_demand_retention");
        }
        if arena_bytes_requested != 0 && !matches!(storage, ExpertStorage::Streamed) {
            return Err("invalid_hy4_stream_arena");
        }
        let profile_arena = prefill_arena && arena_gb.is_none() && !larger_arena;
        Ok((
            Options {
                chunk,
                storage,
                grouped,
                gpu_plan,
                profile_prefill,
                arena_bytes_requested,
                demand_retention,
                routed_half,
                routed_fragments,
                dense_packed,
                dense_device_views,
                dense_ffn_device_views,
                resident_token_embeddings,
                resident_draft_experts,
                draft_head_q5,
                tensor_mla,
                compensated_sparse_mla,
                parallel_router,
                max_drafts,
                context_limit,
            },
            profile_arena,
        ))
    }

    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        if stq_validation(path) {
            return Self::load_stq_validation(path, cancellation);
        }
        let source = source::VerifiedSource::open(path)?;
        for (name, digest) in [
            ("config.json", CONFIG_SHA),
            ("tokenizer.json", TOKENIZER_SHA),
            ("processor_config.json", PROCESSOR_SHA),
            ("chat_template.jinja", TEMPLATE_SHA),
            ("engram_token_map.bin", ENGRAM_MAP_SHA),
        ] {
            verify(&path.join(name), digest)?;
        }
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| "invalid_hy4_tokenizer")?;
        if tokenizer.get_vocab_size(true) != VOCAB {
            return Err("invalid_hy4_tokenizer");
        }
        let engine = Engine::<Metal>::new().map_err(|_| "native_backend_unavailable")?;
        let identity = engine.metal_device_identity();
        let profile_setting = std::env::var("STARK_HY4_LOAD_PROFILE");
        let automatic = matches!(profile_setting, Err(std::env::VarError::NotPresent));
        let storage_override = [
            "STARK_HY4_EXPERT_STORAGE",
            "STARK_HY4_STREAM_ARENA_GB",
            "STARK_HY4_STREAM_ARENA_36GB",
            "STARK_HY4_STREAM_DEMAND_RETENTION",
        ]
        .iter()
        .any(|name| std::env::var_os(name).is_some());
        let mut profiles =
            resolve_load_profiles(profile_setting.as_deref(), engine.supports_metal4(), storage_override)?
                .iter()
                .copied()
                .peekable();
        drop(engine);
        let lookup_enabled = verify_switch("STARK_HY4_MTP_LOOKUP", "invalid_hy4_mtp_lookup", false)?;
        let tail_prefix_enabled =
            verify_switch("STARK_HY4_PREFILL_TAIL_PREFIX", "invalid_hy4_prefill_tail_prefix", false)?;
        let (load_profile, options, native) = loop {
            let load_profile = profiles.next().ok_or("invalid_hy4_load_profile")?;
            let (mut options, profile_arena) = Self::load_options(load_profile, lookup_enabled)?;
            let mut native = Session::load(path, path, options);
            // A busy host can reject the built-in larger arena while admitting the tuned graph.
            if automatic && profile_arena && load_memory_failure(&native) {
                options.arena_bytes_requested = 0;
                native = Session::load(path, path, options);
            }
            if profiles.peek().is_some() && load_memory_failure(&native) {
                continue;
            }
            break (load_profile, options, native?);
        };
        source.verify_unchanged()?;
        let lookup_newline = lookup_enabled.then(|| Arc::new(NewlineTokens::from_tokenizer(&tokenizer, VOCAB)));
        let runtime = json!({
            "backend": identity.backend,
            "device_name": identity.device_name,
            "apple_gpu_family": identity.apple_gpu_family,
            "gpu_core_count": identity.gpu_core_count,
            "supports_mxu": identity.supports_mxu,
            "hy4_load_profile": load_profile,
            "expert_storage": if native.mapped_experts() { "mapped" } else { "streamed" },
            "mapped_expert_residency": matches!(options.storage, ExpertStorage::MappedResident),
            "grouped_prefill_requested": options.grouped,
            "grouped_prefill_operands": if options.routed_half { "f16" } else if options.grouped { "f32" } else { "off" },
            "grouped_prefill_fragments_requested": options.routed_fragments,
            "gpu_expert_plan": options.gpu_plan && native.mapped_experts(),
            "prefill_chunk_tokens": options.chunk,
            "prefill_stage_profile_requested": options.profile_prefill,
            "prefill_dense_packed_requested": options.dense_packed,
            "prefill_dense_device_views_requested": options.dense_device_views,
            "prefill_dense_ffn_device_views_requested": options.dense_ffn_device_views,
            "prefill_tensor_mla_requested": options.tensor_mla,
            "prefill_compensated_sparse_mla_requested": options.compensated_sparse_mla,
            "parallel_router_active": native.parallel_router_active(),
            "token_embeddings_resident": options.resident_token_embeddings,
            "draft_experts_resident": options.resident_draft_experts,
            "draft_head_format": native.draft_head_format(),
            "streamed_expert_demand_retention": options.demand_retention,
            "streamed_expert_arena_bytes": native.arena_bytes(),
            "streamed_expert_arena_requested_bytes": options.arena_bytes_requested,
        });
        Ok(Self {
            native,
            tokenizer,
            cancellation,
            mapper: ProbabilityMapper::default(),
            retained: None,
            pending: None,
            tail_retained: None,
            tail_pending: None,
            stats: DraftStats::default(),
            lookup_stats: LookupStats::default(),
            lookup_newline,
            tail_prefix_enabled,
            runtime,
            chunk: options.chunk,
            stq: false,
            recommended_drafts: RECOMMENDED_DRAFTS,
        })
    }

    fn load_stq_validation(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        let source = stq_source::StqSource::open(path).map_err(|_| "invalid_hy4_stq_artifact")?;
        for (name, digest) in [
            ("tokenizer.json", TOKENIZER_SHA),
            ("processor_config.json", PROCESSOR_SHA),
            ("chat_template.jinja", STQ_TEMPLATE_SHA),
        ] {
            verify(&path.join(name), digest)?;
        }
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| "invalid_hy4_tokenizer")?;
        if tokenizer.get_vocab_size(true) != VOCAB {
            return Err("invalid_hy4_tokenizer");
        }
        let engine = Engine::<Metal>::new().map_err(|_| "native_backend_unavailable")?;
        let identity = engine.metal_device_identity();
        drop(engine);
        let chunk = match std::env::var("STARK_HY4_PREFILL_CHUNK").as_deref() {
            Ok("1024") => 1024,
            Ok("4096") | Err(std::env::VarError::NotPresent) => 4096,
            _ => return Err("invalid_hy4_prefill_chunk"),
        };
        let context =
            optional_bound("STARK_HY4_CONTEXT_LIMIT", 2..=u32::MAX, "invalid_hy4_context_limit")?.unwrap_or(u32::MAX);
        let profile = verify_switch("STARK_HY4_PROFILE_PREFILL", "invalid_hy4_prefill_profile", false)?;
        let round_bf16 = verify_switch("STARK_HY4_STQ_BF16", "invalid_hy4_stq_precision", false)?;
        let pair1_decode = verify_switch("STARK_HY4_STQ_PAIR1_DECODE", "invalid_hy4_stq_decode", false)?;
        let resident = verify_switch("STARK_HY4_STQ_RESIDENT", "invalid_hy4_stq_residency", false)?;
        let tensor_mla = verify_switch("STARK_HY4_STQ_TENSOR_MLA", "invalid_hy4_stq_tensor_mla", false)?;
        let gpu_verify_plan = verify_switch("STARK_HY4_STQ_GPU_VERIFY", "invalid_hy4_stq_gpu_verify", false)?;
        let register_mla = verify_switch("STARK_HY4_STQ_REGISTER_MLA", "invalid_hy4_stq_register_mla", false)?;
        let fused_finish_reduce = verify_switch("STARK_HY4_STQ_FUSED_FINISH", "invalid_hy4_stq_fused_finish", false)?;
        let split_spec1 = verify_switch("STARK_HY4_STQ_SPLIT_SPEC1", "invalid_hy4_stq_split_spec1", false)?;
        let verify_register_mla = verify_switch("STARK_HY4_STQ_VERIFY_MLA", "invalid_hy4_stq_verify_mla", false)?;
        let cache_engram_pages =
            verify_switch("STARK_HY4_STQ_ENGRAM_PAGE_CACHE", "invalid_hy4_stq_engram_cache", false)?;
        let wide_attention = verify_switch("STARK_HY4_STQ_WIDE_ATTENTION", "invalid_hy4_stq_wide_attention", false)?;
        let wide_routed = verify_switch("STARK_HY4_STQ_WIDE_ROUTED", "invalid_hy4_stq_wide_routed", false)?;
        let draft_register_mla =
            verify_switch("STARK_HY4_STQ_DRAFT_REGISTER_MLA", "invalid_hy4_stq_draft_register_mla", false)?;
        let wide_ffn = verify_switch("STARK_HY4_STQ_WIDE_FFN", "invalid_hy4_stq_wide_ffn", false)?;
        let fast_verify = verify_switch("STARK_HY4_STQ_FAST_VERIFY", "invalid_hy4_stq_fast_verify", false)?;
        let fused_attn_hc = verify_switch("STARK_HY4_STQ_FUSED_ATTN_HC", "invalid_hy4_stq_fused_attn_hc", false)?;
        let batch_draft = verify_switch("STARK_HY4_STQ_BATCH_DRAFT", "invalid_hy4_stq_batch_draft", false)?;
        let tail_prefix_enabled =
            verify_switch("STARK_HY4_PREFILL_TAIL_PREFIX", "invalid_hy4_prefill_tail_prefix", false)?;
        // This opt-in validation run borrows an explicitly pinned, immutable
        // local artifact. Its caller supplies the verified weight inventory.
        let native = unsafe {
            Session::load_stq_diagnostic(
                &source,
                stark_engine::engine::hy4::StqOptions {
                    chunk: chunk as u32,
                    context_limit: context,
                    profile,
                    round_bf16,
                    pair1_decode,
                    resident,
                    tensor_mla,
                    gpu_verify_plan,
                    register_mla,
                    fused_finish_reduce,
                    split_spec1,
                    verify_register_mla,
                    cache_engram_pages,
                    wide_attention,
                    wide_routed,
                    draft_register_mla,
                    wide_ffn,
                    fast_verify,
                    fused_attn_hc,
                    batch_draft,
                },
            )
        }
        .map_err(|_| "hy4_stq_native_load_failed")?;
        source.verify_unchanged().map_err(|_| "invalid_hy4_stq_artifact")?;
        let recommended_drafts =
            optional_bound("STARK_HY4_STQ_DRAFT_DEPTH", 1..=native.max_drafts() as u32, "invalid_hy4_stq_depth")?
                .map_or(RECOMMENDED_DRAFTS, |depth| depth as usize);
        let runtime = json!({
            "backend": identity.backend,
            "device_name": identity.device_name,
            "apple_gpu_family": identity.apple_gpu_family,
            "gpu_core_count": identity.gpu_core_count,
            "supports_mxu": identity.supports_mxu,
            "prefill_chunk_tokens": chunk,
            "bf16_projection_boundaries": round_bf16,
            "pair1_single_token_gemv": pair1_decode,
            "mapped_weight_residency": resident,
            "prefill_tensor_mla_requested": tensor_mla,
            "gpu_verify_plan": gpu_verify_plan,
            "prefill_register_mla": register_mla,
            "prefill_fused_expert_finish": fused_finish_reduce,
            "verify_split_spec1": split_spec1,
            "verify_register_mla": verify_register_mla,
            "cache_engram_pages": cache_engram_pages,
            "prefill_stq_wide_attention": wide_attention,
            "prefill_stq_wide_routed": wide_routed,
            "draft_register_mla": draft_register_mla,
            "prefill_stq_wide_ffn": wide_ffn,
            "verify_stq_fast": fast_verify,
            "stq_fused_attn_hc": fused_attn_hc,
            "stq_batch_draft": batch_draft,
            "prefill_stage_profile_requested": profile,
            "draft_head_format": "spec1-k4-dfly2",
        });
        Ok(Self {
            native,
            tokenizer,
            cancellation,
            mapper: ProbabilityMapper::default(),
            retained: None,
            pending: None,
            tail_retained: None,
            tail_pending: None,
            stats: DraftStats::default(),
            lookup_stats: LookupStats::default(),
            lookup_newline: None,
            tail_prefix_enabled,
            runtime,
            chunk,
            stq: true,
            recommended_drafts,
        })
    }

    fn generate_inner(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        image: Option<&ImageInput<'_>>,
        mut on_text: Option<&mut TextCallback<'_>>,
        mut on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let limit = options.context_tokens as usize;
        if input.is_empty()
            || input.len() >= limit
            || options.context_tokens > self.native.context_sizing().context_tokens
        {
            return Err("context_overflow");
        }
        let _activity = InferenceActivity::begin().map_err(|_| "inference_activity_unavailable")?;
        let started = Instant::now();
        let reused = if image.is_none() {
            self.prefix_cache_plan(input, options)
        } else {
            0
        };
        self.retained = None;
        self.pending = None;
        self.tail_retained = None;
        self.tail_pending = None;
        self.stats = DraftStats::default();
        self.lookup_stats = LookupStats::default();
        if self.cancelled(options.request_id)? {
            return Err("generation_cancelled");
        }
        self.native.begin(input, options.context_tokens, reused, options.mtp_drafts, image)?;
        let mut last_reported = Duration::ZERO;
        for rows in input[reused..].chunks(self.chunk) {
            if self.cancelled(options.request_id)? {
                return Err("generation_cancelled");
            }
            self.native.prefill(rows.len())?;
            report_prefill_progress(started.elapsed(), &mut last_reported, on_progress.as_deref_mut())?;
        }

        let budget = (options.max_new_tokens as usize).min(limit - input.len());
        let mut previous = None;
        let mut verified = [0; MAX_DRAFTS + 1];
        let mut next_verified = 0;
        let mut verified_len = 0;
        let mut sampled = 0;
        let mut rng = options.seed;
        let native = &mut self.native;
        let mapper = &mut self.mapper;
        let greedy = matches!(options.sampling, Sampling::Greedy {});
        let drafts = options.mtp_drafts as usize;
        let mut lookup = if options.mtp_scheduling.lookup {
            let newline = self.lookup_newline.as_ref().ok_or("mtp_not_supported")?;
            Some(PromptLookup::new(input.to_vec(), Arc::clone(newline)))
        } else {
            None
        };
        let mut lookup_stats = LookupStats::default();
        let tokens = std::iter::from_fn(|| {
            Some((|| {
                if let Some(token) = previous.take() {
                    native.append(token)?;
                }
                let token = if drafts == 0 || sampled == 0 {
                    let token = if greedy {
                        native.greedy()?
                    } else {
                        sample_with_mapper(native.logits()?, &options.sampling, &mut rng, Some(mapper))?
                    };
                    previous = Some(token);
                    token
                } else {
                    if next_verified == verified_len {
                        let limit = drafts.min(budget - sampled - 1);
                        let proposed = lookup.as_ref().map_or_else(Vec::new, |lookup| lookup.propose(limit));
                        let rows = if proposed.is_empty() {
                            limit + 1
                        } else {
                            proposed.len() + 1
                        };
                        verified_len = native.verify(rows, &proposed, &mut verified)?;
                        if !proposed.is_empty() {
                            lookup_stats.proposed += proposed.len();
                            lookup_stats.accepted += verified_len - 1;
                            lookup_stats.rounds += 1;
                        }
                        next_verified = 0;
                    }
                    let token = verified[next_verified];
                    next_verified += 1;
                    token
                };
                if let Some(lookup) = &mut lookup {
                    lookup.push(token);
                }
                sampled += 1;
                Ok(u64::from(token))
            })())
        });

        let mut decoder = self.tokenizer.decode_stream(false);
        let mut stop = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let mut result = collect_tokens_controlled(
            tokens,
            input.len(),
            options,
            VOCAB,
            started,
            &mut |token| {
                if needs_text && let Some(text) = decoder.step(token).map_err(|_| "decoding_failed")? {
                    raw.push_str(&text);
                    let (text, stopped) = stop.push(&text);
                    if let Some(callback) = on_text.as_deref_mut() {
                        callback(&text)?;
                    }
                    return Ok(!stopped);
                }
                Ok(true)
            },
            on_progress,
            &mut || check_cancelled(&mut self.cancellation, options.request_id),
        )?;
        if needs_text {
            let full = self.tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
            let (tail, stopped) = stop.push(full.strip_prefix(&raw).ok_or("decoding_failed")?);
            if stopped {
                result.finish_reason = FinishReason::StopSequence;
            }
            let tail = tail + &stop.finish();
            if let Some(callback) = on_text {
                callback(&tail)?;
            }
            result.text = Some(stop.text().to_owned());
        }
        self.stats = self.native.stats();
        self.lookup_stats = lookup_stats;
        result.reused_input_tokens = reused;
        let boundary = prefix_cache::chunk_boundary(input.len(), self.chunk);
        if let Some(scope) = options.cache_scope.filter(|_| image.is_none()) {
            let mut tail_saved = false;
            if self.tail_prefix_enabled
                && let Some(tail_boundary) = self.native.prefill_tail_boundary(input.len(), boundary)
            {
                // Keep the last complete IndexPool group before the prompt's
                // final token. The remaining prompt suffix will be prefetched
                // on every hit, reproducing the fresh prompt endpoint exactly.
                if self.native.retain_prefix(tail_boundary).is_ok() {
                    self.tail_pending = Some(BoundaryEntry::new(
                        (options.mtp_drafts, options.mtp_scheduling.lookup),
                        scope,
                        options.context_tokens,
                        &input[..tail_boundary],
                    ));
                    tail_saved = true;
                }
            }
            if boundary > 0 {
                if !tail_saved {
                    self.native.retain_prefix(boundary)?;
                }
                self.pending = Some(PrefixEntry::with_chunk(
                    (options.mtp_drafts, options.mtp_scheduling.lookup),
                    scope,
                    options.context_tokens,
                    input,
                    self.chunk,
                ));
            }
            if boundary == 0 && !tail_saved {
                self.native.clear();
            }
        } else {
            self.native.clear();
        }
        Ok(result)
    }

    fn tail_prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        if !self.tail_prefix_enabled {
            return 0;
        }
        self.tail_retained
            .as_ref()
            .filter(|entry| entry.state == (options.mtp_drafts, options.mtp_scheduling.lookup))
            .map_or(0, |entry| {
                entry.reusable_with_min_suffix(input, options.context_tokens, options.cache_scope.as_ref(), 1)
            })
    }
}

impl GenerationBackend for Hy4Backend {
    type PreparedImage = Hy4PreparedImage;

    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn max_context_length(&self) -> Option<u32> {
        Some(self.native.context_sizing().context_tokens)
    }
    fn context_sizing_metadata(&self) -> Option<Value> {
        Some(json!(self.native.context_sizing()))
    }
    fn supports_automatic_context_planning(&self) -> bool {
        // The native graph budgets and allocates the full capacity at load.
        // Individual text requests can use any smaller logical window.
        true
    }
    fn eos_token_ids(&self) -> &[u32] {
        EOS
    }
    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        self.tokenizer
            .encode(text, special)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|_| "tokenization_failed")
    }
    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str> {
        self.tokenizer.decode(ids, skip_special).map_err(|_| "decoding_failed")
    }
    fn cancelled(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        check_cancelled(&mut self.cancellation, request_id)
    }

    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        if input.contains(&self.native.vision_geometry().image_token) {
            self.clear_prefix_cache();
            return Err("image_placeholder_requires_media");
        }
        let result = self.generate_inner(input, options, None, on_text, on_progress);
        if result.is_err() {
            self.clear_prefix_cache();
        }
        result
    }

    fn supports_images(&self) -> bool {
        true
    }
    fn image_geometry(&self) -> (u32, u32) {
        let geometry = self.native.vision_geometry();
        (geometry.patch, geometry.merge)
    }
    fn image_family_profile(&self) -> Option<(&str, u32)> {
        Some((
            if self.stq {
                HY4_STQ_VISION_PROFILE
            } else {
                HY4_VISION_PROFILE
            },
            1,
        ))
    }
    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        Some(ImageArtifactIdentity {
            schema_version: 7,
            artifact_sha256: if self.stq {
                STQ_ARTIFACT_SHA
            } else {
                source::ARTIFACT_SHA256
            },
            config_sha256: if self.stq {
                STQ_CONFIG_SHA
            } else {
                CONFIG_SHA
            },
            tokenizer_sha256: TOKENIZER_SHA,
            processor_sha256: PROCESSOR_SHA,
            template_sha256: if self.stq {
                STQ_TEMPLATE_SHA
            } else {
                TEMPLATE_SHA
            },
        })
    }
    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        Some(if self.stq {
            (stq_source::REPOSITORY, Some(stq_source::REVISION))
        } else {
            (source::REPOSITORY, Some(source::REVISION))
        })
    }
    fn runtime_identity(&self) -> Option<Value> {
        Some(self.runtime.clone())
    }

    fn prepare_image(
        &mut self,
        input: Vec<u32>,
        encoded: &[u8],
        context: u32,
        request: u64,
    ) -> Result<(Self::PreparedImage, PreparedImageMetadata), &'static str> {
        self.clear_prefix_cache();
        if self.cancelled(request)? {
            return Err("generation_cancelled");
        }
        let result = Hy4PreparedImage::prepare(input, encoded, context, self.native.vision_geometry())?;
        if self.cancelled(request)? {
            return Err("generation_cancelled");
        }
        Ok(result)
    }

    fn generate_image(
        &mut self,
        image: Self::PreparedImage,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.clear_prefix_cache();
        let result = self.generate_inner(&image.tokens, options, Some(&image.image_input()), on_text, on_progress);
        if result.is_err() {
            self.clear_prefix_cache();
        }
        result
    }

    fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        Some(MtpCapabilities {
            max_drafts: self.native.max_drafts(),
            recommended_drafts: self.recommended_drafts,
            sampling: "greedy-target-match-v1",
            verification: if self.stq {
                "hy4-dfly2-greedy-prefix-match-v1"
            } else {
                "hy4-nextn-greedy-prefix-match-v1"
            },
        })
    }
    fn effective_mtp_drafts(
        &self,
        options: &GenerationOptions,
    ) -> Result<u32, &'static str> {
        Ok(if matches!(options.sampling, Sampling::Greedy {}) {
            options.mtp_drafts
        } else {
            0
        })
    }
    fn mtp_metadata(&self) -> Option<Value> {
        Some(if self.lookup_newline.is_some() {
            json!({"stats": self.stats, "lookup_stats": self.lookup_stats})
        } else {
            json!({"stats": self.stats})
        })
    }
    fn supports_mtp_lookup(&self) -> bool {
        self.lookup_newline.is_some()
    }
    fn prefix_cache_supported(&self) -> bool {
        true
    }
    fn prefix_cache_chunk_tokens(&self) -> usize {
        self.chunk
    }
    fn prefix_cache_subchunk_tokens(&self) -> Option<usize> {
        let alignment = self.native.index_pool_group_tokens() as usize;
        (self.tail_prefix_enabled && (1..self.chunk).contains(&alignment)).then_some(alignment)
    }
    fn prefix_cache_key(&self) -> &'static str {
        if self.lookup_newline.is_some() {
            "model-instance+context-tokens+cache-scope+mtp-drafts+mtp-lookup"
        } else {
            "model-instance+context-tokens+cache-scope+mtp-drafts"
        }
    }
    fn prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        let coarse = self
            .retained
            .as_ref()
            .filter(|entry| entry.state == (options.mtp_drafts, options.mtp_scheduling.lookup))
            .map_or(0, |entry| entry.reusable(input, options.context_tokens, options.cache_scope.as_ref()));
        coarse.max(self.tail_prefix_cache_plan(input, options))
    }
    fn prefix_cache_reuse_valid(
        &self,
        reused: usize,
        input_tokens: usize,
    ) -> bool {
        if reused >= input_tokens {
            return false;
        }
        if reused.is_multiple_of(self.chunk) {
            return true;
        }
        self.tail_prefix_enabled && self.tail_retained.as_ref().is_some_and(|entry| entry.boundary() == reused)
    }
    fn publish_prefix_cache(&mut self) {
        self.retained = self.pending.take();
        self.tail_retained = self.tail_pending.take();
    }
    fn clear_prefix_cache(&mut self) {
        self.pending = None;
        self.retained = None;
        self.tail_pending = None;
        self.tail_retained = None;
        self.native.clear();
    }
    fn prefix_cache_retained(&self) -> bool {
        self.retained.is_some() || self.pending.is_some() || self.tail_retained.is_some() || self.tail_pending.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{load_memory_failure, resolve_load_profiles};

    #[test]
    fn implicit_load_prefers_resident_then_streamed_on_metal4() {
        let unset = Err(&std::env::VarError::NotPresent);
        assert_eq!(
            resolve_load_profiles(unset, true, false),
            Ok(&["resident-decode", "streamed-prefill", "default"][..])
        );
        assert_eq!(resolve_load_profiles(unset, true, true), Ok(&["streamed-prefill", "default"][..]));
        assert_eq!(resolve_load_profiles(unset, false, false), Ok(&["default"][..]));
    }

    #[test]
    fn explicit_load_profile_selects_one_candidate() {
        assert_eq!(resolve_load_profiles(Ok("default"), true, false), Ok(&["default"][..]));
        assert_eq!(resolve_load_profiles(Ok("streamed-prefill"), false, false), Ok(&["streamed-prefill"][..]));
        assert_eq!(resolve_load_profiles(Ok("resident-decode"), true, true), Ok(&["resident-decode"][..]));
        assert_eq!(resolve_load_profiles(Ok("map"), true, false), Err("invalid_hy4_load_profile"));
    }

    #[test]
    fn implicit_load_can_retry_memory_admission_errors() {
        assert!(load_memory_failure(&Err("hy4_mapped_experts_exceed_budget")));
        assert!(load_memory_failure(&Err("hy4_larger_arena_exceeds_budget")));
        assert!(load_memory_failure(&Err("hy4_expert_arena_load_failed")));
        assert!(load_memory_failure(&Err("resource_budget_exceeded")));
        assert!(!load_memory_failure(&Err("invalid_hy4_gguf")));
        assert!(!load_memory_failure(&Err("hy4_metal_graph_load_failed")));
    }
}
