// SPDX-License-Identifier: Apache-2.0
//! In-process Hy4 Flash Metal graph and native artifact readers on macOS arm64.

use std::{
    ffi::{CString, c_char, c_int, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::{self, NonNull},
    slice,
};

use super::context::{BackendContextBound, ContextSizing};

pub mod stq_dfly;
pub mod stq_layout;
mod stq_native;
pub use stq_native::StqOptions;
pub mod stq_source;
pub mod stq_trunk;

pub const VOCAB: usize = 120_832;
pub const PREFILL_CHUNK: usize = 1024;
pub const RECOMMENDED_DRAFTS: usize = 2;
pub const MAX_NEXTN_DRAFTS: usize = 4;
pub const MAX_DRAFTS: usize = 7;
pub const SHARD: &str = "Hy4-flash-Q3_K-00001-of-00002.gguf";
pub const ENGRAM_SHARD: &str = "Hy4-flash-Q3_K-00002-of-00002.gguf";
pub const PROJECTOR: &str = "mmproj-Hy4-flash-bf16.gguf";

#[derive(Clone, Copy, Default)]
#[repr(i32)]
pub enum ExpertStorage {
    #[default]
    Auto,
    Streamed,
    Mapped,
    MappedResident,
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct VisionGeometry {
    pub image_token: u32,
    pub patch: u32,
    pub merge: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
}

#[repr(C)]
pub struct ImageInput<'a> {
    rgb: *const u8,
    rgb_len: usize,
    width: u32,
    height: u32,
    _lifetime: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> ImageInput<'a> {
    pub fn new(
        rgb: &'a [u8],
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            rgb: rgb.as_ptr(),
            rgb_len: rgb.len(),
            width,
            height,
            _lifetime: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
#[repr(C)]
pub struct DraftStats {
    pub proposed: u64,
    pub accepted: u64,
    pub rounds: u64,
    pub draft_seconds: f64,
    pub verify_seconds: f64,
    pub catchup_seconds: f64,
    pub prefill_gpu_span_seconds: f64,
    pub prefill_encode_seconds: f64,
    pub prefill_expert_wait_seconds: f64,
    pub prefill_engram_seconds: f64,
    pub decode_gpu_span_seconds: f64,
    pub decode_host_cut_seconds: f64,
    pub prefill_expert_bytes_read: u64,
    pub decode_expert_bytes_read: u64,
    pub prefill_attention_seconds: f64,
    pub prefill_dense_seconds: f64,
    pub prefill_moe_route_seconds: f64,
    pub prefill_moe_expert_seconds: f64,
    pub prefill_other_seconds: f64,
    pub prefill_profiled_blocks: u64,
    pub prefill_grouped_tiles: u64,
    pub prefill_grouped_tail_tiles: u64,
    pub decode_encode_seconds: f64,
    pub decode_commit_seconds: f64,
    pub decode_completion_wait_seconds: f64,
    pub decode_engram_seconds: f64,
    pub decode_command_buffers: u64,
    pub decode_encoders: u64,
    pub decode_dispatches: u64,
    pub prefill_invalid_gpu_spans: u64,
    pub decode_invalid_gpu_spans: u64,
    pub prefill_attention_dense_seconds: f64,
    pub prefill_attention_index_seconds: f64,
    pub prefill_attention_mla_seconds: f64,
    pub prefill_attention_misc_seconds: f64,
    pub verify_mla_seconds: f64,
    pub verify_index_seconds: f64,
    pub verify_dense_seconds: f64,
    pub verify_router_seconds: f64,
    pub verify_expert_seconds: f64,
    pub verify_other_seconds: f64,
    pub verify_profiled_graphs: u64,
    /// Q_A, Q_B, K_B, KV_A, IndexPool projections, V_B, attention gate, output.
    pub prefill_stq_projection_seconds: [f64; 8],
}

#[derive(Clone, Copy)]
pub struct Options {
    pub chunk: usize,
    pub storage: ExpertStorage,
    pub grouped: bool,
    pub gpu_plan: bool,
    pub profile_prefill: bool,
    pub arena_bytes_requested: u64,
    pub demand_retention: bool,
    pub routed_half: bool,
    pub routed_fragments: bool,
    pub dense_packed: bool,
    pub dense_device_views: bool,
    pub dense_ffn_device_views: bool,
    pub resident_token_embeddings: bool,
    pub resident_draft_experts: bool,
    pub draft_head_q5: bool,
    pub tensor_mla: bool,
    pub compensated_sparse_mla: bool,
    pub parallel_router: bool,
    pub max_drafts: usize,
    pub context_limit: Option<u32>,
}

#[repr(C)]
struct ContextMemory {
    checkpoint_context: u32,
    engine_context: u32,
    pool_size: u32,
    draft_slack: u32,
    budget: u64,
    cache_bytes_per_token: u64,
    draft_cache_bytes_per_token: u64,
    pool_bytes_per_row: u64,
    draft_pool_bytes_per_row: u64,
}

impl ContextMemory {
    fn estimate(
        &self,
        context: u32,
    ) -> Option<usize> {
        let context = u64::from(context);
        let draft = context + u64::from(self.draft_slack);
        let pool = u64::from(self.pool_size);
        let cache = context.checked_mul(self.cache_bytes_per_token)?;
        let draft_cache = draft.checked_mul(self.draft_cache_bytes_per_token)?;
        let pooled = (context.checked_div(pool)? + 1).checked_mul(self.pool_bytes_per_row)?;
        let draft_pooled = (draft.checked_div(pool)? + 1).checked_mul(self.draft_pool_bytes_per_row)?;
        usize::try_from(cache.checked_add(draft_cache)?.checked_add(pooled)?.checked_add(draft_pooled)?).ok()
    }
}

struct ContextSelection {
    limit: Option<u32>,
    pool_size: Option<u32>,
    sizing: Option<Result<ContextSizing, &'static str>>,
}

unsafe extern "C" fn select_context(
    memory: *const ContextMemory,
    selection: *mut c_void,
) -> u32 {
    // The native open call owns the borrowed inventory and calls this synchronously.
    let memory = unsafe { &*memory };
    let selection = unsafe { &mut *selection.cast::<ContextSelection>() };
    selection.pool_size = Some(memory.pool_size);
    let budget = stark_resource_budget::inference_memory_budget(memory.budget as usize);
    let sizing = hy4_context_sizing(memory, budget, selection.limit);
    let context_tokens = sizing.as_ref().map_or(0, |sizing| sizing.context_tokens);
    selection.sizing = Some(sizing);
    context_tokens
}

fn hy4_context_sizing(
    memory: &ContextMemory,
    budget: usize,
    cap: Option<u32>,
) -> Result<ContextSizing, &'static str> {
    ContextSizing::fit_loaded(
        memory.checkpoint_context,
        Some(BackendContextBound {
            context_tokens: memory.engine_context,
            source: "hy4_metal_buffer_or_i32_index",
        }),
        memory.checkpoint_context.min(memory.engine_context),
        budget,
        |context| memory.estimate(context),
    )?
    .with_user_cap(cap)
}

unsafe extern "C" {
    fn stark_hy4_open(
        model: *const c_char,
        projector: *const c_char,
        pack: *const c_char,
        chunk: c_int,
        storage: c_int,
        grouped: c_int,
        gpu_plan: c_int,
        profile_prefill: c_int,
        arena_bytes_requested: u64,
        demand_retention: c_int,
        routed_half: c_int,
        dense_packed: c_int,
        resident_token_embeddings: c_int,
        tensor_mla: c_int,
        compensated_sparse_mla: c_int,
        parallel_router: c_int,
        max_drafts: c_int,
        routed_fragments: c_int,
        dense_device_views: c_int,
        dense_ffn_device_views: c_int,
        resident_draft_experts: c_int,
        draft_head_q5: c_int,
        select_context: unsafe extern "C" fn(*const ContextMemory, *mut c_void) -> u32,
        sizing: *mut c_void,
        session: *mut *mut c_void,
        vision: *mut VisionGeometry,
        mapped: *mut c_int,
        arena_bytes: *mut u64,
        actual_draft_head_q5: *mut c_int,
    ) -> c_int;
    fn stark_hy4_close(session: *mut c_void);
    fn stark_hy4_parallel_router_active(session: *const c_void) -> c_int;
    fn stark_hy4_clear(session: *mut c_void);
    fn stark_hy4_retain_prefix(
        session: *mut c_void,
        position: c_int,
    ) -> c_int;
    fn stark_hy4_begin(
        session: *mut c_void,
        tokens: *const u32,
        count: c_int,
        context: c_int,
        reuse: c_int,
        drafts: c_int,
        image: *const ImageInput<'_>,
    ) -> c_int;
    fn stark_hy4_prefill(
        session: *mut c_void,
        rows: c_int,
    ) -> c_int;
    fn stark_hy4_greedy(session: *const c_void) -> c_int;
    fn stark_hy4_logits(session: *const c_void) -> *const f32;
    fn stark_hy4_append(
        session: *mut c_void,
        token: u32,
    ) -> c_int;
    fn stark_hy4_verify(
        session: *mut c_void,
        count: c_int,
        lookup: *const u32,
        tokens: *mut u32,
    ) -> c_int;
    fn stark_hy4_stats(session: *mut c_void) -> DraftStats;
}

pub struct Session {
    inner: NonNull<c_void>,
    vision: VisionGeometry,
    mapped_experts: bool,
    draft_head_q5: bool,
    arena_bytes: u64,
    sizing: ContextSizing,
    pool_size: u32,
    max_drafts: usize,
}

fn path_arg(path: &Path) -> Result<CString, &'static str> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid_hy4_path")
}

fn generation_status(status: c_int) -> Result<(), &'static str> {
    (status == 0).then_some(()).ok_or("native_generation_failed")
}

fn completed_pool_boundary(
    prompt_tokens: usize,
    minimum: usize,
    pool_size: u32,
) -> Option<usize> {
    let before_final_token = prompt_tokens.checked_sub(1)?;
    let pool_size = usize::try_from(pool_size).ok()?;
    if pool_size == 0 {
        return None;
    }
    let boundary = before_final_token / pool_size * pool_size;
    (boundary > minimum).then_some(boundary)
}

impl Session {
    pub fn load(
        model: &Path,
        prepare: &Path,
        options: Options,
    ) -> Result<Self, &'static str> {
        if !(RECOMMENDED_DRAFTS..=MAX_DRAFTS).contains(&options.max_drafts) {
            return Err("invalid_hy4_mtp_max_drafts");
        }
        let shard = path_arg(&model.join(SHARD))?;
        let projector = path_arg(&model.join(PROJECTOR))?;
        let prepare = path_arg(prepare)?;
        let mut inner = ptr::null_mut();
        let mut vision = VisionGeometry::default();
        let mut mapped = 0;
        let mut arena_bytes = 0;
        let mut actual_draft_head_q5 = 0;
        let mut selection = ContextSelection {
            limit: options.context_limit,
            pool_size: None,
            sizing: None,
        };
        // The path arguments remain alive throughout initialization. The returned
        // handle owns all model data and the device queue.
        let code = unsafe {
            stark_hy4_open(
                shard.as_ptr(),
                projector.as_ptr(),
                prepare.as_ptr(),
                c_int::try_from(options.chunk).map_err(|_| "invalid_hy4_prefill_chunk")?,
                options.storage as c_int,
                c_int::from(options.grouped),
                c_int::from(options.gpu_plan),
                c_int::from(options.profile_prefill),
                options.arena_bytes_requested,
                c_int::from(options.demand_retention),
                c_int::from(options.routed_half),
                c_int::from(options.dense_packed),
                c_int::from(options.resident_token_embeddings),
                c_int::from(options.tensor_mla),
                c_int::from(options.compensated_sparse_mla),
                c_int::from(options.parallel_router),
                options.max_drafts as c_int,
                c_int::from(options.routed_fragments),
                c_int::from(options.dense_device_views),
                c_int::from(options.dense_ffn_device_views),
                c_int::from(options.resident_draft_experts),
                c_int::from(options.draft_head_q5),
                select_context,
                ptr::from_mut(&mut selection).cast(),
                &mut inner,
                &mut vision,
                &mut mapped,
                &mut arena_bytes,
                &mut actual_draft_head_q5,
            )
        };
        if code != 0 {
            return Err(match code {
                1 => "invalid_hy4_gguf",
                2 => "hy4_weights_load_failed",
                3 => "hy4_projector_load_failed",
                4 => "hy4_metal_graph_load_failed",
                5 => "hy4_draft_load_failed",
                6 => "hy4_expert_arena_load_failed",
                8 => "hy4_mapped_experts_exceed_budget",
                9 => "hy4_larger_arena_exceeds_budget",
                10 => selection.sizing.and_then(Result::err).unwrap_or("resource_budget_exceeded"),
                _ => "hy4_native_load_failed",
            });
        }
        Ok(Self {
            inner: NonNull::new(inner).ok_or("hy4_native_load_failed")?,
            vision,
            mapped_experts: mapped != 0,
            draft_head_q5: actual_draft_head_q5 != 0,
            arena_bytes,
            sizing: selection.sizing.ok_or("invalid_finite_context_limit")??,
            pool_size: selection.pool_size.ok_or("invalid_hy4_pool_size")?,
            max_drafts: options.max_drafts,
        })
    }

    pub fn vision_geometry(&self) -> VisionGeometry {
        self.vision
    }

    pub fn mapped_experts(&self) -> bool {
        self.mapped_experts
    }

    pub fn parallel_router_active(&self) -> bool {
        unsafe { stark_hy4_parallel_router_active(self.inner.as_ptr()) != 0 }
    }

    pub fn draft_head_format(&self) -> &'static str {
        if self.draft_head_q5 {
            "q5_0"
        } else {
            "q8_0"
        }
    }

    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    pub fn context_sizing(&self) -> ContextSizing {
        self.sizing
    }

    pub fn index_pool_group_tokens(&self) -> u32 {
        self.pool_size
    }

    /// Largest completed IndexPool boundary before the prompt's final token.
    /// Keeps one token available to produce fresh logits on the next request.
    pub fn prefill_tail_boundary(
        &self,
        prompt_tokens: usize,
        minimum: usize,
    ) -> Option<usize> {
        completed_pool_boundary(prompt_tokens, minimum, self.pool_size)
    }

    pub fn max_drafts(&self) -> usize {
        self.max_drafts
    }
    pub fn begin(
        &mut self,
        tokens: &[u32],
        context: u32,
        reused: usize,
        drafts: u32,
        image: Option<&ImageInput<'_>>,
    ) -> Result<(), &'static str> {
        let count = c_int::try_from(tokens.len()).map_err(|_| "context_overflow")?;
        let context = c_int::try_from(context).map_err(|_| "context_overflow")?;
        let reuse = c_int::try_from(reused).map_err(|_| "context_overflow")?;
        let drafts = c_int::try_from(drafts).map_err(|_| "mtp_not_supported")?;
        // The native call copies the token sequence and consumes image pixels
        // synchronously. Neither input pointer is retained.
        generation_status(unsafe {
            stark_hy4_begin(
                self.inner.as_ptr(),
                tokens.as_ptr(),
                count,
                context,
                reuse,
                drafts,
                image.map_or(ptr::null(), ptr::from_ref),
            )
        })
    }

    pub fn prefill(
        &mut self,
        rows: usize,
    ) -> Result<(), &'static str> {
        let rows = c_int::try_from(rows).map_err(|_| "context_overflow")?;
        generation_status(unsafe { stark_hy4_prefill(self.inner.as_ptr(), rows) })
    }

    pub fn greedy(&self) -> Result<u32, &'static str> {
        let token = unsafe { stark_hy4_greedy(self.inner.as_ptr()) };
        (token >= 0 && (token as usize) < VOCAB).then_some(token as u32).ok_or("invalid_generation_output")
    }

    pub fn logits(&self) -> Result<&[f32], &'static str> {
        let logits = unsafe { stark_hy4_logits(self.inner.as_ptr()) };
        if logits.is_null() {
            return Err("invalid_generation_output");
        }
        // The graph resolves exactly VOCAB logits at open. The returned view is
        // borrowed from the handle, and mutation requires an exclusive borrow.
        Ok(unsafe { slice::from_raw_parts(logits, VOCAB) })
    }

    pub fn append(
        &mut self,
        token: u32,
    ) -> Result<(), &'static str> {
        generation_status(unsafe { stark_hy4_append(self.inner.as_ptr(), token) })
    }

    pub fn verify(
        &mut self,
        rows: usize,
        lookup: &[u32],
        output: &mut [u32; MAX_DRAFTS + 1],
    ) -> Result<usize, &'static str> {
        if !(1..=output.len()).contains(&rows) {
            return Err("invalid_generation_output");
        }
        if !lookup.is_empty() && lookup.len() != rows - 1 {
            return Err("invalid_generation_output");
        }
        let lookup = if lookup.is_empty() {
            ptr::null()
        } else {
            lookup.as_ptr()
        };
        let count = unsafe { stark_hy4_verify(self.inner.as_ptr(), rows as c_int, lookup, output.as_mut_ptr()) };
        (count > 0 && count as usize <= rows).then_some(count as usize).ok_or("native_generation_failed")
    }

    pub fn stats(&mut self) -> DraftStats {
        unsafe { stark_hy4_stats(self.inner.as_ptr()) }
    }

    pub fn retain_prefix(
        &mut self,
        rows: usize,
    ) -> Result<(), &'static str> {
        let rows = c_int::try_from(rows).map_err(|_| "context_overflow")?;
        generation_status(unsafe { stark_hy4_retain_prefix(self.inner.as_ptr(), rows) })
    }

    pub fn clear(&mut self) {
        unsafe { stark_hy4_clear(self.inner.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::completed_pool_boundary;

    #[stark_test]
    fn completed_pool_boundary_rounds_down_and_leaves_a_prompt_token() {
        assert_eq!(completed_pool_boundary(16_701, 16_384, 128), Some(16_640));
        assert_eq!(completed_pool_boundary(16_384, 16_000, 128), Some(16_256));
        assert_eq!(completed_pool_boundary(4_097, 4_096, 128), None);
        assert_eq!(completed_pool_boundary(1, 0, 128), None);
        assert_eq!(completed_pool_boundary(16_701, 16_384, 0), None);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { stark_hy4_close(self.inner.as_ptr()) }
    }
}

#[cfg(test)]
mod context_tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::engine::context::ContextConstraint;

    #[stark_test]
    fn hy4_callback_selects_and_records_its_admitted_user_cap() {
        let mut memory = ContextMemory {
            checkpoint_context: 262_144,
            engine_context: 65_536,
            pool_size: 128,
            draft_slack: 16,
            budget: 512 * 1024 * 1024,
            cache_bytes_per_token: 16,
            draft_cache_bytes_per_token: 16,
            pool_bytes_per_row: 16,
            draft_pool_bytes_per_row: 16,
        };
        let mut selection = ContextSelection {
            limit: Some(32_768),
            pool_size: None,
            sizing: None,
        };
        let context = unsafe { select_context(&memory, ptr::from_mut(&mut selection).cast()) };
        assert_eq!(context, 32_768);
        assert_eq!(selection.pool_size, Some(memory.pool_size));
        let sizing = selection.sizing.unwrap().unwrap();
        assert_eq!(sizing.limiting_factor, ContextConstraint::UserCap);
        assert_eq!(sizing.user_context_cap, Some(32_768));
        assert_eq!(sizing.backend_bound.unwrap().source, "hy4_metal_buffer_or_i32_index");
        memory.pool_size = 0;
        assert_eq!(unsafe { select_context(&memory, ptr::from_mut(&mut selection).cast()) }, 0);
        assert_eq!(selection.sizing.unwrap().unwrap_err(), "context_memory_estimate_unavailable");
        memory.engine_context = 1;
        assert_eq!(unsafe { select_context(&memory, ptr::from_mut(&mut selection).cast()) }, 0);
        assert_eq!(selection.sizing.unwrap().unwrap_err(), "invalid_finite_context_limit");
    }
}
