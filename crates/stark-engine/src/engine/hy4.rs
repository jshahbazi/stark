// SPDX-License-Identifier: Apache-2.0
//! In-process AngelSlim Hy4 Flash Metal graph on macOS arm64.

use std::{
    ffi::{CString, c_char, c_int, c_void},
    os::unix::ffi::OsStrExt,
    path::Path,
    ptr::{self, NonNull},
    slice,
};

pub const VOCAB: usize = 120_832;
pub const CONTEXT: u32 = 8192;
pub const PREFILL_CHUNK: usize = 1024;
pub const MAX_DRAFTS: usize = 2;
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
}

unsafe extern "C" {
    fn stark_hy4_open(
        model: *const c_char,
        projector: *const c_char,
        pack: *const c_char,
        context: c_int,
        chunk: c_int,
        storage: c_int,
        grouped: c_int,
        gpu_plan: c_int,
        session: *mut *mut c_void,
        vision: *mut VisionGeometry,
        mapped: *mut c_int,
    ) -> c_int;
    fn stark_hy4_close(session: *mut c_void);
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
        tokens: *mut u32,
    ) -> c_int;
    fn stark_hy4_stats(session: *mut c_void) -> DraftStats;
}

pub struct Session {
    inner: NonNull<c_void>,
    vision: VisionGeometry,
    mapped_experts: bool,
}

fn path_arg(path: &Path) -> Result<CString, &'static str> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid_hy4_path")
}

fn generation_status(status: c_int) -> Result<(), &'static str> {
    (status == 0).then_some(()).ok_or("native_generation_failed")
}

impl Session {
    pub fn load(
        model: &Path,
        prepare: &Path,
        chunk: usize,
        storage: ExpertStorage,
        grouped: bool,
        gpu_plan: bool,
    ) -> Result<Self, &'static str> {
        let shard = path_arg(&model.join(SHARD))?;
        let projector = path_arg(&model.join(PROJECTOR))?;
        let prepare = path_arg(prepare)?;
        let mut inner = ptr::null_mut();
        let mut vision = VisionGeometry::default();
        let mut mapped = 0;
        // The path arguments remain alive throughout initialization. The returned
        // handle owns all model data and the device queue.
        let code = unsafe {
            stark_hy4_open(
                shard.as_ptr(),
                projector.as_ptr(),
                prepare.as_ptr(),
                CONTEXT as c_int,
                c_int::try_from(chunk).map_err(|_| "invalid_hy4_prefill_chunk")?,
                storage as c_int,
                c_int::from(grouped),
                c_int::from(gpu_plan),
                &mut inner,
                &mut vision,
                &mut mapped,
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
                _ => "hy4_native_load_failed",
            });
        }
        Ok(Self {
            inner: NonNull::new(inner).ok_or("hy4_native_load_failed")?,
            vision,
            mapped_experts: mapped != 0,
        })
    }

    pub fn vision_geometry(&self) -> VisionGeometry {
        self.vision
    }

    pub fn mapped_experts(&self) -> bool {
        self.mapped_experts
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
        output: &mut [u32; MAX_DRAFTS + 1],
    ) -> Result<usize, &'static str> {
        if !(1..=output.len()).contains(&rows) {
            return Err("invalid_generation_output");
        }
        let count = unsafe { stark_hy4_verify(self.inner.as_ptr(), rows as c_int, output.as_mut_ptr()) };
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

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { stark_hy4_close(self.inner.as_ptr()) }
    }
}
