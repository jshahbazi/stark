// SPDX-License-Identifier: Apache-2.0
//! Owned serial execution of the pinned EXL3/FP8 GLM pack.
//!
//! The bridge owns all MLX state and returns complete F32 vocabulary readouts.
//! Rust retains a prompt only while its batched prefill is pending. Snapshots
//! retain their private source prefix in memory and are never logged or
//! persisted. The staged library advertises implementation availability;
//! every artifact and precision policy needs separate qualification.

use std::{
    marker::PhantomData,
    path::Path,
    ptr::{self, NonNull},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use super::{Error, MtpCapabilities, NativeDecodeStats, NativeSampling, Result, glm_exl3_ffi as ffi};

const MAX_CONTEXT: usize = 1_048_576;
const MAX_VOCAB: usize = 1_000_000;
const MAX_GPU_BYTES: u64 = 112 << 30;
const WIRED_MARGIN_BYTES: u64 = 2 << 30;
const PREFILL_CHUNK: usize = 16_384;
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

// New symbols exist only in a source-bound native stage selected by the build
// manifest. Ordinary entry points retain their version-one ABI.
#[cfg(any(test, stark_glm_exl3_native_decode_stage))]
mod native_ffi {
    #[cfg(stark_glm_exl3_native_decode_stage)]
    use super::ffi;

    pub const ABI_VERSION: u32 = 1;

    #[repr(C)]
    pub struct Sampling {
        pub struct_size: u32,
        pub abi_version: u32,
        pub mode: u32,
        pub reserved: u32,
        pub temperature: f32,
        pub top_p: f32,
        pub seed: u64,
    }

    #[repr(C)]
    pub struct Round {
        pub struct_size: u32,
        pub abi_version: u32,
        pub count: u32,
        pub accepted: u32,
        pub verified_drafts: u32,
        pub next_pending: u32,
        pub tokens: [u32; 4],
        pub position: u64,
    }

    impl Round {
        pub fn empty() -> Self {
            Self {
                struct_size: size_of::<Self>() as u32,
                abi_version: ABI_VERSION,
                count: 0,
                accepted: 0,
                verified_drafts: 0,
                next_pending: 0,
                tokens: [0; 4],
                position: 0,
            }
        }
    }

    #[cfg(stark_glm_exl3_native_decode_stage)]
    unsafe extern "C" {
        pub fn stark_glm_load_native(
            path: *const u8,
            path_len: usize,
            options: *const ffi::LoadOptions,
            load_mtp_head: u32,
            out: *mut *mut ffi::Model,
        ) -> u32;
        pub fn stark_glm_set_mtp_enabled(
            model: *mut ffi::Model,
            enabled: u32,
        ) -> u32;
        pub fn stark_glm_native_initial(
            model: *mut ffi::Model,
            sampling: *const Sampling,
            out: *mut u32,
        ) -> u32;
        pub fn stark_glm_native_round(
            model: *mut ffi::Model,
            pending: u32,
            accepted_cap: u32,
            out: *mut Round,
        ) -> u32;
        pub fn stark_glm_native_end_request(model: *mut ffi::Model) -> u32;
    }
}

const CAP_NATIVE_DECODE: u64 = 8;
const CAP_MTP_PREFIX: u64 = 16;

fn native_round_cap(
    total: usize,
    committed: usize,
    drafts: usize,
) -> Result<u32> {
    if total == 0 || committed >= total || drafts > 3 {
        return Err(Error::Invalid("GLM EXL3 native stream budget"));
    }
    Ok(drafts.min(total - committed - 1) as u32)
}

struct CommittedRound {
    tokens: [u32; 4],
    count: usize,
    accepted: usize,
    verified_drafts: usize,
    next_pending: u32,
}

fn queue_committed(
    round: &CommittedRound,
    skip_initial: bool,
    queue: &mut [u32; 4],
) -> usize {
    let skip = usize::from(skip_initial);
    let count = round.count - skip;
    queue[..count].copy_from_slice(&round.tokens[skip..round.count]);
    count
}

#[cfg(any(test, stark_glm_exl3_native_decode_stage))]
fn checked_round(
    round: native_ffi::Round,
    pending: u32,
    cap: u32,
    position: usize,
    vocab: usize,
) -> Result<CommittedRound> {
    let count = round.count as usize;
    if round.struct_size != size_of::<native_ffi::Round>() as u32
        || round.abi_version != native_ffi::ABI_VERSION
        || cap > 3
        || !(1..=4).contains(&count)
        || round.accepted.checked_add(1) != Some(round.count)
        || round.accepted > round.verified_drafts
        || round.verified_drafts > cap
        || round.tokens[0] != pending
        || round.next_pending as usize >= vocab
        || round.tokens[..count].iter().any(|&token| token as usize >= vocab)
        || position.checked_add(count).map(|end| end as u64) != Some(round.position)
    {
        return Err(Error::Invalid("GLM EXL3 native round output"));
    }
    Ok(CommittedRound {
        tokens: round.tokens,
        count,
        accepted: round.accepted as usize,
        verified_drafts: round.verified_drafts as usize,
        next_pending: round.next_pending,
    })
}

/// An explicit per-instance allocation policy. The bridge checks its complete
/// phase bill before materializing the model; row counts of zero use its
/// budget-derived choice instead of pack-provided overfill pins.
pub struct LoadOptions {
    pub context_tokens: usize,
    pub gpu_ceiling_bytes: u64,
    pub wired_margin_bytes: u64,
    pub expert_rows: u32,
    pub expert_prefill_rows: u32,
    /// The native library must advertise snapshot support. Each actual
    /// artifact and precision policy still needs its own qualification.
    pub prefix_reuse: bool,
    /// Explicit development-stage load. Never selected by the ordinary loader.
    pub native_decode_stage: bool,
    pub load_mtp_head: bool,
}

impl LoadOptions {
    pub fn new(context_tokens: usize) -> Self {
        Self {
            context_tokens,
            gpu_ceiling_bytes: MAX_GPU_BYTES,
            wired_margin_bytes: WIRED_MARGIN_BYTES,
            expert_rows: 0,
            expert_prefill_rows: 0,
            prefix_reuse: false,
            native_decode_stage: false,
            load_mtp_head: false,
        }
    }

    fn native(&self) -> Result<ffi::LoadOptions> {
        if self.context_tokens == 0
            || self.context_tokens > MAX_CONTEXT
            || self.gpu_ceiling_bytes == 0
            || self.gpu_ceiling_bytes > MAX_GPU_BYTES
            || self.wired_margin_bytes == 0
            || self.wired_margin_bytes >= self.gpu_ceiling_bytes
            || self.expert_rows > 288
            || self.expert_prefill_rows > 288
        {
            return Err(Error::Invalid("GLM EXL3 allocation policy"));
        }
        Ok(ffi::LoadOptions {
            struct_size: size_of::<ffi::LoadOptions>() as u32,
            abi_version: ffi::ABI_VERSION,
            context_tokens: self.context_tokens as u64,
            gpu_ceiling_bytes: self.gpu_ceiling_bytes,
            wired_margin_bytes: self.wired_margin_bytes,
            expert_rows: self.expert_rows,
            expert_prefill_rows: self.expert_prefill_rows,
        })
    }
}

fn path_bytes(path: &Path) -> Result<&[u8]> {
    let bytes = path.to_str().ok_or(Error::Invalid("GLM EXL3 model path"))?.as_bytes();
    if !path.is_absolute() || bytes.is_empty() || bytes.len() > 4096 || bytes.contains(&0) {
        return Err(Error::Invalid("GLM EXL3 model path"));
    }
    Ok(bytes)
}

fn admitted_end(
    position: usize,
    pending: usize,
    count: usize,
    limit: usize,
) -> Result<usize> {
    position
        .checked_add(pending)
        .and_then(|end| end.checked_add(count))
        .filter(|&end| end <= limit)
        .ok_or(Error::Invalid("GLM EXL3 context boundary"))
}

fn status_result(status: u32) -> Result<()> {
    match status {
        ffi::OK => Ok(()),
        ffi::UNSUPPORTED => Err(Error::Unsupported("GLM EXL3 operation unavailable")),
        ffi::INVALID_ARGUMENT => Err(Error::Invalid("GLM EXL3 operation admission")),
        ffi::ABI_MISMATCH => Err(Error::Invalid("GLM EXL3 bridge ABI")),
        ffi::CONTEXT_LIMIT => Err(Error::Invalid("GLM EXL3 context or memory boundary")),
        ffi::CANCELLED => Err(Error::Invalid("GLM EXL3 operation cancelled")),
        ffi::OUT_OF_MEMORY => Err(Error::Invalid("GLM EXL3 allocation failed")),
        ffi::IO_FAILURE => Err(Error::Invalid("GLM EXL3 model I/O failed")),
        ffi::DEVICE_FAILURE => Err(Error::Invalid("GLM EXL3 device operation failed")),
        ffi::RELOAD_REQUIRED => Err(Error::Invalid("GLM EXL3 model requires reload")),
        ffi::BUSY => Err(Error::Invalid("GLM EXL3 model already loaded")),
        ffi::INVALID_STATE => Err(Error::Invalid("GLM EXL3 operation state")),
        _ => Err(Error::Invalid("GLM EXL3 bridge status")),
    }
}

struct NativeSnapshot(NonNull<ffi::Snapshot>);

impl Drop for NativeSnapshot {
    fn drop(&mut self) {
        let mut raw = self.0.as_ptr();
        // Snapshot destruction frees private CPU ownership only. It may outlive
        // the model and does not submit device work or dereference that model.
        unsafe { ffi::stark_glm_prefix_snapshot_free(&mut raw) };
    }
}

/// Private CPU-owned state tied to the instance, failure epoch and context.
/// `Rc` also keeps a deferred restore alive until its suffix is available.
pub struct PrefixSnapshot {
    native: Rc<NativeSnapshot>,
    instance: u64,
    epoch: u64,
    context_tokens: usize,
    tokens: Rc<[u32]>,
    mtp_enabled: bool,
}

impl PrefixSnapshot {
    pub fn position(&self) -> usize {
        self.tokens.len()
    }

    pub fn mtp_enabled(&self) -> bool {
        self.mtp_enabled
    }
}

struct PendingRestore {
    native: Rc<NativeSnapshot>,
    tokens: Rc<[u32]>,
}

/// The native ABI is load-thread-owned. `Rc` prevents moving or sharing this
/// owner across threads; every call that touches device state needs `&mut self`.
pub struct GlmExl3Model {
    native: NonNull<ffi::Model>,
    vocab_size: usize,
    max_context: usize,
    position: usize,
    native_started: bool,
    pending: Vec<u32>,
    instance: u64,
    epoch: u64,
    capture_at: Option<usize>,
    captured: Option<PrefixSnapshot>,
    restore: Option<PendingRestore>,
    prefix_reuse: bool,
    fenced: bool,
    native_decode_stage: bool,
    head_loaded: bool,
    mtp_enabled: bool,
    native_capabilities: u64,
    native_initial: Option<u32>,
    native_request_open: bool,
    native_decode_closed: bool,
    native_cleanup_failed: bool,
    native_stats: NativeDecodeStats,
    _load_thread: PhantomData<Rc<()>>,
}

impl GlmExl3Model {
    pub fn load(
        path: &Path,
        expected_vocab: usize,
        options: LoadOptions,
    ) -> Result<Self> {
        if expected_vocab == 0 || expected_vocab > MAX_VOCAB {
            return Err(Error::Invalid("GLM EXL3 vocabulary size"));
        }
        let path = path_bytes(path)?;
        let native_options = options.native()?;
        if options.load_mtp_head && !options.native_decode_stage {
            return Err(Error::Invalid("GLM EXL3 head requires explicit native stage"));
        }
        #[cfg(not(stark_glm_exl3_native_decode_stage))]
        if options.native_decode_stage {
            return Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"));
        }
        // ABI negotiation and rejected arguments cannot initialize a device.
        if unsafe { ffi::stark_glm_abi_version() } != ffi::ABI_VERSION {
            return Err(Error::Invalid("GLM EXL3 bridge ABI"));
        }
        let mut raw = ptr::null_mut();
        let status = if options.native_decode_stage {
            #[cfg(stark_glm_exl3_native_decode_stage)]
            {
                unsafe {
                    native_ffi::stark_glm_load_native(
                        path.as_ptr(),
                        path.len(),
                        &native_options,
                        u32::from(options.load_mtp_head),
                        &mut raw,
                    )
                }
            }
            #[cfg(not(stark_glm_exl3_native_decode_stage))]
            {
                return Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"));
            }
        } else {
            unsafe { ffi::stark_glm_load(path.as_ptr(), path.len(), &native_options, &mut raw) }
        };
        if let Err(error) = status_result(status) {
            if !raw.is_null() {
                unsafe { ffi::stark_glm_destroy(&mut raw) };
            }
            return Err(error);
        }
        let native = NonNull::new(raw).ok_or(Error::Invalid("GLM EXL3 missing model owner"))?;
        let mut model = Self {
            native,
            vocab_size: expected_vocab,
            max_context: options.context_tokens,
            position: 0,
            native_started: false,
            pending: Vec::new(),
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            capture_at: None,
            captured: None,
            restore: None,
            prefix_reuse: options.prefix_reuse,
            fenced: false,
            native_decode_stage: options.native_decode_stage,
            head_loaded: options.load_mtp_head,
            mtp_enabled: false,
            native_capabilities: 0,
            native_initial: None,
            native_request_open: false,
            native_decode_closed: false,
            native_cleanup_failed: false,
            native_stats: NativeDecodeStats::default(),
            _load_thread: PhantomData,
        };
        let info = model.info()?;
        model.native_capabilities = info.capabilities;
        if info.vocab_size as usize != expected_vocab
            || info.context_tokens != options.context_tokens as u64
            || info.position != 0
            || info.prefill_chunk_tokens as usize != PREFILL_CHUNK
            || info.capabilities & (ffi::CAP_FULL_F32_READOUT | ffi::CAP_SERIAL_DECODE)
                != (ffi::CAP_FULL_F32_READOUT | ffi::CAP_SERIAL_DECODE)
            || info.fenced != 0
            || info.reserved != 0
        {
            return Err(Error::Invalid("GLM EXL3 loaded model metadata"));
        }
        if model.prefix_reuse && info.capabilities & ffi::CAP_PREFIX_SNAPSHOT == 0 {
            return Err(Error::Unsupported("GLM EXL3 native prefix snapshots unavailable"));
        }
        Ok(model)
    }

    fn info(&mut self) -> Result<ffi::Info> {
        let mut info = ffi::Info::empty();
        let status = unsafe { ffi::stark_glm_get_info(self.native.as_ptr(), &mut info) };
        status_result(status)?;
        if info.struct_size != size_of::<ffi::Info>() as u32 || info.abi_version != ffi::ABI_VERSION {
            self.fenced = true;
            return Err(Error::Invalid("GLM EXL3 bridge metadata ABI"));
        }
        self.fenced |= info.fenced != 0;
        Ok(info)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn max_context_length(&self) -> usize {
        self.max_context
    }

    /// Includes admitted prompt tokens queued for the next full prefill.
    pub fn position(&self) -> usize {
        self.position + self.pending.len()
    }

    pub fn is_fenced(&self) -> bool {
        self.fenced
    }

    pub fn prefix_chunk_tokens(&self) -> Option<usize> {
        self.prefix_reuse.then_some(PREFILL_CHUNK)
    }

    pub fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        // Availability is not qualification. The new stage keeps these bits
        // clear until the particular native target and prefix path qualify.
        (cfg!(stark_glm_exl3_native_decode_stage)
            && self.native_decode_stage
            && self.head_loaded
            && self.native_capabilities & CAP_NATIVE_DECODE != 0
            && (!self.prefix_reuse || self.native_capabilities & CAP_MTP_PREFIX != 0))
            .then_some(MtpCapabilities {
                max_drafts: 3,
                recommended_drafts: 3,
                sampling: "native-target-draw-match-v1",
                verification: self.mtp_verification(),
            })
    }

    pub fn mtp_verification(&self) -> &'static str {
        "glm53-exl3-native-round-v1"
    }

    pub fn native_decode_stats(&self) -> NativeDecodeStats {
        self.native_stats
    }

    pub fn set_mtp_enabled(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        if !self.native_decode_stage {
            return if enabled {
                Err(Error::Unsupported("GLM EXL3 MTP unavailable"))
            } else {
                Ok(())
            };
        }
        let result = (|| {
            if self.fenced
                || self.position != 0
                || self.native_started
                || !self.pending.is_empty()
                || self.restore.is_some()
                || self.native_request_open
                || self.native_decode_closed
            {
                return Err(Error::Invalid("GLM EXL3 MTP request mode admission"));
            }
            if enabled && !self.head_loaded {
                return Err(Error::Unsupported("GLM EXL3 MTP head not loaded"));
            }
            #[cfg(stark_glm_exl3_native_decode_stage)]
            {
                status_result(unsafe {
                    native_ffi::stark_glm_set_mtp_enabled(self.native.as_ptr(), u32::from(enabled))
                })?;
                if enabled != self.mtp_enabled {
                    self.invalidate_epoch();
                }
                self.mtp_enabled = enabled;
                Ok(())
            }
            #[cfg(not(stark_glm_exl3_native_decode_stage))]
            Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"))
        })();
        if let Err(error) = result {
            return self.failed(error);
        }
        Ok(())
    }

    pub fn step_native_initial(
        &mut self,
        token: u32,
        sampling: NativeSampling,
    ) -> Result<u32> {
        let result = (|| {
            if !self.native_decode_stage {
                return Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"));
            }
            self.check_tokens(&[token])?;
            if self.native_started || self.native_initial.is_some() || !sampling.is_supported() {
                return Err(Error::Invalid("GLM EXL3 native first draw admission"));
            }
            #[cfg(stark_glm_exl3_native_decode_stage)]
            {
                let (mode, temperature, top_p, seed) = match sampling {
                    NativeSampling::Greedy {
                        seed,
                    } => (0, 0.0, 1.0, seed),
                    NativeSampling::Stochastic {
                        temperature,
                        top_p,
                        seed,
                    } => (1, temperature, top_p, seed),
                };
                let options = native_ffi::Sampling {
                    struct_size: size_of::<native_ffi::Sampling>() as u32,
                    abi_version: native_ffi::ABI_VERSION,
                    mode,
                    reserved: 0,
                    temperature,
                    top_p,
                    seed,
                };
                self.pending.try_reserve(1).map_err(|_| Error::Invalid("GLM EXL3 prompt allocation"))?;
                self.pending.push(token);
                self.forward_prompt()?;
                let mut first = 0;
                status_result(unsafe {
                    native_ffi::stark_glm_native_initial(self.native.as_ptr(), &options, &mut first)
                })?;
                let info = self.info()?;
                if first as usize >= self.vocab_size || info.position != self.position as u64 || info.fenced != 0 {
                    return Err(Error::Invalid("GLM EXL3 native first draw output"));
                }
                self.native_initial = Some(first);
                self.native_request_open = true;
                self.native_stats = NativeDecodeStats::default();
                Ok(first)
            }
            #[cfg(not(stark_glm_exl3_native_decode_stage))]
            Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"))
        })();
        match result {
            Ok(first) => Ok(first),
            Err(error) => self.failed(error),
        }
    }

    pub fn native_decode(
        &mut self,
        first: u32,
        total_budget: usize,
        drafts: usize,
    ) -> Result<NativeDecode<'_>> {
        let admission = (|| {
            if !cfg!(stark_glm_exl3_native_decode_stage) || !self.native_decode_stage {
                return Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"));
            }
            if self.fenced
                || !self.native_started
                || !self.pending.is_empty()
                || !self.native_request_open
                || self.native_decode_closed
                || self.native_initial != Some(first)
                || first as usize >= self.vocab_size
                || total_budget == 0
                || drafts > 3
            {
                return Err(Error::Invalid("GLM EXL3 native stream admission"));
            }
            if drafts > 0 && (!self.head_loaded || !self.mtp_enabled) {
                return Err(Error::Unsupported("GLM EXL3 MTP request disabled"));
            }
            admitted_end(self.position, 0, total_budget, self.max_context)?;
            Ok(())
        })();
        if let Err(error) = admission {
            return self.failed(error);
        }
        self.native_initial = None;
        let mut stream = NativeDecode {
            model: self,
            pending: first,
            total_budget,
            committed: 0,
            drafts,
            skip_initial: true,
            queue: [0; 4],
            queue_at: 0,
            queue_len: 0,
            stopped: false,
        };
        // The worker's outer once(first) can exhaust a one-token budget without
        // requesting a tail. Commit its cap-zero host round now, never in Drop.
        if total_budget == 1 {
            stream.round()?;
        }
        Ok(stream)
    }

    fn native_round(
        &mut self,
        pending: u32,
        cap: u32,
    ) -> Result<CommittedRound> {
        let result = (|| {
            #[cfg(stark_glm_exl3_native_decode_stage)]
            {
                let mut output = native_ffi::Round::empty();
                status_result(unsafe {
                    native_ffi::stark_glm_native_round(self.native.as_ptr(), pending, cap, &mut output)
                })?;
                let round = checked_round(output, pending, cap, self.position, self.vocab_size)?;
                let position = self.position + round.count;
                let info = self.info()?;
                if position > self.max_context || info.position != position as u64 || info.fenced != 0 {
                    return Err(Error::Invalid("GLM EXL3 native round position"));
                }
                self.position = position;
                Ok(round)
            }
            #[cfg(not(stark_glm_exl3_native_decode_stage))]
            {
                let _ = (pending, cap);
                Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"))
            }
        })();
        match result {
            Ok(round) => Ok(round),
            Err(error) => self.failed(error),
        }
    }

    pub fn finish_native_request(&mut self) -> Result<()> {
        if self.native_cleanup_failed {
            return Err(Error::Invalid("GLM EXL3 native request cleanup failed"));
        }
        if !self.native_decode_stage {
            return Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"));
        }
        if !self.native_request_open {
            return Ok(());
        }
        #[cfg(stark_glm_exl3_native_decode_stage)]
        let result = status_result(unsafe { native_ffi::stark_glm_native_end_request(self.native.as_ptr()) });
        #[cfg(not(stark_glm_exl3_native_decode_stage))]
        let result = Err(Error::Unsupported("GLM EXL3 native decode stage unavailable"));
        self.native_request_open = false;
        self.native_decode_closed = true;
        self.native_initial = None;
        if result.is_err() {
            self.native_cleanup_failed = true;
            self.fenced = true;
            self.invalidate_epoch();
            self.captured = None;
            self.capture_at = None;
            self.restore = None;
        }
        result
    }

    fn clear_request(&mut self) {
        self.pending.clear();
        self.capture_at = None;
        self.captured = None;
        self.restore = None;
        self.position = 0;
        self.native_started = false;
        self.native_initial = None;
        self.native_request_open = false;
        self.native_decode_closed = false;
        self.native_cleanup_failed = false;
        self.native_stats = NativeDecodeStats::default();
    }

    fn invalidate_epoch(&mut self) {
        if let Some(epoch) = self.epoch.checked_add(1) {
            self.epoch = epoch;
        } else {
            self.fenced = true;
        }
    }

    /// Reset always drains before releasing the request's private ownership.
    /// A native execution fault keeps the instance fenced until destruction.
    pub fn reset(&mut self) {
        let status = unsafe { ffi::stark_glm_reset(self.native.as_ptr()) };
        self.clear_request();
        if status != ffi::OK {
            self.invalidate_epoch();
            self.fenced = true;
        }
    }

    fn failed<T>(
        &mut self,
        error: Error,
    ) -> Result<T> {
        self.invalidate_epoch();
        self.reset();
        Err(error)
    }

    fn check_tokens(
        &self,
        tokens: &[u32],
    ) -> Result<()> {
        if self.fenced {
            return Err(Error::Invalid("GLM EXL3 model requires reload"));
        }
        if self.native_request_open || self.native_decode_closed {
            return Err(Error::Invalid("GLM EXL3 native request requires reset"));
        }
        admitted_end(self.position, self.pending.len(), tokens.len(), self.max_context)?;
        if tokens.iter().any(|&token| token as usize >= self.vocab_size) {
            return Err(Error::Invalid("GLM EXL3 token ID"));
        }
        Ok(())
    }

    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        self.ingest_prefix(&[token], || Ok(()))
    }

    /// Queue a complete known prefix. The next `step` adds its final token and
    /// runs one native batched prefill, preserving the upstream chunk schedule.
    /// The callback runs before every admitted token and once after admission.
    pub fn ingest_prefix(
        &mut self,
        tokens: &[u32],
        mut before_token: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let result = (|| {
            self.check_tokens(tokens)?;
            self.pending.try_reserve(tokens.len()).map_err(|_| Error::Invalid("GLM EXL3 prompt allocation"))?;
            for &token in tokens {
                before_token()?;
                self.pending.push(token);
            }
            before_token()
        })();
        if let Err(error) = result {
            return self.failed(error);
        }
        Ok(())
    }

    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        let result = self.forward(token);
        if let Err(error) = result {
            return self.failed(error);
        }
        result
    }

    /// Execute a nonempty remaining prompt through the same batching path as
    /// `ingest_prefix` followed by `step`. After a staged restore, pass only
    /// tokens beyond that snapshot's prefix.
    #[cfg(any(test, stark_glm_exl3_prefix_validation))]
    #[allow(dead_code, reason = "Convenience entry point for artifact qualification harnesses")]
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        mut before_token: impl FnMut() -> Result<()>,
    ) -> Result<Vec<f32>> {
        let Some((&last, prefix)) = tokens.split_last() else {
            return self.failed(Error::Invalid("GLM EXL3 empty prompt"));
        };
        self.ingest_prefix(prefix, &mut before_token)?;
        if let Err(error) = before_token() {
            return self.failed(error);
        }
        let logits = self.step(last)?;
        if let Err(error) = before_token() {
            return self.failed(error);
        }
        Ok(logits)
    }

    fn forward(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.check_tokens(&[token])?;
        self.pending.try_reserve(1).map_err(|_| Error::Invalid("GLM EXL3 prompt allocation"))?;
        self.pending.push(token);
        if !self.native_started {
            self.forward_prompt()?;
        } else {
            // Ingest after a readout is a chronological decode continuation.
            // Process every queued token; only the final vocabulary row leaves
            // the native owner.
            for &token in &self.pending {
                status_result(unsafe { ffi::stark_glm_step(self.native.as_ptr(), token) })?;
                self.position += 1;
            }
            self.pending.clear();
        }
        let info = self.info()?;
        if info.position != self.position() as u64 || info.fenced != 0 {
            return Err(Error::Invalid("GLM EXL3 state position"));
        }
        let mut logits = Vec::new();
        logits.try_reserve_exact(self.vocab_size).map_err(|_| Error::Invalid("GLM EXL3 readout allocation"))?;
        logits.resize(self.vocab_size, 0.0);
        status_result(unsafe { ffi::stark_glm_readout_f32(self.native.as_ptr(), logits.as_mut_ptr(), logits.len()) })?;
        if !super::finite_guard::all_finite(&logits) {
            return Err(Error::Invalid("GLM EXL3 nonfinite readout"));
        }
        Ok(logits)
    }

    fn forward_prompt(&mut self) -> Result<()> {
        let mut full_prompt = Vec::new();
        let prefix = self.restore.as_ref().map_or(&[][..], |restore| restore.tokens.as_ref());
        full_prompt
            .try_reserve_exact(prefix.len() + self.pending.len())
            .map_err(|_| Error::Invalid("GLM EXL3 prompt allocation"))?;
        full_prompt.extend_from_slice(prefix);
        full_prompt.extend_from_slice(&self.pending);
        if let Some(position) = self.capture_at {
            status_result(unsafe { ffi::stark_glm_capture_prefix_at(self.native.as_ptr(), position as u64) })?;
        }
        let status = if let Some(restore) = &self.restore {
            unsafe {
                ffi::stark_glm_restore_prefix(
                    self.native.as_ptr(),
                    restore.native.0.as_ptr(),
                    full_prompt.as_ptr(),
                    full_prompt.len(),
                )
            }
        } else {
            unsafe { ffi::stark_glm_prefill(self.native.as_ptr(), full_prompt.as_ptr(), full_prompt.len()) }
        };
        status_result(status)?;
        self.position = full_prompt.len();
        self.pending.clear();
        self.restore = None;
        self.native_started = true;
        if let Some(position) = self.capture_at.take().filter(|&position| position <= full_prompt.len()) {
            let mut raw = ptr::null_mut();
            status_result(unsafe { ffi::stark_glm_take_prefix_snapshot(self.native.as_ptr(), &mut raw) })?;
            // No newly captured snapshot is a normal query result. The worker
            // may retain the immediately preceding successful snapshot.
            self.captured = NonNull::new(raw).map(|native| PrefixSnapshot {
                native: Rc::new(NativeSnapshot(native)),
                instance: self.instance,
                epoch: self.epoch,
                context_tokens: self.max_context,
                tokens: Rc::from(&full_prompt[..position]),
                mtp_enabled: self.mtp_enabled,
            });
        }
        Ok(())
    }

    /// Drain without sampling or reading out token values.
    #[cfg(any(test, stark_glm_exl3_prefix_validation))]
    #[allow(dead_code, reason = "Explicit cleanup entry point for artifact qualification harnesses")]
    pub fn drain(&mut self) -> Result<()> {
        if let Err(error) = status_result(unsafe { ffi::stark_glm_drain(self.native.as_ptr()) }) {
            self.fenced = true;
            return self.failed(error);
        }
        Ok(())
    }

    /// Cancel and drain this serialized request, invalidating all its snapshots.
    /// This method never permits a concurrent caller to retain the raw owner.
    pub fn cancel(&mut self) -> Result<()> {
        self.invalidate_epoch();
        let cancel = status_result(unsafe { ffi::stark_glm_cancel(self.native.as_ptr()) });
        let drain = status_result(unsafe { ffi::stark_glm_drain(self.native.as_ptr()) });
        self.reset();
        cancel.and(drain)?;
        if self.fenced {
            return Err(Error::Invalid("GLM EXL3 model requires reload"));
        }
        Ok(())
    }

    /// Hash committed native state for private fresh/restored qualification.
    /// The ABI returns only a digest; no state payload leaves the bridge.
    #[cfg(stark_glm_exl3_prefix_validation)]
    pub fn validation_state_sha256(&mut self) -> Result<[u8; 32]> {
        let result = (|| {
            if self.fenced || !self.pending.is_empty() || self.restore.is_some() {
                return Err(Error::Invalid("GLM EXL3 state hash requires committed state"));
            }
            self.drain()?;
            let info = self.info()?;
            if info.position != self.position as u64 || info.fenced != 0 {
                return Err(Error::Invalid("GLM EXL3 state hash position"));
            }
            let mut digest = [0; 32];
            status_result(unsafe {
                ffi::stark_glm_validation_state_sha256(self.native.as_ptr(), digest.as_mut_ptr(), digest.len())
            })?;
            Ok(digest)
        })();
        if let Err(error) = result {
            return self.failed(error);
        }
        result
    }

    pub fn capture_prefix_at(
        &mut self,
        position: usize,
    ) -> Result<()> {
        let result = (|| {
            if !self.prefix_reuse {
                return Err(Error::Unsupported("GLM EXL3 prefix snapshots unavailable"));
            }
            if self.fenced
                || self.native_started
                || !self.pending.is_empty()
                || position <= self.position
                || position >= self.max_context
                || !position.is_multiple_of(PREFILL_CHUNK)
                || !self.position.is_multiple_of(PREFILL_CHUNK)
            {
                return Err(Error::Invalid("GLM EXL3 prefix capture position"));
            }
            self.capture_at = Some(position);
            self.captured = None;
            Ok(())
        })();
        if let Err(error) = result {
            return self.failed(error);
        }
        Ok(())
    }

    pub fn take_prefix_snapshot(&mut self) -> Option<PrefixSnapshot> {
        self.captured.take()
    }

    /// Stage an exact restore for the next complete prompt. The native resume
    /// also consumes the suffix, so its actual state mutation occurs in `step`.
    /// Logical position includes the staged prefix during token admission.
    /// The caller must first verify that this immediately preceding snapshot
    /// matches the new prompt's prefix, host cache scope and request options.
    pub fn restore_prefix(
        &mut self,
        snapshot: &PrefixSnapshot,
    ) -> Result<()> {
        let result = (|| {
            if !self.prefix_reuse {
                return Err(Error::Unsupported("GLM EXL3 prefix snapshots unavailable"));
            }
            if self.fenced
                || self.position != 0
                || self.native_started
                || !self.pending.is_empty()
                || self.restore.is_some()
                || self.instance != snapshot.instance
                || self.epoch != snapshot.epoch
                || self.max_context != snapshot.context_tokens
                || self.mtp_enabled != snapshot.mtp_enabled
                || snapshot.position() == 0
                || snapshot.position() >= self.max_context
                || !snapshot.position().is_multiple_of(PREFILL_CHUNK)
            {
                return Err(Error::Invalid("GLM EXL3 prefix restore admission"));
            }
            self.position = snapshot.position();
            self.restore = Some(PendingRestore {
                native: snapshot.native.clone(),
                tokens: snapshot.tokens.clone(),
            });
            Ok(())
        })();
        if let Err(error) = result {
            return self.failed(error);
        }
        Ok(())
    }
}

/// A complete request's committed native tail. The first prefill choice is
/// published by the caller; only its first committed duplicate is suppressed.
/// The next-pending draw stays private until a later round commits it.
pub struct NativeDecode<'a> {
    model: &'a mut GlmExl3Model,
    pending: u32,
    total_budget: usize,
    committed: usize,
    drafts: usize,
    skip_initial: bool,
    queue: [u32; 4],
    queue_at: usize,
    queue_len: usize,
    stopped: bool,
}

impl NativeDecode<'_> {
    fn round(&mut self) -> Result<()> {
        let cap = native_round_cap(self.total_budget, self.committed, self.drafts)?;
        let round = self.model.native_round(self.pending, cap)?;
        let committed = self
            .committed
            .checked_add(round.count)
            .filter(|&count| count <= self.total_budget)
            .ok_or(Error::Invalid("GLM EXL3 native committed budget"))?;
        self.queue_at = 0;
        self.queue_len = queue_committed(&round, self.skip_initial, &mut self.queue);
        self.skip_initial = false;
        self.pending = round.next_pending;
        self.committed = committed;
        self.model.native_stats.rounds += 1;
        self.model.native_stats.verified_drafts += round.verified_drafts;
        self.model.native_stats.accepted += round.accepted;
        self.model.native_stats.committed += round.count;
        if committed == self.total_budget {
            // Final bonus is discarded. End before publishing the final block
            // so an end/drain failure cannot be hidden by budget exhaustion.
            self.model.finish_native_request()?;
        }
        Ok(())
    }
}

impl Iterator for NativeDecode<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped {
            return None;
        }
        while self.queue_at == self.queue_len {
            if self.committed == self.total_budget {
                self.stopped = true;
                return None;
            }
            if let Err(error) = self.round() {
                self.stopped = true;
                return Some(Err(error));
            }
        }
        let token = self.queue[self.queue_at];
        self.queue_at += 1;
        Some(Ok(token))
    }
}

impl Drop for NativeDecode<'_> {
    fn drop(&mut self) {
        // EOS, text stop, cancellation and callback failure can end inside a
        // committed block. Drain and close this request; never execute another
        // round or sample. No chronological continuation is promised until
        // reset. A taken prompt snapshot remains available on successful end.
        let _ = self.model.finish_native_request();
        self.queue.fill(0);
    }
}

impl Drop for GlmExl3Model {
    fn drop(&mut self) {
        // Native destruction drains, joins readers and frees module state before
        // weights/configuration. It nulls the handle after teardown even when a
        // device drain reported a fault. The !Send owner prevents wrong-thread
        // destruction, whose ABI deliberately refuses to release ownership.
        let mut raw = self.native.as_ptr();
        unsafe { ffi::stark_glm_destroy(&mut raw) };
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn native_extension_layout_matches_agreed_version_one_contract() {
        use std::mem::{align_of, offset_of};

        assert_eq!(size_of::<native_ffi::Sampling>(), 32);
        assert_eq!(align_of::<native_ffi::Sampling>(), 8);
        assert_eq!(offset_of!(native_ffi::Sampling, temperature), 16);
        assert_eq!(offset_of!(native_ffi::Sampling, seed), 24);
        assert_eq!(size_of::<native_ffi::Round>(), 48);
        assert_eq!(align_of::<native_ffi::Round>(), 8);
        assert_eq!(offset_of!(native_ffi::Round, tokens), 24);
        assert_eq!(offset_of!(native_ffi::Round, position), 40);
    }

    #[stark_test]
    fn native_sampling_rejects_nonfinite_and_implicit_greedy_temperatures() {
        assert!(
            NativeSampling::Greedy {
                seed: u64::MAX
            }
            .is_supported()
        );
        for temperature in [0.01, 0.7, 100.0] {
            assert!(
                NativeSampling::Stochastic {
                    temperature,
                    top_p: 0.95,
                    seed: 1
                }
                .is_supported()
            );
        }
        for temperature in [f32::NAN, f32::INFINITY, -1.0, 0.0, 0.005, 101.0] {
            assert!(
                !NativeSampling::Stochastic {
                    temperature,
                    top_p: 0.95,
                    seed: 1
                }
                .is_supported()
            );
        }
        for top_p in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1.01] {
            assert!(
                !NativeSampling::Stochastic {
                    temperature: 1.0,
                    top_p,
                    seed: 1
                }
                .is_supported()
            );
        }
    }

    fn valid_native_round() -> native_ffi::Round {
        let mut round = native_ffi::Round::empty();
        round.count = 2;
        round.accepted = 1;
        round.verified_drafts = 2;
        round.tokens = [7, 8, 0, 0];
        round.next_pending = 9;
        round.position = 102;
        round
    }

    #[stark_test]
    fn native_round_output_rejects_bad_abi_counts_positions_or_ids() {
        assert!(checked_round(valid_native_round(), 7, 2, 100, 16).is_ok());
        let invalid: [fn(&mut native_ffi::Round); 10] = [
            |r| r.struct_size = 0,
            |r| r.abi_version = 0,
            |r| r.count = 0,
            |r| r.count = 5,
            |r| r.accepted = 3,
            |r| r.verified_drafts = 3,
            |r| r.tokens[0] = 6,
            |r| r.tokens[1] = 16,
            |r| r.next_pending = 16,
            |r| r.position = 103,
        ];
        for change in invalid {
            let mut round = valid_native_round();
            change(&mut round);
            assert!(checked_round(round, 7, 2, 100, 16).is_err());
        }
        assert!(native_round_cap(0, 0, 3).is_err());
        assert!(native_round_cap(1, 1, 3).is_err());
        assert!(native_round_cap(8, 0, 4).is_err());
    }

    #[stark_test]
    fn native_budget_geometry_and_initial_dedup_match_host_vectors() {
        // Independent host arithmetic oracle: budgets 1..8, forced accepted
        // count 0..3. Initial publication does not consume native commitment.
        let expected: [&[&[usize]]; 4] = [
            &[
                &[1],
                &[1, 1],
                &[1, 1, 1],
                &[1, 1, 1, 1],
                &[1, 1, 1, 1, 1],
                &[1, 1, 1, 1, 1, 1],
                &[1, 1, 1, 1, 1, 1, 1],
                &[1, 1, 1, 1, 1, 1, 1, 1],
            ],
            &[&[1], &[2], &[2, 1], &[2, 2], &[2, 2, 1], &[2, 2, 2], &[2, 2, 2, 1], &[2, 2, 2, 2]],
            &[&[1], &[2], &[3], &[3, 1], &[3, 2], &[3, 3], &[3, 3, 1], &[3, 3, 2]],
            &[&[1], &[2], &[3], &[4], &[4, 1], &[4, 2], &[4, 3], &[4, 4]],
        ];
        for (accepted, oracle) in expected.into_iter().enumerate() {
            for budget in 1..=8 {
                let mut committed = 0;
                let mut skip_initial = true;
                let mut widths = Vec::new();
                let mut visible = vec![0];
                while committed < budget {
                    let cap = native_round_cap(budget, committed, 3).unwrap();
                    let kept = accepted.min(cap as usize);
                    let mut output = native_ffi::Round::empty();
                    output.count = (kept + 1) as u32;
                    output.accepted = kept as u32;
                    output.verified_drafts = cap;
                    output.next_pending = (committed + kept + 1) as u32;
                    output.position = (100 + committed + kept + 1) as u64;
                    for i in 0..=kept {
                        output.tokens[i] = (committed + i) as u32;
                    }
                    let round = checked_round(output, committed as u32, cap, 100 + committed, 32).unwrap();
                    let mut queue = [0; 4];
                    let count = queue_committed(&round, skip_initial, &mut queue);
                    visible.extend_from_slice(&queue[..count]);
                    skip_initial = false;
                    committed += round.count;
                    widths.push(round.count);
                }
                assert!(widths == oracle[budget - 1], "native call geometry changed");
                assert!(visible == (0..budget as u32).collect::<Vec<_>>(), "committed prefix or initial dedup changed");
            }
        }
    }

    #[stark_test]
    fn allocation_policy_rejects_unbounded_or_overfilled_requests_before_native_load() {
        assert!(LoadOptions::new(0).native().is_err());
        assert!(LoadOptions::new(MAX_CONTEXT + 1).native().is_err());
        let mut options = LoadOptions::new(32_768);
        assert!(!options.prefix_reuse);
        assert!(options.native().is_ok());
        options.gpu_ceiling_bytes = MAX_GPU_BYTES + 1;
        assert!(options.native().is_err());
        options.gpu_ceiling_bytes = MAX_GPU_BYTES;
        options.wired_margin_bytes = MAX_GPU_BYTES;
        assert!(options.native().is_err());
        options.wired_margin_bytes = WIRED_MARGIN_BYTES;
        options.expert_prefill_rows = 289;
        assert!(options.native().is_err());
    }

    #[stark_test]
    fn context_admission_checks_staged_prefix_pending_tokens_and_overflow() {
        assert_eq!(admitted_end(16_384, 15, 1, 16_400).unwrap(), 16_400);
        assert!(admitted_end(16_384, 15, 2, 16_400).is_err());
        assert!(admitted_end(usize::MAX, 1, 0, usize::MAX).is_err());
        assert!(admitted_end(usize::MAX - 1, 1, 1, usize::MAX).is_err());
    }

    #[stark_test]
    fn model_path_requires_absolute_utf8_without_interior_nul() {
        assert!(path_bytes(Path::new("relative/model")).is_err());
        assert!(path_bytes(Path::new("/private/model\0suffix")).is_err());
        assert_eq!(path_bytes(Path::new("/private/model")).unwrap(), b"/private/model");
        let too_long = format!("/{}", "x".repeat(4096));
        assert!(path_bytes(Path::new(&too_long)).is_err());
    }
}
