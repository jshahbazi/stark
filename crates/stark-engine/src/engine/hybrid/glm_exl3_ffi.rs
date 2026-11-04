// SPDX-License-Identifier: Apache-2.0
//! Private declarations for Stark's versioned, serial GLM bridge ABI.

use std::ffi::c_void;

pub(super) const ABI_VERSION: u32 = 1;
pub(super) const CAP_FULL_F32_READOUT: u64 = 1;
pub(super) const CAP_SERIAL_DECODE: u64 = 2;

pub(super) type Model = c_void;
pub(super) type Snapshot = c_void;

pub(super) const OK: u32 = 0;
pub(super) const INVALID_ARGUMENT: u32 = 1;
pub(super) const ABI_MISMATCH: u32 = 2;
pub(super) const CONTEXT_LIMIT: u32 = 3;
pub(super) const CANCELLED: u32 = 4;
pub(super) const UNSUPPORTED: u32 = 5;
pub(super) const OUT_OF_MEMORY: u32 = 6;
pub(super) const IO_FAILURE: u32 = 7;
pub(super) const DEVICE_FAILURE: u32 = 8;
pub(super) const RELOAD_REQUIRED: u32 = 9;
pub(super) const BUSY: u32 = 10;
pub(super) const INVALID_STATE: u32 = 11;

#[repr(C)]
pub(super) struct LoadOptions {
    pub struct_size: u32,
    pub abi_version: u32,
    pub context_tokens: u64,
    pub gpu_ceiling_bytes: u64,
    pub wired_margin_bytes: u64,
    pub expert_rows: u32,
    pub expert_prefill_rows: u32,
}

#[repr(C)]
pub(super) struct Info {
    pub struct_size: u32,
    pub abi_version: u32,
    pub vocab_size: u32,
    pub prefill_chunk_tokens: u32,
    pub context_tokens: u64,
    pub position: u64,
    pub capabilities: u64,
    pub fenced: u32,
    pub reserved: u32,
}

impl Info {
    pub fn empty() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION,
            vocab_size: 0,
            prefill_chunk_tokens: 0,
            context_tokens: 0,
            position: 0,
            capabilities: 0,
            fenced: 0,
            reserved: 0,
        }
    }
}

unsafe extern "C" {
    pub(super) fn stark_glm_abi_version() -> u32;
    pub(super) fn stark_glm_load(
        path_utf8: *const u8,
        path_len: usize,
        options: *const LoadOptions,
        out_model: *mut *mut Model,
    ) -> u32;
    pub(super) fn stark_glm_destroy(model: *mut *mut Model) -> u32;
    pub(super) fn stark_glm_reset(model: *mut Model) -> u32;
    pub(super) fn stark_glm_drain(model: *mut Model) -> u32;
    pub(super) fn stark_glm_cancel(model: *mut Model) -> u32;
    pub(super) fn stark_glm_prefill(
        model: *mut Model,
        token_ids: *const u32,
        count: usize,
    ) -> u32;
    pub(super) fn stark_glm_step(
        model: *mut Model,
        token_id: u32,
    ) -> u32;
    pub(super) fn stark_glm_readout_f32(
        model: *mut Model,
        dst: *mut f32,
        count: usize,
    ) -> u32;
    pub(super) fn stark_glm_get_info(
        model: *mut Model,
        info: *mut Info,
    ) -> u32;
    pub(super) fn stark_glm_capture_prefix_at(
        model: *mut Model,
        position: u64,
    ) -> u32;
    pub(super) fn stark_glm_take_prefix_snapshot(
        model: *mut Model,
        out: *mut *mut Snapshot,
    ) -> u32;
    pub(super) fn stark_glm_restore_prefix(
        model: *mut Model,
        snapshot: *const Snapshot,
        full_prompt: *const u32,
        count: usize,
    ) -> u32;
    pub(super) fn stark_glm_prefix_snapshot_free(snapshot: *mut *mut Snapshot) -> u32;
    #[cfg(stark_glm_exl3_prefix_validation)]
    pub(super) fn stark_glm_validation_state_sha256(
        model: *mut Model,
        dst: *mut u8,
        count: usize,
    ) -> u32;
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn abi_layout_matches_version_one_header() {
        assert_eq!(size_of::<LoadOptions>(), 40);
        assert_eq!(align_of::<LoadOptions>(), 8);
        assert_eq!(offset_of!(LoadOptions, context_tokens), 8);
        assert_eq!(offset_of!(LoadOptions, expert_rows), 32);
        assert_eq!(size_of::<Info>(), 48);
        assert_eq!(align_of::<Info>(), 8);
        assert_eq!(offset_of!(Info, context_tokens), 16);
        assert_eq!(offset_of!(Info, capabilities), 32);
        assert_eq!(offset_of!(Info, fenced), 40);
    }
}
