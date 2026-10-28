// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Lily Qwen core vendored for in-process use by Stark.
//! Modified by Stark: the crate exposes no server, CLI, persistence or probes.
//! See UPSTREAM.md for source revision, scope and patch provenance.
pub mod config;
#[cfg(target_os = "macos")]
pub mod engine;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod moe_ffn;
#[cfg(target_os = "macos")]
pub mod qwen4exp;
pub mod safetensors;
pub mod sha256;
#[cfg(target_os = "macos")]
mod stats;
#[cfg(target_os = "macos")]
pub mod tensor;
#[cfg(target_os = "macos")]
pub mod weights;
#[cfg(target_os = "macos")]
mod ablate {
    #[derive(Clone, Copy)]
    pub(crate) enum Group {
        Hc,
        Inject,
        GdnProj,
        GdnStep,
        Attn,
        Head,
        Ple,
        Moe,
        Router,
        Shared,
        Topk,
        Gather,
        Down,
    }
    #[inline(always)]
    pub(crate) const fn on(_: Group) -> bool {
        false
    }
}
