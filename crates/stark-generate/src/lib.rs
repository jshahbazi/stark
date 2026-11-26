//! Serialized, private JSONL text generation. Each request uses a fresh native
//! state unless the host enables consecutive prefix reuse.
mod cancel;
mod generation;
#[cfg(all(feature = "hy4", target_os = "macos", target_arch = "aarch64"))]
mod hy4;
#[cfg(feature = "metal")]
mod hybrid;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod inkling;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod inkling_artifact;
mod media;
#[cfg(any(feature = "metal", test))]
mod model_stops;
#[cfg(feature = "metal")]
mod native;
mod prefix_cache;
pub mod protocol;
mod text_stop;
