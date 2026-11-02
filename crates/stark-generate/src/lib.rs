//! Serialized, private JSONL text generation. Each request uses a fresh native
//! state unless the host enables consecutive prefix reuse.
mod cancel;
mod generation;
#[cfg(feature = "metal")]
mod hybrid;
mod media;
#[cfg(any(feature = "metal", test))]
mod model_stops;
#[cfg(feature = "metal")]
mod native;
mod prefix_cache;
pub mod protocol;
mod text_stop;
