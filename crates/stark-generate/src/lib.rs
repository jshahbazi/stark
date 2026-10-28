//! Serialized, private JSONL text generation using a fresh native state per request.
mod cancel;
mod generation;
#[cfg(feature = "metal")]
mod hybrid;
mod media;
#[cfg(any(feature = "metal", test))]
mod model_stops;
#[cfg(feature = "metal")]
mod native;
pub mod protocol;
mod text_stop;
