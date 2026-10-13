//! Serialized, private JSONL text generation using a fresh native state per request.
mod generation;
#[cfg(feature = "metal")]
mod native;
pub mod protocol;
