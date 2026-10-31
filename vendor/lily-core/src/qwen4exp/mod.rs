// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Qwen3.8-Flash-Next in-process model core.
//! Modified by Stark: diagnostic/replay tools and disk expert services omitted.
pub mod config;
mod host_spec;
#[cfg(debug_assertions)]
mod numerical_trace;
#[cfg(debug_assertions)]
pub use numerical_trace::{NumericalDifference, NumericalTrace};
pub mod model;
pub mod ngram;
pub mod positions;
mod spec;
pub use host_spec::HostVerification;
mod vision;
mod vision_weights;
pub mod weights;
pub use model::{
    DecodeState, Encoder, ImageEmbeds, Qwen4ExpModel, Scratch, VisionInput,
};
pub use ngram::NgramStorage;
pub use positions::{ImageSpan, Positions, positions_for_prompt};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertCacheStats;
