// SPDX-License-Identifier: Apache-2.0
//! Opt-in Metal stage attribution for the connected grouped prefill graph.

use metal::{
    MTL_COUNTER_ERROR_VALUE, MTLCommonCounterSet, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSampleBufferExt, MTLCounterSamplingPoint, MTLCounterSetExt,
    MTLStorageMode,
};
use serde_json::{Value, json};

use super::*;

// Six encoder transitions per visit and one initial encoder. The M5 Max
// supports at most 4096 samples in a buffer.
const SAMPLE_COUNT: usize = (1 + PHYSICAL.len() * 6) * 2;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::engine::inkling_slice) enum PrefillPhase {
    Other,
    AttentionDense,
    RelativeCacheAttention,
    Experts,
}

struct Span {
    phase: PrefillPhase,
    first_dispatch: usize,
    active: bool,
}

#[derive(Default)]
struct Totals {
    attention_dense: f64,
    relative_cache_attention: f64,
    experts: f64,
    other: f64,
}

impl Totals {
    fn record(
        &mut self,
        phase: PrefillPhase,
        ms: f64,
    ) {
        *match phase {
            PrefillPhase::Other => &mut self.other,
            PrefillPhase::AttentionDense => &mut self.attention_dense,
            PrefillPhase::RelativeCacheAttention => &mut self.relative_cache_attention,
            PrefillPhase::Experts => &mut self.experts,
        } += ms;
    }

    fn sum(&self) -> f64 {
        self.attention_dense + self.relative_cache_attention + self.experts + self.other
    }

    fn report(
        &self,
        gpu_ms: f64,
    ) -> Value {
        json!({
            "attention_dense_ms": self.attention_dense,
            "relative_cache_attention_ms": self.relative_cache_attention,
            "experts_ms": self.experts,
            "other_ms": self.other,
            "command_gap_ms": (gpu_ms - self.sum()).max(0.0),
            "whole_command_gpu_ms": gpu_ms,
        })
    }
}

pub(in crate::engine::inkling_slice) struct PrefillTiming {
    device: Retained<Device>,
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    spans: Vec<Span>,
    start: (u64, u64),
    tile: Totals,
    totals: Totals,
}

impl PrefillTiming {
    fn new(runtime: &Runtime) -> Result<Self> {
        let device = &runtime.device;
        if !runtime.pinned_m5 || !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err("Inkling prefill profiling requires M5 Max stage timestamps");
        }
        let timestamp = MTLCommonCounterSet::Timestamp.name();
        let counters = device
            .counter_sets()
            .ok_or("Metal counters unavailable")?
            .into_iter()
            .find(|counter| counter.name() == timestamp)
            .ok_or("Metal timestamp counter unavailable")?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(Some(&counters));
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(SAMPLE_COUNT);
        let samples = device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|_| "Inkling timestamp buffer allocation failed")?;
        Ok(Self {
            device: device.clone(),
            samples,
            spans: Vec::with_capacity(SAMPLE_COUNT / 2),
            start: (0, 0),
            tile: Totals::default(),
            totals: Totals::default(),
        })
    }

    pub(in crate::engine::inkling_slice) fn begin(
        &mut self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Encoder> {
        self.spans.clear();
        self.encoder(command, 0, PrefillPhase::Other).ok_or("Inkling timestamped encoder allocation failed")
    }

    fn encoder(
        &mut self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        first_dispatch: usize,
        phase: PrefillPhase,
    ) -> Option<Encoder> {
        let index = self.spans.len() * 2;
        if index + 1 >= SAMPLE_COUNT {
            return None;
        }
        let descriptor = MTLComputePassDescriptor::new();
        let attachment = descriptor.sample_buffer_attachments().object_at_indexed_subscript(0);
        attachment.set_sample_buffer(Some(&self.samples));
        attachment.set_start_of_encoder_sample_index(index);
        attachment.set_end_of_encoder_sample_index(index + 1);
        let encoder = command.compute_command_encoder_with_descriptor(&descriptor)?;
        self.spans.push(Span {
            phase,
            first_dispatch,
            active: false,
        });
        Some(encoder)
    }

    pub(in crate::engine::inkling_slice) fn transition(
        &mut self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: &mut Option<Encoder>,
        dispatches: usize,
        phase: PrefillPhase,
    ) {
        let span = self.spans.last_mut().expect("profile started");
        if span.phase == phase {
            return;
        }
        if let Some(previous) = encoder.take() {
            span.active = dispatches > span.first_dispatch;
            previous.end_encoding();
            *encoder = self.encoder(command, dispatches, phase);
        }
    }

    pub(in crate::engine::inkling_slice) fn before_submit(
        &mut self,
        dispatches: usize,
    ) {
        let span = self.spans.last_mut().expect("profile started");
        span.active = dispatches > span.first_dispatch;
        self.device.sample_timestamps_gpu_timestamp(&mut self.start.0, &mut self.start.1);
    }

    pub(in crate::engine::inkling_slice) fn complete(
        &mut self,
        timing: &Timing,
    ) -> Result<()> {
        let (mut cpu_end, mut gpu_end) = (0, 0);
        self.device.sample_timestamps_gpu_timestamp(&mut cpu_end, &mut gpu_end);
        if cpu_end <= self.start.0 || gpu_end <= self.start.1 || !timing.gpu_ms.is_finite() || timing.gpu_ms <= 0.0 {
            return Err("Inkling timestamp calibration failed");
        }
        // Paired CPU timestamps are nanoseconds. Convert GPU counter ticks using this command's measured ratio.
        let ms_per_tick = (cpu_end - self.start.0) as f64 / (gpu_end - self.start.1) as f64 * 1e-6;
        let count = self.spans.len() * 2;
        let bytes = self.samples.resolve_counter_range(0..count).ok_or("Inkling timestamp resolve failed")?;
        if bytes.len() != count * size_of::<u64>() {
            return Err("Inkling timestamp payload size invalid");
        }
        let mut tile = Totals::default();
        for (span, samples) in self.spans.iter().zip(bytes.as_chunks::<16>().0) {
            if !span.active {
                continue;
            }
            let [start, end] = samples.as_chunks::<8>().0 else {
                unreachable!("two timestamps per span")
            };
            let (start, end) = (u64::from_ne_bytes(*start), u64::from_ne_bytes(*end));
            if start == 0 || end < start || start == MTL_COUNTER_ERROR_VALUE || end == MTL_COUNTER_ERROR_VALUE {
                return Err("Inkling timestamp sample invalid");
            }
            let ms = (end - start) as f64 * ms_per_tick;
            tile.record(span.phase, ms);
            self.totals.record(span.phase, ms);
        }
        if tile.sum() > timing.gpu_ms + (timing.gpu_ms * 0.01).max(0.001) {
            return Err("Inkling stage timestamps exceed command GPU time");
        }
        self.tile = tile;
        Ok(())
    }
}

/// Attribute cold grouped prefill to GPU stages using the pinned token fixture.
pub fn profile_prefill(
    checkpoint: &Path,
    oracle: &Path,
    count: usize,
    source_geometry: bool,
) -> Result<Value> {
    let first = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let prefix = first.values::<u32>("start", "U32", &[1])?[0] as usize;
    let tile_rows = if source_geometry {
        prefill_trace::CHUNK
    } else {
        BATCHED_PREFILL_ROWS
    };
    if !(2..=prefix).contains(&count) || count % tile_rows == 1 || (source_geometry && count % tile_rows != 0) {
        return Err("Inkling profile requires grouped prefill tiles with at least two rows each");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prefix + 4])?;
    if tokens[..=count].iter().any(|&token| token as usize >= VOCABULARY) {
        return Err("Inkling token is outside the vocabulary");
    }
    let mut session = Session::load(checkpoint, prefix + 4)?;
    let mut profiler = PrefillTiming::new(&session.runtime)?;
    let allocated_before = session.runtime.device.current_allocated_size();
    if source_geometry {
        session.prefill_lanes = Some(PrefillTraceMode::Routed.lanes(&mut session.runtime)?);
    } else {
        session.set_batched_prefill(true)?;
    }
    let allocated = session.runtime.device.current_allocated_size().saturating_sub(allocated_before);
    let baseline_start = std::time::Instant::now();
    session.ingest_many(&tokens[..count])?;
    let baseline_ms = baseline_start.elapsed().as_secs_f64() * 1000.0;
    let expected = session.step(tokens[count])?;
    session.reset();

    let (mut gpu_ms, mut wall_ms, mut dispatches) = (0.0, 0.0, 0);
    let mut tiles = Vec::new();
    for chunk in tokens[..count].chunks(tile_rows) {
        let lanes = session.prefill_lanes.as_ref().ok_or("missing grouped prefill lanes")?;
        let inputs = chunk.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
        let timing = execute_with_profile(&session.runtime, Some(&mut profiler), |gpu| {
            session.encode_prefill_rows(gpu, lanes, &inputs, false);
        })?;
        tiles.push(json!({
            "start_position": session.position,
            "rows": inputs.len(),
            "stages": profiler.tile.report(timing.gpu_ms),
        }));
        gpu_ms += timing.gpu_ms;
        wall_ms += timing.wall_ms;
        dispatches += timing.dispatches;
        session.finish_append(Ok(timing), chunk.len())?;
    }
    let actual = session.step(tokens[count])?;
    Ok(json!({
        "input_tokens": count,
        "context_tokens": session.context(),
        "tile_rows": tile_rows,
        "source_geometry": source_geometry,
        "prefill_additional_metal_allocated_bytes": allocated,
        "uninstrumented_ingest_wall_ms": baseline_ms,
        "instrumented_command_wall_ms": wall_ms,
        "dispatches": dispatches,
        "completion_waits": tiles.len(),
        "exact_continuation_logits": actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "stages": profiler.totals.report(gpu_ms),
        "tiles": tiles,
    }))
}
