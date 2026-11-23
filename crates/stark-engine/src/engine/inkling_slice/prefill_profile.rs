// SPDX-License-Identifier: Apache-2.0
//! Opt-in Metal stage attribution for grouped prefill and single-token decode.

use metal::{
    MTL_COUNTER_ERROR_VALUE, MTLCommonCounterSet, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSampleBufferExt, MTLCounterSamplingPoint, MTLCounterSetExt,
    MTLStorageMode,
};
use serde_json::{Value, json};

use super::*;

// Both stock bulk commands fit in this buffer, including the global subtiles.
const SAMPLE_COUNT: usize = 2048;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::engine::inkling_slice) enum PrefillPhase {
    Other,
    AttentionInputs,
    ExpertInputs,
    ExpertFinish,
    AttentionDense,
    RelativeCacheAttention,
    Experts,
    Nvfp4Quant,
    Nvfp4UpDot,
    Nvfp4DownDot,
    RouterProjection,
    RouteSelect,
    Nvfp4Epilogue,
    Nvfp4ProbeExact,
    Nvfp4ProbeGuard,
    Nvfp4ProbeBare,
    Mxfp8,
    Mxfp8Up,
    Mxfp8Down,
    Mxfp8Replay,
    KeyValue,
    RelativeTable,
    LocalAttention,
    GlobalAttention,
    GlobalPacking,
    GlobalMask,
    GlobalGate,
    ExpertCombine,
    Readout,
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
    nvfp4_quant: f64,
    nvfp4_up_dot: f64,
    nvfp4_down_dot: f64,
    router_projection: f64,
    route_select: f64,
    nvfp4_epilogue: f64,
    probe_exact: f64,
    probe_guard: f64,
    probe_bare: f64,
    mxfp8: f64,
    mxfp8_up: f64,
    mxfp8_down: f64,
    mxfp8_replay: f64,
    other: f64,
    attention_inputs: f64,
    expert_inputs: f64,
    expert_finish: f64,
    key_value: f64,
    relative_table: f64,
    local_attention: f64,
    global_attention: f64,
    global_packing: f64,
    global_mask: f64,
    global_gate: f64,
    expert_combine: f64,
    readout: f64,
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
            PrefillPhase::Nvfp4Quant => &mut self.nvfp4_quant,
            PrefillPhase::Nvfp4UpDot => &mut self.nvfp4_up_dot,
            PrefillPhase::Nvfp4DownDot => &mut self.nvfp4_down_dot,
            PrefillPhase::RouterProjection => &mut self.router_projection,
            PrefillPhase::RouteSelect => &mut self.route_select,
            PrefillPhase::Nvfp4Epilogue => &mut self.nvfp4_epilogue,
            PrefillPhase::Nvfp4ProbeExact => &mut self.probe_exact,
            PrefillPhase::Nvfp4ProbeGuard => &mut self.probe_guard,
            PrefillPhase::Nvfp4ProbeBare => &mut self.probe_bare,
            PrefillPhase::Mxfp8 => &mut self.mxfp8,
            PrefillPhase::Mxfp8Up => &mut self.mxfp8_up,
            PrefillPhase::Mxfp8Down => &mut self.mxfp8_down,
            PrefillPhase::Mxfp8Replay => &mut self.mxfp8_replay,
            PrefillPhase::AttentionInputs => &mut self.attention_inputs,
            PrefillPhase::ExpertInputs => &mut self.expert_inputs,
            PrefillPhase::ExpertFinish => &mut self.expert_finish,
            PrefillPhase::KeyValue => &mut self.key_value,
            PrefillPhase::RelativeTable => &mut self.relative_table,
            PrefillPhase::LocalAttention => &mut self.local_attention,
            PrefillPhase::GlobalAttention => &mut self.global_attention,
            PrefillPhase::GlobalPacking => &mut self.global_packing,
            PrefillPhase::GlobalMask => &mut self.global_mask,
            PrefillPhase::GlobalGate => &mut self.global_gate,
            PrefillPhase::ExpertCombine => &mut self.expert_combine,
            PrefillPhase::Readout => &mut self.readout,
        } += ms;
    }

    fn sum(&self) -> f64 {
        self.attention_dense
            + self.attention()
            + self.nvfp4()
            + self.mxfp8
            + self.mxfp8_up
            + self.mxfp8_down
            + self.mxfp8_replay
            + self.probe_exact
            + self.probe_guard
            + self.probe_bare
            + self.attention_inputs
            + self.expert_inputs
            + self.expert_finish
            + self.expert_combine
            + self.readout
            + self.other
    }

    fn attention(&self) -> f64 {
        self.relative_cache_attention
            + self.key_value
            + self.relative_table
            + self.local_attention
            + self.global_attention
            + self.global_packing
            + self.global_mask
            + self.global_gate
    }

    fn nvfp4(&self) -> f64 {
        self.experts
            + self.router_projection
            + self.route_select
            + self.nvfp4_quant
            + self.nvfp4_up_dot
            + self.nvfp4_down_dot
            + self.nvfp4_epilogue
    }

    fn report(
        &self,
        gpu_ms: f64,
    ) -> Value {
        json!({
            "attention_dense_ms": self.attention_dense,
            "relative_cache_attention_ms": self.attention(),
            "kv_conv_norm_and_cache_ms": self.key_value,
            "relative_table_ms": self.relative_table,
            "local_attention_ms": self.local_attention,
            "global_attention_ms": self.global_attention,
            "global_query_kv_conversion_ms": self.global_packing,
            "global_mask_ms": self.global_mask,
            "global_gate_ms": self.global_gate,
            "experts_ms": self.nvfp4() + self.mxfp8 + self.mxfp8_up + self.mxfp8_down + self.mxfp8_replay + self.expert_combine,
            "nvfp4_experts_ms": self.nvfp4(),
            "nvfp4_routing_ms": self.experts + self.router_projection + self.route_select,
            "nvfp4_router_projection_ms": self.router_projection,
            "nvfp4_route_selection_and_weights_ms": self.route_select,
            "nvfp4_quant_ms": self.nvfp4_quant,
            "nvfp4_dot_ms": self.nvfp4_up_dot + self.nvfp4_down_dot,
            "nvfp4_gate_up_dot_ms": self.nvfp4_up_dot,
            "nvfp4_down_dot_ms": self.nvfp4_down_dot,
            "nvfp4_epilogue_ms": self.nvfp4_epilogue,
            "mxfp8_experts_ms": self.mxfp8 + self.mxfp8_up + self.mxfp8_down + self.mxfp8_replay,
            "mxfp8_replay_ms": self.mxfp8_replay,
            "mxfp8_up_ms": self.mxfp8_up,
            "mxfp8_down_ms": self.mxfp8_down,
            "nvfp4_probe_exact_ms": self.probe_exact,
            "nvfp4_probe_guard_only_ms": self.probe_guard,
            "nvfp4_probe_bare_ms": self.probe_bare,
            "other_ms": self.other + self.readout + self.attention_inputs + self.expert_inputs + self.expert_finish,
            "attention_inputs_ms": self.attention_inputs,
            "expert_inputs_ms": self.expert_inputs,
            "expert_finish_ms": self.expert_finish,
            "expert_combine_ms": self.expert_combine,
            "embedding_readout_ms": self.other + self.readout,
            "readout_ms": self.readout,
            "command_gap_ms": (gpu_ms - self.sum()).max(0.0),
            "whole_command_gpu_ms": gpu_ms,
        })
    }
}

pub(in crate::engine::inkling_slice) struct PrefillTiming {
    device: Retained<Device>,
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    spans: Vec<Span>,
    pub(super) fold_verify_phases: bool,
    pub(super) split_verify_experts: bool,
    start: (u64, u64),
    /// Dispatches per encoded span of the last completed command, with its phase.
    pub(super) span_dispatches: Vec<(PrefillPhase, usize)>,
    submitted_dispatches: usize,
    tile: Totals,
    totals: Totals,
}

impl PrefillTiming {
    pub(super) fn new(runtime: &Runtime) -> Result<Self> {
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
            fold_verify_phases: false,
            split_verify_experts: false,
            start: (0, 0),
            span_dispatches: Vec::new(),
            submitted_dispatches: 0,
            tile: Totals::default(),
            totals: Totals::default(),
        })
    }

    pub(in crate::engine::inkling_slice) fn command_report(
        &self,
        gpu_ms: f64,
    ) -> Value {
        self.tile.report(gpu_ms)
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
        let phase = if self.split_verify_experts {
            match phase {
                PrefillPhase::Experts
                | PrefillPhase::RouterProjection
                | PrefillPhase::RouteSelect
                | PrefillPhase::Nvfp4Quant
                | PrefillPhase::Nvfp4UpDot
                | PrefillPhase::Nvfp4DownDot
                | PrefillPhase::Nvfp4Epilogue
                | PrefillPhase::ExpertCombine
                | PrefillPhase::ExpertFinish
                | PrefillPhase::Mxfp8
                | PrefillPhase::Mxfp8Up
                | PrefillPhase::Mxfp8Down => phase,
                _ => PrefillPhase::Other,
            }
        } else if self.fold_verify_phases {
            match phase {
                PrefillPhase::KeyValue
                | PrefillPhase::RelativeTable
                | PrefillPhase::LocalAttention
                | PrefillPhase::GlobalAttention => PrefillPhase::RelativeCacheAttention,
                PrefillPhase::RouterProjection
                | PrefillPhase::RouteSelect
                | PrefillPhase::Nvfp4Quant
                | PrefillPhase::Nvfp4UpDot
                | PrefillPhase::Nvfp4DownDot
                | PrefillPhase::Nvfp4Epilogue
                | PrefillPhase::ExpertCombine
                | PrefillPhase::Mxfp8
                | PrefillPhase::Mxfp8Up
                | PrefillPhase::Mxfp8Down => PrefillPhase::Experts,
                other => other,
            }
        } else {
            phase
        };
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
        self.submitted_dispatches = dispatches;
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
        self.span_dispatches = self
            .spans
            .iter()
            .enumerate()
            .map(|(index, span)| {
                let end = self.spans.get(index + 1).map_or(self.submitted_dispatches, |next| next.first_dispatch);
                (span.phase, end - span.first_dispatch)
            })
            .collect();
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
    profile(checkpoint, oracle, count, source_geometry, None)
}

/// Attribute the packed prefill candidate to the same GPU stages.
pub fn profile_packed_prefill(
    checkpoint: &Path,
    oracle: &Path,
    count: usize,
    variant: &str,
) -> Result<Value> {
    profile(checkpoint, oracle, count, false, Some(variant))
}

fn profile(
    checkpoint: &Path,
    oracle: &Path,
    count: usize,
    source_geometry: bool,
    packed_variant: Option<&str>,
) -> Result<Value> {
    let first = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let prefix = first.values::<u32>("start", "U32", &[1])?[0] as usize;
    if !(2..=prefix).contains(&count) {
        return Err("Inkling profile requires grouped prefill tiles with at least two rows each");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prefix + 4])?;
    if tokens[..=count].iter().any(|&token| token as usize >= VOCABULARY) {
        return Err("Inkling token is outside the vocabulary");
    }
    let mut session = Session::load(checkpoint, prefix + 4)?;
    let mut profiler = PrefillTiming::new(&session.runtime)?;
    let allocated_before = session.runtime.device.current_allocated_size();
    if let Some(variant) = packed_variant {
        session.packed_prefill = Some(PackedPrefill::diagnostic(&mut session.runtime, variant)?);
    } else if source_geometry {
        session.prefill_lanes = Some(PrefillTraceMode::Routed.lanes(&mut session.runtime)?);
    } else {
        session.prefill_lanes = Some(PrefillLanes::for_m5_worker(&mut session.runtime)?);
    }
    let tile_rows = session.prefill_batch_rows();
    let tail = count % tile_rows;
    if tail == 1
        || (source_geometry && tail != 0)
        || (packed_variant.is_some() && (1..PackedPrefill::MIN_ROWS).contains(&tail))
    {
        return Err("Inkling profile requires fully batched prefill");
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
        let timing = if let Some(packed) = &session.packed_prefill {
            packed.upload(chunk);
            execute_with_profile(&session.runtime, Some(&mut profiler), |gpu| {
                packed.encode(&session, gpu, chunk.len(), false)
            })
        } else {
            let lanes = session.prefill_lanes.as_ref().ok_or("missing grouped prefill lanes")?;
            let inputs = chunk.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
            execute_with_profile(&session.runtime, Some(&mut profiler), |gpu| {
                session.encode_prefill_rows(gpu, lanes, &inputs, false)
            })
        }?;
        tiles.push(json!({
            "start_position": session.position,
            "rows": chunk.len(),
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
        "packed_variant": packed_variant,
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

const STOCK_ROWS: [usize; 4] = [2048, 2044, 3, 1];
const STOCK_INPUT: usize = 4096;
const STOCK_CONTEXT: usize = 4352;

/// Attribute the deployed prefill geometry and the next target token on a real
/// stock-schedule oracle. Instrumentation is compared with the same commands
/// without Metal counter attachments.
pub fn profile_stock_4k(
    checkpoint: &Path,
    draft: &Path,
    oracles: &[std::path::PathBuf],
) -> Result<Value> {
    profile_stock(checkpoint, draft, oracles, false)
}

/// Measure tile utilization and shadow exactness controls on real stock routes.
pub fn probe_stock_nvfp4(
    checkpoint: &Path,
    draft: &Path,
    oracles: &[std::path::PathBuf],
) -> Result<Value> {
    profile_stock(checkpoint, draft, oracles, true)
}

fn profile_stock(
    checkpoint: &Path,
    draft: &Path,
    oracles: &[std::path::PathBuf],
    nvfp4_probe: bool,
) -> Result<Value> {
    if oracles.is_empty() {
        return Err("Inkling stock profile requires at least one oracle");
    }
    let mut session = Session::load(checkpoint, STOCK_CONTEXT)?;
    session.load_dspark(draft)?;
    session.set_batched_prefill(true, true)?;
    if !session.runtime.source_text_active || session.prefill_batch_rows() < STOCK_ROWS[0] {
        return Err("Inkling stock profile requires the source-aligned M5 Max path");
    }
    let mut profiler = PrefillTiming::new(&session.runtime)?;
    if packed_prefill::parallel_route_requested() && !session.runtime.parallel_route {
        return Err("Inkling parallel route experiment did not load");
    }
    if packed_prefill::nvfp4_vector_requested() && !session.runtime.nvfp4_vector {
        return Err("Inkling vector NVFP4 experiment did not load");
    }
    if grouped_pilot::mxfp8_tensor_audit_requested() && !session.runtime.mxfp8_tensor_available {
        return Err("Inkling MXFP8 tensor audit requires the tensor path");
    }
    if nvfp4_probe {
        session.runtime.nvfp4_probe = Some(grouped_pilot::Nvfp4Probe::new(&mut session.runtime)?);
    }
    session.runtime.mxfp8_tensor_profile = session.runtime.mxfp8_tensor_available;
    let mut cases = Vec::with_capacity(oracles.len());
    for (index, oracle) in oracles.iter().enumerate() {
        cases.push(profile_stock_case(&mut session, &mut profiler, oracle, index % 2 == 1)?);
    }
    let false_certificates = session
        .packed_prefill
        .as_ref()
        .filter(|_| grouped_pilot::mxfp8_tensor_audit_requested())
        .and_then(|packed| packed.mxfp8_false_certificates());
    Ok(json!({
        "mxfp8_audit_false_certificates": false_certificates,
        "input_tokens": STOCK_INPUT,
        "context_tokens": STOCK_CONTEXT,
        "prefill_command_rows": STOCK_ROWS,
        "decode_start_position": STOCK_INPUT,
        "vision_and_dspark_resident": true,
        "drafting": false,
        "grouped_nvfp4_kernel": if nvfp4_probe { "ink_group_qmm_probe" } else { session.runtime.grouped_qmm },
        "singleton_router_kernel": session.runtime.singleton_router,
        "singleton_parallel_route": session.runtime.parallel_route,
        "singleton_nvfp4_vector": session.runtime.nvfp4_vector,
        "mxfp8_tensor": session.runtime.mxfp8_tensor_active,
        "fast_nvfp4_kernels": session.runtime.fast_nvfp4,
        "cases": cases,
    }))
}

fn profile_stock_case(
    session: &mut Session,
    profiler: &mut PrefillTiming,
    oracle: &Path,
    instrument_first: bool,
) -> Result<Value> {
    let seed = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let continuation = Tensors::open(&oracle.join("row-1.safetensors"))?;
    if seed.values::<u32>("start", "U32", &[1])?[0] as usize != STOCK_INPUT - 1 {
        return Err("Inkling stock oracle must capture the 4K prefill seed");
    }
    let tokens =
        Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[STOCK_INPUT + 3])?;
    if tokens.iter().any(|&token| token as usize >= VOCABULARY) {
        return Err("Inkling token is outside the vocabulary");
    }
    let expected_seed = seed.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
    let expected_continuation = continuation.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
    let (mut plain, mut profiled, mut decode) = (Value::Null, Value::Null, Value::Null);
    let (mut control_plain, mut control_profiled) = (Value::Null, Value::Null);
    let probe = session.runtime.nvfp4_probe.is_some();
    let selected_qmm = session.runtime.grouped_qmm;
    let tensor = session.runtime.mxfp8_tensor_active;
    // The candidate is the default tensor prefill. The same-loaded control is
    // the bit-exact reference kernels, which must reproduce the pinned
    // capture; the candidate reports its tolerance instead.
    let compare = !probe;
    let mut tolerance = Vec::new();
    let orders = [instrument_first, !instrument_first];
    let instruments = if probe {
        &[true][..]
    } else {
        &orders[..]
    };
    for &instrument in instruments {
        let candidate_first = instrument ^ instrument_first;
        let order = [candidate_first, !candidate_first];
        let arms = if compare {
            &order[..]
        } else {
            &[true][..]
        };
        for &candidate in arms {
            session.runtime.exact_prefill = probe || !candidate;
            session.reset();
            let mut commands = Vec::with_capacity(STOCK_ROWS.len());
            let start = std::time::Instant::now();
            for rows in STOCK_ROWS {
                let position = session.position;
                let readout = position + rows == STOCK_INPUT;
                if let Some(probe) = &mut session.runtime.nvfp4_probe {
                    probe.reverse = instrument_first ^ (position == 0);
                    probe.clear();
                }
                // Time the three-row tail as one real command. It would
                // exceed the timestamp buffer if split into encoder stages.
                commands.push(stock_command(
                    session,
                    &tokens[position..position + rows],
                    readout,
                    (instrument && rows != 3).then_some(&mut *profiler),
                )?);
            }
            let logits = session.read_logits()?;
            let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
            session.runtime.exact_prefill = false;
            if (probe || !candidate) && !same_logits(&logits, &expected_seed) {
                return Err("Inkling stock profile seed differs from pinned logits");
            }
            let result = json!({"harness_wall_ms": prefill_ms, "commands": commands});
            if candidate && !instrument {
                let seed_report = target_validation_report(&logits, &expected_seed);
                let prefilled = session.snapshot()?;
                let live = session.step(tokens[STOCK_INPUT])?;
                session.restore(&prefilled)?;
                decode = profile_stock_decode(session, profiler, tokens[STOCK_INPUT], Some(&live))?;
                tolerance.push(json!({
                    "seed": seed_report,
                    "continuation": target_validation_report(&live, &expected_continuation),
                }));
            } else {
                let live = session.step(tokens[STOCK_INPUT])?;
                if (probe || !candidate) && !same_logits(&live, &expected_continuation) {
                    return Err("Inkling stock prefill changed continuation logits");
                }
            }
            *match (candidate, instrument) {
                (true, false) => &mut plain,
                (true, true) => &mut profiled,
                (false, false) => &mut control_plain,
                (false, true) => &mut control_profiled,
            } = result;
        }
    }
    session.runtime.grouped_qmm = selected_qmm;
    session.runtime.mxfp8_tensor_active = tensor;
    session.runtime.exact_prefill = false;
    Ok(json!({
        "exact_reference_pinned_seed_and_continuation": true,
        "default_prefill_tolerance": tolerance,
        "instrumented_prefill_first": instrument_first,
        "uninstrumented_prefill": plain,
        "instrumented_prefill": profiled,
        "decode_at_same_stock_state": decode,
        "prefill_same_loaded_control": compare.then(|| json!({
            "uninstrumented_prefill": control_plain,
            "instrumented_prefill": control_profiled,
        })),
    }))
}

fn profile_stock_decode(
    session: &mut Session,
    profiler: &mut PrefillTiming,
    token: u32,
    expected: Option<&[f32]>,
) -> Result<Value> {
    let snapshot = session.snapshot()?;
    // Without a pinned continuation the first source run is the reference.
    let mut source_logits = expected.map(<[f32]>::to_vec);
    let candidate_router = session.runtime.singleton_router;
    let parallel_route = session.runtime.parallel_route;
    let original_router = if parallel_route {
        candidate_router
    } else {
        "ink_prefill_router_source"
    };
    let compare_router = parallel_route || candidate_router != original_router;
    let mut source_runs = Vec::new();
    let mut legacy_runs = Vec::new();
    let mut legacy_logits = None;
    // Alternate the order of source and legacy singletons across both timing
    // arms. Only the diagnostic switches precision; the cache is the same
    // stock-prefilled state in every run.
    for (index, instrument) in [false, true, true, false].into_iter().enumerate() {
        for source in [index % 2 == 0, index % 2 != 0] {
            session.restore(&snapshot)?;
            session.runtime.source_text_active = source || compare_router;
            if compare_router && !source {
                session.runtime.singleton_router = original_router;
                session.runtime.parallel_route = false;
            }
            let result = stock_command(session, &[token], true, instrument.then_some(&mut *profiler));
            session.runtime.source_text_active = true;
            session.runtime.singleton_router = candidate_router;
            session.runtime.parallel_route = parallel_route;
            let command = result?;
            let started = std::time::Instant::now();
            let logits = session.read_logits()?;
            let read_ms = started.elapsed().as_secs_f64() * 1000.0;
            let reference = if source || compare_router {
                source_logits.get_or_insert_with(|| logits.clone()).as_slice()
            } else {
                legacy_logits.get_or_insert_with(|| logits.clone()).as_slice()
            };
            if !same_logits(&logits, reference) {
                return Err("Inkling stock singleton profile changed same-mode logits");
            }
            let runs = if source {
                &mut source_runs
            } else {
                &mut legacy_runs
            };
            runs.push(json!({"instrumented": instrument, "read_logits_ms": read_ms, "command": command}));
        }
    }
    if compare_router {
        let read = |session: &Session| {
            session
                .visits
                .iter()
                .flat_map(|visit| {
                    let scratch = &visit.scratch;
                    [(&scratch.logits, EXPERTS), (&scratch.routes, 8), (&scratch.route_weights, 8)]
                        .into_iter()
                        .flat_map(|(buffer, count)| read_shared::<u32>(buffer, count))
                })
                .collect::<Vec<_>>()
        };
        session.restore(&snapshot)?;
        session.runtime.singleton_router = original_router;
        session.runtime.parallel_route = false;
        let result = session.step(token);
        session.runtime.singleton_router = candidate_router;
        session.runtime.parallel_route = parallel_route;
        result?;
        let original = read(session);
        session.restore(&snapshot)?;
        session.step(token)?;
        if original != read(session) {
            return Err("Inkling singleton router logits, IDs or weights differed from the source kernel");
        }
    }
    if compare_router {
        Ok(json!({
            "source": source_runs,
            "original_source_router_control": legacy_runs,
            "exact_router_logits_ids_weights": true,
        }))
    } else {
        Ok(json!({"source": source_runs, "legacy_arithmetic_control": legacy_runs}))
    }
}

fn stock_command(
    session: &mut Session,
    tokens: &[u32],
    readout: bool,
    mut profiler: Option<&mut PrefillTiming>,
) -> Result<Value> {
    let start = session.position;
    let timing = session.encode_append(tokens, readout, profiler.as_deref_mut())?;
    let stages = profiler.map(|profile| profile.tile.report(timing.gpu_ms));
    let nvfp4 = session
        .runtime
        .nvfp4_probe
        .as_ref()
        .filter(|_| tokens.len() >= 2044)
        .map(|probe| probe.report(tokens.len()))
        .transpose()?;
    let stats = json!({
        "start_position": start,
        "rows": tokens.len(),
        "dispatches": timing.dispatches,
        "command_wall_ms": timing.wall_ms,
        "encode_ms": timing.encode_ms,
        "submit_and_wait_ms": timing.wall_ms - timing.encode_ms,
        "gpu_ms": timing.gpu_ms,
        "stages": stages,
        "nvfp4": nvfp4,
        "mxfp8_replay_counts": session.packed_prefill.as_ref()
            .filter(|_| session.runtime.mxfp8_tensor_active && tokens.len() > 64)
            .and_then(|packed| packed.mxfp8_replay_counts()),
    });
    session.finish_append(Ok(timing), tokens.len())?;
    Ok(stats)
}

/// Stage attribution of the worker's cold schedule at any prompt length:
/// alternating uninstrumented and instrumented passes of the default prefill
/// and of the bit-exact reference kernels on the same loaded target. The
/// seed row is excluded. Token files stay private; only timings are emitted.
pub fn profile_schedule(
    checkpoint: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    repeats: usize,
    draft: Option<&Path>,
    command_rows: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>(
        "tokens",
        "U32",
        &[prompt.checked_add(3).ok_or("Inkling prompt length overflow")?],
    )?;
    if tokens.iter().any(|&token| token as usize >= VOCABULARY) || prompt + 3 > context {
        return Err("Inkling schedule profile input is outside the vocabulary or context");
    }
    let mut session = Session::load(checkpoint, context)?;
    // With a draft, follow the worker's drafting setup so prefill also
    // commits every row to the draft cache.
    if let Some(draft) = draft {
        session.load_dspark(draft)?;
    }
    session.set_batched_prefill_width(true, true, command_rows)?;
    session.set_text_prefill_scope(true);
    if draft.is_some() {
        session.set_source_verifier(true)?;
        session.prepare_dspark_int8_head()?;
        session.set_dspark_enabled(true)?;
    }
    if !session.runtime.source_text_active {
        return Err("Inkling schedule profile requires the active source path");
    }
    let mut profiler = PrefillTiming::new(&session.runtime)?;
    let ranges = super::schedule_validation::worker_ranges(prompt, true, session.prefill_batch_rows());
    let mut runs = Vec::new();
    for repeat in 0..repeats.max(1) {
        // With a draft the two arms are drafting on and drafting off, both on
        // the default kernels in the same loaded session; otherwise the
        // default kernels and the exact reference.
        for alternate in [repeat % 2 == 1, repeat % 2 == 0] {
            let exact = alternate && draft.is_none();
            let drafting = draft.is_some() && !alternate;
            for instrument in [false, true] {
                session.runtime.exact_prefill = exact;
                session.reset();
                if draft.is_some() {
                    session.set_dspark_enabled(drafting)?;
                }
                let started = std::time::Instant::now();
                let mut commands = Vec::with_capacity(ranges.len());
                for range in &ranges {
                    let result = stock_command(
                        &mut session,
                        &tokens[range.clone()],
                        false,
                        (instrument && range.len() > 3).then_some(&mut profiler),
                    );
                    if result.is_err() {
                        session.runtime.exact_prefill = false;
                    }
                    commands.push(result?);
                }
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
                session.runtime.exact_prefill = false;
                runs.push(json!({
                    "exact_reference": exact, "drafting": drafting,
                    "instrumented": instrument, "harness_wall_ms": wall_ms,
                    "commands": commands,
                }));
            }
        }
    }
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "prefill_commands": ranges.iter().map(ExactSizeIterator::len).collect::<Vec<_>>(),
        "fast_nvfp4_kernels": session.runtime.fast_nvfp4,
        "drafting": draft.is_some(),
        "runs": runs,
    }))
}

fn target_validation_report(
    actual: &[f32],
    expected: &[f32],
) -> Value {
    let top = |values: &[f32]| values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i);
    let mut report = super::target_validation::report(actual, expected);
    report["top1_matches"] = Value::Bool(top(actual) == top(expected));
    report
}

fn same_logits(
    actual: &[f32],
    expected: &[f32],
) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Replay the generation worker's engine call sequence for one cold text
/// request with drafting and a prefix cache scope, timing every engine call:
/// reset, scope, context, verifier and draft selection, prefill to the
/// 2,048-aligned boundary, snapshot, remaining bulk, tail and seed step.
/// Requests alternate drafting on and off, optionally after an idle pause,
/// and the session is loaded at the worker's initial context. Timings only.
pub fn profile_worker_sequence(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    repeats: usize,
    idle_ms: u64,
    setup: &str,
) -> Result<Value> {
    use std::time::{Duration, Instant};
    const SUFFIX: usize = 4;
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>(
        "tokens",
        "U32",
        &[prompt.checked_add(3).ok_or("Inkling prompt length overflow")?],
    )?;
    if tokens.iter().any(|&token| token as usize >= VOCABULARY) || prompt + 3 > context || prompt <= SUFFIX {
        return Err("Inkling worker sequence input is outside the vocabulary or context");
    }
    // setup: "worker" follows the worker's load sequence; "plain" loads at
    // the request context without a draft; "context" loads at 4,352 and
    // resizes; "draft" adds the draft weights without the verifier.
    let started = Instant::now();
    let with_draft = setup == "worker" || setup == "draft";
    let mut session = Session::load(
        checkpoint,
        if setup == "plain" || setup.starts_with("wide") {
            context
        } else {
            4352
        },
    )?;
    if with_draft {
        session.load_dspark(draft)?;
    }
    // "wide4096" and "wide8192" select the source command width through
    // the worker API at load, without a draft.
    match setup {
        "wide4096" => session.set_batched_prefill_width(true, true, 4096)?,
        "wide8192" => session.set_batched_prefill_width(true, true, 8192)?,
        _ => session.set_batched_prefill(true, true)?,
    }
    if setup == "worker" {
        session.set_source_verifier(true)?;
        session.prepare_dspark_int8_head()?;
        session.set_source_verifier(false)?;
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ms = |since: Instant| since.elapsed().as_secs_f64() * 1000.0;
    let mut requests = Vec::new();
    for index in 0..repeats.max(1) * 2 {
        let drafting = setup == "worker" && (index % 2 == 0) ^ (index / 2 % 2 == 1);
        if idle_ms > 0 {
            std::thread::sleep(Duration::from_millis(idle_ms));
        }
        let request = Instant::now();
        let mut stages = serde_json::Map::new();
        let mark = |name: &str, since: Instant, stages: &mut serde_json::Map<String, Value>| {
            stages.insert(name.into(), json!(ms(since)));
        };
        let t = Instant::now();
        session.reset();
        session.set_text_prefill_scope(true);
        mark("reset_and_scope", t, &mut stages);
        let t = Instant::now();
        session.set_context(context)?;
        mark("set_context", t, &mut stages);
        let t = Instant::now();
        if with_draft {
            if setup == "worker" {
                session.set_source_verifier(drafting)?;
            }
            session.set_dspark_enabled(drafting)?;
        }
        mark("verifier_and_draft_mode", t, &mut stages);
        let rows = session.prefill_batch_rows();
        let bulk_end = prompt - SUFFIX;
        let boundary = bulk_end / Session::SOURCE_PREFILL_ROWS * Session::SOURCE_PREFILL_ROWS;
        let mut commands = Vec::new();
        let ingest = |session: &mut Session, from: usize, to: usize, commands: &mut Vec<f64>| -> Result<()> {
            let mut start = from;
            while start < to {
                let end = to.min(start + rows);
                let t = Instant::now();
                session.ingest_many(&tokens[start..end])?;
                commands.push(ms(t));
                start = end;
            }
            Ok(())
        };
        let t = Instant::now();
        ingest(&mut session, 0, boundary, &mut commands)?;
        mark("prefill_to_boundary", t, &mut stages);
        let t = Instant::now();
        let snapshot = if boundary > 0 {
            Some(session.snapshot()?)
        } else {
            None
        };
        mark("snapshot", t, &mut stages);
        let t = Instant::now();
        ingest(&mut session, boundary, bulk_end, &mut commands)?;
        mark("prefill_after_boundary", t, &mut stages);
        let t = Instant::now();
        session.ingest_many(&tokens[bulk_end..prompt - 1])?;
        mark("tail", t, &mut stages);
        let t = Instant::now();
        let logits = session.step(tokens[prompt - 1])?;
        mark("seed_step", t, &mut stages);
        let total = ms(request);
        drop(snapshot);
        requests.push(json!({
            "drafting": drafting,
            "total_ms": total,
            "stages": stages,
            "commands_ms": commands,
            "seed_logits_sha256": target_validation_digest(&logits),
        }));
    }
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "idle_ms": idle_ms,
        "setup": setup,
        "load_ms": load_ms,
        "requests": requests,
    }))
}

fn target_validation_digest(values: &[f32]) -> String {
    super::target_validation::native_digest(values)
}

/// Expert-product ceiling: prefill one real command and time the default
/// grouped NVFP4 products of one routed visit against diagnostic variants
/// (pre-decoded BF16 weights, tensor work only, decode only). Timings only.
/// Decompose the default dense prefill product on one projection shape:
/// production, tensor work only, device-view loads only, staged loads only
/// and the output store. Synthetic operands; timing only.
pub fn measure_dense_ceiling(
    checkpoint: &Path,
    rows: usize,
    width: usize,
    columns: usize,
    reps: usize,
) -> Result<Value> {
    if rows % 128 != 0 || columns % 64 != 0 || width % 64 != 0 || reps == 0 {
        return Err("Inkling dense ceiling needs rows in 128s, columns and width in 64s");
    }
    let session = Session::load(checkpoint, 4352)?;
    let runtime = &session.runtime;
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let source = concat!(include_str!("dense_fast.metal"), "\n", include_str!("dense_ceiling.metal"));
    let library = runtime
        .device
        .new_library_with_source(source, Some(&options))
        .map_err(|_| "Inkling dense ceiling compilation failed")?;
    let variants: [(&str, &str, usize); 6] = [
        ("production", "ink_dense_fast", 256),
        ("production_repeat", "ink_dense_fast", 256),
        ("tensor_only", "ink_dceil_tensor", 256),
        ("loads_device_view", "ink_dceil_loads_device", 256),
        ("loads_staged", "ink_dceil_loads_staged", 256),
        ("store_only", "ink_dceil_store", 256),
    ];
    // Small finite BF16 values (0x3C00 pattern) keep the tensor unit off
    // subnormal or special operands.
    let fill = |bytes: usize| -> Result<Buffer> {
        let buffer = zeroed(&runtime.device, bytes)?;
        unsafe {
            let words = std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u16>(), bytes / 2);
            for (i, word) in words.iter_mut().enumerate() {
                *word = 0x3C00 | (i as u16 & 0x7F);
            }
        }
        Ok(buffer)
    };
    let input = fill(rows * width * 2)?;
    let weight = fill(columns * width * 2)?;
    let output = zeroed(&runtime.device, rows * columns * 2)?;
    let shape = [width as u32, columns as u32, rows as u32, 0];
    let mut report = Vec::new();
    for (name, kernel, threads) in variants {
        let function = library.new_function_with_name(kernel).ok_or(kernel)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| kernel)?;
        let timing = execute(runtime, |gpu| {
            let Some(encoder) = &gpu.encoder else {
                return;
            };
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in [&input, &weight, &output].into_iter().enumerate() {
                encoder.set_buffer(Some(buffer), 0, index);
            }
            encoder.set_bytes(NonNull::from(&shape).cast(), size_of_val(&shape), 3);
            for _ in 0..reps {
                encoder.dispatch_threadgroups(
                    MTLSize {
                        width: columns / 64,
                        height: rows / 128,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        })?;
        let ms = timing.gpu_ms / reps as f64;
        let flops = 2.0 * rows as f64 * width as f64 * columns as f64;
        let operand_bytes = 2.0 * (rows * width + columns * width) as f64;
        report.push(json!({
            "variant": name,
            "gpu_ms": ms,
            "tflops_equivalent": flops / (ms * 1.0e9),
            "unique_operand_gb_per_s": operand_bytes / (ms * 1.0e6),
        }));
    }
    Ok(json!({"rows": rows, "width": width, "columns": columns, "reps": reps, "variants": report}))
}

pub fn measure_expert_ceiling(
    checkpoint: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    visit: usize,
    reps: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>(
        "tokens",
        "U32",
        &[prompt.checked_add(3).ok_or("Inkling prompt length overflow")?],
    )?;
    let mut session = Session::load(checkpoint, context)?;
    session.set_batched_prefill(true, true)?;
    session.set_text_prefill_scope(true);
    if !session.runtime.source_text_active {
        return Err("Inkling expert ceiling requires the active source path");
    }
    let rows = session.prefill_batch_rows().min(prompt);
    session.runtime.expert_capture =
        Some(super::grouped_pilot::ExpertCapture::new(&session.runtime.device, rows, visit)?);
    let ingested = session.ingest_many(&tokens[..rows]);
    let capture = session.runtime.expert_capture.take();
    ingested?;
    let capture = capture.ok_or("Inkling expert ceiling capture missing")?;
    let mut report = super::grouped_pilot::expert_ceiling(&mut session.runtime, &capture, reps)?;
    report["command_rows"] = json!(rows);
    report["visit"] = json!(visit);
    Ok(report)
}
