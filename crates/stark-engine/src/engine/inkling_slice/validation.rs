// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, path::Path};

use half::bf16;
use serde::Serialize;
use serde_json::{Value, json};

use super::*;

const ROWS: usize = 4;
const SAMPLES: usize = 32;
const WARMUPS: usize = 8;
const VISITS: [usize; 44] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];

fn contents<T: bytemuck::Pod>(
    buffer: &Buffer,
    count: usize,
    offset: usize,
) -> &[T] {
    assert!(offset + count * size_of::<T>() <= buffer.length());
    // The only callers are after terminal completion, with no work in flight.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>().add(offset).cast(), count) }
}

fn write<T: bytemuck::Pod>(
    buffer: &Buffer,
    offset: usize,
    values: &[T],
) {
    let bytes = bytemuck::cast_slice::<T, u8>(values);
    assert!(offset + bytes.len() <= buffer.length());
    // The caller owns the state and has already waited for all GPU work.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().as_ptr().cast::<u8>().add(offset), bytes.len())
    };
}

fn expand(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&x| bf16::from_bits(x).to_f32()).collect()
}

fn load_flow(
    fixture: &Tensors,
    flow: &Flow,
    row: usize,
) -> Result<()> {
    for (name, tensor, dimensions, num) in [
        ("blocks", &flow.blocks, &[ROWS, 2, WIDTH][..], 2),
        ("previous", &flow.previous, &[ROWS, 2, WIDTH][..], 2),
        ("partial", &flow.partial, &[ROWS, WIDTH][..], 1),
    ] {
        let input = fixture.values::<u16>(name, "BF16", dimensions)?;
        write(tensor, 0, &expand(&input[row * num * WIDTH..(row + 1) * num * WIDTH]));
    }
    Ok(())
}

fn seed_visit(
    fixture: &Tensors,
    visit: &Visit,
) -> Result<()> {
    let id = if visit.capacity == LOCAL {
        10
    } else {
        11
    };
    let shape = &[visit.capacity, KV];
    let bytes = visit.capacity * KV * 2;
    fixture.copy_to(&format!("{id}.keys"), "BF16", shape, 0, &visit.keys, 0, bytes)?;
    fixture.copy_to(&format!("{id}.values"), "BF16", shape, 0, &visit.values, 0, bytes)?;
    let mut offset = 0;
    for (buffer, width) in visit.conv.iter().zip([KV, KV, WIDTH, WIDTH]) {
        let bytes = 3 * width * 2;
        fixture.copy_to(&format!("{id}.conv"), "BF16", &[3 * (2 * WIDTH + 2 * KV)], offset, buffer, 0, bytes)?;
        offset += bytes;
    }
    Ok(())
}

#[derive(Default, Serialize)]
struct Parity {
    values: usize,
    unequal: usize,
    worst_row_relative_l2: f64,
    worst_row_peak_error: f64,
    limit: f64,
    failed_rows: usize,
}

impl Parity {
    fn add(
        &mut self,
        actual: &[f32],
        expected: &[f32],
        limit: f64,
    ) {
        let (mut sum, mut residual, mut peak, mut delta) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut finite = true;
        self.limit = limit;
        self.values += actual.len();
        for (&a, &b) in actual.iter().zip(expected) {
            finite &= a.is_finite() && b.is_finite();
            let diff = (a as f64 - b as f64).abs();
            self.unequal += usize::from(a != b);
            sum += (b as f64).powi(2);
            residual += diff * diff;
            peak = peak.max((b as f64).abs());
            delta = delta.max(diff);
        }
        let relative = (residual / sum.max(1e-30)).sqrt();
        let peak_error = delta / peak.max(1e-15);
        self.worst_row_relative_l2 = self.worst_row_relative_l2.max(relative);
        self.worst_row_peak_error = self.worst_row_peak_error.max(peak_error);
        self.failed_rows += usize::from(!finite || relative > limit || peak_error > limit);
    }
}

struct Checks<'a> {
    fixture: &'a Tensors,
    index: usize,
    position: usize,
    row: usize,
    metrics: BTreeMap<String, Parity>,
    route_misses: usize,
    exact_route_misses: usize,
}

impl Checks<'_> {
    fn compare(
        &mut self,
        name: &str,
        expected: &[f32],
        actual: &[f32],
        limit: f64,
    ) {
        self.metrics.entry(format!("{}.{name}", self.index)).or_default().add(actual, expected, limit);
    }

    fn bf16_stage(
        &mut self,
        name: &str,
        buffer: &Buffer,
        width: usize,
        offset: usize,
        limit: f64,
    ) -> Result<()> {
        let values = self.fixture.values::<u16>(&format!("{}.{name}", self.index), "BF16", &[ROWS, width])?;
        let expected = expand(&values[self.row * width..(self.row + 1) * width]);
        self.compare(name, &expected, &expand(contents::<u16>(buffer, width, offset)), limit);
        Ok(())
    }

    fn visit(
        &mut self,
        id: usize,
        visit: &Visit,
    ) -> Result<()> {
        self.index = id;
        let position = self.position + self.row;
        let s = &visit.scratch;
        let offset = position % visit.capacity * KV * 2;
        self.bf16_stage("key_rows", &visit.keys, KV, offset, 0.008)?;
        self.bf16_stage("value_rows", &visit.values, KV, offset, 0.008)?;
        self.bf16_stage("attention", &s.attention, WIDTH, 0, 0.008)?;
        self.bf16_stage("mlp_input", &s.mlp_input, WIDTH, 0, 0.008)?;
        self.bf16_stage("experts", &s.experts, WIDTH, 0, 0.015)?;
        let values = self.fixture.values::<u16>(&format!("{id}.partial"), "BF16", &[ROWS, WIDTH])?;
        self.compare(
            "partial",
            &expand(&values[self.row * WIDTH..(self.row + 1) * WIDTH]),
            contents::<f32>(&s.partial_debug, WIDTH, 0),
            0.008,
        );
        let message = self.fixture.values::<f32>(&format!("{id}.messages"), "F32", &[ROWS, 4, WIDTH])?;
        for slot in 0..4 {
            let start = (self.row * 4 + slot) * WIDTH;
            let actual = contents::<f32>(&s.messages, WIDTH, slot * WIDTH * 4);
            self.compare(&format!("message_{slot}"), &message[start..start + WIDTH], actual, 2e-5);
        }
        let conv_width = 2 * KV + 2 * WIDTH;
        let conv = self.fixture.values::<u16>(&format!("{id}.conv_rows"), "BF16", &[ROWS, conv_width])?;
        let mut start = self.row * conv_width;
        for (slot, width) in [KV, KV, WIDTH, WIDTH].into_iter().enumerate() {
            let actual = contents::<u16>(&visit.conv[slot], width, position % 3 * width * 2);
            self.compare(&format!("conv_{slot}"), &expand(&conv[start..start + width]), &expand(actual), 0.008);
            start += width;
        }
        let route = self.fixture.values::<u32>(&format!("{id}.routes"), "U32", &[ROWS, 8])?;
        let mut expect = route[self.row * 8..(self.row + 1) * 8].to_vec();
        let mut actual = contents::<u32>(&s.routes, 8, 0).to_vec();
        expect.sort_unstable();
        actual.sort_unstable();
        self.route_misses += usize::from(expect != actual);
        Ok(())
    }

    fn expert_replay(
        &mut self,
        runtime: &Runtime,
        layer: &Layer,
        s: &Scratch,
    ) -> Result<()> {
        let id = self.index;
        let begin = self.row * WIDTH;
        let input = self.fixture.values::<u16>(&format!("{id}.mlp_input"), "BF16", &[ROWS, WIDTH])?;
        write(&s.mlp_input, 0, &input[begin..begin + WIDTH]);
        execute(runtime, |gpu| layer.experts(gpu, s))?;
        let expected = self.fixture.values::<f32>(&format!("{id}.up_inner"), "F32", &[ROWS, WIDTH])?;
        self.compare(
            "exact_ingress_up_inner",
            &expected[begin..begin + WIDTH],
            contents::<f32>(&s.up_inner, WIDTH, 0),
            0.,
        );
        let expected = self.fixture.values::<f32>(&format!("{id}.up_max"), "F32", &[ROWS, 1])?;
        self.compare("exact_ingress_up_max", &expected[self.row..self.row + 1], contents::<f32>(&s.up_max, 1, 0), 0.);
        let expected = self.fixture.values::<u16>(&format!("{id}.experts"), "BF16", &[ROWS, WIDTH])?;
        self.compare(
            "exact_ingress_experts",
            &expand(&expected[begin..begin + WIDTH]),
            &expand(contents::<u16>(&s.experts, WIDTH, 0)),
            0.006,
        );
        let routes = self.fixture.values::<u32>(&format!("{id}.routes"), "U32", &[ROWS, 8])?;
        let routes = &routes[self.row * 8..(self.row + 1) * 8];
        let mut expected_set = routes.to_vec();
        let mut actual_set = contents::<u32>(&s.routes, 8, 0).to_vec();
        expected_set.sort_unstable();
        actual_set.sort_unstable();
        self.exact_route_misses += usize::from(expected_set != actual_set);
        let expected_mid = self.fixture.values::<u16>(&format!("{id}.mid"), "BF16", &[ROWS, 8, MID])?;
        let expected_down = self.fixture.values::<f32>(&format!("{id}.down_inner"), "F32", &[ROWS, 8, MID])?;
        let weights = self.fixture.values::<f32>(&format!("{id}.route_weights"), "F32", &[ROWS, 8])?;
        for (i, &expert) in contents::<u32>(&s.routes, 8, 0).iter().enumerate() {
            if let Some(j) = routes.iter().position(|&value| value == expert) {
                let offset = (self.row * 8 + j) * MID;
                self.compare(
                    "exact_ingress_mid",
                    &expand(&expected_mid[offset..offset + MID]),
                    &expand(contents::<u16>(&s.mid, MID, i * MID * 2)),
                    0.006,
                );
                self.compare(
                    "exact_ingress_down_inner",
                    &expected_down[offset..offset + MID],
                    contents::<f32>(&s.down_inner, MID, i * MID * 4),
                    0.006,
                );
                let offset = self.row * 8 + j;
                self.compare(
                    "exact_ingress_route_weights",
                    &weights[offset..offset + 1],
                    contents::<f32>(&s.route_weights, 1, i * 4),
                    2e-5,
                );
            }
        }
        Ok(())
    }
}

fn untouched_cache_mismatches(
    fixture: &Tensors,
    id: usize,
    visit: &Visit,
    position: usize,
) -> Result<usize> {
    let capacity = visit.capacity;
    let mut mismatches = 0;
    for (name, buffer) in [("keys", &visit.keys), ("values", &visit.values)] {
        let expected = fixture.values::<u16>(&format!("{id}.{name}"), "BF16", &[capacity, KV])?;
        let actual = contents::<u16>(buffer, capacity * KV, 0);
        for (slot, (a, b)) in actual.chunks_exact(KV).zip(expected.chunks_exact(KV)).enumerate() {
            if !(position..position + ROWS).any(|pos| pos % capacity == slot) {
                mismatches += a.iter().zip(b).filter(|(x, y)| x != y).count();
            }
        }
    }
    Ok(mismatches)
}

fn check_parity(
    runtime: &Runtime,
    model: &Tensors,
    fixture: &Tensors,
) -> Result<Value> {
    let local = Layer::load(model, &runtime.device, 10)?;
    let global = Layer::load(model, &runtime.device, 11)?;
    let first = Visit::new(&runtime.device, true)?;
    let second = Visit::new(&runtime.device, false)?;
    let flow = Flow::new(&runtime.device)?;
    seed_visit(fixture, &first)?;
    seed_visit(fixture, &second)?;
    let position = fixture.values::<u32>("start", "U32", &[1])?[0] as usize;
    if !(3..=CAPACITY - ROWS).contains(&position) {
        return Err("fixture position outside supported cache geometry");
    }
    let mut checks = Checks {
        fixture,
        row: 0,
        index: 0,
        position,
        metrics: BTreeMap::new(),
        route_misses: 0,
        exact_route_misses: 0,
    };
    let input = VisitInput {
        blocks: 2,
        later: true,
        boundary: false,
        capture: true,
    };
    for row in 0..ROWS {
        load_flow(fixture, &flow, row)?;
        execute(runtime, |gpu| {
            local.encode(gpu, &first, &flow, &flow.partial, input, position + row);
            global.encode(gpu, &second, &flow, &flow.partial, input, position + row);
        })?;
        checks.row = row;
        checks.visit(10, &first)?;
        checks.expert_replay(runtime, &local, &first.scratch)?;
        checks.visit(11, &second)?;
        checks.expert_replay(runtime, &global, &second.scratch)?;
    }
    let cache_mismatches = untouched_cache_mismatches(fixture, 10, &first, position)?
        + untouched_cache_mismatches(fixture, 11, &second, position)?;
    let (replay_checks, connected_checks): (BTreeMap<_, _>, BTreeMap<_, _>) =
        checks.metrics.into_iter().partition(|(name, _)| name.contains(".exact_ingress"));
    let failed = |stages: &BTreeMap<String, Parity>| stages.values().map(|metric| metric.failed_rows).sum::<usize>();

    // Feed the global layer the reference's BF16 local output. Isolate its
    // attention and cache writes from errors propagated by the local experts.
    seed_visit(fixture, &second)?;
    let partial = fixture.values::<u16>("10.partial", "BF16", &[ROWS, WIDTH])?;
    let mut global_checks = Checks {
        fixture,
        row: 0,
        index: 11,
        position,
        metrics: BTreeMap::new(),
        route_misses: 0,
        exact_route_misses: 0,
    };
    for row in 0..ROWS {
        load_flow(fixture, &flow, row)?;
        write(&flow.partial, 0, &expand(&partial[row * WIDTH..(row + 1) * WIDTH]));
        execute(runtime, |gpu| global.encode(gpu, &second, &flow, &flow.partial, input, position + row))?;
        global_checks.row = row;
        let offset = (position + row) * KV * 2;
        global_checks.bf16_stage("key_rows", &second.keys, KV, offset, 0.008)?;
        global_checks.bf16_stage("value_rows", &second.values, KV, offset, 0.008)?;
        global_checks.bf16_stage("attention", &second.scratch.attention, WIDTH, 0, 0.008)?;
    }
    let connected_passed = failed(&connected_checks) == 0 && checks.route_misses == 0 && cache_mismatches == 0;
    let replay_passed = failed(&replay_checks) == 0 && checks.exact_route_misses == 0;
    let global_passed = failed(&global_checks.metrics) == 0;
    Ok(json!({
        "passed": connected_passed && replay_passed && global_passed,
        "starting_position": position,
        "sequential_positions": ROWS,
        "physical_layers": [10, 11],
        "route_rows": ROWS * 2,
        "connected": {
            "passed": connected_passed,
            "route_set_mismatches": checks.route_misses,
            "untouched_kv_values_checked": 2 * (LOCAL + CAPACITY - 2 * ROWS) * KV,
            "untouched_kv_mismatches": cache_mismatches,
            "failed_stage_rows": failed(&connected_checks),
            "stages": connected_checks,
        },
        "exact_expert_ingress": {
            "passed": replay_passed,
            "route_set_mismatches": checks.exact_route_misses,
            "failed_stage_rows": failed(&replay_checks),
            "stages": replay_checks,
        },
        "exact_global_visit_ingress": {
            "passed": global_passed,
            "failed_stage_rows": failed(&global_checks.metrics),
            "stages": global_checks.metrics,
        },
    }))
}

fn distributions(
    samples: &[Timing],
    metric: impl Fn(&Timing) -> f64,
) -> Value {
    let mut series: Vec<_> = samples.iter().map(metric).collect();
    series.sort_by(f64::total_cmp);
    let q = |f: f64| {
        let rank = f * (series.len() - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        series[lo] + (rank - lo as f64) * (series[hi] - series[lo])
    };
    json!({"min": q(0.), "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": q(1.)})
}

fn benchmark(
    runtime: &Runtime,
    model: &Tensors,
    fixture: &Tensors,
) -> Result<Value> {
    let layers = (0..20).map(|i| Layer::load(model, &runtime.device, i)).collect::<Result<Vec<_>>>()?;
    let visits = VISITS.iter().map(|&i| Visit::new(&runtime.device, layers[i].local)).collect::<Result<Vec<_>>>()?;
    for visit in &visits {
        seed_visit(fixture, visit)?;
    }
    let flow = Flow::new(&runtime.device)?;
    let source = fixture.values::<u16>("partial", "BF16", &[ROWS, WIDTH])?;
    let ingress = expand(&source);
    let mut samples = Vec::with_capacity(SAMPLES);
    for round in 0..WARMUPS + SAMPLES {
        let begin = (round % ROWS) * WIDTH;
        write(&flow.partial, 0, &ingress[begin..begin + WIDTH]);
        let timing = execute(runtime, |gpu| {
            let mut blocks = 0;
            for (i, (&physical, visit)) in VISITS.iter().zip(&visits).enumerate() {
                if i == 6 {
                    gpu.copy(binding(&flow.partial), binding(&flow.prelude));
                }
                if i == 18 || i == 30 {
                    // The previous core pass contributes its last frozen block
                    // and ending partial. Restart from the saved prelude.
                    gpu.copy((&flow.blocks, 2 * WIDTH * 4), binding(&flow.previous));
                    gpu.copy(binding(&flow.partial), (&flow.previous, WIDTH * 4));
                    gpu.copy(binding(&flow.prelude), binding(&flow.partial));
                    blocks = 1;
                }
                let boundary = physical % 6 == 0;
                let state = VisitInput {
                    blocks,
                    later: (18..42).contains(&i),
                    boundary,
                    capture: false,
                };
                layers[physical].encode(gpu, visit, &flow, &flow.partial, state, START + round);
                blocks += usize::from(boundary);
            }
        })?;
        if round >= WARMUPS {
            samples.push(timing);
        }
    }
    if contents::<f32>(&flow.partial, WIDTH, 0).iter().any(|x| !x.is_finite()) {
        return Err("nonfinite slice terminal state");
    }
    let dispatches = samples[0].dispatches;
    if samples.iter().any(|s| s.dispatches != dispatches || s.gpu_ms <= 0.0) {
        return Err("invalid slice timing");
    }
    Ok(json!({
        "starting_position": START,
        "first_timed_position": START + WARMUPS,
        "last_timed_position": START + WARMUPS + SAMPLES - 1,
        "warmups": WARMUPS,
        "samples": samples.len(),
        "physical_layers": layers.len(),
        "nvfp4_visits": visits.len(),
        "global_visits": visits.iter().filter(|visit| visit.capacity != LOCAL).count(),
        "gated_visits": 24,
        "command_buffers_per_step": 1,
        "compute_encoders_per_step": 1,
        "completion_waits_per_step": 1,
        "dispatches_per_step": dispatches,
        "ingress_bytes_per_step": WIDTH * 4,
        "readback_bytes_per_timed_step": 0,
        "metal_allocated_bytes": runtime.device.current_allocated_size(),
        "wall_ms": distributions(&samples, |s| s.wall_ms),
        "gpu_ms": distributions(&samples, |s| s.gpu_ms),
        "wall_minus_gpu_ms": distributions(&samples, |s| s.wall_ms - s.gpu_ms),
    }))
}

/// Run original-checkpoint decode validation on private, source-shaped fixtures.
/// The returned values are aggregate-only and may be recorded in benchmark docs.
pub fn measure(
    model: &Path,
    fixture: &Path,
    parity_only: bool,
) -> Result<Value> {
    let model = Tensors::open(model)?;
    let fixture = Tensors::open(fixture)?;
    let runtime = Runtime::new()?;
    let parity = check_parity(&runtime, &model, &fixture)?;
    let timing = if !parity_only && parity["starting_position"] == START {
        Some(benchmark(&runtime, &model, &fixture)?)
    } else {
        None
    };
    Ok(json!({"parity": parity, "decode_slice": timing}))
}
