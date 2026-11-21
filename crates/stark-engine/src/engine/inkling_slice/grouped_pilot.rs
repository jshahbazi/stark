// SPDX-License-Identifier: Apache-2.0
//! Compare the original packed NVFP4 banks using fixed, GPU-selected routes.

use half::bf16;
use serde_json::{Value, json};

use super::*;

const TOP: usize = 8;
const TILE: usize = 32;
const REPEATS: usize = 5;
const PHYSICAL: usize = 10;

fn run_slots(rows: usize) -> usize {
    (rows * TOP).div_ceil(TILE) + EXPERTS - 1
}

pub(super) struct GroupedExperts {
    capacity: usize,
    pub(super) min_rows: usize,
    grouped_mxfp8: bool,
    input: Buffer,
    output: Outputs,
    logits: Buffer,
    ids: Buffer,
    weights: Buffer,
    order: Buffer,
    runs: Buffer,
    up_inner: Buffer,
    up_max: Buffer,
    down_inner: Buffer,
    down_max: Buffer,
    activation: Buffer,
    quantum: Buffer,
    maxima: Buffer,
    projected: Buffer,
}

struct ExpertOrder<'a> {
    ids: &'a Buffer,
    weights: &'a Buffer,
    indices: &'a Buffer,
    runs: &'a Buffer,
}

impl GroupedExperts {
    pub(super) fn new(
        runtime: &mut Runtime,
        capacity: usize,
    ) -> Result<Self> {
        if !runtime.pinned_m5 || !(64..=128).contains(&capacity) {
            return Err("grouped Inkling experts require M5 Max and capacity of 64–128 rows");
        }
        register_grouped(runtime)?;
        Self::buffers(&runtime.device, capacity)
    }

    pub(super) fn rows(&self) -> usize {
        self.capacity
    }

    pub(super) fn input_at(
        &self,
        index: usize,
    ) -> Binding<'_> {
        row(&self.input, index, WIDTH * 2)
    }

    pub(super) fn output_at(
        &self,
        index: usize,
    ) -> Binding<'_> {
        row(&self.output.merged, index, WIDTH * 2)
    }

    pub(super) fn enable_mxfp8(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        register_grouped_mxfp8(runtime)?;
        self.grouped_mxfp8 = true;
        Ok(())
    }

    fn buffers(
        device: &Device,
        capacity: usize,
    ) -> Result<Self> {
        let routes = capacity * TOP;
        Ok(Self {
            capacity,
            min_rows: 64,
            grouped_mxfp8: false,
            input: zeroed(device, capacity * WIDTH * 2)?,
            output: Outputs::new(device, capacity)?,
            logits: zeroed(device, capacity * EXPERTS * 4)?,
            ids: zeroed(device, routes * 4)?,
            weights: zeroed(device, routes * 4)?,
            order: zeroed(device, routes * 4)?,
            runs: zeroed(device, run_slots(capacity) * 3 * 4)?,
            up_inner: zeroed(device, capacity * WIDTH * 4)?,
            up_max: zeroed(device, capacity * 4)?,
            down_inner: zeroed(device, routes * MID * 4)?,
            down_max: zeroed(device, routes * 4)?,
            activation: zeroed(device, routes * WIDTH * 2)?,
            quantum: zeroed(device, routes * (WIDTH / 32) * 4)?,
            maxima: zeroed(device, routes * (WIDTH / 32) * 4)?,
            projected: zeroed(device, routes * WIDTH * 4)?,
        })
    }

    pub(super) fn supports(
        &self,
        layer: &Layer,
        rows: usize,
    ) -> bool {
        (self.min_rows..=self.capacity).contains(&rows)
            && (matches!(layer.experts, Experts::Nvfp4 { .. }) || self.grouped_mxfp8)
    }

    pub(super) fn copy_input(
        &self,
        gpu: &mut Stream<'_>,
        source: Binding<'_>,
        index: usize,
    ) {
        assert!(index < self.capacity);
        gpu.dispatch("ink_group_copy_row", &[source, row(&self.input, index, WIDTH * 2)], (), (WIDTH / 128, 1), 128);
    }

    pub(super) fn copy_output(
        &self,
        gpu: &mut Stream<'_>,
        index: usize,
        destination: Binding<'_>,
    ) {
        assert!(index < self.capacity);
        gpu.dispatch(
            "ink_group_copy_row",
            &[row(&self.output.merged, index, WIDTH * 2), destination],
            (),
            (WIDTH / 128, 1),
            128,
        );
    }

    pub(super) fn routing(
        &self,
        index: usize,
    ) -> [Binding<'_>; 3] {
        [row(&self.logits, index, EXPERTS * 4), row(&self.ids, index, TOP * 4), row(&self.weights, index, TOP * 4)]
    }

    pub(super) fn encode(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        rows: usize,
        source_router: bool,
    ) {
        assert!(self.supports(layer, rows));
        if source_router && matches!(layer.experts, Experts::Nvfp4 { .. }) {
            gpu.one(
                "ink_prefill_router_source",
                &[&self.input, &layer.router, &self.logits],
                rows as u32,
                (EXPERTS / 32, rows.div_ceil(16)),
                32,
            );
            gpu.one(
                "ink_prefill_select_source",
                &[&self.logits, &layer.bias, &self.ids, &self.weights],
                (),
                (rows, 1),
                EXPERTS,
            );
        } else {
            self.encode_router(gpu, &self.input, &layer.router, &layer.bias, rows);
        }
        self.encode_buckets(gpu, &self.ids, rows);
        match &layer.experts {
            Experts::Nvfp4 {
                up,
                down,
            } => {
                self.encode_projected(gpu, &self.input, up, down, rows, self.active_order());
            },
            Experts::Mxfp8 {
                up,
                down,
            } => self.encode_mxfp8(gpu, up, down, rows),
        }
    }

    fn active_order(&self) -> ExpertOrder<'_> {
        ExpertOrder {
            ids: &self.ids,
            weights: &self.weights,
            indices: &self.order,
            runs: &self.runs,
        }
    }

    fn encode_router(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        router: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) {
        gpu.one("ink_group_router", &[input, router, &self.logits], (), (EXPERTS, rows), 32);
        gpu.one("ink_group_select", &[&self.logits, bias, &self.ids, &self.weights], (), (rows, 1), EXPERTS);
    }

    fn encode_buckets(
        &self,
        gpu: &mut Stream<'_>,
        ids: &Buffer,
        rows: usize,
    ) {
        gpu.one("ink_group_bucket", &[ids, &self.order, &self.runs], (rows * TOP) as u32, (1, 1), EXPERTS);
    }

    fn encode_dot(
        &self,
        gpu: &mut Stream<'_>,
        bank: &Bank,
        inner: &Buffer,
        routing: &ExpertOrder<'_>,
        rows: usize,
        width: usize,
        gather: bool,
    ) {
        gpu.one(
            "ink_group_ingress",
            &[inner, routing.indices, &self.activation, &self.quantum, &self.maxima],
            [width as u32, u32::from(gather)],
            ((width / 32).div_ceil(128), rows * TOP),
            128,
        );
        gpu.one(
            "ink_group_qmm",
            &[&self.activation, &bank.codes, &bank.blocks, routing.runs, &self.quantum, &self.maxima, &self.projected],
            width as u32,
            (WIDTH / 64, run_slots(rows)),
            128,
        );
    }

    fn encode_projected(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        up: &Bank,
        down: &Bank,
        rows: usize,
        routing: ExpertOrder<'_>,
    ) {
        let out = &self.output;
        gpu.one("ink_nvfp4_activation", &[input, &self.up_inner, &self.up_max], WIDTH as u32, (rows, 1), 256);
        self.encode_dot(gpu, up, &self.up_inner, &routing, rows, WIDTH, true);
        gpu.one(
            "ink_group_gate_up",
            &[&self.projected, &self.up_max, routing.ids, routing.weights, routing.indices, &up.maxima, &out.middle],
            up.denominator,
            (MID / 128, rows * TOP),
            128,
        );
        gpu.one(
            "ink_nvfp4_activation",
            &[&out.middle, &self.down_inner, &self.down_max],
            MID as u32,
            (rows * TOP, 1),
            256,
        );
        self.encode_dot(gpu, down, &self.down_inner, &routing, rows, MID, false);
        gpu.one(
            "ink_group_down_epilogue",
            &[&self.projected, &self.down_max, routing.ids, routing.indices, &down.maxima, &out.down],
            down.denominator,
            (WIDTH / 128, rows * TOP),
            128,
        );
        gpu.one("ink_group_combine", &[&out.down, &out.merged], (), (WIDTH / 128, rows), 128);
    }

    fn encode_mxfp8(
        &self,
        gpu: &mut Stream<'_>,
        up: &Mxfp8Bank,
        down: &Mxfp8Bank,
        rows: usize,
    ) {
        let out = &self.output;
        let slots = run_slots(rows);
        gpu.one("ink_mxfp8_activation", &[&self.input, &self.up_inner], WIDTH as u32, (WIDTH / 32, rows), 32);
        gpu.one(
            "ink_group_mxfp8_up",
            &[&self.up_inner, &self.order, &self.runs, &self.weights, &up.codes, &up.exponents, &out.middle],
            (),
            (MID / 4, slots),
            128,
        );
        gpu.one("ink_mxfp8_activation", &[&out.middle, &self.down_inner], MID as u32, (MID / 32, rows * TOP), 32);
        gpu.one(
            "ink_group_mxfp8_down",
            &[&self.down_inner, &self.order, &self.runs, &down.codes, &down.exponents, &out.down],
            (),
            (WIDTH / 4, slots),
            128,
        );
        gpu.one("ink_group_combine", &[&out.down, &out.merged], (), (WIDTH / 128, rows), 128);
    }
}

fn register_grouped_mxfp8(runtime: &mut Runtime) -> Result<()> {
    if runtime.pipelines.contains_key("ink_group_mxfp8_up") {
        return Ok(());
    }
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let library = runtime
        .device
        .new_library_with_source(
            concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n",
                include_str!("experts.metal"),
                "\n",
                include_str!("mxfp8.metal"),
                "\n",
                include_str!("mxfp8_grouped.metal")
            ),
            Some(&options),
        )
        .map_err(|_| "Inkling grouped MXFP8 shader compilation failed")?;
    for name in ["ink_group_mxfp8_up", "ink_group_mxfp8_down"] {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    Ok(())
}

fn register_grouped(runtime: &mut Runtime) -> Result<()> {
    if runtime.pipelines.contains_key("ink_group_qmm") {
        return Ok(());
    }
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let source = concat!(
        "#include <metal_stdlib>\nusing namespace metal;\n",
        include_str!("experts.metal"),
        "\n",
        include_str!("grouped_pilot.metal")
    );
    let library = runtime
        .device
        .new_library_with_source(source, Some(&options))
        .map_err(|_| "Inkling grouped library compilation failed")?;
    let names = [
        "ink_group_router",
        "ink_group_select",
        "ink_group_bucket",
        "ink_group_copy_row",
        "ink_group_combine",
        "ink_group_ingress",
        "ink_group_qmm",
        "ink_group_gate_up",
        "ink_group_down_epilogue",
    ];
    for name in names {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    Ok(())
}

fn read<T: bytemuck::Pod>(
    buffer: &Buffer,
    elements: usize,
) -> &[T] {
    assert!(elements * size_of::<T>() <= buffer.length());
    // These buffers are read only after the diagnostic waits for completion.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast(), elements) }
}

fn upload<T: bytemuck::Pod>(
    device: &Device,
    values: &[T],
) -> Result<Buffer> {
    let bytes: &[u8] = bytemuck::cast_slice(values);
    let result = zeroed(device, bytes.len())?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.contents().as_ptr().cast(), bytes.len()) };
    Ok(result)
}

fn row(
    buffer: &Buffer,
    index: usize,
    stride: usize,
) -> Binding<'_> {
    (buffer, index * stride)
}

fn ink_bank(
    source: &Tensors,
    device: &Device,
    suffix: &str,
    width: usize,
) -> Result<Bank> {
    Bank::load(source, device, &format!("model.llm.layers.{PHYSICAL}.mlp.experts.{suffix}"), width)
}

struct Routes {
    ids: Buffer,
    weights: Buffer,
    logits: Buffer,
    forced: bool,
}

impl Routes {
    fn new(
        device: &Device,
        rows: usize,
        forced: bool,
    ) -> Result<Self> {
        let routes = rows * TOP;
        let (ids, weights) = if forced {
            let ids: Vec<_> = (0..routes).map(|i| ((i * 53 + 29) % EXPERTS) as u32).collect();
            let weights: Vec<_> = (0..routes).map(|i| 0.5625 + (i % TOP) as f32 * 0.125).collect();
            (upload(device, &ids)?, upload(device, &weights)?)
        } else {
            (zeroed(device, routes * 4)?, zeroed(device, routes * 4)?)
        };
        Ok(Self {
            ids,
            weights,
            logits: zeroed(device, EXPERTS * 4)?,
            forced,
        })
    }

    fn encode(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        router: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) {
        if self.forced {
            return;
        }
        for i in 0..rows {
            gpu.dispatch(
                "ink_router_logits",
                &[row(input, i, WIDTH * 2), binding(router), binding(&self.logits)],
                WIDTH as u32,
                (EXPERTS, 1),
                32,
            );
            gpu.dispatch(
                "ink_router_select",
                &[binding(&self.logits), binding(bias), row(&self.ids, i, TOP * 4), row(&self.weights, i, TOP * 4)],
                (),
                (1, 1),
                128,
            );
        }
    }
}

struct Outputs {
    middle: Buffer,
    down: Buffer,
    merged: Buffer,
}

impl Outputs {
    fn new(
        device: &Device,
        rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            middle: zeroed(device, rows * TOP * MID * 2)?,
            down: zeroed(device, rows * TOP * WIDTH * 2)?,
            merged: zeroed(device, rows * WIDTH * 2)?,
        })
    }
}

struct Trial {
    rows: usize,
    input: Buffer,
    router: Buffer,
    bias: Buffer,
    routes: Routes,
    serial: Outputs,
    grouped: GroupedExperts,
    up_inner: Buffer,
    up_max: Buffer,
    down_inner: Buffer,
    down_max: Buffer,
}

impl Trial {
    fn new(
        runtime: &Runtime,
        source: &Tensors,
        anchors: &[Vec<u16>],
        rows: usize,
        forced: bool,
    ) -> Result<Self> {
        let device = &runtime.device;
        let mut seed = 0x8765_1234u32;
        let mut values = Vec::with_capacity(rows * WIDTH);
        for _ in 0..rows {
            let weights: [f32; 8] = std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let sample = ((seed >> 16) as f32 + 1.0) / 65536.0;
                sample * sample * sample
            });
            let total = weights.iter().sum::<f32>();
            for channel in 0..WIDTH {
                let value = weights
                    .iter()
                    .enumerate()
                    .map(|(i, &weight)| weight * bf16::from_bits(anchors[i / 4][(i % 4) * WIDTH + channel]).to_f32())
                    .sum::<f32>();
                values.push(bf16::from_f32(value / total).to_bits());
            }
        }
        let input = upload(device, &values)?;
        let routed = rows * TOP;
        let stem = format!("model.llm.layers.{PHYSICAL}.mlp.gate");
        Ok(Self {
            rows,
            routes: Routes::new(device, rows, forced)?,
            router: source.buffer(device, &format!("{stem}.weight"), "BF16", &[EXPERTS, WIDTH])?,
            bias: source.buffer(device, &format!("{stem}.bias"), "F32", &[EXPERTS])?,
            input,
            serial: Outputs::new(device, rows)?,
            grouped: GroupedExperts::buffers(device, rows)?,
            up_inner: zeroed(device, rows * WIDTH * 4)?,
            up_max: zeroed(device, rows * 4)?,
            down_inner: zeroed(device, routed * MID * 4)?,
            down_max: zeroed(device, routed * 4)?,
        })
    }

    fn sum(
        &self,
        gpu: &mut Stream<'_>,
        out: &Outputs,
    ) {
        for i in 0..self.rows {
            gpu.dispatch(
                "ink_expert_sum",
                &[row(&out.down, i, TOP * WIDTH * 2), row(&out.merged, i, WIDTH * 2)],
                WIDTH as u32,
                (WIDTH / 128, 1),
                128,
            );
        }
    }

    fn encode_serial(
        &self,
        gpu: &mut Stream<'_>,
        up: &Bank,
        down: &Bank,
    ) {
        let rows = self.rows;
        let out = &self.serial;
        self.routes.encode(gpu, &self.input, &self.router, &self.bias, rows);
        gpu.one("ink_nvfp4_activation", &[&self.input, &self.up_inner, &self.up_max], WIDTH as u32, (rows, 1), 256);
        for i in 0..rows {
            gpu.dispatch(
                "ink_nvfp4_up",
                &[
                    row(&self.up_inner, i, WIDTH * 4),
                    row(&self.up_max, i, 4),
                    row(&self.routes.ids, i, TOP * 4),
                    row(&self.routes.weights, i, TOP * 4),
                    binding(&up.codes),
                    binding(&up.blocks),
                    binding(&up.maxima),
                    row(&out.middle, i, TOP * MID * 2),
                ],
                up.args(),
                (MID / 4, TOP),
                128,
            );
        }
        gpu.one(
            "ink_nvfp4_activation",
            &[&out.middle, &self.down_inner, &self.down_max],
            MID as u32,
            (rows * TOP, 1),
            256,
        );
        for i in 0..rows {
            gpu.dispatch(
                "ink_nvfp4_down",
                &[
                    row(&self.down_inner, i, TOP * MID * 4),
                    row(&self.down_max, i, TOP * 4),
                    row(&self.routes.ids, i, TOP * 4),
                    binding(&down.codes),
                    binding(&down.blocks),
                    binding(&down.maxima),
                    row(&out.down, i, TOP * WIDTH * 2),
                ],
                down.args(),
                (WIDTH / 4, TOP),
                128,
            );
        }
        self.sum(gpu, out);
    }

    fn encode_grouped(
        &self,
        gpu: &mut Stream<'_>,
        up: &Bank,
        down: &Bank,
    ) {
        let rows = self.rows;
        let grouped = &self.grouped;
        let routing = if self.routes.forced {
            ExpertOrder {
                ids: &self.routes.ids,
                weights: &self.routes.weights,
                indices: &grouped.order,
                runs: &grouped.runs,
            }
        } else {
            grouped.encode_router(gpu, &self.input, &self.router, &self.bias, rows);
            grouped.active_order()
        };
        grouped.encode_buckets(gpu, routing.ids, rows);
        grouped.encode_projected(gpu, &self.input, up, down, rows, routing);
    }

    fn measure(
        &self,
        runtime: &Runtime,
        up: &Bank,
        down: &Bank,
    ) -> Result<Value> {
        let serial = || execute(runtime, |gpu| self.encode_serial(gpu, up, down));
        let grouped = || execute(runtime, |gpu| self.encode_grouped(gpu, up, down));
        serial()?;
        grouped()?;
        let mut samples = Vec::with_capacity(REPEATS);
        for index in 0..REPEATS {
            let (serial, grouped) = if index % 2 == 0 {
                let baseline = serial()?;
                (baseline, grouped()?)
            } else {
                let candidate = grouped()?;
                (serial()?, candidate)
            };
            samples.push(json!({"serial_wall_ms": serial.wall_ms, "serial_gpu_ms":serial.gpu_ms,
                "grouped_wall_ms":grouped.wall_ms,"grouped_gpu_ms":grouped.gpu_ms,
                "serial_dispatches":serial.dispatches,"grouped_dispatches":grouped.dispatches}));
        }
        let pairs = self.rows * TOP;
        let serial_ids = read::<u32>(&self.routes.ids, pairs);
        let serial_weights = read::<u32>(&self.routes.weights, pairs);
        let grouped = &self.grouped;
        let router_matches = self.routes.forced
            || (serial_ids == read::<u32>(&grouped.ids, pairs)
                && serial_weights == read::<u32>(&grouped.weights, pairs));
        let order =
            bucket_reference(serial_ids, read(&grouped.order, pairs), read(&grouped.runs, run_slots(self.rows) * 3))?;
        Ok(json!({
            "rows":self.rows,"forced_even_routes":self.routes.forced,"exact_router":(!self.routes.forced).then_some(router_matches),
            "buckets":order.summary,"pairs":samples,
            "bf16_weighted_middle":difference(&self.serial.middle, &grouped.output.middle, MID, pairs, Some(&order.order))?,
            "bf16_down":difference(&self.serial.down, &grouped.output.down, WIDTH, pairs, None)?,
            "bf16_expert_sum":difference(&self.serial.merged, &grouped.output.merged, WIDTH, self.rows, None)?,
        }))
    }
}

struct BucketReference {
    order: Vec<usize>,
    summary: Value,
}

fn bucket_reference(
    ids: &[u32],
    grouped_order: &[u32],
    grouped_runs: &[u32],
) -> Result<BucketReference> {
    let mut order: Vec<_> = (0..ids.len()).collect();
    order.sort_by_key(|&index| ids[index]);
    let mut runs = vec![0u32; grouped_runs.len()];
    let (mut start, mut live, mut experts, mut longest) = (0, 0, 0, 0);
    while start < order.len() {
        let expert = ids[order[start]];
        if expert as usize >= EXPERTS {
            return Err("invalid diagnostic expert route");
        }
        let end = start + order[start..].partition_point(|&index| ids[index] == expert);
        longest = longest.max(end - start);
        experts += 1;
        for block in (start..end).step_by(TILE) {
            runs[live * 3..][..3].copy_from_slice(&[block as u32, (end - block).min(TILE) as u32, expert]);
            live += 1;
        }
        start = end;
    }
    let exact =
        runs == grouped_runs && order.iter().zip(grouped_order).all(|(&expected, &actual)| expected == actual as usize);
    Ok(BucketReference {
        order,
        summary: json!({"exact":exact,"experts_used":experts,"max_rows_per_expert":longest,"live_runs":live,"run_slots":runs.len()/3}),
    })
}

fn difference(
    serial: &Buffer,
    grouped: &Buffer,
    width: usize,
    rows: usize,
    order: Option<&[usize]>,
) -> Result<Value> {
    let serial = read::<u16>(serial, width * rows);
    let grouped = read::<u16>(grouped, width * rows);
    let mut changed = 0;
    let mut maximum = 0.0f64;
    let mut norm = 0.0f64;
    let mut error = 0.0f64;
    let mut worst_row = 0.0f64;
    for (row, candidate) in grouped.chunks_exact(width).enumerate() {
        let expected = &serial[order.map_or(row, |order| order[row]) * width..][..width];
        let mut row_norm = 0.0;
        let mut row_error = 0.0;
        for (&a, &b) in expected.iter().zip(candidate) {
            changed += usize::from(a != b);
            let (a, b) = (f64::from(bf16::from_bits(a).to_f32()), f64::from(bf16::from_bits(b).to_f32()));
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite grouped projection output");
            }
            maximum = maximum.max((a - b).abs());
            row_norm += a * a;
            row_error += (a - b) * (a - b);
        }
        norm += row_norm;
        error += row_error;
        worst_row = worst_row.max((row_error / row_norm.max(f64::MIN_POSITIVE)).sqrt());
    }
    Ok(
        json!({"words":rows*width,"different":changed,"relative_l2":(error / norm.max(f64::MIN_POSITIVE)).sqrt(),"worst_row_relative_l2":worst_row,"max_absolute":maximum}),
    )
}

pub fn measure_grouped_nvfp4(
    checkpoint: &std::path::Path,
    short: &std::path::Path,
    long: &std::path::Path,
) -> Result<Value> {
    let mut runtime = Runtime::new()?;
    if !runtime.pinned_m5 {
        return Err("grouped Inkling diagnostic requires M5 Max Metal 4");
    }
    register_grouped(&mut runtime)?;
    let source = Tensors::open(checkpoint)?;
    let up = ink_bank(&source, &runtime.device, "w13", WIDTH)?;
    let down = ink_bank(&source, &runtime.device, "w2", MID)?;
    let key = format!("{PHYSICAL}.mlp_input");
    let anchors = [short, long]
        .into_iter()
        .map(|fixture| Tensors::open(fixture)?.values(&key, "BF16", &[4, WIDTH]))
        .collect::<Result<Vec<Vec<u16>>>>()?;
    let shapes = [(64, false), (128, false), (64, true), (128, true)]
        .into_iter()
        .map(|(rows, forced)| Trial::new(&runtime, &source, &anchors, rows, forced)?.measure(&runtime, &up, &down))
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"physical_layer":PHYSICAL,"grouping_in_timing":true,"source_anchors":8,"cases":shapes}))
}
