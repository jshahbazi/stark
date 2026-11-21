// SPDX-License-Identifier: Apache-2.0
//! Compare the original packed NVFP4 banks using fixed, GPU-selected routes.

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{session::read_shared, *};

const TOP: usize = 8;
const TILE: usize = 32;
const ROWS64_QMM: Kernel = "ink_group_qmm_reused_guard_rows64";
const EXPERIMENT_BUCKET: Kernel = "ink_group_bucket_experiment";
const REPEATS: usize = 5;
const PHYSICAL: usize = 10;
const PROBE_FIELDS: usize = 5;
const PROBE_DOTS: usize = 44 * 2;
const MXFP8_DOTS: usize = 8;
const PROBE_SAMPLE: [usize; 6] = [0, 1, 42, 43, 84, 85];

pub(super) struct Nvfp4Probe {
    counts: Buffer,
    pub(super) reverse: bool,
}

impl Nvfp4Probe {
    pub(super) fn new(runtime: &mut Runtime) -> Result<Self> {
        let kernel = "ink_group_qmm_probe";
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = runtime
            .device
            .new_library_with_source(
                concat!(
                    "#include <metal_stdlib>\nusing namespace metal;\n",
                    "#define INK_GROUP_QMM_PROBE\n#define ink_group_qmm ink_group_qmm_probe\n",
                    include_str!("experts.metal"),
                    "\n",
                    include_str!("grouped_pilot.metal"),
                ),
                Some(&options),
            )
            .map_err(|_| "Inkling NVFP4 telemetry compilation failed")?;
        let function = library.new_function_with_name(kernel).ok_or(kernel)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| kernel)?;
        runtime.pipelines.insert(kernel, pipeline);
        let counts = zeroed(&runtime.device, (PROBE_DOTS + PROBE_SAMPLE.len() * 3) * PROBE_FIELDS * 4)?;
        Ok(Self {
            counts,
            reverse: false,
        })
    }

    pub(super) fn clear(&self) {
        // Each preceding command has completed before the shared counters are cleared.
        unsafe { self.counts.contents().as_ptr().cast::<u8>().write_bytes(0, self.counts.length()) };
    }

    pub(super) fn report(
        &self,
        rows: usize,
    ) -> Result<Value> {
        let count = read_shared::<u32>(&self.counts, PROBE_DOTS * PROBE_FIELDS);
        let mut phases = Vec::with_capacity(2);
        for (phase, width) in [WIDTH, MID].into_iter().enumerate() {
            let mut totals = [0u64; PROBE_FIELDS];
            for record in count.chunks_exact(PROBE_FIELDS).skip(phase).step_by(2) {
                if record[1] as usize != rows * TOP || record[0] == 0 {
                    return Err("Inkling NVFP4 telemetry did not cover every real routed row");
                }
                for (total, value) in totals.iter_mut().zip(record) {
                    *total += u64::from(*value);
                }
            }
            let [live_tiles, routed_rows, active_blocks, guard_failures, replay_entries] = totals;
            phases.push(json!({
                "phase": if phase == 0 { "gate_up" } else { "down" },
                "live_run_tiles": live_tiles,
                "launched_run_slots": run_slots(rows) * (PROBE_DOTS / 2),
                "routed_rows": routed_rows,
                "live_row_capacity": live_tiles * TILE as u64,
                "active_simd_blocks": active_blocks,
                "guard_failed_simd_blocks": guard_failures,
                "replay_entries": replay_entries,
                "eligible_output_kblocks": routed_rows * WIDTH as u64 * (width / 32) as u64,
            }));
        }
        Ok(json!({"timed_dot_ordinals": PROBE_SAMPLE, "occupancy": phases}))
    }
}

fn run_slots(rows: usize) -> usize {
    run_slots_for(rows, TILE)
}

fn run_slots_for(
    rows: usize,
    tile: usize,
) -> usize {
    (rows * TOP).div_ceil(tile) + EXPERTS - 1
}

fn active_rows64(gpu: &Stream<'_>) -> bool {
    gpu.runtime.source_text_active && gpu.runtime.nvfp4_probe.is_none() && gpu.runtime.grouped_qmm == ROWS64_QMM
}

pub(super) struct GroupedExperts {
    capacity: usize,
    pub(super) min_rows: usize,
    grouped_mxfp8: bool,
    wide_mxfp8: bool,
    source_mxfp8: bool,
    mxfp8_tensor: Option<Mxfp8Tensor>,
    gate_up: Kernel,
    source_log_weights: Option<(Buffer, Buffer)>,
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

struct Mxfp8Tensor {
    input_norms: Buffer,
    masks: Buffer,
    counts: Buffer,
    false_certificates: Buffer,
}

pub(super) fn mxfp8_tensor_audit_requested() -> bool {
    std::env::var_os("STARK_INKLING_EXPERIMENT_MXFP8_TENSOR_AUDIT").as_deref() == Some(std::ffi::OsStr::new("1"))
}

pub(super) fn mxfp8_tensor_requested() -> bool {
    std::env::var_os("STARK_INKLING_EXPERIMENT_MXFP8_TENSOR").as_deref() == Some(std::ffi::OsStr::new("1"))
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
        if !runtime.pinned_m5 || !(64..=2048).contains(&capacity) {
            return Err("grouped Inkling experts require M5 Max and capacity of 64–2,048 rows");
        }
        register_grouped(runtime)?;
        Self::buffers(&runtime.device, capacity)
    }

    pub(super) fn rows(&self) -> usize {
        self.capacity
    }

    pub(super) fn input(&self) -> &Buffer {
        &self.input
    }

    pub(super) fn output(&self) -> &Buffer {
        &self.output.merged
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
        register_grouped_mxfp8(runtime, false, false)?;
        self.grouped_mxfp8 = true;
        Ok(())
    }

    pub(super) fn enable_wide_mxfp8(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        register_grouped_mxfp8(runtime, true, false)?;
        self.wide_mxfp8 = true;
        Ok(())
    }

    pub(super) fn enable_source_mxfp8(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        register_grouped_mxfp8(runtime, true, true)?;
        self.source_mxfp8 = true;
        Ok(())
    }

    pub(super) fn enable_mxfp8_tensor(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let source = concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n#define INK_GROUP_MXFP8_SOURCE\n#define INK_MX_SOURCE_VECTOR_LOAD\n",
            include_str!("experts.metal"),
            "\n",
            include_str!("mxfp8.metal"),
            "\n",
            include_str!("mxfp8_grouped.metal"),
            "\n",
            include_str!("mxfp8_tensor.metal"),
        );
        let audit_source;
        let source = if mxfp8_tensor_audit_requested() {
            audit_source = format!("#define INK_MX_AUDIT\n{source}");
            audit_source.as_str()
        } else {
            source
        };
        let library = runtime
            .device
            .new_library_with_source(source, Some(&options))
            .map_err(|_| "Inkling MXFP8 tensor library compilation failed")?;
        for name in [
            "ink_mx_weight_norm",
            "ink_mx_tensor_ingress",
            "ink_mx_tensor_product",
            "ink_mx_replay_reset",
            "ink_mx_replay_source",
            "ink_mx_tensor_finish_up",
            "ink_mx_tensor_finish_down",
        ] {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
        let device = &runtime.device;
        self.mxfp8_tensor = Some(Mxfp8Tensor {
            input_norms: zeroed(device, self.capacity * TOP * size_of::<f32>())?,
            masks: zeroed(device, run_slots(self.capacity) * WIDTH * size_of::<u32>())?,
            counts: zeroed(device, MXFP8_DOTS * size_of::<u32>())?,
            false_certificates: zeroed(device, MXFP8_DOTS * size_of::<u32>())?,
        });
        Ok(())
    }

    pub(super) fn mxfp8_tensor_enabled(&self) -> bool {
        self.mxfp8_tensor.is_some()
    }

    pub(super) fn mxfp8_replay_counts(&self) -> Option<Vec<u32>> {
        self.mxfp8_tensor.as_ref().map(|tensor| read_shared(&tensor.counts, MXFP8_DOTS))
    }

    pub(super) fn mxfp8_false_certificates(&self) -> Option<Vec<u32>> {
        self.mxfp8_tensor.as_ref().map(|tensor| read_shared(&tensor.false_certificates, MXFP8_DOTS))
    }

    pub(super) fn enable_source_epilogue(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        let name = "ink_group_gate_up_source";
        if !runtime.pipelines.contains_key(name) {
            let options = MTLCompileOptions::new();
            options.set_math_mode(MTLMathMode::Safe);
            options.set_language_version(MTLLanguageVersion::Version4_0);
            let source = concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n",
                "#define INK_GROUP_SOURCE_EPILOGUE\n#define ink_group_gate_up ink_group_gate_up_source\n",
                include_str!("experts.metal"),
                "\n",
                include_str!("grouped_pilot.metal"),
            );
            let library = runtime
                .device
                .new_library_with_source(source, Some(&options))
                .map_err(|_| "source expert epilogue compilation failed")?;
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
        self.gate_up = name;
        Ok(())
    }

    pub(super) fn enable_source_softmax(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        const SOFTMAX: Kernel = "block_softmax_float32";
        if !runtime.pipelines.contains_key(SOFTMAX) {
            let stock = packed_source_library(&runtime.device)?;
            for kernel in [SOFTMAX, "vs_LogAddExpfloat32", "ink_singleton_route_weights"] {
                let function = stock.new_function_with_name(kernel).ok_or(kernel)?;
                let pipeline =
                    runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| kernel)?;
                runtime.pipelines.insert(kernel, pipeline);
            }

            let options = MTLCompileOptions::new();
            options.set_math_mode(MTLMathMode::Safe);
            options.set_language_version(MTLLanguageVersion::Version4_0);
            let selector = runtime.device.new_library_with_source(concat!(
                "#define INK_PREFILL_LOG_WEIGHTS\n#define ink_prefill_select_source ink_prefill_select_log_weights\n",
                include_str!("prefill_dense_source.metal"),
            ), Some(&options)).map_err(|_| "source log-weight selector compilation failed")?;
            for name in [
                "ink_prefill_select_log_weights",
                "ink_prefill_scale_route_weights",
                "ink_prefill_negate_route_weights",
            ] {
                let function = selector.new_function_with_name(name).ok_or(name)?;
                let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
                runtime.pipelines.insert(name, pipeline);
            }
        }
        self.source_log_weights =
            Some((zeroed(&runtime.device, self.capacity * TOP * 4)?, zeroed(&runtime.device, 4)?));
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
            wide_mxfp8: false,
            source_mxfp8: false,
            mxfp8_tensor: None,
            gate_up: "ink_group_gate_up",
            source_log_weights: None,
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
            && (matches!(layer.experts, Experts::Nvfp4 { .. }) || self.grouped_mxfp8 || self.wide_mxfp8)
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
        source_packed: bool,
    ) {
        assert!(self.supports(layer, rows));
        if source_router && (self.source_log_weights.is_some() || matches!(layer.experts, Experts::Nvfp4 { .. })) {
            self.encode_source_router(gpu, &self.input, &layer.router, &layer.bias, rows);
        } else {
            self.encode_router(gpu, &self.input, &layer.router, &layer.bias, rows);
        }
        let bucket = if matches!(layer.experts, Experts::Nvfp4 { .. }) && active_rows64(gpu) {
            EXPERIMENT_BUCKET
        } else {
            "ink_group_bucket"
        };
        self.encode_buckets(gpu, &self.ids, rows, bucket);
        match &layer.experts {
            Experts::Nvfp4 {
                up,
                down,
            } => {
                self.encode_projected(gpu, &self.input, up, down, rows, self.active_order(), source_packed, None);
            },
            Experts::Mxfp8 {
                up,
                down,
            } => self.encode_mxfp8(gpu, up, down, rows, source_packed),
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

    fn encode_source_router(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        router: &Buffer,
        bias: &Buffer,
        rows: usize,
    ) {
        gpu.one(
            "ink_prefill_router_source",
            &[input, router, &self.logits],
            rows as u32,
            (EXPERTS / 32, rows.div_ceil(16)),
            32,
        );
        let select = if self.source_log_weights.is_some() {
            "ink_prefill_select_log_weights"
        } else {
            "ink_prefill_select_source"
        };
        let route_weights = self.source_log_weights.as_ref().map_or(&self.weights, |(weights, _)| weights);
        gpu.one(select, &[&self.logits, bias, &self.ids, route_weights], (), (rows, 1), EXPERTS);
        if let Some((log_weights, zero)) = &self.source_log_weights {
            gpu.one("vs_LogAddExpfloat32", &[log_weights, zero, log_weights], (rows * TOP) as u32, (rows, 1), TOP);
            gpu.one("ink_prefill_negate_route_weights", &[log_weights], (), (rows, 1), TOP);
            gpu.one("block_softmax_float32", &[log_weights, &self.weights], TOP as i32, (rows, 1), 32);
            gpu.one(
                "ink_prefill_scale_route_weights",
                &[&self.weights],
                (rows * TOP) as u32,
                ((rows * TOP).div_ceil(128), 1),
                128,
            );
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
        kernel: Kernel,
    ) {
        gpu.one(kernel, &[ids, &self.order, &self.runs], (rows * TOP) as u32, (1, 1), EXPERTS);
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
        gpu.phase(PrefillPhase::Nvfp4Quant);
        gpu.one(
            "ink_group_ingress",
            &[inner, routing.indices, &self.activation, &self.quantum, &self.maxima],
            [width as u32, u32::from(gather)],
            ((width / 32).div_ceil(128), rows * TOP),
            128,
        );
        gpu.phase(if gather {
            PrefillPhase::Nvfp4UpDot
        } else {
            PrefillPhase::Nvfp4DownDot
        });
        let buffers =
            [&self.activation, &bank.codes, &bank.blocks, routing.runs, &self.quantum, &self.maxima, &self.projected];
        let grid = if active_rows64(gpu) {
            (WIDTH / 32, run_slots_for(rows, 64))
        } else {
            (WIDTH / 64, run_slots(rows))
        };
        if let Some(probe) = &gpu.runtime.nvfp4_probe
            && gpu.nvfp4_dots < PROBE_DOTS
        {
            let ordinal = gpu.nvfp4_dots;
            gpu.nvfp4_dots += 1;
            let buffers =
                [buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6], &probe.counts];
            if let Some(sample) = PROBE_SAMPLE.iter().position(|&dot| dot == ordinal) {
                let phases =
                    [PrefillPhase::Nvfp4ProbeExact, PrefillPhase::Nvfp4ProbeGuard, PrefillPhase::Nvfp4ProbeBare];
                let order = if probe.reverse {
                    [2, 1, 0]
                } else {
                    [0, 1, 2]
                };
                for mode in order {
                    gpu.phase(phases[mode]);
                    let slot = PROBE_DOTS + sample * 3 + mode;
                    gpu.one(
                        "ink_group_qmm_probe",
                        &buffers,
                        [width as u32, mode as u32, (slot * PROBE_FIELDS) as u32],
                        grid,
                        128,
                    );
                }
                gpu.phase(if gather {
                    PrefillPhase::Nvfp4UpDot
                } else {
                    PrefillPhase::Nvfp4DownDot
                });
            }
            // The regular result overwrites every shadow result before the epilogue reads it.
            gpu.one("ink_group_qmm_probe", &buffers, [width as u32, 0, (ordinal * PROBE_FIELDS) as u32], grid, 128);
        } else {
            let kernel = if gpu.runtime.source_text_active && gpu.runtime.nvfp4_probe.is_none() {
                gpu.runtime.grouped_qmm
            } else {
                "ink_group_qmm"
            };
            gpu.one(kernel, &buffers, width as u32, grid, 128);
        }
    }

    fn encode_projected(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        up: &Bank,
        down: &Bank,
        rows: usize,
        routing: ExpertOrder<'_>,
        source_packed: bool,
        up_dot_tap: Option<&Buffer>,
    ) {
        let out = &self.output;
        gpu.phase(PrefillPhase::Nvfp4Quant);
        gpu.one("ink_nvfp4_activation", &[input, &self.up_inner, &self.up_max], WIDTH as u32, (rows, 1), 256);
        self.encode_dot(gpu, up, &self.up_inner, &routing, rows, WIDTH, true);
        if let Some(tap) = up_dot_tap {
            let words = rows * TOP * WIDTH;
            gpu.one("ink_trace_copy", &[&self.projected, tap], words as u32, (words.div_ceil(128), 1), 128);
        }
        gpu.phase(PrefillPhase::Nvfp4Epilogue);
        gpu.one(
            if source_packed {
                self.gate_up
            } else {
                "ink_group_gate_up"
            },
            &[&self.projected, &self.up_max, routing.ids, routing.weights, routing.indices, &up.maxima, &out.middle],
            up.denominator,
            (MID / 128, rows * TOP),
            128,
        );
        gpu.phase(PrefillPhase::Nvfp4Quant);
        gpu.one(
            "ink_nvfp4_activation",
            &[&out.middle, &self.down_inner, &self.down_max],
            MID as u32,
            (rows * TOP, 1),
            256,
        );
        self.encode_dot(gpu, down, &self.down_inner, &routing, rows, MID, false);
        gpu.phase(PrefillPhase::Nvfp4Epilogue);
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
        source_packed: bool,
    ) {
        let out = &self.output;
        let slots = run_slots(rows);
        let [up_kernel, down_kernel] = grouped_mxfp8_names(self.wide_mxfp8, self.source_mxfp8 && source_packed);
        let tensor = self.mxfp8_tensor.as_ref().filter(|_| {
            source_packed && gpu.runtime.source_text_active && gpu.runtime.mxfp8_tensor_active && rows > 64
        });
        let up_buffers =
            [&self.up_inner, &self.order, &self.runs, &self.weights, &up.codes, &up.exponents, &out.middle];
        let down_buffers = [&self.down_inner, &self.order, &self.runs, &down.codes, &down.exponents, &out.down];

        gpu.one("ink_mxfp8_activation", &[&self.input, &self.up_inner], WIDTH as u32, (WIDTH / 32, rows), 32);
        gpu.phase(PrefillPhase::Mxfp8Up);
        if let Some(tensor) = tensor {
            self.encode_mxfp8_tensor(gpu, tensor, up, &self.up_inner, rows, WIDTH);
            gpu.phase(PrefillPhase::Mxfp8);
            gpu.one(
                "ink_mx_tensor_finish_up",
                &[&self.projected, &self.order, &self.weights, &out.middle],
                (),
                (MID.div_ceil(128), rows * TOP),
                128,
            );
        } else {
            gpu.one(up_kernel, &up_buffers, (), (MID / 4, slots), 128);
        }

        gpu.phase(PrefillPhase::Mxfp8);
        gpu.one("ink_mxfp8_activation", &[&out.middle, &self.down_inner], MID as u32, (MID / 32, rows * TOP), 32);
        gpu.phase(PrefillPhase::Mxfp8Down);
        if let Some(tensor) = tensor {
            self.encode_mxfp8_tensor(gpu, tensor, down, &self.down_inner, rows, MID);
            gpu.phase(PrefillPhase::Mxfp8);
            gpu.one(
                "ink_mx_tensor_finish_down",
                &[&self.projected, &self.order, &out.down],
                (),
                (WIDTH.div_ceil(128), rows * TOP),
                128,
            );
        } else {
            gpu.one(down_kernel, &down_buffers, (), (WIDTH / 4, slots), 128);
        }
        gpu.phase(PrefillPhase::Mxfp8);
        gpu.one("ink_group_combine", &[&out.down, &out.merged], (), (WIDTH / 128, rows), 128);
    }

    fn encode_mxfp8_tensor(
        &self,
        gpu: &mut Stream<'_>,
        tensor: &Mxfp8Tensor,
        bank: &Mxfp8Bank,
        inner: &Buffer,
        rows: usize,
        width: usize,
    ) {
        let norms = bank.tensor_norms.as_ref().expect("MXFP8 tensor norms prepared at startup");
        let routes = rows * TOP;
        let phase = gpu.mxfp8_dots;
        assert!(phase < MXFP8_DOTS);
        gpu.mxfp8_dots += 1;
        let count = (gpu.runtime.mxfp8_tensor_profile).then_some(phase);
        if let Some(index) = count {
            gpu.dispatch("ink_mx_replay_reset", &[(&tensor.counts, index * size_of::<u32>())], (), (1, 1), 1);
        }
        gpu.one(
            "ink_mx_tensor_ingress",
            &[inner, &self.order, &self.activation, &tensor.input_norms],
            [width as u32, routes as u32],
            (routes.div_ceil(4), 1),
            128,
        );
        gpu.one(
            "ink_mx_tensor_product",
            &[
                &self.activation,
                &self.runs,
                &bank.codes,
                &bank.exponents,
                &tensor.input_norms,
                norms,
                &self.projected,
                &tensor.masks,
                &tensor.counts,
            ],
            [width as u32, count.map_or(0, |index| index as u32 + 1)],
            (WIDTH / 64, run_slots(rows)),
            128,
        );
        gpu.phase(PrefillPhase::Mxfp8Replay);
        gpu.one(
            "ink_mx_replay_source",
            &[
                inner,
                &self.order,
                &self.runs,
                &bank.codes,
                &bank.exponents,
                &self.projected,
                &tensor.masks,
                &tensor.false_certificates,
            ],
            [width as u32, phase as u32],
            (WIDTH / 4, run_slots(rows)),
            128,
        );
    }
}

fn grouped_mxfp8_names(
    wide: bool,
    source: bool,
) -> [&'static str; 2] {
    if source {
        ["ink_group_mxfp8_source_up", "ink_group_mxfp8_source_down"]
    } else if wide {
        ["ink_group_mxfp8_wide_up", "ink_group_mxfp8_wide_down"]
    } else {
        ["ink_group_mxfp8_up", "ink_group_mxfp8_down"]
    }
}

fn register_grouped_mxfp8(
    runtime: &mut Runtime,
    wide: bool,
    source_mode: bool,
) -> Result<()> {
    let names = grouped_mxfp8_names(wide, source_mode);
    if names.iter().all(|name| runtime.pipelines.contains_key(name)) {
        return Ok(());
    }
    let source = concat!(
        "#include <metal_stdlib>\nusing namespace metal;\n",
        include_str!("experts.metal"),
        "\n",
        include_str!("mxfp8.metal"),
        "\n",
        include_str!("mxfp8_grouped.metal")
    );
    let wide_source;
    let source = if source_mode {
        wide_source = format!(
            "#define INK_GROUP_MXFP8_WIDE\n#define INK_GROUP_MXFP8_SOURCE\n#define ink_group_mxfp8_wide_up ink_group_mxfp8_source_up\n#define ink_group_mxfp8_wide_down ink_group_mxfp8_source_down\n{source}"
        );
        wide_source.as_str()
    } else if wide {
        wide_source = format!("#define INK_GROUP_MXFP8_WIDE\n{source}");
        wide_source.as_str()
    } else {
        source
    };
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let library = runtime
        .device
        .new_library_with_source(source, Some(&options))
        .map_err(|_| "Inkling grouped MXFP8 shader compilation failed")?;
    for name in names {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    Ok(())
}

pub(super) fn enable_shared_scales(runtime: &mut Runtime) -> Result<()> {
    let kernel = "ink_group_qmm_shared_scales";
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let source = concat!(
        "#include <metal_stdlib>\nusing namespace metal;\n",
        "#define INK_GROUP_QMM_SHARED_SCALES\n#define ink_group_qmm ink_group_qmm_shared_scales\n",
        include_str!("experts.metal"),
        "\n",
        include_str!("grouped_pilot.metal")
    );
    let library = runtime
        .device
        .new_library_with_source(source, Some(&options))
        .map_err(|_| "Inkling shared-scale NVFP4 compilation failed")?;
    let function = library.new_function_with_name(kernel).ok_or(kernel)?;
    let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| kernel)?;
    runtime.pipelines.insert(kernel, pipeline);
    runtime.grouped_qmm = kernel;
    Ok(())
}

pub(super) fn enable_rows64(runtime: &mut Runtime) -> Result<()> {
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let shader = concat!(include_str!("experts.metal"), "\n", include_str!("grouped_pilot.metal"));
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n#define INK_GROUP_QMM_REUSE_BOUND\n#define INK_GROUP_QMM_TILE_ROWS 64\n#define INK_GROUP_QMM_TILE_COLS 32\n#define ink_group_qmm {ROWS64_QMM}\n#define ink_group_bucket {EXPERIMENT_BUCKET}\n{shader}"
    );
    let library = runtime
        .device
        .new_library_with_source(&source, Some(&options))
        .map_err(|_| "Inkling 64-row NVFP4 compilation failed")?;
    for name in [ROWS64_QMM, EXPERIMENT_BUCKET] {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    runtime.grouped_qmm = ROWS64_QMM;
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
    layer: usize,
    suffix: &str,
    width: usize,
) -> Result<Bank> {
    Bank::load(source, device, &format!("model.llm.layers.{layer}.mlp.experts.{suffix}"), width)
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
        grouped.encode_buckets(gpu, routing.ids, rows, "ink_group_bucket");
        grouped.encode_projected(gpu, &self.input, up, down, rows, routing, false, None);
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
    let up = ink_bank(&source, &runtime.device, PHYSICAL, "w13", WIDTH)?;
    let down = ink_bank(&source, &runtime.device, PHYSICAL, "w2", MID)?;
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

/// Replay the first matched prefill visit with two sets of routing weights.
pub fn probe_prefill_expert_weights(
    checkpoint: &std::path::Path,
    oracle: &std::path::Path,
) -> Result<Value> {
    const ROWS: usize = 512;
    const BATCH: usize = 64;
    let capture = Tensors::open(&oracle.join("prefill.safetensors"))?;
    let (physical, focus) = match capture.values::<u32>("chunk", "U32", &[3])?.as_slice() {
        [0, 512, 0] => (0, 52),
        [0, 512, 4] => (4, 204),
        _ => return Err("expected a chunk-zero 512-row Inkling expert trace at visit 0 or 4"),
    };
    let input = capture.values::<u16>("stage.mlp_input", "BF16", &[1, ROWS, WIDTH])?;
    let expected_logits = capture.values::<f32>("stage.trace.router_logits", "F32", &[ROWS, EXPERTS])?;
    let expected_ids = capture.values::<u32>("stage.routes", "U32", &[ROWS, TOP])?;
    let expected_weights = capture.values::<f32>("stage.route_weights", "F32", &[ROWS, TOP])?;
    let expected_log_weights = capture
        .contains("stage.log_weights")
        .then(|| capture.values::<f32>("stage.log_weights", "F32", &[ROWS, TOP]))
        .transpose()?;
    let expected = capture.values::<u16>("stage.experts", "BF16", &[1, ROWS, WIDTH])?;

    let mut runtime = Runtime::new()?;
    let mut grouped = GroupedExperts::new(&mut runtime, BATCH)?;
    if std::env::var_os("INKLING_SOURCE_EPILOGUE").is_some() {
        grouped.enable_source_epilogue(&mut runtime)?;
    }
    if std::env::var_os("INKLING_SOURCE_SOFTMAX").is_some() {
        grouped.enable_source_softmax(&mut runtime)?;
    }
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let library = runtime
        .device
        .new_library_with_source(include_str!("prefill_dense_source.metal"), Some(&options))
        .map_err(|_| "Inkling source router library compilation failed")?;
    for name in ["ink_prefill_router_source", "ink_prefill_select_source"] {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    let weights = Tensors::open(checkpoint)?;
    let device = &runtime.device;
    let mut stage_trace = capture
        .contains("stage.expert_order")
        .then(|| ExpertStageTrace::new(&capture, device, ROWS, BATCH))
        .transpose()?;
    let prefix = format!("model.llm.layers.{physical}.mlp.gate");
    let router = weights.buffer(device, &format!("{prefix}.weight"), "BF16", &[EXPERTS, WIDTH])?;
    let bias = weights.buffer(device, &format!("{prefix}.bias"), "F32", &[EXPERTS])?;
    let up = ink_bank(&weights, device, physical, "w13", WIDTH)?;
    let down = ink_bank(&weights, device, physical, "w2", MID)?;

    let mut logits = Vec::with_capacity(ROWS * EXPERTS);
    let mut aligned_weights = Vec::with_capacity(ROWS * TOP);
    let mut aligned_log_weights = Vec::new();
    let mut native_sums = Vec::with_capacity(ROWS * WIDTH);
    let mut injected_sums = Vec::with_capacity(ROWS * WIDTH);
    let mut reordered_rows = 0;
    for start in (0..ROWS).step_by(BATCH) {
        let rows = start..start + BATCH;
        let route_range = rows.start * TOP..rows.end * TOP;
        let source_ids = &expected_ids[route_range.clone()];
        let input = upload(device, &input[rows.start * WIDTH..rows.end * WIDTH])?;
        execute(&runtime, |gpu| grouped.encode_source_router(gpu, &input, &router, &bias, BATCH))?;
        logits.extend(read_shared::<f32>(&grouped.logits, BATCH * EXPERTS));
        let actual_ids = read_shared::<u32>(&grouped.ids, BATCH * TOP);
        let actual_weights = read_shared::<f32>(&grouped.weights, BATCH * TOP);
        let actual_log_weights =
            grouped.source_log_weights.as_ref().map(|(weights, _)| read_shared::<f32>(weights, BATCH * TOP));
        let first_weight = aligned_weights.len();
        for (row, (source_row, actual_row)) in
            source_ids.chunks_exact(TOP).zip(actual_ids.chunks_exact(TOP)).enumerate()
        {
            reordered_rows += usize::from(source_row != actual_row);
            let mut seen = 0u32;
            for id in source_row {
                let slot = actual_row.iter().position(|actual| actual == id).ok_or("Inkling expert ID set differs")?;
                if seen & (1 << slot) != 0 {
                    return Err("duplicate source route in Inkling expert trace");
                }
                seen |= 1 << slot;
                aligned_weights.push(actual_weights[row * TOP + slot]);
                if let Some(values) = &actual_log_weights {
                    aligned_log_weights.push(values[row * TOP + slot]);
                }
            }
        }
        let ids = upload(device, source_ids)?;
        for (scores, sums, trace) in [
            (&aligned_weights[first_weight..], &mut native_sums, None),
            (&expected_weights[route_range], &mut injected_sums, stage_trace.as_mut()),
        ] {
            let scores = upload(device, scores)?;
            execute(&runtime, |gpu| {
                grouped.encode_buckets(gpu, &ids, BATCH, "ink_group_bucket");
                let order = ExpertOrder {
                    ids: &ids,
                    weights: &scores,
                    indices: &grouped.order,
                    runs: &grouped.runs,
                };
                grouped.encode_projected(
                    gpu,
                    &input,
                    &up,
                    &down,
                    BATCH,
                    order,
                    true,
                    trace.as_ref().map(|t| &t.up_dot_tap),
                );
            })?;
            sums.extend(read_shared::<u16>(grouped.output(), BATCH * WIDTH));
            if let Some(trace) = trace {
                trace.compare_batch(&grouped, start, BATCH, focus)?;
            }
        }
    }

    let row = focus;
    let route_row = row * TOP..(row + 1) * TOP;
    let output_row = row * WIDTH..(row + 1) * WIDTH;
    let compare = |actual: &[u16], want: &[u16]| {
        json!({
            "all": expert_probe_summary(actual, want),
            "focus_row": expert_probe_summary(&actual[output_row.clone()], &want[output_row.clone()]),
        })
    };
    let mut report = json!({
        "visit": physical, "physical_layer": physical, "chunk_rows": ROWS, "native_expert_batch_rows": BATCH,
        "focus_row_index": focus,
        "native_log_weights_in_source_order": expected_log_weights.as_ref().filter(|values| values.len() == aligned_log_weights.len())
            .map(|values| expert_probe_summary(&aligned_log_weights, values)),
        "router": "source-shaped native Metal logits and selection",
        "expert_id_sets_equal": true, "route_order_different_rows": reordered_rows,
        "native_router_logits": expert_probe_summary(&logits, &expected_logits),
        "native_weights_in_source_order": {
            "all": expert_probe_summary(&aligned_weights, &expected_weights),
            "focus_row": expert_probe_summary(&aligned_weights[route_row.clone()], &expected_weights[route_row]),
        },
        "native_weights_vs_reference": compare(&native_sums, &expected),
        "reference_weights_vs_reference": compare(&injected_sums, &expected),
        "native_vs_reference_weights": compare(&native_sums, &injected_sums),
    });
    if let Some(trace) = stage_trace {
        report["reference_weight_stage_comparison"] = trace.finish();
    }
    Ok(report)
}

struct ExpertStageTrace {
    up_dot_tap: Buffer,
    up_inner: ExpertStage<f32>,
    up_max: ExpertStage<f32>,
    up_dot: ExpertStage<f32>,
    mid: ExpertStage<u16>,
    down_inner: ExpertStage<f32>,
    down_max: ExpertStage<f32>,
    down_dot: ExpertStage<f32>,
    down_raw: ExpertStage<u16>,
}

impl ExpertStageTrace {
    fn new(
        capture: &Tensors,
        device: &Device,
        rows: usize,
        batch: usize,
    ) -> Result<Self> {
        let routes = rows * TOP;
        let order = capture.values::<u32>("stage.expert_order", "U32", &[routes])?;
        let inverse = invert_expert_order(&order)?;
        let token_f32 = |name, width| ExpertStage::new(capture, name, "F32", rows, width, None);
        let route_f32 = |name, width| ExpertStage::new(capture, name, "F32", routes, width, Some(&inverse));
        let route_bf16 = |name, width| ExpertStage::new(capture, name, "BF16", routes, width, Some(&inverse));
        Ok(Self {
            up_dot_tap: zeroed(device, batch * TOP * WIDTH * 4)?,
            up_inner: token_f32("stage.up_inner", WIDTH)?,
            up_max: token_f32("stage.up_max", 1)?,
            up_dot: route_f32("stage.up_dot", WIDTH)?,
            mid: route_bf16("stage.mid", MID)?,
            down_inner: route_f32("stage.down_inner", MID)?,
            down_max: route_f32("stage.down_max", 1)?,
            down_dot: route_f32("stage.down_dot", WIDTH)?,
            down_raw: route_bf16("stage.down_raw", WIDTH)?,
        })
    }

    fn compare_batch(
        &mut self,
        grouped: &GroupedExperts,
        start: usize,
        rows: usize,
        focus: usize,
    ) -> Result<()> {
        let inverse = invert_expert_order(&read_shared::<u32>(&grouped.order, rows * TOP))?;
        self.up_inner.compare(&grouped.up_inner, start, rows, focus, None);
        self.up_max.compare(&grouped.up_max, start, rows, focus, None);
        self.up_dot.compare(&self.up_dot_tap, start, rows, focus, Some(&inverse));
        self.mid.compare(&grouped.output.middle, start, rows, focus, Some(&inverse));
        self.down_inner.compare(&grouped.down_inner, start, rows, focus, Some(&inverse));
        self.down_max.compare(&grouped.down_max, start, rows, focus, Some(&inverse));
        self.down_dot.compare(&grouped.projected, start, rows, focus, Some(&inverse));
        self.down_raw.compare(&grouped.output.down, start, rows, focus, None);
        Ok(())
    }

    fn finish(self) -> Value {
        json!({
            "up_inner": self.up_inner.finish(), "up_max": self.up_max.finish(), "up_dot": self.up_dot.finish(),
            "mid": self.mid.finish(), "down_inner": self.down_inner.finish(), "down_max": self.down_max.finish(),
            "down_dot": self.down_dot.finish(), "down_raw": self.down_raw.finish(),
        })
    }
}

fn invert_expert_order(order: &[u32]) -> Result<Vec<usize>> {
    let mut inverse = vec![usize::MAX; order.len()];
    for (sorted, &route) in order.iter().enumerate() {
        let index = inverse.get_mut(route as usize).ok_or("Inkling expert order route is out of range")?;
        if *index != usize::MAX {
            return Err("Inkling expert order contains a duplicate route");
        }
        *index = sorted;
    }
    Ok(inverse)
}

fn undo_expert_order<T: Copy>(
    values: &[T],
    width: usize,
    inverse: &[usize],
) -> Vec<T> {
    inverse.iter().flat_map(|&row| values[row * width..(row + 1) * width].iter().copied()).collect()
}

struct ExpertStage<T> {
    expected: Vec<T>,
    row_width: usize,
    actual_hash: Sha256,
    unequal: usize,
    first_row: Option<usize>,
    focus: Value,
}

impl<T: bytemuck::Pod> ExpertStage<T> {
    fn new(
        capture: &Tensors,
        name: &str,
        dtype: &str,
        rows: usize,
        width: usize,
        inverse: Option<&[usize]>,
    ) -> Result<Self> {
        let flat = if width == 1 {
            capture.values(name, dtype, &[rows])
        } else {
            capture.values(name, dtype, &[rows, width])
        }
        .or_else(|_| capture.values(name, dtype, &[rows, 1, width]))
        .or_else(|_| capture.values(name, dtype, &[rows, 1, 1, width]))?;
        Ok(Self {
            expected: match inverse {
                Some(order) => undo_expert_order(&flat, width, order),
                None => flat,
            },
            row_width: if inverse.is_some() {
                TOP * width
            } else {
                width
            },
            actual_hash: Sha256::new(),
            unequal: 0,
            first_row: None,
            focus: Value::Null,
        })
    }

    fn compare(
        &mut self,
        buffer: &Buffer,
        start: usize,
        rows: usize,
        focus: usize,
        inverse: Option<&[usize]>,
    ) {
        let actual = read_shared::<T>(buffer, rows * self.row_width);
        let actual = match inverse {
            Some(order) => undo_expert_order(&actual, self.row_width / TOP, order),
            None => actual,
        };
        let expected = &self.expected[start * self.row_width..(start + rows) * self.row_width];
        let actual_bytes = bytemuck::cast_slice::<T, u8>(&actual);
        let expected_bytes = bytemuck::cast_slice::<T, u8>(expected);
        self.actual_hash.update(actual_bytes);
        for (index, (a, b)) in
            actual_bytes.chunks_exact(size_of::<T>()).zip(expected_bytes.chunks_exact(size_of::<T>())).enumerate()
        {
            if a != b {
                self.unequal += 1;
                self.first_row.get_or_insert(start + index / self.row_width);
            }
        }
        if (start..start + rows).contains(&focus) {
            let range = (focus - start) * self.row_width..(focus - start + 1) * self.row_width;
            self.focus = expert_probe_summary(&actual[range.clone()], &expected[range]);
        }
    }

    fn finish(self) -> Value {
        json!({"all": {"values": self.expected.len(), "unequal": self.unequal, "first_row": self.first_row,
            "actual_sha256": format!("{:x}", self.actual_hash.finalize()),
            "expected_sha256": format!("{:x}", Sha256::digest(bytemuck::cast_slice::<T, u8>(&self.expected)))},
            "focus_row": self.focus})
    }
}

fn expert_probe_summary<T: bytemuck::Pod>(
    actual: &[T],
    expected: &[T],
) -> Value {
    let size = std::mem::size_of::<T>();
    let actual_bytes = bytemuck::cast_slice::<T, u8>(actual);
    let expected_bytes = bytemuck::cast_slice::<T, u8>(expected);
    let unequal =
        actual_bytes.chunks_exact(size).zip(expected_bytes.chunks_exact(size)).filter(|(a, b)| a != b).count();
    json!({
        "values": actual.len(), "unequal": unequal,
        "actual_sha256": format!("{:x}", Sha256::digest(actual_bytes)),
        "expected_sha256": format!("{:x}", Sha256::digest(expected_bytes)),
    })
}
