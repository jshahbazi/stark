// SPDX-License-Identifier: Apache-2.0
//! Pack independent rows for the source-order controller expansion.

use super::*;

const NORM_BATCH: usize = 4;
const QKV_SLOTS: usize = 3;
const QKV_ROWS: usize = QKV_SLOTS * NORM_BATCH;
const QKV_INPUTS: usize = QKV_ROWS + NORM_BATCH;
const CONTROLLER_RANK: usize = 64;
const NORM_KERNELS: [&str; 3] =
    ["ink_prefill_controller_norm_pair", "ink_prefill_controller_pack_rank", "ink_prefill_controller_norm_output"];

pub(super) struct PrefillControllers {
    input_message: Buffer,
    compressed: Buffer,
    gated: Buffer,
    rows: usize,
    qkv: Option<QkvControllers>,
}

struct QkvControllers {
    raw: Buffer,
    normalized: Buffer,
    rank: Buffer,
    mixed: Buffer,
}

impl PrefillControllers {
    pub(super) fn new(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        let device = &runtime.device;
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = device
            .new_library_with_source(include_str!("prefill_controller.metal"), Some(&options))
            .map_err(|_| "Inkling prefill controller shader compilation failed")?;
        for name in ["ink_prefill_controller_pack", "ink_prefill_controller_expand"] {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
        Ok(Self {
            input_message: zeroed(device, rows * 2 * WIDTH * 4)?,
            compressed: zeroed(device, rows * 64 * 2)?,
            gated: zeroed(device, rows * WIDTH * 4)?,
            rows,
            qkv: None,
        })
    }

    pub(super) fn enable_verify_qkv_batch(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        packed_prefill::register(
            runtime,
            concat!(
                "#define INK_QKV_NORM\n",
                include_str!("long_position.metal"),
                include_str!("serial.metal"),
                "\n",
                include_str!("verify_qkv_controller.metal")
            ),
            &["ink_verify_qkv_norm_pair", "ink_verify_qkv_norm_output"],
        )?;
        packed_prefill::register(
            runtime,
            concat!(
                "#define INK_QKV_RANK\n",
                include_str!("prefill.metal"),
                "\n",
                include_str!("verify_qkv_controller.metal")
            ),
            &["ink_verify_qkv_rank"],
        )?;
        packed_prefill::register(
            runtime,
            concat!(
                "#define INK_QKV_EXPAND\n",
                include_str!("verify_controller.metal"),
                "\n",
                include_str!("verify_qkv_controller.metal")
            ),
            &["ink_verify_qkv_expand"],
        )?;
        let device = &runtime.device;
        self.qkv = Some(QkvControllers {
            raw: zeroed(device, QKV_ROWS * 2 * WIDTH * 4)?,
            normalized: zeroed(device, QKV_ROWS * 2 * WIDTH * 2)?,
            rank: zeroed(device, QKV_ROWS * CONTROLLER_RANK * 2)?,
            mixed: zeroed(device, QKV_ROWS * WIDTH * 4)?,
        });
        Ok(())
    }

    pub(super) fn has_verify_qkv(&self) -> bool {
        self.qkv.is_some()
    }

    pub(super) fn try_encode_verify_qkv<'a>(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        state: VisitInput,
        rows: impl Iterator<Item = (&'a Flow, &'a Scratch)>,
    ) -> bool {
        let Some(packed) = &self.qkv else {
            return false;
        };
        let rows: Vec<_> = rows.collect();
        let count = rows.len();
        debug_assert!((1..=NORM_BATCH).contains(&count));
        // Match the single-slot kernels' padded bindings. Only live rows write outputs.
        let at = |index: usize| rows[index.min(count - 1)];
        let inputs: [Binding<'_>; QKV_INPUTS + 2] = std::array::from_fn(|i| match i {
            i if i < QKV_ROWS => {
                let (flow, scratch) = at(i % NORM_BATCH);
                layer.attention_norm_input(&flow.partial, scratch, state, i / NORM_BATCH)
            },
            i if i < QKV_INPUTS => binding(&at(i - QKV_ROWS).1.messages),
            QKV_INPUTS => binding(&packed.raw),
            _ => binding(&packed.normalized),
        });
        gpu.dispatch("ink_verify_qkv_norm_pair", &inputs, count as u32, (2, QKV_SLOTS * count), WIDTH / 4);
        let [q, k, v] = [&layer.gates[0], &layer.gates[1], &layer.gates[2]];
        gpu.one(
            "ink_verify_qkv_rank",
            &[&q.down, &k.down, &v.down, &packed.normalized, &packed.rank],
            count as u32,
            (CONTROLLER_RANK / 4, QKV_SLOTS),
            128,
        );
        gpu.one(
            "ink_verify_qkv_expand",
            &[&packed.rank, &q.gain, &k.gain, &v.gain, &q.forget, &k.forget, &v.forget, &packed.raw, &packed.mixed],
            count as u32,
            (WIDTH / 16, QKV_SLOTS),
            32,
        );
        let outputs: [Binding<'_>; 1 + QKV_SLOTS + QKV_ROWS] = std::array::from_fn(|i| match i {
            0 => binding(&packed.mixed),
            1..=QKV_SLOTS => binding(&layer.qkv_norm[i - 1]),
            _ => {
                let index = i - (1 + QKV_SLOTS);
                binding(&at(index % NORM_BATCH).1.qkv_input[index / NORM_BATCH])
            },
        });
        gpu.dispatch("ink_verify_qkv_norm_output", &outputs, count as u32, (QKV_SLOTS * count, 1), WIDTH / 4);
        true
    }

    pub(super) fn compile_batched_norm(runtime: &mut Runtime) -> Result<()> {
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = runtime
            .device
            .new_library_with_source(
                concat!(
                    include_str!("long_position.metal"),
                    include_str!("serial.metal"),
                    "\n",
                    include_str!("prefill_controller_norm.metal")
                ),
                Some(&options),
            )
            .map_err(|_| "Inkling prefill batch RMS shader compilation failed")?;
        for name in NORM_KERNELS {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
        Ok(())
    }

    pub(super) fn encode<'a>(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        slot: usize,
        rows: impl Iterator<Item = (&'a Scratch, Binding<'a>, Binding<'a>)> + Clone,
        count: usize,
        wide_gemv: bool,
        batched_norm: bool,
        single_native: bool,
    ) {
        let scale = if slot == 3 {
            &layer.mlp_norm
        } else {
            &layer.qkv_norm[slot]
        };
        let gate = &layer.gates[slot];
        for start in (0..count).step_by(self.rows) {
            let size = (count - start).min(self.rows);
            let tile = rows.clone().skip(start).take(size);
            let staged = batched_norm.then(|| tile.clone().collect::<Vec<_>>());
            if let Some(staged) = &staged {
                for (batch, rows) in staged.chunks(NORM_BATCH).enumerate() {
                    let packed = (&self.input_message, batch * NORM_BATCH * 2 * WIDTH * 4);
                    let mut buffers = [packed; 3 * NORM_BATCH + 1];
                    for row in 0..NORM_BATCH {
                        let (scratch, input, _) = rows.get(row).copied().unwrap_or(rows[0]);
                        buffers[row] = input;
                        buffers[NORM_BATCH + row] = (&scratch.messages, slot * WIDTH * 4);
                        buffers[2 * NORM_BATCH + row] = binding(&scratch.controller_pair);
                    }
                    gpu.dispatch(NORM_KERNELS[0], &buffers, (), (2, rows.len()), WIDTH / 4);
                }
            } else {
                for (scratch, input, _) in tile.clone() {
                    let message = (&scratch.messages, slot * WIDTH * 4);
                    gpu.norm(input, scale, binding(&scratch.controller_pair), false);
                    gpu.norm(message, scale, (&scratch.controller_pair, WIDTH * 2), false);
                }
            }
            // The single-row source decode takes the verifier's native
            // rank and expansion products with one live row.
            let native =
                (gpu.runtime.verify_nvfp4_matvec && gpu.runtime.verify_combined_short && (2..=4).contains(&size))
                    || (single_native && size == 1);
            if let Some(staged) = staged.as_ref().filter(|_| native) {
                let mut bindings = [binding(&gate.down); 9];
                for row in 0..NORM_BATCH {
                    let live = row.min(size - 1);
                    bindings[row + 1] = binding(&staged[live].0.controller_pair);
                    bindings[row + 5] = (&self.compressed, live * 64 * 2);
                }
                gpu.dispatch("ink_matvec_prefill", &bindings, [(2 * WIDTH) as u32, 64, size as u32], (16, 1), 128);
            } else {
                project_rows(
                    gpu,
                    None,
                    &gate.down,
                    tile.clone().map(|(s, _, _)| (binding(&s.controller_pair), &s.compressed)),
                    2 * WIDTH,
                    64,
                    wide_gemv,
                );
                if let Some(staged) = &staged {
                    for (batch, rows) in staged.chunks(NORM_BATCH).enumerate() {
                        let packed = (&self.compressed, batch * NORM_BATCH * 64 * 2);
                        let mut buffers = [packed; NORM_BATCH + 1];
                        for (row, buffer) in buffers[..NORM_BATCH].iter_mut().enumerate() {
                            *buffer = binding(&rows.get(row).unwrap_or(&rows[0]).0.compressed);
                        }
                        gpu.dispatch(NORM_KERNELS[1], &buffers, (), (rows.len(), 1), 64);
                    }
                } else {
                    for (row, (scratch, input, _)) in tile.clone().enumerate() {
                        gpu.dispatch(
                            "ink_prefill_controller_pack",
                            &[
                                input,
                                (&scratch.messages, slot * WIDTH * 4),
                                binding(&scratch.compressed),
                                (&self.input_message, row * 2 * WIDTH * 4),
                                (&self.compressed, row * 64 * 2),
                            ],
                            (),
                            (WIDTH / 128, 1),
                            128,
                        );
                    }
                }
            }
            let (kernel, columns, tiles) = if native {
                ("ink_verify_controller_native_m16n16", 16, 1)
            } else {
                ("ink_prefill_controller_expand", 32, size.div_ceil(16))
            };
            gpu.one(
                kernel,
                &[&self.compressed, &gate.gain, &gate.forget, &self.input_message, &self.gated],
                size as u32,
                (WIDTH / columns, tiles),
                32,
            );
            if let Some(staged) = &staged {
                for (batch, rows) in staged.chunks(NORM_BATCH).enumerate() {
                    let gated = (&self.gated, batch * NORM_BATCH * WIDTH * 4);
                    let mut buffers = [gated; NORM_BATCH + 2];
                    buffers[1] = binding(scale);
                    for (row, buffer) in buffers[2..].iter_mut().enumerate() {
                        *buffer = rows.get(row).unwrap_or(&rows[0]).2;
                    }
                    gpu.dispatch(NORM_KERNELS[2], &buffers, (), (rows.len(), 1), WIDTH / 4);
                }
            } else {
                for (row, (_, _, output)) in tile.enumerate() {
                    gpu.norm((&self.gated, row * WIDTH * 4), scale, output, true);
                }
            }
        }
    }
}
