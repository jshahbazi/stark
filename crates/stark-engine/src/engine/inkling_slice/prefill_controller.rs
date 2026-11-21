// SPDX-License-Identifier: Apache-2.0
//! Pack independent rows for the source-order controller expansion.

use super::*;

pub(super) struct PrefillControllers {
    input_message: Buffer,
    compressed: Buffer,
    gated: Buffer,
    rows: usize,
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
        })
    }

    pub(super) fn encode<'a>(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        slot: usize,
        rows: impl Iterator<Item = (&'a Scratch, Binding<'a>, &'a Buffer)> + Clone,
        count: usize,
        wide_gemv: bool,
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
            for (scratch, input, _) in tile.clone() {
                let message = (&scratch.messages, slot * WIDTH * 4);
                gpu.norm(input, scale, binding(&scratch.controller_pair), false);
                gpu.norm(message, scale, (&scratch.controller_pair, WIDTH * 2), false);
            }
            project_rows(
                gpu,
                None,
                &gate.down,
                tile.clone().map(|(s, _, _)| (binding(&s.controller_pair), &s.compressed)),
                2 * WIDTH,
                64,
                wide_gemv,
            );
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
            gpu.one(
                "ink_prefill_controller_expand",
                &[&self.compressed, &gate.gain, &gate.forget, &self.input_message, &self.gated],
                size as u32,
                (WIDTH / 32, size.div_ceil(16)),
                32,
            );
            for (row, (_, _, output)) in tile.enumerate() {
                gpu.norm((&self.gated, row * WIDTH * 4), scale, binding(output), true);
            }
        }
    }
}
