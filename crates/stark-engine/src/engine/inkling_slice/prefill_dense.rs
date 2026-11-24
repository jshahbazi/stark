// SPDX-License-Identifier: Apache-2.0
//! BF16 prefill products with source-shaped Metal tensor accumulators.

use super::*;

pub(super) struct PrefillDense {
    input: Buffer,
    output: Buffer,
}

impl PrefillDense {
    pub(super) fn new(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        let device = &runtime.device;
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = device
            .new_library_with_source(include_str!("prefill_dense_source.metal"), Some(&options))
            .map_err(|_| "Inkling dense matrix shader compilation failed")?;
        for name in ["ink_prefill_dense_source", "ink_prefill_router_source", "ink_prefill_select_source"] {
            let function = library.new_function_with_name(name).ok_or("missing Inkling dense matrix kernel")?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| "Inkling dense matrix pipeline compilation failed")?;
            runtime.pipelines.insert(name, pipeline);
        }
        Ok(Self {
            input: zeroed(device, rows * 3072 * 2)?,
            output: zeroed(device, rows * 3072 * 2)?,
        })
    }

    pub(super) fn project(
        &self,
        gpu: &mut Stream<'_>,
        weight: &Buffer,
        rows: &[(Binding<'_>, &Buffer)],
        width: usize,
        columns: usize,
    ) {
        let copy = |gpu: &mut Stream<'_>, source, output, count: usize| {
            gpu.dispatch("ink_trace_copy", &[source, output], (count / 2) as u32, (count.div_ceil(256), 1), 128);
        };
        for (row, (input, _)) in rows.iter().enumerate() {
            copy(gpu, *input, (&self.input, row * width * 2), width);
        }
        let max = columns.max(rows.len());
        let split = if width >= 3 * max || (max <= 1024 && width > 2 * max) {
            2048
        } else {
            width
        };
        gpu.one(
            "ink_prefill_dense_source",
            &[&self.input, weight, &self.output],
            [width as u32, columns as u32, rows.len() as u32, split as u32],
            (columns / 32, rows.len().div_ceil(16)),
            32,
        );
        for (row, (_, output)) in rows.iter().enumerate() {
            copy(gpu, (&self.output, row * columns * 2), binding(output), columns);
        }
    }
}
