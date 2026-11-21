// SPDX-License-Identifier: Apache-2.0
//! Layer-major prompt scheduling with shared BF16 attention projections.

use super::*;
use crate::engine::inkling_slice::grouped_pilot::GroupedExperts;

pub(super) struct PrefillLanes {
    flows: Vec<Flow>,
    features: Vec<Buffer>,
    scratch: Vec<Scratch>,
    grouped: GroupedExperts,
}

impl PrefillLanes {
    pub(super) fn new(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        if !runtime.pinned_m5 {
            return Err("Inkling shared projection pilot requires the pinned M5 Max");
        }
        let grouped = GroupedExperts::new(runtime, rows)?;
        let device = &runtime.device;
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = device
            .new_library_with_source(include_str!("prefill.metal"), Some(&options))
            .map_err(|_| "Inkling shared projection shader compilation failed")?;
        let function =
            library.new_function_with_name("ink_matvec_prefill").ok_or("missing Inkling shared projection")?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| "Inkling shared projection pipeline compilation failed")?;
        runtime.pipelines.insert("ink_matvec_prefill", pipeline);
        Ok(Self {
            flows: (1..rows).map(|_| Flow::new(device)).collect::<Result<_>>()?,
            features: (1..rows).map(|_| zeroed(device, 4 * WIDTH * 2)).collect::<Result<_>>()?,
            // Each query finishes using the shared visit attention workspace
            // before the next row uses it. Other activations are per row.
            scratch: (1..rows).map(|_| Scratch::new(device, 0)).collect::<Result<_>>()?,
            grouped,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.flows.len() + 1
    }
}

fn project_rows<'a>(
    gpu: &mut Stream<'_>,
    weight: &Buffer,
    inputs_and_outputs: impl IntoIterator<Item = (&'a Buffer, &'a Buffer)>,
    columns: usize,
    rows: usize,
) {
    let lanes = inputs_and_outputs.into_iter().collect::<Vec<_>>();
    for tile in lanes.chunks(4) {
        if let [(input, output)] = tile {
            gpu.project(binding(input), weight, output, columns, rows);
            continue;
        }
        let first = tile[0];
        let mut buffers = [binding(weight); 9];
        for row in 0..4 {
            let (input, output) = tile.get(row).copied().unwrap_or(first);
            buffers[1 + row] = binding(input);
            buffers[5 + row] = binding(output);
        }
        gpu.dispatch(
            "ink_matvec_prefill",
            &buffers,
            [columns as u32, rows as u32, tile.len() as u32],
            (rows.div_ceil(4), 1),
            if rows <= 64 {
                128
            } else {
                64
            },
        );
    }
}

impl Session {
    pub(super) fn encode_prefill_rows(
        &self,
        gpu: &mut Stream<'_>,
        lanes: &PrefillLanes,
        inputs: &[TokenInput<'_>],
        readout: bool,
    ) {
        let row_count = inputs.len();
        let skip = lanes.rows() - row_count;
        let flows = lanes.flows.iter().chain(std::iter::once(&self.flow)).skip(skip);
        let taps = lanes.features.iter().chain(std::iter::once(&self.draft_features)).skip(skip);
        let draft = self.active_dspark();
        for (flow, &input) in flows.clone().zip(inputs) {
            self.encode_embedding(gpu, flow, input);
        }

        let tap_rows = taps.map(Some).enumerate().map(|(row, taps)| {
            if draft.is_some() || (self.capture_draft_features && row + 1 == row_count) {
                taps
            } else {
                None
            }
        });
        let mut blocks = 0;
        for (index, (&physical, visit)) in PHYSICAL.iter().zip(&self.visits).enumerate() {
            let state = self.visit_input(index, physical, &mut blocks);
            let layer = &self.layers[physical];
            let scratches = lanes.scratch.iter().chain(std::iter::once(&visit.scratch)).skip(skip);
            for ((flow, taps), scratch) in flows.clone().zip(tap_rows.clone()).zip(scratches.clone()) {
                Self::prepare_visit(gpu, flow, index, taps);
                layer.attention_inputs(gpu, flow, &flow.partial, scratch, state);
            }

            for (index, (_, weight, _, columns)) in layer.attention_projections(&visit.scratch).into_iter().enumerate()
            {
                let bindings = scratches.clone().map(|s| {
                    let (input, _, output, _) = layer.attention_projections(s)[index];
                    (input, output)
                });
                project_rows(gpu, weight, bindings, WIDTH, columns);
            }
            if let Some(weights) = &layer.dynamic {
                for (index, weight) in weights.iter().enumerate() {
                    let bindings = scratches.clone().map(|s| (&s.qkv_input[index + 1], &s.dynamic[index]));
                    project_rows(gpu, weight, bindings, WIDTH, 32);
                }
            }
            for (row, scratch) in scratches.clone().enumerate() {
                layer.attention_read_cache(gpu, scratch, visit, self.position + row);
            }
            let output = scratches.clone().map(|s| (&s.attended, &s.attn_raw));
            project_rows(gpu, &layer.output, output, 3072, WIDTH);

            for (row, (flow, scratch)) in flows.clone().zip(scratches.clone()).enumerate() {
                let position = self.position + row;
                Layer::convolution(
                    gpu,
                    &scratch.attn_raw,
                    &layer.attn_conv,
                    None,
                    &visit.conv[2],
                    &scratch.attention,
                    WIDTH,
                    position,
                );
                layer.expert_inputs(gpu, flow, &flow.partial, scratch, state);
            }

            if lanes.grouped.supports(layer, row_count) {
                for (row, scratch) in scratches.clone().enumerate() {
                    lanes.grouped.copy_input(gpu, binding(&scratch.mlp_input), row);
                }
                lanes.grouped.encode(gpu, layer, row_count);
                for (row, scratch) in scratches.clone().enumerate() {
                    lanes.grouped.copy_output(gpu, row, binding(&scratch.experts));
                }
            } else {
                for scratch in scratches.clone() {
                    layer.experts(gpu, scratch);
                }
            }
            for (row, (flow, scratch)) in flows.clone().zip(scratches).enumerate() {
                layer.finish_experts(gpu, visit, &flow.partial, scratch, state, self.position + row);
            }
        }

        for (row, (flow, taps)) in flows.zip(tap_rows).enumerate() {
            self.finish_forward(gpu, flow, blocks, readout && row + 1 == row_count, taps);
            if let (Some(draft), Some(taps)) = (draft, taps) {
                draft.commit(gpu, taps, self.position + row);
            }
        }
    }
}
