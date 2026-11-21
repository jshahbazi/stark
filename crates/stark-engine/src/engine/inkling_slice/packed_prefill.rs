// SPDX-License-Identifier: Apache-2.0
//! Layer-major prefill with contiguous activations shared by adjacent stages.

use super::*;
use crate::engine::inkling_slice::grouped_pilot::GroupedExperts;

const QUERY: usize = HEADS * 128;
const RELATIVE: usize = HEADS * 16;
const ATTENTION_SUBTILE: usize = 16;

const M5_PACKED_ROWS: usize = 512;

pub(super) struct PackedPrefill {
    rows: usize,
    tokens: Buffer,
    partial: Buffer,
    prelude: Buffer,
    blocks: Buffer,
    previous: Buffer,
    mixed: Buffer,
    messages: Buffer,
    controller_input: Buffer,
    compressed: Buffer,
    gated: Buffer,
    norms: [Buffer; 3],
    q: Buffer,
    k: Buffer,
    v: Buffer,
    r: Buffer,
    gate: Buffer,
    sink: Buffer,
    dynamic: [Buffer; 2],
    conv_k: Buffer,
    conv_v: Buffer,
    keys: Buffer,
    values: Buffer,
    table: Buffer,
    parts: Buffer,
    attended: Buffer,
    attention_raw: Buffer,
    attention_output: Buffer,
    mlp: Buffer,
    experts: GroupedExperts,
}

impl PackedPrefill {
    pub(super) const MIN_ROWS: usize = 16;

    pub(super) fn for_m5_worker(runtime: &mut Runtime) -> Result<Self> {
        Self::new(runtime, M5_PACKED_ROWS, true)
    }

    pub(super) fn diagnostic(
        runtime: &mut Runtime,
        variant: &str,
    ) -> Result<Self> {
        let variant = variant.strip_prefix("packed").ok_or("expected packed prefill variant")?;
        let (rows, wide) = variant.strip_suffix("+mxwide").map_or((variant, false), |rows| (rows, true));
        let rows = rows.parse().map_err(|_| "invalid packed prefill rows")?;
        Self::new(runtime, rows, wide)
    }

    fn new(
        runtime: &mut Runtime,
        rows: usize,
        wide_mxfp8: bool,
    ) -> Result<Self> {
        if !runtime.pinned_m5 || !(128..=M5_PACKED_ROWS).contains(&rows) {
            return Err("packed Inkling prefill requires M5 Max and 128 to 512 rows");
        }
        register(
            runtime,
            concat!(include_str!("prefill.metal"), "\n", include_str!("packed_dense.metal")),
            &["ink_packed_matvec"],
        )?;
        register(
            runtime,
            include_str!("packed_residual.metal"),
            &[
                "ink_packed_embedding",
                "ink_packed_prepare_visit",
                "ink_packed_mix",
                "ink_packed_norm",
                "ink_packed_controller_norm",
                "ink_packed_controller_expand",
                "ink_packed_after_attention",
                "ink_packed_after_experts",
            ],
        )?;
        register(
            runtime,
            include_str!("packed_attention.metal"),
            &[
                "ink_packed_conv",
                "ink_packed_commit_conv",
                "ink_packed_heads",
                "ink_packed_commit_kv",
                "ink_packed_attention_table",
                "ink_packed_attention_parts",
                "ink_packed_attention_gate",
            ],
        )?;
        let mut experts = GroupedExperts::new(runtime, rows)?;
        experts.min_rows = Self::MIN_ROWS;
        if wide_mxfp8 {
            experts.enable_wide_mxfp8(runtime)?;
        } else {
            experts.enable_mxfp8(runtime)?;
        }
        let device = &runtime.device;
        let bf16 = |stride| zeroed(device, rows * stride * 2);
        let f32 = |stride| zeroed(device, rows * stride * 4);
        Ok(Self {
            rows,
            tokens: zeroed(device, rows * 4)?,
            partial: f32(WIDTH)?,
            prelude: f32(WIDTH)?,
            blocks: f32(4 * WIDTH)?,
            previous: f32(2 * WIDTH)?,
            mixed: f32(3 * WIDTH)?,
            messages: f32(4 * WIDTH)?,
            controller_input: bf16(2 * WIDTH)?,
            compressed: bf16(64)?,
            gated: f32(WIDTH)?,
            norms: [bf16(WIDTH)?, bf16(WIDTH)?, bf16(WIDTH)?],
            q: bf16(QUERY)?,
            k: bf16(KV)?,
            v: bf16(KV)?,
            r: bf16(RELATIVE)?,
            gate: bf16(QUERY)?,
            sink: bf16(HEADS)?,
            dynamic: [bf16(32)?, bf16(32)?],
            conv_k: bf16(KV)?,
            conv_v: bf16(KV)?,
            keys: bf16(KV)?,
            values: bf16(KV)?,
            table: bf16(HEADS * LOCAL)?,
            parts: zeroed(device, ATTENTION_SUBTILE * HEADS * 64 * 130 * 4)?,
            attended: bf16(QUERY)?,
            attention_raw: bf16(WIDTH)?,
            attention_output: bf16(WIDTH)?,
            mlp: bf16(WIDTH)?,
            experts,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn upload(
        &self,
        tokens: &[u32],
    ) {
        assert!(tokens.len() <= self.rows);
        // Session append is serialized and the prior command has completed.
        unsafe {
            self.tokens.contents().as_ptr().cast::<u32>().copy_from_nonoverlapping(tokens.as_ptr(), tokens.len());
        }
    }

    fn project(
        &self,
        gpu: &mut Stream<'_>,
        weight: &Buffer,
        input: &Buffer,
        output: &Buffer,
        width: usize,
        columns: usize,
        rows: usize,
    ) {
        gpu.one(
            "ink_packed_matvec",
            &[weight, input, output],
            [width as u32, columns as u32, rows as u32, 0],
            (columns.div_ceil(4), rows.div_ceil(8)),
            if columns <= 64 {
                128
            } else {
                64
            },
        );
    }

    // input and output are FP32 tensors, with independent token-row strides.
    #[allow(clippy::too_many_arguments)]
    fn mix(
        &self,
        gpu: &mut Stream<'_>,
        sources: Binding<'_>,
        partial: Binding<'_>,
        weights: &Buffer,
        output: Binding<'_>,
        rows: usize,
        queries: usize,
        count: usize,
        live_partial: bool,
        source_stride: usize,
        result_stride: usize,
    ) {
        gpu.dispatch(
            "ink_packed_mix",
            &[sources, partial, binding(weights), output],
            [count as u32, u32::from(live_partial), source_stride as u32, result_stride as u32],
            (queries, rows),
            WIDTH / 4,
        );
    }

    fn norm(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        input: Binding<'_>,
        stride: usize,
        output: Binding<'_>,
        slot: usize,
        rows: usize,
        later: bool,
    ) {
        let weight = if slot == 3 {
            &layer.mlp_norm
        } else {
            &layer.qkv_norm[slot]
        };
        let (input, stride) = if later {
            let message = (&self.messages, slot * WIDTH * 4);
            gpu.dispatch(
                "ink_packed_controller_norm",
                &[input, message, binding(&self.controller_input)],
                [stride as u32, (4 * WIDTH) as u32],
                (2, rows),
                WIDTH / 4,
            );
            let gate = &layer.gates[slot];
            self.project(gpu, &gate.down, &self.controller_input, &self.compressed, 2 * WIDTH, 64, rows);
            gpu.dispatch(
                "ink_packed_controller_expand",
                &[
                    binding(&self.compressed),
                    binding(&gate.gain),
                    binding(&gate.forget),
                    input,
                    message,
                    binding(&self.gated),
                ],
                [rows as u32, stride as u32, (4 * WIDTH) as u32, WIDTH as u32],
                (WIDTH / 32, rows.div_ceil(16)),
                32,
            );
            (binding(&self.gated), WIDTH)
        } else {
            (input, stride)
        };
        gpu.dispatch(
            "ink_packed_norm",
            &[input, binding(weight), output],
            [stride as u32, WIDTH as u32],
            (rows, 1),
            WIDTH / 4,
        );
    }

    fn convolution(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        weight: &Buffer,
        dynamic: Option<&Buffer>,
        history: &Buffer,
        output: &Buffer,
        width: usize,
        start: usize,
        rows: usize,
    ) {
        let args = [width as u32, start as u32, u32::from(dynamic.is_some()), rows as u32];
        gpu.one(
            "ink_packed_conv",
            &[input, weight, dynamic.unwrap_or(weight), history, output],
            args,
            (width / 128, rows),
            128,
        );
        gpu.one("ink_packed_commit_conv", &[input, history], args, (width / 128, rows.min(3)), 128);
    }

    fn attention(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        visit: &Visit,
        start: usize,
        rows: usize,
    ) {
        let window = usize::from(layer.local) * LOCAL;
        let args = [start as u32, rows as u32, visit.capacity as u32, window as u32];
        self.convolution(
            gpu,
            &self.k,
            &layer.key_conv,
            layer.dynamic.as_ref().map(|_| &self.dynamic[0]),
            &visit.conv[0],
            &self.conv_k,
            KV,
            start,
            rows,
        );
        self.convolution(
            gpu,
            &self.v,
            &layer.value_conv,
            layer.dynamic.as_ref().map(|_| &self.dynamic[1]),
            &visit.conv[1],
            &self.conv_v,
            KV,
            start,
            rows,
        );
        gpu.one(
            "ink_packed_heads",
            &[&self.q, &self.conv_k, &self.conv_v, &self.r, &layer.q_scale, &layer.r_scale, &self.keys, &self.values],
            [start as u32, window as u32],
            (16, rows),
            128,
        );
        gpu.one(
            "ink_packed_attention_table",
            &[&self.r, &layer.relative, &self.table],
            (),
            (LOCAL / 128, rows * HEADS),
            128,
        );
        for offset in (0..rows).step_by(ATTENTION_SUBTILE) {
            let live = (rows - offset).min(ATTENTION_SUBTILE);
            let params = [args[0], args[1], args[2], args[3], offset as u32];
            let shards = if (start + offset + live).min(visit.capacity) < LOCAL {
                8
            } else {
                16
            };
            gpu.one(
                "ink_packed_attention_parts",
                &[&self.q, &visit.keys, &visit.values, &self.table, &self.sink, &self.parts, &self.keys, &self.values],
                params,
                (shards, live * HEADS),
                128,
            );
            gpu.one(
                "ink_packed_attention_gate",
                &[&self.parts, &self.gate, &self.attended],
                params,
                (8, live * HEADS),
                128,
            );
        }
        gpu.one(
            "ink_packed_commit_kv",
            &[&self.keys, &self.values, &visit.keys, &visit.values],
            args,
            (KV / 128, rows.min(visit.capacity)),
            128,
        );
    }

    pub(super) fn encode(
        &self,
        session: &Session,
        gpu: &mut Stream<'_>,
        rows: usize,
        readout: bool,
    ) {
        let start = session.position;
        let last = rows - 1;
        let last_partial = (&self.partial, last * WIDTH * 4);
        gpu.one(
            "ink_packed_embedding",
            &[&session.embedding, &session.embed_norm, &self.partial, &self.tokens],
            (),
            (rows, 1),
            WIDTH / 4,
        );
        let mut blocks = 0;
        for (index, (&physical, visit)) in PHYSICAL.iter().zip(&session.visits).enumerate() {
            gpu.phase(PrefillPhase::AttentionInputs);
            let state = session.visit_input(index, physical, &mut blocks);
            let layer = &session.layers[physical];
            if matches!(index, 6 | 18 | 30) {
                gpu.one(
                    "ink_packed_prepare_visit",
                    &[&self.partial, &self.prelude, &self.blocks, &self.previous],
                    index as u32,
                    (WIDTH / 128, rows),
                    128,
                );
            }
            if session.capture_draft_features
                && let Some(tap) = DRAFT_TAPS.iter().position(|&i| i == index)
            {
                gpu.dispatch(
                    "ink_capture_feature",
                    &[last_partial, (&session.draft_features, tap * WIDTH * 2)],
                    (),
                    (WIDTH / 128, 1),
                    128,
                );
            }
            if state.blocks > 0 {
                let queries = layer.attn_queries.as_ref().unwrap_or(&layer.mlp_query);
                self.mix(
                    gpu,
                    binding(&self.blocks),
                    binding(&self.partial),
                    queries,
                    binding(&self.mixed),
                    rows,
                    if layer.local {
                        1
                    } else {
                        3
                    },
                    state.blocks,
                    true,
                    4 * WIDTH,
                    3 * WIDTH,
                );
            }
            if state.later {
                self.mix(
                    gpu,
                    binding(&self.previous),
                    binding(&self.partial),
                    layer.gate_queries.as_ref().unwrap(),
                    binding(&self.messages),
                    rows,
                    4,
                    2,
                    false,
                    2 * WIDTH,
                    4 * WIDTH,
                );
            }
            for slot in 0..3 {
                let (input, stride) = if state.blocks == 0 {
                    (binding(&self.partial), WIDTH)
                } else {
                    (
                        (
                            &self.mixed,
                            if layer.local {
                                0
                            } else {
                                slot * WIDTH * 4
                            },
                        ),
                        3 * WIDTH,
                    )
                };
                self.norm(gpu, layer, input, stride, binding(&self.norms[slot]), slot, rows, state.later);
            }
            gpu.phase(PrefillPhase::AttentionDense);
            for (slot, weight, output, columns) in [
                (0, &layer.q, &self.q, QUERY),
                (1, &layer.k, &self.k, KV),
                (2, &layer.v, &self.v, KV),
                (0, &layer.r, &self.r, RELATIVE),
                (0, &layer.attn_gate, &self.gate, QUERY),
                (0, &layer.sink, &self.sink, HEADS),
            ] {
                self.project(gpu, weight, &self.norms[slot], output, WIDTH, columns, rows);
            }
            if let Some(dynamic) = &layer.dynamic {
                self.project(gpu, &dynamic[0], &self.norms[1], &self.dynamic[0], WIDTH, 32, rows);
                self.project(gpu, &dynamic[1], &self.norms[2], &self.dynamic[1], WIDTH, 32, rows);
            }
            gpu.phase(PrefillPhase::RelativeCacheAttention);
            self.attention(gpu, layer, visit, start, rows);
            gpu.phase(PrefillPhase::AttentionDense);
            self.project(gpu, &layer.output, &self.attended, &self.attention_raw, QUERY, WIDTH, rows);
            gpu.phase(PrefillPhase::ExpertInputs);
            self.convolution(
                gpu,
                &self.attention_raw,
                &layer.attn_conv,
                None,
                &visit.conv[2],
                &self.attention_output,
                WIDTH,
                start,
                rows,
            );
            gpu.one(
                "ink_packed_after_attention",
                &[&self.partial, &self.attention_output, &self.blocks],
                [u32::from(state.boundary), state.blocks as u32],
                (WIDTH / 128, rows),
                128,
            );
            self.mix(
                gpu,
                binding(&self.blocks),
                binding(&self.partial),
                &layer.mlp_query,
                binding(&self.mixed),
                rows,
                1,
                state.blocks + usize::from(state.boundary),
                true,
                4 * WIDTH,
                3 * WIDTH,
            );
            self.norm(gpu, layer, binding(&self.mixed), 3 * WIDTH, binding(self.experts.input()), 3, rows, state.later);
            gpu.phase(match layer.experts {
                Experts::Nvfp4 {
                    ..
                } => PrefillPhase::Experts,
                Experts::Mxfp8 {
                    ..
                } => PrefillPhase::Mxfp8,
            });
            self.experts.encode(gpu, layer, rows, false);
            gpu.phase(PrefillPhase::ExpertFinish);
            self.convolution(
                gpu,
                self.experts.output(),
                &layer.mlp_conv,
                None,
                &visit.conv[3],
                &self.mlp,
                WIDTH,
                start,
                rows,
            );
            gpu.one("ink_packed_after_experts", &[&self.partial, &self.mlp], (), (WIDTH / 128, rows), 128);
        }
        if readout || session.capture_draft_features {
            gpu.phase(PrefillPhase::Other);
            self.mix(
                gpu,
                (&self.blocks, last * 4 * WIDTH * 4),
                last_partial,
                &session.output_query,
                binding(&session.mixed),
                1,
                1,
                blocks,
                true,
                4 * WIDTH,
                WIDTH,
            );
            gpu.norm(binding(&session.mixed), &session.output_norm, binding(&session.hidden), true);
            if session.capture_draft_features {
                gpu.dispatch(
                    "ink_trace_copy",
                    &[binding(&session.hidden), (&session.draft_features, 3 * WIDTH * 2)],
                    (WIDTH / 2) as u32,
                    (WIDTH.div_ceil(256), 1),
                    128,
                );
            }
            if readout {
                session.encode_output(gpu);
            }
        }
    }
}

fn register(
    runtime: &mut Runtime,
    source: &str,
    kernels: &[Kernel],
) -> Result<()> {
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let library = runtime
        .device
        .new_library_with_source(source, Some(&options))
        .map_err(|_| "Inkling packed prefill library compilation failed")?;
    for &name in kernels {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
    }
    Ok(())
}
