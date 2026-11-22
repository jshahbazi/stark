// SPDX-License-Identifier: Apache-2.0
//! Layer-major prompt scheduling with shared BF16 attention projections.

use super::{prefill_dense::PrefillDense, prefill_relative::PrefillRelative, prefill_trace::PrefillTrace, *};
use crate::engine::inkling_slice::grouped_pilot::GroupedExperts;

#[path = "prefill_controller.rs"]
mod prefill_controller;
use prefill_controller::PrefillControllers;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum VerifyExperts {
    Singleton,
    Grouped,
}

pub(super) struct PrefillLanes {
    flows: Vec<Flow>,
    features: Vec<Buffer>,
    scratch: Vec<Scratch>,
    grouped: GroupedExperts,
    controllers: Option<PrefillControllers>,
    pub(super) dense: Option<PrefillDense>,
    pub(super) relative: Option<PrefillRelative>,
    pub(super) source_router: bool,
    wide_gemv: bool,
    batched_norm: bool,
    pub(super) direct_experts: bool,
}

impl PrefillLanes {
    pub(super) fn new(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        if !runtime.pinned_m5 {
            return Err("Inkling shared projection pilot requires the pinned M5 Max");
        }
        let grouped = GroupedExperts::new(
            runtime,
            if rows <= 128 {
                rows.max(BATCHED_PREFILL_ROWS)
            } else {
                BATCHED_PREFILL_ROWS
            },
        )?;
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
            controllers: None,
            dense: None,
            relative: None,
            source_router: false,
            wide_gemv: false,
            batched_norm: false,
            direct_experts: false,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.flows.len() + 1
    }

    pub(super) fn for_m5_worker(runtime: &mut Runtime) -> Result<Self> {
        Self::for_m5_rows(runtime, M5_PREFILL_ROWS)
    }

    pub(super) fn for_m5_rows(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        let mut lanes = Self::new(runtime, rows)?;
        lanes.enable_controllers(runtime)?;
        lanes.enable_batched_norm(runtime)?;
        lanes.direct_experts = true;
        lanes.enable_mxfp8(runtime)?;
        lanes.enable_wide_gemv(runtime)?;
        lanes.enable_short_expert_tiles();
        Ok(lanes)
    }

    pub(super) fn verify_taps<'a>(
        &'a self,
        session: &'a Session,
        rows: usize,
    ) -> impl Iterator<Item = &'a Buffer> + Clone {
        self.features.iter().chain(std::iter::once(&session.draft_features)).skip(self.rows() - rows)
    }

    pub(super) fn enable_verify_groups(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        self.grouped.enable_source_epilogue(runtime)?;
        if std::env::var_os("STARK_INKLING_EXPERIMENT_VERIFY_COMPACT_RUNS").as_deref()
            == Some(std::ffi::OsStr::new("1"))
        {
            packed_prefill::register(runtime, include_str!("verify_bucket.metal"), &["ink_verify_bucket_short"])?;
            self.grouped.enable_verify_compaction(runtime)?;
        }
        self.grouped.enable_source_softmax(runtime)?;
        self.grouped.enable_source_mxfp8(runtime)?;
        let fused_experts = std::env::var_os("STARK_INKLING_EXPERIMENT_VERIFY_FUSED_FINISH").as_deref()
            == Some(std::ffi::OsStr::new("1"));
        let fused_attention = std::env::var_os("STARK_INKLING_EXPERIMENT_VERIFY_FUSED_ATTENTION").as_deref()
            == Some(std::ffi::OsStr::new("1"));
        if fused_experts || fused_attention {
            packed_prefill::register(
                runtime,
                include_str!("verify_finish.metal"),
                &["ink_verify_finish_experts", "ink_verify_finish_attention"],
            )?;
            runtime.verify_fused_finish = fused_experts;
            runtime.verify_fused_attention = fused_attention;
        }
        if std::env::var_os("STARK_INKLING_EXPERIMENT_VERIFY_SHORT_ROUTER").as_deref()
            == Some(std::ffi::OsStr::new("1"))
        {
            packed_prefill::register(
                runtime,
                concat!("#define INK_SOURCE_VERIFY_ROUTER\n", include_str!("prefill_dense_source.metal")),
                &["ink_verify_router_source"],
            )?;
            runtime.verify_short_router = true;
            if std::env::var_os("STARK_INKLING_EXPERIMENT_VERIFY_NATIVE_ROUTER").as_deref()
                == Some(std::ffi::OsStr::new("1"))
                && runtime.verify_native_router.is_none()
            {
                packed_prefill::register(
                    runtime,
                    include_str!("verify_router_native.metal"),
                    &["ink_verify_router_native", "ink_verify_router_join"],
                )?;
                runtime.verify_native_router = Some(
                    runtime
                        .device
                        .new_buffer(
                            2 * MAX_VERIFY_ROWS * EXPERTS * size_of::<f32>(),
                            MTLResourceOptions::STORAGE_MODE_PRIVATE,
                        )
                        .ok_or("Inkling verifier router allocation failed")?,
                );
            }
        }
        self.grouped.min_rows = 2;
        Ok(())
    }

    pub(super) fn enable_verify_qkv_batch(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        self.controllers.as_mut().ok_or("Inkling verifier controllers unavailable")?.enable_verify_qkv_batch(runtime)
    }

    pub(super) fn enable_controllers(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        self.controllers = Some(PrefillControllers::new(runtime, self.rows().min(BATCHED_PREFILL_ROWS))?);
        Ok(())
    }

    pub(super) fn enable_batched_norm(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        PrefillControllers::compile_batched_norm(runtime)?;
        self.batched_norm = true;
        Ok(())
    }

    pub(super) fn enable_mxfp8(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        self.grouped.enable_mxfp8(runtime)
    }

    pub(super) fn enable_short_expert_tiles(&mut self) {
        self.grouped.min_rows = 16;
    }

    pub(super) fn enable_wide_gemv(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = runtime
            .device
            .new_library_with_source(include_str!("prefill.metal"), Some(&options))
            .map_err(|_| "Inkling wide projection shader compilation failed")?;
        let name = "ink_matvec_prefill_wide";
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        runtime.pipelines.insert(name, pipeline);
        self.wide_gemv = true;
        Ok(())
    }
}

pub(super) fn project_rows<'a>(
    gpu: &mut Stream<'_>,
    dense: Option<&PrefillDense>,
    weight: &Buffer,
    inputs_and_outputs: impl IntoIterator<Item = (Binding<'a>, &'a Buffer)>,
    columns: usize,
    rows: usize,
    wide_gemv: bool,
) {
    let lanes = inputs_and_outputs.into_iter().collect::<Vec<_>>();
    if let Some(dense) = dense.filter(|_| lanes.len() > 64 && rows > 64) {
        dense.project(gpu, weight, &lanes, columns, rows);
        return;
    }
    for tile in lanes.chunks(if wide_gemv {
        8
    } else {
        4
    }) {
        if let [(input, output)] = tile {
            gpu.project(*input, weight, output, columns, rows);
            continue;
        }
        let first = tile[0];
        let live = gpu.runtime.verify_nvfp4_matvec && gpu.runtime.verify_combined_short;
        let (kernel, width) = match tile.len() {
            2 if live => ("ink_matvec_verify2", 2),
            3 if live => ("ink_matvec_verify3", 3),
            2..=4 => ("ink_matvec_prefill", 4),
            _ => ("ink_matvec_prefill_wide", 8),
        };
        let mut buffers = [binding(weight); 17];
        for row in 0..width {
            let (input, output) = tile.get(row).copied().unwrap_or(first);
            buffers[1 + row] = input;
            buffers[1 + width + row] = binding(output);
        }
        gpu.dispatch(
            kernel,
            &buffers[..1 + 2 * width],
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
        self.encode_prefill_trace(gpu, lanes, inputs, readout, None);
    }

    pub(super) fn encode_prefill_trace(
        &self,
        gpu: &mut Stream<'_>,
        lanes: &PrefillLanes,
        inputs: &[TokenInput<'_>],
        readout: bool,
        trace: Option<&PrefillTrace>,
    ) {
        self.encode_lanes(gpu, lanes, inputs, readout, trace, None, None, None);
    }

    pub(super) fn encode_verify_rows(
        &self,
        gpu: &mut Stream<'_>,
        lanes: &PrefillLanes,
        tokens: &[u32],
        experts: VerifyExperts,
    ) {
        let inputs = tokens.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
        self.encode_lanes(gpu, lanes, &inputs, true, None, Some(experts), None, None);
    }

    pub(super) fn encode_verify_transaction(
        &self,
        gpu: &mut Stream<'_>,
        lanes: &PrefillLanes,
        tokens: &[u32],
        experts: VerifyExperts,
        undo: &verify_state::VerifyWorkspace,
        proposals: Option<&Buffer>,
    ) {
        let inputs = tokens.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
        self.encode_lanes(gpu, lanes, &inputs, true, None, Some(experts), Some(undo), proposals);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_lanes(
        &self,
        gpu: &mut Stream<'_>,
        lanes: &PrefillLanes,
        inputs: &[TokenInput<'_>],
        readout: bool,
        trace: Option<&PrefillTrace>,
        verify: Option<VerifyExperts>,
        undo: Option<&verify_state::VerifyWorkspace>,
        proposals: Option<&Buffer>,
    ) {
        let allow_groups = verify != Some(VerifyExperts::Singleton);
        let source_groups = verify == Some(VerifyExperts::Grouped);
        let row_count = inputs.len();
        let skip = lanes.rows() - row_count;
        let flows = lanes.flows.iter().chain(std::iter::once(&self.flow)).skip(skip);
        let taps = lanes.verify_taps(self, row_count);
        let draft = self.active_dspark();
        for (row, (flow, &input)) in flows.clone().zip(inputs).enumerate() {
            if row > 0
                && let Some(proposals) = proposals
            {
                gpu.one(
                    "ink_embedding_draft_ids",
                    &[&self.embedding, &self.embed_norm, &flow.partial, proposals],
                    [(row - 1) as u32, VOCABULARY as u32],
                    (1, 1),
                    WIDTH / 4,
                );
            } else {
                self.encode_embedding(gpu, flow, input);
            }
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
            let tap = trace.filter(|trace| trace.visit == index);
            let state = self.visit_input(index, physical, &mut blocks);
            let layer = &self.layers[physical];
            let scratches = lanes.scratch.iter().chain(std::iter::once(&visit.scratch)).skip(skip);
            let controllers = lanes.controllers.as_ref().filter(|_| state.later && row_count > 1);
            gpu.phase(PrefillPhase::AttentionInputs);
            for (row, ((flow, taps), scratch)) in flows.clone().zip(tap_rows.clone()).zip(scratches.clone()).enumerate()
            {
                Self::prepare_visit(gpu, flow, index, taps);
                if let Some(tap) = tap {
                    tap.input(gpu, row, flow);
                }
                if controllers.is_some() {
                    layer.prepare_attention_inputs(gpu, flow, &flow.partial, scratch, state);
                } else {
                    layer.attention_inputs(gpu, flow, &flow.partial, scratch, state);
                }
            }
            if let Some(controllers) = controllers
                && !(undo.is_some()
                    && controllers.try_encode_verify_qkv(gpu, layer, state, flows.clone().zip(scratches.clone())))
            {
                for slot in 0..3 {
                    let rows = flows.clone().zip(scratches.clone()).map(|(flow, scratch)| {
                        (
                            scratch,
                            layer.attention_norm_input(&flow.partial, scratch, state, slot),
                            binding(&scratch.qkv_input[slot]),
                        )
                    });
                    controllers.encode(gpu, layer, slot, rows, row_count, lanes.wide_gemv, lanes.batched_norm);
                }
            }

            gpu.phase(PrefillPhase::AttentionDense);
            for (index, (_, weight, _, columns)) in layer.attention_projections(&visit.scratch).into_iter().enumerate()
            {
                let bindings = scratches.clone().map(|s| {
                    let (input, _, output, _) = layer.attention_projections(s)[index];
                    (binding(input), output)
                });
                project_rows(gpu, lanes.dense.as_ref(), weight, bindings, WIDTH, columns, lanes.wide_gemv);
            }
            if let Some(weights) = &layer.dynamic {
                for (index, weight) in weights.iter().enumerate() {
                    let bindings = scratches.clone().map(|s| (binding(&s.qkv_input[index + 1]), &s.dynamic[index]));
                    project_rows(gpu, None, weight, bindings, WIDTH, 32, lanes.wide_gemv);
                }
            }
            gpu.phase(PrefillPhase::RelativeCacheAttention);
            if let Some(shared) = undo.filter(|workspace| workspace.shares_local_attention())
                && layer.local
                && visit.capacity == LOCAL
                && self.position >= LOCAL
                && trace.is_none()
            {
                let rows = scratches.clone().collect::<Vec<_>>();
                shared.encode_shared_local(gpu, index, layer, visit, self.position, &rows);
            } else if let Some(shared) = undo.filter(|workspace| workspace.shares_global_attention())
                && !layer.local
                && self.position >= LOCAL
                && self.position + row_count <= visit.capacity
                && tap.is_none()
                && trace.is_none()
            {
                let rows = scratches.clone().collect::<Vec<_>>();
                shared.encode_shared_global(gpu, layer, visit, self.position, &rows);
            } else if let Some(relative) = lanes.relative.as_ref().filter(|_| layer.local && visit.capacity == LOCAL) {
                relative.gather(gpu, visit, self.position, row_count, true);
                for (row, scratch) in scratches.clone().enumerate() {
                    if let Some(tap) = tap {
                        tap.projected(gpu, row, scratch);
                    }
                    let position = self.position + row;
                    layer.attention_write_cache(gpu, scratch, visit, [position as u32, LOCAL as u32, LOCAL as u32, 0]);
                    relative.pack(gpu, scratch, row);
                }
                relative.gather(gpu, visit, self.position, row_count, false);
                relative.encode(gpu, &layer.relative, self.position, row_count);
                for (row, scratch) in scratches.clone().enumerate() {
                    relative.unpack(gpu, scratch, row, tap.is_some());
                    if let Some(tap) = tap {
                        tap.attended(gpu, row, scratch, visit, self.position + row);
                    }
                }
            } else {
                for (row, scratch) in scratches.clone().enumerate() {
                    if let Some(tap) = tap {
                        tap.projected(gpu, row, scratch);
                    }
                    layer.attention_read_cache(gpu, scratch, visit, self.position + row);
                    if let Some(tap) = tap {
                        tap.attended(gpu, row, scratch, visit, self.position + row);
                    }
                }
            }
            gpu.phase(PrefillPhase::AttentionDense);
            let output = scratches.clone().map(|s| (binding(&s.attended), &s.attn_raw));
            project_rows(gpu, lanes.dense.as_ref(), &layer.output, output, 3072, WIDTH, lanes.wide_gemv);

            gpu.phase(PrefillPhase::ExpertInputs);
            let direct = allow_groups
                && lanes.direct_experts
                && trace.is_none()
                && lanes.grouped.supports(layer, row_count)
                && (!state.later || controllers.is_some());
            let fused_attention = gpu.runtime.verify_fused_attention
                && undo.is_some()
                && tap.is_none()
                && trace.is_none()
                && !state.capture
                && row_count <= MAX_VERIFY_ROWS;
            if fused_attention {
                let scratch: [&Scratch; MAX_VERIFY_ROWS] =
                    std::array::from_fn(|i| scratches.clone().nth(i.min(row_count - 1)).expect("verified scratch"));
                let flow: [&Flow; MAX_VERIFY_ROWS] =
                    std::array::from_fn(|i| flows.clone().nth(i.min(row_count - 1)).expect("verified flow"));
                gpu.dispatch(
                    "ink_verify_finish_attention",
                    &[
                        binding(&layer.attn_conv),
                        binding(&visit.conv[2]),
                        binding(&scratch[0].attn_raw),
                        binding(&scratch[1].attn_raw),
                        binding(&scratch[2].attn_raw),
                        binding(&scratch[3].attn_raw),
                        binding(&flow[0].partial),
                        binding(&flow[1].partial),
                        binding(&flow[2].partial),
                        binding(&flow[3].partial),
                        (&flow[0].blocks, state.blocks.min(3) * WIDTH * 4),
                        (&flow[1].blocks, state.blocks.min(3) * WIDTH * 4),
                        (&flow[2].blocks, state.blocks.min(3) * WIDTH * 4),
                        (&flow[3].blocks, state.blocks.min(3) * WIDTH * 4),
                    ],
                    [self.position as u32, row_count as u32, u32::from(state.boundary), 0],
                    (WIDTH / 128, 1),
                    128,
                );
            }
            for (row, (flow, scratch)) in flows.clone().zip(scratches.clone()).enumerate() {
                if fused_attention {
                    layer.mix_expert_inputs(gpu, flow, &flow.partial, scratch, state);
                } else {
                    Layer::convolution(
                        gpu,
                        binding(&scratch.attn_raw),
                        &layer.attn_conv,
                        None,
                        &visit.conv[2],
                        &scratch.attention,
                        WIDTH,
                        self.position + row,
                    );
                    layer.prepare_expert_inputs(gpu, flow, &flow.partial, scratch, state);
                }
                if controllers.is_none() {
                    if direct {
                        gpu.norm(binding(&scratch.mixed), &layer.mlp_norm, lanes.grouped.input_at(row), true);
                    } else {
                        layer.norms(gpu, scratch, binding(&scratch.mixed), &scratch.mlp_input, 3, state.later);
                    }
                }
            }
            if let Some(controllers) = controllers {
                let rows = scratches.clone().enumerate().map(|(row, scratch)| {
                    let output = if direct {
                        lanes.grouped.input_at(row)
                    } else {
                        binding(&scratch.mlp_input)
                    };
                    (scratch, binding(&scratch.mixed), output)
                });
                controllers.encode(gpu, layer, 3, rows, row_count, lanes.wide_gemv, lanes.batched_norm);
            }

            gpu.phase(match layer.experts {
                Experts::Nvfp4 {
                    ..
                } => PrefillPhase::Experts,
                Experts::Mxfp8 {
                    ..
                } => PrefillPhase::Mxfp8,
            });
            let fused_finish = gpu.runtime.verify_fused_finish
                && undo.is_some()
                && direct
                && tap.is_none()
                && !state.capture
                && row_count <= MAX_VERIFY_ROWS;
            gpu.defer_expert_merge = fused_finish;
            let expert_rows = lanes.grouped.rows();
            for start in (0..row_count).step_by(expert_rows) {
                let count = (row_count - start).min(expert_rows);
                let tile = scratches.clone().skip(start).take(count);
                if allow_groups && lanes.grouped.supports(layer, count) {
                    if direct {
                        lanes.grouped.encode(gpu, layer, count, source_groups || lanes.source_router, source_groups);
                        continue;
                    }
                    for (row, scratch) in tile.clone().enumerate() {
                        lanes.grouped.copy_input(gpu, binding(&scratch.mlp_input), row);
                    }
                    lanes.grouped.encode(gpu, layer, count, source_groups || lanes.source_router, source_groups);
                    for (row, scratch) in tile.enumerate() {
                        lanes.grouped.copy_output(gpu, row, binding(&scratch.experts));
                        if let Some(tap) = tap {
                            tap.routing(gpu, start + row, lanes.grouped.routing(row));
                        }
                    }
                } else {
                    for (row, scratch) in tile.enumerate() {
                        layer.experts(gpu, scratch);
                        if let Some(tap) = tap {
                            tap.routing(
                                gpu,
                                start + row,
                                [&scratch.logits, &scratch.routes, &scratch.route_weights].map(binding),
                            );
                        }
                    }
                }
            }
            gpu.defer_expert_merge = false;
            gpu.phase(PrefillPhase::ExpertFinish);
            if fused_finish {
                let partials =
                    std::array::from_fn(|i| &flows.clone().nth(i.min(row_count - 1)).expect("verified row").partial);
                lanes.grouped.finish_verify_rows(gpu, layer, visit, self.position, row_count, partials);
                if row_count == MAX_VERIFY_ROWS
                    && let Some(undo) = undo
                {
                    let first = scratches.clone().next().expect("verified anchor scratch");
                    undo.save_anchor(gpu, index, first, lanes.grouped.output_at(0));
                }
                continue;
            }
            for (row, (flow, scratch)) in flows.clone().zip(scratches).enumerate() {
                let output = if direct {
                    lanes.grouped.output_at(row)
                } else {
                    binding(&scratch.experts)
                };
                layer.finish_experts(gpu, visit, &flow.partial, scratch, output, state, self.position + row);
                if row == 0
                    && row_count == MAX_VERIFY_ROWS
                    && let Some(undo) = undo
                {
                    undo.save_anchor(gpu, index, scratch, output);
                }
                if let Some(tap) = tap {
                    tap.output(gpu, row, scratch);
                }
                if let Some(trace) = trace {
                    trace.partial(gpu, index, row, flow);
                }
            }
        }

        gpu.phase(PrefillPhase::Other);
        for (row, (flow, taps)) in flows.zip(tap_rows).enumerate() {
            if verify.is_some() {
                self.finish_hidden(gpu, flow, blocks, (&self.hidden, row * WIDTH * 2), taps);
            } else {
                self.finish_forward(gpu, flow, blocks, readout && row + 1 == row_count, taps);
            }
            if let (None, Some(draft), Some(taps)) = (undo, draft, taps) {
                draft.commit(gpu, taps, self.position + row);
            }
        }
        if verify.is_some() {
            self.encode_output(gpu);
            for row in 1..row_count {
                self.cap_logits(gpu, row);
            }
        }
    }
}
