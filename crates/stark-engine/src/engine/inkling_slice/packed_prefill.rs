// SPDX-License-Identifier: Apache-2.0
//! Layer-major prefill with contiguous activations shared by adjacent stages.

use super::{prefill_trace::PrefillTrace, *};
use crate::engine::inkling_slice::grouped_pilot::GroupedExperts;

#[path = "packed_global_probe.rs"]
mod packed_global_probe;
use packed_global_probe::GlobalProbe;

const QUERY: usize = HEADS * 128;
const RELATIVE: usize = HEADS * 16;
const ATTENTION_SUBTILE: usize = 16;

const M5_PACKED_ROWS: usize = 512;

struct SourcePrefill {
    ordered_kv: [Buffer; 2],
    routed: bool,
    global: Option<GlobalProbe>,
}

pub(super) struct PackedPrefill {
    rows: usize,
    source: Option<SourcePrefill>,
    source_allowed: bool,
    /// Context-sized global workspace, present with the source global probe.
    context_workspace: bool,
    draft_taps: Option<Buffer>,
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

/// Default dense projection kernel and its tile (rows, columns), from
/// dense_fast.metal. The staged variants (ink_dense_staged*) are under test.
/// Default local prefill attention: BF16 query operands.
const LOCAL_ATTENTION: Kernel = "ink_prefill_relative_fast";
const DENSE_TILE: (Kernel, usize, usize) = ("ink_dense_fast", 128, 64);

impl PackedPrefill {
    pub(super) const MIN_ROWS: usize = 16;
    /// Smallest command the default source path packs. Lowering it below
    /// MIN_ROWS would run the worker's three-row tail on the packed source
    /// kernels instead of serial decode steps (not yet validated; parked at
    /// MIN_ROWS). The exact reference and other scopes keep MIN_ROWS.
    pub(super) const SOURCE_TAIL_ROWS: usize = Self::MIN_ROWS;

    pub(super) fn min_command_rows(
        &self,
        runtime: &Runtime,
    ) -> usize {
        if runtime.source_text_active && !runtime.exact_prefill && self.source.is_some() && self.source_allowed {
            Self::SOURCE_TAIL_ROWS
        } else {
            Self::MIN_ROWS
        }
    }
    /// Smallest packed command that uses the source kernels. Shorter packed
    /// chunks keep the incumbent exact kernels.
    pub(super) const SOURCE_MIN_ROWS: usize = 65;
    /// Largest source command width the packed workspace accepts.
    pub(super) const SOURCE_COMMAND_ROWS: usize = 8192;

    /// Largest request context the source path admits.
    pub(super) fn source_context_limit(context_workspace: bool) -> usize {
        if context_workspace {
            MAX_M5_CONTEXT
        } else {
            packed_global_probe::MAX_KEYS
        }
    }

    /// Global key capacity allocated for a context under the context-sized workspace.
    pub(super) fn source_global_capacity(context: usize) -> usize {
        packed_global_probe::capacity_for(context)
    }

    /// Key-extent global workspace bytes at a capacity.
    pub(super) fn source_global_workspace_bytes(keys: usize) -> usize {
        packed_global_probe::workspace_bytes(keys)
    }

    pub(super) fn source_global_mask_bytes(keys: usize) -> usize {
        packed_global_probe::mask_bytes(keys)
    }

    /// Bytes above the default global workspace, which the load-time source
    /// reserve already covers, needed at a context.
    pub(super) fn source_global_extra_bytes(
        context_workspace: bool,
        context: usize,
    ) -> usize {
        if !context_workspace || context > Self::source_context_limit(true) {
            return 0;
        }
        Self::source_global_workspace_bytes(Self::source_global_capacity(context))
            - Self::source_global_workspace_bytes(packed_global_probe::MAX_KEYS)
    }

    pub(super) const EXACT_WORKER_ROWS: usize = M5_PACKED_ROWS;

    pub(super) fn for_m5_worker(
        runtime: &mut Runtime,
        layers: &mut [Layer],
        source_allowed: bool,
        source_rows: usize,
    ) -> Result<Self> {
        if source_allowed && let Ok(mut packed) = Self::source_for_m5_worker(runtime, source_rows) {
            // Keep the source scheduler and ordinary grouped kernels if an
            // optional fast pipeline or its scratch fails to initialize.
            let _ = grouped_pilot::enable_rows64(runtime);
            let _ = packed.experts.enable_mxfp8_tensor(runtime, layers);
            runtime.parallel_route = register_parallel_route(runtime).is_ok();
            runtime.nvfp4_vector = register_nvfp4_vector(runtime).is_ok();
            return Ok(packed);
        }
        Self::new(runtime, M5_PACKED_ROWS, true, None, false, false, false)
    }

    fn source_for_m5_worker(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        if !Session::SOURCE_COMMAND_WIDTHS.contains(&rows) {
            return Err("Inkling source prefill width must be 2,048, 4,096 or 8,192 rows");
        }
        let packed = Self::new(runtime, rows, true, Some((true, true)), true, true, true)?;
        register(
            runtime,
            concat!(
                "#define INK_SOURCE_SINGLETON_CONV\n#define ink_conv ink_conv_source\n",
                include_str!("long_position.metal"),
                include_str!("serial.metal"),
            ),
            &["ink_conv_source"],
        )?;
        register(
            runtime,
            concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n",
                "#define INK_SOURCE_SINGLETON_EPILOGUE\n#define ink_nvfp4_up ink_nvfp4_up_source\n",
                include_str!("experts.metal"),
            ),
            &["ink_nvfp4_up_source"],
        )?;
        register(
            runtime,
            concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n",
                "#define INK_SOURCE_SINGLETON_MXFP8\n#define ink_mxfp8_up ink_mxfp8_source_up\n",
                "#define ink_mxfp8_down ink_mxfp8_source_down\n",
                include_str!("experts.metal"),
                "\n",
                include_str!("mxfp8.metal"),
            ),
            &["ink_mxfp8_source_up", "ink_mxfp8_source_down"],
        )?;
        Ok(packed)
    }

    pub(super) fn diagnostic(
        runtime: &mut Runtime,
        variant: &str,
    ) -> Result<Self> {
        let variant = variant.strip_prefix("packed").ok_or("expected packed prefill variant")?;
        let (variant, mxsrc) = variant.strip_suffix("+mxsrc").map_or((variant, false), |name| (name, true));
        let (variant, softmax) = variant.strip_suffix("+softmax").map_or((variant, false), |name| (name, true));
        let (variant, epilogue) = variant.strip_suffix("+epi").map_or((variant, false), |name| (name, true));
        let (variant, global) = variant.strip_suffix("+global").map_or((variant, false), |name| (name, true));
        let (variant, source) = if let Some(variant) = variant.strip_suffix("+routed") {
            (variant, Some((true, global)))
        } else if let Some(variant) = variant.strip_suffix("+source") {
            (variant, Some((false, global)))
        } else if global {
            return Err("global MLX probe requires packed source prefill");
        } else {
            (variant, None)
        };
        let (rows, wide) = variant.strip_suffix("+mxwide").map_or((variant, false), |rows| (rows, true));
        let rows = rows.parse().map_err(|_| "invalid packed prefill rows")?;
        Self::new(runtime, rows, wide, source, epilogue, softmax, mxsrc)
    }

    fn new(
        runtime: &mut Runtime,
        rows: usize,
        wide_mxfp8: bool,
        source: Option<(bool, bool)>,
        epilogue: bool,
        softmax: bool,
        mxsrc: bool,
    ) -> Result<Self> {
        if !runtime.pinned_m5 || !(128..=Self::SOURCE_COMMAND_ROWS.max(2048)).contains(&rows) {
            return Err("packed Inkling prefill requires M5 Max and 128 rows up to the source command width");
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
                "ink_packed_perturb",
            ],
        )?;
        register(
            runtime,
            concat!(include_str!("long_position.metal"), include_str!("packed_attention.metal")),
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
        if source.is_some() {
            let source_kernels = ["ink_prefill_dense_source", "ink_prefill_router_source", "ink_prefill_select_source"];
            let singleton = "ink_singleton_router_source";
            let [dense, router, select] = source_kernels;
            runtime.singleton_router = if register(
                runtime,
                concat!("#define INK_SOURCE_SINGLETON_ROUTER\n", include_str!("prefill_dense_source.metal")),
                &[dense, router, select, singleton],
            )
            .is_ok()
            {
                singleton
            } else {
                register(runtime, include_str!("prefill_dense_source.metal"), &source_kernels)?;
                router
            };
            register(
                runtime,
                include_str!("dense_fast.metal"),
                &["ink_dense_fast", DENSE_TILE.0, "ink_router_compare"],
            )?;
            runtime.router_tiles =
                register(runtime, include_str!("prefill_dense_source.metal"), &["ink_prefill_router_tiles"]).is_ok();
            register(
                runtime,
                include_str!("prefill_relative_source.metal"),
                &["ink_prefill_relative_table_source", "ink_prefill_relative_source"],
            )?;
            register(
                runtime,
                concat!(
                    "#define INK_RELATIVE_BF16_QUERY\n#define ink_prefill_relative_source ink_prefill_relative_fast\n",
                    "#define ink_prefill_relative_table_source ink_prefill_relative_table_fast_unused\n",
                    "#define ink_prefill_relative_gather ink_prefill_relative_gather_fast_unused\n",
                    include_str!("prefill_relative_source.metal"),
                ),
                &["ink_prefill_relative_fast"],
            )?;
            register(
                runtime,
                concat!(include_str!("long_position.metal"), include_str!("packed_attention.metal")),
                &["ink_packed_chronological_kv"],
            )?;
            register(
                runtime,
                concat!(
                    "#define INK_PACKED_SOURCE_CONV\n#define ink_packed_conv ink_packed_conv_source\n",
                    include_str!("long_position.metal"),
                    include_str!("packed_attention.metal"),
                ),
                &["ink_packed_conv_source"],
            )?;
            register(
                runtime,
                concat!(
                    "#define INK_PACKED_SOURCE_HEADS\n#define ink_packed_heads ink_packed_heads_source\n",
                    include_str!("long_position.metal"),
                    include_str!("packed_attention.metal"),
                ),
                &["ink_packed_heads_source"],
            )?;
        }
        let mut experts = GroupedExperts::new(runtime, rows)?;
        experts.min_rows = Self::SOURCE_TAIL_ROWS;
        if wide_mxfp8 {
            experts.enable_wide_mxfp8(runtime)?;
        } else {
            experts.enable_mxfp8(runtime)?;
        }
        if mxsrc {
            experts.enable_source_mxfp8(runtime)?;
        }
        if epilogue {
            experts.enable_source_epilogue(runtime)?;
        }
        if softmax {
            experts.enable_source_softmax(runtime)?;
        }
        let global = if source.is_some_and(|(_, global)| global) {
            Some(GlobalProbe::new(runtime)?)
        } else {
            None
        };
        let context_workspace = global.is_some();
        let device = &runtime.device;
        let bf16 = |stride| zeroed(device, rows * stride * 2);
        let f32 = |stride| zeroed(device, rows * stride * 4);
        Ok(Self {
            rows,
            context_workspace,
            source: if let Some((routed, _)) = source {
                let bytes = (LOCAL + rows + 2 * 128) * KV * 2;
                Some(SourcePrefill {
                    ordered_kv: [zeroed(device, bytes)?, zeroed(device, bytes)?],
                    routed,
                    global,
                })
            } else {
                None
            },
            source_allowed: true,
            draft_taps: None,
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

    pub(super) fn mxfp8_tensor_enabled(&self) -> bool {
        self.experts.mxfp8_tensor_enabled()
    }

    pub(super) fn mxfp8_replay_counts(&self) -> Option<Vec<u32>> {
        self.experts.mxfp8_replay_counts()
    }

    pub(super) fn mxfp8_false_certificates(&self) -> Option<Vec<u32>> {
        self.experts.mxfp8_false_certificates()
    }

    pub(super) fn set_source_scope(
        &mut self,
        enabled: bool,
    ) {
        self.source_allowed = enabled;
    }

    pub(super) fn supports_source_context(
        &self,
        context: usize,
    ) -> bool {
        self.source.is_some() && context <= Self::source_context_limit(self.context_workspace)
    }

    /// Whether the allocated global workspace covers a context. Without the
    /// context-sized workspace this equals `supports_source_context`.
    pub(super) fn source_ready(
        &self,
        context: usize,
    ) -> bool {
        self.supports_source_context(context)
            && self
                .source
                .as_ref()
                .and_then(|source| source.global.as_ref())
                .is_none_or(|global| context <= global.capacity())
    }

    pub(super) fn context_workspace(&self) -> bool {
        self.context_workspace
    }

    /// Size the context-sized global workspace for a session context. This
    /// does nothing without the source global probe.
    pub(super) fn set_source_context(
        &mut self,
        device: &Device,
        context: usize,
    ) -> Result<()> {
        if !self.context_workspace || !self.supports_source_context(context) {
            return Ok(());
        }
        match self.source.as_mut().and_then(|source| source.global.as_mut()) {
            Some(global) => global.resize(device, Self::source_global_capacity(context)),
            None => Ok(()),
        }
    }

    /// Additional global workspace bytes the session allocates at a context.
    pub(super) fn source_context_extra_bytes(
        &self,
        context: usize,
    ) -> usize {
        Self::source_global_extra_bytes(self.context_workspace, context)
    }

    /// Resizing releases the old staging buffers before replacing them.
    pub(super) fn source_workspace_growth_bytes(
        &self,
        context: usize,
    ) -> usize {
        if !self.context_workspace {
            return 0;
        }
        self.source.as_ref().and_then(|source| source.global.as_ref()).map_or(0, |global| {
            Self::source_global_workspace_bytes(Self::source_global_capacity(context))
                .saturating_sub(Self::source_global_workspace_bytes(global.capacity()))
        })
    }

    fn source_for_rows(
        &self,
        rows: usize,
        exact: bool,
    ) -> Option<&SourcePrefill> {
        // The default path runs every packed command on the same source
        // kernels; the exact reference keeps the pinned 65-row threshold.
        let min_rows = if exact {
            Self::SOURCE_MIN_ROWS
        } else {
            Self::SOURCE_TAIL_ROWS
        };
        self.source.as_ref().filter(|_| self.source_allowed && rows >= min_rows)
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
        // Default source prefill: one tensor-unit tiling with a single K
        // order at every command width. The exact reference keeps the
        // pinned thin tilings and width-dependent K splits below.
        if self.source_for_rows(rows, gpu.runtime.exact_prefill).is_some() && !gpu.runtime.exact_prefill {
            // Both operands staged in threadgroup memory per 128 x 64 tile;
            // same per-output arithmetic as ink_dense_fast.
            let (kernel, tile_rows, tile_columns) = DENSE_TILE;
            gpu.one(
                kernel,
                &[input, weight, output],
                [width as u32, columns as u32, rows as u32, 0],
                (columns.div_ceil(tile_columns), rows.div_ceil(tile_rows)),
                tile_rows / 32 * (tile_columns / 32) * 32,
            );
            return;
        }
        if rows > 64
            && columns <= 64
            && let Some(global) =
                self.source_for_rows(rows, gpu.runtime.exact_prefill).and_then(|source| source.global.as_ref())
        {
            global.project_thin(gpu, input, weight, output, width, columns, rows);
            return;
        }
        if self.source_for_rows(rows, gpu.runtime.exact_prefill).is_some() && columns > 64 {
            let largest = columns.max(rows);
            let split = if width >= 3 * largest || (largest <= 1024 && width > 2 * largest) {
                2048
            } else {
                width
            };
            gpu.one(
                "ink_prefill_dense_source",
                &[input, weight, output],
                [width as u32, columns as u32, rows as u32, split as u32],
                (columns.div_ceil(32), rows.div_ceil(16)),
                32,
            );
            return;
        }
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
            if self.source_for_rows(rows, gpu.runtime.exact_prefill).is_some() {
                "ink_packed_conv_source"
            } else {
                "ink_packed_conv"
            },
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
        gpu.phase(PrefillPhase::KeyValue);
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
            if self.source_for_rows(rows, gpu.runtime.exact_prefill).is_some() {
                "ink_packed_heads_source"
            } else {
                "ink_packed_heads"
            },
            &[&self.q, &self.conv_k, &self.conv_v, &self.r, &layer.q_scale, &layer.r_scale, &self.keys, &self.values],
            [start as u32, window as u32],
            (16, rows),
            128,
        );
        if let Some(source) = self.source_for_rows(rows, gpu.runtime.exact_prefill)
            && layer.local
        {
            let first = start.saturating_sub(LOCAL - 1);
            let aligned = first / 128 * 128;
            let backing = (start + rows - aligned).next_multiple_of(128);
            let [keys, values] = &source.ordered_kv;
            gpu.one(
                "ink_packed_chronological_kv",
                &[&visit.keys, &visit.values, &self.keys, &self.values, keys, values],
                [start as u32, rows as u32, visit.capacity as u32, aligned as u32, first as u32],
                (KV / 128, backing),
                128,
            );
            gpu.phase(PrefillPhase::RelativeTable);
            gpu.one(
                "ink_prefill_relative_table_source",
                &[&self.r, &layer.relative, &self.table],
                (rows * HEADS) as u32,
                (LOCAL / 32, (rows * HEADS).div_ceil(16)),
                32,
            );
            gpu.phase(PrefillPhase::LocalAttention);
            gpu.one(
                if gpu.runtime.exact_prefill {
                    "ink_prefill_relative_source"
                } else {
                    LOCAL_ATTENTION
                },
                &[&self.q, keys, values, &self.table, &self.sink, &self.gate, &self.attended],
                [start as u32, rows as u32, aligned as u32, backing as u32],
                (rows.div_ceil(16), HEADS),
                128,
            );
        } else if (rows > 64 || !gpu.runtime.exact_prefill)
            && let Some(global) =
                self.source_for_rows(rows, gpu.runtime.exact_prefill).and_then(|source| source.global.as_ref())
        {
            gpu.phase(PrefillPhase::RelativeTable);
            gpu.one(
                "ink_prefill_relative_table_source",
                &[&self.r, &layer.relative, &self.table],
                (rows * HEADS) as u32,
                (LOCAL / 32, (rows * HEADS).div_ceil(16)),
                32,
            );
            global.encode(gpu, self, visit, start, rows);
        } else {
            self.exact_attention(gpu, layer, visit, args, start, rows);
        }
        gpu.phase(PrefillPhase::KeyValue);
        gpu.one(
            "ink_packed_commit_kv",
            &[&self.keys, &self.values, &visit.keys, &visit.values],
            args,
            (KV / 128, rows.min(visit.capacity)),
            128,
        );
    }

    fn exact_attention(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        visit: &Visit,
        args: [u32; 4],
        start: usize,
        rows: usize,
    ) {
        gpu.phase(PrefillPhase::RelativeTable);
        gpu.one(
            "ink_packed_attention_table",
            &[&self.r, &layer.relative, &self.table],
            (),
            (LOCAL / 128, rows * HEADS),
            128,
        );
        gpu.phase(if layer.local {
            PrefillPhase::LocalAttention
        } else {
            PrefillPhase::GlobalAttention
        });
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
    }

    pub(super) fn enable_draft_taps(
        &mut self,
        runtime: &mut Runtime,
    ) -> Result<()> {
        register(runtime, include_str!("packed_draft.metal"), &["ink_packed_draft_tap", "ink_packed_draft_hidden"])?;
        self.draft_taps = Some(zeroed(&runtime.device, self.rows * 4 * WIDTH * 2)?);
        runtime.packed_draft_commit =
            register(runtime, include_str!("draft_commit.metal"), &["ink_draft_commit_mix", "ink_draft_commit_values"])
                .is_ok()
                && runtime.pipelines.contains_key("ink_dense_fast");
        Ok(())
    }

    pub(super) fn captures_all_draft_taps(&self) -> bool {
        self.draft_taps.is_some()
    }

    pub(super) fn packed_draft_row_matches(
        &self,
        row: usize,
        recorded: &[u16],
    ) -> bool {
        self.draft_taps.as_ref().is_some_and(|taps| {
            if row >= self.rows || recorded.len() != 4 * WIDTH {
                return false;
            }
            // The diagnostic reads only after the packed command completes.
            let base = unsafe { taps.contents().as_ptr().cast::<u16>().add(row * 4 * WIDTH) };
            unsafe { std::slice::from_raw_parts(base, 4 * WIDTH) == recorded }
        })
    }

    pub(super) fn encode(
        &self,
        session: &Session,
        gpu: &mut Stream<'_>,
        rows: usize,
        readout: bool,
    ) {
        self.encode_traced(session, gpu, rows, readout, None);
    }

    pub(super) fn encode_traced(
        &self,
        session: &Session,
        gpu: &mut Stream<'_>,
        rows: usize,
        readout: bool,
        trace: Option<&PrefillTrace>,
    ) {
        let start = session.position;
        let last = rows - 1;
        let last_partial = (&self.partial, last * WIDTH * 4);
        let draft = session.active_dspark().zip(self.draft_taps.as_ref());
        gpu.one(
            "ink_packed_embedding",
            &[&session.embedding, &session.embed_norm, &self.partial, &self.tokens],
            (),
            (rows, 1),
            WIDTH / 4,
        );
        if let Some(trace) = trace {
            trace.packed_partial(gpu, PHYSICAL.len(), &self.partial);
        }
        let mut blocks = 0;
        for (index, (&physical, visit)) in PHYSICAL.iter().zip(&session.visits).enumerate() {
            let taps = trace.filter(|trace| trace.visit == index);
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
            if let Some(tap) = DRAFT_TAPS.iter().position(|&i| i == index) {
                if let Some((_, features)) = draft {
                    gpu.one("ink_packed_draft_tap", &[&self.partial, features], tap as u32, (WIDTH / 128, rows), 128);
                }
                if session.capture_draft_features {
                    gpu.dispatch(
                        "ink_capture_feature",
                        &[last_partial, (&session.draft_features, tap * WIDTH * 2)],
                        (),
                        (WIDTH / 128, 1),
                        128,
                    );
                }
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
            if let Some(taps) = taps {
                taps.packed_stages(
                    gpu,
                    &[
                        ("trace.attn_norm_0", binding(&self.norms[0])),
                        ("trace.attn_norm_1", binding(&self.norms[1])),
                        ("trace.attn_norm_2", binding(&self.norms[2])),
                        ("trace.q_raw", binding(&self.q)),
                        ("trace.key_raw", binding(&self.k)),
                        ("trace.value_raw", binding(&self.v)),
                        ("trace.r_raw", binding(&self.r)),
                        ("trace.gate_raw", binding(&self.gate)),
                        ("trace.sink_raw", binding(&self.sink)),
                        ("trace.key_dynamic", binding(&self.dynamic[0])),
                        ("trace.value_dynamic", binding(&self.dynamic[1])),
                    ],
                );
            }
            self.attention(gpu, layer, visit, start, rows);
            if let Some(taps) = taps {
                taps.packed_stages(
                    gpu,
                    &[
                        ("trace.key_conv", binding(&self.conv_k)),
                        ("trace.value_conv", binding(&self.conv_v)),
                        ("trace.query_clipped", binding(&self.q)),
                        ("trace.relative_clipped", binding(&self.r)),
                        ("trace.relative_table", binding(&self.table)),
                        ("trace.attended", binding(&self.attended)),
                        ("key_rows", binding(&self.keys)),
                        ("value_rows", binding(&self.values)),
                    ],
                );
            }
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
            self.experts.encode(
                gpu,
                layer,
                rows,
                self.source_for_rows(rows, gpu.runtime.exact_prefill).is_some_and(|source| source.routed),
                self.source_for_rows(rows, gpu.runtime.exact_prefill).is_some(),
            );
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
            if let Some((visit, position, element)) = gpu.runtime.prefill_perturb
                && visit == index
                && (start..start + rows).contains(&position)
            {
                let at = [(position - start) as u32, element as u32];
                gpu.one("ink_packed_perturb", &[&self.partial], at, (1, 1), 1);
            }
            if let Some(taps) = taps {
                let [logits, routes, weights] = self.experts.routing(0);
                taps.packed_stages(
                    gpu,
                    &[
                        ("trace.attention_raw", binding(&self.attention_raw)),
                        ("attention", binding(&self.attention_output)),
                        ("mlp_input", binding(self.experts.input())),
                        ("trace.router_logits", logits),
                        ("routes", routes),
                        ("route_weights", weights),
                        ("experts", binding(self.experts.output())),
                        ("trace.mlp_conv", binding(&self.mlp)),
                    ],
                );
            }
            if let Some(trace) = trace {
                trace.packed_partial(gpu, index, &self.partial);
            }
        }
        if let Some((draft, taps)) = draft {
            gpu.phase(PrefillPhase::Other);
            self.mix(
                gpu,
                binding(&self.blocks),
                binding(&self.partial),
                &session.output_query,
                binding(&self.mixed),
                rows,
                1,
                blocks,
                true,
                4 * WIDTH,
                WIDTH,
            );
            gpu.one(
                "ink_packed_norm",
                &[&self.mixed, &session.output_norm, &self.mlp],
                [WIDTH as u32, WIDTH as u32],
                (rows, 1),
                WIDTH / 4,
            );
            gpu.one("ink_packed_draft_hidden", &[&self.mlp, taps], (), (WIDTH / 128, rows), 128);
            if gpu.runtime.packed_draft_commit && !gpu.runtime.exact_prefill {
                draft.commit_rows(gpu, taps, start, rows, [&self.mixed, &self.mlp, &self.k]);
            } else {
                for row in 0..rows {
                    draft.commit_from(gpu, (taps, row * 4 * WIDTH * 2), start + row);
                }
            }
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

/// Retained default; kept for the profile diagnostics.
pub(super) fn parallel_route_requested() -> bool {
    true
}

pub(super) fn register_parallel_route(runtime: &mut Runtime) -> Result<()> {
    let library = packed_source_library(&runtime.device)?;
    let function = library.new_function_with_name(PARALLEL_ROUTE_KERNEL).ok_or(PARALLEL_ROUTE_KERNEL)?;
    let pipeline =
        runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| PARALLEL_ROUTE_KERNEL)?;
    runtime.pipelines.insert(PARALLEL_ROUTE_KERNEL, pipeline);
    Ok(())
}

/// Retained default; kept for the profile diagnostics.
pub(super) fn nvfp4_vector_requested() -> bool {
    true
}

pub(super) fn register_nvfp4_vector(runtime: &mut Runtime) -> Result<()> {
    register(
        runtime,
        concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n",
            "#define INK_NVFP4_VECTOR_BLOCK\n#define INK_SOURCE_SINGLETON_EPILOGUE\n",
            "#define ink_nvfp4_up ink_nvfp4_vector_up\n#define ink_nvfp4_down ink_nvfp4_vector_down\n",
            include_str!("experts.metal"),
        ),
        &["ink_nvfp4_vector_up", "ink_nvfp4_vector_down"],
    )
}

pub(super) fn register_verify_nvfp4_matvec(runtime: &mut Runtime) -> Result<()> {
    register(
        runtime,
        concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n#define INK_NVFP4_VECTOR_BLOCK\n",
            include_str!("experts.metal"),
            include_str!("verify_nvfp4.metal"),
        ),
        &["ink_verify_nvfp4_group_dot"],
    )?;
    {
        register(
            runtime,
            concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n#define INK_NVFP4_VECTOR_BLOCK\n",
                "#define INK_VERIFY_EXACT_RUN_WIDTH\n#define ink_verify_nvfp4_group_dot ink_verify_nvfp4_exact_run_width\n",
                include_str!("experts.metal"),
                include_str!("verify_nvfp4.metal"),
            ),
            &["ink_verify_nvfp4_exact_run_width"],
        )?;
        runtime.verify_nvfp4_kernel = "ink_verify_nvfp4_exact_run_width";
    }
    {
        let source = concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n#define INK_NVFP4_VECTOR_BLOCK\n",
            "#define INK_VERIFY_EXACT_RUN_WIDTH\n#define INK_VERIFY_HALF_ACT\n",
            "#define ink_verify_nvfp4_group_dot ink_verify_nvfp4_half_dot\n",
            "#define ink_nvfp4_activation ink_verify_half_activation\n",
            include_str!("experts.metal"),
            include_str!("verify_nvfp4.metal"),
        );
        // The retained byte-pair coefficient table takes precedence over the
        // affine decoder (INK_VERIFY_AFFINE_FP4), which is now unused.
        let decoder = "#define INK_VERIFY_BYTE_COEFFICIENTS\n";
        let source = format!("{decoder}{source}");
        register(runtime, &source, &["ink_verify_nvfp4_half_dot", "ink_verify_half_activation"])?;
        runtime.verify_half_act = true;
    }
    Ok(())
}

pub(super) fn register(
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
