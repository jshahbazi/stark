// SPDX-License-Identifier: Apache-2.0
//! Decode slice for the pinned original Inkling 2 Mini checkpoint.
//! This is an explicit engine diagnostic. The generation worker does not
//! admit the family until the missing epilogue and normal entry points qualify.

mod checkpoint;
mod trace;
mod validation;

use std::{collections::HashMap, ptr::NonNull};

use checkpoint::Tensors;
use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLGPUFamily,
    MTLLanguageVersion, MTLLibraryExt, MTLMathMode, MTLResourceOptions, MTLSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
pub use validation::measure;

const WIDTH: usize = 2304;
const EXPERTS: usize = 128;
const MID: usize = 1152;
const KV: usize = 1024;
const LOCAL: usize = 1024;
const CAPACITY: usize = 4352;
const HEADS: usize = 24;
const START: usize = 4096;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = ProtocolObject<dyn MTLDevice>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Kernel = &'static str;
type Result<T> = std::result::Result<T, &'static str>;
type Binding<'a> = (&'a Buffer, usize);

fn binding(buffer: &Buffer) -> Binding<'_> {
    (buffer, 0)
}

fn zeroed(
    device: &Device,
    bytes: usize,
) -> Result<Buffer> {
    let buffer = device.new_buffer(bytes, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or("Metal allocation failed")?;
    // State initialization is outside the timed command. Shared buffers are
    // read and reset only after terminal completion.
    unsafe { buffer.contents().as_ptr().cast::<u8>().write_bytes(0, bytes) };
    Ok(buffer)
}

struct Runtime {
    device: Retained<Device>,
    queue: Queue,
    pipelines: HashMap<Kernel, Pipeline>,
    pinned_m5: bool,
}

impl Runtime {
    fn new() -> Result<Self> {
        let device = <dyn MTLDevice>::system_default().ok_or("Metal device unavailable")?;
        let queue = device.new_command_queue().ok_or("Metal queue unavailable")?;
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let source = concat!(
            include_str!("serial.metal"),
            "\n",
            include_str!("experts.metal"),
            "\n",
            include_str!("attention_source.metal")
        );
        let library = device
            .new_library_with_source(source, Some(&options))
            .map_err(|_| "Inkling Metal library compilation failed")?;
        let names = [
            "ink_matvec",
            "ink_matvec_source",
            "ink_mix",
            "ink_norm",
            "ink_controller_expand",
            "ink_conv",
            "ink_heads_and_cache",
            "ink_relative_table",
            "ink_attention_parts",
            "ink_attention_gate",
            "ink_attention_source_parts",
            "ink_attention_source_gate",
            "ink_after_attention",
            "ink_after_experts",
            "ink_router_logits",
            "ink_router_select",
            "ink_nvfp4_activation",
            "ink_nvfp4_up",
            "ink_nvfp4_down",
            "ink_expert_sum",
            "ink_copy",
            "ink_trace_copy",
        ];
        let mut pipelines = HashMap::new();
        for name in names {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            pipelines.insert(name, pipeline);
        }
        let pinned_m5 = device.name() == "Apple M5 Max"
            && device.supports_family(MTLGPUFamily::Apple10)
            && device.supports_family(MTLGPUFamily::Metal4);
        if pinned_m5 {
            const NAME: &str = "ink_controller_source";
            let library = device
                .new_library_with_source(include_str!("controller_source.metal"), Some(&options))
                .map_err(|_| NAME)?;
            let function = library.new_function_with_name(NAME).ok_or(NAME)?;
            let pipeline = device.new_compute_pipeline_state_with_function(&function).map_err(|_| NAME)?;
            pipelines.insert(NAME, pipeline);
        }
        Ok(Self {
            device,
            queue,
            pipelines,
            pinned_m5,
        })
    }
}

struct Stream<'a> {
    encoder: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
    runtime: &'a Runtime,
    dispatches: usize,
}

impl Stream<'_> {
    fn dispatch<T>(
        &mut self,
        kernel: Kernel,
        buffers: &[Binding<'_>],
        params: T,
        groups: (usize, usize),
        threads: usize,
    ) {
        self.encoder.set_compute_pipeline_state(&self.runtime.pipelines[kernel]);
        for (index, &(buffer, offset)) in buffers.iter().enumerate() {
            self.encoder.set_buffer(Some(buffer), offset, index);
        }
        if size_of::<T>() != 0 {
            self.encoder.set_bytes(NonNull::from(&params).cast(), size_of::<T>(), buffers.len());
        }
        self.encoder.dispatch_threadgroups(
            MTLSize {
                width: groups.0,
                height: groups.1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        self.dispatches += 1;
    }

    fn one<T>(
        &mut self,
        kernel: Kernel,
        buffers: &[&Buffer],
        params: T,
        groups: (usize, usize),
        threads: usize,
    ) {
        let mut bindings = [binding(buffers[0]); 9];
        for (slot, buffer) in bindings.iter_mut().zip(buffers) {
            *slot = binding(buffer);
        }
        self.dispatch(kernel, &bindings[..buffers.len()], params, groups, threads);
    }

    fn copy(
        &mut self,
        source: Binding<'_>,
        dest: Binding<'_>,
    ) {
        self.dispatch("ink_copy", &[source, dest], (), (WIDTH / 128, 1), 128);
    }

    fn project(
        &mut self,
        input: Binding<'_>,
        weight: &Buffer,
        output: &Buffer,
        columns: usize,
        rows: usize,
    ) {
        let (kernel, threads) = if self.runtime.pinned_m5 {
            ("ink_matvec_source", if rows <= 64 { 128 } else { 64 })
        } else {
            ("ink_matvec", 128)
        };
        self.dispatch(
            kernel,
            &[input, binding(weight), binding(output)],
            [columns as u32, rows as u32],
            (rows.div_ceil(4), 1),
            threads,
        );
    }

    fn mix(
        &mut self,
        blocks: &Buffer,
        partial: &Buffer,
        queries: &Buffer,
        out: &Buffer,
        count: usize,
        partial_live: bool,
        rows: usize,
    ) {
        self.one(
            "ink_mix",
            &[blocks, partial, queries, out],
            [count as u32, u32::from(partial_live)],
            (rows, 1),
            WIDTH / 4,
        );
    }

    fn norm(
        &mut self,
        input: Binding<'_>,
        weights: &Buffer,
        output: Binding<'_>,
        weighted: bool,
    ) {
        self.dispatch("ink_norm", &[input, binding(weights), output], u32::from(weighted), (1, 1), WIDTH / 4);
    }
}

struct Bank {
    codes: Buffer,
    blocks: Buffer,
    maxima: Buffer,
    denominator: f32,
}

impl Bank {
    fn load(
        source: &Tensors,
        device: &Device,
        prefix: &str,
        width: usize,
    ) -> Result<Self> {
        let key = format!("{prefix}_weight");
        let denominator = source.f32(&format!("{key}_outer_scale_denominator"))?;
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err("invalid expert outer denominator");
        }
        Ok(Self {
            codes: source.buffer(device, &key, "U8", &[EXPERTS, WIDTH, width / 2])?,
            blocks: source.buffer(device, &format!("{key}_scale"), "F8_E4M3", &[EXPERTS, WIDTH, width / 16])?,
            maxima: source.buffer(device, &format!("{key}_amax"), "F32", &[EXPERTS])?,
            denominator,
        })
    }

    fn args(&self) -> [u32; 3] {
        [WIDTH as u32, MID as u32, self.denominator.to_bits()]
    }
}

struct Gate {
    down: Buffer,
    gain: Buffer,
    forget: Buffer,
}

struct Layer {
    local: bool,
    attn_queries: Option<Buffer>,
    mlp_query: Buffer,
    gate_queries: Option<Buffer>,
    gates: Vec<Gate>,
    qkv_norm: [Buffer; 3],
    mlp_norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    r: Buffer,
    attn_gate: Buffer,
    sink: Buffer,
    output: Buffer,
    q_scale: Buffer,
    r_scale: Buffer,
    relative: Buffer,
    key_conv: Buffer,
    value_conv: Buffer,
    dynamic: Option<[Buffer; 2]>,
    attn_conv: Buffer,
    mlp_conv: Buffer,
    router: Buffer,
    bias: Buffer,
    up: Bank,
    down: Bank,
}

impl Layer {
    fn load(
        source: &Tensors,
        device: &Device,
        layer: usize,
    ) -> Result<Self> {
        const LLM: &str = "model.llm.";
        let stem = format!("{LLM}layers.{layer}.");
        let local = !matches!(layer, 5 | 11 | 17 | 23);
        let weight = |name: &str, shape: &[usize]| source.buffer(device, &format!("{stem}{name}"), "BF16", shape);
        let norm = |name| weight(name, &[WIDTH]);
        let projection = |name, rows, cols| weight(name, &[rows, cols]);
        let qkv_norm = if local {
            let shared = norm("attn_norm.weight")?;
            [shared.clone(), shared.clone(), shared]
        } else {
            [norm("q_input_norm.weight")?, norm("k_input_norm.weight")?, norm("v_input_norm.weight")?]
        };
        let attn_query_shape: &[usize] = if local {
            &[WIDTH]
        } else {
            &[3, WIDTH]
        };
        let attn_queries = layer
            .checked_sub(1)
            .map(|index| source.buffer(device, &format!("{LLM}attn_res.attn_queries.{index}"), "F32", attn_query_shape))
            .transpose()?;
        let mlp_query = source.buffer(device, &format!("{LLM}attn_res.mlp_queries.{layer}"), "F32", &[WIDTH])?;
        let (gate_queries, gates) = if (6..18).contains(&layer) {
            let index = 2 * (layer - 6);
            let queries = zeroed(device, 4 * WIDTH * 4)?;
            for (dest, name, row) in [
                (0, "middle_cycle_cross_pass_queries", index),
                (1, "middle_cycle_cross_pass_kv_queries", index),
                (2, "middle_cycle_cross_pass_kv_queries", index + 1),
                (3, "middle_cycle_cross_pass_queries", index + 1),
            ] {
                source.copy_to(
                    &format!("{LLM}{name}"),
                    "F32",
                    &[24, WIDTH],
                    row * WIDTH * 4,
                    &queries,
                    dest * WIDTH * 4,
                    WIDTH * 4,
                )?;
            }
            let mut gates = Vec::new();
            for slot in ["attn", "k", "v", "mlp"] {
                let w = |part, shape: &[usize]| weight(&format!("controllers.{slot}.{part}_weight"), shape);
                gates.push(Gate {
                    down: w("down", &[64, 2 * WIDTH])?,
                    gain: w("gain", &[WIDTH, 64])?,
                    forget: w("forget", &[WIDTH, 64])?,
                });
            }
            (Some(queries), gates)
        } else {
            (None, Vec::new())
        };
        let (key_conv, value_conv, dynamic) = if local {
            (weight("attn.k_sconv.weight", &[KV, 1, 4])?, weight("attn.v_sconv.weight", &[KV, 1, 4])?, None)
        } else {
            (
                weight("attn.k_sconv.static_w", &[4, KV])?,
                weight("attn.v_sconv.static_w", &[4, KV])?,
                Some([
                    projection("attn.k_sconv.weight_proj", 32, WIDTH)?,
                    projection("attn.v_sconv.weight_proj", 32, WIDTH)?,
                ]),
            )
        };
        Ok(Self {
            local,
            attn_queries,
            mlp_query,
            gate_queries,
            gates,
            qkv_norm,
            mlp_norm: norm("mlp_norm.weight")?,
            q: projection("attn.wq_du.weight", 3072, WIDTH)?,
            k: projection("attn.wk_dv.weight", KV, WIDTH)?,
            v: projection("attn.wv_dv.weight", KV, WIDTH)?,
            r: projection("attn.wr_du.weight", HEADS * 16, WIDTH)?,
            attn_gate: projection("attn.gate_attn.weight", 3072, WIDTH)?,
            sink: projection("attn.attn_sink_proj.weight", HEADS, WIDTH)?,
            output: projection("attn.wo_ud.weight", WIDTH, 3072)?,
            q_scale: weight("attn.q_l2_clip.scale", &[HEADS, 128])?,
            r_scale: weight("attn.rel_l2_clip.scale", &[HEADS, 16])?,
            relative: weight("attn.rel_logits_proj.proj", &[16, LOCAL])?,
            key_conv,
            value_conv,
            dynamic,
            attn_conv: weight("attn_sconv.weight", &[WIDTH, 1, 4])?,
            mlp_conv: weight("mlp_sconv.weight", &[WIDTH, 1, 4])?,
            router: weight("mlp.gate.weight", &[EXPERTS, WIDTH])?,
            bias: source.buffer(device, &format!("{stem}mlp.gate.bias"), "F32", &[EXPERTS])?,
            up: Bank::load(source, device, &format!("{stem}mlp.experts.w13"), WIDTH)?,
            down: Bank::load(source, device, &format!("{stem}mlp.experts.w2"), MID)?,
        })
    }

    fn norms(
        &self,
        gpu: &mut Stream<'_>,
        s: &Scratch,
        input: Binding<'_>,
        output: &Buffer,
        slot: usize,
        later: bool,
    ) {
        let scale = if slot == 3 {
            &self.mlp_norm
        } else {
            &self.qkv_norm[slot]
        };
        if later {
            let msg = (&s.messages, slot * WIDTH * 4);
            gpu.norm(input, scale, binding(&s.controller_pair), false);
            gpu.norm(msg, scale, (&s.controller_pair, WIDTH * 2), false);
            s.record(gpu, trace::CONTROLLER_PAIR[slot], &s.controller_pair);
            let gate = &self.gates[slot];
            gpu.project(binding(&s.controller_pair), &gate.down, &s.compressed, 2 * WIDTH, 64);
            s.record(gpu, trace::CONTROLLER_RANK[slot], &s.compressed);
            let (kernel, groups, threads) = if gpu.runtime.pinned_m5 {
                ("ink_controller_source", WIDTH / 32, 32)
            } else {
                ("ink_controller_expand", WIDTH / 4, 128)
            };
            gpu.dispatch(
                kernel,
                &[binding(&s.compressed), binding(&gate.gain), binding(&gate.forget), input, msg, binding(&s.gated)],
                (),
                (groups, 1),
                threads,
            );
            s.record(gpu, trace::CONTROLLER_OUT[slot], &s.gated);
            gpu.norm(binding(&s.gated), scale, binding(output), true);
        } else {
            gpu.norm(input, scale, binding(output), true);
        }
        if slot < 3 {
            s.record(gpu, trace::ATTN_NORM[slot], output);
        }
    }

    fn experts(
        &self,
        gpu: &mut Stream<'_>,
        s: &Scratch,
    ) {
        gpu.one("ink_router_logits", &[&s.mlp_input, &self.router, &s.logits], WIDTH as u32, (EXPERTS, 1), 32);
        gpu.one("ink_router_select", &[&s.logits, &self.bias, &s.routes, &s.route_weights], (), (1, 1), 128);
        gpu.one("ink_nvfp4_activation", &[&s.mlp_input, &s.up_inner, &s.up_max], WIDTH as u32, (1, 1), 256);
        gpu.one(
            "ink_nvfp4_up",
            &[
                &s.up_inner,
                &s.up_max,
                &s.routes,
                &s.route_weights,
                &self.up.codes,
                &self.up.blocks,
                &self.up.maxima,
                &s.mid,
            ],
            self.up.args(),
            (MID / 4, 8),
            128,
        );
        gpu.one("ink_nvfp4_activation", &[&s.mid, &s.down_inner, &s.down_max], MID as u32, (8, 1), 256);
        gpu.one(
            "ink_nvfp4_down",
            &[&s.down_inner, &s.down_max, &s.routes, &self.down.codes, &self.down.blocks, &self.down.maxima, &s.down],
            self.down.args(),
            (WIDTH / 4, 8),
            128,
        );
        gpu.one("ink_expert_sum", &[&s.down, &s.experts], WIDTH as u32, (WIDTH / 128, 1), 128);
    }

    fn convolution(
        gpu: &mut Stream<'_>,
        input: &Buffer,
        weight: &Buffer,
        extra: Option<&Buffer>,
        history: &Buffer,
        output: &Buffer,
        width: usize,
        position: usize,
    ) {
        gpu.one(
            "ink_conv",
            &[input, weight, extra.unwrap_or(weight), history, output],
            [width as u32, position as u32, u32::from(extra.is_some()), 0],
            (width / 128, 1),
            128,
        );
    }

    fn attention(
        &self,
        gpu: &mut Stream<'_>,
        s: &Scratch,
        cache: &Visit,
        position: usize,
    ) {
        for ((input, weight, out, rows), name) in [
            (&s.qkv_input[0], &self.q, &s.q, 3072),
            (&s.qkv_input[1], &self.k, &s.k, KV),
            (&s.qkv_input[2], &self.v, &s.v, KV),
            (&s.qkv_input[0], &self.r, &s.r, HEADS * 16),
            (&s.qkv_input[0], &self.attn_gate, &s.attn_gate, 3072),
            (&s.qkv_input[0], &self.sink, &s.sink, HEADS),
        ]
        .into_iter()
        .zip(trace::RAW_PROJECTIONS)
        {
            gpu.project(binding(input), weight, out, WIDTH, rows);
            s.record(gpu, name, out);
        }
        if let Some(dynamic) = &self.dynamic {
            for (i, weight) in dynamic.iter().enumerate() {
                gpu.project(binding(&s.qkv_input[i + 1]), weight, &s.dynamic[i], WIDTH, 32);
            }
        }
        let width = cache.capacity as u32;
        let window = if self.local {
            LOCAL as u32
        } else {
            0
        };
        let visible = (position + 1).min(cache.capacity);
        let shards = if visible < LOCAL {
            32
        } else {
            64
        };
        let args = [position as u32, width, window, shards];
        Self::convolution(
            gpu,
            &s.k,
            &self.key_conv,
            self.dynamic.as_ref().map(|_| &s.dynamic[0]),
            &cache.conv[0],
            &s.conv_k,
            KV,
            position,
        );
        Self::convolution(
            gpu,
            &s.v,
            &self.value_conv,
            self.dynamic.as_ref().map(|_| &s.dynamic[1]),
            &cache.conv[1],
            &s.conv_v,
            KV,
            position,
        );
        gpu.one(
            "ink_heads_and_cache",
            &[&s.q, &s.conv_k, &s.conv_v, &s.r, &self.q_scale, &self.r_scale, &cache.keys, &cache.values],
            args,
            (16, 1),
            128,
        );
        gpu.one("ink_relative_table", &[&s.r, &self.relative, &s.table], (), (HEADS * LOCAL / 128, 1), 128);
        let (parts_kernel, gate_kernel, parts_groups, gate_groups, shape) = if gpu.runtime.pinned_m5 {
            (
                "ink_attention_source_parts",
                "ink_attention_source_gate",
                (shards as usize / 4, HEADS),
                (8, HEADS),
                [shards, shards],
            )
        } else {
            let parts = visible.div_ceil(32);
            (
                "ink_attention_parts",
                "ink_attention_gate",
                (parts, HEADS),
                (HEADS, 1),
                [parts as u32, width.div_ceil(32)],
            )
        };
        gpu.one(
            parts_kernel,
            &[&s.q, &cache.keys, &cache.values, &s.table, &s.sink, &s.parts],
            args,
            parts_groups,
            128,
        );
        gpu.one(gate_kernel, &[&s.parts, &s.attn_gate, &s.attended], shape, gate_groups, 128);
        s.record(gpu, "attended", &s.attended);
        gpu.project(binding(&s.attended), &self.output, &s.attn_raw, 3072, WIDTH);
        s.record(gpu, "attn_raw", &s.attn_raw);
        Self::convolution(gpu, &s.attn_raw, &self.attn_conv, None, &cache.conv[2], &s.attention, WIDTH, position);
    }

    fn encode(
        &self,
        gpu: &mut Stream<'_>,
        cache: &Visit,
        flow: &Flow,
        partial: &Buffer,
        state: VisitInput,
        position: usize,
    ) {
        let s = &cache.scratch;
        let queries = self.attn_queries.as_ref().unwrap_or(&self.mlp_query);
        if state.blocks != 0 {
            gpu.mix(
                &flow.blocks,
                partial,
                queries,
                &s.mixed,
                state.blocks,
                true,
                if self.local {
                    1
                } else {
                    3
                },
            );
            s.record(gpu, "attn_mixed", &s.mixed);
        }
        if state.later {
            gpu.mix(&flow.previous, partial, self.gate_queries.as_ref().unwrap(), &s.messages, 2, false, 4);
        }
        for slot in 0..3 {
            let offset = if self.local {
                0
            } else {
                slot * WIDTH * 4
            };
            let input = if state.blocks == 0 {
                binding(partial)
            } else {
                (&s.mixed, offset)
            };
            self.norms(gpu, s, input, &s.qkv_input[slot], slot, state.later);
        }
        self.attention(gpu, s, cache, position);
        if state.boundary {
            gpu.copy(binding(partial), (&flow.blocks, state.blocks * WIDTH * 4));
        }
        gpu.one("ink_after_attention", &[partial, &s.attention], u32::from(state.boundary), (WIDTH / 128, 1), 128);
        gpu.mix(&flow.blocks, partial, &self.mlp_query, &s.mixed, state.blocks + usize::from(state.boundary), true, 1);
        s.record(gpu, "mlp_mixed", &s.mixed);
        self.norms(gpu, s, binding(&s.mixed), &s.mlp_input, 3, state.later);
        self.experts(gpu, s);
        s.record(gpu, "router_logits", &s.logits);
        Self::convolution(gpu, &s.experts, &self.mlp_conv, None, &cache.conv[3], &s.mlp, WIDTH, position);
        gpu.one("ink_after_experts", &[partial, &s.mlp], (), (WIDTH / 128, 1), 128);
        if state.capture {
            gpu.copy(binding(partial), binding(&s.partial_debug));
        }
    }
}

#[derive(Clone, Copy)]
struct VisitInput {
    blocks: usize,
    later: bool,
    boundary: bool,
    capture: bool,
}

struct Flow {
    blocks: Buffer,
    previous: Buffer,
    partial: Buffer,
    prelude: Buffer,
}

impl Flow {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            blocks: zeroed(device, 4 * WIDTH * 4)?,
            previous: zeroed(device, 2 * WIDTH * 4)?,
            partial: zeroed(device, WIDTH * 4)?,
            prelude: zeroed(device, WIDTH * 4)?,
        })
    }
}

struct Scratch {
    trace: Option<trace::Trace>,
    mixed: Buffer,
    messages: Buffer,
    controller_pair: Buffer,
    compressed: Buffer,
    gated: Buffer,
    qkv_input: [Buffer; 3],
    q: Buffer,
    k: Buffer,
    v: Buffer,
    r: Buffer,
    attn_gate: Buffer,
    sink: Buffer,
    dynamic: [Buffer; 2],
    conv_k: Buffer,
    conv_v: Buffer,
    table: Buffer,
    parts: Buffer,
    attended: Buffer,
    attn_raw: Buffer,
    attention: Buffer,
    mlp_input: Buffer,
    logits: Buffer,
    routes: Buffer,
    route_weights: Buffer,
    up_inner: Buffer,
    up_max: Buffer,
    mid: Buffer,
    down_inner: Buffer,
    down_max: Buffer,
    down: Buffer,
    experts: Buffer,
    mlp: Buffer,
    partial_debug: Buffer,
}

impl Scratch {
    fn record(
        &self,
        gpu: &mut Stream<'_>,
        name: &str,
        source: &Buffer,
    ) {
        if let Some(trace) = &self.trace {
            trace.encode(gpu, name, source);
        }
    }
}

struct Visit {
    capacity: usize,
    keys: Buffer,
    values: Buffer,
    conv: [Buffer; 4],
    scratch: Scratch,
}

impl Visit {
    fn new(
        device: &Device,
        local: bool,
    ) -> Result<Self> {
        let capacity = if local {
            LOCAL
        } else {
            CAPACITY
        };
        let bf = |n| zeroed(device, n * 2);
        let wide = |n| zeroed(device, n * 4);
        Ok(Self {
            capacity,
            keys: bf(capacity * KV)?,
            values: bf(capacity * KV)?,
            conv: [bf(3 * KV)?, bf(3 * KV)?, bf(3 * WIDTH)?, bf(3 * WIDTH)?],
            scratch: Scratch {
                trace: None,
                mixed: wide(3 * WIDTH)?,
                messages: wide(4 * WIDTH)?,
                controller_pair: bf(2 * WIDTH)?,
                compressed: bf(64)?,
                gated: wide(WIDTH)?,
                qkv_input: [bf(WIDTH)?, bf(WIDTH)?, bf(WIDTH)?],
                q: bf(3072)?,
                k: bf(KV)?,
                v: bf(KV)?,
                r: bf(HEADS * 16)?,
                attn_gate: bf(3072)?,
                sink: bf(HEADS)?,
                dynamic: [bf(32)?, bf(32)?],
                conv_k: bf(KV)?,
                conv_v: bf(KV)?,
                table: bf(HEADS * LOCAL)?,
                parts: wide(HEADS * capacity.div_ceil(32).max(64) * 130)?,
                attended: bf(3072)?,
                attn_raw: bf(WIDTH)?,
                attention: bf(WIDTH)?,
                mlp_input: bf(WIDTH)?,
                logits: wide(EXPERTS)?,
                routes: wide(8)?,
                route_weights: wide(8)?,
                up_inner: wide(WIDTH)?,
                up_max: wide(1)?,
                mid: bf(8 * MID)?,
                down_inner: wide(8 * MID)?,
                down_max: wide(8)?,
                down: bf(8 * WIDTH)?,
                experts: bf(WIDTH)?,
                mlp: bf(WIDTH)?,
                partial_debug: wide(WIDTH)?,
            },
        })
    }
}

struct Timing {
    dispatches: usize,
    wall_ms: f64,
    gpu_ms: f64,
}

fn execute(
    runtime: &Runtime,
    encode: impl FnOnce(&mut Stream<'_>),
) -> Result<Timing> {
    let start = std::time::Instant::now();
    let command = runtime.queue.command_buffer().ok_or("Metal command allocation failed")?;
    let encoder = command.compute_command_encoder().ok_or("Metal encoder allocation failed")?;
    let mut gpu = Stream {
        encoder: &encoder,
        runtime,
        dispatches: 0,
    };
    encode(&mut gpu);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err("Inkling command failed");
    }
    Ok(Timing {
        dispatches: gpu.dispatches,
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        gpu_ms: (command.gpu_end_time() - command.gpu_start_time()) * 1000.0,
    })
}
