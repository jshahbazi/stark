// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only comparison at the pinned source's 512-row prompt boundary.

use half::bf16;
use serde_json::{Value, json};

use super::*;

pub(super) const CHUNK: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrefillTraceMode {
    Baseline,
    Dense,
    Local,
    Routed,
}

impl PrefillTraceMode {
    pub(super) fn lanes(
        self,
        runtime: &mut Runtime,
    ) -> Result<PrefillLanes> {
        let mut lanes = PrefillLanes::new(runtime, CHUNK)?;
        if self != Self::Baseline {
            lanes.dense = Some(prefill_dense::PrefillDense::new(runtime, CHUNK)?);
        }
        if matches!(self, Self::Local | Self::Routed) {
            lanes.relative = Some(prefill_relative::PrefillRelative::new(runtime, CHUNK)?);
        }
        lanes.source_router = self == Self::Routed;
        Ok(lanes)
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Token,
    TokenHeads(usize),
    Heads(usize),
    Matrix,
}

struct Tap {
    name: &'static str,
    width: usize,
    dtype: &'static str,
    layout: Layout,
    buffer: Buffer,
}

pub(super) struct PrefillTrace {
    pub(super) visit: usize,
    partials: Buffer,
    stages: Vec<Tap>,
}

impl PrefillTrace {
    fn new(
        device: &Device,
        source: &Tensors,
        visit: usize,
    ) -> Result<Self> {
        use Layout::*;
        let specs = [
            ("trace.attn_norm_0", WIDTH, "BF16", Token),
            ("trace.attn_norm_1", WIDTH, "BF16", Token),
            ("trace.attn_norm_2", WIDTH, "BF16", Token),
            ("trace.q_raw", HEADS * 128, "BF16", TokenHeads(HEADS)),
            ("trace.key_raw", KV, "BF16", Token),
            ("trace.value_raw", KV, "BF16", Token),
            ("trace.r_raw", HEADS * 16, "BF16", TokenHeads(HEADS)),
            ("trace.gate_raw", HEADS * 128, "BF16", Token),
            ("trace.sink_raw", HEADS, "BF16", Heads(HEADS)),
            ("trace.key_dynamic", 32, "BF16", TokenHeads(8)),
            ("trace.value_dynamic", 32, "BF16", TokenHeads(8)),
            ("trace.key_conv", KV, "BF16", TokenHeads(8)),
            ("trace.value_conv", KV, "BF16", TokenHeads(8)),
            ("trace.query_clipped", HEADS * 128, "BF16", Heads(HEADS)),
            ("key_rows", KV, "BF16", Matrix),
            ("value_rows", KV, "BF16", Matrix),
            ("trace.relative_clipped", HEADS * 16, "BF16", TokenHeads(HEADS)),
            ("trace.relative_table", HEADS * LOCAL, "BF16", Heads(HEADS)),
            ("trace.attended", HEADS * 128, "BF16", Token),
            ("trace.attention_raw", WIDTH, "BF16", Token),
            ("attention", WIDTH, "BF16", Token),
            ("mlp_input", WIDTH, "BF16", Token),
            ("trace.router_logits", EXPERTS, "F32", Matrix),
            ("routes", 8, "U32", Matrix),
            ("route_weights", 8, "F32", Matrix),
            ("experts", WIDTH, "BF16", Token),
            ("trace.mlp_conv", WIDTH, "BF16", Token),
        ];
        let stages = specs
            .into_iter()
            .filter(|(name, ..)| source.contains(&format!("stage.{name}")))
            .map(|(name, width, dtype, layout)| {
                let bytes = CHUNK
                    * width
                    * if dtype == "BF16" {
                        2
                    } else {
                        4
                    };
                Ok(Tap {
                    name,
                    width,
                    dtype,
                    layout,
                    buffer: zeroed(device, bytes)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            visit,
            partials: zeroed(device, (PHYSICAL.len() + 1) * CHUNK * WIDTH * 4)?,
            stages,
        })
    }

    fn copy(
        gpu: &mut Stream<'_>,
        source: Binding<'_>,
        destination: Binding<'_>,
        bytes: usize,
    ) {
        let words = bytes / 4;
        gpu.dispatch("ink_trace_copy", &[source, destination], words as u32, (words.div_ceil(128), 1), 128);
    }

    fn stage(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
        name: &str,
        source: Binding<'_>,
    ) {
        if let Some(tap) = self.stages.iter().find(|tap| tap.name == name) {
            let bytes = tap.buffer.length() / CHUNK;
            Self::copy(gpu, source, (&tap.buffer, row * bytes), bytes);
        }
    }

    pub(super) fn input(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
        flow: &Flow,
    ) {
        // The residual work buffer is F32 and every committed value was rounded to BF16.
        gpu.copy(binding(&flow.partial), (&self.partials, (PHYSICAL.len() * CHUNK + row) * WIDTH * 4));
    }

    pub(super) fn projected(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
        s: &Scratch,
    ) {
        for (name, buffer) in [
            ("trace.attn_norm_0", &s.qkv_input[0]),
            ("trace.attn_norm_1", &s.qkv_input[1]),
            ("trace.attn_norm_2", &s.qkv_input[2]),
            ("trace.q_raw", &s.q),
            ("trace.key_raw", &s.k),
            ("trace.value_raw", &s.v),
            ("trace.r_raw", &s.r),
            ("trace.gate_raw", &s.attn_gate),
            ("trace.sink_raw", &s.sink),
            ("trace.key_dynamic", &s.dynamic[0]),
            ("trace.value_dynamic", &s.dynamic[1]),
        ] {
            self.stage(gpu, row, name, binding(buffer));
        }
    }

    pub(super) fn attended(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
        s: &Scratch,
        visit: &Visit,
        position: usize,
    ) {
        for (name, buffer) in [
            ("trace.key_conv", &s.conv_k),
            ("trace.value_conv", &s.conv_v),
            ("trace.query_clipped", &s.q),
            ("trace.relative_clipped", &s.r),
            ("trace.relative_table", &s.table),
            ("trace.attended", &s.attended),
        ] {
            self.stage(gpu, row, name, binding(buffer));
        }
        let offset = (position % visit.capacity) * KV * 2;
        self.stage(gpu, row, "key_rows", (&visit.keys, offset));
        self.stage(gpu, row, "value_rows", (&visit.values, offset));
    }

    pub(super) fn routing(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
        buffers: [Binding<'_>; 3],
    ) {
        for (name, buffer) in ["trace.router_logits", "routes", "route_weights"].into_iter().zip(buffers) {
            self.stage(gpu, row, name, buffer);
        }
    }

    pub(super) fn output(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
        s: &Scratch,
    ) {
        for (name, buffer) in [
            ("trace.attention_raw", &s.attn_raw),
            ("attention", &s.attention),
            ("mlp_input", &s.mlp_input),
            ("experts", &s.experts),
            ("trace.mlp_conv", &s.mlp),
        ] {
            self.stage(gpu, row, name, binding(buffer));
        }
    }

    pub(super) fn partial(
        &self,
        gpu: &mut Stream<'_>,
        visit: usize,
        row: usize,
        flow: &Flow,
    ) {
        gpu.copy(binding(&flow.partial), (&self.partials, (visit * CHUNK + row) * WIDTH * 4));
    }

    fn compare(
        &self,
        source: &Tensors,
    ) -> Result<Value> {
        let partials = read_shared::<f32>(&self.partials, (PHYSICAL.len() + 1) * CHUNK * WIDTH);
        let residual = |index, name: &str| -> Result<Value> {
            let start = index * CHUNK * WIDTH;
            let actual = partials[start..start + CHUNK * WIDTH].iter().map(|&f| bf16::from_f32(f).to_bits());
            let expected = source.values::<u16>(name, "BF16", &[1, CHUNK, WIDTH])?;
            Ok(differences(WIDTH, actual, &expected))
        };
        let visits = (0..PHYSICAL.len())
            .map(|i| residual(i, &format!("full.visit.{i:02}.partial")))
            .collect::<Result<Vec<_>>>()?;
        let mut stages = vec![json!({
            "stage": "visit_input",
            "words": residual(PHYSICAL.len(), "stage.trace.embedding_rms")?,
        })];
        for tap in &self.stages {
            let diff = if tap.dtype == "BF16" {
                tap.compare::<u16>(source)?
            } else {
                tap.compare::<u32>(source)?
            };
            stages.push(json!({"stage":tap.name,"words":diff}));
        }
        Ok(json!({"visits":visits,"selected_visit":self.visit,"stages":stages}))
    }
}

impl Tap {
    fn compare<T: bytemuck::Pod + Eq>(
        &self,
        source: &Tensors,
    ) -> Result<Value> {
        let n = self.width;
        let shape = match self.layout {
            Layout::Token => vec![1, CHUNK, n],
            Layout::TokenHeads(h) => vec![1, CHUNK, h, n / h],
            Layout::Heads(h) => vec![1, h, CHUNK, n / h],
            Layout::Matrix => vec![CHUNK, n],
        };
        let expected = source.values::<T>(&format!("stage.{}", self.name), self.dtype, &shape)?;
        let actual = read_shared::<T>(&self.buffer, CHUNK * n);
        if let Layout::Heads(heads) = self.layout {
            let width = n / heads;
            let expected = (0..CHUNK)
                .flat_map(|row| {
                    (0..heads).flat_map({
                        let expected = &expected;
                        move |head| {
                            expected[(head * CHUNK + row) * width..(head * CHUNK + row + 1) * width].iter().copied()
                        }
                    })
                })
                .collect::<Vec<_>>();
            Ok(differences(n, actual.into_iter(), &expected))
        } else {
            Ok(differences(n, actual.into_iter(), &expected))
        }
    }
}

fn differences<T: Eq>(
    width: usize,
    actual: impl Iterator<Item = T>,
    expected: &[T],
) -> Value {
    let mut unequal = 0;
    let mut first = None;
    let mut last_row = None;
    let mut rows = 0;
    for (i, (a, b)) in actual.zip(expected).enumerate() {
        if &a != b {
            unequal += 1;
            first.get_or_insert(i / width);
            if last_row != Some(i / width) {
                last_row = Some(i / width);
                rows += 1;
            }
        }
    }
    json!({"values":expected.len(),"unequal":unequal,"unequal_rows":rows,"first_row":first})
}

pub fn trace_prefill(
    checkpoint: &Path,
    oracle: &Path,
    mode: PrefillTraceMode,
) -> Result<Value> {
    let source = Tensors::open(&oracle.join("prefill.safetensors"))?;
    let chunk = source.values::<u32>("chunk", "U32", &[3])?;
    let [offset, rows, visit] = [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
    if rows != CHUNK || offset % CHUNK != 0 || offset >= 4096 || visit >= PHYSICAL.len() {
        return Err("expected a valid pinned 512-row Inkling trace");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[4100])?;
    if tokens.iter().any(|&token| token as usize >= VOCABULARY) {
        return Err("Inkling oracle token is outside the vocabulary");
    }
    let mut session = Session::load(checkpoint, 4352)?;
    let lanes = mode.lanes(&mut session.runtime)?;
    let trace = PrefillTrace::new(&session.runtime.device, &source, visit)?;
    let mut comparison = Value::Null;
    let mut wall_ms = 0.0;
    for (index, inputs) in tokens[..4096].chunks_exact(CHUNK).enumerate() {
        let inputs = inputs.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
        let capture = index * CHUNK == offset;
        let timing = execute(&session.runtime, |gpu| {
            session.encode_prefill_trace(gpu, &lanes, &inputs, false, capture.then_some(&trace));
        })?;
        wall_ms += timing.wall_ms;
        session.position += CHUNK;
        if capture {
            comparison = trace.compare(&source)?;
        }
    }
    let mut teacher_logits = Vec::new();
    for (index, &token) in tokens[4096..].iter().enumerate() {
        let file = Tensors::open(&oracle.join(format!("row-{index}.safetensors")))?;
        let expected = file.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
        let actual = session.step(token)?;
        let top = |logits: &[f32]| logits.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i);
        teacher_logits.push(json!({
            "error": target_validation::report(&actual, &expected),
            "greedy_equal": top(&actual) == top(&expected),
        }));
    }
    Ok(json!({"chunk_offset":offset,"chunk_rows":CHUNK,"mode":format!("{mode:?}"),"comparison":comparison,
        "instrumented_prefill_ms":wall_ms,"teacher_logits":teacher_logits}))
}
