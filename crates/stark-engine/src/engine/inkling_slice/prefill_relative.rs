// SPDX-License-Identifier: Apache-2.0
//! Pinned local prefill attention with a chronological view of the native ring.

use super::*;

const HEAD_WIDTH: usize = 128;
const QUERY_WIDTH: usize = HEADS * HEAD_WIDTH;

pub(super) struct PrefillRelative {
    query: Buffer,
    relative: Buffer,
    table: Buffer,
    sink: Buffer,
    gate: Buffer,
    attended: Buffer,
    keys: Buffer,
    values: Buffer,
}

impl PrefillRelative {
    pub(super) fn new(
        runtime: &mut Runtime,
        rows: usize,
    ) -> Result<Self> {
        let device = &runtime.device;
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = device
            .new_library_with_source(include_str!("prefill_relative_source.metal"), Some(&options))
            .map_err(|_| "Inkling batched relative attention shader compilation failed")?;
        for name in ["ink_prefill_relative_gather", "ink_prefill_relative_table_source", "ink_prefill_relative_source"]
        {
            let function = library.new_function_with_name(name).ok_or("missing Inkling relative attention kernel")?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| "Inkling relative attention pipeline compilation failed")?;
            runtime.pipelines.insert(name, pipeline);
        }
        let workspace = |width| zeroed(device, rows * width * 2);
        let history = || zeroed(device, (LOCAL + rows + 2 * HEAD_WIDTH) * KV * 2);
        Ok(Self {
            query: workspace(QUERY_WIDTH)?,
            relative: workspace(HEADS * 16)?,
            table: workspace(HEADS * LOCAL)?,
            sink: workspace(HEADS)?,
            gate: workspace(QUERY_WIDTH)?,
            attended: workspace(QUERY_WIDTH)?,
            keys: history()?,
            values: history()?,
        })
    }

    pub(super) fn gather(
        &self,
        gpu: &mut Stream<'_>,
        visit: &Visit,
        chunk_start: usize,
        rows: usize,
        prior: bool,
    ) {
        let first = chunk_start.saturating_sub(LOCAL - 1);
        let end = chunk_start + rows;
        let aligned = first / HEAD_WIDTH * HEAD_WIDTH;
        let (from, to) = if prior {
            (aligned, chunk_start)
        } else {
            (chunk_start, end.div_ceil(HEAD_WIDTH) * HEAD_WIDTH)
        };
        if from < to {
            gpu.one(
                "ink_prefill_relative_gather",
                &[&visit.keys, &visit.values, &self.keys, &self.values],
                [aligned as i32, from as i32, first as i32, end as i32],
                (KV / 4 / 128, to - from),
                128,
            );
        }
    }

    pub(super) fn pack(
        &self,
        gpu: &mut Stream<'_>,
        scratch: &Scratch,
        row: usize,
    ) {
        for (source, dest, width) in [
            (&scratch.q, &self.query, QUERY_WIDTH),
            (&scratch.r, &self.relative, HEADS * 16),
            (&scratch.sink, &self.sink, HEADS),
            (&scratch.attn_gate, &self.gate, QUERY_WIDTH),
        ] {
            Self::copy(gpu, binding(source), (dest, row * width * 2), width);
        }
    }

    pub(super) fn encode(
        &self,
        gpu: &mut Stream<'_>,
        weight: &Buffer,
        chunk_start: usize,
        rows: usize,
    ) {
        let first = chunk_start.saturating_sub(LOCAL - 1);
        gpu.one(
            "ink_prefill_relative_table_source",
            &[&self.relative, weight, &self.table],
            (rows * HEADS) as u32,
            (LOCAL / 32, (rows * HEADS).div_ceil(16)),
            32,
        );
        gpu.one(
            "ink_prefill_relative_source",
            &[&self.query, &self.keys, &self.values, &self.table, &self.sink, &self.gate, &self.attended],
            [chunk_start as i32, rows as i32, first as i32, (chunk_start + rows - first) as i32],
            (rows.div_ceil(16), HEADS),
            128,
        );
    }

    pub(super) fn unpack(
        &self,
        gpu: &mut Stream<'_>,
        scratch: &Scratch,
        row: usize,
        trace_table: bool,
    ) {
        Self::copy(gpu, (&self.attended, row * QUERY_WIDTH * 2), binding(&scratch.attended), QUERY_WIDTH);
        if trace_table {
            Self::copy(gpu, (&self.table, row * HEADS * LOCAL * 2), binding(&scratch.table), HEADS * LOCAL);
        }
    }

    fn copy(
        gpu: &mut Stream<'_>,
        input: Binding<'_>,
        output: Binding<'_>,
        bfloat_elements: usize,
    ) {
        let words = bfloat_elements / 2;
        gpu.dispatch("ink_trace_copy", &[input, output], words as u32, (words.div_ceil(128), 1), 128);
    }
}
