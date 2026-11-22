// SPDX-License-Identifier: Apache-2.0
//! Original Inkling 2 Mini DSpark, with one target-verified proposal.
//!
//! Graph and rounding reference: Ollama 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8,
//! mlxrunner/model/inkling2/draft.go and mlxrunner/dflash.go. The bidirectional
//! block always has one anchor and seven masks, even for one proposal.

use std::fs::File;

use half::bf16;
use metal::MTLBlitCommandEncoder;

use super::{session::PADDED_VOCABULARY, *};

const ROWS: usize = 8;
const TILE_ROWS: usize = 16;
const HEAD_DIM: usize = 128;
const ATTENTION: usize = HEADS * HEAD_DIM;
const FF: usize = 9_216;
const RANK: usize = 256;
const RNN: usize = 2 * (RANK + WIDTH);
const STORAGE_VOCABULARY: usize = 201_024;
const SHARDS: usize = 32;
const GROUPS: usize = VOCABULARY.div_ceil(4);

fn private(
    device: &Device,
    bytes: usize,
) -> Result<Buffer> {
    device.new_buffer(bytes, MTLResourceOptions::STORAGE_MODE_PRIVATE).ok_or("Inkling draft allocation failed")
}

struct Int8Head {
    weight: Buffer,
    scales: Buffer,
}

impl Int8Head {
    const BYTES: usize = PADDED_VOCABULARY * (WIDTH + size_of::<f32>());
    const SHAPE: [u32; 2] = [WIDTH as u32, PADDED_VOCABULARY as u32];

    fn load(
        runtime: &mut Runtime,
        unembedding: &Buffer,
    ) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = runtime
            .device
            .new_library_with_source(include_str!("draft_int8_head.metal"), Some(&options))
            .map_err(|_| "Inkling INT8 draft head Metal library compilation failed")?;
        for name in ["ink_draft_pack_int8_head", "ink_draft_int8_head", "ink_draft_scale_int8_head"] {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
        let weight = private(&runtime.device, PADDED_VOCABULARY * WIDTH)?;
        let scales = private(&runtime.device, PADDED_VOCABULARY * size_of::<f32>())?;
        execute(runtime, |gpu| {
            gpu.one(
                "ink_draft_pack_int8_head",
                &[unembedding, &weight, &scales],
                Self::SHAPE,
                (PADDED_VOCABULARY, 1),
                128,
            );
        })?;
        Ok(Self {
            weight,
            scales,
        })
    }

    fn encode(
        &self,
        gpu: &mut Stream<'_>,
        scratch: &Scratch,
        count: usize,
    ) {
        gpu.one(
            "ink_draft_int8_head",
            &[&scratch.norm, &self.weight, &scratch.raw],
            Self::SHAPE,
            (PADDED_VOCABULARY / 32, 1),
            32,
        );
        gpu.one(
            "ink_draft_scale_int8_head",
            &[&scratch.raw, &self.scales],
            [PADDED_VOCABULARY as u32, count as u32],
            (PADDED_VOCABULARY.div_ceil(128), count),
            128,
        );
    }
}

struct DraftLayer {
    input_norm: Buffer,
    ff_norm: Buffer,
    context_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    context_k_norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    context_k: Buffer,
    context_v: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    local: bool,
}

impl DraftLayer {
    fn load(
        source: &Tensors,
        device: &Device,
        index: usize,
    ) -> Result<Self> {
        let tensor =
            |name: &str, shape: &[usize]| source.buffer(device, &format!("layers.{index}.{name}"), "BF16", shape);
        Ok(Self {
            input_norm: tensor("input_layernorm.weight", &[WIDTH])?,
            ff_norm: tensor("post_attention_layernorm.weight", &[WIDTH])?,
            context_norm: tensor("dflare_norm.weight", &[WIDTH])?,
            q_norm: tensor("self_attn.q_norm.weight", &[HEAD_DIM])?,
            k_norm: tensor("self_attn.k_norm.weight", &[HEAD_DIM])?,
            context_k_norm: tensor("self_attn.k_norm_ctx.weight", &[HEAD_DIM])?,
            q: tensor("self_attn.q_proj.weight", &[ATTENTION, WIDTH])?,
            k: tensor("self_attn.k_proj.weight", &[KV, WIDTH])?,
            v: tensor("self_attn.v_proj.weight", &[KV, WIDTH])?,
            context_k: tensor("self_attn.k_proj_ctx.weight", &[KV, WIDTH])?,
            context_v: tensor("self_attn.v_proj_ctx.weight", &[KV, WIDTH])?,
            o: tensor("self_attn.o_proj.weight", &[WIDTH, ATTENTION])?,
            gate: tensor("mlp.gate_proj.weight", &[FF, WIDTH])?,
            up: tensor("mlp.up_proj.weight", &[FF, WIDTH])?,
            down: tensor("mlp.down_proj.weight", &[WIDTH, FF])?,
            local: index != 1,
        })
    }
}

struct Cache {
    keys: Buffer,
    values: Buffer,
    capacity: usize,
}

impl Cache {
    fn new(
        device: &Device,
        capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            keys: private(device, capacity * KV * 2)?,
            values: private(device, capacity * KV * 2)?,
            capacity,
        })
    }
}

struct Scratch {
    anchor: Buffer,
    residual: Buffer,
    norm: Buffer,
    context_mixed: Buffer,
    raw: Buffer,
    q: Buffer,
    temp: Buffer,
    k: Buffer,
    v: Buffer,
    attended: Buffer,
    parts: Buffer,
    gate: Buffer,
    up: Buffer,
    activated: Buffer,
    rnn_input: Buffer,
    latent: Buffer,
    groups: Buffer,
    winner: Buffer,
}

impl Scratch {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            anchor: private(device, WIDTH * 4)?,
            residual: private(device, ROWS * WIDTH * 4)?,
            norm: zeroed(device, TILE_ROWS * WIDTH * 2)?,
            context_mixed: private(device, WIDTH * 4)?,
            raw: private(device, TILE_ROWS * PADDED_VOCABULARY * 4)?,
            q: private(device, ROWS * ATTENTION * 2)?,
            temp: private(device, ROWS * ATTENTION * 2)?,
            k: private(device, ROWS * KV * 2)?,
            v: private(device, ROWS * KV * 2)?,
            attended: zeroed(device, TILE_ROWS * ATTENTION * 2)?,
            parts: private(device, ROWS * HEADS * SHARDS * (HEAD_DIM + 2) * 4)?,
            gate: private(device, ROWS * FF * 2)?,
            up: private(device, ROWS * FF * 2)?,
            activated: zeroed(device, TILE_ROWS * FF * 2)?,
            rnn_input: zeroed(device, TILE_ROWS * RNN * 2)?,
            latent: private(device, RANK * 2)?,
            groups: private(device, GROUPS * 8)?,
            winner: zeroed(device, size_of::<u32>())?,
        })
    }

    fn matmul(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        weight: &Buffer,
        cols: usize,
        outputs: usize,
    ) {
        gpu.one(
            "ink_unembed_tensor",
            &[input, weight, &self.raw],
            [cols as u32, outputs as u32],
            (outputs / 32, 1),
            32,
        );
    }

    fn project(
        &self,
        gpu: &mut Stream<'_>,
        input: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        shape: (usize, usize),
    ) {
        self.matmul(gpu, input, weight, shape.0, shape.1);
        round(gpu, &self.raw, out, ROWS * shape.1);
    }
}

pub(super) struct DraftBlock {
    state: Buffer,
    gates: Buffer,
    winners: Buffer,
}

impl DraftBlock {
    pub(super) fn winners(&self) -> &Buffer {
        &self.winners
    }

    pub(super) fn new(runtime: &mut Runtime) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.set_math_mode(MTLMathMode::Safe);
        options.set_language_version(MTLLanguageVersion::Version4_0);
        let library = runtime
            .device
            .new_library_with_source(
                concat!("#define INK_DRAFT_MARKOV_BLOCK\n", include_str!("dspark.metal")),
                Some(&options),
            )
            .map_err(|_| "Inkling draft block diagnostic library compilation failed")?;
        for name in ["ink_draft_markov_input", "ink_draft_markov_state"] {
            let function = library.new_function_with_name(name).ok_or(name)?;
            let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
            runtime.pipelines.insert(name, pipeline);
        }
        let device = &runtime.device;
        Ok(Self {
            state: private(device, RANK * 2)?,
            gates: private(device, TILE_ROWS * 3 * RANK * 4)?,
            winners: zeroed(device, (ROWS - 1) * size_of::<u32>())?,
        })
    }
}

pub(super) struct Dspark {
    layers: [DraftLayer; 3],
    cache: [Cache; 3],
    fuse: Buffer,
    mask: Buffer,
    norm: Buffer,
    feature_norm: Buffer,
    feature: Buffer,
    previous: Buffer,
    recurrent: Buffer,
    recurrent_bias: Buffer,
    folded: Buffer,
    folded_bias: Buffer,
    int8_head: Option<Int8Head>,
    scratch: Scratch,
}

impl Dspark {
    pub(super) const INT8_HEAD_BYTES: usize = Int8Head::BYTES;

    pub(super) fn prepare_int8_head(
        &mut self,
        runtime: &mut Runtime,
        unembedding: &Buffer,
    ) -> Result<()> {
        if self.int8_head.is_none() {
            self.int8_head = Some(Int8Head::load(runtime, unembedding)?);
        }
        Ok(())
    }

    pub(super) fn int8_head_resident_bytes(&self) -> usize {
        self.int8_head.as_ref().map_or(0, |_| Self::INT8_HEAD_BYTES)
    }

    pub(super) fn load(
        runtime: &mut Runtime,
        checkpoint: File,
        unembedding: &Buffer,
        context: usize,
    ) -> Result<Self> {
        if !runtime.pinned_m5 {
            return Err("Inkling DSpark requires the qualified Metal tensor kernel");
        }
        register(runtime)?;
        let device = &runtime.device;
        let source = Tensors::from_file(checkpoint)?;
        let tensor = |name, shape: &[usize]| source.buffer(device, name, "BF16", shape);
        let fuse = zeroed(device, 4 * 4 * 2)?;
        let values = source.values::<u16>("fuse_weight", "BF16", &[4, 4])?;
        for (row, values) in values.chunks_exact(4).enumerate() {
            let logits: [f32; 4] = std::array::from_fn(|index| bf16::from_bits(values[index]).to_f32());
            if logits.iter().any(|x| !x.is_finite()) {
                return Err("Inkling draft fusion contains a nonfinite value");
            }
            let peak = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let exponentials = logits.map(|logit| (logit - peak).exp());
            let total: f32 = exponentials.into_iter().sum();
            for (column, value) in exponentials.into_iter().enumerate() {
                unsafe {
                    fuse.contents()
                        .as_ptr()
                        .cast::<u16>()
                        .add(row * 4 + column)
                        .write(bf16::from_f32(value / total).to_bits());
                }
            }
        }
        let latent = tensor("markov_head.hidden_out.weight", &[WIDTH, RANK])?;
        let latent_bias = tensor("markov_head.hidden_out.bias", &[WIDTH])?;
        let folded = private(device, PADDED_VOCABULARY * RANK * 2)?;
        let folded_bias = private(device, PADDED_VOCABULARY * 2)?;
        {
            // Load-time work only. Preserve the two target-weight products
            // before BF16 rounding them, as the source Markov head does.
            let raw = private(device, PADDED_VOCABULARY * RANK * 4)?;
            let transposed = private(device, WIDTH * RANK * 2)?;
            execute(runtime, |gpu| {
                gpu.one("ink_draft_fold_input", &[&latent, &transposed], (), (WIDTH * RANK / 128, 1), 128);
                gpu.one(
                    "ink_draft_fold",
                    &[&transposed, unembedding, &raw],
                    PADDED_VOCABULARY as u32,
                    (PADDED_VOCABULARY / 32, RANK / 16),
                    32,
                );
                gpu.one(
                    "ink_draft_fold_output",
                    &[&raw, &folded],
                    PADDED_VOCABULARY as u32,
                    (PADDED_VOCABULARY / 32, RANK / 32),
                    256,
                );
                gpu.one(
                    "ink_draft_fold_bias",
                    &[&latent_bias, unembedding, &folded_bias],
                    (),
                    (PADDED_VOCABULARY / 4, 1),
                    128,
                );
            })?;
        }
        Ok(Self {
            layers: [
                DraftLayer::load(&source, device, 0)?,
                DraftLayer::load(&source, device, 1)?,
                DraftLayer::load(&source, device, 2)?,
            ],
            cache: Self::new_cache(device, context)?,
            fuse,
            mask: tensor("mask_embed", &[WIDTH])?,
            norm: tensor("norm.weight", &[WIDTH])?,
            feature_norm: tensor("markov_head.dflare_norm.weight", &[WIDTH])?,
            feature: private(device, WIDTH * 2)?,
            previous: tensor("markov_head.markov_w1.weight", &[STORAGE_VOCABULARY, RANK])?,
            recurrent: tensor("markov_head.rnn_proj.weight", &[3 * RANK, RNN])?,
            recurrent_bias: tensor("markov_head.rnn_proj.bias", &[3 * RANK])?,
            folded,
            folded_bias,
            int8_head: None,
            scratch: Scratch::new(device)?,
        })
    }

    fn new_cache(
        device: &Device,
        context: usize,
    ) -> Result<[Cache; 3]> {
        Ok([
            Cache::new(device, context.min(LOCAL))?,
            Cache::new(device, context)?,
            Cache::new(device, context.min(LOCAL))?,
        ])
    }

    pub(super) fn set_context(
        &mut self,
        device: &Device,
        context: usize,
    ) -> Result<()> {
        self.cache = Self::new_cache(device, context)?;
        Ok(())
    }

    pub(super) fn state_bytes(position: usize) -> usize {
        (2 * position.min(LOCAL) + position) * 2 * KV * 2 + WIDTH * 2
    }

    pub(super) fn probe(
        &self,
        runtime: &Runtime,
        token: usize,
    ) -> Result<[Vec<u16>; 5]> {
        let buffers = [
            (&self.scratch.norm, 0, WIDTH),
            (&self.scratch.latent, 0, RANK),
            (&self.feature, 0, WIDTH),
            (&self.folded, token * RANK, RANK),
            (&self.folded_bias, token, 1),
        ];
        let elements = buffers.iter().map(|(_, _, length)| length).sum::<usize>();
        let dest = zeroed(&runtime.device, elements * 2)?;
        let command = runtime.queue.command_buffer().ok_or("Inkling draft probe command allocation failed")?;
        let encoder = command.blit_command_encoder().ok_or("Inkling draft probe encoder allocation failed")?;
        let mut offset = 0;
        for (buffer, start, length) in buffers {
            encoder.copy_buffer_to_buffer(buffer, start * 2, &dest, offset * 2, length * 2);
            offset += length;
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("Inkling draft probe command failed");
        }
        let words = unsafe { std::slice::from_raw_parts(dest.contents().as_ptr().cast::<u16>(), elements) };
        let mut offset = 0;
        Ok(buffers.map(|(_, _, length)| {
            let result = words[offset..offset + length].to_vec();
            offset += length;
            result
        }))
    }

    pub(super) fn state_buffers(
        &self,
        position: usize,
    ) -> impl Iterator<Item = (&Buffer, usize)> {
        self.cache
            .iter()
            .flat_map(move |cache| {
                let bytes = position.min(cache.capacity) * KV * 2;
                [(&cache.keys, bytes), (&cache.values, bytes)]
            })
            .chain(std::iter::once((&self.feature, WIDTH * 2)))
    }

    fn mix(
        &self,
        gpu: &mut Stream<'_>,
        taps: Binding<'_>,
        index: usize,
        norm: &Buffer,
        out: &Buffer,
    ) {
        let mixed = &self.scratch.context_mixed;
        gpu.dispatch(
            "ink_draft_mix",
            &[taps, binding(&self.fuse), binding(mixed)],
            index as u32,
            (WIDTH / 128, 1),
            128,
        );
        gpu.norm(binding(mixed), norm, binding(out), true);
    }

    pub(super) fn commit(
        &self,
        gpu: &mut Stream<'_>,
        taps: &Buffer,
        position: usize,
    ) {
        self.commit_from(gpu, binding(taps), position);
    }

    pub(super) fn commit_from(
        &self,
        gpu: &mut Stream<'_>,
        taps: Binding<'_>,
        position: usize,
    ) {
        let s = &self.scratch;
        for (index, (layer, cache)) in self.layers.iter().zip(&self.cache).enumerate() {
            self.mix(gpu, taps, index, &layer.context_norm, &s.norm);
            gpu.project(binding(&s.norm), &layer.context_k, &s.temp, WIDTH, KV);
            rope(gpu, &s.temp, &layer.context_k_norm, &cache.keys, KV, position, cache.capacity, 1);
            gpu.project(binding(&s.norm), &layer.context_v, &s.temp, WIDTH, KV);
            gpu.one(
                "ink_draft_copy_value",
                &[&s.temp, &cache.values],
                (position % cache.capacity) as u32,
                (KV / 128, 1),
                128,
            );
        }
        self.mix(gpu, taps, self.layers.len(), &self.feature_norm, &self.feature);
    }

    fn rank_geometry(runtime: &Runtime) -> (Kernel, usize, usize) {
        if runtime.source_text_active && runtime.draft_shortlist {
            let groups = VOCABULARY.div_ceil(128);
            ("ink_draft_shortlist", groups, groups * 4)
        } else {
            ("ink_draft_rank", GROUPS, GROUPS)
        }
    }

    pub(super) fn propose(
        &self,
        runtime: &Runtime,
        anchor: u32,
        position: usize,
        embedding: &Buffer,
        embed_norm: &Buffer,
        unembedding: &Buffer,
    ) -> Result<u32> {
        let s = &self.scratch;
        let (rank, dispatch_groups, winner_groups) = Self::rank_geometry(runtime);
        execute(runtime, |gpu| {
            self.encode_block(gpu, anchor, position, embedding, embed_norm);
            // Row 0 is the anchor. The first mask row supplies the one
            // proposal, while all eight rows participated in attention.
            gpu.norm((&s.residual, WIDTH * 4), &self.norm, binding(&s.norm), true);
            gpu.one(
                "ink_draft_rnn_input",
                &[&self.previous, &s.norm, &self.feature, &s.rnn_input],
                anchor,
                (RNN / 128, 1),
                128,
            );
            s.matmul(gpu, &s.rnn_input, &self.recurrent, RNN, 3 * RANK);
            gpu.one("ink_draft_rnn_latent", &[&s.raw, &self.recurrent_bias, &s.latent], (), (RANK / 128, 1), 128);
            s.matmul(gpu, &s.norm, unembedding, WIDTH, PADDED_VOCABULARY);
            gpu.one(
                rank,
                &[&s.raw, &self.folded, &self.folded_bias, &s.latent, &s.groups],
                VOCABULARY as u32,
                (dispatch_groups, 1),
                128,
            );
            gpu.one("ink_draft_winner", &[&s.groups, &s.winner], winner_groups as u32, (1, 1), 256);
        })?;
        let winner = unsafe { s.winner.contents().as_ptr().cast::<u32>().read() };
        if winner as usize >= VOCABULARY {
            return Err("Inkling draft did not produce a finite vocabulary score");
        }
        Ok(winner)
    }

    pub(super) fn propose_many(
        &self,
        runtime: &Runtime,
        anchor: u32,
        position: usize,
        embedding: &Buffer,
        embed_norm: &Buffer,
        unembedding: &Buffer,
        count: usize,
        block: &DraftBlock,
    ) -> Result<Vec<u32>> {
        self.propose_many_with_submit(
            runtime,
            anchor,
            position,
            embedding,
            embed_norm,
            unembedding,
            count,
            block,
            || Ok(()),
        )
        .map(|(winners, _)| winners)
    }

    pub(super) fn propose_many_with_submit(
        &self,
        runtime: &Runtime,
        anchor: u32,
        position: usize,
        embedding: &Buffer,
        embed_norm: &Buffer,
        unembedding: &Buffer,
        count: usize,
        block: &DraftBlock,
        on_submit: impl FnOnce() -> Result<()>,
    ) -> Result<(Vec<u32>, Timing)> {
        if !(1..ROWS).contains(&count) || anchor as usize >= VOCABULARY {
            return Err("Inkling draft proposal block is out of range");
        }
        let timing = self.execute_proposal_block(
            runtime,
            anchor,
            position,
            embedding,
            embed_norm,
            unembedding,
            count,
            block,
            None,
            on_submit,
        )?;
        let winners = session::read_shared::<u32>(&block.winners, count);
        if winners.iter().any(|&token| token as usize >= VOCABULARY) {
            return Err("Inkling draft did not produce a finite vocabulary score");
        }
        Ok((winners, timing))
    }

    pub(super) fn teacher_proposal_logits(
        &self,
        runtime: &Runtime,
        anchor: u32,
        position: usize,
        embedding: &Buffer,
        embed_norm: &Buffer,
        unembedding: &Buffer,
        previous: &[u32],
        block: &DraftBlock,
        scores: &Buffer,
    ) -> Result<()> {
        debug_assert!(previous.len() < ROWS - 1);
        // This diagnostic owns the scratch. Metal commands complete before reuse.
        unsafe {
            std::ptr::copy_nonoverlapping(
                previous.as_ptr(),
                block.winners.contents().as_ptr().cast::<u32>(),
                previous.len(),
            );
        }
        self.execute_proposal_block(
            runtime,
            anchor,
            position,
            embedding,
            embed_norm,
            unembedding,
            previous.len() + 1,
            block,
            Some(scores),
            || Ok(()),
        )
        .map(|_| ())
    }

    fn execute_proposal_block(
        &self,
        runtime: &Runtime,
        anchor: u32,
        position: usize,
        embedding: &Buffer,
        embed_norm: &Buffer,
        unembedding: &Buffer,
        count: usize,
        block: &DraftBlock,
        scores: Option<&Buffer>,
        on_submit: impl FnOnce() -> Result<()>,
    ) -> Result<Timing> {
        let s = &self.scratch;
        let (rank, dispatch_groups, winner_groups) = Self::rank_geometry(runtime);
        execute_prepared_with_submit(
            runtime,
            None,
            |_| Ok(()),
            |gpu| {
                self.encode_block(gpu, anchor, position, embedding, embed_norm);
                for row in 0..count {
                    gpu.norm((&s.residual, (row + 1) * WIDTH * 4), &self.norm, (&s.norm, row * WIDTH * 2), true);
                }
                // The source's padded vocabulary tile projects all live mask rows
                // together. A separate, small recurrent output preserves those scores.
                if let Some(head) = self.int8_head.as_ref().filter(|_| scores.is_none()) {
                    head.encode(gpu, s, count);
                } else {
                    s.matmul(gpu, &s.norm, unembedding, WIDTH, PADDED_VOCABULARY);
                }
                for row in 0..count {
                    gpu.dispatch(
                        "ink_draft_markov_input",
                        &[
                            binding(&self.previous),
                            (&s.norm, row * WIDTH * 2),
                            binding(&self.feature),
                            binding(&s.rnn_input),
                            binding(&block.state),
                            binding(&block.winners),
                        ],
                        [row as u32, anchor, VOCABULARY as u32, 0],
                        (RNN / 128, 1),
                        128,
                    );
                    gpu.one(
                        "ink_unembed_tensor",
                        &[&s.rnn_input, &self.recurrent, &block.gates],
                        [RNN as u32, (3 * RANK) as u32],
                        (3 * RANK / 32, 1),
                        32,
                    );
                    gpu.one(
                        "ink_draft_markov_state",
                        &[&block.gates, &self.recurrent_bias, &s.latent, &block.state],
                        row as u32,
                        (RANK / 128, 1),
                        128,
                    );
                    let (pipeline, output, groups) = match scores {
                        Some(scores) => ("ink_draft_export_scores", (scores, row * PADDED_VOCABULARY * 4), GROUPS),
                        None => (rank, binding(&s.groups), dispatch_groups),
                    };
                    gpu.dispatch(
                        pipeline,
                        &[
                            (&s.raw, row * PADDED_VOCABULARY * 4),
                            binding(&self.folded),
                            binding(&self.folded_bias),
                            binding(&s.latent),
                            output,
                        ],
                        VOCABULARY as u32,
                        (groups, 1),
                        128,
                    );
                    if scores.is_none() {
                        gpu.dispatch(
                            "ink_draft_winner",
                            &[binding(&s.groups), (&block.winners, row * size_of::<u32>())],
                            winner_groups as u32,
                            (1, 1),
                            256,
                        );
                    }
                }
            },
            on_submit,
        )
    }

    fn encode_block(
        &self,
        gpu: &mut Stream<'_>,
        anchor: u32,
        position: usize,
        embedding: &Buffer,
        embed_norm: &Buffer,
    ) {
        let s = &self.scratch;
        gpu.one("ink_embedding", &[embedding, embed_norm, &s.anchor], anchor, (1, 1), WIDTH / 4);
        gpu.one("ink_draft_block", &[&s.anchor, &self.mask, &s.residual], (), (WIDTH / 128, ROWS), 128);
        for (layer, cache) in self.layers.iter().zip(&self.cache) {
            self.decode_layer(gpu, layer, cache, position);
        }
    }

    fn decode_layer(
        &self,
        gpu: &mut Stream<'_>,
        layer: &DraftLayer,
        cache: &Cache,
        position: usize,
    ) {
        let s = &self.scratch;
        normalize_rows(gpu, &s.residual, &layer.input_norm, &s.norm);
        s.project(gpu, &s.norm, &layer.q, &s.temp, (WIDTH, ATTENTION));
        rope(gpu, &s.temp, &layer.q_norm, &s.q, ATTENTION, position, 0, ROWS);
        s.project(gpu, &s.norm, &layer.k, &s.temp, (WIDTH, KV));
        rope(gpu, &s.temp, &layer.k_norm, &s.k, KV, position, 0, ROWS);
        s.project(gpu, &s.norm, &layer.v, &s.v, (WIDTH, KV));
        let start = if layer.local {
            position.saturating_sub(LOCAL)
        } else {
            0
        };
        gpu.one(
            "ink_draft_attend_parts",
            &[&s.q, &cache.keys, &cache.values, &s.k, &s.v, &s.parts],
            [start as u32, position as u32, cache.capacity as u32, 0],
            (SHARDS, ROWS * HEADS),
            32,
        );
        gpu.one("ink_draft_attend_join", &[&s.parts, &s.attended], (), (ROWS * HEADS, 1), HEAD_DIM);
        s.project(gpu, &s.attended, &layer.o, &s.temp, (ATTENTION, WIDTH));
        add_residual(gpu, &s.residual, &s.temp);
        normalize_rows(gpu, &s.residual, &layer.ff_norm, &s.norm);
        s.project(gpu, &s.norm, &layer.gate, &s.gate, (WIDTH, FF));
        s.project(gpu, &s.norm, &layer.up, &s.up, (WIDTH, FF));
        gpu.one("ink_draft_swiglu", &[&s.gate, &s.up, &s.activated], (), (ROWS * FF / 128, 1), 128);
        s.project(gpu, &s.activated, &layer.down, &s.temp, (FF, WIDTH));
        add_residual(gpu, &s.residual, &s.temp);
    }
}

fn register(runtime: &mut Runtime) -> Result<()> {
    let names = [
        "ink_draft_mix",
        "ink_draft_rope",
        "ink_draft_copy_value",
        "ink_draft_block",
        "ink_draft_round",
        "ink_draft_fold_input",
        "ink_draft_fold",
        "ink_draft_fold_output",
        "ink_draft_fold_bias",
        "ink_draft_attend_parts",
        "ink_draft_attend_join",
        "ink_draft_residual",
        "ink_draft_swiglu",
        "ink_draft_rnn_input",
        "ink_draft_rnn_latent",
        "ink_draft_rank",
        "ink_draft_winner",
    ];
    if runtime.pipelines.contains_key(names[0]) {
        return Ok(());
    }
    let options = MTLCompileOptions::new();
    options.set_math_mode(MTLMathMode::Safe);
    options.set_language_version(MTLLanguageVersion::Version4_0);
    let library = runtime
        .device
        .new_library_with_source(include_str!("dspark.metal"), Some(&options))
        .map_err(|_| "Inkling draft Metal library compilation failed")?;
    let mut pipelines = Vec::with_capacity(names.len());
    for name in names {
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        pipelines.push((name, pipeline));
    }
    if std::env::var_os("STARK_INKLING_EXPERIMENT_DRAFT_SHORTLIST").as_deref() == Some(std::ffi::OsStr::new("1")) {
        let library = runtime
            .device
            .new_library_with_source(
                concat!("#define INK_DRAFT_SHORTLIST\n", include_str!("dspark.metal")),
                Some(&options),
            )
            .map_err(|_| "Inkling draft shortlist library compilation failed")?;
        let name = "ink_draft_shortlist";
        let function = library.new_function_with_name(name).ok_or(name)?;
        let pipeline = runtime.device.new_compute_pipeline_state_with_function(&function).map_err(|_| name)?;
        pipelines.push((name, pipeline));
        runtime.draft_shortlist = true;
    }
    runtime.pipelines.extend(pipelines);
    Ok(())
}

fn round(
    gpu: &mut Stream<'_>,
    raw: &Buffer,
    out: &Buffer,
    elements: usize,
) {
    gpu.one("ink_draft_round", &[raw, out], elements as u32, (elements.div_ceil(128), 1), 128);
}

fn normalize_rows(
    gpu: &mut Stream<'_>,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
) {
    for row in 0..ROWS {
        gpu.norm((input, row * WIDTH * 4), weight, (output, row * WIDTH * 2), true);
    }
}

fn add_residual(
    gpu: &mut Stream<'_>,
    residual: &Buffer,
    update: &Buffer,
) {
    gpu.one("ink_draft_residual", &[residual, update], (), (ROWS * WIDTH / 128, 1), 128);
}

fn rope(
    gpu: &mut Stream<'_>,
    input: &Buffer,
    weights: &Buffer,
    out: &Buffer,
    features: usize,
    position: usize,
    capacity: usize,
    rows: usize,
) {
    gpu.one(
        "ink_draft_rope",
        &[input, weights, out],
        [features as u32, position as u32, capacity as u32, 0],
        (features / HEAD_DIM, rows),
        32,
    );
}
