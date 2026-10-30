//! Qwen3.8 hashed per-layer embeddings on the standard Uzu backend.
//!
//! Hash and convolution semantics follow Qwen4Exp, with algorithm provenance
//! recorded in docs/QWEN38_UZU_PROVENANCE.md. All projections use LinearMatmul.

use super::{
    Qwen38Error,
    weights::{PagedMlxEmbedding, Qwen38WeightLoader},
};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, InvalidInput, Kernels,
        kernel::{
            Qwen38HcNormKernel, Qwen38PleConvKernel, Qwen38PleGateKernel, Qwen38PleGatherKernel, matmul::MatmulA,
        },
    },
    config::model::qwen38_language_model::Qwen38ArchitectureConfig,
    data_type::DataType,
    encodable_block::linear::{Gather, LinearMatmul},
};

pub struct Ple<B: Backend> {
    key: LinearMatmul<B>,
    value: LinearMatmul<B>,
    query_norm: B::GlobalBuffer,
    key_norm: B::GlobalBuffer,
    conv_norm: B::GlobalBuffer,
    conv_weight: B::GlobalBuffer,
    shards: Vec<PagedMlxEmbedding>,
    multipliers: [u64; 3],
    sizes: Box<[u64]>,
    offsets: Box<[u64]>,
    eos: u32,
    heads_per_ngram: u32,
    width: u32,
    streams: u32,
    embedding_width: u32,
    head_width: u32,
    group_size: u32,
    kernel_size: u32,
    dilation: u32,
    epsilon: f32,
    norm: <B::Kernels as Kernels>::Qwen38HcNormKernel,
    gather: <B::Kernels as Kernels>::Qwen38PleGatherKernel,
    gate: <B::Kernels as Kernels>::Qwen38PleGateKernel,
    conv: <B::Kernels as Kernels>::Qwen38PleConvKernel,
}

pub struct PleState<B: Backend> {
    history: [u32; 2],
    window: B::GlobalBuffer,
}

impl<B: Backend> Ple<B> {
    pub fn new(
        context: &B::Context,
        cfg: &Qwen38ArchitectureConfig,
        layer: u32,
        loader: &Qwen38WeightLoader<'_, B>,
    ) -> Result<Self, Qwen38Error<B>> {
        let heads = cfg.heads_per_ngram.checked_mul(2).ok_or(Qwen38Error::Invalid("PLE head count overflow"))?;
        let embedding_width = cfg.ple_embed_dim.unwrap_or(cfg.hidden_size);
        let wide = cfg.hidden_size.checked_mul(cfg.hc_count).ok_or(Qwen38Error::Invalid("PLE width overflow"))?;
        let window = cfg
            .ple_conv_kernel_size
            .checked_sub(1)
            .and_then(|n| n.checked_mul(cfg.ngram_size))
            .ok_or(Qwen38Error::Invalid("PLE window overflow"))?;
        if cfg.ngram_size != 3
            || heads == 0
            || !embedding_width.is_multiple_of(heads)
            || cfg.split_ngram_parts == 0
            || !cfg.ple.padded_vocab_size.is_multiple_of(cfg.split_ngram_parts)
            || cfg.ple.layer_multipliers.len() != 3
            || cfg.ple.ngram_heads_vocab_sizes.len() != heads as usize
            || cfg.ple.ngram_heads_offsets.len() != heads as usize
            || window == 0
            || window > 16
            || cfg.eos_token_id.is_empty()
            || cfg.quantization.ngram_embedding.bits != 4
        {
            return Err(Qwen38Error::Invalid("unsupported PLE geometry"));
        }
        let head_width = embedding_width / heads;
        let group_size = cfg.quantization.ngram_embedding.group_size;
        if group_size == 0 || !head_width.is_multiple_of(group_size) || !head_width.is_multiple_of(2) {
            return Err(Qwen38Error::Invalid("invalid PLE quantized row geometry"));
        }
        for (&size, &offset) in cfg.ple.ngram_heads_vocab_sizes.iter().zip(&cfg.ple.ngram_heads_offsets) {
            if size == 0 || offset.checked_add(size).is_none_or(|end| end > cfg.ple.padded_vocab_size as u64) {
                return Err(Qwen38Error::Invalid("PLE hash range exceeds table"));
            }
        }
        let prefix = format!("model.language_model.layers.{layer}.ple");
        let rows = cfg.ple.padded_vocab_size / cfg.split_ngram_parts;
        let mut shards = Vec::with_capacity(cfg.split_ngram_parts as usize);
        for index in 0..cfg.split_ngram_parts {
            let stem = format!("{prefix}.ple_embedding.ngram_embedding.shard_{index}");
            let binding = loader.binding(&format!("{stem}.weight"))?;
            if binding.shape.first().copied() != Some(rows) {
                return Err(Qwen38Error::Invalid("PLE shard row binding mismatch"));
            }
            let shard = loader.paged_embedding(&stem, rows, head_width)?;
            if shard.rows != rows
                || shard.columns != head_width
                || shard.bits != 4
                || shard.group_size != group_size
                || shard.code_row_bytes != head_width as usize / 2
                || shard.parameter_row_bytes != (head_width / group_size) as usize * 2
            {
                return Err(Qwen38Error::Invalid("PLE shard staging geometry mismatch"));
            }
            shards.push(shard);
        }
        Ok(Self {
            key: loader.linear(&format!("{prefix}.key_proj"), embedding_width, wide, DataType::BF16, DataType::BF16)?,
            value: loader.linear(
                &format!("{prefix}.value_proj"),
                embedding_width,
                cfg.hidden_size,
                DataType::BF16,
                DataType::BF16,
            )?,
            query_norm: loader.control_f32(&format!("{prefix}.norm_query.weight"), &[wide])?,
            key_norm: loader.control_f32(&format!("{prefix}.norm_key.weight"), &[wide])?,
            conv_norm: loader.control_f32(&format!("{prefix}.norm_conv.weight"), &[wide])?,
            conv_weight: loader
                .control_f32(&format!("{prefix}.conv1d.weight"), &[wide, 1, cfg.ple_conv_kernel_size])?,
            shards,
            multipliers: [cfg.ple.layer_multipliers[0], cfg.ple.layer_multipliers[1], cfg.ple.layer_multipliers[2]],
            sizes: cfg.ple.ngram_heads_vocab_sizes.clone(),
            offsets: cfg.ple.ngram_heads_offsets.clone(),
            eos: cfg.eos_token_id[0],
            heads_per_ngram: cfg.heads_per_ngram,
            width: cfg.hidden_size,
            streams: cfg.hc_count,
            embedding_width,
            head_width,
            group_size,
            kernel_size: cfg.ple_conv_kernel_size,
            dilation: cfg.ngram_size,
            epsilon: cfg.rms_norm_eps,
            norm: <B::Kernels as Kernels>::Qwen38HcNormKernel::new(context).map_err(Qwen38Error::Backend)?,
            gather: <B::Kernels as Kernels>::Qwen38PleGatherKernel::new(context).map_err(Qwen38Error::Backend)?,
            gate: <B::Kernels as Kernels>::Qwen38PleGateKernel::new(context).map_err(Qwen38Error::Backend)?,
            conv: <B::Kernels as Kernels>::Qwen38PleConvKernel::new(context).map_err(Qwen38Error::Backend)?,
        })
    }

    pub fn create_state(
        &self,
        context: &B::Context,
    ) -> Result<PleState<B>, B::Error> {
        let values = (self.width as usize)
            .checked_mul(self.streams as usize)
            .and_then(|n| n.checked_mul((self.kernel_size - 1) as usize * self.dilation as usize))
            .and_then(|n| n.checked_mul(2))
            .ok_or(InvalidInput("PLE state size overflow"))?;
        let mut window = context.create_buffer(values)?;
        (&mut window).as_slice_mut::<u8>().fill(0);
        Ok(PleState {
            history: [self.eos; 2],
            window,
        })
    }

    /// Host IDs are supplied by the standard language-model runner before
    /// encoding; no unfinished GPU token buffer is read on the CPU.
    pub fn encode(
        &self,
        hyper: impl BufferRef<Backend = B>,
        tokens: &[u32],
        state: &mut PleState<B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let rows = u32::try_from(tokens.len()).map_err(|_| InvalidInput("PLE batch overflow"))?;
        let wide = self.width as usize * self.streams as usize;
        let expected = tokens
            .len()
            .checked_mul(wide)
            .and_then(|n| n.checked_mul(2))
            .ok_or(InvalidInput("PLE input size overflow"))?;
        if rows == 0 || hyper.size() != expected {
            return Err(InvalidInput("PLE requires a nonempty exact flat input").into());
        }
        let ids = self.hash_ids(tokens, state.history);
        let code_row = (self.head_width / 2) as usize;
        let parameter_row = (self.head_width / self.group_size) as usize * 2;
        let mut codes = vec![0u8; ids.len() * code_row];
        let mut scales = vec![0u8; ids.len() * parameter_row];
        let mut biases = vec![0u8; ids.len() * parameter_row];
        for (index, &id) in ids.iter().enumerate() {
            let mut base = 0u64;
            let mut found = false;
            for shard in &self.shards {
                let end = base + shard.rows as u64;
                if (id as u64) < end {
                    let row = id as u64 - base;
                    shard
                        .codes
                        .read_exact_into(row * code_row as u64, &mut codes[index * code_row..(index + 1) * code_row])
                        .map_err(|_| InvalidInput("PLE code row read failed"))?;
                    shard
                        .scales
                        .read_exact_into(
                            row * parameter_row as u64,
                            &mut scales[index * parameter_row..(index + 1) * parameter_row],
                        )
                        .map_err(|_| InvalidInput("PLE scale row read failed"))?;
                    shard
                        .biases
                        .read_exact_into(
                            row * parameter_row as u64,
                            &mut biases[index * parameter_row..(index + 1) * parameter_row],
                        )
                        .map_err(|_| InvalidInput("PLE bias row read failed"))?;
                    found = true;
                    break;
                }
                base = end;
            }
            if !found {
                return Err(InvalidInput("PLE row exceeds validated table").into());
            }
        }
        // Each allocation belongs to this command buffer and survives GPU completion.
        let codes = commands.allocate_constant_from_slice(&codes)?;
        let scales = commands.allocate_constant_from_slice(&scales)?;
        let biases = commands.allocate_constant_from_slice(&biases)?;
        let mut embedded = commands.allocate_scratch_for_shape(&[rows, self.embedding_width], DataType::BF16)?;
        self.gather.encode(
            &codes,
            &scales,
            &biases,
            &mut embedded,
            self.head_width,
            self.group_size,
            self.sizes.len() as u32,
            rows,
            commands,
        );
        let key = self.key.encode_with_a(
            MatmulA::FullPrecision {
                values: &embedded,
                offset: 0,
            },
            rows,
            None::<Gather<&B::ScratchBuffer>>,
            commands,
        )?;
        let value = self.value.encode_with_a(
            MatmulA::FullPrecision {
                values: &embedded,
                offset: 0,
            },
            rows,
            None::<Gather<&B::ScratchBuffer>>,
            commands,
        )?;
        let mut query = commands.allocate_scratch(expected)?;
        let mut normalized_key = commands.allocate_scratch(expected)?;
        for row in 0..rows as usize {
            let span = row * wide * 2..(row + 1) * wide * 2;
            self.norm.encode(
                hyper.subrange(span.clone()),
                &self.query_norm,
                (&mut query).subrange_mut(span.clone()),
                self.width,
                self.streams,
                1,
                self.epsilon,
                commands,
            );
            self.norm.encode(
                (&key).subrange(span.clone()),
                &self.key_norm,
                (&mut normalized_key).subrange_mut(span),
                self.width,
                self.streams,
                1,
                self.epsilon,
                commands,
            );
        }
        let mut gated = commands.allocate_scratch(expected)?;
        self.gate.encode(&normalized_key, &query, &value, &mut gated, self.width, self.streams, rows, commands);
        let mut normalized_gated = commands.allocate_scratch(expected)?;
        for row in 0..rows as usize {
            let span = row * wide * 2..(row + 1) * wide * 2;
            self.norm.encode(
                (&gated).subrange(span.clone()),
                &self.conv_norm,
                (&mut normalized_gated).subrange_mut(span),
                self.width,
                self.streams,
                1,
                self.epsilon,
                commands,
            );
        }
        let mut output = commands.allocate_scratch(expected)?;
        commands.encode_copy(hyper, &mut output);
        self.conv.encode(
            &mut state.window,
            &normalized_gated,
            &self.conv_weight,
            &gated,
            &mut output,
            wide as u32,
            self.kernel_size,
            self.dilation,
            rows,
            commands,
        );
        state.history = advance_history(state.history, tokens);
        Ok(output)
    }

    fn hash_ids(
        &self,
        tokens: &[u32],
        history: [u32; 2],
    ) -> Vec<u32> {
        hash_ids(tokens, history, self.eos, self.multipliers, &self.sizes, &self.offsets, self.heads_per_ngram)
    }
}

fn advance_history(
    history: [u32; 2],
    tokens: &[u32],
) -> [u32; 2] {
    match tokens {
        [] => history,
        [token] => [history[1], *token],
        [.., a, b] => [*a, *b],
    }
}

fn hash_ids(
    tokens: &[u32],
    history: [u32; 2],
    eos: u32,
    multipliers: [u64; 3],
    sizes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
) -> Vec<u32> {
    let mut ids = Vec::with_capacity(tokens.len() * sizes.len());
    for (row, &token) in tokens.iter().enumerate() {
        let p1 = if row == 0 {
            history[1]
        } else {
            tokens[row - 1]
        };
        let p2 = match row {
            0 => history[0],
            1 => history[1],
            _ => tokens[row - 2],
        };
        let p1 = if p1 == eos {
            eos
        } else {
            p1
        };
        let p2 = if p1 == eos || p2 == eos {
            eos
        } else {
            p2
        };
        let bigram = (token as u64).wrapping_mul(multipliers[0]) ^ (p1 as u64).wrapping_mul(multipliers[1]);
        let trigram = bigram ^ (p2 as u64).wrapping_mul(multipliers[2]);
        for (head, (&size, &offset)) in sizes.iter().zip(offsets).enumerate() {
            let mixed = if head / heads_per_ngram as usize == 0 {
                bigram
            } else {
                trigram
            };
            ids.push(((mixed % size) + offset) as u32);
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn hash_streaming_chunks_and_segment_boundaries_match() {
        let tokens = [7, 9, 2, 11, 13, 17, 2, 23];
        let sizes = [11, 13, 17, 19];
        let offsets = [0, 11, 24, 41];
        let multipliers = [u64::MAX - 2, 0x123456789abcdef, 0x7fffffffffffffed];
        let all = hash_ids(&tokens, [2, 2], 2, multipliers, &sizes, &offsets, 2);
        for split in 0..=tokens.len() {
            let mut chunked = hash_ids(&tokens[..split], [2, 2], 2, multipliers, &sizes, &offsets, 2);
            let history = advance_history([2, 2], &tokens[..split]);
            chunked.extend(hash_ids(&tokens[split..], history, 2, multipliers, &sizes, &offsets, 2));
            assert_eq!(chunked, all);
        }
        assert_eq!(advance_history([3, 5], &[]), [3, 5]);
        assert_eq!(advance_history([3, 5], &[7]), [5, 7]);
    }
}
