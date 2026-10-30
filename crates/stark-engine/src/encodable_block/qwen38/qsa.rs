//! Qwen3.8 compressed-indexer attention through standard Uzu projections,
//! backend buffers, generated kernels and command encoding.

use super::{Qwen38Error, weights::Qwen38WeightLoader};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, InvalidInput, Kernels,
        kernel::{
            Qwen38QsaAttentionKernel, Qwen38QsaBlockKeysKernel, Qwen38QsaCombineKernel, Qwen38QsaPrepareIndexKernel,
            Qwen38QsaPrepareKeyKernel, Qwen38QsaPrepareQueryKernel, Qwen38QsaScoresKernel, Qwen38QsaSelectKernel,
            Qwen38QsaStoreValueKernel, matmul::MatmulA,
        },
    },
    config::model::qwen38_language_model::Qwen38ArchitectureConfig,
    data_type::DataType,
    encodable_block::linear::{Gather, LinearMatmul},
};

const HEAD_DIM: u32 = 256;
const INDEX_DIM: u32 = 128;
const CACHE_GROUP: u32 = 32;
const ATTENTION_SPLIT: u32 = 64;
pub(super) const MAX_CONTEXT_TOKENS: u32 = 65_536;
// The shader compacts at most CACHE32 * THREADS512 keys per packed scan.
// Its two 16-bit counters are per chunk, not per complete score row. Context
// 64K at ratios 1/2/4 therefore uses up to four/two/one safe chunks.
const SELECT_CHUNK_BLOCKS: u32 = 32 * 512;
const MAX_VISIBLE_BLOCKS: u32 = 65_536;
const _: () = assert!(SELECT_CHUNK_BLOCKS < (1 << 16));

#[cfg(test)]
#[path = "qsa_boundary_tests.rs"]
mod boundary_tests;

pub(super) fn context_limit(
    maximum_positions: u32,
    compression_ratio: u32,
) -> u32 {
    maximum_positions.min(MAX_CONTEXT_TOKENS).min(MAX_VISIBLE_BLOCKS.saturating_mul(compression_ratio))
}

pub struct Qsa<B: Backend> {
    q: LinearMatmul<B>,
    k: LinearMatmul<B>,
    v: LinearMatmul<B>,
    output: LinearMatmul<B>,
    index: LinearMatmul<B>,
    q_norm: B::GlobalBuffer,
    k_norm: B::GlobalBuffer,
    index_q_norm: B::GlobalBuffer,
    index_k_norm: B::GlobalBuffer,
    hidden: u32,
    query_heads: u32,
    kv_heads: u32,
    index_heads: u32,
    ratio: u32,
    block_budget: u32,
    rotary: u32,
    theta: f32,
    epsilon: f32,
    max_context: u32,
    prepare_q: <B::Kernels as Kernels>::Qwen38QsaPrepareQueryKernel,
    prepare_k: <B::Kernels as Kernels>::Qwen38QsaPrepareKeyKernel,
    store_v: <B::Kernels as Kernels>::Qwen38QsaStoreValueKernel,
    prepare_index: <B::Kernels as Kernels>::Qwen38QsaPrepareIndexKernel,
    block_keys: <B::Kernels as Kernels>::Qwen38QsaBlockKeysKernel,
    scores: <B::Kernels as Kernels>::Qwen38QsaScoresKernel,
    select: <B::Kernels as Kernels>::Qwen38QsaSelectKernel,
    attention: <B::Kernels as Kernels>::Qwen38QsaAttentionKernel,
    combine: <B::Kernels as Kernels>::Qwen38QsaCombineKernel,
}

/// Request-owned Q8 K/V and BF16 raw/compressed indexer caches. Capacity is
/// physical storage; only the initialized prefix `length` may be attended.
pub struct QsaState<B: Backend> {
    capacity: u32,
    length: u32,
    keys: B::GlobalBuffer,
    values: B::GlobalBuffer,
    key_scales: B::GlobalBuffer,
    value_scales: B::GlobalBuffer,
    raw_index: B::GlobalBuffer,
    block_index: B::GlobalBuffer,
}

impl<B: Backend> QsaState<B> {
    pub fn length(&self) -> u32 {
        self.length
    }
}

impl<B: Backend> Qsa<B> {
    pub fn new(
        context: &B::Context,
        cfg: &Qwen38ArchitectureConfig,
        layer: u32,
        loader: &Qwen38WeightLoader<'_, B>,
    ) -> Result<Self, Qwen38Error<B>> {
        let rotary = (cfg.head_dim as f32 * cfg.rope_parameters.partial_rotary_factor) as u32;
        if cfg.head_dim != HEAD_DIM
            || cfg.indexer_head_dim != INDEX_DIM
            || cfg.indexer_kv_heads != 1
            || cfg.num_key_value_heads == 0
            || cfg.num_attention_heads == 0
            || !cfg.num_attention_heads.is_multiple_of(cfg.num_key_value_heads)
            || cfg.indexer_n_heads == 0
            || cfg.indexer_n_heads > 4
            || cfg.indexer_compress_ratio == 0
            || cfg.indexer_budget < cfg.indexer_compress_ratio
            || !cfg.indexer_budget.is_multiple_of(cfg.indexer_compress_ratio)
            || rotary == 0
            || rotary > INDEX_DIM
            || !rotary.is_multiple_of(2)
            || !cfg.rope_parameters.rope_theta.is_finite()
            || cfg.rope_parameters.rope_theta <= 0.0
        {
            return Err(Qwen38Error::Invalid("unsupported QSA geometry"));
        }
        let q_width =
            cfg.num_attention_heads.checked_mul(HEAD_DIM).ok_or(Qwen38Error::Invalid("QSA width overflow"))?;
        let q_gate_width = q_width.checked_mul(2).ok_or(Qwen38Error::Invalid("QSA gated width overflow"))?;
        let kv_width =
            cfg.num_key_value_heads.checked_mul(HEAD_DIM).ok_or(Qwen38Error::Invalid("QSA width overflow"))?;
        let index_width =
            (cfg.indexer_n_heads + 1).checked_mul(INDEX_DIM).ok_or(Qwen38Error::Invalid("QSA index width overflow"))?;
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let ix = format!("{prefix}.indexer");
        Ok(Self {
            q: loader.linear(
                &format!("{prefix}.q_proj"),
                cfg.hidden_size,
                q_gate_width,
                DataType::BF16,
                DataType::BF16,
            )?,
            k: loader.linear(&format!("{prefix}.k_proj"), cfg.hidden_size, kv_width, DataType::BF16, DataType::BF16)?,
            v: loader.linear(&format!("{prefix}.v_proj"), cfg.hidden_size, kv_width, DataType::BF16, DataType::BF16)?,
            output: loader.linear(
                &format!("{prefix}.o_proj"),
                q_width,
                cfg.hidden_size,
                DataType::BF16,
                DataType::BF16,
            )?,
            index: loader.linear(
                &format!("{ix}.index_qk_proj"),
                cfg.hidden_size,
                index_width,
                DataType::BF16,
                DataType::BF16,
            )?,
            q_norm: loader.control_f32(&format!("{prefix}.q_norm.weight"), &[HEAD_DIM])?,
            k_norm: loader.control_f32(&format!("{prefix}.k_norm.weight"), &[HEAD_DIM])?,
            index_q_norm: loader.control_f32(&format!("{ix}.q_layernorm.weight"), &[INDEX_DIM])?,
            index_k_norm: loader.control_f32(&format!("{ix}.k_layernorm.weight"), &[INDEX_DIM])?,
            hidden: cfg.hidden_size,
            query_heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            index_heads: cfg.indexer_n_heads,
            ratio: cfg.indexer_compress_ratio,
            block_budget: cfg.indexer_budget / cfg.indexer_compress_ratio,
            rotary,
            theta: cfg.rope_parameters.rope_theta,
            epsilon: cfg.rms_norm_eps,
            max_context: context_limit(cfg.max_position_embeddings, cfg.indexer_compress_ratio),
            prepare_q: <B::Kernels as Kernels>::Qwen38QsaPrepareQueryKernel::new(context)
                .map_err(Qwen38Error::Backend)?,
            prepare_k: <B::Kernels as Kernels>::Qwen38QsaPrepareKeyKernel::new(context)
                .map_err(Qwen38Error::Backend)?,
            store_v: <B::Kernels as Kernels>::Qwen38QsaStoreValueKernel::new(context).map_err(Qwen38Error::Backend)?,
            prepare_index: <B::Kernels as Kernels>::Qwen38QsaPrepareIndexKernel::new(context)
                .map_err(Qwen38Error::Backend)?,
            block_keys: <B::Kernels as Kernels>::Qwen38QsaBlockKeysKernel::new(context)
                .map_err(Qwen38Error::Backend)?,
            scores: <B::Kernels as Kernels>::Qwen38QsaScoresKernel::new(context).map_err(Qwen38Error::Backend)?,
            select: <B::Kernels as Kernels>::Qwen38QsaSelectKernel::new(context).map_err(Qwen38Error::Backend)?,
            attention: <B::Kernels as Kernels>::Qwen38QsaAttentionKernel::new(context).map_err(Qwen38Error::Backend)?,
            combine: <B::Kernels as Kernels>::Qwen38QsaCombineKernel::new(context).map_err(Qwen38Error::Backend)?,
        })
    }

    pub fn create_state(
        &self,
        context: &B::Context,
        capacity: u32,
    ) -> Result<QsaState<B>, B::Error> {
        if capacity == 0 || capacity > self.max_context {
            return Err(InvalidInput("QSA requires a bounded finite cache capacity").into());
        }
        let bytes = |rows: u32, width: u32, size: usize| -> Result<usize, B::Error> {
            (rows as usize)
                .checked_mul(width as usize)
                .and_then(|n| n.checked_mul(size))
                .filter(|&n| n > 0 && n <= isize::MAX as usize)
                .ok_or_else(|| InvalidInput("QSA cache size overflow").into())
        };
        let cache_rows = self.kv_heads.checked_mul(capacity).ok_or(InvalidInput("QSA cache rows overflow"))?;
        Ok(QsaState {
            capacity,
            length: 0,
            keys: context.create_buffer(bytes(cache_rows, HEAD_DIM, 1)?)?,
            values: context.create_buffer(bytes(cache_rows, HEAD_DIM, 1)?)?,
            key_scales: context.create_buffer(bytes(cache_rows, HEAD_DIM / CACHE_GROUP, 2)?)?,
            value_scales: context.create_buffer(bytes(cache_rows, HEAD_DIM / CACHE_GROUP, 2)?)?,
            raw_index: context.create_buffer(bytes(capacity, INDEX_DIM, 2)?)?,
            block_index: context.create_buffer(bytes((capacity / self.ratio).max(1), INDEX_DIM, 2)?)?,
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        rows: u32,
        base_position: u32,
        state: &mut QsaState<B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let end = base_position.checked_add(rows).ok_or(InvalidInput("QSA position overflow"))?;
        let input_bytes = (rows as usize)
            .checked_mul(self.hidden as usize)
            .and_then(|n| n.checked_mul(2))
            .ok_or(InvalidInput("QSA input size overflow"))?;
        if rows == 0 || base_position != state.length || end > state.capacity || input.size() != input_bytes {
            return Err(InvalidInput("QSA requires an exact consecutive flat batch").into());
        }
        let project = |linear: &LinearMatmul<B>, commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding| {
            linear.encode_with_a(
                MatmulA::FullPrecision {
                    values: input,
                    offset: 0,
                },
                rows,
                None::<Gather<&B::ScratchBuffer>>,
                commands,
            )
        };
        let qg = project(&self.q, commands)?;
        let key = project(&self.k, commands)?;
        let value = project(&self.v, commands)?;
        let index = project(&self.index, commands)?;
        let q_width = self.query_heads * HEAD_DIM;
        let kv_width = self.kv_heads * HEAD_DIM;
        let index_width = (self.index_heads + 1) * INDEX_DIM;
        let mut gated = commands.allocate_scratch_for_shape(&[rows, q_width], DataType::BF16)?;
        for row in 0..rows {
            let position = base_position + row;
            let visible = (position + 1) / self.ratio;
            let selected = visible.min(self.block_budget);
            let tokens = selected * self.ratio + (position + 1) % self.ratio;
            let splits = tokens.div_ceil(ATTENTION_SPLIT);
            let group = self.query_heads / self.kv_heads;
            let head_groups = group.div_ceil(4);
            let mut query = commands.allocate_scratch_for_shape(&[self.query_heads, HEAD_DIM], DataType::BF16)?;
            let mut gate = commands.allocate_scratch_for_shape(&[self.query_heads, HEAD_DIM], DataType::BF16)?;
            let mut index_query =
                commands.allocate_scratch_for_shape(&[self.index_heads, INDEX_DIM], DataType::BF16)?;
            self.prepare_q.encode(
                (&qg).subrange(row as usize * q_width as usize * 4..(row + 1) as usize * q_width as usize * 4),
                &self.q_norm,
                &mut query,
                &mut gate,
                self.query_heads,
                self.rotary,
                position,
                self.theta,
                self.epsilon,
                commands,
            );
            self.prepare_k.encode(
                (&key).subrange(row as usize * kv_width as usize * 2..(row + 1) as usize * kv_width as usize * 2),
                &self.k_norm,
                &mut state.keys,
                &mut state.key_scales,
                self.kv_heads,
                state.capacity,
                self.rotary,
                position,
                self.theta,
                self.epsilon,
                commands,
            );
            self.store_v.encode(
                (&value).subrange(row as usize * kv_width as usize * 2..(row + 1) as usize * kv_width as usize * 2),
                &mut state.values,
                &mut state.value_scales,
                self.kv_heads,
                state.capacity,
                position,
                commands,
            );
            self.prepare_index.encode(
                (&index)
                    .subrange(row as usize * index_width as usize * 2..(row + 1) as usize * index_width as usize * 2),
                &self.index_q_norm,
                &mut index_query,
                &mut state.raw_index,
                self.index_heads,
                self.rotary,
                position,
                self.theta,
                self.epsilon,
                commands,
            );
            if (position + 1).is_multiple_of(self.ratio) {
                self.block_keys.encode(
                    &state.raw_index,
                    &self.index_k_norm,
                    &mut state.block_index,
                    self.ratio,
                    visible - 1,
                    self.rotary,
                    self.theta,
                    self.epsilon,
                    commands,
                );
            }
            let mut scores = commands.allocate_scratch(visible.max(1) as usize * 4)?;
            let selection_capacity = self.block_budget.min(state.capacity / self.ratio).max(1);
            let mut selection = commands.allocate_scratch(selection_capacity as usize * 4)?;
            let mut count = commands.allocate_scratch(4)?;
            if visible > self.block_budget {
                self.scores.encode(&index_query, &state.block_index, &mut scores, self.index_heads, visible, commands);
            }
            self.select.encode(&scores, &mut selection, &mut count, visible, self.block_budget, commands);
            let mut partials =
                commands.allocate_scratch_for_shape(&[self.query_heads, splits, HEAD_DIM], DataType::F32)?;
            let mut stats = commands.allocate_scratch_for_shape(&[self.query_heads, splits, 2], DataType::F32)?;
            self.attention.encode(
                &query,
                &state.keys,
                &state.values,
                &state.key_scales,
                &state.value_scales,
                &selection,
                &count,
                &mut partials,
                &mut stats,
                state.capacity,
                group,
                splits,
                self.block_budget,
                self.ratio,
                position,
                (HEAD_DIM as f32).sqrt().recip(),
                self.query_heads,
                head_groups,
                self.kv_heads,
                commands,
            );
            self.combine.encode(
                &partials,
                &stats,
                &gate,
                (&mut gated)
                    .subrange_mut(row as usize * q_width as usize * 2..(row + 1) as usize * q_width as usize * 2),
                self.query_heads,
                splits,
                commands,
            );
        }
        let output = self.output.encode_with_a(
            MatmulA::FullPrecision {
                values: &gated,
                offset: 0,
            },
            rows,
            None::<Gather<&B::ScratchBuffer>>,
            commands,
        )?;
        state.length = end;
        Ok(output)
    }
}
