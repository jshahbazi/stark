//! BF16 eager boundaries for source-bound owned or committed-query paths.

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            ReferenceCausalScoresBF16Kernel, VisionCastBF16Kernel, VisionHeadCopyBF16Kernel, VisionMatmulBF16Kernel,
            VisionSoftmaxBF16Kernel,
        },
    },
    config::token_mixer::attention::AttentionConfig,
    data_type::DataType,
};

pub(crate) struct ReferenceBF16Attention<B: Backend> {
    dot: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    score: <B::Kernels as Kernels>::ReferenceCausalScoresBF16Kernel,
    softmax: <B::Kernels as Kernels>::VisionSoftmaxBF16Kernel,
    cast: <B::Kernels as Kernels>::VisionCastBF16Kernel,
    copy: <B::Kernels as Kernels>::VisionHeadCopyBF16Kernel,
    heads: u32,
    groups: u32,
    head_dim: u32,
    scale: f32,
}

impl<B: Backend> ReferenceBF16Attention<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &AttentionConfig,
    ) -> Result<Self, B::Error> {
        if !config.is_causal
            || config.is_kv_sharing
            || config.sliding_window_size.is_some()
            || config.has_sinks
            || config.logit_soft_cap.is_some()
        {
            return Err(InvalidInput("reference BF16 attention requires owned full causal cache").into());
        }
        Self::kernels(context, config)
    }

    /// A single query over already committed, temporally ordered rows. The
    /// caller validates the window and row count before entering this path.
    /// This grants no mutable shared-cache or partial-ring acceptance support.
    pub(crate) fn for_committed_target(
        context: &B::Context,
        config: &AttentionConfig,
    ) -> Result<Self, B::Error> {
        if !config.is_causal
            || !config.is_kv_sharing
            || config.has_sinks
            || config.logit_soft_cap.is_some()
            || config.sliding_window_size.is_some_and(|window| window != 513)
        {
            return Err(InvalidInput("invalid committed BF16 assistant attention profile").into());
        }
        Self::kernels(context, config)
    }

    fn kernels(
        context: &B::Context,
        config: &AttentionConfig,
    ) -> Result<Self, B::Error> {
        crate::config::validation::validate_heads(config.num_heads, config.num_groups, config.head_dim)?;
        Ok(Self {
            dot: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context)?,
            score: <B::Kernels as Kernels>::ReferenceCausalScoresBF16Kernel::new(context)?,
            softmax: <B::Kernels as Kernels>::VisionSoftmaxBF16Kernel::new(context)?,
            cast: <B::Kernels as Kernels>::VisionCastBF16Kernel::new(context)?,
            copy: <B::Kernels as Kernels>::VisionHeadCopyBF16Kernel::new(context)?,
            heads: config.num_heads,
            groups: config.num_groups,
            head_dim: config.head_dim,
            scale: config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
        })
    }

    pub(crate) fn matches_geometry(
        &self,
        heads: u32,
        groups: u32,
        head_dim: u32,
    ) -> bool {
        self.heads == heads && self.groups == groups && self.head_dim == head_dim
    }

    pub(crate) fn encode(
        &self,
        queries: impl BufferRef<Backend = B>,
        keys: impl BufferRef<Backend = B>,
        values: impl BufferRef<Backend = B>,
        rows: u32,
        prefix_len: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let key_count = prefix_len.checked_add(rows).ok_or(InvalidInput("reference attention cache overflow"))?;
        let head_elements = rows as usize * self.head_dim as usize;
        let cache_stride = self.groups * self.head_dim;
        let required_cache = key_count as usize * cache_stride as usize * size_of::<half::bf16>();
        if rows == 0
            || queries.size() != head_elements * self.heads as usize * size_of::<half::bf16>()
            || keys.size() < required_cache
            || values.size() < required_cache
        {
            return Err(InvalidInput("invalid reference attention buffer shape").into());
        }
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.heads, self.head_dim], DataType::BF16)?;
        // Reuse each head's intermediates in dispatch order. Scratch grows with
        // rows*keys rather than retaining a scores tensor for every query head.
        let mut dot = commands.allocate_scratch_for_shape(&[rows, key_count], DataType::F32)?;
        let mut probabilities = commands.allocate_scratch_for_shape(&[rows, key_count], DataType::BF16)?;
        let mut av = commands.allocate_scratch_for_shape(&[rows, self.head_dim], DataType::F32)?;
        let mut head = commands.allocate_scratch_for_shape(&[rows, self.head_dim], DataType::BF16)?;
        let queries_per_group = self.heads / self.groups;
        for head_index in 0..self.heads {
            let q_start = head_index as usize * head_elements * size_of::<half::bf16>();
            let q_end = q_start + head_elements * size_of::<half::bf16>();
            let kv_start = (head_index / queries_per_group * self.head_dim) as usize * size_of::<half::bf16>();
            self.dot.encode(
                queries.subrange(q_start..q_end),
                keys.subrange(kv_start..),
                &mut dot,
                rows,
                key_count,
                self.head_dim,
                cache_stride,
                1,
                32,
                commands,
            );
            self.score.encode(&dot, &mut probabilities, key_count, rows, prefix_len, self.scale, commands);
            self.softmax.encode(&mut probabilities, key_count, rows, commands);
            self.dot.encode(
                &probabilities,
                values.subrange(kv_start..),
                &mut av,
                rows,
                self.head_dim,
                key_count,
                cache_stride,
                0,
                4,
                commands,
            );
            self.cast.encode(&av, &mut head, rows * self.head_dim, commands);
            self.copy.encode(
                &head,
                &mut output,
                rows * self.head_dim,
                self.head_dim,
                self.heads * self.head_dim,
                head_index,
                commands,
            );
        }
        Ok(output)
    }

    /// Ephemeral first-divergence instrumentation; ordinary encoding is unchanged.
    #[cfg(test)]
    pub(crate) fn trace_single_query(
        &self,
        queries: impl BufferRef<Backend = B>,
        keys: impl BufferRef<Backend = B>,
        values: impl BufferRef<Backend = B>,
        key_count: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Vec<(String, B::ScratchBuffer)>, B::Error> {
        let mut stages = Vec::new();
        let stride = self.groups * self.head_dim;
        for index in 0..self.heads {
            let start = index as usize * self.head_dim as usize * 2;
            let group_start = (index / (self.heads / self.groups) * self.head_dim) as usize * 2;
            let mut dot = commands.allocate_scratch_for_shape(&[1, key_count], DataType::F32)?;
            let mut probabilities = commands.allocate_scratch_for_shape(&[1, key_count], DataType::BF16)?;
            let mut scores = commands.allocate_scratch_for_shape(&[1, key_count], DataType::BF16)?;
            let mut weighted = commands.allocate_scratch_for_shape(&[1, self.head_dim], DataType::F32)?;
            let mut output = commands.allocate_scratch_for_shape(&[1, self.head_dim], DataType::BF16)?;
            self.dot.encode(
                queries.subrange(start..start + self.head_dim as usize * 2),
                keys.subrange(group_start..),
                &mut dot,
                1,
                key_count,
                self.head_dim,
                stride,
                1,
                32,
                commands,
            );
            self.score.encode(&dot, &mut probabilities, key_count, 1, key_count - 1, self.scale, commands);
            commands.encode_copy(&probabilities, &mut scores);
            self.softmax.encode(&mut probabilities, key_count, 1, commands);
            self.dot.encode(
                &probabilities,
                values.subrange(group_start..),
                &mut weighted,
                1,
                self.head_dim,
                key_count,
                stride,
                0,
                4,
                commands,
            );
            self.cast.encode(&weighted, &mut output, self.head_dim, commands);
            stages.push((format!("head{index}.scores"), scores));
            stages.push((format!("head{index}.probabilities"), probabilities));
            stages.push((format!("head{index}.weighted_values"), output));
        }
        Ok(stages)
    }
}
