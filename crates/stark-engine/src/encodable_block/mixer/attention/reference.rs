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
    is_causal: bool,
}

#[cfg(test)]
mod decider_tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{
        backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
        tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_backend},
    };

    fn output<B: Backend>(
        causal: bool,
        future: f32,
    ) -> Vec<half::bf16> {
        let context = create_context::<B>();
        let config: AttentionConfig = serde_json::from_value(serde_json::json!({
            "type":"AttentionConfig", "qkvg_projection_config":{}, "out_projection_config":{},
            "query_norm_config":null, "key_norm_config":null, "num_heads":1, "num_groups":1,
            "head_dim":64, "is_causal":causal, "scale":1.0, "sliding_window_size":null,
            "logit_soft_cap":null, "has_sinks":false, "has_qkvg_biases":false,
            "has_out_biases":false, "has_gate":false, "normalize_values":false,
            "is_kv_sharing":false, "reference_bf16":true
        }))
        .unwrap();
        let attention = ReferenceBF16Attention::<B>::new(&context, &config).unwrap();
        let zeros = vec![half::bf16::ZERO; 128];
        let q = create_buffer_with_data::<B, _>(&context, &zeros);
        let k = create_buffer_with_data::<B, _>(&context, &zeros);
        let mut values = vec![half::bf16::ZERO; 64];
        values.extend(std::iter::repeat_n(half::bf16::from_f32(future), 64));
        let v = create_buffer_with_data::<B, _>(&context, &values);
        let mut command = context.create_command_buffer(None, None).unwrap();
        let result = attention.encode(&q, &k, &v, 2, 0, None, &mut command).unwrap();
        let mut readback = context.create_buffer(256).unwrap();
        command.encode_copy(&result, &mut readback);
        command.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<B, half::bf16>(&readback)
    }

    fn check<B: Backend>() {
        for future in [2.0, 4.0] {
            let causal = output::<B>(true, future);
            let bidirectional = output::<B>(false, future);
            assert!(causal[..64].iter().all(|value| value.to_f32() == 0.0));
            assert!(causal[64..].iter().all(|value| value.to_f32() == future / 2.0));
            assert!(bidirectional.iter().all(|value| value.to_f32() == future / 2.0));
        }
    }

    #[stark_test]
    fn decision_block_attention_is_bidirectional_within_two_blocks() {
        for_each_backend!(|B| {
            let context = create_context::<B>();
            let config: AttentionConfig = serde_json::from_value(serde_json::json!({
                "type":"AttentionConfig", "qkvg_projection_config":{}, "out_projection_config":{},
                "query_norm_config":null, "key_norm_config":null, "num_heads":1, "num_groups":1,
                "head_dim":64, "is_causal":true, "scale":1.0, "sliding_window_size":null,
                "logit_soft_cap":null, "has_sinks":false, "has_qkvg_biases":false,
                "has_out_biases":false, "has_gate":false, "normalize_values":false,
                "is_kv_sharing":false, "reference_bf16":true
            }))
            .unwrap();
            let attention = ReferenceBF16Attention::<B>::new(&context, &config).unwrap();
            let q = create_buffer_with_data::<B, _>(&context, &vec![half::bf16::ZERO; 3 * 64]);
            let k = create_buffer_with_data::<B, _>(&context, &vec![half::bf16::ZERO; 3 * 64]);
            let values = [0.0, 10.0, 20.0]
                .into_iter()
                .flat_map(|value| std::iter::repeat_n(half::bf16::from_f32(value), 64))
                .collect::<Vec<_>>();
            let v = create_buffer_with_data::<B, _>(&context, &values);
            let mut command = context.create_command_buffer(None, None).unwrap();
            let result = attention.encode(&q, &k, &v, 3, 0, Some(2), &mut command).unwrap();
            let mut readback = context.create_buffer(3 * 64 * 2).unwrap();
            command.encode_copy(&result, &mut readback);
            command.end_encoding().submit().wait_until_completed().unwrap();
            let values = buffer_to_vec::<B, half::bf16>(&readback);
            assert_eq!(values[0].to_f32(), 5.0);
            assert_eq!(values[64].to_f32(), 5.0);
            assert!((values[128].to_f32() - 10.0).abs() < 0.125);
        });
    }

    #[stark_test]
    fn decider_noncausal_attention_reads_future_rows_and_causal_attention_does_not() {
        for_each_backend!(|B| check::<B>());
    }
}

impl<B: Backend> ReferenceBF16Attention<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &AttentionConfig,
    ) -> Result<Self, B::Error> {
        if config.is_kv_sharing
            || config.sliding_window_size.is_some()
            || config.has_sinks
            || config.logit_soft_cap.is_some()
        {
            return Err(InvalidInput("reference BF16 attention requires owned full cache").into());
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
            is_causal: config.is_causal,
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
        block_boundary: Option<u32>,
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
            // The eager score kernel masks columns beyond prefix + query row.
            // A full key-count prefix admits every column for noncausal input.
            let mask_prefix = if self.is_causal {
                prefix_len
            } else {
                key_count
            };
            self.score.encode(
                &dot,
                &mut probabilities,
                key_count,
                rows,
                mask_prefix,
                block_boundary.unwrap_or(0),
                self.scale,
                commands,
            );
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
            self.score.encode(&dot, &mut probabilities, key_count, 1, key_count - 1, 0, self.scale, commands);
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
