use crate::backends::common::Backend;

mod fallback;
mod gemm;
mod gemm_grouped;
mod single_pass;
mod source_softmax;
mod two_pass;

use crate::backends::{
    common::{
        BufferRef, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
    },
    metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaskKind {
    None,
    Causal,
    Trie,
}

impl MaskKind {
    fn for_attention(
        is_causal: bool,
        is_trie: bool,
    ) -> Option<Self> {
        match (is_causal, is_trie) {
            (true, true) => Some(Self::Trie),
            (true, false) => Some(Self::Causal),
            (false, false) => Some(Self::None),
            (false, true) => None,
        }
    }

    fn is_causal(self) -> bool {
        !matches!(self, Self::None)
    }

    fn is_trie(self) -> bool {
        matches!(self, Self::Trie)
    }
}

const SINGLE_PASS_KV_THRESHOLD: u32 = 1_024;
const D256_SINGLE_PASS_KV_THRESHOLD: u32 = 150_000;
const D128_SHORT_KV_GEMM_THRESHOLD: u32 = 2_048;

pub struct AttentionMetalKernel {
    head_dim: u32,
    is_causal: bool,
    grouped: Option<gemm_grouped::AttentionGemmGrouped>,
    gemm: Option<gemm::AttentionGemm>,
    fallback: Option<fallback::AttentionFallback>,
    two_pass: two_pass::AttentionTwoPass,
    single_pass: single_pass::AttentionSinglePass,
    materialize_prefill_scores: bool,
}

impl AttentionKernel for AttentionMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, MetalError> {
        crate::config::validation::validate_heads(config.num_q_heads, config.num_groups, config.head_dim)?;
        config.validate_prefill_materialization()?;
        let grouped = gemm_grouped::AttentionGemmGrouped::is_supported(&config, context)
            .then(|| gemm_grouped::AttentionGemmGrouped::new(&config));
        let gemm = gemm::AttentionGemm::is_supported(&config).then(|| gemm::AttentionGemm::new(&config));
        let fallback = fallback::AttentionFallback::is_supported(&config)
            .then(|| fallback::AttentionFallback::new(&config, context))
            .transpose()?;
        Ok(Self {
            head_dim: config.head_dim,
            is_causal: config.is_causal,
            grouped,
            gemm,
            fallback,
            two_pass: two_pass::AttentionTwoPass::new(&config),
            single_pass: single_pass::AttentionSinglePass::new(&config),
            materialize_prefill_scores: config.materialize_prefill_scores,
        })
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        command_buffer.push_debug_group("attention core");
        let result = self.encode_impl(arguments, command_buffer);
        command_buffer.pop_debug_group();
        result
    }
}

impl AttentionMetalKernel {
    #[cfg(test)]
    pub(crate) fn encode_source_scale_prefill_for_test(
        config: AttentionKernelConfig,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        gemm::AttentionGemm::new(&config).encode_source_scale_for_test(arguments, command_buffer)
    }

    #[cfg(test)]
    pub(crate) fn encode_mxu_prefill_for_test(
        config: AttentionKernelConfig,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        gemm::AttentionGemm::new(&config).encode_forced_mxu_for_test(arguments, command_buffer)
    }

    #[cfg(test)]
    pub(crate) fn encode_unfused_prefill_for_test(
        context: &MetalContext,
        config: AttentionKernelConfig,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        fallback::AttentionFallback::new(&config, context)?.encode(arguments, command_buffer)
    }

    #[cfg(test)]
    pub(crate) fn encode_fallback_trace_for_test(
        context: &MetalContext,
        config: AttentionKernelConfig,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(<Metal as Backend>::ScratchBuffer, Vec<(&'static str, <Metal as Backend>::ScratchBuffer)>), MetalError>
    {
        fallback::AttentionFallback::new(&config, context)?.encode_trace(arguments, command_buffer)
    }

    fn encode_impl(
        &self,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let suffix_length = arguments.suffix_length;
        let kv_length = arguments.cache.prefix_len() + suffix_length;
        let is_trie = arguments.trie.is_some();
        let is_ring = arguments.cache.ring_params().is_some();
        if self.materialize_prefill_scores
            && ((9..1024).contains(&suffix_length) || (self.head_dim == 512 && kv_length < 1024))
        {
            if is_trie {
                return Err(crate::backends::common::InvalidInput(
                    "materialized prefill attention does not support tries",
                )
                .into());
            }
            return self
                .fallback
                .as_ref()
                .ok_or(crate::backends::common::InvalidInput("materialized prefill attention fallback is missing"))?
                .encode(arguments, command_buffer);
        }
        if !is_trie
            && let Some(gemm) = &self.gemm
            && gemm.uses_source_mxu_prefill(suffix_length, command_buffer.context().supports_mxu)
        {
            return gemm.encode(arguments, command_buffer);
        }
        let mask = MaskKind::for_attention(self.is_causal, is_trie);

        if !is_ring
            && let Some(mask) = mask
            && let Some(grouped) = &self.grouped
        {
            if grouped.should_encode(mask, suffix_length, kv_length) {
                return grouped.encode(mask, arguments, command_buffer);
            }

            // These measured sibling exceptions apply only when the
            // grouped core was built for this layer.
            if self.head_dim == 256 && (1..=16).contains(&suffix_length) && kv_length > D256_SINGLE_PASS_KV_THRESHOLD {
                return self.single_pass.encode(arguments, command_buffer);
            }
            if (self.head_dim == 256 && (9..=16).contains(&suffix_length))
                || (self.head_dim == 128
                    && (9..=16).contains(&suffix_length)
                    && kv_length > D128_SHORT_KV_GEMM_THRESHOLD)
            {
                return self.two_pass.encode(arguments, command_buffer);
            }
        }

        if suffix_length > 8 {
            if let Some(gemm) = &self.gemm {
                return gemm.encode(arguments, command_buffer);
            }
            if !is_trie && let Some(fallback) = &self.fallback {
                return fallback.encode(arguments, command_buffer);
            }
        }

        if kv_length > SINGLE_PASS_KV_THRESHOLD {
            self.two_pass.encode(arguments, command_buffer)
        } else {
            self.single_pass.encode(arguments, command_buffer)
        }
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/attention/mod.rs"]
mod tests;
