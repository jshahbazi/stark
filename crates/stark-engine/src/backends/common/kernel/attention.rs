use super::AttentionImageSpanKernel;
use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels},
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
};

/// Image blocks extend the causal/window mask. Text calls retain their dispatch.
pub(crate) fn encode_image_span<B: Backend>(
    config: AttentionKernelConfig,
    arguments: AttentionArguments<
        '_,
        B,
        impl BufferRef<Backend = B>,
        impl BufferRef<Backend = B>,
        impl BufferRef<Backend = B>,
        impl BufferRef<Backend = B>,
    >,
    commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<B::ScratchBuffer, B::Error> {
    let span = arguments.image_span.ok_or(InvalidInput("missing image attention block"))?;
    let prefix = arguments.cache.prefix_len();
    let rows = arguments.suffix_length;
    let keys = prefix.checked_add(rows).ok_or(InvalidInput("image attention context overflow"))?;
    let element_bytes = config.data_type.size_in_bytes();
    let query_bytes = (rows as usize)
        .checked_mul(config.num_q_heads as usize)
        .and_then(|n| n.checked_mul(config.head_dim as usize))
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or(InvalidInput("image attention query shape overflow"))?;
    let cache_bytes = (keys as usize)
        .checked_mul(config.num_groups as usize)
        .and_then(|n| n.checked_mul(config.head_dim as usize))
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or(InvalidInput("image attention cache shape overflow"))?;
    crate::config::validation::validate_heads(config.num_q_heads, config.num_groups, config.head_dim)?;
    if !config.is_causal
        || config.has_sinks
        || arguments.sinks.is_some()
        || arguments.trie.is_some()
        || rows == 0
        || rows > 1024
        || span[0] >= span[1]
        || span[1] > rows
        || !matches!(config.data_type, DataType::BF16 | DataType::F32)
        || !matches!(config.head_dim, 64 | 128 | 256 | 512)
        || config.is_kv_cache_ring != arguments.cache.ring_params().is_some()
        || arguments.queries.size() != query_bytes
        || arguments.keys.size() < cache_bytes
        || arguments.values.size() < cache_bytes
    {
        return Err(InvalidInput("invalid complete image attention batch").into());
    }
    let ring = arguments.cache.ring_params();
    if ring.is_some_and(|r| {
        r.ring_length != prefix || (prefix == 0 && r.ring_offset != 0) || (prefix > 0 && r.ring_offset >= prefix)
    }) {
        return Err(InvalidInput("invalid image attention ring state").into());
    }
    let kernel =
        <B::Kernels as Kernels>::AttentionImageSpanKernel::new(commands.context(), config.data_type, config.head_dim)?;
    let mut output =
        commands.allocate_scratch_for_shape(&[rows, config.num_q_heads, config.head_dim], config.data_type)?;
    kernel.encode(
        arguments.queries,
        arguments.keys,
        arguments.values,
        &mut output,
        config.num_q_heads / config.num_groups,
        config.num_q_heads,
        rows,
        prefix,
        ring.map_or(0, |r| r.ring_offset),
        u32::from(ring.is_some()),
        config.sliding_window_size.unwrap_or(0),
        span[0],
        span[1],
        config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
        commands,
    );
    Ok(output)
}

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, <Self::Backend as Backend>::Error>;
}

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
}

pub struct AttentionArguments<
    'a,
    B: Backend,
    QT: BufferRef<Backend = B>,
    TT: BufferRef<Backend = B>,
    KT: BufferRef<Backend = B>,
    VT: BufferRef<Backend = B>,
> {
    pub queries: QT,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub sinks: Option<&'a B::GlobalBuffer>,
    pub cache: KVCacheView,
    /// Batch-local [start, end) containing a complete bidirectional image block.
    /// Image blocks must never straddle prefill batches; trie batches cannot use this mask.
    pub image_span: Option<[u32; 2]>,
}
