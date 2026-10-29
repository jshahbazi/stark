use std::collections::{HashMap, hash_map::Entry};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionKernelConfig, AttentionSinglePassKernel},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::{AttentionSinglePassMetalKernel, SourceAttentionSinglePassMetalKernel},
        },
    },
    data_type::DataType,
};

pub struct AttentionSinglePass {
    kernels: Mutex<HashMap<bool, AttentionSinglePassMetalKernel>>,
    source_kernel: Mutex<Option<SourceAttentionSinglePassMetalKernel>>,
    config: AttentionKernelConfig,
}

impl AttentionSinglePass {
    pub fn new(config: &AttentionKernelConfig) -> Self {
        Self {
            kernels: Mutex::new(HashMap::new()),
            source_kernel: Mutex::new(None),
            config: *config,
        }
    }

    fn get_or_create_source(
        &self,
        context: &MetalContext,
    ) -> Result<MappedMutexGuard<'_, SourceAttentionSinglePassMetalKernel>, MetalError> {
        let mut kernel = self.source_kernel.lock();
        if kernel.is_none() {
            *kernel = Some(SourceAttentionSinglePassMetalKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                false,
                self.config.sliding_window_size.is_some(),
            )?);
        }
        Ok(MutexGuard::map(kernel, |kernel| kernel.as_mut().expect("source kernel was just initialized")))
    }

    fn get_or_create(
        &self,
        context: &MetalContext,
        is_trie: bool,
    ) -> Result<MappedMutexGuard<'_, AttentionSinglePassMetalKernel>, MetalError> {
        let mut kernels = self.kernels.lock();
        if let Entry::Vacant(entry) = kernels.entry(is_trie) {
            let kernel = AttentionSinglePassMetalKernel::new(
                context,
                self.config.data_type,
                self.config.head_dim,
                self.config.has_sinks,
                self.config.is_kv_cache_ring,
                self.config.is_causal,
                is_trie,
                self.config.sliding_window_size.is_some(),
            )?;
            entry.insert(kernel);
        }
        Ok(MutexGuard::map(kernels, |kernels| kernels.get_mut(&is_trie).expect("kernel was just initialized")))
    }

    pub fn encode(
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
        let config = self.config;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        macro_rules! encode_vector {
            ($kernel:expr $(, $source_origin:expr)?) => {
                $kernel.encode(
                    arguments.queries,
                    arguments.keys,
                    arguments.values,
                    &mut output,
                    config.num_q_heads / config.num_groups,
                    arguments.cache.prefix_len() + arguments.suffix_length,
                    config.head_dim,
                    config.num_groups * config.head_dim,
                    config.head_dim,
                    config.num_groups * config.head_dim,
                    arguments.cache.ring_params(),
                    $($source_origin,)?
                    config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
                    arguments.trie,
                    config.sliding_window_size,
                    arguments.sinks,
                    config.num_q_heads,
                    arguments.suffix_length,
                    command_buffer,
                );
            };
        }
        let source = (config.source_bf16_prefill || config.materialize_prefill_scores)
            && config.data_type == DataType::BF16
            && matches!(config.head_dim, 64 | 128 | 256)
            && config.is_causal
            && !config.has_sinks
            && arguments.trie.is_none()
            && arguments.suffix_length == 1
            && arguments.cache.prefix_len() + arguments.suffix_length < 1024;
        if source {
            let kernel = self.get_or_create_source(command_buffer.context())?;
            encode_vector!(kernel, arguments.cache.source_ring_origin());
        } else {
            let kernel = self.get_or_create(command_buffer.context(), arguments.trie.is_some())?;
            encode_vector!(kernel);
        }
        Ok(output)
    }
}
