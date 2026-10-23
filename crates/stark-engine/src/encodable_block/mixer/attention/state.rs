use std::any::Any;

use crate::{
    backends::common::{
        Backend, Buffer, CommandBuffer, Context, DeviceCapabilities, InvalidInput, Kernels, SparseBuffer,
        gpu_types::{Copy, ring::RingParams},
        kernel::KVCacheUpdateKernel,
    },
    data_type::DataType,
    encodable_block::mixer::{MixerState, attention::Attention},
};

pub const ATTENTION_SUFFIX_CAPACITY: u32 = 1024; // TODO: remove hardcoded suffix capacity

fn checked_cache_capacity(
    prefix: u32,
    width: u32,
    element_bytes: usize,
) -> Result<(u32, usize), InvalidInput> {
    let rows = prefix.checked_add(ATTENTION_SUFFIX_CAPACITY).ok_or(InvalidInput("attention capacity overflow"))?;
    let bytes = (rows as usize)
        .checked_mul(width as usize)
        .and_then(|size| size.checked_mul(element_bytes))
        .filter(|&size| size > 0 && size <= isize::MAX as usize)
        .ok_or(InvalidInput("attention byte capacity overflow"))?;
    Ok((rows, bytes))
}

fn checked_preparation_rows(
    prefix: u32,
    suffix: u32,
    capacity: u32,
) -> Result<u32, InvalidInput> {
    let rows = prefix.checked_add(suffix).ok_or(InvalidInput("attention preparation overflow"))?;
    if suffix > ATTENTION_SUFFIX_CAPACITY || rows > capacity {
        return Err(InvalidInput("attention preparation exceeds cache capacity"));
    }
    Ok(rows)
}

#[derive(Clone, Copy)]
pub struct KVCacheView {
    prefix_len: u32,
    ring: Option<RingParams>,
}

impl KVCacheView {
    pub fn full(prefix_len: u32) -> Self {
        Self {
            prefix_len,
            ring: None,
        }
    }

    pub fn ring(
        prefix_len: u32,
        offset: u32,
    ) -> Self {
        Self {
            prefix_len,
            ring: Some(RingParams {
                ring_offset: offset,
                ring_length: prefix_len,
            }),
        }
    }

    pub fn prefix_len(self) -> u32 {
        self.prefix_len
    }

    pub fn ring_params(self) -> Option<RingParams> {
        self.ring
    }
}

enum KVCacheState {
    Full {
        length: u32,
    },
    Ring {
        offset: u32,
        length: u32,
        capacity: u32,
    },
}

impl KVCacheState {
    fn full() -> Self {
        Self::Full {
            length: 0,
        }
    }

    fn ring(capacity: u32) -> Self {
        assert!(capacity > 0, "zero ring capacity");
        Self::Ring {
            offset: 0,
            length: 0,
            capacity,
        }
    }

    pub fn view(&self) -> KVCacheView {
        match self {
            Self::Full {
                length,
            } => KVCacheView::full(*length),
            Self::Ring {
                offset,
                length,
                capacity: _,
            } => KVCacheView::ring(*length, *offset),
        }
    }

    fn required_prefix_len(
        &self,
        context_length: u32,
    ) -> u32 {
        match self {
            Self::Full {
                ..
            } => context_length,
            Self::Ring {
                capacity,
                ..
            } => context_length.min(*capacity),
        }
    }

    fn accept(
        &mut self,
        accepted_indices: &[u32],
    ) -> Vec<Copy> {
        assert!(accepted_indices.is_sorted_by(|a, b| a < b), "unsorted accept");
        let suffix_base = self.view().prefix_len();

        match self {
            Self::Full {
                length,
            } => {
                let mut copies = Vec::with_capacity(accepted_indices.len());
                for (destination_index, accepted_index) in accepted_indices.iter().copied().enumerate() {
                    let source = suffix_base + accepted_index;
                    let destination = suffix_base + destination_index as u32;
                    if source != destination {
                        copies.push(Copy {
                            source,
                            destination,
                        });
                    }
                }
                *length += accepted_indices.len() as u32;
                copies
            },
            Self::Ring {
                offset,
                length,
                capacity,
            } => {
                let count = accepted_indices.len() as u32;
                let mut copies = Vec::with_capacity(accepted_indices.len().min(*capacity as usize));
                for (index, accepted_index) in accepted_indices.iter().copied().enumerate() {
                    let source = suffix_base + accepted_index;
                    let destination = ((u64::from(*offset) + u64::from(*length)) % u64::from(*capacity)) as u32;
                    // A copy is dead when a later accept in the same batch overwrites its
                    // destination; emitting it would alias in a single kernel dispatch.
                    if index as u32 + *capacity >= count && source != destination {
                        copies.push(Copy {
                            source,
                            destination,
                        });
                    }

                    if length < capacity {
                        *length += 1;
                    } else {
                        *offset = (*offset + 1) % *capacity;
                    }
                }
                copies
            },
        }
    }
}

pub struct AttentionState<B: Backend> {
    pub element_dim: u32,
    pub data_type: DataType,
    cache: KVCacheState,
    capacity_elements: u32,
    pub is_sparse: bool,
    pub keys: Box<dyn Buffer<Backend = B>>,
    pub values: Box<dyn Buffer<Backend = B>>,
    pub kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel,
}

impl<B: Backend> AttentionState<B> {
    pub fn view(&self) -> KVCacheView {
        self.cache.view()
    }

    pub fn create_empty(
        attention: &Attention<B>,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        if max_context_length
            .is_some_and(|length| length == 0 || attention.max_rope_length.is_some_and(|maximum| length > maximum))
        {
            return Err(InvalidInput("context exceeds model capacity").into());
        }

        let data_type = attention.data_type;

        let max_prefix_elements = match attention.ring_capacity {
            Some(window) => max_context_length.map_or(window, |limit| window.min(limit)),
            None => max_context_length
                .or(attention.max_rope_length)
                .ok_or(InvalidInput("full attention requires a finite context capacity"))?,
        };

        let cache =
            attention.ring_capacity.map_or_else(KVCacheState::full, |_| KVCacheState::ring(max_prefix_elements));

        let element_dim = attention
            .num_kv_heads
            .and_then(|heads| heads.checked_mul(attention.head_dim))
            .filter(|&width| width > 0)
            .ok_or(InvalidInput("invalid attention cache width"))?;
        let (max_elements, kv_buffer_bytes) =
            checked_cache_capacity(max_prefix_elements, element_dim, data_type.size_in_bytes())?;

        let is_sparse = context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS);

        let (keys, values): (Box<dyn Buffer<Backend = B>>, Box<dyn Buffer<Backend = B>>) = if is_sparse {
            (
                Box::new(context.create_sparse_buffer(kv_buffer_bytes)?),
                Box::new(context.create_sparse_buffer(kv_buffer_bytes)?),
            )
        } else {
            (Box::new(context.create_buffer(kv_buffer_bytes)?), Box::new(context.create_buffer(kv_buffer_bytes)?))
        };

        let kv_cache_update = <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, data_type)?;

        Ok(Self {
            element_dim,
            data_type,
            cache,
            capacity_elements: max_elements,
            is_sparse,
            keys,
            values,
            kv_cache_update,
        })
    }
}

impl<B: Backend> MixerState<B> for AttentionState<B> {
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        let elements_required = checked_preparation_rows(
            self.cache.required_prefix_len(context_length),
            suffix_length,
            self.capacity_elements,
        )?;

        if self.is_sparse {
            let bytes_required = (elements_required as usize)
                .checked_mul(self.element_dim as usize)
                .and_then(|size| size.checked_mul(self.data_type.size_in_bytes()))
                .ok_or(InvalidInput("attention mapping size overflow"))?;

            let keys = (self.keys.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>().unwrap();
            let values = (self.values.as_mut() as &mut dyn Any).downcast_mut::<B::SparseBuffer>().unwrap();

            keys.map(bytes_required)?;
            values.map(bytes_required)?;
        }

        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let copies = self.cache.accept(accepted_indices);

        if !copies.is_empty() {
            self.kv_cache_update.encode(
                self.keys.as_mut(),
                self.values.as_mut(),
                &copies,
                copies.len() as u32,
                self.element_dim,
                command_buffer,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/kv_cache_state_test.rs"]
mod tests;
