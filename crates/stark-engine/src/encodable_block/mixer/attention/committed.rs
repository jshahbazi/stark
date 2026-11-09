//! Immutable target-KV borrow for Gemma's single-position external drafter.
//! Ordinary shared layers read an uncommitted suffix; this view instead
//! contains only committed target rows, including the current final row.
use crate::{
    backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput},
    encodable_block::mixer::attention::{AttentionState, KVCacheView},
};

pub struct CommittedAttentionKV<'a, B: Backend> {
    pub(super) state: &'a AttentionState<B>,
    owner_layer: u32,
    position: u32,
    pub(super) view: KVCacheView,
}

impl<'a, B: Backend> CommittedAttentionKV<'a, B> {
    pub(crate) fn new(
        state: &'a AttentionState<B>,
        owner_layer: u32,
        context: u32,
    ) -> Option<Self> {
        let view = state.committed_view(context)?;
        let bytes = (view.prefix_len() as usize)
            .checked_mul(state.element_dim as usize)?
            .checked_mul(state.data_type.size_in_bytes())?;
        if state.element_dim == 0 || state.keys.size() < bytes || state.values.size() < bytes {
            return None;
        }
        Some(Self {
            state,
            owner_layer,
            position: context,
            view,
        })
    }

    pub fn owner_layer(&self) -> u32 {
        self.owner_layer
    }
    /// Position of the unprocessed target draw after these committed rows.
    /// Keep this constant throughout one assistant proposal round.
    pub fn position(&self) -> u32 {
        self.position
    }
    pub fn retained_rows(&self) -> u32 {
        self.view.prefix_len()
    }
    pub fn is_ring(&self) -> bool {
        self.view.ring_params().is_some()
    }

    /// Account for both temporal K/V scratch buffers. Full caches are borrowed.
    pub fn scratch_bytes(&self) -> Option<usize> {
        if self.view.ring_params().is_none() {
            return Some(0);
        }
        (self.retained_rows() as usize)
            .checked_mul(self.state.element_dim as usize)?
            .checked_mul(self.state.data_type.size_in_bytes())?
            .checked_mul(2)
    }

    /// The existing attention ABI has one query row and `prefix+1` key rows.
    /// Treat the final committed row as that last row, never append a phantom KV.
    pub(super) fn query_cache_view(&self) -> KVCacheView {
        let prefix = self.retained_rows() - 1;
        if self.view.ring_params().is_some() {
            KVCacheView::ring(prefix, 0)
        } else {
            KVCacheView::full(prefix)
        }
    }

    #[cfg(test)]
    pub(crate) fn validation_copy(
        &self,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        if self.is_ring() {
            return self.linearize_ring(commands);
        }
        let bytes =
            self.retained_rows() as usize * self.state.element_dim as usize * self.state.data_type.size_in_bytes();
        let mut keys = commands.allocate_scratch(bytes)?;
        let mut values = commands.allocate_scratch(bytes)?;
        commands.encode_copy(self.state.keys.as_ref().subrange(..bytes), &mut keys);
        commands.encode_copy(self.state.values.as_ref().subrange(..bytes), &mut values);
        Ok((keys, values))
    }

    pub(super) fn linearize_ring(
        &self,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        let ring = self.view.ring_params().ok_or(InvalidInput("external KV is not a ring"))?;
        let rows = self.retained_rows();
        if rows == 0 || rows > 1024 || ring.ring_offset >= rows {
            return Err(InvalidInput("external KV ring exceeds bounded target view").into());
        }
        let stride = self.state.element_dim as usize * self.state.data_type.size_in_bytes();
        let bytes = self.scratch_bytes().ok_or(InvalidInput("external KV scratch overflow"))? / 2;
        let mut keys = commands.allocate_scratch(bytes)?;
        let mut values = commands.allocate_scratch(bytes)?;
        let split = ring.ring_offset as usize * stride;
        let tail = bytes - split;
        for (source, output) in [(&self.state.keys, &mut keys), (&self.state.values, &mut values)] {
            commands.encode_copy(source.as_ref().subrange(split..bytes), (&mut *output).subrange_mut(..tail));
            if split > 0 {
                commands.encode_copy(source.as_ref().subrange(..split), (&mut *output).subrange_mut(tail..));
            }
        }
        Ok((keys, values))
    }
}
