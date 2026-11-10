//! Immutable target-KV borrow for Gemma's single-position external drafter.
//! Ordinary shared layers read an uncommitted suffix; this view instead
//! contains only committed target rows, including the current final row.
use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, InvalidInput,
    },
    encodable_block::mixer::attention::{AttentionState, KVCacheView},
};

pub struct CommittedAttentionKV<'a, B: Backend> {
    pub(super) state: &'a AttentionState<B>,
    owner_layer: u32,
    position: u32,
    pub(super) view: KVCacheView,
    temporal_handoff: Option<(&'a B::GlobalBuffer, &'a B::GlobalBuffer)>,
}

/// Immutable source view captured before ordinary target ring acceptance can
/// discard the oldest of the assistant's inclusive 513 visible local rows.
pub(crate) struct AssistantAttentionHandoff<B: Backend> {
    keys: B::GlobalBuffer,
    values: B::GlobalBuffer,
    owner_layer: u32,
    position: u32,
    rows: u32,
}

impl<B: Backend> AssistantAttentionHandoff<B> {
    pub(crate) fn payload_bytes(&self) -> usize {
        self.keys.size() + self.values.size()
    }
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
            temporal_handoff: None,
        })
    }

    pub(crate) fn with_assistant_handoff(
        mut self,
        handoff: &'a AssistantAttentionHandoff<B>,
    ) -> Option<Self> {
        let bytes = handoff.rows as usize * self.state.element_dim as usize * self.state.data_type.size_in_bytes();
        if !self.is_ring()
            || handoff.owner_layer != self.owner_layer
            || handoff.position != self.position
            || handoff.rows != self.position.min(513)
            || handoff.keys.size() != bytes
            || handoff.values.size() != bytes
        {
            return None;
        }
        self.view = KVCacheView::ring(handoff.rows, 0);
        self.temporal_handoff = Some((&handoff.keys, &handoff.values));
        Some(self)
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
        if let Some((source_keys, source_values)) = self.temporal_handoff {
            commands.encode_copy(source_keys, &mut keys);
            commands.encode_copy(source_values, &mut values);
            return Ok((keys, values));
        }
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

impl<B: Backend> AttentionState<B> {
    pub(crate) fn capture_assistant_handoff(
        &self,
        owner_layer: u32,
        position: u32,
        suffix_rows: u32,
        context: &B::Context,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<AssistantAttentionHandoff<B>>, B::Error> {
        let view = self.view();
        let Some(ring) = view.ring_params() else {
            return Ok(None);
        };
        let prefix = view.prefix_len();
        let total = prefix.checked_add(suffix_rows).ok_or(InvalidInput("assistant handoff row overflow"))?;
        if suffix_rows == 0
            || suffix_rows > super::ATTENTION_SUFFIX_CAPACITY
            || self.data_type != crate::data_type::DataType::BF16
            || prefix > 512
            || prefix != position.saturating_sub(suffix_rows).min(512)
            || self.element_dim == 0
            || position < suffix_rows
            || (prefix > 0 && ring.ring_offset >= prefix)
            || (prefix == 0 && ring.ring_offset != 0)
        {
            return Err(InvalidInput("invalid preaccept assistant local handoff").into());
        }
        let rows = position.min(513);
        if rows == 0 || total < rows {
            return Err(InvalidInput("incomplete preaccept assistant local handoff").into());
        }
        let stride = self.element_dim as usize * self.data_type.size_in_bytes();
        if self.keys.size() < total as usize * stride || self.values.size() < total as usize * stride {
            return Err(InvalidInput("assistant local handoff exceeds selected cache").into());
        }
        let bytes = rows as usize * stride;
        let mut keys = context.create_buffer(bytes)?;
        let mut values = context.create_buffer(bytes)?;
        let keep_start = total - rows;
        let mut logical = 0;
        let mut output_row = 0usize;
        for (start, end) in [(ring.ring_offset, prefix), (0, ring.ring_offset), (prefix, total)] {
            let length = end - start;
            let skip = keep_start.saturating_sub(logical).min(length);
            let count = length - skip;
            if count > 0 {
                let source = (start + skip) as usize * stride..end as usize * stride;
                let destination = output_row * stride..(output_row + count as usize) * stride;
                commands.encode_copy(
                    self.keys.as_ref().subrange(source.clone()),
                    (&mut keys).subrange_mut(destination.clone()),
                );
                commands.encode_copy(self.values.as_ref().subrange(source), (&mut values).subrange_mut(destination));
                output_row += count as usize;
            }
            logical += length;
        }
        if output_row != rows as usize {
            return Err(InvalidInput("incomplete assistant local copy").into());
        }
        Ok(Some(AssistantAttentionHandoff {
            keys,
            values,
            owner_layer,
            position,
            rows,
        }))
    }
}
