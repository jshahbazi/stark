use std::any::Any;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::{rope::AnyRoPEConfig, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{
            attention::{Attention, AttentionNewError, rope::PrecalculatedRoPE},
            delta_net::{DeltaNet, DeltaNetNewError},
            mamba2::{Mamba2, Mamba2NewError},
            short_conv::{ShortConv, ShortConvNewError},
        },
    },
    parameters::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

pub mod attention;
pub mod delta_net;
pub mod mamba2;
pub mod short_conv;

pub trait MixerState<B: Backend>: Any + Send {
    /// The coordinator fences before publishing device-owned snapshot payloads.
    fn capture_prefix(
        &self,
        context_length: u32,
        _context: &B::Context,
        _command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<Box<dyn MixerStateSnapshot<B>>>, B::Error> {
        Ok(self
            .can_rewind(context_length)
            .then(|| Box::new(RewindSnapshot(context_length)) as Box<dyn MixerStateSnapshot<B>>))
    }
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error>;

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error>;

    /// Whether `rewind(context_length)` would succeed. Recurrent and ring
    /// states cannot drop their newest rows and never rewind.
    fn can_rewind(
        &self,
        _context_length: u32,
    ) -> bool {
        false
    }

    /// Keep only the first `context_length` retained rows. Returns false and
    /// leaves the state unchanged when `can_rewind` is false.
    fn rewind(
        &mut self,
        _context_length: u32,
    ) -> bool {
        false
    }
}

pub trait MixerStateSnapshot<B: Backend>: Send {
    fn can_restore(
        &self,
        state: &dyn MixerState<B>,
    ) -> bool;
    fn restore(
        &self,
        state: &mut dyn MixerState<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> bool;
    fn payload_bytes(&self) -> usize;
}

struct RewindSnapshot(u32);
impl<B: Backend> MixerStateSnapshot<B> for RewindSnapshot {
    fn can_restore(
        &self,
        state: &dyn MixerState<B>,
    ) -> bool {
        state.can_rewind(self.0)
    }
    fn restore(
        &self,
        state: &mut dyn MixerState<B>,
        _: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> bool {
        state.rewind(self.0)
    }
    fn payload_bytes(&self) -> usize {
        0
    }
}

impl<'a, B: Backend> MaybeMut<'a, dyn MixerState<B>> {
    pub fn downcast<T: MixerState<B>>(self) -> Option<MaybeMut<'a, T>> {
        match self {
            MaybeMut::Const(value) => (value as &dyn Any).downcast_ref::<T>().map(MaybeMut::Const),
            MaybeMut::Mut(value) => (value as &mut dyn Any).downcast_mut::<T>().map(MaybeMut::Mut),
        }
    }
}

pub trait Mixer<B: Backend>: Any + Send + Sync {
    fn speculation_supported(&self) -> bool;

    fn max_context_length(&self) -> Option<u32>;

    fn memory_estimate(
        &self,
        _context_tokens: u32,
        _input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        None
    }

    /// Whether every state this mixer creates can be rewound to a shorter
    /// prefix. Only full, non-ring attention caches qualify.
    fn prefix_reuse_supported(&self) -> bool {
        false
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error>;

    fn encode(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error>;
}

#[derive(Debug, Error)]
pub enum MixerNewError<B: Backend> {
    #[error("Attention mixer error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Delta net mixer error: {0}")]
    DeltaNet(#[from] DeltaNetNewError<B>),
    #[error("Mamba2 mixer error: {0}")]
    Mamba2(#[from] Mamba2NewError<B>),
    #[error("Short conv mixer error: {0}")]
    ShortConv(#[from] ShortConvNewError<B>),
}

impl<B: Backend> dyn Mixer<B> {
    pub fn new(
        hidden_dim: u32,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AnyTokenMixerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Mixer<B>>, Option<B::GlobalBuffer>), MixerNewError<B>> {
        match config {
            AnyTokenMixerConfig::AttentionConfig(config) => {
                let (attention, in_projection_input_hadamard_factors) =
                    Attention::new(hidden_dim, data_type, rope_config, config, parameter_tree, context)?;

                Ok((Box::new(attention), in_projection_input_hadamard_factors))
            },
            AnyTokenMixerConfig::DeltaNetConfig(config) => {
                assert!(rope_config.is_none(), "unexpected rope for delta net mixer");

                let (delta_net, in_projection_input_hadamard_factors) =
                    DeltaNet::new(hidden_dim, data_type, config, parameter_tree, context)?;

                Ok((Box::new(delta_net), in_projection_input_hadamard_factors))
            },
            AnyTokenMixerConfig::Mamba2Config(config) => {
                assert!(rope_config.is_none(), "unexpected rope for mamba2 mixer");

                let (mamba2, in_projection_input_hadamard_factors) =
                    Mamba2::new(hidden_dim, data_type, config, parameter_tree, context)?;

                Ok((Box::new(mamba2), in_projection_input_hadamard_factors))
            },
            AnyTokenMixerConfig::ShortConvConfig(config) => {
                assert!(rope_config.is_none(), "unexpected rope for short conv mixer");

                let (short_conv, in_projection_input_hadamard_factors) =
                    ShortConv::new(hidden_dim, data_type, config, parameter_tree, context)?;

                Ok((Box::new(short_conv), in_projection_input_hadamard_factors))
            },
        }
    }
}
