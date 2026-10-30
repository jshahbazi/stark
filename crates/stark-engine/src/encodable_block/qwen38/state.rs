//! Request-owned state for the full Qwen3.8 Uzu decoder.
use super::{Encoding, gdn::GdnState, ple::PleState, qsa::QsaState};
use crate::backends::common::{Backend, InvalidInput};

pub(super) enum MixerState<B: Backend> {
    Gdn(GdnState<B>),
    Qsa(QsaState<B>),
}
pub(super) struct LayerState<B: Backend> {
    pub mixer: MixerState<B>,
    pub ple: Option<PleState<B>>,
}

pub struct Qwen38State<B: Backend> {
    pub(super) layers: Vec<LayerState<B>>,
    pub(super) capacity: u32,
    context_length: u32,
    pending: u32,
    failed: bool,
}

impl<B: Backend> Qwen38State<B> {
    pub(super) fn new(
        layers: Vec<LayerState<B>>,
        capacity: u32,
    ) -> Self {
        Self {
            layers,
            capacity,
            context_length: 0,
            pending: 0,
            failed: false,
        }
    }
    pub fn context_length(&self) -> u32 {
        self.context_length
    }
    pub fn is_valid(&self) -> bool {
        !self.failed
    }
    pub fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        if self.failed || context_length.checked_add(suffix_length).is_none_or(|n| n > self.capacity) {
            return Err(InvalidInput("Qwen state is invalid or context capacity exceeded").into());
        }
        // Generic scoring may reserve a future final chunk before encoding
        // its preceding chunks. State is already capacity-sized; no rewind is
        // performed here. Actual encoding always uses committed context.
        Ok(())
    }
    pub(super) fn begin_encode(
        &mut self,
        count: u32,
    ) -> Result<(), B::Error> {
        if self.failed
            || self.pending != 0
            || count == 0
            || self.context_length.checked_add(count).is_none_or(|n| n > self.capacity)
        {
            return Err(InvalidInput("invalid Qwen encoding state").into());
        }
        if self
            .layers
            .iter()
            .any(|layer| matches!(&layer.mixer, MixerState::Qsa(cache) if cache.length() != self.context_length))
        {
            self.failed = true;
            return Err(InvalidInput("Qwen cache prefix does not match committed context").into());
        }
        // Any encoding error leaves this state fenced. Recurrent/cache
        // buffers may already be referenced by submitted command work.
        self.failed = true;
        self.pending = count;
        Ok(())
    }
    pub(super) fn finish_encode(&mut self) {
        self.failed = false;
    }
    pub fn invalidate(&mut self) {
        self.failed = true;
    }
    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        _commands: &mut Encoding<B>,
    ) -> Result<(), B::Error> {
        if self.failed
            || self.pending == 0
            || accepted_indices.len() != self.pending as usize
            || accepted_indices.iter().enumerate().any(|(index, &value)| index as u32 != value)
        {
            self.failed = true;
            return Err(InvalidInput("Qwen requires acceptance of the entire flat suffix").into());
        }
        self.context_length += self.pending;
        self.pending = 0;
        Ok(())
    }
}

#[cfg(all(test, backend = "cpu"))]
mod tests {
    use super::*;
    use crate::backends::{common::Context, cpu::Cpu};

    fn commands() -> Encoding<Cpu> {
        let context = <<Cpu as Backend>::Context as Context>::new().unwrap();
        context.create_command_buffer(None, None).unwrap()
    }

    #[stark_engine_macros::stark_test]
    fn future_reservation_does_not_advance_context_and_full_chunks_commit_once() {
        let mut state = Qwen38State::<Cpu>::new(Vec::new(), 8);
        let mut commands = commands();
        state.prepare(6, 2).unwrap();
        assert_eq!(state.context_length(), 0);
        state.begin_encode(3).unwrap();
        state.finish_encode();
        state.encode_accept(&[0, 1, 2], &mut commands).unwrap();
        assert_eq!(state.context_length(), 3);
        assert!(state.encode_accept(&[0], &mut commands).is_err());
        assert_eq!(state.context_length(), 3);
    }

    #[stark_engine_macros::stark_test]
    fn partial_reordered_duplicate_or_failed_encoding_fences_recurrent_state() {
        for accepted in [&[0][..], &[1, 0][..], &[0, 0][..]] {
            let mut state = Qwen38State::<Cpu>::new(Vec::new(), 8);
            state.begin_encode(2).unwrap();
            state.finish_encode();
            assert!(state.encode_accept(accepted, &mut commands()).is_err());
            assert!(state.begin_encode(1).is_err());
            assert_eq!(state.context_length(), 0);
        }
        let mut state = Qwen38State::<Cpu>::new(Vec::new(), 8);
        state.begin_encode(2).unwrap();
        assert!(state.encode_accept(&[0, 1], &mut commands()).is_err());
        assert!(state.prepare(0, 1).is_err());
    }

    #[stark_engine_macros::stark_test]
    fn exact_context_capacity_is_usable_and_overflow_is_rejected() {
        let mut state = Qwen38State::<Cpu>::new(Vec::new(), 2);
        assert!(state.prepare(u32::MAX, 1).is_err());
        state.begin_encode(2).unwrap();
        state.finish_encode();
        state.encode_accept(&[0, 1], &mut commands()).unwrap();
        assert_eq!(state.context_length(), 2);
        assert!(state.begin_encode(1).is_err());
    }

    #[stark_engine_macros::stark_test]
    fn full_64k_capacity_commits_the_last_token_once_without_wrapping() {
        let mut state = Qwen38State::<Cpu>::new(Vec::new(), 65_536);
        state.begin_encode(65_535).unwrap();
        state.finish_encode();
        state.encode_accept(&(0..65_535).collect::<Vec<_>>(), &mut commands()).unwrap();
        assert_eq!(state.context_length(), 65_535);
        state.begin_encode(1).unwrap();
        state.finish_encode();
        state.encode_accept(&[0], &mut commands()).unwrap();
        assert_eq!(state.context_length(), 65_536);
        assert!(state.prepare(65_536, 1).is_err());
        assert!(state.prepare(u32::MAX, 1).is_err());
        assert!(state.begin_encode(1).is_err());
        assert_eq!(state.context_length(), 65_536);
    }
}
