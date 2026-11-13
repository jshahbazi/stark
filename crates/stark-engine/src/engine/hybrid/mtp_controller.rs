//! Backend-neutral, synchronous MTP control foundation. No backend capability
//! is enabled by this module. Native integration must supply complete private,
//! model/policy-bound target AND draft checkpoints before using this stream.
//!
//! The initial target verifier uses ordinary single-token steps. The sampler
//! owns exact stochastic rejection and its independent draft randomness; this
//! controller makes no same-seed MTP-on/off identity claim.

#![allow(dead_code, reason = "model-free MTP foundation awaiting qualified native-head integration")]

use std::{collections::VecDeque, time::Instant};

use super::{Error, MtpSampler, MtpStats, Result};

/// A loaded backend borrowed by one serialized stream. All methods are
/// synchronous: errors must drain submitted work before returning or fencing.
pub(super) trait MtpBackend {
    type Checkpoint;

    fn vocab_size(&self) -> usize;
    fn is_fenced(&self) -> bool;
    fn checkpoint(&mut self) -> Result<Self::Checkpoint>;

    /// Produce the next complete-vocabulary proposal distribution. `depth`
    /// starts at zero, with the preceding stream's pending token as `input`.
    /// Speculative mutations may affect target and draft owners; verification
    /// always rewinds to the round checkpoint before any target step.
    fn draft_logits(
        &mut self,
        input: u32,
        depth: usize,
    ) -> Result<Vec<f32>>;

    /// An ordinary serial target step consuming exactly `input`.
    fn target_logits(
        &mut self,
        input: u32,
    ) -> Result<Vec<f32>>;

    /// Restore the COMPLETE target/draft checkpoint, then commit exactly this
    /// causal input prefix. Replay must not sample or use an RNG. This includes
    /// draft-cache alignment, position/history, recurrent/convolution, attention
    /// and PLE state. Reject stale/cross-model/policy checkpoints. Backends may
    /// optimize restoration only when the resulting state is identical.
    fn rewind(
        &mut self,
        checkpoint: &Self::Checkpoint,
        committed_inputs: &[u32],
    ) -> Result<()>;

    /// Infallibly invalidate the loaded execution state and all checkpoints.
    /// This must be idempotent. Subsequent inference must reject until explicit
    /// recovery outside this stream.
    fn fence(&mut self);
}

pub(super) struct MtpDecode<'a, B: MtpBackend + ?Sized> {
    backend: &'a mut B,
    sampler: &'a mut dyn MtpSampler,
    stats: &'a mut MtpStats,
    vocab: usize,
    pending: u32,
    remaining: usize,
    drafts: usize,
    checkpoint: Option<B::Checkpoint>,
    committed_inputs: Vec<u32>,
    queue: VecDeque<u32>,
    delivered: usize,
    stopped: bool,
    failed: bool,
}

impl<'a, B: MtpBackend + ?Sized> MtpDecode<'a, B> {
    /// `first` has already been sampled but has not been consumed by the
    /// backend. `remaining` counts additional outputs after `first`. Context
    /// capacity and initial-readout provenance remain backend admission duties.
    pub(super) fn new(
        backend: &'a mut B,
        sampler: &'a mut dyn MtpSampler,
        stats: &'a mut MtpStats,
        first: u32,
        remaining: usize,
        drafts: usize,
    ) -> Result<Self> {
        let vocab = backend.vocab_size();
        if backend.is_fenced() || vocab == 0 || vocab > u32::MAX as usize || first as usize >= vocab || drafts > 3 {
            backend.fence();
            return Err(Error::Invalid("MTP controller admission"));
        }
        Ok(Self {
            backend,
            sampler,
            stats,
            vocab,
            pending: first,
            remaining,
            drafts,
            checkpoint: None,
            committed_inputs: Vec::new(),
            queue: VecDeque::new(),
            delivered: 0,
            stopped: false,
            failed: false,
        })
    }

    fn readout(
        &self,
        logits: &[f32],
    ) -> Result<()> {
        if logits.len() != self.vocab || logits.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("MTP controller readout"));
        }
        Ok(())
    }

    fn token(
        &self,
        token: u32,
    ) -> Result<()> {
        if token as usize >= self.vocab {
            return Err(Error::Invalid("MTP controller output ID"));
        }
        Ok(())
    }

    fn invalidate(&mut self) {
        let needs_fence = !self.failed;
        self.failed = true;
        self.stopped = true;
        self.queue.clear();
        self.checkpoint = None;
        self.committed_inputs.clear();
        if needs_fence {
            self.backend.fence();
        }
    }

    fn rewind(
        &mut self,
        inputs: &[u32],
    ) -> Result<()> {
        let checkpoint = self.checkpoint.as_ref().ok_or(Error::Invalid("MTP controller checkpoint missing"))?;
        let started = Instant::now();
        let result = self.backend.rewind(checkpoint, inputs);
        self.stats.rollback_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn round(&mut self) -> Result<()> {
        self.checkpoint = None;
        self.committed_inputs.clear();
        self.delivered = 0;
        self.checkpoint = Some(self.backend.checkpoint()?);
        self.sampler.begin_round();
        let count = self.drafts.min(self.remaining.saturating_sub(1));
        let mut proposals = Vec::with_capacity(count);
        let started = Instant::now();
        let draft_result: Result<()> = (|| {
            let mut input = self.pending;
            for depth in 0..count {
                let logits = self.backend.draft_logits(input, depth)?;
                self.readout(&logits)?;
                let proposal = self.sampler.draft(&logits)?;
                self.token(proposal)?;
                proposals.push(proposal);
                input = proposal;
            }
            Ok(())
        })();
        self.stats.draft_seconds += started.elapsed().as_secs_f64();
        draft_result?;
        if count > 0 {
            self.rewind(&[])?;
        }
        let started = Instant::now();
        let mut outputs = VecDeque::with_capacity(count + 1);
        let mut inputs = Vec::with_capacity(count + 1);
        let mut accepted = 0;
        let verify_result: Result<()> = (|| {
            let mut input = self.pending;
            for index in 0..=count {
                let logits = self.backend.target_logits(input)?;
                self.readout(&logits)?;
                inputs.push(input);
                let proposal = proposals.get(index).copied();
                let selected = self.sampler.target(&logits, proposal)?;
                self.token(selected)?;
                outputs.push_back(selected);
                if proposal != Some(selected) {
                    break;
                }
                accepted += 1;
                input = selected;
            }
            Ok(())
        })();
        self.stats.verify_seconds += started.elapsed().as_secs_f64();
        verify_result?;
        // The last delivered output remains pending, exactly as ordinary
        // generation. Rewind also removes all unaccepted draft-cache effects.
        self.rewind(&inputs)?;
        self.pending = *outputs.back().ok_or(Error::Invalid("MTP controller empty round"))?;
        self.remaining -= outputs.len();
        self.stats.rounds += 1;
        self.stats.proposed += proposals.len();
        self.stats.accepted += accepted;
        // Only serial target readouts count. Rewind replay is excluded.
        self.stats.verified_rows += inputs.len();
        self.committed_inputs = inputs;
        self.queue = outputs;
        Ok(())
    }

    /// Stop at the last delivered output, acknowledging cleanup failures.
    /// Call before declaring a collector successful; Drop can only fence a
    /// cleanup failure, which the backend's owning request must also inspect.
    pub(super) fn finish(&mut self) -> Result<()> {
        if self.failed || self.backend.is_fenced() {
            self.invalidate();
            return Err(Error::Invalid("MTP controller fenced"));
        }
        if !self.queue.is_empty() {
            let inputs = self.committed_inputs[..self.delivered].to_vec();
            if let Err(error) = self.rewind(&inputs) {
                self.invalidate();
                return Err(error);
            }
        }
        self.queue.clear();
        self.checkpoint = None;
        self.committed_inputs.clear();
        self.stopped = true;
        Ok(())
    }

    /// Restore the delivered boundary, then invalidate the cancelled request.
    /// Neither this method nor finish/Drop invokes any sampler callback.
    pub(super) fn cancel(&mut self) -> Result<()> {
        let result = self.finish();
        self.invalidate();
        result
    }
}

impl<B: MtpBackend + ?Sized> Iterator for MtpDecode<'_, B> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped {
            return None;
        }
        if self.backend.is_fenced() {
            self.invalidate();
            return Some(Err(Error::Invalid("MTP controller fenced")));
        }
        if self.queue.is_empty() {
            if self.remaining == 0 {
                self.checkpoint = None;
                self.stopped = true;
                return None;
            }
            if let Err(error) = self.round() {
                self.invalidate();
                return Some(Err(error));
            }
        }
        let token = self.queue.pop_front()?;
        self.delivered += 1;
        Some(Ok(token))
    }
}

impl<B: MtpBackend + ?Sized> Drop for MtpDecode<'_, B> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use stark_engine_macros::stark_test;

    use super::*;

    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        Checkpoint,
        Draft,
        Target,
        TargetAt(usize),
        Rewind(usize),
        DraftShape,
        TargetShape,
        DraftNonfinite,
        TargetNonfinite,
    }

    struct Backend {
        target: Vec<u32>,
        draft: Vec<u32>,
        fenced: bool,
        fault: Fault,
        rewinds: usize,
        target_calls: usize,
        draft_calls: usize,
        first_fenced_state: Option<(Vec<u32>, Vec<u32>)>,
    }

    impl Backend {
        fn new(fault: Fault) -> Self {
            Self {
                target: Vec::new(),
                draft: Vec::new(),
                fenced: false,
                fault,
                rewinds: 0,
                target_calls: 0,
                draft_calls: 0,
                first_fenced_state: None,
            }
        }
    }

    impl MtpBackend for Backend {
        type Checkpoint = (Vec<u32>, Vec<u32>);

        fn vocab_size(&self) -> usize {
            8
        }
        fn is_fenced(&self) -> bool {
            self.fenced
        }
        fn checkpoint(&mut self) -> Result<Self::Checkpoint> {
            if self.fault == Fault::Checkpoint {
                return Err(Error::Invalid("fake checkpoint fault"));
            }
            Ok((self.target.clone(), self.draft.clone()))
        }
        fn draft_logits(
            &mut self,
            input: u32,
            _depth: usize,
        ) -> Result<Vec<f32>> {
            self.draft_calls += 1;
            self.draft.push(input);
            // Deliberately dirty BOTH owners to require complete restoration.
            self.target.push(7);
            match self.fault {
                Fault::Draft => Err(Error::Invalid("fake draft fault")),
                Fault::DraftShape => Ok(vec![0.0; 7]),
                Fault::DraftNonfinite => Ok(vec![f32::NAN; 8]),
                _ => Ok(vec![1.0; 8]),
            }
        }
        fn target_logits(
            &mut self,
            input: u32,
        ) -> Result<Vec<f32>> {
            self.target_calls += 1;
            self.target.push(input);
            if self.fault == Fault::TargetAt(self.target_calls) {
                return Err(Error::Invalid("fake later target fault"));
            }
            match self.fault {
                Fault::Target => Err(Error::Invalid("fake target fault")),
                Fault::TargetShape => Ok(vec![]),
                Fault::TargetNonfinite => Ok(vec![f32::INFINITY; 8]),
                _ => Ok(vec![2.0; 8]),
            }
        }
        fn rewind(
            &mut self,
            checkpoint: &Self::Checkpoint,
            inputs: &[u32],
        ) -> Result<()> {
            self.rewinds += 1;
            if self.fault == Fault::Rewind(self.rewinds) {
                return Err(Error::Invalid("fake rewind fault"));
            }
            self.target = checkpoint.0.clone();
            self.draft = checkpoint.1.clone();
            self.target.extend_from_slice(inputs);
            self.draft.extend_from_slice(inputs);
            Ok(())
        }
        fn fence(&mut self) {
            if !self.fenced {
                self.first_fenced_state = Some((self.target.clone(), self.draft.clone()));
            }
            self.fenced = true;
            self.target.clear();
            self.draft.clear();
        }
    }

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    struct Calls {
        begin: usize,
        draft: usize,
        target: usize,
    }

    struct Sampler {
        proposals: VecDeque<u32>,
        outputs: VecDeque<u32>,
        calls: Rc<RefCell<Calls>>,
        draft_error: bool,
        target_error: bool,
        seen_proposals: Vec<Option<u32>>,
    }

    impl Sampler {
        fn new(
            proposals: &[u32],
            outputs: &[u32],
        ) -> Self {
            Self {
                proposals: proposals.iter().copied().collect(),
                outputs: outputs.iter().copied().collect(),
                calls: Rc::new(RefCell::new(Calls::default())),
                draft_error: false,
                target_error: false,
                seen_proposals: Vec::new(),
            }
        }
    }

    impl MtpSampler for Sampler {
        fn begin_round(&mut self) {
            self.calls.borrow_mut().begin += 1;
        }
        fn draft(
            &mut self,
            logits: &[f32],
        ) -> Result<u32> {
            assert_eq!(logits, &[1.0; 8]);
            self.calls.borrow_mut().draft += 1;
            if self.draft_error {
                return Err(Error::Invalid("fake sampler draft fault"));
            }
            self.proposals.pop_front().ok_or(Error::Invalid("fake proposals exhausted"))
        }
        fn target(
            &mut self,
            logits: &[f32],
            proposal: Option<u32>,
        ) -> Result<u32> {
            assert_eq!(logits, &[2.0; 8]);
            self.calls.borrow_mut().target += 1;
            self.seen_proposals.push(proposal);
            if self.target_error {
                return Err(Error::Invalid("fake sampler target fault"));
            }
            self.outputs.pop_front().ok_or(Error::Invalid("fake outputs exhausted"))
        }
    }

    #[stark_test]
    fn mtp_controller_every_proposal_and_acceptance_length() {
        for drafts in 0..=3 {
            for accepted in 0..=drafts {
                let proposals = [1, 2, 3];
                let mut expected = proposals[..accepted].to_vec();
                expected.push(5);
                let mut backend = Backend::new(Fault::None);
                let mut sampler = Sampler::new(&proposals[..drafts], &expected);
                let mut stats = MtpStats::default();
                let delivered = {
                    let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 4, drafts).unwrap();
                    let delivered = stream.by_ref().take(expected.len()).collect::<Result<Vec<_>>>().unwrap();
                    stream.finish().unwrap();
                    delivered
                };
                assert_eq!(delivered, expected);
                let mut expected_state = vec![0];
                expected_state.extend_from_slice(&proposals[..accepted]);
                assert_eq!(backend.target, expected_state);
                assert_eq!(backend.draft, expected_state);
                assert!(!backend.fenced);
                assert_eq!(backend.target_calls, accepted + 1);
                assert_eq!(stats.proposed, drafts);
                assert_eq!(stats.accepted, accepted);
                assert_eq!(stats.verified_rows, accepted + 1);
                assert_eq!(
                    sampler.seen_proposals,
                    (0..=accepted).map(|index| proposals[..drafts].get(index).copied()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[stark_test]
    fn mtp_controller_budget_tails_and_multiple_rounds() {
        for budget in [0, 1, 5] {
            let mut backend = Backend::new(Fault::None);
            let mut sampler = Sampler::new(&[1, 2, 3], &[1, 2, 3, 4, 5]);
            let mut stats = MtpStats::default();
            let outputs = {
                let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, budget, 3).unwrap();
                let outputs = stream.by_ref().collect::<Result<Vec<_>>>().unwrap();
                stream.finish().unwrap();
                outputs
            };
            assert_eq!(outputs, (1..=budget as u32).collect::<Vec<_>>());
            assert_eq!(backend.target, (0..budget as u32).collect::<Vec<_>>());
            assert_eq!(backend.draft, backend.target);
            assert_eq!(
                backend.draft_calls,
                if budget == 5 {
                    3
                } else {
                    0
                }
            );
            assert_eq!(backend.target_calls, budget);
            assert_eq!(stats.verified_rows, budget);
            assert_eq!(sampler.calls.borrow().target, budget);
            assert_eq!(
                sampler.calls.borrow().begin,
                if budget == 5 {
                    2
                } else {
                    budget
                }
            );
        }
    }

    #[stark_test]
    fn mtp_controller_partial_delivery_drop_and_finish_never_sample_cleanup() {
        for delivered in 0..=4 {
            for explicit in [false, true] {
                let mut backend = Backend::new(Fault::None);
                let mut sampler = Sampler::new(&[1, 2, 3], &[1, 2, 3, 4]);
                let calls = Rc::clone(&sampler.calls);
                let mut stats = MtpStats::default();
                let before;
                {
                    let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 4, 3).unwrap();
                    let outputs = stream.by_ref().take(delivered).collect::<Result<Vec<_>>>().unwrap();
                    assert_eq!(outputs, (1..=delivered as u32).collect::<Vec<_>>());
                    before = *calls.borrow();
                    if explicit {
                        stream.finish().unwrap();
                        assert!(stream.next().is_none());
                    }
                }
                assert_eq!(*calls.borrow(), before);
                assert_eq!(backend.target, (0..delivered as u32).collect::<Vec<_>>());
                assert_eq!(backend.draft, backend.target);
                assert!(!backend.fenced);
            }
        }
    }

    #[stark_test]
    fn mtp_controller_rejection_then_next_round_consumes_correction_once() {
        let mut backend = Backend::new(Fault::None);
        let mut sampler = Sampler::new(&[1, 2, 6], &[5, 6, 7]);
        let mut stats = MtpStats::default();
        let outputs = {
            let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 3, 3).unwrap();
            let outputs = stream.by_ref().collect::<Result<Vec<_>>>().unwrap();
            stream.finish().unwrap();
            outputs
        };
        assert_eq!(outputs, [5, 6, 7]);
        assert_eq!(backend.target, [0, 5, 6]);
        assert_eq!(backend.draft, [0, 5, 6]);
        assert_eq!(sampler.seen_proposals, [Some(1), Some(6), None]);
        assert_eq!(stats.rounds, 2);
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.proposed, 3);
        assert_eq!(stats.verified_rows, 3);
        assert_eq!(backend.target_calls, 3);
    }

    #[stark_test]
    fn mtp_controller_cancel_restores_then_fences_without_sampling() {
        for delivered in 0..=3 {
            let mut backend = Backend::new(Fault::None);
            let mut sampler = Sampler::new(&[1, 2, 3], &[1, 2, 3, 4]);
            let calls = Rc::clone(&sampler.calls);
            let mut stats = MtpStats::default();
            let before;
            {
                let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 4, 3).unwrap();
                for expected in 1..=delivered {
                    assert_eq!(stream.next().unwrap().unwrap(), expected);
                }
                before = *calls.borrow();
                stream.cancel().unwrap();
                assert!(stream.next().is_none());
                assert!(stream.finish().is_err());
            }
            assert_eq!(*calls.borrow(), before);
            assert!(backend.fenced);
            assert!(backend.target.is_empty() && backend.draft.is_empty());
            let expected_state = (0..delivered).collect::<Vec<_>>();
            assert_eq!(backend.first_fenced_state, Some((expected_state.clone(), expected_state)));
        }
    }

    #[stark_test]
    fn mtp_controller_backend_and_malformed_readout_errors_fence() {
        for fault in [
            Fault::Checkpoint,
            Fault::Draft,
            Fault::Target,
            Fault::TargetAt(2),
            Fault::Rewind(1),
            Fault::Rewind(2),
            Fault::DraftShape,
            Fault::TargetShape,
            Fault::DraftNonfinite,
            Fault::TargetNonfinite,
        ] {
            let mut backend = Backend::new(fault);
            let mut sampler = Sampler::new(&[1, 2, 3], &[1, 2, 3, 4]);
            let mut stats = MtpStats::default();
            {
                let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 4, 3).unwrap();
                assert!(stream.next().unwrap().is_err());
                assert!(stream.next().is_none());
                assert!(stream.finish().is_err());
            }
            assert!(backend.fenced);
            assert!(backend.target.is_empty() && backend.draft.is_empty());
            assert_eq!(stats.rounds, 0);
            if matches!(fault, Fault::DraftShape | Fault::DraftNonfinite) {
                assert_eq!(sampler.calls.borrow().draft, 0);
                assert_eq!(sampler.calls.borrow().target, 0);
            }
            if matches!(fault, Fault::TargetShape | Fault::TargetNonfinite) {
                assert_eq!(sampler.calls.borrow().target, 0);
            }
        }
    }

    #[stark_test]
    fn mtp_controller_sampler_errors_and_invalid_output_ids_fence() {
        for case in 0..4 {
            let mut backend = Backend::new(Fault::None);
            let mut sampler = Sampler::new(
                if case == 2 {
                    &[8]
                } else {
                    &[1, 2, 3]
                },
                if case == 3 {
                    &[8]
                } else {
                    &[1, 2, 3, 4]
                },
            );
            sampler.draft_error = case == 0;
            sampler.target_error = case == 1;
            let mut stats = MtpStats::default();
            {
                let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 4, 3).unwrap();
                assert!(stream.next().unwrap().is_err());
                assert!(stream.finish().is_err());
            }
            assert!(backend.fenced);
            assert!(backend.target.is_empty() && backend.draft.is_empty());
        }
    }

    #[stark_test]
    fn mtp_controller_invalid_admission_never_samples() {
        for (first, drafts, fenced) in [(8, 3, false), (0, 4, false), (0, 3, true)] {
            let mut backend = Backend::new(Fault::None);
            backend.fenced = fenced;
            let mut sampler = Sampler::new(&[], &[]);
            let mut stats = MtpStats::default();
            assert!(MtpDecode::new(&mut backend, &mut sampler, &mut stats, first, 4, drafts).is_err());
            assert_eq!(*sampler.calls.borrow(), Calls::default());
            assert!(backend.fenced);
            assert!(backend.target.is_empty() && backend.draft.is_empty());
        }
    }

    #[stark_test]
    fn mtp_controller_cleanup_failure_is_acknowledged_and_drop_fences() {
        for explicit in [false, true] {
            let mut backend = Backend::new(Fault::Rewind(3));
            let mut sampler = Sampler::new(&[1, 2, 3], &[1, 2, 3, 4]);
            let calls = Rc::clone(&sampler.calls);
            let mut stats = MtpStats::default();
            let before;
            {
                let mut stream = MtpDecode::new(&mut backend, &mut sampler, &mut stats, 0, 4, 3).unwrap();
                assert_eq!(stream.next().unwrap().unwrap(), 1);
                before = *calls.borrow();
                if explicit {
                    assert!(stream.finish().is_err());
                }
            }
            assert_eq!(*calls.borrow(), before);
            assert!(backend.fenced);
            assert!(backend.target.is_empty() && backend.draft.is_empty());
        }
    }
}
