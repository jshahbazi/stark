use half::bf16;
use num_traits::{Float, NumCast};
use stark_engine_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ring::RingParams};

#[kernel(RepetitionPenalty)]
#[variants(T, f32, bf16)]
pub fn repetition_penalty<T: ArrayElement + Float + NumCast>(
    original_logits: *const T,
    logits_copy: *mut T,
    context_ring: *const u32,
    token_ids: *const u32,
    repetition_penalty: f32,
    suffix_repetition_length: u32,
    vocab_size: u32,
    sampling_start: u32,
    sampling_length: u32,
) {
    fn apply(
        original_logits: *const (impl ArrayElement + Float),
        logits_copy: *mut (impl ArrayElement + Float),
        vocab_size: usize,
        sample_index: usize,
        token_id: usize,
        repetition_penalty: f32,
    ) {
        if token_id >= vocab_size {
            return;
        }
        let offset = sample_index * vocab_size + token_id;
        let logit = unsafe { (*original_logits.add(offset)).to_f32().unwrap() };
        let penalized = if logit > 0.0 {
            logit / repetition_penalty
        } else {
            logit * repetition_penalty
        };
        unsafe {
            *logits_copy.add(offset) = NumCast::from(penalized).unwrap();
        }
    }

    let ring = unsafe { &*context_ring.cast::<RingParams>() };
    let ring_tokens = unsafe { context_ring.add(2) };
    let vocab_size = vocab_size as usize;

    for sample_index in 0..sampling_length {
        let batch_prefix_length = sampling_start + sample_index + 1;
        let total_length = ring.ring_length + batch_prefix_length;
        let window_length = total_length.min(suffix_repetition_length);
        let skipped_length = total_length - window_length;

        for window_index in 0..window_length {
            let source_index = skipped_length + window_index;
            let token_id = if source_index < ring.ring_length {
                let slot = (ring.ring_offset + source_index) % suffix_repetition_length;
                (unsafe { *ring_tokens.add(slot as usize) }) as usize
            } else {
                (unsafe { *token_ids.add((source_index - ring.ring_length) as usize) }) as usize
            };
            apply(original_logits, logits_copy, vocab_size, sample_index as usize, token_id, repetition_penalty);
        }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::repetition_penalty;

    #[stark_test]
    fn repetition_penalty_ignores_out_of_vocabulary_indices() {
        let logits = [2.0f32, 4.0];
        let ring = [0u32, 0];
        for token in [2, 3, u32::MAX] {
            let mut copy = logits;
            repetition_penalty(logits.as_ptr(), copy.as_mut_ptr(), ring.as_ptr(), [token].as_ptr(), 2.0, 1, 2, 0, 1);
            assert_eq!(copy, logits);
        }
        let mut copy = logits;
        repetition_penalty(logits.as_ptr(), copy.as_mut_ptr(), ring.as_ptr(), [1].as_ptr(), 2.0, 1, 2, 0, 1);
        assert_eq!(copy, [2.0, 2.0]);
        let retained = [0u32, 1, u32::MAX];
        repetition_penalty(logits.as_ptr(), copy.as_mut_ptr(), retained.as_ptr(), [0].as_ptr(), 2.0, 2, 2, 0, 1);
        assert_eq!(copy, [1.0, 2.0]);
    }
}
