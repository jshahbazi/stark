//! Checked allocation admission for automatic contexts, without allocating state.

const SCRATCH_CLASSES: usize = 32;
const MAX_SCRATCH_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Persistent payload and conservative cached scratch pages for one request.
/// The scratch allocator caches power-of-two pages across sequential layers.
#[derive(Clone, Debug, Default)]
pub struct MemoryEstimate {
    pub state_bytes: usize,
    pub scratch_bytes: usize,
    pub max_buffer_bytes: usize,
    pub prefix_snapshot_bytes: usize,
    state_buffer_bytes: usize,
    scratch_pages: [usize; SCRATCH_CLASSES],
}

pub fn checked_bytes(
    shape: &[u32],
    element_bytes: usize,
) -> Option<usize> {
    shape
        .iter()
        .try_fold(element_bytes, |size, &dimension| size.checked_mul(dimension as usize))
        .filter(|&size| size <= isize::MAX as usize)
}

/// Sparse finite caches reserve address space for the selected context, but
/// only commit rows reached by this request. Dense and undeclared contexts
/// continue to admit their entire selected allocation.
pub fn generation_allocation_context(
    input_tokens: u32,
    context_tokens: u32,
    max_new_tokens: u32,
    sparse_finite_context: bool,
) -> Option<u32> {
    let requested = input_tokens.checked_add(max_new_tokens)?;
    if input_tokens == 0 || max_new_tokens == 0 || requested > context_tokens {
        return None;
    }
    Some(if sparse_finite_context {
        requested
    } else {
        context_tokens
    })
}

pub fn attention_estimate(
    context_tokens: u32,
    input_tokens: u32,
    head_dim: u32,
    query_heads: u32,
    kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    projection_dim: u32,
    element_bytes: usize,
    reference: bool,
) -> Option<MemoryEstimate> {
    let suffix = crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY;
    let cache_rows = ring_capacity.map_or(context_tokens, |window| window.min(context_tokens)).checked_add(suffix)?;
    let state_bytes = match kv_heads {
        Some(heads) => {
            // attention_prepare.metal still computes K/V element offsets in
            // uint32_t, even when the device accepts a larger byte buffer.
            cache_rows.checked_mul(heads)?.checked_mul(head_dim)?;
            checked_bytes(&[cache_rows, heads, head_dim, 2], element_bytes)?
        },
        None => 0, // Shared KV is accounted by its owning transformer layer.
    };
    let groups = kv_heads.unwrap_or(1).max(1);
    if query_heads == 0 || !query_heads.is_multiple_of(groups) {
        return None;
    }
    let scratch = |rows: u32, sequence_rows: u32| {
        let mut estimate = MemoryEstimate::default();
        estimate.add_scratch(&[rows, projection_dim], element_bytes, 2)?;
        estimate.add_scratch(&[rows, query_heads, head_dim], element_bytes, 3)?;
        estimate.add_scratch(&[rows, groups, head_dim], element_bytes, 2)?;
        if reference {
            estimate.add_scratch(&[rows, sequence_rows], 4, 1)?;
            estimate.add_scratch(&[rows, sequence_rows], 2, 1)?;
            estimate.add_scratch(&[rows, head_dim], 4, 1)?;
            estimate.add_scratch(&[rows, head_dim], 2, 1)?;
        } else {
            // Includes the untiled fallback, even when the selected Metal
            // kernel uses less workspace for the same geometry.
            estimate.add_scratch(&[query_heads, rows, sequence_rows], element_bytes, 1)?;
            estimate.add_scratch(&[query_heads / groups, rows, sequence_rows], element_bytes, 1)?;
            estimate.add_scratch(&[query_heads / groups, rows, head_dim], element_bytes, 1)?;
        }
        Some(estimate)
    };
    let prefill_rows = input_tokens.clamp(1, suffix);
    let prefill_sequence = input_tokens.max(1).min(cache_rows);
    let decode_sequence = context_tokens.min(cache_rows);
    let mut result = scratch(prefill_rows, prefill_sequence)?.sequential(scratch(1, decode_sequence)?)?;
    result.set_state_buffers(state_bytes, state_bytes / 2);
    Some(result)
}

impl MemoryEstimate {
    pub fn state(state_bytes: usize) -> Self {
        Self {
            state_bytes,
            max_buffer_bytes: state_bytes,
            state_buffer_bytes: state_bytes,
            ..Self::default()
        }
    }

    pub fn set_state_buffers(
        &mut self,
        state_bytes: usize,
        max_buffer_bytes: usize,
    ) {
        self.state_bytes = state_bytes;
        self.state_buffer_bytes = max_buffer_bytes;
        let scratch_max = self.scratch_pages.iter().rposition(|&pages| pages != 0).map_or(0, |class| 1usize << class);
        self.max_buffer_bytes = max_buffer_bytes.max(scratch_max);
    }

    pub fn without_state(mut self) -> Self {
        self.set_state_buffers(0, 0);
        self.prefix_snapshot_bytes = 0;
        self
    }

    pub fn add_state_buffers(
        &mut self,
        state_bytes: usize,
        max_buffer_bytes: usize,
    ) -> Option<()> {
        self.set_state_buffers(
            self.state_bytes.checked_add(state_bytes)?,
            self.state_buffer_bytes.max(max_buffer_bytes),
        );
        Some(())
    }

    pub fn add_scratch(
        &mut self,
        shape: &[u32],
        element_bytes: usize,
        copies: usize,
    ) -> Option<()> {
        let bytes = checked_bytes(shape, element_bytes)?;
        if bytes == 0 || copies == 0 {
            return Some(());
        }
        let page = bytes.checked_next_power_of_two().filter(|&size| size <= MAX_SCRATCH_BYTES)?;
        let class = page.trailing_zeros() as usize;
        self.scratch_pages[class] = self.scratch_pages[class].checked_add(copies)?;
        self.scratch_bytes = self.scratch_bytes.checked_add(page.checked_mul(copies)?)?;
        self.max_buffer_bytes = self.max_buffer_bytes.max(page);
        Some(())
    }

    /// Sequential work can reuse released pages, but each size class remains
    /// resident in the pool, so max total bytes alone would undercount it.
    pub fn sequential(
        mut self,
        other: Self,
    ) -> Option<Self> {
        self.state_bytes = self.state_bytes.checked_add(other.state_bytes)?;
        self.prefix_snapshot_bytes = self.prefix_snapshot_bytes.checked_add(other.prefix_snapshot_bytes)?;
        self.state_buffer_bytes = self.state_buffer_bytes.max(other.state_buffer_bytes);
        self.max_buffer_bytes = self.max_buffer_bytes.max(other.max_buffer_bytes);
        self.scratch_bytes = 0;
        for class in 0..SCRATCH_CLASSES {
            self.scratch_pages[class] = self.scratch_pages[class].max(other.scratch_pages[class]);
            self.scratch_bytes =
                self.scratch_bytes.checked_add((1usize << class).checked_mul(self.scratch_pages[class])?)?;
        }
        Some(self)
    }

    pub fn total_bytes(&self) -> Option<usize> {
        self.state_bytes.checked_add(self.scratch_bytes)
    }

    /// Scoring submits one independently owned scratch pool per prompt chunk
    /// and retains every pending submission until the complete prompt is
    /// encoded. Its peak therefore includes all chunk pools simultaneously.
    pub fn scoring_total_bytes(
        &self,
        input_tokens: u32,
        separate_final_row: bool,
    ) -> Option<usize> {
        if input_tokens == 0 {
            return None;
        }
        let chunks = input_tokens.div_ceil(1024) as usize;
        // A shorter last chunk or the 1023+1 readout split can populate an
        // additional power-of-two page class. Earlier complete chunks use the
        // profiled geometry; only the final pool needs a second inventory.
        let last_rows = (input_tokens - 1) % 1024 + 1;
        let extra_final = (input_tokens > 1024 && last_rows < 1024) || (separate_final_row && last_rows > 1);
        self.state_bytes
            .checked_add(self.scratch_bytes.checked_mul(chunks.checked_add(usize::from(extra_final))?)?)?
            .checked_add((1024 * 1024usize).checked_mul(chunks)?)
    }
}

/// Combine dense MTP's simultaneously retained target/predictor states and
/// its bounded residual/logit handoff. Transformer work runs sequentially.
pub fn dense_mtp_estimate(
    target: MemoryEstimate,
    predictor: MemoryEstimate,
    input_tokens: u32,
    model_dim: u32,
    vocab_size: u32,
) -> Option<MemoryEstimate> {
    let rows = input_tokens.clamp(4, 1024);
    let snapshot_bytes = target.prefix_snapshot_bytes.checked_add(predictor.prefix_snapshot_bytes)?;
    let mut estimate = target.sequential(predictor)?;
    let row_bytes = checked_bytes(&[model_dim], 2)?;
    let chunk_bytes = checked_bytes(&[rows, model_dim], 2)?;
    // TargetPass.hidden and the shifted priming residual coexist; keep last
    // hidden, prefix hidden and their in-flight replacements as distinct rows.
    let hidden_bytes = chunk_bytes.checked_mul(2)?.checked_add(row_bytes.checked_mul(4)?)?;
    estimate.add_state_buffers(hidden_bytes, chunk_bytes.max(row_bytes))?;
    // Replacing a retained boundary constructs the new snapshot before the
    // previous snapshot is dropped, so both payloads can coexist briefly.
    estimate.add_state_buffers(snapshot_bytes.checked_mul(2)?, estimate.max_buffer_bytes)?;
    // Session and prefix logits, up to four verify rows, plus the temporary
    // BF16-to-F32 copyout: eight full-F32 row equivalents at peak.
    estimate.state_bytes = estimate.state_bytes.checked_add(checked_bytes(&[8, vocab_size], 4)?)?;
    estimate.add_scratch(&[4, vocab_size], 2, 1)?;
    Some(estimate)
}

/// Find the greatest admitted context while preserving room for one output.
pub fn automatic_context(
    input_tokens: u32,
    maximum: u32,
    memory_budget_bytes: usize,
    mut estimate: impl FnMut(u32) -> Option<MemoryEstimate>,
) -> Option<u32> {
    let mut low = input_tokens.checked_add(1)?;
    if low > maximum || estimate(low)?.total_bytes()? > memory_budget_bytes {
        return None;
    }
    let mut high = maximum;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if estimate(middle).and_then(|value| value.total_bytes()).is_some_and(|bytes| bytes <= memory_budget_bytes) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Some(low)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn sequential_layers_reuse_pages_and_retain_distinct_size_classes() {
        let mut first = MemoryEstimate::state(10);
        first.add_scratch(&[17], 1, 2).unwrap();
        let mut next = MemoryEstimate::state(20);
        next.add_scratch(&[17], 1, 1).unwrap();
        next.add_scratch(&[33], 1, 1).unwrap();
        let combined = first.sequential(next).unwrap();
        assert_eq!(combined.state_bytes, 30);
        assert_eq!(combined.scratch_bytes, 2 * 32 + 64);
        let shared = MemoryEstimate::state(1000).without_state();
        assert_eq!(combined.sequential(shared).unwrap().state_bytes, 30);
    }

    #[stark_test]
    fn automatic_context_finds_exact_boundary_and_rejects_overflow() {
        let profile = |context: u32| Some(MemoryEstimate::state(context as usize * 8));
        assert_eq!(automatic_context(20, 1000, 800, profile), Some(100));
        assert_eq!(automatic_context(100, 1000, 800, profile), None);
        assert_eq!(automatic_context(u32::MAX, u32::MAX, usize::MAX, profile), None);
        assert_eq!(automatic_context(20, 100, usize::MAX, |_| None), None);
        assert!(checked_bytes(&[u32::MAX, u32::MAX], 8).is_none());
    }

    #[stark_test]
    fn finite_sparse_admission_counts_requested_growth_and_dense_reservation() {
        let profile = |context: u32| Some(MemoryEstimate::state(context as usize * 8));
        let short = generation_allocation_context(10, 1000, 10, true).unwrap();
        assert_eq!(short, 20);
        assert_eq!(automatic_context(10, short, 800, profile), Some(short));
        let growing = generation_allocation_context(10, 1000, 500, true).unwrap();
        assert_eq!(automatic_context(10, growing, 800, profile), Some(100));
        let dense = generation_allocation_context(10, 1000, 10, false).unwrap();
        assert_eq!(dense, 1000);
        assert_eq!(automatic_context(10, dense, 800, profile), Some(100));
        assert!(generation_allocation_context(u32::MAX, u32::MAX, 1, true).is_none());
        assert!(generation_allocation_context(10, 20, 11, true).is_none());
    }

    #[stark_test]
    fn scoring_admission_retains_all_chunk_pools_and_allows_full_prompt_context() {
        let mut estimate = MemoryEstimate::state(100);
        estimate.add_scratch(&[17], 1, 2).unwrap();
        let one = estimate.scoring_total_bytes(1024, false).unwrap();
        let two = estimate.scoring_total_bytes(1025, false).unwrap();
        assert_eq!(one, 100 + 64 + 1024 * 1024);
        assert_eq!(two, 100 + 3 * 64 + 2 * 1024 * 1024);
        assert_eq!(estimate.scoring_total_bytes(1024, true), Some(100 + 128 + 1024 * 1024));
        let mut readout = estimate.clone();
        readout.add_state_buffers(64 * 1000, 4 * 1000).unwrap();
        assert_eq!(readout.scoring_total_bytes(1024, false).unwrap() - one, 64 * 1000);
        assert_eq!(readout.scoring_total_bytes(1025, false).unwrap() - two, 64 * 1000);
        assert_eq!(estimate.scoring_total_bytes(0, false), None);
        estimate.scratch_bytes = usize::MAX;
        assert_eq!(estimate.scoring_total_bytes(2, true), None);
    }

    #[stark_test]
    fn attention_geometry_accounts_for_precision_ring_and_shared_cache() {
        let full = attention_estimate(4096, 20, 64, 8, Some(2), None, 768, 2, false).unwrap();
        assert_eq!(full.state_bytes, (4096 + 1024) * 2 * 64 * 2 * 2);
        assert!(full.max_buffer_bytes >= full.state_bytes / 2);
        let f32 = attention_estimate(4096, 20, 64, 8, Some(2), None, 768, 4, false).unwrap();
        assert_eq!(f32.state_bytes, 2 * full.state_bytes);
        let ring = attention_estimate(4096, 20, 64, 8, Some(2), Some(256), 768, 2, false).unwrap();
        assert_eq!(ring.state_bytes, (256 + 1024) * 2 * 64 * 2 * 2);
        let shared = attention_estimate(4096, 20, 64, 8, None, None, 512, 2, false).unwrap();
        assert_eq!(shared.state_bytes, 0);
        assert!(shared.scratch_bytes > 0);
        assert!(attention_estimate(u32::MAX, 1, 64, 8, Some(2), None, 768, 2, false).is_none());
        let indexed_limit = u32::MAX / (8 * 128) - 1024;
        assert!(attention_estimate(indexed_limit, 1, 128, 8, Some(8), None, 3072, 2, false).is_some());
        assert!(attention_estimate(indexed_limit + 1, 1, 128, 8, Some(8), None, 3072, 2, false).is_none());
    }

    #[stark_test]
    fn individual_buffer_limit_does_not_clamp_combined_model_memory() {
        let profile = |context: u32| {
            let one_layer = MemoryEstimate::state(context as usize * 4);
            let combined = one_layer.clone().sequential(one_layer)?;
            (combined.max_buffer_bytes <= 64).then_some(combined)
        };
        assert_eq!(automatic_context(1, 100, 1024, profile), Some(16));
        let mut shared = MemoryEstimate::state(1000);
        shared.add_scratch(&[8], 1, 1).unwrap();
        assert_eq!(shared.without_state().max_buffer_bytes, 8);
        let mut scratch = MemoryEstimate::default();
        scratch.add_scratch(&[65], 1, 1).unwrap();
        assert_eq!(scratch.max_buffer_bytes, 128);
    }

    #[stark_test]
    fn dense_mtp_no_rope_profile_includes_both_attention_states_and_handoffs() {
        let geometry = |capacity| attention_estimate(capacity, 4, 128, 1, Some(1), None, 384, 2, false);
        let plain = geometry(4096).unwrap();
        let dense = dense_mtp_estimate(plain.clone(), geometry(4096).unwrap(), 1, 128, 256).unwrap();
        assert!(dense.state_bytes > 2 * plain.state_bytes);
        assert!(dense.total_bytes().unwrap() > plain.total_bytes().unwrap());
        let budget = dense.total_bytes().unwrap();
        let selected = automatic_context(1, 4096, budget, |capacity| {
            dense_mtp_estimate(geometry(capacity)?, geometry(capacity)?, 1, 128, 256)
        });
        assert_eq!(selected, Some(4096));
        assert!(
            automatic_context(1, 4096, 1, |capacity| {
                dense_mtp_estimate(geometry(capacity)?, geometry(capacity)?, 1, 128, 256)
            })
            .is_none()
        );
    }
}
