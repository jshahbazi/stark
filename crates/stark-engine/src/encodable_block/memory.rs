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

/// Additional peak for a retained token Vec. Bulk appends follow the prefill
/// chunk grid; decode appends one token at a time. A replacement may coexist
/// with its previous allocation, whose original capacity is already resident.
pub fn token_vector_growth_bytes<T>(
    initial_capacity: usize,
    retained_tokens: usize,
    bulk_tokens: usize,
    single_tokens: usize,
    chunk_tokens: usize,
) -> Option<usize> {
    let element_bytes = size_of::<T>();
    if element_bytes == 0 || chunk_tokens == 0 || retained_tokens > initial_capacity {
        return None;
    }
    let prefill_end = retained_tokens.checked_add(bulk_tokens)?;
    let end = prefill_end.checked_add(single_tokens)?;
    let initial_bytes = initial_capacity.checked_mul(element_bytes).filter(|&bytes| bytes <= isize::MAX as usize)?;
    end.checked_mul(element_bytes).filter(|&bytes| bytes <= isize::MAX as usize)?;
    let minimum = if element_bytes == 1 {
        8
    } else if element_bytes <= 1024 {
        4
    } else {
        1
    };
    let mut capacity = initial_capacity;
    let mut length = retained_tokens;
    let mut peak = initial_capacity;
    while length < prefill_end {
        if prefill_end <= capacity {
            break;
        }
        // Skip complete chunks that fit. Only a logarithmic number of actual
        // allocation boundaries need simulation even for large model windows.
        let fitting = capacity.checked_sub(length)?.checked_div(chunk_tokens)?.checked_mul(chunk_tokens)?;
        length = length.checked_add(fitting)?.min(prefill_end);
        if length == prefill_end {
            break;
        }
        length = length.checked_add((prefill_end - length).min(chunk_tokens))?;
        let replacement = capacity.checked_mul(2)?.max(length).max(minimum);
        peak = peak.max(capacity.checked_add(replacement)?);
        capacity = replacement;
    }
    while capacity < end {
        let replacement = capacity.checked_mul(2)?.max(capacity.checked_add(1)?).max(minimum);
        peak = peak.max(capacity.checked_add(replacement)?);
        capacity = replacement;
    }
    peak.checked_mul(element_bytes).filter(|&bytes| bytes <= isize::MAX as usize)?.checked_sub(initial_bytes)
}

/// Image boundaries can split the ordinary prefill grid. Without their exact
/// ranges, bound any amortized Vec schedule by its final old/new overlap.
pub fn irregular_token_history_bytes(
    initial_capacity: usize,
    final_tokens: usize,
) -> Option<usize> {
    if final_tokens <= initial_capacity {
        return Some(0);
    }
    final_tokens
        .checked_mul(3)?
        .max(4)
        .checked_sub(initial_capacity)?
        .checked_mul(size_of::<u64>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
}

/// Generation-only host token owners. Returned IDs are one preallocated u32
/// Vec; ordinary streams also own a u64 prompt copy and an amortized history.
/// Dense MTP consumes borrowed u32 prompts and has no persistent token history.
pub fn generation_token_storage_bytes(
    input_tokens: u32,
    max_new_tokens: u32,
    dense_mtp: bool,
    irregular_prefill: bool,
) -> Option<usize> {
    let final_tokens = input_tokens.checked_add(max_new_tokens)? as usize;
    let history = if dense_mtp {
        0
    } else if irregular_prefill {
        irregular_token_history_bytes(0, final_tokens)?
    } else {
        token_vector_growth_bytes::<u64>(0, 0, input_tokens as usize, max_new_tokens as usize, 1024)?
    };
    let prompt_copies = (input_tokens as usize).checked_mul(if dense_mtp {
        4
    } else {
        12
    })?;
    let returned_tokens = (max_new_tokens as usize).checked_mul(size_of::<u32>())?;
    history.checked_add(prompt_copies)?.checked_add(returned_tokens)
}

#[cfg(test)]
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
    attention_estimate_with_prefix(
        context_tokens,
        input_tokens,
        input_tokens,
        head_dim,
        query_heads,
        kv_heads,
        ring_capacity,
        projection_dim,
        element_bytes,
        reference,
    )
}

pub fn attention_estimate_with_prefix(
    context_tokens: u32,
    input_tokens: u32,
    sequence_tokens: u32,
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
    let prefill_sequence = sequence_tokens.max(1).min(cache_rows);
    let decode_sequence = context_tokens.min(cache_rows);
    let mut result = scratch(prefill_rows, prefill_sequence)?.sequential(scratch(1, decode_sequence)?)?;
    result.set_state_buffers(state_bytes, state_bytes / 2);
    Some(result)
}

/// Sparse full-attention buffers map only rows reached by this request.
/// Rings additionally need one live prefill batch beyond the retained window.
pub fn attention_physical_state_bytes(
    context_tokens: u32,
    input_tokens: u32,
    sequence_tokens: u32,
    irregular_prefill: bool,
    head_dim: u32,
    kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    element_bytes: usize,
    page_bytes: usize,
) -> Option<usize> {
    if page_bytes == 0 || input_tokens == 0 || sequence_tokens < input_tokens {
        return None;
    }
    // Stream/scoring preparation maps the largest batch at the final batch's
    // start, even when that last batch is short. Images can shift that start
    // off the ordinary 1024-row grid, so their decoder uses a layout bound.
    let batch = input_tokens.min(1024);
    let last_start = if irregular_prefill {
        sequence_tokens.checked_sub(1)?
    } else {
        sequence_tokens.checked_sub(input_tokens)?.checked_add((input_tokens - 1) / 1024 * 1024)?
    };
    let prefill_rows = ring_capacity.map_or(last_start, |window| last_start.min(window)).checked_add(batch)?;
    let decode_rows = match ring_capacity {
        Some(window) => context_tokens.min(window).checked_add(1)?,
        None => context_tokens,
    };
    let rows = prefill_rows.max(decode_rows);
    match kv_heads {
        Some(heads) => {
            let buffer = checked_bytes(&[rows, heads, head_dim], element_bytes)?;
            buffer.checked_add(page_bytes - 1)?.checked_div(page_bytes)?.checked_mul(page_bytes)?.checked_mul(2)
        },
        None => Some(0),
    }
}

/// Address reservation for one K/V cache buffer, independent of physical
/// sparse mapping and of the request's shorter execution workspace.
pub fn attention_cache_buffer_bytes(
    context_tokens: u32,
    head_dim: u32,
    kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    element_bytes: usize,
) -> Option<usize> {
    let rows = ring_capacity
        .map_or(context_tokens, |window| window.min(context_tokens))
        .checked_add(crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY)?;
    match kv_heads {
        Some(heads) => {
            rows.checked_mul(heads)?.checked_mul(head_dim)?;
            checked_bytes(&[rows, heads, head_dim], element_bytes)
        },
        None => Some(0),
    }
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
    fn retained_token_history_charges_capacity_growth_and_replacement_overlap() {
        let bytes = |capacity, retained, bulk, output, chunk| {
            token_vector_growth_bytes::<u64>(capacity, retained, bulk, output, chunk)
        };
        assert_eq!(bytes(1024, 1024, 1, 8, 1024), Some(2048 * 8));
        assert_eq!(bytes(2048, 1024, 1, 8, 1024), Some(0));
        assert_eq!(bytes(2048, 1, 1024, 8, 1024), Some(0));
        assert_eq!(bytes(1025, 1024, 1, 1, 1024), Some(2050 * 8));
        // A non-power-of-two bulk allocation can double on the next push.
        assert_eq!(bytes(0, 0, 1025, 1, 1025), Some((1025 + 2050) * 8));
        // Ordinary prefill appends on its fixed 1024-row chunk grid.
        assert_eq!(bytes(0, 0, 1025, 1, 1024), Some((1024 + 2048) * 8));
        assert_eq!(bytes(0, 0, 1025, 1024, 1024), Some((2048 + 4096) * 8));
        assert_eq!(bytes(4, 5, 1, 1, 1024), None);
        assert_eq!(bytes(0, 0, 1, 1, 0), None);
        assert_eq!(bytes(usize::MAX, 0, 1, 1, 1024), None);
        assert_eq!(bytes(0, 0, usize::MAX, 1, 1024), None);
    }

    #[stark_test]
    fn token_history_inventory_matches_real_vec_capacity_transitions() {
        for (capacity, retained, bulk, output, chunk) in [
            (0, 0, 1025, 1, 1024),
            (0, 0, 1025, 1, 1025),
            (6, 4, 3, 40, 2),
            (1024, 1024, 1, 8, 1024),
            (1025, 1024, 1, 1, 1024),
            (2048, 1024, 1, 8, 1024),
            (1024, 1024, 1025, 1, 1024),
            (4096, 1024, 1025, 2048, 1024),
        ] {
            let mut tokens = Vec::<u64>::with_capacity(capacity);
            tokens.resize(retained, 0);
            let initial = tokens.capacity();
            let mut peak = initial;
            let mut append = |rows| {
                let previous = tokens.capacity();
                tokens.extend(std::iter::repeat_n(0, rows));
                if previous != tokens.capacity() {
                    peak = peak.max(previous + tokens.capacity());
                }
            };
            let mut remaining = bulk;
            while remaining > 0 {
                let rows = remaining.min(chunk);
                append(rows);
                remaining -= rows;
            }
            for _ in 0..output {
                append(1);
            }
            assert_eq!(
                token_vector_growth_bytes::<u64>(capacity, retained, bulk, output, chunk),
                Some((peak - initial) * 8)
            );
        }
        // Arbitrary image chunk boundaries retain their separate safe bound.
        assert_eq!(irregular_token_history_bytes(2048, 1033), Some(0));
        assert!(
            irregular_token_history_bytes(0, 2049).unwrap()
                >= token_vector_growth_bytes::<u64>(0, 0, 2048, 1, 1000).unwrap()
        );
        assert_eq!(irregular_token_history_bytes(0, usize::MAX), None);
    }

    #[stark_test]
    fn generation_token_storage_tracks_executed_length_and_selected_path() {
        let history = token_vector_growth_bytes::<u64>(0, 0, 1025, 1, 1024).unwrap();
        assert_eq!(generation_token_storage_bytes(1025, 1, false, false), Some(history + 1025 * 12 + 4));
        // Dense MTP has only the possible retained u32 prefix key and one
        // returned-token Vec, even when a larger dense KV context is selected.
        assert_eq!(generation_token_storage_bytes(1025, 1, true, false), Some(1025 * 4 + 4));
        assert_eq!(generation_token_storage_bytes(1025, 2, true, false), Some(1025 * 4 + 8));
        assert_eq!(generation_token_storage_bytes(u32::MAX, 1, true, false), None);
    }

    #[stark_test]
    fn reused_attention_suffix_keeps_full_prefix_score_width() {
        let warm = attention_estimate_with_prefix(8400, 128, 8320, 64, 8, Some(2), None, 768, 2, false).unwrap();
        let wrong = attention_estimate_with_prefix(8400, 128, 128, 64, 8, Some(2), None, 768, 2, false).unwrap();
        let cold = attention_estimate_with_prefix(8400, 8320, 8320, 64, 8, Some(2), None, 768, 2, false).unwrap();
        assert!(warm.scratch_bytes > wrong.scratch_bytes);
        assert!(warm.scratch_bytes < cold.scratch_bytes);
        assert_eq!(warm.state_bytes, cold.state_bytes);
        assert_eq!(warm.state_bytes, wrong.state_bytes);
    }

    #[stark_test]
    fn sparse_physical_cache_counts_mapping_pages_without_unused_suffix_reservation() {
        let page = 256 * 1024;
        let full = attention_physical_state_bytes(8192, 128, 8192, false, 64, Some(2), None, 2, page).unwrap();
        assert_eq!(full, 8192 * 64 * 2 * 2 * 2);
        let reserved = attention_cache_buffer_bytes(8192, 64, Some(2), None, 2).unwrap();
        assert!(reserved * 2 > full);
        let ring = attention_physical_state_bytes(8192, 128, 8192, false, 64, Some(2), Some(512), 2, page).unwrap();
        assert_eq!(ring, 2 * page);
        let wide_batch =
            attention_physical_state_bytes(8192, 1024, 8192, false, 64, Some(2), Some(512), 2, page).unwrap();
        assert_eq!(wide_batch, 4 * page);
        assert_eq!(attention_physical_state_bytes(8192, 128, 8192, false, 64, None, None, 2, page), Some(0));
        assert!(attention_physical_state_bytes(8192, 128, 8192, false, 64, Some(2), None, 2, 0).is_none());
    }

    #[stark_test]
    fn sparse_physical_prefill_covers_padded_final_batch_and_shifted_image_grid() {
        let rows = |context, suffix, sequence, irregular, ring| {
            attention_physical_state_bytes(context, suffix, sequence, irregular, 1, Some(1), ring, 1, 1)
                .map(|bytes| bytes / 2)
        };
        assert_eq!(rows(1026, 1025, 1025, false, None), Some(2048));
        assert_eq!(rows(2050, 1025, 2049, false, None), Some(3072));
        assert_eq!(rows(1026, 1025, 1025, false, Some(512)), Some(1536));
        assert_eq!(rows(2049, 2048, 2048, false, None), Some(2049));
        // Image span 1000..1300 shifts chunks to 0..1000, 1000..2024,
        // 2024..2048; stream preparation therefore maps 2024+1024 rows.
        let image = rows(2049, 2048, 2048, true, None).unwrap();
        assert_eq!(image, 3071);
        assert!(image >= 2024 + 1024);
        assert!(image <= 2049 + 1024);
        assert_eq!(rows(2049, 2048, 2048, true, Some(512)), Some(1536));
    }

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
        assert_eq!(attention_cache_buffer_bytes(4096, 64, Some(2), None, 2), Some(full.state_bytes / 2));
        assert_eq!(attention_cache_buffer_bytes(4096, 64, Some(2), Some(256), 2), Some(ring.state_bytes / 2));
        assert_eq!(attention_cache_buffer_bytes(4096, 64, None, None, 2), Some(0));
        assert!(attention_cache_buffer_bytes(indexed_limit + 1, 128, Some(8), None, 2).is_none());
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
    fn sparse_short_request_still_checks_full_cache_address_reservation() {
        let request = attention_estimate(200, 100, 64, 8, Some(2), None, 768, 2, false).unwrap();
        let device_limit = 1024 * 1024;
        assert!(request.max_buffer_bytes < device_limit);
        assert!(attention_cache_buffer_bytes(2048, 64, Some(2), None, 2).unwrap() <= device_limit);
        assert!(attention_cache_buffer_bytes(4096, 64, Some(2), None, 2).unwrap() > device_limit);
        // Physical request growth can fit while the unchanged selected
        // context's sparse address reservation exceeds the device limit.
        assert_eq!(generation_allocation_context(100, 4096, 100, true), Some(200));
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
