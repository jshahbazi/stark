//! Stable, memory-sized context windows for loaded generation models.

use stark_resource_budget::{available_memory_bytes, inference_memory_budget};

const LOAD_CONTEXT_PERCENT: u64 = 85;
const CONTEXT_ALIGNMENT: u32 = 1024;

/// Price full cache capacity with two prefill chunks so the load probe
/// includes scratch and a prefix boundary. Actual prompt-dependent workspace
/// is admitted on each request.
pub const LOAD_PREFILL_PROBE_TOKENS: u32 = 2048;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ContextSizing {
    pub checkpoint_context_tokens: u32,
    pub engine_context_tokens: u32,
    pub memory_context_tokens: u32,
    pub context_tokens: u32,
    pub memory_budget_bytes: usize,
}

impl ContextSizing {
    /// Plan a window with an estimator bounded by its execution geometry.
    /// Missing estimates leave the backend's existing capacity policy in use.
    pub fn fit_loaded(
        checkpoint_context_tokens: u32,
        engine_context_tokens: u32,
        memory_budget_bytes: usize,
        estimate: impl Fn(u32) -> Option<usize>,
    ) -> Result<Option<Self>, &'static str> {
        let limit = checkpoint_context_tokens.min(engine_context_tokens);
        if limit < 2 || estimate(2).is_none() {
            return Ok(None);
        }
        Self::fit(checkpoint_context_tokens, engine_context_tokens, limit, memory_budget_bytes, estimate)
            .map(Some)
            .ok_or("resource_budget_exceeded")
    }

    /// Select a window for the lifetime of the loaded model. The estimate must
    /// describe cold execution and be nondecreasing over the search range.
    /// `estimate_limit` is independent of the checkpoint and engine bounds
    /// when the backend can estimate memory beyond its runnable geometry.
    pub fn fit(
        checkpoint_context_tokens: u32,
        engine_context_tokens: u32,
        estimate_limit: u32,
        memory_budget_bytes: usize,
        estimate: impl Fn(u32) -> Option<usize>,
    ) -> Option<Self> {
        let limit = checkpoint_context_tokens.min(engine_context_tokens);
        let memory_context_tokens = largest_fitting_context(2, estimate_limit, memory_budget_bytes, &estimate)?;
        let scaled = (u64::from(memory_context_tokens) * LOAD_CONTEXT_PERCENT / 100) as u32;
        let aligned = if scaled < CONTEXT_ALIGNMENT {
            scaled
        } else {
            scaled - scaled % CONTEXT_ALIGNMENT
        };
        // Bounded estimators cannot sample the margin beyond their geometry.
        // Keep the full model window if it fits with spare bytes instead.
        let context_tokens = if limit == estimate_limit
            && memory_context_tokens == limit
            && estimate(limit)? as u128 * 100 <= memory_budget_bytes as u128 * u128::from(LOAD_CONTEXT_PERCENT)
        {
            limit
        } else {
            aligned.min(limit)
        };
        // Scratch size classes can overlap differently after rounding down.
        // Publish only a capacity that the inventory confirms fits.
        let context_tokens =
            if context_tokens >= 2 && estimate(context_tokens).is_some_and(|bytes| bytes <= memory_budget_bytes) {
                context_tokens
            } else {
                largest_fitting_context(2, context_tokens, memory_budget_bytes, estimate)?
            };
        Some(Self {
            checkpoint_context_tokens,
            engine_context_tokens,
            memory_context_tokens,
            context_tokens,
            memory_budget_bytes,
        })
    }
}

/// Bytes available for an additional context allocation at this instant.
/// The OS and Metal describe overlapping memory, so use their tighter bound.
pub fn current_memory_budget(metal_headroom: Option<usize>) -> usize {
    let unified = inference_memory_budget(available_memory_bytes().unwrap_or(0));
    metal_headroom.map_or(unified, |bytes| unified.min(inference_memory_budget(bytes)))
}

#[cfg(feature = "metal")]
pub fn metal_memory_headroom() -> Option<usize> {
    use metal::MTLDevice;

    let device = <dyn MTLDevice>::system_default()?;
    Some(
        usize::try_from(device.recommended_max_working_set_size())
            .unwrap_or(usize::MAX)
            .saturating_sub(device.current_allocated_size()),
    )
}

/// Find the greatest length admitted by a monotone memory estimate.
pub fn largest_fitting_context(
    minimum: u32,
    maximum: u32,
    budget: usize,
    mut estimate: impl FnMut(u32) -> Option<usize>,
) -> Option<u32> {
    let mut fits = |context| estimate(context).is_some_and(|bytes| bytes <= budget);
    if minimum > maximum || !fits(minimum) {
        return None;
    }
    let mut low = minimum;
    let mut high = maximum;
    while low < high {
        let candidate = low + (high - low).div_ceil(2);
        if fits(candidate) {
            low = candidate;
        } else {
            high = candidate - 1;
        }
    }
    Some(low)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn sizing_caps_memory_and_preserves_fully_fitting_geometry() {
        let bytes = |context: u32| Some(context as usize * 8);
        let fit = |budget| ContextSizing::fit(131_072, 100_000, 100_000, budget, bytes).unwrap();
        assert_eq!(fit(400_000).memory_context_tokens, 50_000);
        assert_eq!(fit(400_000).context_tokens, 41_984);
        assert_eq!(fit(800_000).context_tokens, 84_992);
        assert_eq!(fit(1_000_000).context_tokens, 100_000);
        assert!(ContextSizing::fit(131_072, 100_000, 100_000, 16, bytes).is_none());
    }

    #[stark_test]
    fn unavailable_estimates_and_insufficient_memory_have_distinct_outcomes() {
        assert!(ContextSizing::fit_loaded(4096, 4096, 0, |_| None).unwrap().is_none());
        assert_eq!(ContextSizing::fit_loaded(4096, 4096, 0, |_| Some(16)).unwrap_err(), "resource_budget_exceeded");
    }

    #[stark_test]
    fn aligned_window_is_checked_against_the_estimator() {
        let bytes = |context: u32| {
            Some(if context == 8192 {
                12_000
            } else {
                context as usize
            })
        };
        let sizing = ContextSizing::fit(20_000, 20_000, 20_000, 10_000, bytes).unwrap();
        assert_eq!(sizing.memory_context_tokens, 10_000);
        assert_eq!(sizing.context_tokens, 8191);
    }

    #[stark_test]
    fn uncapped_estimates_apply_margin_before_geometry_limits() {
        let bytes = |context: u32| Some(context as usize);
        let fit = |budget, engine| {
            ContextSizing::fit(1_048_576, engine, u32::MAX, budget, bytes).map(|sizing| sizing.context_tokens)
        };
        assert_eq!(fit(500_000, 1_048_576), Some(424_960));
        assert_eq!(fit(1_500_000, 1_048_576), Some(1_048_576));
        assert_eq!(fit(1_500_000, 786_432), Some(786_432));
        assert_eq!(fit(20, 1_048_576), Some(17));
        assert_eq!(fit(2, 1_048_576), None);
    }
}
