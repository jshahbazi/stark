//! Stable, memory-sized context windows for loaded generation models.

use stark_resource_budget::{available_memory_bytes, inference_memory_budget};

const LOAD_CONTEXT_PERCENT: u64 = 85;
const CONTEXT_ALIGNMENT: u32 = 1024;

/// Price full cache capacity with two prefill chunks so the load probe
/// includes scratch and a prefix boundary. Actual prompt-dependent workspace
/// is admitted on each request.
pub const LOAD_PREFILL_PROBE_TOKENS: u32 = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct BackendContextBound {
    pub context_tokens: u32,
    pub source: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextConstraint {
    Checkpoint,
    Backend,
    Memory,
    UserCap,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ContextSizing {
    pub checkpoint_context_tokens: u32,
    pub engine_context_tokens: u32,
    pub backend_bound: Option<BackendContextBound>,
    pub user_context_cap: Option<u32>,
    pub limiting_factor: ContextConstraint,
    pub memory_context_tokens: u32,
    pub context_tokens: u32,
    pub memory_budget_bytes: usize,
}

impl ContextSizing {
    /// Select a window for the lifetime of the loaded model. The estimate must
    /// describe cold execution and be nondecreasing over the search range.
    /// The estimator range can extend beyond the runnable geometry.
    pub fn fit_loaded(
        checkpoint_context_tokens: u32,
        backend_bound: Option<BackendContextBound>,
        estimate_limit: u32,
        memory_budget_bytes: usize,
        estimate: impl Fn(u32) -> Option<usize>,
    ) -> Result<Self, &'static str> {
        let engine_context_tokens = backend_bound.map_or(checkpoint_context_tokens, |bound| bound.context_tokens);
        let limit = checkpoint_context_tokens.min(engine_context_tokens);
        if limit < 2 || limit == u32::MAX || estimate_limit < 2 {
            return Err("invalid_finite_context_limit");
        }
        if estimate(2).is_none() {
            return Err("context_memory_estimate_unavailable");
        }
        let fit = || {
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
            let limiting_factor = if context_tokens < limit {
                ContextConstraint::Memory
            } else if engine_context_tokens < checkpoint_context_tokens {
                ContextConstraint::Backend
            } else {
                ContextConstraint::Checkpoint
            };
            Some(Self {
                checkpoint_context_tokens,
                engine_context_tokens,
                backend_bound,
                user_context_cap: None,
                limiting_factor,
                memory_context_tokens,
                context_tokens,
                memory_budget_bytes,
            })
        };
        fit().ok_or("resource_budget_exceeded")
    }

    pub fn with_user_cap(
        mut self,
        cap: Option<u32>,
    ) -> Result<Self, &'static str> {
        if let Some(cap) = cap {
            if cap < 2 {
                return Err("invalid_finite_context_limit");
            }
            if cap < self.context_tokens {
                self.context_tokens = cap;
                self.limiting_factor = ContextConstraint::UserCap;
            }
        }
        self.user_context_cap = cap;
        Ok(self)
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

    fn bound(context_tokens: u32) -> BackendContextBound {
        BackendContextBound {
            context_tokens,
            source: "metal_test_buffer",
        }
    }

    #[stark_test]
    fn sizing_caps_memory_and_preserves_fully_fitting_geometry() {
        let bytes = |context: u32| Some(context as usize * 8);
        let fit = |budget| ContextSizing::fit_loaded(131_072, Some(bound(100_000)), 100_000, budget, bytes).unwrap();
        assert_eq!(fit(400_000).memory_context_tokens, 50_000);
        assert_eq!(fit(400_000).context_tokens, 41_984);
        assert_eq!(fit(800_000).context_tokens, 84_992);
        assert_eq!(fit(1_000_000).context_tokens, 100_000);
        assert!(ContextSizing::fit_loaded(131_072, Some(bound(100_000)), 100_000, 16, bytes).is_err());
    }

    #[stark_test]
    fn evidence_distinguishes_checkpoint_backend_memory_and_user_cap() {
        let bytes = |context: u32| Some(context as usize * 8);
        let checkpoint = ContextSizing::fit_loaded(40_960, None, 40_960, 1_000_000, bytes).unwrap();
        assert_eq!(checkpoint.context_tokens, 40_960);
        assert_eq!(checkpoint.engine_context_tokens, 40_960);
        assert_eq!(checkpoint.backend_bound, None);
        assert_eq!(checkpoint.limiting_factor, ContextConstraint::Checkpoint);
        let backend = ContextSizing::fit_loaded(131_072, Some(bound(65_536)), 65_536, 1_000_000, bytes).unwrap();
        assert_eq!(backend.limiting_factor, ContextConstraint::Backend);
        assert_eq!(backend.backend_bound, Some(bound(65_536)));
        let memory = ContextSizing::fit_loaded(131_072, Some(bound(65_536)), 65_536, 320_000, bytes).unwrap();
        assert_eq!(memory.limiting_factor, ContextConstraint::Memory);
        assert_eq!(memory.memory_context_tokens, 40_000);
        let capped = backend.with_user_cap(Some(32_768)).unwrap();
        assert_eq!(capped.context_tokens, 32_768);
        assert_eq!(capped.user_context_cap, Some(32_768));
        assert_eq!(capped.limiting_factor, ContextConstraint::UserCap);
    }

    #[stark_test]
    fn unavailable_estimates_and_insufficient_memory_have_distinct_outcomes() {
        assert_eq!(
            ContextSizing::fit_loaded(4096, None, 4096, 0, |_| None).unwrap_err(),
            "context_memory_estimate_unavailable"
        );
        assert_eq!(ContextSizing::fit_loaded(1, None, 1, 0, |_| None).unwrap_err(), "invalid_finite_context_limit");
        assert_eq!(
            ContextSizing::fit_loaded(4096, None, 4096, 0, |_| Some(16)).unwrap_err(),
            "resource_budget_exceeded"
        );
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
        let sizing = ContextSizing::fit_loaded(20_000, None, 20_000, 10_000, bytes).unwrap();
        assert_eq!(sizing.memory_context_tokens, 10_000);
        assert_eq!(sizing.context_tokens, 8191);
    }

    #[stark_test]
    fn uncapped_estimates_apply_margin_before_geometry_limits() {
        let bytes = |context: u32| Some(context as usize);
        let fit = |budget, engine| {
            ContextSizing::fit_loaded(1_048_576, Some(bound(engine)), u32::MAX, budget, bytes)
                .ok()
                .map(|sizing| sizing.context_tokens)
        };
        assert_eq!(fit(500_000, 1_048_576), Some(424_960));
        assert_eq!(fit(1_500_000, 1_048_576), Some(1_048_576));
        assert_eq!(fit(1_500_000, 786_432), Some(786_432));
        assert_eq!(fit(20, 1_048_576), Some(17));
        assert_eq!(fit(2, 1_048_576), None);
    }
}
