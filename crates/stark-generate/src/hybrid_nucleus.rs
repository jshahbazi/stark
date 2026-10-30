//! Exact small-prefix nucleus sampling with the reference order as fallback.

use rayon::prelude::*;

use super::{
    PARALLEL_PROBABILITY_MIN_CANDIDATES, ProbabilityMapper, Sampling, descending_sample_key, finite_guard,
    probability_exp_f32, profile, sample_with_mapper, sampling_indices, uniform,
};

const MAX_VOCABULARY: usize = 1_000_000;
const PREFIX: usize = 64;

#[derive(Default)]
pub(super) struct NucleusSampler {
    indices: Vec<usize>,
    probabilities: Vec<f64>,
    #[cfg(test)]
    last_route: Option<Route>,
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Ineligible,
    Invalid,
    SelectionBudget,
    Proved,
    UncertainReuse,
    InsufficientReuse,
}

#[derive(Clone, Copy)]
struct Prefix {
    kept: usize,
    cumulative: f64,
}

enum Attempt {
    Proved(Prefix),
    FullOrder,
    Reference,
}

#[derive(Clone, Copy)]
struct TotalInterval {
    lower: f64,
    upper: f64,
}

/// Higham (1993), section 2, eq. (2.6), bounds each recursive ordering's error
/// by gamma_(N-1) times the exact nonnegative sum. Applying it to two orders
/// gives |sorted_sum-id_sum| <= 2*a/(1-2*a)*id_sum for a=N*u, u=EPS/2.
/// For N<=1e6 the outward-rounded 4*N*EPS bound below has ample slack.
/// https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf
/// https://doi.org/10.1137/0914050
///
/// Both paths share the same rounded exp terms. Nonnegative subnormal
/// additions are exact on the binary64 grid; other additions use RN-even.
fn total_interval(
    sum: f64,
    length: usize,
) -> Option<TotalInterval> {
    if length == 0 || length > MAX_VOCABULARY || !sum.is_finite() || sum < 1.0 {
        return None;
    }
    let error = (4.0 * length as f64 * f64::EPSILON * sum).next_up();
    let lower = (sum - error).next_down();
    let upper = (sum + error).next_up();
    (lower > 0.0 && upper.is_finite()).then_some(TotalInterval {
        lower,
        upper,
    })
}

#[derive(PartialEq, Eq)]
enum StopDecision {
    Stop,
    Continue,
    Uncertain,
}

fn stop_decision(
    cumulative: f64,
    total: TotalInterval,
    top_p: f64,
) -> StopDecision {
    let lower = (cumulative / total.upper).next_down();
    let upper = (cumulative / total.lower).next_up();
    if lower >= top_p {
        StopDecision::Stop
    } else if upper < top_p {
        StopDecision::Continue
    } else {
        StopDecision::Uncertain
    }
}

fn eligible(
    logits: &[f32],
    sampling: &Sampling,
) -> bool {
    !logits.is_empty()
        && logits.len() <= MAX_VOCABULARY
        && matches!(sampling,Sampling::Stochastic{temperature,top_k:None,top_p:Some(top_p),min_p:None}
            if temperature.is_finite() && *temperature>0.0 && top_p.is_finite() && *top_p<1.0)
}

impl ProbabilityMapper {
    fn map_id_order_into(
        &mut self,
        logits: &[f32],
        max: f32,
        temperature: f32,
        probabilities: &mut Vec<f64>,
    ) {
        if logits.len() >= PARALLEL_PROBABILITY_MIN_CANDIDATES && !self.attempted {
            self.attempted = true;
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
            if threads > 1 {
                self.pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok();
            }
        }
        probabilities.clear();
        if logits.len() >= PARALLEL_PROBABILITY_MIN_CANDIDATES
            && let Some(pool) = &self.pool
            && probabilities.try_reserve_exact(logits.len()).is_ok()
        {
            probabilities.resize(logits.len(), 0.0);
            pool.install(|| {
                probabilities.par_iter_mut().zip(logits.par_iter()).for_each(|(p, &logit)| {
                    *p = probability_exp_f32((logit - max) / temperature);
                });
            });
        } else {
            probabilities.extend(logits.iter().map(|&logit| probability_exp_f32((logit - max) / temperature)));
        }
    }
}

/// Same exact bounded insertion order as the reference selector. Budget
/// exhaustion falls back before probability mapping, keeping scratch <=64 IDs.
fn bounded_top64_into(
    logits: &[f32],
    indices: &mut Vec<usize>,
) -> bool {
    let k = PREFIX.min(logits.len());
    indices.clear();
    indices.reserve_exact(PREFIX);
    indices.extend(0..k);
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    let mut worst_key = descending_sample_key(logits[indices[k - 1]]);
    let mut replacements = 0;
    for index in k..logits.len() {
        let key = descending_sample_key(logits[index]);
        if key >= worst_key {
            continue;
        }
        replacements += 1;
        if replacements > 1_024 {
            return false;
        }
        let position = indices.partition_point(|&candidate| descending_sample_key(logits[candidate]) <= key);
        indices.copy_within(position..k - 1, position + 1);
        indices[position] = index;
        worst_key = descending_sample_key(logits[indices[k - 1]]);
    }
    true
}

fn draw_id_probabilities(
    indices: &[usize],
    probabilities: &[f64],
    prefix: Prefix,
    rng: &mut u64,
) -> u32 {
    let target = uniform(rng) * prefix.cumulative;
    let mut mass = 0.0;
    for &index in &indices[..prefix.kept] {
        mass += probabilities[index];
        if target < mass {
            return index as u32;
        }
    }
    indices[prefix.kept - 1] as u32
}

fn ordered_prefix(
    indices: &[usize],
    probabilities: &[f64],
    top_p: f64,
) -> Prefix {
    let total = indices.iter().map(|&index| probabilities[index]).sum::<f64>();
    let mut kept = 0;
    let mut cumulative = 0.0;
    for &index in indices {
        if kept > 0 && cumulative / total >= top_p {
            break;
        }
        cumulative += probabilities[index];
        kept += 1;
    }
    Prefix {
        kept,
        cumulative,
    }
}

impl NucleusSampler {
    pub(super) fn clear(&mut self) {
        self.indices.clear();
        self.probabilities.clear();
    }

    // No RNG argument: selection and proof cannot advance the random stream.
    fn try_prefix(
        &mut self,
        logits: &[f32],
        sampling: &Sampling,
        mapper: &mut ProbabilityMapper,
    ) -> Attempt {
        {
            let _scope = profile::Scope::new("sample.nucleus.finite");
            if !finite_guard::all_finite(logits) {
                #[cfg(test)]
                {
                    self.last_route = Some(Route::Invalid);
                }
                return Attempt::Reference;
            }
        }
        let Sampling::Stochastic {
            temperature,
            top_p: Some(top_p),
            ..
        } = sampling
        else {
            unreachable!();
        };
        {
            let _scope = profile::Scope::new("sample.nucleus.select_top64");
            if !bounded_top64_into(logits, &mut self.indices) {
                #[cfg(test)]
                {
                    self.last_route = Some(Route::SelectionBudget);
                }
                return Attempt::Reference;
            }
        }
        let interval = {
            let _scope = profile::Scope::new("sample.nucleus.id_order_probability");
            let max = logits[self.indices[0]];
            mapper.map_id_order_into(logits, max, *temperature, &mut self.probabilities);
            // Exact max gives one term equal to1; every rounded term is in[0,1].
            let Some(interval) = total_interval(self.probabilities.iter().sum(), logits.len()) else {
                #[cfg(test)]
                {
                    self.last_route = Some(Route::Invalid);
                }
                return Attempt::Reference;
            };
            interval
        };
        let _scope = profile::Scope::new("sample.nucleus.prove_prefix");
        let mut kept = 0;
        let mut cumulative = 0.0;
        for &index in &self.indices {
            if kept > 0 {
                match stop_decision(cumulative, interval, *top_p as f64) {
                    StopDecision::Stop => {
                        return Attempt::Proved(Prefix {
                            kept,
                            cumulative,
                        });
                    },
                    StopDecision::Continue => {},
                    StopDecision::Uncertain => {
                        #[cfg(test)]
                        {
                            self.last_route = Some(Route::UncertainReuse);
                        }
                        return Attempt::FullOrder;
                    },
                }
            }
            cumulative += self.probabilities[index];
            kept += 1;
        }
        if kept == logits.len() {
            return Attempt::Proved(Prefix {
                kept,
                cumulative,
            });
        }
        #[cfg(test)]
        {
            self.last_route = Some(Route::InsufficientReuse);
        }
        match stop_decision(cumulative, interval, *top_p as f64) {
            StopDecision::Stop => Attempt::Proved(Prefix {
                kept,
                cumulative,
            }),
            StopDecision::Continue => Attempt::FullOrder,
            StopDecision::Uncertain => {
                #[cfg(test)]
                {
                    self.last_route = Some(Route::UncertainReuse);
                }
                Attempt::FullOrder
            },
        }
    }

    pub(super) fn sample(
        &mut self,
        logits: &[f32],
        sampling: &Sampling,
        rng: &mut u64,
        mapper: &mut ProbabilityMapper,
    ) -> Result<u32, &'static str> {
        self.clear();
        // Common top-k/greedy settings take the reference route before any
        // shortcut profile scope, probability allocation or full finite scan.
        if !eligible(logits, sampling) {
            #[cfg(test)]
            {
                self.last_route = Some(Route::Ineligible);
            }
            return sample_with_mapper(logits, sampling, rng, Some(mapper));
        }
        let _scope = profile::Scope::new("sample.nucleus.total");
        let result = match self.try_prefix(logits, sampling, mapper) {
            Attempt::Proved(prefix) => {
                #[cfg(test)]
                {
                    self.last_route = Some(Route::Proved);
                }
                let _scope = profile::Scope::new("sample.nucleus.draw");
                Ok(draw_id_probabilities(&self.indices, &self.probabilities, prefix, rng))
            },
            Attempt::FullOrder => {
                // Reuse every already-rounded probability. Only ordering and
                // F64 sum/filter/CDF are redone in the exact reference order.
                // Full indices are local and drop; retained prefix stays <=64.
                (|| {
                    let indices = {
                        let _scope = profile::Scope::new("sample.nucleus.fallback_order");
                        sampling_indices(logits, None)?
                    };
                    let Sampling::Stochastic {
                        top_p: Some(top_p),
                        ..
                    } = sampling
                    else {
                        unreachable!();
                    };
                    let prefix = {
                        let _scope = profile::Scope::new("sample.nucleus.fallback_filter");
                        ordered_prefix(&indices, &self.probabilities, *top_p as f64)
                    };
                    let _scope = profile::Scope::new("sample.nucleus.draw");
                    Ok(draw_id_probabilities(&indices, &self.probabilities, prefix, rng))
                })()
            },
            Attempt::Reference => sample_with_mapper(logits, sampling, rng, Some(mapper)),
        };
        self.clear();
        result
    }
}

#[cfg(test)]
#[path = "hybrid_nucleus_tests.rs"]
mod tests;
