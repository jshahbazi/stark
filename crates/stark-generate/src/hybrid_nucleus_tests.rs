//! Synthetic exact controls. Assertions do not print logits or probabilities.

use super::*;
use crate::hybrid::probability_values;

fn stochastic(top_p: f32) -> Sampling {
    Sampling::Stochastic {
        temperature: 1.0,
        top_k: None,
        top_p: Some(top_p),
        min_p: None,
    }
}

fn clear(sampler: &NucleusSampler) -> bool {
    sampler.indices.is_empty() && sampler.probabilities.is_empty()
}

fn next_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut word = *state;
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
    word ^ (word >> 31)
}

fn logits(
    length: usize,
    seed: u64,
) -> Vec<f32> {
    let mut state = seed;
    (0..length)
        .map(|index| match index % 17 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1),
            3 => -f32::from_bits(1),
            4 => f32::MIN_POSITIVE,
            5 => -f32::MIN_POSITIVE,
            _ => ((next_word(&mut state) % 200_001) as i32 - 100_000) as f32 / 137.0,
        })
        .collect()
}

// The pre-change probability expression is deliberately retained only here.
fn old_probability_values(
    logits: &[f32],
    indices: &[usize],
    max: f32,
    temperature: f32,
    pool: Option<&rayon::ThreadPool>,
) -> Vec<f64> {
    if let Some(pool) = pool {
        let mut probabilities = Vec::new();
        if probabilities.try_reserve_exact(indices.len()).is_ok() {
            probabilities.resize(indices.len(), 0.0);
            pool.install(|| {
                probabilities.par_iter_mut().zip(indices.par_iter()).for_each(|(p, &i)| {
                    *p = (((logits[i] - max) / temperature) as f64).exp();
                })
            });
            return probabilities;
        }
    }
    indices.iter().map(|&i| (((logits[i] - max) / temperature) as f64).exp()).collect()
}

fn old_sample(
    logits: &[f32],
    sampling: &Sampling,
    rng: &mut u64,
    pool: Option<&rayon::ThreadPool>,
) -> Result<u32, &'static str> {
    if logits.is_empty() || !finite_guard::all_finite(logits) {
        return Err("native_generation_failed");
    }
    let Sampling::Stochastic {
        temperature,
        top_k,
        top_p,
        min_p,
    } = sampling
    else {
        let mut best = 0;
        for index in 1..logits.len() {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        return Ok(best as u32);
    };
    let indices = sampling_indices(logits, *top_k)?;
    let pool = pool.filter(|_| indices.len() >= PARALLEL_PROBABILITY_MIN_CANDIDATES);
    let probabilities = old_probability_values(logits, &indices, logits[indices[0]], *temperature, pool);
    let total = probabilities.iter().sum::<f64>();
    let mut kept = 0;
    let mut cumulative = 0.0;
    for &p in &probabilities {
        if kept > 0
            && (min_p.is_some_and(|minimum| p < minimum as f64)
                || top_p.is_some_and(|limit| cumulative / total >= limit as f64))
        {
            break;
        }
        cumulative += p;
        kept += 1;
    }
    let target = uniform(rng) * cumulative;
    let mut mass = 0.0;
    for (&index, &p) in indices[..kept].iter().zip(&probabilities) {
        mass += p;
        if target < mass {
            return Ok(index as u32);
        }
    }
    Ok(indices[kept - 1] as u32)
}

fn reference_prefix(
    probabilities: &[f64],
    top_p: f64,
) -> Prefix {
    let total = probabilities.iter().sum::<f64>();
    let mut kept = 0;
    let mut cumulative = 0.0;
    for &p in probabilities {
        if kept > 0 && cumulative / total >= top_p {
            break;
        }
        cumulative += p;
        kept += 1;
    }
    Prefix {
        kept,
        cumulative,
    }
}

fn exact_terms_and_prefix(
    logits: &[f32],
    sampling: &Sampling,
    mapper: &mut ProbabilityMapper,
    sampler: &mut NucleusSampler,
) {
    let Sampling::Stochastic {
        temperature,
        top_p: Some(top_p),
        top_k: None,
        min_p: None,
    } = sampling
    else {
        panic!("nucleus control required");
    };
    let indices = sampling_indices(logits, None).unwrap();
    let max = logits[indices[0]];
    let old = old_probability_values(logits, &indices, max, *temperature, mapper.pool.as_ref());
    let current = mapper.map(logits, &indices, max, *temperature);
    assert!(
        old.len() == current.len() && old.iter().zip(&current).all(|(a, b)| a.to_bits() == b.to_bits()),
        "common exp helper changed previous probability bits"
    );
    let total = old.iter().sum::<f64>();
    let mut id_probabilities = Vec::new();
    mapper.map_id_order_into(logits, max, *temperature, &mut id_probabilities);
    assert!(
        indices.iter().zip(&old).all(|(&i, p)| id_probabilities[i].to_bits() == p.to_bits()),
        "ID map changed probability bits"
    );
    let interval = total_interval(id_probabilities.iter().sum(), logits.len()).expect("eligible interval");
    assert!(interval.lower <= total && total <= interval.upper, "sorted reference total escaped interval");
    let expected = reference_prefix(&old, *top_p as f64);
    let restored = ordered_prefix(&indices, &id_probabilities, *top_p as f64);
    assert!(
        expected.kept == restored.kept && expected.cumulative.to_bits() == restored.cumulative.to_bits(),
        "ID lookup changed ordered filter or CDF prefix"
    );
    sampler.clear();
    match sampler.try_prefix(logits, sampling, mapper) {
        Attempt::Proved(prefix) => {
            assert!(
                prefix.kept == expected.kept && prefix.cumulative.to_bits() == expected.cumulative.to_bits(),
                "proved prefix changed reference filter"
            );
            assert!(sampler.indices[..prefix.kept] == indices[..prefix.kept], "proved prefix changed total order");
        },
        Attempt::FullOrder => {
            let actual = ordered_prefix(&indices, &sampler.probabilities, *top_p as f64);
            assert!(
                actual.kept == expected.kept && actual.cumulative.to_bits() == expected.cumulative.to_bits(),
                "fallback reuse changed reference prefix"
            );
        },
        Attempt::Reference => {
            assert!(sampler.last_route == Some(Route::SelectionBudget), "unexpected early reference route")
        },
    }
    sampler.clear();
}

fn exact_draws(
    logits: &[f32],
    sampling: &Sampling,
    seeds: &[u64],
    mapper: &mut ProbabilityMapper,
    sampler: &mut NucleusSampler,
) {
    for &seed in seeds {
        let mut old_rng = seed;
        let old = old_sample(logits, sampling, &mut old_rng, mapper.pool.as_ref());
        let mut reference_rng = seed;
        let reference = sample_with_mapper(logits, sampling, &mut reference_rng, Some(mapper));
        let mut actual_rng = seed;
        let actual = sampler.sample(logits, sampling, &mut actual_rng, mapper);
        assert!(old == reference && old == actual, "sampled token or error changed");
        assert!(old_rng == reference_rng && old_rng == actual_rng, "RNG advancement changed");
        assert!(clear(sampler), "draw retained semantic scratch");
    }
}

#[test]
fn conservative_total_interval_contains_both_orders_with_subnormal_terms() {
    let mut state = 42;
    for length in [1, 2, 3, 63, 64, 65, 127, 4096, 65_537, 1_000_000] {
        let mut terms: Vec<f64> = (0..length)
            .map(|index| match index % 7 {
                0 => 0.0,
                1 => f64::from_bits(1),
                2 => f64::MIN_POSITIVE,
                3 => f64::EPSILON / 2.0,
                4 => 1.0,
                _ => (next_word(&mut state) >> 11) as f64 / (1_u64 << 53) as f64,
            })
            .collect();
        terms[0] = 1.0;
        let sum = terms.iter().sum();
        let interval = total_interval(sum, length).expect("eligible interval");
        for descending in [false, true] {
            terms.sort_unstable_by(|a, b| {
                if descending {
                    b.total_cmp(a)
                } else {
                    a.total_cmp(b)
                }
            });
            let sorted = terms.iter().sum::<f64>();
            assert!(interval.lower <= sorted && sorted <= interval.upper, "ordered sum escaped interval");
        }
    }
    for (sum, length) in [(0.0, 1), (f64::NAN, 1), (f64::INFINITY, 1), (1.0, 0), (1.0, 1_000_001)] {
        assert!(total_interval(sum, length).is_none(), "invalid interval admitted");
    }
}

#[test]
fn bounded_top64_matches_reference_or_stops_before_exp_on_adversarial_order() {
    let mut indices = Vec::new();
    let mut saw_budget = false;
    for length in [1, 2, 63, 64, 65, 257, 1024, 1088, 1089, 4096, 65_537] {
        let mut values = logits(length, 42);
        for order in 0..3 {
            if order == 1 {
                values.sort_unstable_by(f32::total_cmp);
            }
            if order == 2 {
                values.reverse();
            }
            let original = sampling_indices(&values, Some(64)).unwrap();
            if bounded_top64_into(&values, &mut indices) {
                assert!(indices == original, "bounded prefix changed total order");
            } else {
                saw_budget = true;
            }
            assert!(indices.len() <= 64 && indices.capacity() == 64, "prefix scratch grew beyond fixed budget");
        }
    }
    assert!(saw_budget, "adversarial selector budget not covered");
    let mut mapper = ProbabilityMapper {
        pool: None,
        attempted: true,
    };
    let mut sampler = NucleusSampler::default();
    let values: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    exact_draws(&values, &stochastic(0.95), &[0, 42, u64::MAX], &mut mapper, &mut sampler);
    assert!(
        sampler.last_route == Some(Route::SelectionBudget) && sampler.probabilities.capacity() == 0,
        "selector fallback mapped exp or retained vocabulary state"
    );
}

#[test]
fn common_scalar_exp_matches_old_serial_parallel_and_ieee_edge_expressions() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut values = logits(65_537, 123);
    values.extend([f32::MIN, f32::MAX, -0.0, 0.0]);
    let indices = sampling_indices(&values, None).unwrap();
    let max = values[indices[0]];
    for temperature in [f32::from_bits(1), f32::MIN_POSITIVE, 0.01, 0.7, 1.0, 2.0, f32::MAX] {
        let previous = old_probability_values(&values, &indices, max, temperature, None);
        for pool in [None, Some(&pool)] {
            let current = probability_values(&values, &indices, max, temperature, pool);
            assert!(
                previous.len() == current.len()
                    && previous.iter().zip(&current).all(|(a, b)| a.to_bits() == b.to_bits()),
                "shared helper changed old exp bits"
            );
        }
    }
    for exponent in 0..=254_u32 {
        for fraction in [0, 1, 0x3fffff, 0x7fffff] {
            for sign in [0, 0x80000000] {
                let value = f32::from_bits(sign | (exponent << 23) | fraction);
                assert!(
                    probability_exp_f32(value).to_bits() == ((value as f64).exp()).to_bits(),
                    "scalar exp changed IEEE edge encoding"
                );
            }
        }
    }
}

#[test]
fn exact_random_ties_signed_zero_and_top_p_boundary_draws_match_old_sampler() {
    let mut mapper = ProbabilityMapper {
        pool: None,
        attempted: true,
    };
    let mut sampler = NucleusSampler::default();
    for length in [1, 2, 3, 63, 64, 65, 97, 257, 1024, 4096] {
        for seed in [0, 1, 42, 0x123456789abcdef0] {
            let mut values = logits(length, seed);
            for order in 0..3 {
                if order == 1 {
                    values.sort_unstable_by(f32::total_cmp);
                }
                if order == 2 {
                    values.reverse();
                }
                for top_p in [0.0, f32::from_bits(1), 0.01, 0.5, 0.9, 0.95, f32::from_bits(1.0_f32.to_bits() - 1)] {
                    let sampling = stochastic(top_p);
                    exact_terms_and_prefix(&values, &sampling, &mut mapper, &mut sampler);
                    exact_draws(&values, &sampling, &[0, 42, u64::MAX], &mut mapper, &mut sampler);
                }
            }
        }
    }
}

#[test]
fn reuse_fallback_preserves_ordered_total_prefix_and_rng_for_many_seeds() {
    let mut mapper = ProbabilityMapper {
        pool: None,
        attempted: true,
    };
    let mut sampler = NucleusSampler::default();
    let seeds: Vec<u64> = (0..256).map(|seed| seed * 0x1234567).collect();
    for top_p in [0.5_f32.next_down(), 0.5, 0.5_f32.next_up()] {
        exact_terms_and_prefix(&[0.0, -0.0], &stochastic(top_p), &mut mapper, &mut sampler);
        exact_draws(&[0.0, -0.0], &stochastic(top_p), &seeds, &mut mapper, &mut sampler);
        if top_p == 0.5 {
            assert!(
                sampler.last_route == Some(Route::UncertainReuse),
                "threshold uncertainty did not reuse full order"
            );
        }
    }
    for length in [65, 97, 1024] {
        let values = vec![-0.0; length];
        let sampling = stochastic(0.999);
        exact_terms_and_prefix(&values, &sampling, &mut mapper, &mut sampler);
        exact_draws(&values, &sampling, &seeds, &mut mapper, &mut sampler);
        assert!(sampler.last_route == Some(Route::InsufficientReuse), "large nucleus did not reuse full order");
        assert!(sampler.indices.capacity() == 64, "full ordering leaked into retained prefix scratch");
    }
}

#[test]
fn parallel_reuse_proved_routes_and_rounded_exp_underflow_match_reference() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut mapper = ProbabilityMapper {
        pool: Some(pool),
        attempted: true,
    };
    let mut sampler = NucleusSampler::default();
    for length in [65_536, 65_537, 248_320] {
        let values = vec![0.0; length];
        let sampling = stochastic(0.95);
        exact_terms_and_prefix(&values, &sampling, &mut mapper, &mut sampler);
        exact_draws(&values, &sampling, &[0, 1, 42, 99, u64::MAX], &mut mapper, &mut sampler);
        assert!(sampler.last_route == Some(Route::InsufficientReuse), "parallel fallback reuse not covered");
        let mut concentrated = vec![-80.0; length];
        concentrated[length / 2] = 0.0;
        for temperature in [f32::from_bits(1), f32::MIN_POSITIVE, 0.01, 1.0] {
            let sampling = Sampling::Stochastic {
                temperature,
                top_k: None,
                top_p: Some(0.95),
                min_p: None,
            };
            exact_terms_and_prefix(&concentrated, &sampling, &mut mapper, &mut sampler);
            exact_draws(&concentrated, &sampling, &[0, 1, 42, 99, u64::MAX], &mut mapper, &mut sampler);
            assert!(sampler.last_route == Some(Route::Proved), "concentrated/underflow case not proved");
        }
    }
}

#[test]
fn ineligible_requests_and_failures_allocate_no_shortcut_state_and_clear_values() {
    let mut mapper = ProbabilityMapper {
        pool: None,
        attempted: true,
    };
    let mut sampler = NucleusSampler::default();
    let cases = [
        Sampling::Greedy {},
        Sampling::Stochastic {
            temperature: 1.0,
            top_k: Some(20),
            top_p: Some(0.95),
            min_p: None,
        },
        Sampling::Stochastic {
            temperature: 1.0,
            top_k: Some(0),
            top_p: Some(0.95),
            min_p: None,
        },
        Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1.0),
            min_p: None,
        },
        Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
        },
        Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.95),
            min_p: Some(0.01),
        },
        Sampling::Stochastic {
            temperature: 0.0,
            top_k: None,
            top_p: Some(0.95),
            min_p: None,
        },
        Sampling::Stochastic {
            temperature: f32::INFINITY,
            top_k: None,
            top_p: Some(0.95),
            min_p: None,
        },
    ];
    for sampling in &cases {
        exact_draws(&[0.0, -0.0, 1.0], sampling, &[0, 42], &mut mapper, &mut sampler);
    }
    assert!(
        sampler.indices.capacity() == 0 && sampler.probabilities.capacity() == 0,
        "ineligible request allocated shortcut state"
    );
    for values in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![f32::NEG_INFINITY]] {
        sampler.indices.push(1);
        sampler.probabilities.push(2.0);
        exact_draws(&values, &stochastic(0.95), &[0, 42], &mut mapper, &mut sampler);
    }
    let concentrated = [0.0, -80.0, -80.0];
    exact_draws(&concentrated, &stochastic(0.95), &[42], &mut mapper, &mut sampler);
    let capacities = (sampler.indices.capacity(), sampler.probabilities.capacity());
    for _ in 0..8 {
        exact_draws(&concentrated, &stochastic(0.95), &[42], &mut mapper, &mut sampler);
        assert!(
            capacities == (sampler.indices.capacity(), sampler.probabilities.capacity()),
            "warm sampler capacity changed"
        );
    }
}
