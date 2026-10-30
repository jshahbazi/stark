//! Exact speculative sampling over the same filtered distributions as decode.
//!
//! Proposals are sampled from q, accepted with min(1, p/q), and rejection
//! samples from the normalized positive part of p-q. Draft distributions live
//! only until their verify round finishes. Draft randomness has a separate
//! stream. Verification stops at the first rejection; cleanup never invokes
//! this sampler. A verified group can contain draws beyond a later text stop.

use std::collections::VecDeque;

use stark_engine::engine::hybrid::{Error, MtpSampler, Result};

use super::{Distribution, ProbabilityMapper, Sampling, sample_with_mapper, sampling_distribution, uniform};

const SPARSE_LOOKUP_LIMIT: usize = 256;
const DRAFT_SEED_DOMAIN: u64 = 0x4d54_505f_4452_4146;

enum ProbabilityLookup {
    Sparse(Vec<(usize, f64)>),
    Dense(Vec<f64>),
}

impl ProbabilityLookup {
    fn new(
        distribution: &Distribution,
        vocabulary: usize,
    ) -> Self {
        if distribution.indices.len() <= SPARSE_LOOKUP_LIMIT {
            let mut entries: Vec<_> = distribution
                .indices
                .iter()
                .zip(&distribution.weights)
                .map(|(&id, &weight)| (id, weight / distribution.total))
                .collect();
            entries.sort_unstable_by_key(|&(id, _)| id);
            Self::Sparse(entries)
        } else {
            let mut values = vec![0.0; vocabulary];
            for (&id, &weight) in distribution.indices.iter().zip(&distribution.weights) {
                values[id] = weight / distribution.total;
            }
            Self::Dense(values)
        }
    }

    fn get(
        &self,
        id: usize,
    ) -> f64 {
        match self {
            Self::Sparse(entries) => entries.binary_search_by_key(&id, |&(id, _)| id).map_or(0.0, |at| entries[at].1),
            Self::Dense(values) => values.get(id).copied().unwrap_or(0.0),
        }
    }
}

struct Proposal {
    token: u32,
    probabilities: ProbabilityLookup,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Drafting,
    Verifying,
    Finished,
}

pub(super) struct ExactMtpSampler<'a> {
    sampling: &'a Sampling,
    rng: &'a mut u64,
    draft_rng: u64,
    mapper: &'a mut ProbabilityMapper,
    proposals: VecDeque<Proposal>,
    phase: Phase,
    vocabulary: Option<usize>,
    failed: bool,
}

impl<'a> ExactMtpSampler<'a> {
    pub(super) fn new(
        sampling: &'a Sampling,
        rng: &'a mut u64,
        mapper: &'a mut ProbabilityMapper,
    ) -> Self {
        let draft_rng = *rng ^ DRAFT_SEED_DOMAIN;
        Self {
            sampling,
            rng,
            draft_rng,
            mapper,
            proposals: VecDeque::new(),
            phase: Phase::Drafting,
            vocabulary: None,
            failed: false,
        }
    }

    /// The prompt's first draw has no proposal. It uses exactly the ordinary
    /// sampler, before the first draft round begins.
    pub(super) fn sample_initial(
        &mut self,
        logits: &[f32],
    ) -> std::result::Result<u32, &'static str> {
        if self.failed || self.phase != Phase::Drafting || !self.proposals.is_empty() || self.vocabulary.is_some() {
            self.failed = true;
            return Err("native_generation_failed");
        }
        let result = sample_with_mapper(logits, self.sampling, self.rng, Some(&mut *self.mapper));
        self.failed |= result.is_err();
        if result.is_ok() {
            self.vocabulary = Some(logits.len());
        }
        result
    }

    fn distribution(
        &mut self,
        logits: &[f32],
    ) -> Result<Distribution> {
        if self.failed || self.vocabulary.is_some_and(|size| size != logits.len()) {
            return Err(Error::Invalid("MTP sampler vocabulary or state"));
        }
        let distribution = sampling_distribution(logits, self.sampling, Some(&mut *self.mapper))
            .map_err(|_| Error::Invalid("MTP sampling distribution"))?;
        self.vocabulary = Some(logits.len());
        Ok(distribution)
    }

    fn draft_inner(
        &mut self,
        logits: &[f32],
    ) -> Result<u32> {
        if self.phase != Phase::Drafting {
            return Err(Error::Invalid("MTP draft after verification"));
        }
        let distribution = self.distribution(logits)?;
        let token = draw(&distribution, &mut self.draft_rng, matches!(self.sampling, Sampling::Greedy {}))?;
        self.proposals.push_back(Proposal {
            token,
            probabilities: ProbabilityLookup::new(&distribution, logits.len()),
        });
        Ok(token)
    }

    fn target_inner(
        &mut self,
        logits: &[f32],
        proposal: Option<u32>,
    ) -> Result<u32> {
        if self.phase == Phase::Finished {
            return Err(Error::Invalid("MTP verify after round completion"));
        }
        let distribution = self.distribution(logits)?;
        self.phase = Phase::Verifying;
        let greedy = matches!(self.sampling, Sampling::Greedy {});
        match proposal {
            None => {
                if !self.proposals.is_empty() {
                    return Err(Error::Invalid("MTP bonus before proposal verification"));
                }
                self.phase = Phase::Finished;
                draw(&distribution, self.rng, greedy)
            },
            Some(token) => {
                let drafted = self.proposals.pop_front().ok_or(Error::Invalid("MTP proposal distribution missing"))?;
                if token != drafted.token {
                    return Err(Error::Invalid("MTP proposal distribution mismatch"));
                }
                let p = probability(&distribution, token as usize);
                let q = drafted.probabilities.get(token as usize);
                let acceptance = acceptance_probability(p, q)?;
                if acceptance == 1.0 || (!greedy && uniform(self.rng) < acceptance) {
                    return Ok(token);
                }
                // Future proposals were conditioned on a rejected prefix.
                self.proposals.clear();
                self.phase = Phase::Finished;
                let residual = residual_distribution(&distribution, &drafted.probabilities)?;
                draw(&residual, self.rng, greedy)
            },
        }
    }
}

impl MtpSampler for ExactMtpSampler<'_> {
    fn begin_round(&mut self) {
        self.proposals.clear();
        self.phase = Phase::Drafting;
    }

    fn draft(
        &mut self,
        logits: &[f32],
    ) -> Result<u32> {
        let result = self.draft_inner(logits);
        self.failed |= result.is_err();
        result
    }

    fn target(
        &mut self,
        logits: &[f32],
        proposal: Option<u32>,
    ) -> Result<u32> {
        let result = self.target_inner(logits, proposal);
        self.failed |= result.is_err();
        result
    }
}

fn probability(
    distribution: &Distribution,
    id: usize,
) -> f64 {
    distribution
        .indices
        .iter()
        .zip(&distribution.weights)
        .find_map(|(&candidate, &weight)| (candidate == id).then_some(weight / distribution.total))
        .unwrap_or(0.0)
}

fn acceptance_probability(
    p: f64,
    q: f64,
) -> Result<f64> {
    if !p.is_finite() || !q.is_finite() || p < 0.0 || q <= 0.0 {
        return Err(Error::Invalid("MTP proposal probability"));
    }
    Ok((p / q).min(1.0))
}

fn residual_distribution(
    target: &Distribution,
    draft: &ProbabilityLookup,
) -> Result<Distribution> {
    let mut indices = Vec::with_capacity(target.indices.len());
    let mut weights = Vec::with_capacity(target.weights.len());
    let mut total = 0.0;
    for (&id, &weight) in target.indices.iter().zip(&target.weights) {
        let residual = (weight / target.total - draft.get(id)).max(0.0);
        if residual > 0.0 {
            indices.push(id);
            weights.push(residual);
            total += residual;
        }
    }
    if !total.is_finite() || total <= 0.0 || indices.is_empty() {
        return Err(Error::Invalid("MTP rejection residual"));
    }
    Ok(Distribution {
        indices,
        weights,
        total,
    })
}

fn draw(
    distribution: &Distribution,
    rng: &mut u64,
    greedy: bool,
) -> Result<u32> {
    if distribution.indices.is_empty() || !distribution.total.is_finite() || distribution.total <= 0.0 {
        return Err(Error::Invalid("MTP draw distribution"));
    }
    if greedy {
        return Ok(distribution.indices[0] as u32);
    }
    let target = uniform(rng) * distribution.total;
    let mut mass = 0.0;
    for (&id, &weight) in distribution.indices.iter().zip(&distribution.weights) {
        mass += weight;
        if target < mass {
            return Ok(id as u32);
        }
    }
    Ok(*distribution.indices.last().ok_or(Error::Invalid("MTP draw distribution"))? as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> Sampling {
        Sampling::Stochastic {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
        }
    }

    fn from_weights(weights: &[f64]) -> Distribution {
        Distribution {
            indices: (0..weights.len()).collect(),
            weights: weights.to_vec(),
            total: weights.iter().sum(),
        }
    }

    #[test]
    fn rejection_equation_recovers_every_target_mass() {
        let cases: &[(&[f64], &[f64])] = &[
            (&[0.2, 0.3, 0.5], &[0.6, 0.1, 0.3]),
            (&[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]),
            (&[0.25, 0.5, 0.25], &[0.25, 0.5, 0.25]),
            (&[0.25, 0.0, 0.75, 0.0], &[0.0, 0.5, 0.0, 0.5]),
            (&[1e-200, 1.0, 0.5], &[0.1, 0.2, 0.7]),
        ];
        for &(p, q) in cases {
            let target = from_weights(p);
            let draft = from_weights(q);
            let lookup = ProbabilityLookup::new(&draft, q.len());
            let accepted: Vec<_> = (0..p.len())
                .map(|id| {
                    let q = lookup.get(id);
                    if q > 0.0 {
                        q * acceptance_probability(probability(&target, id), q).unwrap()
                    } else {
                        0.0
                    }
                })
                .collect();
            let rejection = 1.0 - accepted.iter().sum::<f64>();
            let residual = residual_distribution(&target, &lookup);
            for (id, &accepted_mass) in accepted.iter().enumerate() {
                let rejected_mass = residual.as_ref().map_or(0.0, |r| rejection * probability(r, id));
                assert!((accepted_mass + rejected_mass - probability(&target, id)).abs() < 2e-15);
            }
        }
    }

    #[test]
    fn residual_excludes_rejected_token_and_honors_support_holes() {
        let target = from_weights(&[0.1, 0.0, 0.4, 0.5]);
        let draft = ProbabilityLookup::new(&from_weights(&[0.8, 0.2, 0.0, 0.0]), 4);
        let residual = residual_distribution(&target, &draft).unwrap();
        assert!(residual.indices == [2, 3]);
        assert!(probability(&residual, 0) == 0.0 && probability(&residual, 1) == 0.0);
        assert!((probability(&residual, 2) - 4.0 / 9.0).abs() < 1e-15);
        assert!(acceptance_probability(0.0, 1.0).unwrap() == 0.0);
        assert!(acceptance_probability(1.0, 0.0).is_err());
    }

    #[test]
    fn matching_distributions_accept_and_bonus_uses_target_stream() {
        let sampling = options();
        let logits = [0.5, 1.5, -0.5];
        let mut rng = 37;
        let mut reference_rng = rng;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
        sampler.begin_round();
        let first = sampler.draft(&logits).unwrap();
        let second = sampler.draft(&logits).unwrap();
        assert!(sampler.target(&logits, Some(first)).unwrap() == first);
        assert!(sampler.target(&logits, Some(second)).unwrap() == second);
        let bonus = sampler.target(&logits, None).unwrap();
        let reference = super::super::sample(&logits, &sampling, &mut reference_rng).unwrap();
        assert!(bonus == reference && *sampler.rng == reference_rng);
    }

    #[test]
    fn exact_sampling_uses_the_existing_temperature_and_filter_policy() {
        let logits = [1.0, 1.75, 0.5, -1.0, 2.0, 1.5];
        let sampling = Sampling::Stochastic {
            temperature: 0.7,
            top_k: Some(4),
            top_p: Some(0.8),
            min_p: Some(0.1),
        };
        for seed in 0..256 {
            let mut rng = seed;
            let mut reference_rng = seed;
            let mut mapper = ProbabilityMapper::default();
            let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
            sampler.begin_round();
            let draw = sampler.target(&logits, None).unwrap();
            assert!(draw == super::super::sample(&logits, &sampling, &mut reference_rng).unwrap());
            assert!(*sampler.rng == reference_rng);
        }
    }

    #[test]
    fn prompt_draw_matches_ordinary_sampling_and_can_only_run_once() {
        let sampling = options();
        let logits = [0.5, -1.0, 1.25];
        let mut rng = 98;
        let mut reference_rng = rng;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
        let actual = sampler.sample_initial(&logits).unwrap();
        let expected = super::super::sample(&logits, &sampling, &mut reference_rng).unwrap();
        assert!(actual == expected && *sampler.rng == reference_rng);
        assert!(sampler.sample_initial(&logits).is_err());
        sampler.begin_round();
        assert!(sampler.draft(&logits).is_err());
    }

    #[test]
    fn sampled_rejection_recovers_target_distribution() {
        let cases = [
            (options(), vec![0.1_f32.ln(), 0.3_f32.ln(), 0.6_f32.ln()], vec![0.7_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()]),
            (
                Sampling::Stochastic {
                    temperature: 0.7,
                    top_k: Some(3),
                    top_p: Some(0.9),
                    min_p: Some(0.1),
                },
                vec![1.0, 2.0, -5.0, 0.0, 3.0],
                vec![1.8, 2.0, -5.0, 3.0, -4.0],
            ),
        ];
        for (sampling, target_logits, draft_logits) in cases {
            let target = sampling_distribution(&target_logits, &sampling, None).unwrap();
            let mut counts = vec![0_usize; target_logits.len()];
            let mut mapper = ProbabilityMapper::default();
            let trials = 60_000;
            for seed in 0..trials {
                let mut rng = seed as u64;
                let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
                sampler.begin_round();
                let proposal = sampler.draft(&draft_logits).unwrap();
                let output = sampler.target(&target_logits, Some(proposal)).unwrap();
                counts[output as usize] += 1;
            }
            for (id, &count) in counts.iter().enumerate() {
                assert!((count as f64 / trials as f64 - probability(&target, id)).abs() < 0.012);
                if probability(&target, id) == 0.0 {
                    assert!(count == 0);
                }
            }
        }
    }

    #[test]
    fn tiny_residual_stays_normalized_and_excludes_the_rejection() {
        let delta = f64::EPSILON;
        let target = from_weights(&[0.5 + delta, 0.5 - delta]);
        let draft = ProbabilityLookup::new(&from_weights(&[0.5, 0.5]), 2);
        let residual = residual_distribution(&target, &draft).unwrap();
        assert!(acceptance_probability(probability(&target, 1), draft.get(1)).unwrap() < 1.0);
        assert!(residual.total > 0.0 && residual.total < 1e-14);
        assert!(probability(&residual, 0) == 1.0 && probability(&residual, 1) == 0.0);
    }

    #[test]
    fn underflowed_candidates_remain_impossible_when_rejecting() {
        let sampling = options();
        let target_logits = [f32::MAX, -f32::MAX, 0.0];
        let draft_logits = [-f32::MAX, f32::MAX, 0.0];
        for seed in 0..256 {
            let mut rng = seed;
            let mut mapper = ProbabilityMapper::default();
            let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
            sampler.begin_round();
            let proposal = sampler.draft(&draft_logits).unwrap();
            assert!(proposal == 1);
            assert!(sampler.target(&target_logits, Some(proposal)).unwrap() == 0);
        }
    }

    #[test]
    fn full_vocabulary_is_not_capped_to_vendor_candidate_limit() {
        let sampling = options();
        let logits = vec![0.0; 2048];
        let mut rng = 3;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
        sampler.begin_round();
        let proposal = sampler.draft(&logits).unwrap();
        let distribution = sampler.proposals.front().unwrap();
        assert!(distribution.probabilities.get(2047) == 1.0 / 2048.0);
        assert!(sampler.target(&logits, Some(proposal)).unwrap() == proposal);
    }

    #[test]
    fn disjoint_greedy_support_rejects_and_discards_later_proposals() {
        let sampling = Sampling::Greedy {};
        let mut rng = 11;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
        sampler.begin_round();
        let first = sampler.draft(&[2.0, 1.0]).unwrap();
        sampler.draft(&[2.0, 1.0]).unwrap();
        assert!(sampler.target(&[1.0, 2.0], Some(first)).unwrap() == 1);
        assert!(sampler.proposals.is_empty() && *sampler.rng == 11);
        sampler.begin_round();
        assert!(sampler.target(&[1.0, 2.0], None).unwrap() == 1);
    }

    #[test]
    fn sampler_is_seeded_and_draft_randomness_does_not_advance_target_rng() {
        fn replay(seed: u64) -> Vec<u32> {
            let sampling = options();
            let mut rng = seed;
            let mut mapper = ProbabilityMapper::default();
            let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
            let mut draws = Vec::new();
            for _ in 0..16 {
                sampler.begin_round();
                let first = sampler.draft(&[1.0, 2.0, 3.0]).unwrap();
                let second = sampler.draft(&[1.0, 2.0, 3.0]).unwrap();
                let output = sampler.target(&[3.0, 2.0, 1.0], Some(first)).unwrap();
                draws.push(output);
                if output == first {
                    let output = sampler.target(&[3.0, 2.0, 1.0], Some(second)).unwrap();
                    draws.push(output);
                    if output == second {
                        draws.push(sampler.target(&[3.0, 2.0, 1.0], None).unwrap());
                    }
                }
            }
            draws
        }
        assert!(replay(27) == replay(27));
        assert!(replay(27) != replay(28));
        let sampling = options();
        let mut rng = 17;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
        sampler.draft(&[0.0, 0.0]).unwrap();
        assert!(*sampler.rng == 17);
    }

    #[test]
    fn invalid_order_nonfinite_and_changed_geometry_fail_closed() {
        let sampling = options();
        for case in 0..5 {
            let mut rng = 0;
            let mut mapper = ProbabilityMapper::default();
            let mut sampler = ExactMtpSampler::new(&sampling, &mut rng, &mut mapper);
            sampler.begin_round();
            let proposal = sampler.draft(&[0.0, 1.0]).unwrap();
            let result = match case {
                0 => sampler.target(&[0.0, 1.0], Some(1 - proposal)),
                1 => sampler.target(&[0.0, 1.0], None),
                2 => sampler.target(&[0.0, 1.0, 2.0], Some(proposal)),
                3 => sampler.target(&[f32::NAN, 1.0], Some(proposal)),
                _ => {
                    sampler.target(&[0.0, 1.0], Some(proposal)).unwrap();
                    sampler.draft(&[0.0, 1.0])
                },
            };
            assert!(result.is_err());
            sampler.begin_round();
            assert!(sampler.target(&[0.0, 1.0], None).is_err());
        }
    }
}
