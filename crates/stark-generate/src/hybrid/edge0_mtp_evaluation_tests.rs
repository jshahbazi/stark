// SPDX-License-Identifier: Apache-2.0
//! Opt-in actual Edge0 evaluation. Complete readouts and token sequences stay
//! in RAM. The ordinary generation worker never calls the evaluation hook.

use std::path::PathBuf;

use stark_engine::engine::hybrid::{Model, MtpStats};

use super::*;

struct Recording<'a> {
    exact: ExactMtpSampler<'a>,
    target_rows: Vec<Vec<u32>>,
    draft_calls: usize,
}

impl MtpSampler for Recording<'_> {
    fn begin_round(&mut self) {
        self.exact.begin_round();
    }

    fn draft(
        &mut self,
        logits: &[f32],
    ) -> Result<u32> {
        self.draft_calls += 1;
        self.exact.draft(logits)
    }

    fn target(
        &mut self,
        logits: &[f32],
        proposal: Option<u32>,
    ) -> Result<u32> {
        assert!(logits.len() == 248_320 && logits.iter().all(|value| value.is_finite()), "evaluation_target_shape");
        self.target_rows.push(logits.iter().map(|value| value.to_bits()).collect());
        self.exact.target(logits, proposal)
    }
}

fn ordinary_rows(
    model: &mut Model,
    prompt: &[u32],
    outputs: &[u32],
    count: usize,
) -> Vec<Vec<u32>> {
    model.reset();
    for &input in &prompt[..prompt.len() - 1] {
        model.step(input).unwrap_or_else(|_| panic!("evaluation_reference_ingest"));
    }
    let mut rows = Vec::new();
    let mut input = *prompt.last().unwrap_or_else(|| panic!("evaluation_prompt"));
    for index in 0..count {
        let row = model.step(input).unwrap_or_else(|_| panic!("evaluation_reference_step"));
        rows.push(row.into_iter().map(f32::to_bits).collect());
        if index + 1 < count {
            input = outputs[index];
        }
    }
    rows
}

fn assert_stats(
    stats: MtpStats,
    budget: usize,
    draft_calls: usize,
) {
    assert!(
        stats.verified_rows == budget.saturating_sub(1)
            && stats.proposed == draft_calls
            && stats.accepted <= stats.proposed
            && stats.proposed <= 3 * stats.rounds
            && [stats.draft_seconds, stats.verify_seconds, stats.rollback_seconds]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
        "evaluation_stats"
    );
    if budget <= 2 {
        assert!(draft_calls == 0, "evaluation_tail_no_proposal");
    } else {
        assert!(draft_calls > 0, "evaluation_drafts_exercised");
    }
}

#[test]
#[ignore = "requires pinned local Edge0 artifact and external serialized resource watchdog"]
fn edge0_actual_exact_sampling_and_restoration() {
    assert!(
        std::env::var("STARK_EDGE0_REFERENCE").ok().as_deref() == Some("1")
            && std::env::var("STARK_EDGE0_METAL").ok().as_deref() == Some("1")
            && std::env::var("STARK_EDGE0_RESIDENT").ok().as_deref() != Some("1"),
        "evaluation_policy"
    );
    let path =
        PathBuf::from(std::env::var_os("STARK_EDGE0_SOURCE_PATH").unwrap_or_else(|| panic!("evaluation_source")));
    let mut model = Model::load(&path).unwrap_or_else(|_| panic!("evaluation_model_load"));
    assert!(
        model.supports_mtp()
            && model.mtp_capabilities().is_some_and(|caps| caps.max_drafts == 3)
            && model.default_mtp_drafts() == 0,
        "evaluation_worker_mtp_opt_in"
    );
    let text = "A quiet morning brings clear light through the window. ".repeat(4);
    let encoding = model.tokenizer().encode(text, false).unwrap_or_else(|_| panic!("evaluation_tokenizer"));
    let prompt = encoding.get_ids().get(..12).unwrap_or_else(|| panic!("evaluation_prompt_length"));
    let stochastic = Sampling::Stochastic {
        temperature: 0.7,
        top_k: None,
        top_p: Some(0.9),
        min_p: None,
    };
    let greedy = Sampling::Greedy {};
    let mut cases = 0;
    let mut paired_rows = 0;
    let mut proposals = 0;
    let mut accepted = 0;
    let mut depths_exercised = 0;
    for sampling in [&greedy, &stochastic] {
        for depth in 1..=3 {
            let mut rng = 0x45_44_47_45_30;
            let mut mapper = ProbabilityMapper::default();
            let mut sampler = Recording {
                exact: ExactMtpSampler::new(sampling, &mut rng, &mut mapper),
                target_rows: Vec::new(),
                draft_calls: 0,
            };
            let mut outputs = Vec::new();
            let stats = model
                .evaluate_edge0_mtp(prompt, 5, depth, &mut sampler, &mut |token| {
                    outputs.push(token);
                    Ok(true)
                })
                .unwrap_or_else(|_| panic!("evaluation_sampled_case"));
            assert!(outputs.len() == 5 && sampler.target_rows.len() == 5, "evaluation_output_budget");
            assert_stats(stats, 5, sampler.draft_calls);
            let reference = ordinary_rows(&mut model, prompt, &outputs, outputs.len());
            assert!(sampler.target_rows == reference, "evaluation_serial_target_exact");
            cases += 1;
            paired_rows += outputs.len();
            proposals += stats.proposed;
            accepted += stats.accepted;
            depths_exercised |= 1 << (depth - 1);
        }
    }
    for budget in 0..=3 {
        let mut rng = 17;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = Recording {
            exact: ExactMtpSampler::new(&greedy, &mut rng, &mut mapper),
            target_rows: Vec::new(),
            draft_calls: 0,
        };
        let mut outputs = Vec::new();
        let stats = model
            .evaluate_edge0_mtp(prompt, budget, 3, &mut sampler, &mut |token| {
                outputs.push(token);
                Ok(true)
            })
            .unwrap_or_else(|_| panic!("evaluation_tail_case"));
        assert!(outputs.len() == budget && sampler.target_rows.len() == budget, "evaluation_tail_budget");
        assert_stats(stats, budget, sampler.draft_calls);
        if budget > 0 {
            assert!(
                sampler.target_rows == ordinary_rows(&mut model, prompt, &outputs, budget),
                "evaluation_tail_exact"
            );
        }
        cases += 1;
        paired_rows += budget;
    }
    // A callback stop may land inside an already verified group. The returned
    // state must leave the last delivered output pending, just like decode.
    let mut rng = 23;
    let mut mapper = ProbabilityMapper::default();
    let mut sampler = Recording {
        exact: ExactMtpSampler::new(&greedy, &mut rng, &mut mapper),
        target_rows: Vec::new(),
        draft_calls: 0,
    };
    let mut outputs = Vec::new();
    model
        .evaluate_edge0_mtp(prompt, 5, 3, &mut sampler, &mut |token| {
            outputs.push(token);
            Ok(outputs.len() < 2)
        })
        .unwrap_or_else(|_| panic!("evaluation_early_stop"));
    assert!(outputs.len() == 2 && sampler.target_rows.len() >= 2, "evaluation_early_stop_count");
    let next = model.step(outputs[1]).unwrap_or_else(|_| panic!("evaluation_early_stop_next"));
    let reference = ordinary_rows(&mut model, prompt, &outputs, 3);
    assert!(
        sampler.target_rows[..2] == reference[..2] && next.iter().map(|v| v.to_bits()).eq(reference[2].iter().copied()),
        "evaluation_early_stop_boundary"
    );
    paired_rows += 3;
    cases += 1;

    // Callback cancellation and sampler failure release the leased head. Each
    // next evaluation must recover on this same loaded model, without reload.
    for fail_sampler in [false, true] {
        struct FailingSampler<'a> {
            inner: ExactMtpSampler<'a>,
        }
        impl MtpSampler for FailingSampler<'_> {
            fn begin_round(&mut self) {
                self.inner.begin_round();
            }
            fn draft(
                &mut self,
                _: &[f32],
            ) -> Result<u32> {
                Err(Error::Invalid("evaluation_injected_sampler"))
            }
            fn target(
                &mut self,
                logits: &[f32],
                proposal: Option<u32>,
            ) -> Result<u32> {
                self.inner.target(logits, proposal)
            }
        }
        let mut rng = 29;
        let mut mapper = ProbabilityMapper::default();
        let mut exact = ExactMtpSampler::new(&greedy, &mut rng, &mut mapper);
        let mut failing_rng = 29;
        let mut failing_mapper = ProbabilityMapper::default();
        let mut failing = FailingSampler {
            inner: ExactMtpSampler::new(&greedy, &mut failing_rng, &mut failing_mapper),
        };
        let sampler: &mut dyn MtpSampler = if fail_sampler {
            &mut failing
        } else {
            &mut exact
        };
        let mut delivered = 0;
        let result = model.evaluate_edge0_mtp(prompt, 5, 3, sampler, &mut |_| {
            delivered += 1;
            if !fail_sampler && delivered == 2 {
                Err(Error::Invalid("evaluation_injected_callback"))
            } else {
                Ok(true)
            }
        });
        assert!(
            result.is_err()
                && delivered
                    == if fail_sampler {
                        1
                    } else {
                        2
                    },
            "evaluation_failure_propagated"
        );
        cases += 1;
        let mut rng = 31;
        let mut mapper = ProbabilityMapper::default();
        let mut sampler = Recording {
            exact: ExactMtpSampler::new(&greedy, &mut rng, &mut mapper),
            target_rows: Vec::new(),
            draft_calls: 0,
        };
        let mut outputs = Vec::new();
        model
            .evaluate_edge0_mtp(prompt, 1, 3, &mut sampler, &mut |token| {
                outputs.push(token);
                Ok(true)
            })
            .unwrap_or_else(|_| panic!("evaluation_same_model_recovery"));
        assert!(
            outputs.len() == 1 && sampler.target_rows == ordinary_rows(&mut model, prompt, &outputs, 1),
            "evaluation_recovery_exact"
        );
        paired_rows += 1;
        cases += 1;
    }
    assert!(
        cases == 15 && paired_rows == 41 && depths_exercised == 7 && proposals > 0 && accepted <= proposals,
        "evaluation_gate_counts"
    );
    assert!(model.supports_mtp() && model.default_mtp_drafts() == 0, "evaluation_worker_default_stays_off");
    println!(
        "{}",
        serde_json::json!({
            "actual_artifact_forward": true, "exact_mtp_sampler": true,
            "cases": cases, "serial_target_bit_exact_pairs": paired_rows,
            "draft_depths_exercised": 3, "sampled_full_cases": 6, "tail_cases": 4,
            "early_stop_cases": 1, "failure_cases": 2, "same_model_recoveries": 2,
        "proposed": proposals, "accepted": accepted, "worker_mtp_available": true
        })
    );
}
