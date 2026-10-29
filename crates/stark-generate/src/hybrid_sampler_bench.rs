//! Private component comparison on identical real model readouts.

use std::{hint::black_box, path::PathBuf, time::Instant};

use stark_engine::engine::hybrid::{Model, math};

use super::{descending_sample_key, sample, sampling_indices};
use crate::generation::Sampling;

fn timed_order(
    logits: &[f32],
    reference: bool,
) -> (Vec<usize>, f64) {
    let started = Instant::now();
    let indices = if reference {
        math::topk(logits, logits.len()).unwrap_or_else(|_| panic!("reference selection failed"))
    } else {
        sampling_indices(logits, None).unwrap_or_else(|_| panic!("radix selection failed"))
    };
    let elapsed = started.elapsed().as_secs_f64();
    (black_box(indices), elapsed)
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    (values[middle - 1] + values[middle]) / 2.0
}

/// Source-bound controls from the selector immediately before the packed-key
/// and bounded-prefix changes. This code exists only in the explicit test.
fn previous_sampling_indices(
    logits: &[f32],
    top_k: Option<u32>,
) -> Vec<usize> {
    if let Some(k) = top_k {
        let k = (k as usize).min(logits.len());
        let compare = |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b));
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.select_nth_unstable_by(k - 1, compare);
        indices.truncate(k);
        indices.sort_unstable_by(compare);
        return indices;
    }
    let mut histograms = [[0_usize; 256]; 4];
    for &logit in logits {
        let key = descending_sample_key(logit);
        for (pass, histogram) in histograms.iter_mut().enumerate() {
            histogram[((key >> (pass * 8)) & 0xff) as usize] += 1;
        }
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    let mut scratch = vec![0; logits.len()];
    for (pass, histogram) in histograms.iter().enumerate() {
        let mut offsets = [0_usize; 256];
        let mut end = 0;
        for (offset, &count) in offsets.iter_mut().zip(histogram) {
            *offset = end;
            end += count;
        }
        for &index in &indices {
            let digit = ((descending_sample_key(logits[index]) >> (pass * 8)) & 0xff) as usize;
            let destination = &mut offsets[digit];
            scratch[*destination] = index;
            *destination += 1;
        }
        std::mem::swap(&mut indices, &mut scratch);
    }
    indices
}

fn timed_previous_selection(
    logits: &[f32],
    top_k: Option<u32>,
    previous: bool,
) -> (Vec<usize>, f64) {
    let started = Instant::now();
    let indices = if previous {
        previous_sampling_indices(logits, top_k)
    } else {
        sampling_indices(logits, top_k).unwrap_or_else(|_| panic!("candidate selection failed"))
    };
    (black_box(indices), started.elapsed().as_secs_f64())
}

#[test]
#[ignore = "requires the complete private model, public corpus and exclusive GPU slot"]
fn private_model_selection_cpu_benchmark() {
    // This explicitly selected native benchmark emits only authored labels on
    // failure; model readouts and token vectors never enter panic diagnostics.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| eprintln!("private selection benchmark failed")));
    let model_path = PathBuf::from(std::env::var_os("STARK_LILY_TEST_MODEL_PATH").expect("model path required"));
    let corpus_path = PathBuf::from(std::env::var_os("STARK_SAMPLE_CORPUS_PATH").expect("corpus path required"));
    let corpus = std::fs::read_to_string(corpus_path).unwrap_or_else(|_| panic!("corpus unavailable"));
    let mut model = Model::load(&model_path).unwrap_or_else(|_| panic!("model unavailable"));
    let encoded = model.tokenizer().encode(corpus, false).unwrap_or_else(|_| panic!("corpus tokenization failed"));
    assert!(encoded.len() >= 4096, "corpus is too short");
    let input = &encoded.get_ids()[..4096];
    for &token in &input[..input.len() - 1] {
        model.ingest(token).unwrap_or_else(|_| panic!("prefill admission failed"));
    }
    let mut logits = model.step(input[input.len() - 1]).unwrap_or_else(|_| panic!("prefill failed"));
    let sampling = Sampling::Stochastic {
        temperature: 1.0,
        top_k: None,
        top_p: Some(0.95),
        min_p: None,
    };
    let mut rng = 42;
    let mut reference_times = Vec::with_capacity(256);
    let mut radix_times = Vec::with_capacity(256);
    let mut previous_radix_times = Vec::with_capacity(256);
    let mut packed_radix_times = Vec::with_capacity(256);
    let mut previous_partial_times = Vec::with_capacity(256);
    let mut bounded_times = Vec::with_capacity(256);
    for step in 0usize..256 {
        let ((reference, reference_time), (radix, radix_time)) = if step.is_multiple_of(2) {
            let reference = timed_order(&logits, true);
            let radix = timed_order(&logits, false);
            (reference, radix)
        } else {
            let radix = timed_order(&logits, false);
            let reference = timed_order(&logits, true);
            (reference, radix)
        };
        assert!(reference == radix, "candidate ordering changed");
        reference_times.push(reference_time);
        radix_times.push(radix_time);
        drop(reference);
        drop(radix);
        for (top_k, previous_times, candidate_times) in [
            (None, &mut previous_radix_times, &mut packed_radix_times),
            (Some(20), &mut previous_partial_times, &mut bounded_times),
        ] {
            let ((previous, previous_time), (candidate, candidate_time)) = if step.is_multiple_of(2) {
                let previous = timed_previous_selection(&logits, top_k, true);
                let candidate = timed_previous_selection(&logits, top_k, false);
                (previous, candidate)
            } else {
                let candidate = timed_previous_selection(&logits, top_k, false);
                let previous = timed_previous_selection(&logits, top_k, true);
                (previous, candidate)
            };
            assert!(previous == candidate, "previous and candidate selection orders changed");
            previous_times.push(previous_time);
            candidate_times.push(candidate_time);
        }
        let token = sample(&logits, &sampling, &mut rng).unwrap_or_else(|_| panic!("sampling failed"));
        if step + 1 < 256 {
            logits = model.step(token).unwrap_or_else(|_| panic!("decode failed"));
        }
    }
    let report = serde_json::json!({
        "scope": "qwen38_private_sampler_selection",
        "input_tokens": input.len(),
        "vocabulary": model.vocab_size(),
        "readouts": reference_times.len(),
        "comparison_order": "alternating_ab_ba",
        "reference_mean_ms": reference_times.iter().sum::<f64>() * 1000.0 / 256.0,
        "radix_mean_ms": radix_times.iter().sum::<f64>() * 1000.0 / 256.0,
        "reference_median_ms": median(&reference_times) * 1000.0,
        "radix_median_ms": median(&radix_times) * 1000.0,
        "previous_byte_radix_mean_ms": previous_radix_times.iter().sum::<f64>() * 1000.0 / 256.0,
        "packed_radix_mean_ms": packed_radix_times.iter().sum::<f64>() * 1000.0 / 256.0,
        "previous_byte_radix_median_ms": median(&previous_radix_times) * 1000.0,
        "packed_radix_median_ms": median(&packed_radix_times) * 1000.0,
        "previous_partial_top20_mean_ms": previous_partial_times.iter().sum::<f64>() * 1000.0 / 256.0,
        "bounded_top20_mean_ms": bounded_times.iter().sum::<f64>() * 1000.0 / 256.0,
        "previous_partial_top20_median_ms": median(&previous_partial_times) * 1000.0,
        "bounded_top20_median_ms": median(&bounded_times) * 1000.0,
        "all_index_orders_equal": true,
        "passed": true,
    });
    println!("{report}");
    std::panic::set_hook(previous_hook);
}
