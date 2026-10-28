// SPDX-License-Identifier: Apache-2.0
//! Split attention checks. Only aggregate benchmark durations are emitted.

use std::time::Instant;

use metal::MTLBuffer;
use stark_engine_macros::stark_test;

use super::*;

struct Inputs {
    heads: usize,
    kv_heads: usize,
    dim: usize,
    count: usize,
    capacity: usize,
    query: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    gates: Vec<f32>,
    visible: Vec<bool>,
}

impl Inputs {
    fn new(
        heads: usize,
        kv_heads: usize,
        dim: usize,
        count: usize,
    ) -> Self {
        let capacity = count.next_power_of_two();
        let values =
            |len: usize, seed: usize| (0..len).map(|i| (((i * 13 + seed * 7) % 127) as f32 - 63.0) / 53.0).collect();
        Self {
            heads,
            kv_heads,
            dim,
            count,
            capacity,
            query: values(heads * dim, 1),
            keys: values(capacity * kv_heads * dim, 2),
            values: values(capacity * kv_heads * dim, 3),
            gates: values(heads * dim, 4),
            visible: (0..count).map(|token| token % 5 != 1 && (token / 256) % 3 != 1).collect(),
        }
    }
}

fn encode(
    graph: &mut DecodeGraph,
    inputs: &Inputs,
    buffers: &[DeviceVector; 5],
    output: &DeviceVector,
    split: bool,
) {
    let [query, keys, values, gates, visible] = buffers;
    if split {
        let splits = inputs.count.div_ceil(256);
        let partials = graph.scratch(inputs.heads * splits * (inputs.dim + 2)).unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_split_attention",
                &[query, keys, values, visible, &partials],
                &[inputs.count as u32, inputs.heads as u32, inputs.dim as u32, inputs.kv_heads as u32, splits as u32],
                inputs.heads * splits * 256,
                256,
            )
            .unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_combine_attention",
                &[&partials, gates, output],
                &[inputs.heads as u32, inputs.dim as u32, splits as u32],
                inputs.heads * inputs.dim,
                256,
            )
            .unwrap();
    } else {
        let scores = graph.scratch(inputs.heads * inputs.capacity).unwrap();
        let probabilities = graph.scratch(inputs.heads * inputs.capacity).unwrap();
        let parameters = [
            inputs.count as u32,
            inputs.heads as u32,
            inputs.dim as u32,
            inputs.kv_heads as u32,
            inputs.capacity as u32,
        ];
        graph
            .kernel(
                "qwen_graph_qsa_scores",
                &[query, keys, visible, &scores],
                &parameters,
                inputs.heads * inputs.count,
                64,
            )
            .unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_softmax",
                &[&scores, visible, &probabilities],
                &[inputs.count as u32, inputs.heads as u32, inputs.capacity as u32],
                inputs.heads,
                1,
            )
            .unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_output",
                &[&probabilities, values, gates, visible, output],
                &parameters,
                inputs.heads * inputs.dim,
                256,
            )
            .unwrap();
    }
}

fn run(
    inputs: &Inputs,
    split: bool,
    repeat: usize,
    poison: Option<(usize, usize, f32)>,
) -> (Result<Vec<Vec<f32>>>, f64) {
    let mut graph = DecodeGraph::new().unwrap();
    let upload = |values: &[f32]| {
        if values.len() <= 8_000_000 {
            DecodeGraph::upload(values).unwrap()
        } else {
            // Real K/V storage is persistent cache allocation, not a host
            // activation upload. Initialize the same owned cache shape here.
            assert!(values.iter().all(|value| value.is_finite()));
            let cache = DecodeGraph::zeros(values.len()).unwrap();
            unsafe {
                cache.buffer.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
            cache
        }
    };
    let buffers = [
        upload(&inputs.query),
        upload(&inputs.keys),
        upload(&inputs.values),
        upload(&inputs.gates),
        DecodeGraph::upload(&inputs.visible.iter().map(|&keep| f32::from_bits(u32::from(keep))).collect::<Vec<_>>())
            .unwrap(),
    ];
    if let Some((buffer, index, value)) = poison {
        // Exclusive test storage has no submitted GPU users.
        unsafe { buffers[buffer].buffer.contents().as_ptr().cast::<f32>().add(index).write(value) };
    }
    let output = graph.scratch(inputs.heads * inputs.dim).unwrap();
    for _ in 0..repeat {
        encode(&mut graph, inputs, &buffers, &output, split);
    }
    let start = Instant::now();
    let result = graph.finish(&[output]);
    (result, start.elapsed().as_secs_f64() / repeat as f64)
}

fn reference(inputs: &Inputs) -> Vec<f32> {
    let mut output = vec![0.0; inputs.heads * inputs.dim];
    for head in 0..inputs.heads {
        let kv_head = head / (inputs.heads / inputs.kv_heads);
        let mut scores = vec![f64::NEG_INFINITY; inputs.count];
        for (token, score) in scores.iter_mut().enumerate() {
            if !inputs.visible[token] {
                continue;
            }
            *score = (0..inputs.dim)
                .map(|d| {
                    f64::from(inputs.query[head * inputs.dim + d])
                        * f64::from(inputs.keys[token * inputs.kv_heads * inputs.dim + kv_head * inputs.dim + d])
                })
                .sum::<f64>()
                / (inputs.dim as f64).sqrt();
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for d in 0..inputs.dim {
            let value: f64 = weights
                .iter()
                .enumerate()
                .filter(|(token, _)| inputs.visible[*token])
                .map(|(token, &p)| {
                    p * f64::from(inputs.values[token * inputs.kv_heads * inputs.dim + kv_head * inputs.dim + d])
                })
                .sum();
            let gate = 1.0 / (1.0 + (-f64::from(inputs.gates[head * inputs.dim + d])).exp());
            output[head * inputs.dim + d] = (value / denominator * gate) as f32;
        }
    }
    output
}

#[stark_test]
fn qwen_split_attention_matches_scalar_and_f64_with_masks_grouping_and_tails() {
    for (heads, kv_heads, dim, count) in [
        (4, 2, 3, 1),
        (4, 2, 33, 7),
        (2, 1, 128, 255),
        (4, 2, 128, 256),
        (4, 1, 128, 257),
        (4, 2, 256, 513),
        (8, 2, 128, 4099),
        (8, 2, 128, 16385),
    ] {
        let inputs = Inputs::new(heads, kv_heads, dim, count);
        let scalar = run(&inputs, false, 1, None).0.unwrap().remove(0);
        let split = run(&inputs, true, 1, None).0.unwrap().remove(0);
        let oracle = reference(&inputs);
        assert!(split.iter().zip(&scalar).all(|(&a, &b)| (a - b).abs() <= 2e-5));
        assert!(split.iter().zip(&oracle).all(|(&a, &b)| (a - b).abs() <= 2e-5));
    }
}

#[stark_test]
fn qwen_split_attention_rejects_nonfinite_visible_data_and_empty_masks() {
    let mut inputs = Inputs::new(4, 2, 33, 513);
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for buffer in [0, 1, 2, 3] {
            assert!(
                run(&inputs, true, 1, Some((buffer, 0, poison))).0.is_err(),
                "split attention accepted poisoned buffer {buffer}"
            );
        }
    }
    // Token one is hidden; its value must never be consumed.
    assert!(run(&inputs, true, 1, Some((2, inputs.kv_heads * inputs.dim, f32::NAN))).0.is_ok());
    inputs.visible.fill(false);
    assert!(run(&inputs, true, 1, None).0.is_err());
}

#[stark_test]
fn qwen_split_attention_normalizes_values_before_large_finite_sums() {
    let mut inputs = Inputs::new(2, 1, 33, 513);
    inputs.values.fill(f32::MAX * 0.125);
    let actual = run(&inputs, true, 1, None).0.unwrap().remove(0);
    let expected = reference(&inputs);
    assert!(actual.iter().zip(expected).all(|(&a, b)| a.is_finite() && (a - b).abs() <= b.abs() * 2e-5));
}

#[stark_test]
#[ignore]
fn qwen_split_attention_long_context_microbenchmark() {
    for count in [4096, 16384] {
        let inputs = Inputs::new(32, 4, 256, count);
        for _ in 0..3 {
            let (scalar, scalar_seconds) = run(&inputs, false, 8, None);
            let (split, split_seconds) = run(&inputs, true, 8, None);
            let scalar = scalar.unwrap().remove(0);
            let split = split.unwrap().remove(0);
            let oracle = reference(&inputs);
            let max_error = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!(
                "qwen_attention_error count={count} split_scalar={:.8} split_f64={:.8} scalar_f64={:.8}",
                max_error(&split, &scalar),
                max_error(&split, &oracle),
                max_error(&scalar, &oracle)
            );
            // Qualify against the independent wide oracle. The long scalar
            // softmax's ordered F32 sum is itself measurably less accurate;
            // its error is reported separately rather than treated as truth.
            assert!(split.iter().zip(&oracle).all(|(&a, &b)| (a - b).abs() <= 2e-5));
            eprintln!(
                "qwen_attention_completion_ms count={count} scalar={:.3} split={:.3}",
                scalar_seconds * 1e3,
                split_seconds * 1e3
            );
        }
    }
}
