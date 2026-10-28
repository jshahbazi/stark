// SPDX-License-Identifier: Apache-2.0
//! Focused Metal control-kernel checks; full Qwen graph parity lives in qwen.rs.

use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn qwen_graph_router_preserves_total_order_ties_and_original_softmax_scores() {
    let scores = [1.0, 1.0, -0.0, 0.0, 2.0, -1.0];
    let mut graph = DecodeGraph::new().unwrap();
    let input = DecodeGraph::upload(&scores).unwrap();
    let ids = DecodeGraph::zeros(5).unwrap();
    let probabilities = DecodeGraph::zeros(5).unwrap();
    graph.kernel("qwen_graph_router", &[&input, &ids, &probabilities], &[scores.len() as u32, 5], 1, 1).unwrap();
    let values = graph.finish(&[ids, probabilities]).unwrap();
    let selected = values[0].iter().map(|value| value.to_bits() as usize).collect::<Vec<_>>();
    assert_eq!(selected, [4, 0, 1, 3, 2]);
    let expected =
        crate::engine::hybrid::math::softmax(&[scores[4], scores[0], scores[1], scores[3], scores[2]]).unwrap();
    assert!(values[1].iter().zip(expected).all(|(&actual, expected)| (actual - expected).abs() < 1e-6));
}

#[stark_test]
fn qwen_graph_qsa_pool_selection_keeps_complete_block_and_current_partial_tail() {
    let index_projection = [1.0, 0.0, 1.0, 0.0]; // one index query, then current index key
    let index_keys = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let q_norm = [1.0, 1.0];
    let k_norm = [1.0, 1.0];
    let mut graph = DecodeGraph::new().unwrap();
    let index_projection = DecodeGraph::upload(&index_projection).unwrap();
    let index_keys = DecodeGraph::upload(&index_keys).unwrap();
    let q_norm = DecodeGraph::upload(&q_norm).unwrap();
    let k_norm = DecodeGraph::upload(&k_norm).unwrap();
    let visible = DecodeGraph::zeros(5).unwrap();
    let block_scores = DecodeGraph::zeros(2).unwrap();
    let top_ids = DecodeGraph::zeros(1).unwrap();
    let top_scores = DecodeGraph::zeros(1).unwrap();
    let parameters = [5, 1, 2, 2, 0, 1e-6f32.to_bits(), 10_000.0f32.to_bits(), 2, 2];
    graph
        .kernel(
            "qwen_graph_qsa_visible",
            &[&index_projection, &index_keys, &q_norm, &k_norm, &visible, &block_scores, &top_ids, &top_scores],
            &parameters,
            1,
            1,
        )
        .unwrap();
    let actual = graph.finish(&[visible]).unwrap().remove(0);
    assert_eq!(actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), [1, 1, 0, 0, 1]);
}

#[stark_test]
fn qwen_graph_qsa_prepare_writes_first_token_at_zero_based_cache_position() {
    let index = [1.0, 0.0, 0.25, -0.5];
    let query_gate = [3.0, 4.0, -0.2, 0.7];
    let key = [-2.0, 1.0];
    let value = [0.3, -0.8];
    let query_norm = [2.0, 3.0];
    let key_norm = [4.0, 5.0];
    let mut graph = DecodeGraph::new().unwrap();
    let index = DecodeGraph::upload(&index).unwrap();
    let query_gate = DecodeGraph::upload(&query_gate).unwrap();
    let key = DecodeGraph::upload(&key).unwrap();
    let value = DecodeGraph::upload(&value).unwrap();
    let query_norm = DecodeGraph::upload(&query_norm).unwrap();
    let key_norm = DecodeGraph::upload(&key_norm).unwrap();
    let index_cache = DecodeGraph::zeros(2).unwrap();
    let query_cache = DecodeGraph::zeros(2).unwrap();
    let key_cache = DecodeGraph::zeros(2).unwrap();
    let value_cache = DecodeGraph::zeros(2).unwrap();
    let gates = DecodeGraph::zeros(2).unwrap();
    let params = [1, 1, 2, 1, 2, 1, 2, 0, 1e-6f32.to_bits(), 10_000.0f32.to_bits()];
    graph
        .kernel(
            "qwen_graph_qsa_prepare",
            &[
                &index,
                &query_gate,
                &key,
                &value,
                &query_norm,
                &key_norm,
                &index_cache,
                &query_cache,
                &key_cache,
                &value_cache,
                &gates,
            ],
            &params,
            3,
            1,
        )
        .unwrap();
    let actual = graph.finish(&[index_cache, query_cache, key_cache, value_cache, gates]).unwrap();
    let rms_q = (25.0f32 / 2.0 + 1e-6).sqrt();
    let rms_k = (5.0f32 / 2.0 + 1e-6).sqrt();
    assert_eq!(actual[0], [0.25, -0.5]);
    assert!((actual[1][0] - (3.0 / rms_q) * 2.0).abs() < 1e-6);
    assert!((actual[1][1] - (4.0 / rms_q) * 3.0).abs() < 1e-6);
    assert!((actual[2][0] - (-2.0 / rms_k) * 4.0).abs() < 1e-6);
    assert!((actual[2][1] - (1.0 / rms_k) * 5.0).abs() < 1e-6);
    assert_eq!(actual[3], [0.3, -0.8]);
    assert_eq!(actual[4], [-0.2, 0.7]);
}

fn qwen_graph_ple_step(
    history: &DeviceVector,
    key: &[f32],
    value: &[f32],
    query: &[f32],
) -> Vec<f32> {
    let (streams, dim, kernel, dilation) = (2u32, 2u32, 3u32, 2u32);
    let width = streams as usize * dim as usize;
    let mut graph = DecodeGraph::new().unwrap();
    let key = DecodeGraph::upload(key).unwrap();
    let value = DecodeGraph::upload(value).unwrap();
    let query = DecodeGraph::upload(query).unwrap();
    let key_norm = DecodeGraph::upload(&[1.0, 0.8, 1.2, 0.9]).unwrap();
    let query_norm = DecodeGraph::upload(&[0.9, 1.1, 0.7, 1.3]).unwrap();
    let conv_norm = DecodeGraph::upload(&[1.0, 0.8, 1.2, 0.9]).unwrap();
    // Channel-major [channel,kernel] weights, matching the source PLE layout.
    let conv_weights =
        DecodeGraph::upload(&[0.25, -0.5, 1.0, -0.75, 0.5, 0.8, 0.4, -0.2, 0.6, 0.7, 0.3, -0.9]).unwrap();
    let output = graph.scratch(width).unwrap();
    let scratch = DecodeGraph::zeros(width).unwrap();
    graph
        .kernel(
            "qwen_graph_ple",
            &[&key, &value, &query, &key_norm, &query_norm, &conv_norm, &conv_weights, history, &output, &scratch],
            &[streams, dim, kernel, dilation, 1e-6f32.to_bits()],
            streams as usize,
            1,
        )
        .unwrap();
    graph
        .kernel(
            "qwen_graph_ple_convolve",
            &[&conv_weights, history, &output, &scratch],
            &[streams, dim, kernel, dilation],
            width,
            256,
        )
        .unwrap();
    graph.finish(&[output]).unwrap().remove(0)
}

fn qwen_graph_ple_reference_step(
    history: &mut Vec<f32>,
    key: &[f32],
    value: &[f32],
    query: &[f32],
) -> Vec<f32> {
    let streams = 2;
    let dim = 2;
    let kernel = 3;
    let dilation = 2;
    let eps = 1e-6;
    let width = streams * dim;
    let key_weights = [1.0, 0.8, 1.2, 0.9];
    let query_weights = [0.9, 1.1, 0.7, 1.3];
    let conv_weights = [0.25, -0.5, 1.0, -0.75, 0.5, 0.8, 0.4, -0.2, 0.6, 0.7, 0.3, -0.9];
    let conv_norm = key_weights;
    let mut gated = vec![0.0; width];
    for stream in 0..streams {
        let start = stream * dim;
        let normalized_key =
            crate::engine::hybrid::math::rms_norm(&key[start..start + dim], &key_weights[start..start + dim], eps)
                .unwrap();
        let normalized_query =
            crate::engine::hybrid::math::rms_norm(&query[start..start + dim], &query_weights[start..start + dim], eps)
                .unwrap();
        let dot = normalized_key.iter().zip(&normalized_query).map(|(a, b)| a * b).sum::<f32>();
        let score = dot / (dim as f32).sqrt();
        let sign = if score > 0.0 {
            1.0
        } else if score < 0.0 {
            -1.0
        } else {
            0.0
        };
        let gate = crate::engine::hybrid::math::sigmoid(sign * score.abs().max(1e-6).sqrt());
        for channel in 0..dim {
            gated[start + channel] = gate * value[channel];
        }
    }
    let mut normalized = Vec::with_capacity(width);
    for stream in 0..streams {
        let start = stream * dim;
        normalized.extend(
            crate::engine::hybrid::math::rms_norm(&gated[start..start + dim], &conv_norm[start..start + dim], eps)
                .unwrap(),
        );
    }
    if history.is_empty() {
        history.resize((kernel - 1) * dilation * width, 0.0);
    }
    let mut output = gated;
    for channel in 0..width {
        let mut sum = normalized[channel] * conv_weights[channel * kernel + kernel - 1];
        for tap in 0..kernel - 1 {
            sum += history[tap * dilation * width + channel] * conv_weights[channel * kernel + tap];
        }
        output[channel] += crate::engine::hybrid::math::silu(sum);
    }
    history.copy_within(width.., 0);
    let last = history.len() - width;
    history[last..].copy_from_slice(&normalized);
    output
}

#[stark_test]
fn qwen_graph_ple_multistep_history_and_fresh_reset_replay_match() {
    let width = 4usize;
    let history_len = width * (3 - 1) * 2;
    let first_key = [0.2, 0.7, -0.4, 0.3];
    let first_value = [0.4, -0.3];
    let first_query = [-0.3, 1.1, 0.6, 0.25];
    let second_key = [0.5, -0.1, 0.8, -0.7];
    let second_value = [-0.2, 0.8];
    let second_query = [0.6, 0.25, -0.2, 0.9];

    let inputs = [
        (&first_key, &first_value, &first_query),
        (&second_key, &second_value, &second_query),
        (&first_key, &first_value, &first_query),
        (&second_key, &second_value, &second_query),
        (&first_key, &first_value, &first_query),
    ];
    let history = DecodeGraph::zeros(history_len).unwrap();
    let mut expected_history = Vec::new();
    let mut outputs = Vec::new();
    for (key, value, query) in inputs {
        let actual = qwen_graph_ple_step(&history, key, value, query);
        let expected = qwen_graph_ple_reference_step(&mut expected_history, key, value, query);
        assert!(actual.iter().zip(expected).all(|(&actual, expected)| (actual - expected).abs() < 2e-5));
        outputs.push(actual);
    }
    // Identical inputs at positions 0, 2 and 4 must consume different causal
    // history: position 2 reaches the dilated tap and position 4 the oldest tap.
    assert_ne!(outputs[0], outputs[2]);
    assert_ne!(outputs[2], outputs[4]);
    let fresh_history = DecodeGraph::zeros(history_len).unwrap();
    for ((key, value, query), expected) in inputs.into_iter().zip(outputs) {
        assert_eq!(qwen_graph_ple_step(&fresh_history, key, value, query), expected);
    }
}

#[stark_test]
fn qwen_graph_parallel_attention_preserves_grouped_heads_mask_and_prefix_stride() {
    use crate::engine::hybrid::math::{sigmoid, softmax};

    for (heads, kv_heads, dim, count, capacity) in [(4, 2, 3, 5, 8), (2, 1, 33, 7, 16)] {
        let values = |len: usize, seed: usize| {
            (0..len).map(|index| (((index * 13 + seed * 7) % 47) as f32 - 23.0) / 29.0).collect::<Vec<_>>()
        };
        let query = values(heads * dim, 1);
        let keys = values(capacity * kv_heads * dim, 2);
        let cache_values = values(capacity * kv_heads * dim, 3);
        let gates = values(heads * dim, 4);
        let visible = (0..count).map(|token| token != 1 && token != 3).collect::<Vec<_>>();
        let selected =
            visible.iter().enumerate().filter_map(|(token, &keep)| keep.then_some(token)).collect::<Vec<_>>();
        let mut graph = DecodeGraph::new().unwrap();
        let input_query = DecodeGraph::upload(&query).unwrap();
        let input_keys = DecodeGraph::upload(&keys).unwrap();
        let input_values = DecodeGraph::upload(&cache_values).unwrap();
        let input_gates = DecodeGraph::upload(&gates).unwrap();
        let input_visible =
            DecodeGraph::upload(&visible.iter().map(|&keep| f32::from_bits(u32::from(keep))).collect::<Vec<_>>())
                .unwrap();
        let scores = graph.scratch(heads * capacity).unwrap();
        let probabilities = graph.scratch(heads * capacity).unwrap();
        let output = graph.scratch(heads * dim).unwrap();
        let parameters = [count as u32, heads as u32, dim as u32, kv_heads as u32, capacity as u32];
        graph
            .kernel(
                "qwen_graph_qsa_scores",
                &[&input_query, &input_keys, &input_visible, &scores],
                &parameters,
                heads * count,
                64,
            )
            .unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_softmax",
                &[&scores, &input_visible, &probabilities],
                &[count as u32, heads as u32, capacity as u32],
                heads,
                1,
            )
            .unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_output",
                &[&probabilities, &input_values, &input_gates, &input_visible, &output],
                &parameters,
                heads * dim,
                256,
            )
            .unwrap();
        let actual = graph.finish(&[output]).unwrap().remove(0);
        let mut expected = vec![0.0; heads * dim];
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let scores = selected
                .iter()
                .map(|&token| {
                    let mut score = 0.0;
                    for channel in 0..dim {
                        score += query[head * dim + channel] * keys[token * kv_heads * dim + kv_head * dim + channel];
                    }
                    score / (dim as f32).sqrt()
                })
                .collect::<Vec<_>>();
            let probabilities = softmax(&scores).unwrap();
            for (&token, probability) in selected.iter().zip(probabilities) {
                for channel in 0..dim {
                    expected[head * dim + channel] +=
                        probability * cache_values[token * kv_heads * dim + kv_head * dim + channel];
                }
            }
            for channel in 0..dim {
                expected[head * dim + channel] *= sigmoid(gates[head * dim + channel]);
            }
        }
        assert!(actual.iter().zip(expected).all(|(&a, b)| (a - b).abs() < 2e-6));
    }
}
