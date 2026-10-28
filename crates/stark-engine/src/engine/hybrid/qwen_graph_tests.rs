// SPDX-License-Identifier: Apache-2.0
//! Focused Metal control-kernel checks; full Qwen graph parity lives in qwen.rs.

use metal::MTLBuffer;
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
fn qwen_graph_parallel_router_matches_scalar_ids_and_probability_bits() {
    for experts in [1usize, 7, 255, 256, 257, 512, 513] {
        let scores = (0..experts)
            .map(|i| match i % 13 {
                0 => -0.0,
                1 => 0.0,
                2..=4 => 1.0,
                _ => ((i * 31 % 173) as f32 - 86.0) * 0.027,
            })
            .collect::<Vec<_>>();
        for k in [1, 10.min(experts), experts.min(32)] {
            let mut expected = None;
            for (name, threads) in [("qwen_graph_router", 1), ("qwen_graph_router_parallel", 256)] {
                let mut graph = DecodeGraph::new().unwrap();
                let input = graph.upload_into(&scores).unwrap();
                let ids = graph.scratch(k).unwrap();
                let probabilities = graph.scratch(k).unwrap();
                graph
                    .kernel(name, &[&input, &ids, &probabilities], &[experts as u32, k as u32], threads, threads)
                    .unwrap();
                let actual = graph
                    .finish(&[ids, probabilities])
                    .unwrap()
                    .into_iter()
                    .map(|row| row.into_iter().map(f32::to_bits).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected, "parallel router changed selection or softmax arithmetic");
                } else {
                    expected = Some(actual);
                }
            }
        }
    }
}

#[stark_test]
fn qwen_graph_parallel_router_invalid_input_leaves_invalid_gather_indices() {
    for poison in [f32::NAN, f32::from_bits(0xffc0_0001)] {
        for poison_index in [0usize, 255, 256, 511] {
            let mut graph = DecodeGraph::new().unwrap();
            let input = DecodeGraph::zeros(512).unwrap();
            let ids = DecodeGraph::zeros(10).unwrap();
            let probabilities = DecodeGraph::zeros(10).unwrap();
            // Test input bypasses the finite host-upload contract deliberately.
            unsafe { input.buffer.contents().as_ptr().cast::<f32>().add(poison_index).write(poison) };
            graph.kernel("qwen_graph_router_parallel", &[&input, &ids, &probabilities], &[512, 10], 256, 256).unwrap();
            assert!(graph.finish(&[]).is_err());
            let actual = unsafe { std::slice::from_raw_parts(ids.buffer.contents().as_ptr().cast::<u32>(), 10) };
            assert!(actual.iter().all(|&id| id == u32::MAX));
        }
    }
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

fn qwen_graph_qsa_selection_case(
    parameters: [u32; 9],
    tied: bool,
    scalar: bool,
) -> (Vec<Vec<f32>>, f64) {
    let [count, heads, dim, _, _, _, _, ratio, budget] = parameters;
    let data = |len: usize, salt: usize| {
        (0..len)
            .map(|i| {
                if tied {
                    0.0
                } else {
                    (((i * 17 + salt) % 41) as f32 - 20.0) * 0.031
                }
            })
            .collect::<Vec<_>>()
    };
    let index = DecodeGraph::upload(&data(((heads + 1) * dim) as usize, 3)).unwrap();
    let keys = DecodeGraph::upload(&data((count * dim) as usize, 5)).unwrap();
    let q_norm = DecodeGraph::upload(&vec![1.1; dim as usize]).unwrap();
    let k_norm = DecodeGraph::upload(&vec![0.9; dim as usize]).unwrap();
    let visible = DecodeGraph::zeros(count as usize).unwrap();
    let blocks = (count / ratio) as usize;
    let scores = DecodeGraph::zeros(blocks.max(1)).unwrap();
    let ids = DecodeGraph::zeros((budget / ratio) as usize).unwrap();
    let top_scores = DecodeGraph::zeros((budget / ratio) as usize).unwrap();
    let queries = DecodeGraph::zeros((heads * dim) as usize).unwrap();
    let pooled_keys = DecodeGraph::zeros(blocks.max(1) * dim as usize).unwrap();
    if !scalar && count > budget {
        // Cache construction is prefix work. Ordinary decode below updates
        // only the current query, just as a partial block does at runtime.
        let mut preparation = DecodeGraph::new().unwrap();
        let mut preparation_parameters = parameters.to_vec();
        preparation_parameters.extend([0, blocks as u32]);
        preparation
            .kernel(
                "qwen_graph_qsa_index_prepare",
                &[&index, &keys, &q_norm, &k_norm, &queries, &pooled_keys],
                &preparation_parameters,
                heads as usize + blocks,
                1,
            )
            .unwrap();
        preparation.finish(&[]).unwrap();
    }
    let mut graph = DecodeGraph::new().unwrap();
    if scalar {
        graph
            .kernel(
                "qwen_graph_qsa_visible",
                &[&index, &keys, &q_norm, &k_norm, &visible, &scores, &ids, &top_scores],
                &parameters,
                1,
                1,
            )
            .unwrap();
    } else {
        if count > budget {
            let mut preparation_parameters = parameters.to_vec();
            preparation_parameters.extend([0, 0]);
            graph
                .kernel(
                    "qwen_graph_qsa_index_prepare",
                    &[&index, &keys, &q_norm, &k_norm, &queries, &pooled_keys],
                    &preparation_parameters,
                    heads as usize,
                    1,
                )
                .unwrap();
            graph
                .kernel(
                    "qwen_graph_qsa_index_cached_scores",
                    &[&queries, &pooled_keys, &scores],
                    &[blocks as u32, heads, dim],
                    blocks,
                    64,
                )
                .unwrap();
            graph
                .kernel(
                    "qwen_graph_qsa_index_threshold",
                    &[&scores, &ids, &top_scores],
                    &[blocks as u32, (budget / ratio).min(blocks as u32)],
                    256,
                    256,
                )
                .unwrap();
        }
        graph
            .kernel(
                "qwen_graph_qsa_index_mask",
                &[&scores, &ids, &top_scores, &visible],
                &[count, ratio, budget],
                count as usize,
                256,
            )
            .unwrap();
    }
    let started = std::time::Instant::now();
    let result = graph.finish(&[visible, scores]).unwrap();
    (result, started.elapsed().as_secs_f64())
}

#[stark_test]
fn qwen_graph_parallel_sparse_selection_matches_scalar_scores_masks_ties_and_tails() {
    for (count, heads, dim, rotary, ratio, budget) in [
        (3, 1, 8, 4, 4, 16),
        (16, 2, 8, 4, 4, 16),
        (17, 2, 8, 4, 4, 16),
        (63, 4, 8, 4, 4, 16),
        (67, 4, 8, 4, 4, 16),
        (9, 2, 8, 4, 1, 4),
        (129, 4, 128, 32, 4, 64),
    ] {
        let parameters =
            [count, heads, dim, rotary, count - 1, 1e-6f32.to_bits(), 10_000_000.0f32.to_bits(), ratio, budget];
        for tied in [false, true] {
            let expected = qwen_graph_qsa_selection_case(parameters, tied, true).0;
            let actual = qwen_graph_qsa_selection_case(parameters, tied, false).0;
            assert!(actual[0] == expected[0], "parallel sparse selection changed the visible mask");
            assert!(actual[1] == expected[1], "parallel sparse selection changed block scores");
        }
    }
}

#[stark_test]
fn qwen_graph_pooled_index_keys_append_only_on_complete_blocks_and_survive_growth() {
    let dim = 8usize;
    let heads = 2usize;
    let index = DecodeGraph::upload(&(0..24).map(|i| (i as f32 - 11.0) * 0.07).collect::<Vec<_>>()).unwrap();
    let keys = DecodeGraph::upload(&(0..72).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect::<Vec<_>>()).unwrap();
    let norm = DecodeGraph::upload(&vec![1.1; dim]).unwrap();
    let query = DecodeGraph::zeros(heads * dim).unwrap();
    let prepare = |graph: &mut DecodeGraph,
                   pooled: &DeviceVector,
                   query: &DeviceVector,
                   count: usize,
                   first: usize,
                   blocks: usize| {
        graph
            .kernel(
                "qwen_graph_qsa_index_prepare",
                &[&index, &keys, &norm, &norm, query, pooled],
                &[
                    count as u32,
                    heads as u32,
                    dim as u32,
                    4,
                    (count - 1) as u32,
                    1e-6f32.to_bits(),
                    10_000_000.0f32.to_bits(),
                    4,
                    4,
                    first as u32,
                    blocks as u32,
                ],
                heads + blocks,
                1,
            )
            .unwrap();
    };
    let initial = DecodeGraph::zeros(dim).unwrap();
    let mut first = DecodeGraph::new().unwrap();
    prepare(&mut first, &initial, &query, 4, 0, 1);
    let first_key = first.finish(&[initial.clone()]).unwrap().remove(0);
    let mut partial = DecodeGraph::new().unwrap();
    prepare(&mut partial, &initial, &query, 5, 0, 0);
    assert!(
        partial.finish(&[initial.clone()]).unwrap().remove(0) == first_key,
        "partial sparse block changed a completed pooled key"
    );
    let grown = DecodeGraph::zeros(dim * 4).unwrap();
    let mut growth = DecodeGraph::new().unwrap();
    growth.copy_into(&initial, &grown.slice(0, dim).unwrap()).unwrap();
    prepare(&mut growth, &grown, &query, 8, 1, 1);
    prepare(&mut growth, &grown, &query, 9, 0, 0);
    let actual = growth.finish(&[grown.slice(0, dim * 2).unwrap(), query.clone()]).unwrap();
    let batch = DecodeGraph::zeros(dim * 2).unwrap();
    let batch_query = DecodeGraph::zeros(heads * dim).unwrap();
    let mut expected_graph = DecodeGraph::new().unwrap();
    prepare(&mut expected_graph, &batch, &batch_query, 9, 0, 2);
    let expected = expected_graph.finish(&[batch, batch_query]).unwrap();
    assert!(actual == expected, "incremental pooled index state differs after cache growth");
    assert!(actual[0][..dim] == first_key, "appending the next pooled key changed its predecessor");
    let reset = DecodeGraph::zeros(dim).unwrap();
    let reset_query = DecodeGraph::zeros(heads * dim).unwrap();
    let mut fresh = DecodeGraph::new().unwrap();
    prepare(&mut fresh, &reset, &reset_query, 4, 0, 1);
    assert!(fresh.finish(&[reset]).unwrap().remove(0) == first_key, "fresh pooled state differs after reset");
}

#[stark_test]
fn qwen_graph_radix_selection_preserves_total_order_and_ascending_block_ties() {
    let scores = [0.0, -0.0, 1.0, 1.0, -3.0, 3.0, f32::from_bits(1.0f32.to_bits() + 1), 0.0];
    for keep in 1..scores.len() {
        let mut graph = DecodeGraph::new().unwrap();
        let input = DecodeGraph::upload(&scores).unwrap();
        let ids = DecodeGraph::zeros(1).unwrap();
        let threshold = DecodeGraph::zeros(1).unwrap();
        let visible = DecodeGraph::zeros(scores.len()).unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_index_threshold",
                &[&input, &ids, &threshold],
                &[scores.len() as u32, keep as u32],
                256,
                256,
            )
            .unwrap();
        graph
            .kernel(
                "qwen_graph_qsa_index_mask",
                &[&input, &ids, &threshold, &visible],
                &[scores.len() as u32, 1, keep as u32],
                scores.len(),
                256,
            )
            .unwrap();
        let actual = graph.finish(&[visible]).unwrap().remove(0);
        let mut order = (0..scores.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        assert!(
            actual.iter().enumerate().all(|(i, value)| value.to_bits() == u32::from(order[..keep].contains(&i))),
            "radix top-k differs from total_cmp"
        );
    }
}

#[stark_test]
#[ignore]
fn qwen_graph_parallel_sparse_selection_microbenchmark() {
    let parameters = [4099, 4, 128, 32, 4098, 1e-6f32.to_bits(), 10_000_000.0f32.to_bits(), 4, 2048];
    for _ in 0..3 {
        let expected = qwen_graph_qsa_selection_case(parameters, false, true);
        let actual = qwen_graph_qsa_selection_case(parameters, false, false);
        assert!(actual.0 == expected.0, "sparse selection benchmark changed values");
        eprintln!("qwen_qsa_selection_4k_wall_ms scalar={:.3} parallel={:.3}", expected.1 * 1e3, actual.1 * 1e3);
    }
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
fn qwen_graph_parallel_ple_matches_scalar_across_wide_tails_and_history() {
    for (streams, dim) in [(2usize, 2usize), (4, 33), (4, 2560)] {
        let width = streams * dim;
        let parameters = [streams as u32, dim as u32, 3, 2, 1e-6f32.to_bits()];
        let history = [DecodeGraph::zeros(width * 4).unwrap(), DecodeGraph::zeros(width * 4).unwrap()];
        let values = |len: usize, seed: usize| {
            (0..len).map(|i| (((i * 13 + seed * 7) % 47) as f32 - 23.0) / 29.0).collect::<Vec<_>>()
        };
        let norm = DecodeGraph::upload(&values(width, 4)).unwrap();
        let conv = DecodeGraph::upload(&values(width * 3, 5)).unwrap();
        for step in 0..6 {
            let key = DecodeGraph::upload(&values(width, step)).unwrap();
            let value = DecodeGraph::upload(&values(dim, step + 1)).unwrap();
            let query = DecodeGraph::upload(&values(width, step + 2)).unwrap();
            let mut results = Vec::new();
            for (index, name) in ["qwen_graph_ple", "qwen_graph_ple_parallel"].into_iter().enumerate() {
                let mut graph = DecodeGraph::new().unwrap();
                let output = graph.scratch(width).unwrap();
                let scratch = graph.scratch(width).unwrap();
                graph
                    .kernel(
                        name,
                        &[&key, &value, &query, &norm, &norm, &norm, &conv, &history[index], &output, &scratch],
                        &parameters,
                        streams
                            * if index == 0 {
                                1
                            } else {
                                256
                            },
                        if index == 0 {
                            1
                        } else {
                            256
                        },
                    )
                    .unwrap();
                graph
                    .kernel(
                        "qwen_graph_ple_convolve",
                        &[&conv, &history[index], &output, &scratch],
                        &parameters[..4],
                        width,
                        256,
                    )
                    .unwrap();
                results.push(graph.finish(&[output, scratch, history[index].clone()]).unwrap());
            }
            for (scalar, parallel) in results[0].iter().zip(&results[1]) {
                assert!(scalar.iter().zip(parallel).all(|(&a, &b)| (a - b).abs() < 2e-5));
            }
        }
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
