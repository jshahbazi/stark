// SPDX-License-Identifier: Apache-2.0
//! Snapshot chronology gates independent of model weights.

use stark_engine_macros::stark_test;

use super::*;

fn same(
    actual: &[f32],
    expected: &[f32],
) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[stark_test]
fn glm_kda_prefix_snapshot_restores_convolution_and_recurrent_bits() {
    let heads = 2;
    let dim = 7;
    let kernel = 4;
    let inner = heads * dim;
    let values = |len: usize, seed: usize| {
        (0..len).map(|i| ((i * 13 + seed * 7) % 43) as f32 * 0.013 - 0.27).collect::<Vec<_>>()
    };
    let convolutions = [values(inner * kernel, 1), values(inner * kernel, 2), values(inner * kernel, 3)];
    let create = || {
        Kda::new(
            Geometry {
                heads,
                dim,
                kernel,
                rms_eps: 1e-5,
                lower_bound: -5.0,
            },
            [&convolutions[0], &convolutions[1], &convolutions[2]],
            &values(inner, 4),
            &vec![1.0; dim],
            &vec![-1.25; heads],
        )
        .unwrap()
    };
    let step = |state: &mut Kda, seed| {
        let inputs = [
            values(inner, seed),
            values(inner, seed + 1),
            values(inner, seed + 2),
            values(inner, seed + 3),
            values(inner, seed + 4),
        ];
        state.step([&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]], &values(heads, seed + 5)).unwrap()
    };
    for transposed in [false, true] {
        let mut state = create();
        state.transposed_state = transposed;
        for seed in 0..7 {
            step(&mut state, seed);
        }
        let snapshot = state.snapshot().unwrap();
        let expected = (7..12).map(|seed| step(&mut state, seed)).collect::<Vec<_>>();
        let final_state = state.snapshot().unwrap();
        state.reset();
        state.restore(&snapshot).unwrap();
        for (seed, expected) in (7..12).zip(expected) {
            assert!(same(&step(&mut state, seed), &expected), "GLM KDA resume changed output bits");
        }
        assert!(same(&state.snapshot().unwrap().values, &final_state.values), "GLM KDA resume changed final state");
        state.transposed_state = !transposed;
        assert!(state.restore(&snapshot).is_err(), "GLM KDA accepted a different recurrent layout");
    }
}

#[stark_test]
fn glm_attention_prefix_snapshot_restores_incomplete_pool_after_growth() {
    for pool in [3, 4] {
        for prefix in [61, 65] {
            let mut state = Attention::new(
                AttentionGeometry {
                    rank: 3,
                    heads: 1,
                    query_dim: 2,
                    index_dim: 2,
                    index_heads: 1,
                    pool,
                    top_k: pool * 2,
                    context: 130,
                },
                &vec![0.0; pool * 2],
            )
            .unwrap();
            let step = |state: &mut Attention, position: usize| {
                let mut graph = DecodeGraph::new().unwrap();
                state
                    .append(
                        &mut graph,
                        &DecodeGraph::upload(&[position as f32 * 0.01, 0.3, -0.2]).unwrap(),
                        &DecodeGraph::upload(&[position as f32 * 0.013, 0.2]).unwrap(),
                        &DecodeGraph::upload(&[0.1, -0.2]).unwrap(),
                    )
                    .unwrap();
                let output = state
                    .attend(
                        &mut graph,
                        &DecodeGraph::upload(&[0.7, -0.1, 0.3]).unwrap(),
                        &DecodeGraph::upload(&[0.5, 0.3]).unwrap(),
                        &DecodeGraph::upload(&[0.7]).unwrap(),
                    )
                    .unwrap();
                graph.finish(&[output]).unwrap().remove(0)
            };
            for position in 0..prefix {
                step(&mut state, position);
            }
            let snapshot = state.snapshot().unwrap();
            assert!(snapshot.tail_keys.len() > 0, "test must retain an incomplete pool");
            let expected = (prefix..prefix + 10).map(|position| step(&mut state, position)).collect::<Vec<_>>();
            state.reset();
            state.restore(&snapshot).unwrap();
            for (position, expected) in (prefix..prefix + 10).zip(expected) {
                assert!(same(&step(&mut state, position), &expected), "GLM MLA/indexer resume changed output bits");
            }
            assert!(state.restore(&snapshot).is_err(), "GLM MLA restore accepted an active sequence");
        }
    }
}
