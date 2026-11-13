// SPDX-License-Identifier: Apache-2.0
//! Synthetic operator trajectories; never read a model or publish vectors.
use stark_engine_macros::stark_test;

use super::*;

fn data(
    length: usize,
    seed: usize,
) -> Vec<f32> {
    (0..length).map(|i| (((i * 13 + seed * 7) % 71) as f32 - 35.0) * 0.009).collect()
}

fn create(
    transposed: bool,
    kernel: usize,
) -> Kda {
    let (heads, dim) = (2, 128);
    let inner = heads * dim;
    let convolution = [data(inner * kernel, 1), data(inner * kernel, 2), data(inner * kernel, 3)];
    let mut state = Kda::new(
        Geometry {
            heads,
            dim,
            kernel,
            rms_eps: 1e-5,
            lower_bound: -5.0,
        },
        [&convolution[0], &convolution[1], &convolution[2]],
        &data(inner, 4),
        &data(dim, 5),
        &[-1.25, -0.75],
    )
    .unwrap();
    state.transposed_state = transposed;
    // Nonzero convolution history and recurrent state exercise prefix-resume
    // semantics, including entries that zero-state tests cannot observe.
    let values = data(state.state_values, 6);
    unsafe { state.state.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(values.as_ptr(), values.len()) };
    state
}

fn projections(
    tokens: usize,
    inner: usize,
    heads: usize,
    seed: usize,
) -> [Vec<f32>; 6] {
    std::array::from_fn(|part| {
        data(
            tokens
                * if part == 5 {
                    heads
                } else {
                    inner
                },
            seed + part * 11,
        )
    })
}

fn encode(
    state: &mut Kda,
    inputs: &[Vec<f32>; 6],
    tokens: usize,
) -> (DecodeGraph, DeviceVector) {
    encode_with_policy(state, inputs, tokens, None)
}

fn encode_with_policy(
    state: &mut Kda,
    inputs: &[Vec<f32>; 6],
    tokens: usize,
    parallel_norm: Option<bool>,
) -> (DecodeGraph, DeviceVector) {
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let vectors = inputs
        .iter()
        .map(|values| {
            let finite = values
                .iter()
                .map(|&value| {
                    if value.is_finite() {
                        value
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let vector = graph.upload_into(&finite).unwrap();
            // Exercise GPU rejection rather than the host upload guard. This fresh
            // allocation belongs to an unsubmitted graph and has no active writer.
            unsafe {
                vector
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(vector.offset)
                    .cast::<f32>()
                    .copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
            vector
        })
        .collect::<Vec<_>>();
    let output = graph.scratch(tokens * state.inner).unwrap();
    let admitted = if let Some(parallel_norm) = parallel_norm {
        state.encode_prefill_register_policy(
            &mut graph,
            &vectors[0],
            &vectors[1],
            &vectors[2],
            &vectors[3],
            &vectors[4],
            &vectors[5],
            &output,
            tokens,
            parallel_norm,
        )
    } else {
        state.encode_prefill_register_inner(
            &mut graph,
            &vectors[0],
            &vectors[1],
            &vectors[2],
            &vectors[3],
            &vectors[4],
            &vectors[5],
            &output,
            tokens,
        )
    };
    assert!(admitted.unwrap(), "test hardware declined KDA register kernel");
    (graph, output)
}

fn run(
    state: &mut Kda,
    inputs: &[Vec<f32>; 6],
    tokens: usize,
) -> Result<Vec<f32>> {
    let (graph, output) = encode(state, inputs, tokens);
    let result = graph.finish(&[output])?.remove(0);
    state.check_completed()?;
    Ok(result)
}

fn run_with_policy(
    state: &mut Kda,
    inputs: &[Vec<f32>; 6],
    tokens: usize,
    parallel_norm: bool,
) -> Vec<f32> {
    let (graph, output) = encode_with_policy(state, inputs, tokens, Some(parallel_norm));
    let result = graph.finish(&[output]).unwrap().remove(0);
    state.check_completed().unwrap();
    assert!(result.iter().all(|value| value.is_finite()), "nonfinite KDA trajectory");
    result
}

fn equal_bits(
    actual: &[f32],
    expected: &[f32],
) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[stark_test]
fn kda_register_prefill_plan_covers_each_row_once_with_bounded_head_slots() {
    assert_eq!(
        plan_for(64, 40, 1024),
        Some(Plan {
            groups: 80,
            rows: 104
        })
    );
    for heads in [1, 2, 3, 48, 64, 256] {
        for cores in [1, 8, 40, 80, 256] {
            for cap in [32, 64, 256, 768, 1024] {
                let plan = plan_for(heads, cores, cap).unwrap();
                assert!(plan.rows * 8 <= cap && plan.rows <= 128);
                let total = heads * 128;
                let mut previous = 0;
                for group in 0..plan.groups {
                    let first = 4 * ((group * (total / 4)) / plan.groups);
                    let end = 4 * (((group + 1) * (total / 4)) / plan.groups);
                    assert_eq!(first, previous);
                    assert!(end > first && end - first <= plan.rows);
                    assert!((end - 1) / 128 - first / 128 < 2);
                    previous = end;
                }
                assert_eq!(previous, total);
            }
        }
    }
    assert_eq!(plan_for(64, 0, 1024), None);
    assert_eq!(plan_for(0, 40, 1024), None);
    assert_eq!(plan_for(64, 40, 31), None);
}

#[stark_test]
fn kda_register_prefill_matches_ordered_nonzero_trajectories_and_chunk_continuations() {
    for transposed in [false, true] {
        for kernel in [2, 4, 32] {
            let mut reference = create(transposed, kernel);
            let mut candidate = create(transposed, kernel);
            for (chunk, tokens) in [1, 3, 7, 8, 9, 31, 128].into_iter().enumerate() {
                let inputs = projections(tokens, reference.inner, reference.heads, chunk * 17);
                let mut expected = Vec::with_capacity(tokens * reference.inner);
                for token in 0..tokens {
                    let span = token * reference.inner..(token + 1) * reference.inner;
                    expected.extend(
                        reference
                            .step(
                                [
                                    &inputs[0][span.clone()],
                                    &inputs[1][span.clone()],
                                    &inputs[2][span.clone()],
                                    &inputs[3][span.clone()],
                                    &inputs[4][span],
                                ],
                                &inputs[5][token * reference.heads..(token + 1) * reference.heads],
                            )
                            .unwrap(),
                    );
                }
                let actual_output = run(&mut candidate, &inputs, tokens).unwrap();
                assert!(actual_output.iter().chain(&expected).all(|value| value.is_finite()));
                assert!(equal_bits(&actual_output, &expected), "KDA register output changed source bits");
                let actual = candidate.snapshot().unwrap();
                let expected = reference.snapshot().unwrap();
                let history = 3 * reference.inner * (kernel - 1);
                assert!(
                    equal_bits(&actual.values[..history], &expected.values[..history]),
                    "KDA convolution chronology changed"
                );
                assert!(
                    equal_bits(&actual.values[history..], &expected.values[history..]),
                    "KDA register recurrent state changed source bits"
                );
            }
        }
    }
}

fn reference_pipeline(
    transposed: bool,
    parallel_norm: bool,
    fused_state: bool,
) -> Pipeline {
    let source = format!(
        "#define STARK_GLM_KDA_TRANSPOSED {}\n#define STARK_GLM_KDA_FUSED_STATE {}\n#define STARK_GLM_KDA_PARALLEL_NORM {}\n{}",
        usize::from(transposed),
        usize::from(fused_state),
        usize::from(parallel_norm),
        include_str!("glm_metal.metal"),
    );
    let e = executor().unwrap();
    let options = metal::MTLCompileOptions::new();
    options.set_math_mode(metal::MTLMathMode::Safe);
    let library = e
        .device
        .new_library_with_source(&source, Some(&options))
        .unwrap_or_else(|_| panic!("KDA exact reference compilation failed"));
    let function = library.new_function_with_name("glm_kda_decode").unwrap();
    e.device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|_| panic!("KDA exact reference pipeline failed"))
}

fn reference_step(
    state: &Kda,
    pipeline: &Pipeline,
    inputs: &[Vec<f32>; 6],
    token: usize,
) -> Vec<f32> {
    // All preceding reference commands have completed before reusing input.
    for (part, values) in inputs.iter().enumerate() {
        let width = if part == 5 {
            state.heads
        } else {
            state.inner
        };
        unsafe {
            state
                .input
                .contents()
                .as_ptr()
                .cast::<f32>()
                .add(part * state.inner)
                .copy_from_nonoverlapping(values.as_ptr().add(token * width), width);
        }
    }
    let e = executor().unwrap();
    let command = e.queue.command_buffer().unwrap();
    let encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in [&state.input, &state.weights, &state.state, &state.output].into_iter().enumerate() {
        encoder.set_buffer(Some(buffer), 0, index);
    }
    encoder.set_bytes(NonNull::from(&state.parameters).cast(), std::mem::size_of_val(&state.parameters), 4);
    encoder.set_buffer(Some(&state.failure), 0, 5);
    encoder.dispatch_threadgroups(
        MTLSize {
            width: state.heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: state.dim(),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    state.check_completed().unwrap();
    let result =
        unsafe { std::slice::from_raw_parts(state.output.contents().as_ptr().cast::<f32>(), state.inner) }.to_vec();
    assert!(result.iter().all(|value| value.is_finite()), "nonfinite KDA reference trajectory");
    result
}

#[stark_test]
fn kda_register_prefill_matches_source_bits_for_ordered_parallel_and_extreme_policies() {
    for (transposed, parallel_norm, fused_state) in
        [(false, false, false), (true, false, false), (true, false, true), (true, true, true)]
    {
        let pipeline = reference_pipeline(transposed, parallel_norm, fused_state);
        // q and k independently exercise the helper's joint ordered fallback;
        // v exercises the output RMS fallback. All inputs remain finite.
        for extreme in [None, Some((0, 1e17f32)), Some((1, 1e17)), Some((2, 1e19))] {
            let reference = create(transposed, 4);
            let mut candidate = create(transposed, 4);
            for chunk in 0..2 {
                let mut inputs = projections(17, reference.inner, reference.heads, 51 + chunk * 17);
                if let Some((part, factor)) = extreme {
                    inputs[part].iter_mut().for_each(|value| *value *= factor);
                }
                let expected =
                    (0..17).flat_map(|token| reference_step(&reference, &pipeline, &inputs, token)).collect::<Vec<_>>();
                let actual = run_with_policy(&mut candidate, &inputs, 17, parallel_norm);
                assert!(equal_bits(&actual, &expected), "KDA selected policy output changed source bits");
                assert!(
                    equal_bits(&candidate.snapshot().unwrap().values, &reference.snapshot().unwrap().values),
                    "KDA selected policy state changed source bits"
                );
            }
        }
    }
}

#[stark_test]
fn kda_register_prefill_selected_policies_restore_prefix_and_eight_plus_one_bits() {
    for (transposed, parallel_norm) in [(false, false), (true, false), (true, true)] {
        let mut state = create(transposed, 4);
        let inputs = projections(9, state.inner, state.heads, 73);
        let before = state.snapshot().unwrap();
        let expected = run_with_policy(&mut state, &inputs, 9, parallel_norm);
        let final_state = state.snapshot().unwrap();
        state.restore(&before).unwrap();
        let first = std::array::from_fn(|part| {
            let width = if part == 5 {
                state.heads
            } else {
                state.inner
            };
            inputs[part][..8 * width].to_vec()
        });
        let mut split = run_with_policy(&mut state, &first, 8, parallel_norm);
        let tail = std::array::from_fn(|part| {
            let width = if part == 5 {
                state.heads
            } else {
                state.inner
            };
            inputs[part][8 * width..].to_vec()
        });
        split.extend(run_with_policy(&mut state, &tail, 1, parallel_norm));
        assert!(equal_bits(&split, &expected), "selected KDA norm policy changed chunk bits");
        assert!(equal_bits(&state.snapshot().unwrap().values, &final_state.values));
        state.restore(&before).unwrap();
        assert!(equal_bits(&run_with_policy(&mut state, &inputs, 9, parallel_norm), &expected));
        assert!(equal_bits(&state.snapshot().unwrap().values, &final_state.values));
    }
}

#[stark_test]
fn kda_register_prefill_prefix_restore_and_graph_ownership_preserve_state() {
    for transposed in [false, true] {
        let mut state = create(transposed, 4);
        let inputs = projections(9, state.inner, state.heads, 99);
        let before = state.snapshot().unwrap();
        let (unsubmitted, _) = encode(&mut state, &inputs, 9);
        drop(unsubmitted);
        assert!(equal_bits(&state.snapshot().unwrap().values, &before.values), "abandoned encoding wrote KDA state");
        let expected = run(&mut state, &inputs, 9).unwrap();
        let final_state = state.snapshot().unwrap();
        state.restore(&before).unwrap();
        assert!(
            equal_bits(&run(&mut state, &inputs, 9).unwrap(), &expected),
            "register prefix restore changed output bits"
        );
        assert!(equal_bits(&state.snapshot().unwrap().values, &final_state.values));
        // Changing only the token staging boundary must preserve exact state,
        // as prefix reuse may resume with a shorter final chunk.
        state.restore(&before).unwrap();
        let first = std::array::from_fn(|part| {
            let width = if part == 5 {
                state.heads
            } else {
                state.inner
            };
            inputs[part][..8 * width].to_vec()
        });
        let mut split = run(&mut state, &first, 8).unwrap();
        let tail = std::array::from_fn(|part| {
            let width = if part == 5 {
                state.heads
            } else {
                state.inner
            };
            inputs[part][8 * width..].to_vec()
        });
        split.extend(run(&mut state, &tail, 1).unwrap());
        assert!(equal_bits(&split, &expected), "register chunk boundary changed output bits");
        assert!(equal_bits(&state.snapshot().unwrap().values, &final_state.values));
        state.restore(&before).unwrap();
        let (graph, output) = encode(&mut state, &inputs, 9);
        let pending = graph.submit(&[output]).unwrap();
        drop(pending); // the draining owner fences before reset/snapshot
        assert!(equal_bits(&state.snapshot().unwrap().values, &final_state.values));
        state.reset();
        let (graph, output) = encode(&mut state, &inputs, 9);
        let pending = graph.submit(&[output]).unwrap();
        drop(state); // the unretained transport's graph retains bound resources
        assert!(pending.wait().unwrap()[0].iter().all(|v| v.is_finite()));
    }
}

#[stark_test]
fn kda_register_prefill_rejects_masked_nonfinite_controls_and_keeps_status_sticky() {
    for transposed in [false, true] {
        let mut state = create(transposed, 4);
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for part in 0..6 {
                state.reset();
                let mut inputs = projections(9, state.inner, state.heads, 23);
                let last = inputs[part].len() - 1;
                inputs[part][last] = poison;
                assert!(run(&mut state, &inputs, 9).is_err(), "masked nonfinite KDA input accepted");
                let inputs = projections(9, state.inner, state.heads, 24);
                assert!(run(&mut state, &inputs, 9).is_err(), "KDA status was not sticky");
                state.reset();
                assert!(run(&mut state, &inputs, 9).is_ok(), "KDA reset did not recover");
            }
        }
    }
}
