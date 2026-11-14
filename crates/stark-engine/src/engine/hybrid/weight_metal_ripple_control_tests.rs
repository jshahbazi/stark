// SPDX-License-Identifier: Apache-2.0
//! Private synthetic Ripple controls: numerical, boundary and GPU components.
use stark_engine_macros::stark_test;

use super::*;

const HEADS: usize = 80;
const KV: usize = 8;
const DIM: usize = 128;
const GUARD: usize = 4;
const SENTINEL: f32 = 7.25;

struct Case {
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    p: [u32; 8],
    q: Vec<f32>,
    k: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

fn pipeline(name: &str) -> PipelineState {
    let executor = executor().unwrap();
    let function = executor.library.new_function_with_name(name).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

fn case(
    position: usize,
    slot: usize,
    capacity: usize,
    rope: bool,
    seed: usize,
) -> Case {
    let values = |n, phase: f32| (0..n).map(|i| ((i + seed) as f32 * 0.017 + phase).sin() * 2.0).collect::<Vec<_>>();
    let q = values(HEADS * DIM, 0.3);
    let k = values(KV * DIM, 1.7);
    let v = values(KV * DIM, 2.9);
    let qn = (0..DIM).map(|i| 0.7 + (i % 31) as f32 * 0.017).collect::<Vec<_>>();
    let kn = (0..DIM).map(|i| 0.6 + (i % 23) as f32 * 0.023).collect::<Vec<_>>();
    let output = |n| vec![SENTINEL; n + GUARD * 2];
    let mut buffers = Vec::new();
    for x in [&q, &k, &v, &qn, &kn, &output(HEADS * DIM), &output(capacity * KV * DIM), &output(capacity * KV * DIM)] {
        buffers.push(buffer(executor().unwrap(), bytemuck::cast_slice(x)).unwrap());
    }
    buffers.push(buffer(executor().unwrap(), bytemuck::cast_slice(&[0u32])).unwrap());
    Case {
        buffers,
        p: [
            HEADS as u32,
            KV as u32,
            DIM as u32,
            position as u32,
            slot as u32,
            u32::from(rope),
            1e-6f32.to_bits(),
            10000f32.to_bits(),
        ],
        q,
        k,
        qn,
        kn,
    }
}

fn dispatch(
    pipeline: &PipelineState,
    c: &Case,
    parallel: bool,
    repeats: usize,
) -> f64 {
    let executor = executor().unwrap();
    let command = executor.queue.command_buffer().unwrap();
    let encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in c.buffers[..8].iter().enumerate() {
        let offset = if index >= 5 {
            GUARD * 4
        } else {
            0
        };
        encoder.set_buffer(Some(buffer), offset, index);
    }
    encoder.set_bytes(NonNull::from(&c.p).cast(), std::mem::size_of_val(&c.p), 8);
    encoder.set_buffer(Some(&c.buffers[8]), 0, 30);
    for _ in 0..repeats {
        encoder.dispatch_threadgroups(
            MTLSize {
                width: HEADS + KV,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: if parallel {
                    32
                } else {
                    1
                },
                height: 1,
                depth: 1,
            },
        );
        encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    command.gpu_end_time() - command.gpu_start_time()
}

fn read(
    c: &Case,
    index: usize,
) -> Vec<f32> {
    let b = &c.buffers[index];
    // SAFETY: callers wait for the sole writer before reading shared storage.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<f32>(), b.length() / 4).to_vec() }
}

fn failure(c: &Case) -> u32 {
    // SAFETY: callers wait for the sole writer before reading this status.
    unsafe { *c.buffers[8].contents().as_ptr().cast::<u32>() }
}

fn oracle(
    c: &Case,
    query: bool,
) -> Vec<f64> {
    let (input, norm, heads) = if query {
        (&c.q, &c.qn, HEADS)
    } else {
        (&c.k, &c.kn, KV)
    };
    let mut output = vec![0.0; heads * DIM];
    for head in 0..heads {
        let x = &input[head * DIM..(head + 1) * DIM];
        let scale =
            1.0 / (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / DIM as f64 + f64::from(1e-6f32)).sqrt();
        for d in 0..DIM {
            let value = f64::from(x[d]) * scale * f64::from(norm[d]);
            output[head * DIM + d] = if c.p[5] != 0 {
                let r = d % (DIM / 2);
                let partner = if d < DIM / 2 {
                    d + DIM / 2
                } else {
                    d - DIM / 2
                };
                let companion = f64::from(x[partner]) * scale * f64::from(norm[partner]);
                let phase = f64::from(c.p[3]) * 10000f64.powf(-2.0 * r as f64 / DIM as f64);
                let (sine, cosine) = phase.sin_cos();
                if d < DIM / 2 {
                    value * cosine - companion * sine
                } else {
                    companion * sine + value * cosine
                }
            } else {
                value
            };
        }
    }
    output
}

#[stark_test]
fn ripple_parallel_prepare_preserves_scalar_order_oracle_and_bounds() {
    let old = pipeline("ripple_graph_attention_prepare");
    let new = pipeline("ripple_graph_attention_prepare_parallel");
    for rope in [false, true] {
        for (position, slot, capacity) in [(0, 0, 1), (7, 2, 3), (31, 31, 67), (63, 66, 67), (1000, 3, 67)] {
            let c = case(position, slot, capacity, rope, 29);
            dispatch(&old, &c, false, 1);
            let original = (5..8).map(|index| read(&c, index)).collect::<Vec<_>>();
            assert_eq!(failure(&c), 0);
            for b in &c.buffers[5..8] {
                // SAFETY: the previous dispatch has completed; reset every
                // output so missing writes cannot inherit reference values.
                unsafe {
                    std::slice::from_raw_parts_mut(b.contents().as_ptr().cast::<f32>(), b.length() / 4).fill(SENTINEL);
                }
            }
            dispatch(&new, &c, true, 1);
            for (index, expected) in (5..8).zip(&original) {
                let actual = read(&c, index);
                assert!(
                    actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "parallel prepare changed scalar bits"
                );
                assert!(actual[..GUARD].iter().all(|&x| x == SENTINEL));
                assert!(actual[actual.len() - GUARD..].iter().all(|&x| x == SENTINEL));
            }
            assert_eq!(failure(&c), 0);
            for query in [false, true] {
                let actual = read(
                    &c,
                    if query {
                        5
                    } else {
                        6
                    },
                );
                let begin = GUARD
                    + if query {
                        0
                    } else {
                        slot * KV * DIM
                    };
                let expected = oracle(&c, query);
                // F64 trigonometry is independent; F32 phase rounding grows at
                // larger positions. Scalar bit parity above remains exact.
                for (&a, &b) in actual[begin..begin + expected.len()].iter().zip(&expected) {
                    assert!((f64::from(a) - b).abs() <= 0.0003 + 0.00003 * b.abs(), "F64 prepare mismatch");
                }
            }
        }
    }
    for index in [0usize, 1, 2, 3, 4] {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let c = case(63, 3, 67, true, 31);
            // SAFETY: no dispatch is in flight and each selected buffer holds
            // at least one writable F32 synthetic input element.
            unsafe {
                *c.buffers[index].contents().as_ptr().cast::<f32>() = poison;
            }
            dispatch(&new, &c, true, 1);
            // F32::MAX in value or norm can remain finite without overflow;
            // NaN/Inf and square overflow in Q/K must always be rejected.
            if poison != f32::MAX || index < 2 {
                assert_ne!(failure(&c), 0);
            }
        }
    }
}

#[stark_test]
#[ignore]
fn ripple_prepare_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let pipelines = [pipeline("ripple_graph_attention_prepare"), pipeline("ripple_graph_attention_prepare_parallel")];
    for rope in [false, true] {
        let banks = (0..16).map(|i| case(1000 + i, i % 67, 67, rope, i * 29)).collect::<Vec<_>>();
        let mut times = [Vec::new(), Vec::new()];
        let mut tick = 0;
        for replay in 0..9 {
            for index in if replay % 2 == 0 {
                [0, 1, 1, 0]
            } else {
                [1, 0, 0, 1]
            } {
                let c = &banks[tick % banks.len()];
                tick += 1;
                let seconds = dispatch(&pipelines[index], c, index == 1, 32) / 32.0;
                assert_eq!(failure(c), 0);
                if replay >= 2 {
                    times[index].push(seconds);
                }
            }
        }
        for (index, values) in times.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"component":"ripple_attention_prepare", "rope":rope,"heads":HEADS,"kv_heads":KV,"dim":DIM,"schedule":if index==0 {"scalar"} else {"parallel_scalar_rms"},"math_mode":"safe","timer_scope":"serial_ABBA_16_rotating_cases_32_dispatch_GPU_timestamps","warm_samples":values.len(),"median_gpu_microseconds":values[values.len()/2]*1e6})
            );
        }
    }
}

struct SoftmaxCase {
    scores: Retained<ProtocolObject<dyn MTLBuffer>>,
    failure: Retained<ProtocolObject<dyn MTLBuffer>>,
    p: [u32; 6],
    initial: Vec<f32>,
}

fn softmax_case(
    count: usize,
    capacity: usize,
    seed: usize,
) -> SoftmaxCase {
    let mut initial = vec![SENTINEL; HEADS * capacity + GUARD * 2];
    for head in 0..HEADS {
        for token in 0..count {
            initial[GUARD + head * capacity + token] = ((head * 31 + token + seed) as f32 * 0.017).sin() * 8.0;
        }
    }
    SoftmaxCase {
        scores: buffer(executor().unwrap(), bytemuck::cast_slice(&initial)).unwrap(),
        failure: buffer(executor().unwrap(), bytemuck::cast_slice(&[0u32])).unwrap(),
        p: [count as u32, HEADS as u32, KV as u32, DIM as u32, capacity as u32, 0],
        initial,
    }
}

fn reset_softmax(c: &SoftmaxCase) {
    // SAFETY: no GPU writer is in flight; buffers retain complete initialized
    // synthetic score storage and one initialized atomic status element.
    unsafe {
        std::ptr::copy_nonoverlapping(c.initial.as_ptr(), c.scores.contents().as_ptr().cast::<f32>(), c.initial.len());
        *c.failure.contents().as_ptr().cast::<u32>() = 0;
    }
}

fn read_softmax(c: &SoftmaxCase) -> (Vec<f32>, u32) {
    // SAFETY: callers have completed the sole GPU writer of both buffers.
    unsafe {
        (
            std::slice::from_raw_parts(c.scores.contents().as_ptr().cast::<f32>(), c.initial.len()).to_vec(),
            *c.failure.contents().as_ptr().cast::<u32>(),
        )
    }
}

fn dispatch_softmax(
    pipeline: &PipelineState,
    c: &SoftmaxCase,
    parallel: bool,
    repeats: usize,
) -> f64 {
    let executor = executor().unwrap();
    let command = executor.queue.command_buffer().unwrap();
    let encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(Some(&c.scores), GUARD * 4, 0);
    encoder.set_bytes(NonNull::from(&c.p).cast(), std::mem::size_of_val(&c.p), 1);
    encoder.set_buffer(Some(&c.failure), 0, 30);
    for _ in 0..repeats {
        encoder.dispatch_threadgroups(
            MTLSize {
                width: HEADS,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: if parallel {
                    32
                } else {
                    1
                },
                height: 1,
                depth: 1,
            },
        );
        encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    command.gpu_end_time() - command.gpu_start_time()
}

#[stark_test]
fn ripple_parallel_softmax_preserves_order_and_finite_failures() {
    let old = pipeline("ripple_graph_attention_softmax");
    let new = pipeline("ripple_graph_attention_softmax_parallel");
    for count in [1usize, 31, 32, 33, 64, 1024] {
        let c = softmax_case(count, count + 3, 29);
        dispatch_softmax(&old, &c, false, 1);
        let reference = read_softmax(&c);
        reset_softmax(&c);
        dispatch_softmax(&new, &c, true, 1);
        let actual = read_softmax(&c);
        assert_eq!(reference.1, 0);
        assert_eq!(actual.1, 0);
        assert!(
            actual.0.iter().zip(&reference.0).all(|(a, b)| a.to_bits() == b.to_bits()),
            "softmax finite bits changed"
        );
        for head in 0..HEADS {
            let x = &c.initial[GUARD + head * (count + 3)..GUARD + head * (count + 3) + count];
            let maximum = x.iter().map(|&v| f64::from(v)).fold(f64::NEG_INFINITY, f64::max);
            let denom = x.iter().map(|&v| (f64::from(v) - maximum).exp()).sum::<f64>();
            for (token, &v) in x.iter().enumerate() {
                let expected = (f64::from(v) - maximum).exp() / denom;
                let observed = f64::from(actual.0[GUARD + head * (count + 3) + token]);
                assert!((observed - expected).abs() <= 2e-6, "softmax F64 mismatch");
            }
        }
    }
    for all in [false, true] {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, -f32::MAX] {
            let mut c = softmax_case(33, 37, 31);
            if all {
                for head in 0..HEADS {
                    c.initial[GUARD + head * 37..GUARD + head * 37 + 33].fill(poison);
                }
            } else {
                c.initial[GUARD] = poison;
            }
            reset_softmax(&c);
            dispatch_softmax(&old, &c, false, 1);
            let reference = read_softmax(&c);
            reset_softmax(&c);
            dispatch_softmax(&new, &c, true, 1);
            let actual = read_softmax(&c);
            assert_eq!(actual.1, reference.1, "softmax rejection changed");
            assert!(
                actual.0.iter().zip(&reference.0).all(|(a, b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()),
                "softmax nonfinite behavior changed"
            );
        }
    }
}

#[stark_test]
#[ignore]
fn ripple_softmax_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let pipelines = [pipeline("ripple_graph_attention_softmax"), pipeline("ripple_graph_attention_softmax_parallel")];
    for count in [1usize, 8, 16, 32, 64, 1024] {
        let banks = (0..16).map(|i| softmax_case(count, count + 3, i * 29)).collect::<Vec<_>>();
        let mut times = [Vec::new(), Vec::new()];
        let mut tick = 0;
        for replay in 0..9 {
            for index in if replay % 2 == 0 {
                [0, 1, 1, 0]
            } else {
                [1, 0, 0, 1]
            } {
                let c = &banks[tick % banks.len()];
                tick += 1;
                reset_softmax(c);
                let seconds = dispatch_softmax(&pipelines[index], c, index == 1, 32) / 32.0;
                assert_eq!(read_softmax(c).1, 0);
                if replay >= 2 {
                    times[index].push(seconds);
                }
            }
        }
        for (index, values) in times.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"component":"ripple_attention_softmax","count":count,"capacity":count+3,"heads":HEADS,"schedule":if index==0 {"scalar"} else {"parallel_ordered_reductions"},"math_mode":"safe","timer_scope":"serial_ABBA_16_rotating_cases_32_dispatch_GPU_timestamps","warm_samples":values.len(),"median_gpu_microseconds":values[values.len()/2]*1e6})
            );
        }
    }
}
