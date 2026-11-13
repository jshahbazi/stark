// SPDX-License-Identifier: Apache-2.0
//! Synthetic component qualification only. No runtime attention selection.
use stark_engine_macros::stark_test;

use super::*;

const HEADS: usize = 80;
const KV: usize = 8;
const DIM: usize = 128;
const GUARD: usize = 3;
const SENTINEL: f32 = 7.25;
struct Case {
    p: [u32; 8],
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probabilities: Vec<f32>,
    gates: Vec<f32>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}
fn pipeline(name: &str) -> PipelineState {
    static CACHE: OnceLock<Mutex<HashMap<String, PipelineState>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(pipeline) = cache.get(name) {
        return pipeline.clone();
    }
    let executor = executor().unwrap();
    let f = executor.library.new_function_with_name(name).unwrap();
    let pipeline = executor.device.new_compute_pipeline_state_with_function(&f).unwrap();
    cache.insert(name.to_owned(), pipeline.clone());
    pipeline
}
fn padded(values: &[f32]) -> Vec<f32> {
    let mut result = vec![SENTINEL; values.len() + 2 * GUARD];
    result[GUARD..GUARD + values.len()].copy_from_slice(values);
    result
}
fn case(
    count: usize,
    capacity: usize,
    start: usize,
    gated: bool,
    tile: usize,
) -> Case {
    assert!(count > 0 && count <= capacity && count <= 4096 && start < capacity && matches!(tile, 64 | 128));
    let q = (0..HEADS * DIM).map(|i| ((i as f32 * 0.013 + 0.7).sin() * 1.3)).collect::<Vec<_>>();
    let mut used = vec![false; capacity];
    for token in 0..count {
        used[(start + token) % capacity] = true;
    }
    let mut k = vec![f32::NAN; capacity * KV * DIM];
    let mut v = k.clone();
    for slot in 0..capacity {
        if used[slot] {
            for i in 0..KV * DIM {
                k[slot * KV * DIM + i] = ((slot * 17 + i) as f32 * 0.019 + 1.3).sin() * 1.7;
                v[slot * KV * DIM + i] = ((slot * 13 + i) as f32 * 0.023 + 2.9).sin() * 2.1;
            }
        }
    }
    let mut probabilities = vec![f32::NAN; HEADS * capacity];
    for head in 0..HEADS {
        let weights = (0..count).map(|token| ((token * 3 + head * 11) as f64 * 0.017).sin().exp()).collect::<Vec<_>>();
        let denominator: f64 = weights.iter().sum();
        for token in 0..count {
            probabilities[head * capacity + token] = (weights[token] / denominator) as f32;
        }
    }
    let gates = (0..HEADS * DIM)
        .map(|i| {
            if gated {
                (i as f32 * 0.037 + 0.3).sin() * 3.0
            } else {
                f32::NAN
            }
        })
        .collect::<Vec<_>>();
    let mut c = Case {
        p: [
            count as u32,
            HEADS as u32,
            KV as u32,
            DIM as u32,
            capacity as u32,
            start as u32,
            u32::from(gated),
            tile as u32,
        ],
        q,
        k,
        v,
        probabilities,
        gates,
        buffers: Vec::new(),
    };
    for x in [
        &c.q,
        &c.k,
        &vec![SENTINEL; HEADS * capacity],
        &c.probabilities,
        &c.v,
        &c.gates,
        &vec![SENTINEL; HEADS * DIM],
        &vec![SENTINEL; HEADS * count.div_ceil(tile) * DIM],
    ] {
        c.buffers.push(buffer(executor().unwrap(), bytemuck::cast_slice(&padded(x))).unwrap());
    }
    c.buffers.push(buffer(executor().unwrap(), bytemuck::cast_slice(&[0u32])).unwrap());
    c
}
fn sync_inputs(c: &Case) {
    for (index, x) in [(0, &c.q), (1, &c.k), (3, &c.probabilities), (4, &c.v), (5, &c.gates)] {
        // SAFETY: no command is outstanding and these Shared buffers own spans.
        unsafe {
            std::ptr::copy_nonoverlapping(
                x.as_ptr(),
                c.buffers[index].contents().as_ptr().cast::<f32>().add(GUARD),
                x.len(),
            );
        }
    }
}
fn reset(
    c: &Case,
    sticky: u32,
) {
    for index in [2usize, 6, 7] {
        // SAFETY: no GPU command is outstanding; buffers are owned Shared spans.
        unsafe {
            std::slice::from_raw_parts_mut(
                c.buffers[index].contents().as_ptr().cast::<f32>(),
                c.buffers[index].length() / 4,
            )
            .fill(SENTINEL);
        }
    }
    unsafe {
        *c.buffers[8].contents().as_ptr().cast::<u32>() = sticky;
    }
}
fn read(
    c: &Case,
    index: usize,
) -> Vec<f32> {
    // SAFETY: every caller has waited for the sole GPU writer.
    let all = unsafe {
        std::slice::from_raw_parts(c.buffers[index].contents().as_ptr().cast::<f32>(), c.buffers[index].length() / 4)
    };
    assert!(
        all[..GUARD].iter().chain(&all[all.len() - GUARD..]).all(|&v| v.to_bits() == SENTINEL.to_bits()),
        "attention guard overwritten"
    );
    all[GUARD..all.len() - GUARD].to_vec()
}
fn failure(c: &Case) -> u32 {
    unsafe { *c.buffers[8].contents().as_ptr().cast::<u32>() }
}
fn bind(
    encoding: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    c: &Case,
    indices: &[usize],
) {
    for (slot, &index) in indices.iter().enumerate() {
        encoding.set_buffer(Some(&c.buffers[index]), GUARD * 4, slot);
    }
    encoding.set_buffer(Some(&c.buffers[8]), 0, 30);
}
fn dispatch(
    c: &Case,
    component: usize,
    variant: usize,
    repeats: usize,
) -> f64 {
    let old_score = pipeline("ripple_graph_attention_scores");
    let new_score = pipeline("ripple_graph_attention_scores_parallel");
    let old_output = pipeline("ripple_graph_attention_output");
    let partial = pipeline("ripple_graph_attention_output_partial");
    let reduce = pipeline("ripple_graph_attention_output_reduce");
    let count = c.p[0] as usize;
    let groups = HEADS * count.div_ceil(c.p[7] as usize);
    let executor = executor().unwrap();
    let command = executor.queue.command_buffer().unwrap();
    let e = command.compute_command_encoder().unwrap();
    for _ in 0..repeats {
        if component == 0 {
            bind(&e, c, &[0, 1, 2]);
            e.set_bytes(NonNull::from(&c.p).cast(), std::mem::size_of_val(&c.p), 3);
            e.set_compute_pipeline_state(if variant == 0 {
                &old_score
            } else {
                &new_score
            });
            // Match production dispatch_threads, including count=1/8 tails.
            e.dispatch_threads(
                MTLSize {
                    width: if variant == 0 {
                        HEADS * count
                    } else {
                        HEADS * count * 32
                    },
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: if variant == 0 {
                        256
                    } else {
                        32
                    },
                    height: 1,
                    depth: 1,
                },
            );
        } else if variant == 0 {
            bind(&e, c, &[3, 4, 5, 6]);
            e.set_bytes(NonNull::from(&c.p).cast(), std::mem::size_of_val(&c.p), 4);
            e.set_compute_pipeline_state(&old_output);
            e.dispatch_threadgroups(
                MTLSize {
                    width: (HEADS * DIM).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            bind(&e, c, &[3, 4, 7]);
            e.set_bytes(NonNull::from(&c.p).cast(), std::mem::size_of_val(&c.p), 3);
            e.set_compute_pipeline_state(&partial);
            e.dispatch_threadgroups(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: DIM,
                    height: 1,
                    depth: 1,
                },
            );
            e.memory_barrier_with_scope(MTLBarrierScope::Buffers);
            bind(&e, c, &[7, 3, 4, 5, 6]);
            e.set_bytes(NonNull::from(&c.p).cast(), std::mem::size_of_val(&c.p), 5);
            e.set_compute_pipeline_state(&reduce);
            e.dispatch_threadgroups(
                MTLSize {
                    width: (HEADS * DIM).div_ceil(128),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
        e.memory_barrier_with_scope(MTLBarrierScope::Buffers);
    }
    e.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    (command.gpu_end_time() - command.gpu_start_time()) / repeats as f64
}
fn oracle(
    c: &Case,
    component: usize,
) -> Vec<f64> {
    let count = c.p[0] as usize;
    let capacity = c.p[4] as usize;
    let start = c.p[5] as usize;
    let mut output = Vec::new();
    for head in 0..HEADS {
        for j in 0..if component == 0 {
            count
        } else {
            DIM
        } {
            let mut sum = 0.0f64;
            if component == 0 {
                let slot = (start + j) % capacity;
                for d in 0..DIM {
                    sum += f64::from(c.q[head * DIM + d])
                        * f64::from(c.k[slot * KV * DIM + head / (HEADS / KV) * DIM + d]);
                }
                sum /= (DIM as f64).sqrt();
            } else {
                for token in 0..count {
                    let slot = (start + token) % capacity;
                    sum += f64::from(c.probabilities[head * capacity + token])
                        * f64::from(c.v[slot * KV * DIM + head / (HEADS / KV) * DIM + j]);
                }
                if c.p[6] != 0 {
                    sum *= 1.0 / (1.0 + (-f64::from(c.gates[head * DIM + j])).exp());
                }
            }
            output.push(sum);
        }
    }
    output
}
fn active(
    c: &Case,
    component: usize,
) -> Vec<f32> {
    if component == 1 {
        return read(c, 6);
    }
    let all = read(c, 2);
    let count = c.p[0] as usize;
    let cap = c.p[4] as usize;
    let mut values = Vec::new();
    for head in 0..HEADS {
        values.extend_from_slice(&all[head * cap..head * cap + count]);
        assert!(all[head * cap + count..(head + 1) * cap].iter().all(|&v| v.to_bits() == SENTINEL.to_bits()));
    }
    values
}
fn compare(
    values: &[f32],
    reference: &[f64],
) -> f64 {
    let mut error = 0.0f64;
    for (&v, &r) in values.iter().zip(reference) {
        error = error.max((f64::from(v) - r).abs());
        assert!((f64::from(v) - r).abs() <= 2e-5 + 2e-5 * r.abs(), "attention F64 bound");
    }
    error
}

#[stark_test]
fn ripple_parallel_attention_matches_f64_ring_wrap_tails_and_gates() {
    let mut max_error = [0.0f64; 2];
    for count in [1usize, 8, 16, 64, 128, 256, 257, 1024] {
        for start in [0usize, count + 4] {
            for gated in [false, true] {
                for tile in [64usize, 128] {
                    let c = case(count, count + 7, start, gated, tile);
                    for component in 0..2 {
                        let reference = oracle(&c, component);
                        for variant in 0..2 {
                            reset(&c, 0);
                            dispatch(&c, component, variant, 1);
                            assert_eq!(failure(&c), 0);
                            max_error[component] =
                                max_error[component].max(compare(&active(&c, component), &reference));
                            if variant == 1 && component == 1 {
                                read(&c, 7);
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"scope":"synthetic_attention_f64","max_abs_score_error":max_error[0],"max_abs_output_error":max_error[1],"heads":HEADS,"kv_heads":KV,"dim":DIM,"counts":[1,8,16,64,128,256,257,1024],"gates_on_off":true,"nonpower_ring_wrap":true})
    );
}

#[stark_test]
fn ripple_parallel_attention_preserves_extreme_fallback_and_sticky_failures() {
    for component in 0..2 {
        for poison in 0..8 {
            let mut c = case(256, 263, 260, true, 64);
            if component == 0 {
                match poison {
                    0 => c.q.fill(f32::MAX),
                    1 => c.q.fill(f32::NAN),
                    2 => c.k[(260 * KV) * DIM] = f32::INFINITY,
                    3 => {
                        c.q.fill(f32::MAX);
                        c.k.fill(0.0);
                    },
                    4 => {
                        c.q.fill(1e20);
                        c.k.fill(1e20);
                    },
                    5 => {
                        c.q.fill(f32::MAX);
                        c.k.fill(0.0);
                        c.k[260 * KV * DIM] = 1.0;
                    },
                    6 => {},
                    _ => {
                        c.q.fill(1e20);
                        c.k.fill(-1e20);
                    },
                }
            } else {
                match poison {
                    0 => c.v.fill(f32::MAX),
                    1 => c.probabilities[0] = f32::NAN,
                    2 => c.v[260 * KV * DIM] = f32::INFINITY,
                    3 => {
                        c.v.fill(f32::MAX);
                        for head in 0..HEADS {
                            for token in 0..256 {
                                c.probabilities[head * 263 + token] = 0.0;
                            }
                        }
                    },
                    4 => c.gates[0] = f32::NAN,
                    5 => {
                        c.v.fill(1e20);
                        for head in 0..HEADS {
                            for token in 0..256 {
                                c.probabilities[head * 263 + token] = 1e20;
                            }
                        }
                    },
                    6 => {},
                    _ => {
                        c.p[6] = 0;
                        c.gates.fill(f32::NAN);
                    },
                }
            }
            sync_inputs(&c);
            let sticky = u32::from(poison == 6);
            reset(&c, sticky);
            dispatch(&c, component, 0, 1);
            let old = active(&c, component);
            let old_failure = failure(&c);
            reset(&c, sticky);
            dispatch(&c, component, 1, 1);
            let new = active(&c, component);
            let new_failure = failure(&c);
            assert_eq!(old_failure, new_failure, "parallel attention failure contract");
            if (component == 0 && matches!(poison, 0 | 1 | 3 | 4 | 5 | 7))
                || (component == 1 && matches!(poison, 0 | 3 | 5))
            {
                assert!(
                    old.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
                    "extreme fallback changed original bits"
                );
            } else if poison == 6 {
                assert_ne!(new_failure, 0);
            } else {
                assert!(
                    old.iter().zip(&new).all(|(a, b)| {
                        (a.is_nan() && b.is_nan())
                            || a.to_bits() == b.to_bits()
                            || (a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5 + 2e-5 * a.abs())
                    }),
                    "parallel attention invalid-input output contract"
                );
            }
        }
    }
}

#[stark_test]
#[ignore]
fn ripple_parallel_attention_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    for count in [1usize, 8, 16, 64, 128, 256, 512, 1024, 4096] {
        for component in 0..2 {
            if component == 1 && count < 256 {
                continue;
            }
            for tile in if component == 0 {
                vec![64usize]
            } else {
                vec![64usize, 128]
            } {
                let c = case(count, count + 7, count + 4, true, tile);
                let reference = oracle(&c, component);
                let mut times = [Vec::new(), Vec::new()];
                let mut max_error = [0.0f64; 2];
                for replay in 0..9 {
                    for variant in if replay % 2 == 0 {
                        [0usize, 1, 1, 0]
                    } else {
                        [1usize, 0, 0, 1]
                    } {
                        reset(&c, 0);
                        let seconds = dispatch(&c, component, variant, 16);
                        assert_eq!(failure(&c), 0);
                        max_error[variant] = max_error[variant].max(compare(&active(&c, component), &reference));
                        if replay > 0 {
                            times[variant].push(seconds);
                        }
                    }
                }
                for (variant, t) in times.iter_mut().enumerate() {
                    t.sort_by(f64::total_cmp);
                    println!(
                        "{}",
                        serde_json::json!({"component":if component==0 {"attention_scores"} else {"attention_output"},"count":count,"capacity":count+7,"start":count+4,"tile":tile,"parallel":variant==1,"production_admitted":false,"heads":HEADS,"kv_heads":KV,"dim":DIM,"warm_samples":t.len(),"repeats_per_sample":16,"median_gpu_microseconds":(t[7]+t[8])*0.5e6,"minimum_gpu_microseconds":t[0]*1e6,"maximum_gpu_microseconds":t[t.len()-1]*1e6,"max_abs_error_vs_f64":max_error[variant],"timer_scope":"serial_ABBA_GPU_timestamps_including_two_output_dispatches_and_barrier","partial_bytes":HEADS*count.div_ceil(tile)*DIM*4})
                    );
                }
            }
        }
    }
}

#[stark_test]
fn ripple_parallel_attention_rejects_active_nonfinite_gates() {
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut c = case(257, 264, 261, true, 128);
        c.gates[0] = poison;
        sync_inputs(&c);
        reset(&c, 0);
        dispatch(&c, 1, 1, 1);
        assert_ne!(failure(&c), 0);
        // Gating-disabled calls never consume the gate payload. The actual
        // production graph separately checks each enabled gate projection.
        c.p[6] = 0;
        reset(&c, 0);
        dispatch(&c, 1, 1, 1);
        assert_eq!(failure(&c), 0);
        compare(&active(&c, 1), &oracle(&c, 1));
    }
}
