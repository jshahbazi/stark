// SPDX-License-Identifier: Apache-2.0
//! Component ceiling only: validated bounded inputs, omitted scan cost.
//! The bounded shader wrapper is test-only and never selected by a runtime graph.
use stark_engine_macros::stark_test;

use super::*;

fn pipeline(
    kind: u32,
    bounded: bool,
) -> PipelineState {
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in
        [(0, kind), (4, 256), (5, weights::block_layout(kind).unwrap().1 as u32), (73, u32::from(bounded))]
    {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let name = match (kind, bounded) {
        (12, false) => "hybrid_q4_grouped32_matvec",
        (12, true) => "hybrid_ripple_q4_grouped32_bounded_matvec",
        (14, _) => "hybrid_q6_grouped16_matvec",
        _ => unreachable!(),
    };
    // Deliberately private PSOs: function constant 73 must never populate a
    // production cache keyed only by MatrixFormat.
    let executor = executor().unwrap();
    let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

fn oracle(
    kind: u32,
    encoded: &[u8],
    input: &[f32],
) -> (f64, f64) {
    let bytes = weights::block_layout(kind).unwrap().1;
    let mut total = 0.0f64;
    let mut absolute = 0.0f64;
    for (block, x) in encoded.chunks_exact(bytes).zip(input.chunks_exact(256)) {
        let mut decoded = [0.0f32; 256];
        weights::decode_block(kind, block, &mut decoded).unwrap();
        for (&w, &x) in decoded.iter().zip(x) {
            let value = f64::from(w) * f64::from(x);
            total += value;
            absolute += value.abs();
        }
    }
    (total, absolute)
}

fn dispatch(
    pipeline: &PipelineState,
    kind: u32,
    encoded: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    columns: usize,
    banks: usize,
    replay: usize,
) -> (Vec<f32>, f64) {
    let executor = executor().unwrap();
    let matrix_bytes = rows * columns / 256 * weights::block_layout(kind).unwrap().1;
    objc2::rc::autoreleasepool(|_| {
        let command = executor.queue.command_buffer().unwrap();
        let encoding = command.compute_command_encoder().unwrap();
        encoding.set_compute_pipeline_state(pipeline);
        encoding.set_buffer(Some(input), 0, 1);
        let p = [rows as u32, columns as u32, kind, 256, weights::block_layout(kind).unwrap().1 as u32, 0, 0, 0];
        let offsets = [0u32; 4];
        encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
        for i in 0..banks {
            let bank = (i + replay) % banks;
            encoding.set_buffer(Some(encoded), bank * matrix_bytes, 0);
            encoding.set_buffer(Some(output), bank * rows * 4, 2);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: rows.div_ceil(if kind == 12 {
                        2
                    } else {
                        4
                    }),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoding.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: owned Shared output banks have completed their only writers.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * banks) }.to_vec();
        (values, (command.gpu_end_time() - command.gpu_start_time()) / banks as f64)
    })
}

#[stark_test]
fn ripple_bounded_component_preserves_normal_input_bits_and_f64() {
    for kind in [12u32, 14] {
        let pipelines = [pipeline(kind, false), pipeline(kind, true)];
        for columns in [256usize, 768, 5120] {
            let rows = 13;
            let bytes = super::tests::finite_packed_codes(kind, rows, columns, 31);
            let row_bytes = bytes.len() / rows;
            let encoded = buffer(executor().unwrap(), &bytes).unwrap();
            let output =
                executor().unwrap().device.new_buffer(rows * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
            for magnitude in [0.0f32, 1e-20, 1.0, 1e10, 1e20] {
                let x = (0..columns).map(|i| (i as f32 * 0.017).sin() * magnitude).collect::<Vec<_>>();
                assert!(x.iter().all(|v| v.is_finite() && v.abs() <= 1e20));
                let input = buffer(executor().unwrap(), bytemuck::cast_slice(&x)).unwrap();
                let reference = dispatch(&pipelines[0], kind, &encoded, &input, &output, rows, columns, 1, 0).0;
                let bounded = dispatch(&pipelines[1], kind, &encoded, &input, &output, rows, columns, 1, 0).0;
                assert!(
                    reference.iter().zip(&bounded).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "bounded component changed normal output bits"
                );
                for row in 0..rows {
                    let (expected, absolute) = oracle(kind, &bytes[row * row_bytes..(row + 1) * row_bytes], &x);
                    assert!((f64::from(bounded[row]) - expected).abs() <= 2e-6 * absolute.max(1.0));
                }
            }
        }
    }
}

#[stark_test]
#[ignore]
fn ripple_bounded_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let executor = executor().unwrap();
    for (kind, label, rows, columns) in [
        (12u32, "q4_gate_up", 40960usize, 5120usize),
        (12, "q4_down", 5120, 40960),
        (12, "q4_q_gate", 10240, 5120),
        (12, "q4_o", 5120, 10240),
        (14, "q6_down", 5120, 40960),
        (14, "q6_k_v", 1024, 5120),
        (14, "q6_head", 100352, 5120),
    ] {
        let pipelines = [pipeline(kind, false), pipeline(kind, true)];
        let matrix_bytes = rows * columns / 256 * weights::block_layout(kind).unwrap().1;
        let banks = (320usize * 1024 * 1024).div_ceil(matrix_bytes).max(2);
        let bytes = super::tests::finite_packed_codes(kind, rows * banks, columns, 37);
        let x = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        assert!(x.iter().all(|v| v.is_finite() && v.abs() <= 1e20));
        let row_bytes = matrix_bytes / rows;
        let mut expected = Vec::new();
        for bank in 0..banks {
            for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                let at = bank * matrix_bytes + row * row_bytes;
                expected.push((bank * rows + row, oracle(kind, &bytes[at..at + row_bytes], &x)));
            }
        }
        let encoded = buffer(executor, &bytes).unwrap();
        let input = buffer(executor, bytemuck::cast_slice(&x)).unwrap();
        let output = executor.device.new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let mut times = [Vec::new(), Vec::new()];
        let mut reference = Vec::new();
        let mut max_f64 = [0.0f64; 2];
        for replay in 0..9 {
            for variant in if replay % 2 == 0 {
                [0usize, 1, 1, 0]
            } else {
                [1usize, 0, 0, 1]
            } {
                let (values, seconds) =
                    dispatch(&pipelines[variant], kind, &encoded, &input, &output, rows, columns, banks, replay);
                assert!(values.iter().all(|v| v.is_finite()));
                if reference.is_empty() {
                    reference = values.clone();
                }
                assert!(
                    values.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "bounded experiment changed normal output bits"
                );
                for &(row, (target, absolute)) in &expected {
                    max_f64[variant] =
                        max_f64[variant].max((f64::from(values[row]) - target).abs() / absolute.max(1.0));
                }
                assert!(max_f64[variant] <= 2e-6);
                if replay > 0 {
                    times[variant].push(seconds);
                }
            }
        }
        for (variant, samples) in times.iter_mut().enumerate() {
            samples.sort_by(f64::total_cmp);
            let seconds = (samples[7] + samples[8]) * 0.5;
            println!(
                "{}",
                serde_json::json!({"component":label,"kind":kind,"rows":rows,"columns":columns,"bounded_only":variant==1,"production_admitted":false,"input_envelope_proven_on_host":true,"envelope_scan_cost_included":false,"normal_outputs_bit_exact":true,"math_mode":"safe","timer_scope":"serial_ABBA_rotating_banks_GPU_timestamps","encoded_population_bytes":matrix_bytes*banks,"rotating_banks":banks,"warm_samples":samples.len(),"median_gpu_microseconds_per_matrix":seconds*1e6,"encoded_gigabytes_per_second":matrix_bytes as f64/seconds/1e9,"maximum_f64_error_over_absolute_sum":max_f64[variant]})
            );
        }
    }
}
