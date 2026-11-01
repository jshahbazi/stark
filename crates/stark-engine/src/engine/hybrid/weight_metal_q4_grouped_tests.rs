// SPDX-License-Identifier: Apache-2.0
//! Q4 head schedule numerical, admission and component checks.
use stark_engine_macros::stark_test;

use super::*;

fn pipeline(name: &str) -> PipelineState {
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, 12u32), (4, 256), (5, 144), (73, 0)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let executor = executor().unwrap();
    let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

fn codes(
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let mut state = 0x9135_acedu32;
    let mut codes = vec![0u8; rows * columns / 256 * 144];
    for byte in &mut codes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    for (index, block) in codes.chunks_exact_mut(144).enumerate() {
        block[..2]
            .copy_from_slice(&half::f16::from_f32(0.001 + (index % 257) as f32 * 0.00003125).to_bits().to_le_bytes());
        block[2..4]
            .copy_from_slice(&half::f16::from_f32(0.0004 + (index % 131) as f32 * 0.000015625).to_bits().to_le_bytes());
    }
    codes
}

fn run(
    pipeline: &PipelineState,
    codes: &[u8],
    input: &[f32],
    rows: usize,
    columns: usize,
    group: usize,
) -> Vec<f32> {
    let executor = executor().unwrap();
    // Exercise nonzero and deliberately unaligned mapped-code offsets. The
    // byte-reading shader must never reinterpret this prefix as encoded data.
    let mut prefixed = vec![0xadu8; 19];
    prefixed.extend_from_slice(codes);
    let codes = buffer(executor, &prefixed).unwrap();
    let input = buffer(executor, bytemuck::cast_slice(input)).unwrap();
    let output = buffer(executor, bytemuck::cast_slice(&vec![7.25f32; rows + 2])).unwrap();
    dispatch(pipeline, &codes, &input, &output, rows, columns, group, 4, 19);
    // SAFETY: dispatch waits for the owned shared output writer.
    let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows + 2) };
    assert_eq!(values[0], 7.25);
    assert_eq!(values[rows + 1], 7.25);
    values[1..rows + 1].to_vec()
}

fn dispatch(
    pipeline: &PipelineState,
    codes: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    columns: usize,
    group: usize,
    output_offset: usize,
    code_offset: u32,
) -> f64 {
    let executor = executor().unwrap();
    let command = executor.queue.command_buffer().unwrap();
    let encoder = command.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(Some(codes), 0, 0);
    encoder.set_buffer(Some(input), 0, 1);
    encoder.set_buffer(Some(output), output_offset, 2);
    let p = [rows as u32, columns as u32, 12, 256, 144, 0, group as u32, 0];
    let offsets = [code_offset, 0, 0, 0];
    encoder.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
    encoder.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows.div_ceil(2),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    command.gpu_end_time() - command.gpu_start_time()
}

#[stark_test]
fn q4_grouped_matches_f64_and_preserves_extreme_fallback() {
    let old = pipeline("hybrid_packed_factored_matvec");
    let new = pipeline("hybrid_q4_grouped32_matvec");
    for columns in [256usize, 768, 4096, 8192] {
        let (rows, batch_rows) = (39, 13);
        let codes = codes(rows, columns);
        let input = (0..columns * 3).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let actual = run(&new, &codes, &input, rows, columns, batch_rows);
        for row in 0..rows {
            let mut expected = 0.0f64;
            let mut absolute = 0.0f64;
            for (encoded, x) in codes[row * columns / 256 * 144..(row + 1) * columns / 256 * 144]
                .chunks_exact(144)
                .zip(input[row / batch_rows * columns..(row / batch_rows + 1) * columns].chunks_exact(256))
            {
                let mut decoded = [0.0f32; 256];
                weights::decode_block(12, encoded, &mut decoded).unwrap();
                for (&weight, &value) in decoded.iter().zip(x) {
                    let product = f64::from(weight) * f64::from(value);
                    expected += product;
                    absolute += product.abs();
                }
            }
            assert!((f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0));
        }
        for value in [1e30f32, f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let input = vec![value; columns * 3];
            let reference = run(&old, &codes, &input, rows, columns, batch_rows);
            let actual = run(&new, &codes, &input, rows, columns, batch_rows);
            for (a, b) in actual.iter().zip(reference) {
                assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan());
            }
        }
    }
}

#[stark_test]
fn q4_grouped_production_admits_only_head_and_keeps_safe_cache() {
    let format = MatrixFormat::Packed(12);
    let enabled = q4_grouped32_shape(format, 154_880, 4096);
    for (rows, columns) in [(154_880, 0), (154_879, 4096), (154_880, 8192), (13, 768)] {
        assert!(!q4_grouped32_shape(format, rows, columns));
    }
    assert!(!q4_grouped32_shape(MatrixFormat::Packed(13), 154_880, 4096));
    if !enabled {
        return;
    }
    let executor = executor().unwrap();
    // Populate the generic format-only cache before requesting the head PSO.
    executor.pipeline_for_geometry(format, 13, 768).unwrap();
    assert_eq!(matrix_groups(format, 154_880, 4096), 77_440);
    assert_eq!(matrix_threads(format, 154_880, 4096), 64);
    let (rows, columns) = (154_880, 4096);
    let codes = codes(rows, columns);
    let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let parameters = [rows as u32, columns as u32, 12, 256, 144, 0, 0, 0];
    let expected = run(&pipeline("hybrid_q4_grouped32_matvec"), &codes, &input, rows, columns, 0);
    let actual = executor.run_inner(&codes, &input, parameters, None, TestMatrixSchedule::Production).unwrap();
    assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(
        executor.run_inner(&codes, &vec![f32::NAN; columns], parameters, None, TestMatrixSchedule::Production).is_err()
    );
    assert!(
        executor
            .run_inner(&codes, &vec![f32::INFINITY; columns], parameters, None, TestMatrixSchedule::Production)
            .is_err()
    );
}

#[stark_test]
#[ignore]
fn q4_grouped_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let pipelines = [pipeline("hybrid_packed_factored_matvec"), pipeline("hybrid_q4_grouped32_matvec")];
    let (rows, columns) = (154_880, 4096);
    let executor = executor().unwrap();
    let codes = buffer(executor, &codes(rows, columns)).unwrap();
    let values = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let input = buffer(executor, bytemuck::cast_slice(&values)).unwrap();
    let output = buffer(executor, bytemuck::cast_slice(&vec![0.0f32; rows])).unwrap();
    let mut times = [Vec::new(), Vec::new()];
    let mut reference = Vec::new();
    let mut max_error = 0.0f64;
    for replay in 0..9 {
        for index in if replay % 2 == 0 {
            [0usize, 1, 1, 0]
        } else {
            [1usize, 0, 0, 1]
        } {
            let seconds = dispatch(&pipelines[index], &codes, &input, &output, rows, columns, 0, 0, 0);
            // SAFETY: dispatch completed the shared output writer.
            let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows) };
            assert!(values.iter().all(|v| v.is_finite()));
            if reference.is_empty() {
                reference = values.to_vec();
            }
            max_error = max_error.max(
                values.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0, f64::max),
            );
            if replay > 0 {
                times[index].push(seconds);
            }
        }
    }
    for (index, values) in times.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"component":"glm_q4_head", "schedule":(["factored2", "grouped32"][index]),
            "median_gpu_microseconds": (values[7] + values[8]) * 0.5e6, "warm_samples": values.len(),
            "maximum_absolute_error_vs_original": max_error, "math_mode":"safe", "timer_scope":"same_process_ABBA_GPU_timestamps"})
        );
    }
}
