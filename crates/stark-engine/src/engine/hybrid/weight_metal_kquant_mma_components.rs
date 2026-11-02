// SPDX-License-Identifier: Apache-2.0
//! Ignored component timings for experimental multi-token Q5/Q6 projections.

use stark_engine_macros::stark_test;

use super::*;

const BANKS: usize = 8;
const SAMPLE_ROWS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Timing {
    gpu: f64,
    encode: f64,
    commit: f64,
    wait: f64,
    roundtrip: f64,
}

#[derive(Clone, Copy)]
enum Schedule {
    TokenMajor,
    ProjectionMajor,
    Mma,
}

impl Schedule {
    fn label(self) -> &'static str {
        match self {
            Self::TokenMajor => "token_major_serial",
            Self::ProjectionMajor => "projection_major_serial",
            Self::Mma => "mma_bank_major",
        }
    }
}

fn dispatch(
    shape: Shape,
    schedule: Schedule,
    serial: &PipelineState,
    mma: &PipelineState,
    geometry: MmaGeometry,
    codes: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    code_bytes: usize,
    rotation: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    failure: &ProtocolObject<dyn MTLBuffer>,
    capture: bool,
) -> Result<(Timing, Option<Vec<f32>>)> {
    objc2::rc::autoreleasepool(|_| {
        let executor = executor()?;
        if codes.iter().any(|code| code.length() < code_bytes) {
            return Err(Error::Invalid("MMA component bank span"));
        }
        // The previous dispatch has completed before this single reusable
        // status word is cleared for the next command.
        unsafe { failure.contents().as_ptr().cast::<u32>().write(0) };
        let started = std::time::Instant::now();
        let command = executor.queue.command_buffer().ok_or(Error::Invalid("Kquant MMA command allocation failed"))?;
        let encoder =
            command.compute_command_encoder().ok_or(Error::Invalid("Kquant MMA encoder allocation failed"))?;
        match schedule {
            Schedule::TokenMajor => {
                for token in 0..shape.tokens {
                    for step in 0..codes.len() {
                        let bank = (step + rotation) % codes.len();
                        let code = &codes[bank];
                        encode_serial(
                            &encoder,
                            shape,
                            serial,
                            code,
                            input,
                            output,
                            [0, 0, bank * shape.tokens * shape.rows],
                            token,
                        )?;
                    }
                }
            },
            Schedule::ProjectionMajor => {
                for step in 0..codes.len() {
                    let bank = (step + rotation) % codes.len();
                    let code = &codes[bank];
                    for token in 0..shape.tokens {
                        encode_serial(
                            &encoder,
                            shape,
                            serial,
                            code,
                            input,
                            output,
                            [0, 0, bank * shape.tokens * shape.rows],
                            token,
                        )?;
                    }
                }
            },
            Schedule::Mma => {
                for step in 0..codes.len() {
                    let bank = (step + rotation) % codes.len();
                    let code = &codes[bank];
                    encode_mma(
                        &encoder,
                        shape,
                        mma,
                        geometry,
                        code,
                        input,
                        output,
                        failure,
                        [0, 0, bank * shape.tokens * shape.rows],
                    )?;
                }
            },
        }
        encoder.end_encoding();
        let encode_seconds = started.elapsed().as_secs_f64();
        let mark = std::time::Instant::now();
        command.commit();
        let commit_seconds = mark.elapsed().as_secs_f64();
        let mark = std::time::Instant::now();
        command.wait_until_completed();
        let wait_seconds = mark.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("Kquant MMA command failed"));
        }
        if unsafe { *failure.contents().as_ptr().cast::<u32>() } != 0 {
            return Err(Error::Invalid("Kquant MMA kernel reported failure"));
        }
        let roundtrip_seconds = started.elapsed().as_secs_f64();
        let gpu_seconds = command.gpu_end_time() - command.gpu_start_time();
        let values = if capture {
            let count = BANKS * shape.tokens * shape.rows;
            // SAFETY: the command completed and is the sole writer to this shared result buffer.
            Some(unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count) }.to_vec())
        } else {
            None
        };
        Ok((
            Timing {
                gpu: gpu_seconds,
                encode: encode_seconds,
                commit: commit_seconds,
                wait: wait_seconds,
                roundtrip: roundtrip_seconds,
            },
            values,
        ))
    })
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let upper = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[upper - 1] + sorted[upper]) * 0.5
    } else {
        sorted[upper]
    }
}

fn hash_f32(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};

    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

#[stark_test]
#[ignore]
fn kquant_mma_multi_token_projection_components() {
    let _activity = crate::engine::hybrid::activity::InferenceActivity::begin().unwrap();
    let cases = [
        (13u32, 8192usize, 4096usize, "q5k_main"),
        (13, 4096, 8192, "q5k_output"),
        (14, 8192, 4096, "q6k_main"),
        (14, 4096, 8192, "q6k_output"),
    ];
    for (kind, rows, columns, label) in cases {
        for tokens in [2usize, 4, 8, 14] {
            let shape = Shape {
                kind,
                rows,
                columns,
                tokens,
            };
            let (mma, geometry) = mma_pipeline(shape).unwrap();
            let serial = serial_pipeline(kind).unwrap();
            let block_bytes = crate::engine::hybrid::weights::block_layout(kind).unwrap().1;
            let matrix_bytes = rows * columns / 256 * block_bytes;
            assert!(matrix_bytes * BANKS > 160 * 1024 * 1024, "component population fits the excluded cache size");
            let code_payloads = (0..BANKS).map(|bank| finite_codes(kind, rows, columns, bank + 1)).collect::<Vec<_>>();
            assert!(code_payloads.iter().all(|payload| payload.len() == matrix_bytes));
            let input_payload = (0..tokens * columns)
                .map(|index| ((index % columns) as f32 * 0.017 + (index / columns) as f32 * 0.113).sin())
                .collect::<Vec<_>>();
            let executor = executor().unwrap();
            let codes = code_payloads.iter().map(|payload| buffer(executor, payload).unwrap()).collect::<Vec<_>>();
            let input = buffer(executor, bytemuck::cast_slice(&input_payload)).unwrap();
            let output = buffer(executor, bytemuck::cast_slice(&vec![0.0f32; BANKS * tokens * rows])).unwrap();
            let failure = buffer(executor, bytemuck::cast_slice(&[0u32])).unwrap();

            let (_, source) = dispatch(
                shape,
                Schedule::TokenMajor,
                &serial,
                &mma,
                geometry,
                &codes,
                matrix_bytes,
                0,
                &input,
                &output,
                &failure,
                true,
            )
            .unwrap();
            let source = source.unwrap();
            assert!(source.iter().all(|value| value.is_finite()));

            // Keep each pair's MMA measurements separate: pooling them would
            // hide a clock/system drift between the two ABBA comparisons.
            let mut timings = vec![Vec::<Timing>::new(); 4];
            let mut last_outputs: Vec<Option<Vec<f32>>> = vec![None; 4];
            let mut rotation = 1usize;
            for (pair_index, (left, right)) in
                [(Schedule::TokenMajor, Schedule::Mma), (Schedule::ProjectionMajor, Schedule::Mma)]
                    .into_iter()
                    .enumerate()
            {
                let pair = [left, right, right, left];
                // Three ABBA warmup rounds issue twelve untimed dispatches per pair.
                for _ in 0..3 {
                    for schedule in pair {
                        let _ = dispatch(
                            shape,
                            schedule,
                            &serial,
                            &mma,
                            geometry,
                            &codes,
                            matrix_bytes,
                            rotation,
                            &input,
                            &output,
                            &failure,
                            false,
                        )
                        .unwrap();
                        rotation = (rotation + 1) % BANKS;
                    }
                }
                for _ in 0..8 {
                    for schedule in pair {
                        let (timing, values) = dispatch(
                            shape,
                            schedule,
                            &serial,
                            &mma,
                            geometry,
                            &codes,
                            matrix_bytes,
                            rotation,
                            &input,
                            &output,
                            &failure,
                            true,
                        )
                        .unwrap();
                        let index = pair_index * 2 + usize::from(matches!(schedule, Schedule::Mma));
                        timings[index].push(timing);
                        last_outputs[index] = values;
                        rotation = (rotation + 1) % BANKS;
                    }
                }
            }

            for (index, schedule) in
                [Schedule::TokenMajor, Schedule::Mma, Schedule::ProjectionMajor, Schedule::Mma].into_iter().enumerate()
            {
                let actual = last_outputs[index].as_ref().unwrap();
                let bitwise_equal = actual.iter().zip(&source).all(|(a, b)| a.to_bits() == b.to_bits());
                let mut max_rrms = 0.0f64;
                let mut max_f64_error = 0.0f64;
                let mut max_absolute_error = 0.0f64;
                for bank in 0..BANKS {
                    let start = bank * tokens * rows;
                    let end = start + tokens * rows;
                    let (rrms, f64_error, absolute_error) = numerical_errors(
                        shape,
                        &code_payloads[bank],
                        &input_payload,
                        &actual[start..end],
                        &source[start..end],
                        SAMPLE_ROWS,
                    );
                    max_rrms = max_rrms.max(rrms);
                    max_f64_error = max_f64_error.max(f64_error);
                    max_absolute_error = max_absolute_error.max(absolute_error);
                }
                if matches!(schedule, Schedule::Mma) {
                    assert!(max_rrms <= 2e-5, "Kquant MMA relative RMS gate");
                    assert!(max_f64_error <= 2e-6, "Kquant MMA F64 gate");
                } else {
                    assert!(bitwise_equal, "serial schedule changed output bits");
                }
                let samples = &timings[index];
                let gpu = samples.iter().map(|sample| sample.gpu / BANKS as f64).collect::<Vec<_>>();
                let encode = samples.iter().map(|sample| sample.encode / BANKS as f64).collect::<Vec<_>>();
                let commit = samples.iter().map(|sample| sample.commit / BANKS as f64).collect::<Vec<_>>();
                let wait = samples.iter().map(|sample| sample.wait / BANKS as f64).collect::<Vec<_>>();
                let roundtrip = samples.iter().map(|sample| sample.roundtrip / BANKS as f64).collect::<Vec<_>>();
                let median_gpu = median(&gpu);
                let logical_weight_bytes = matrix_bytes
                    * if matches!(schedule, Schedule::Mma) {
                        1
                    } else {
                        tokens
                    };
                println!(
                    "{}",
                    serde_json::json!({
                        "component": label,
                        "kind": kind,
                        "rows": rows,
                        "columns": columns,
                        "tokens": tokens,
                        "banks": BANKS,
                        "weight_population_bytes": matrix_bytes * BANKS,
                        "schedule": schedule.label(),
                        "comparison": if index < 2 { "token_major_vs_mma" } else { "projection_major_vs_mma" },
                        "simdgroups_per_threadgroup": if matches!(schedule, Schedule::Mma) {geometry.nsg} else {2},
                        "threads_per_threadgroup": if matches!(schedule, Schedule::Mma) {geometry.nsg * 32} else {64},
                        "output_rows_per_threadgroup": if matches!(schedule, Schedule::Mma) {geometry.nt * 8} else if kind == 13 {2} else {4},
                        "token_tile_capacity": if matches!(schedule, Schedule::Mma) {geometry.rt * 8} else {1},
                        "shared_memory_bytes": if matches!(schedule, Schedule::Mma) {geometry.shared_bytes} else {0},
                        "measurement_dispatches": samples.len(),
                        "median_gpu_microseconds_per_matrix": median_gpu * 1e6,
                        "median_gpu_microseconds_per_token": median_gpu * 1e6 / tokens as f64,
                        "median_cpu_encode_microseconds_per_matrix": median(&encode) * 1e6,
                        "median_cpu_encode_microseconds_per_token": median(&encode) * 1e6 / tokens as f64,
                        "median_cpu_commit_microseconds_per_matrix": median(&commit) * 1e6,
                        "median_cpu_wait_microseconds_per_matrix": median(&wait) * 1e6,
                        "median_cpu_roundtrip_microseconds_per_matrix": median(&roundtrip) * 1e6,
                        "logical_weight_bytes_per_matrix": logical_weight_bytes,
                        "logical_weight_gigabytes_per_second_not_physical_dram_bandwidth": logical_weight_bytes as f64 / median_gpu / 1e9,
                        "maximum_relative_rms_vs_serial": max_rrms,
                        "maximum_f64_error_over_absolute_sum": max_f64_error,
                        "maximum_absolute_error_vs_serial": max_absolute_error,
                        "bitwise_equal_to_serial": bitwise_equal,
                        "output_sha256": hash_f32(actual),
                        "math_mode": "safe",
                        "timer_scope": "same_process_ABBA_command_create_encode_commit_wait_gpu_timestamps_upload_excluded",
                    })
                );
            }
        }
    }
}
