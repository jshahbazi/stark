// SPDX-License-Identifier: Apache-2.0
//! Safe KDA norm reference/probe and opt-in runtime admission checks.
use stark_engine_macros::stark_test;

use super::*;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn pipeline(parallel: bool) -> Pipeline {
    let source = format!(
        "#define STARK_GLM_KDA_TRANSPOSED 1\n#define STARK_GLM_KDA_FUSED_STATE 1\n#define STARK_GLM_KDA_PARALLEL_NORM {}\n{}",
        usize::from(parallel),
        include_str!("glm_metal.metal")
    );
    let e = executor().unwrap();
    let options = metal::MTLCompileOptions::new();
    options.set_math_mode(metal::MTLMathMode::Safe);
    let library = e
        .device
        .new_library_with_source(&source, Some(&options))
        .unwrap_or_else(|_| panic!("KDA norm trial library compilation failed"));
    let function = library.new_function_with_name("glm_kda_decode").expect("KDA norm trial function");
    e.device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|_| panic!("KDA norm trial pipeline creation failed"))
}

fn create(
    heads: usize,
    seed: usize,
) -> Kda {
    let (dim, kernel) = (128usize, 4usize);
    let inner = heads * dim;
    let values = |n: usize, offset: usize| {
        (0..n).map(|i| (((i * 13 + offset * 7 + seed * 11) % 43) as f32 - 21.0) * 0.013).collect::<Vec<_>>()
    };
    let convolution = [values(inner * kernel, 1), values(inner * kernel, 2), values(inner * kernel, 3)];
    let mut result = Kda::new(
        Geometry {
            heads,
            dim,
            kernel,
            rms_eps: 1e-5,
            lower_bound: -5.0,
        },
        [&convolution[0], &convolution[1], &convolution[2]],
        &values(inner, 4),
        &vec![1.0; dim],
        &vec![-1.25; heads],
    )
    .unwrap();
    result.transposed_state = true;
    result
}

fn encode(
    state: &Kda,
    pipeline: &Pipeline,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
) {
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
}

fn stage(
    state: &Kda,
    step: usize,
) {
    let values = (0..5 * state.inner + state.heads)
        .map(|i| (((i * 17 + step * 11) % 47) as f32 - 23.0) * 0.011)
        .collect::<Vec<_>>();
    // No submitted command uses this independently owned staging allocation.
    unsafe {
        state.input.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(values.as_ptr(), values.len());
    }
}

fn run(
    state: &Kda,
    pipeline: &Pipeline,
) -> Result<Vec<f32>> {
    objc2::rc::autoreleasepool(|_| {
        let e = executor()?;
        let command = e.queue.command_buffer().ok_or(Error::Invalid("KDA norm trial command"))?;
        let encoder = command.compute_command_encoder().ok_or(Error::Invalid("KDA norm trial encoder"))?;
        encode(state, pipeline, &encoder);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("KDA norm trial execution"));
        }
        state.check_completed()?;
        let output =
            unsafe { std::slice::from_raw_parts(state.output.contents().as_ptr().cast::<f32>(), state.inner) }.to_vec();
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("KDA norm trial output"));
        }
        Ok(output)
    })
}

fn relative_rms(
    actual: &[f32],
    expected: &[f32],
) -> f64 {
    let error: f64 = actual.iter().zip(expected).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
    let power: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
    (error / power.max(f64::MIN_POSITIVE)).sqrt()
}

#[stark_test]
fn kda_parallel_norm_multistep_preserves_source_tolerance_and_sticky_failures() {
    let psos = [pipeline(false), pipeline(true)];
    let mut states = [create(2, 0), create(2, 0)];
    for _ in 0..2 {
        for state in &mut states {
            state.reset();
        }
        for step in 0..33 {
            for state in &states {
                stage(state, step);
            }
            let reference = run(&states[0], &psos[0]).unwrap();
            let candidate = run(&states[1], &psos[1]).unwrap();
            assert!(relative_rms(&candidate, &reference) <= 2e-5, "KDA parallel norm output source RMS");
            let reference = unsafe {
                std::slice::from_raw_parts(states[0].state.contents().as_ptr().cast::<f32>(), states[0].state_values)
            };
            let candidate = unsafe {
                std::slice::from_raw_parts(states[1].state.contents().as_ptr().cast::<f32>(), states[1].state_values)
            };
            assert!(relative_rms(candidate, reference) <= 2e-5, "KDA parallel norm recurrent source RMS");
        }
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [0, 3 * states[0].inner, 4 * states[0].inner, 5 * states[0].inner] {
            for (variant, state) in states.iter_mut().enumerate() {
                state.reset();
                stage(state, 0);
                unsafe {
                    state.input.contents().as_ptr().cast::<f32>().add(index).write(poison);
                }
                assert!(run(state, &psos[variant]).is_err(), "KDA norm trial masked nonfinite control");
                // The unchanged status remains sticky across another step.
                stage(state, 1);
                assert!(run(state, &psos[variant]).is_err(), "KDA norm trial cleared sticky failure");
                state.reset();
                stage(state, 1);
                assert!(run(state, &psos[variant]).is_ok(), "KDA norm trial reset did not recover");
            }
        }
    }
}

fn norm_probe(
    input: &[f32],
    rms: bool,
    parallel: bool,
) -> Result<Vec<f32>> {
    static PROBE: OnceLock<Pipeline> = OnceLock::new();
    let e = executor()?;
    let pso = PROBE.get_or_init(|| {
        let source = format!("#define STARK_GLM_KDA_PARALLEL_NORM 1\n{}", include_str!("glm_metal.metal"));
        let options = metal::MTLCompileOptions::new();
        options.set_math_mode(metal::MTLMathMode::Safe);
        let library = e
            .device
            .new_library_with_source(&source, Some(&options))
            .unwrap_or_else(|_| panic!("KDA norm probe library compilation failed"));
        let function = library.new_function_with_name("glm_kda_parallel_norm_probe").expect("KDA norm probe function");
        e.device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|_| panic!("KDA norm probe pipeline creation failed"))
    });
    let source = e
        .device
        .new_buffer_with_data(bytemuck::cast_slice(input), MTLResourceOptions::STORAGE_MODE_SHARED)
        .ok_or(Error::Invalid("KDA norm probe input"))?;
    let output = DecodeGraph::zeros(input.len())?;
    let failure = DecodeGraph::zeros(1)?;
    let parameters = [
        input.len() as u32,
        u32::from(rms),
        u32::from(parallel),
        if rms {
            1e-5f32
        } else {
            1e-6f32
        }
        .to_bits(),
    ];
    objc2::rc::autoreleasepool(|_| {
        let command = e.queue.command_buffer().ok_or(Error::Invalid("KDA norm probe command"))?;
        let encoder = command.compute_command_encoder().ok_or(Error::Invalid("KDA norm probe encoder"))?;
        encoder.set_compute_pipeline_state(&pso);
        encoder.set_buffer(Some(&source), 0, 0);
        encoder.set_buffer(Some(&output.buffer), output.offset, 1);
        encoder.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 2);
        encoder.set_buffer(Some(&failure.buffer), failure.offset, 30);
        encoder.dispatch_threadgroups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("KDA norm probe execution"));
        }
        if unsafe { *failure.buffer.contents().as_ptr().cast::<u32>() } != 0 {
            return Err(Error::Invalid("KDA norm probe finite failure"));
        }
        Ok(unsafe { std::slice::from_raw_parts(output.buffer.contents().as_ptr().cast::<f32>(), input.len()) }.to_vec())
    })
}

#[stark_test]
fn kda_parallel_norm_matches_f64_and_exact_ordered_extreme_fallback() {
    for width in [7usize, 32, 128, 129, 256] {
        for rms in [false, true] {
            for magnitude in [0.0f32, 1.0, 1e15] {
                let values = (0..width).map(|i| ((i % 43) as f32 - 21.0) * 0.013 * magnitude).collect::<Vec<_>>();
                let actual = norm_probe(&values, rms, true).unwrap();
                let sum: f64 = values.iter().map(|&v| f64::from(v).powi(2)).sum();
                let epsilon = if rms {
                    1e-5
                } else {
                    1e-6
                };
                let denominator = if rms {
                    (sum / width as f64 + epsilon).sqrt()
                } else {
                    (sum + epsilon).sqrt()
                };
                for (&a, &v) in actual.iter().zip(&values) {
                    let expected = f64::from(v) / denominator;
                    assert!((f64::from(a) - expected).abs() <= 2e-6 * expected.abs().max(1.0), "KDA norm F64 gate");
                }
            }
            for magnitude in [1e16f32, 1e30, f32::MAX] {
                let values = vec![magnitude; width];
                let reference = norm_probe(&values, rms, false).unwrap();
                let actual = norm_probe(&values, rms, true).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "KDA norm ordered extreme fallback bits"
                );
            }
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut values = vec![0.0; width];
                values[width - 1] = poison;
                assert!(norm_probe(&values, rms, true).is_err(), "KDA norm nonfinite probe escaped");
            }
        }
    }
}

#[stark_test]
fn kda_parallel_norm_large_output_keeps_original_kernel_bits() {
    let psos = [pipeline(false), pipeline(true)];
    let inner = 128usize;
    let convolution = [0.0f32, 1.0].repeat(inner);
    let create = || {
        Kda::new(
            Geometry {
                heads: 1,
                dim: 128,
                kernel: 2,
                rms_eps: 1e-5,
                lower_bound: -5.0,
            },
            [&convolution; 3],
            &vec![0.0; inner],
            &vec![1.0; 128],
            &[-1.0],
        )
        .unwrap()
    };
    let mut states = [create(), create()];
    for magnitude in [1e17f32, 1e30, f32::MAX] {
        let mut outputs = Vec::new();
        for (variant, state) in states.iter_mut().enumerate() {
            state.reset();
            let mut input = vec![0.0f32; 5 * inner + 1];
            input[0] = 1.0;
            input[inner] = 1.0;
            input[2 * inner..3 * inner].fill(magnitude);
            unsafe {
                state.input.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(input.as_ptr(), input.len());
            }
            outputs.push(run(state, &psos[variant]).unwrap());
        }
        assert!(
            outputs[0].iter().zip(&outputs[1]).all(|(a, b)| a.to_bits() == b.to_bits()),
            "KDA large-output norm changed reference kernel bits"
        );
        let a = unsafe {
            std::slice::from_raw_parts(states[0].state.contents().as_ptr().cast::<f32>(), states[0].state_values)
        };
        let b = unsafe {
            std::slice::from_raw_parts(states[1].state.contents().as_ptr().cast::<f32>(), states[1].state_values)
        };
        assert!(
            a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "KDA large-output fallback changed state bits"
        );
    }
}

#[stark_test]
#[ignore]
fn kda_parallel_norm_serial_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let psos = [pipeline(false), pipeline(true)];
    let states =
        [(0..34).map(|i| create(64, i)).collect::<Vec<_>>(), (0..34).map(|i| create(64, i)).collect::<Vec<_>>()];
    for list in &states {
        for state in list {
            stage(state, 0);
        }
    }
    let mut samples = [Vec::new(), Vec::new()];
    for replay in 0..9 {
        for variant in if replay % 2 == 0 {
            [0usize, 1, 1, 0]
        } else {
            [1usize, 0, 0, 1]
        } {
            let seconds = objc2::rc::autoreleasepool(|_| {
                let e = executor().unwrap();
                let command = e.queue.command_buffer().unwrap();
                let encoder = command.compute_command_encoder().unwrap();
                for state in &states[variant] {
                    encode(state, &psos[variant], &encoder);
                }
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                for state in &states[variant] {
                    state.check_completed().unwrap();
                }
                command.gpu_end_time() - command.gpu_start_time()
            });
            if replay > 0 {
                samples[variant].push(seconds / 34.0);
            }
        }
    }
    for (variant, times) in samples.iter_mut().enumerate() {
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"component":"kda_parallel_norm_recurrence",
            "parallel_norm":variant==1,"math_mode":"safe","activity":true,"serial_encoder":true,
            "transposed_state":true,"fused_state":true,"heads":64,"dim":128,"layers_per_command":34,
            "warm_samples":times.len(),"paired_abba":true,
            "median_gpu_microseconds_per_layer":(times[7]+times[8])*0.5e6,
            "minimum_gpu_microseconds_per_layer":times[0]*1e6,
            "maximum_gpu_microseconds_per_layer":times[times.len()-1]*1e6})
        );
    }
}

#[stark_test]
fn kda_parallel_norm_runtime_admission_requires_transpose_fused_and_dim128() {
    let executor = executor().unwrap();
    let state = create(2, 0);
    let configured = std::env::var_os("STARK_GLM53_KDA_PARALLEL_NORM").is_some_and(|v| v == "1")
        && std::env::var_os("STARK_GLM53_KDA_FUSED_STATE").is_some_and(|v| v == "1");
    let cached = executor.parallel_norm_pipeline.is_some();
    assert!(!cached || configured, "KDA norm PSO escaped opt-in/fused admission");
    assert_eq!(state.parallel_norm_admitted(executor), cached);
    let mut row_major = create(2, 0);
    row_major.transposed_state = false;
    assert!(!row_major.parallel_norm_admitted(executor), "KDA row-major reference admitted parallel norm");
    let tiny = Kda::new(
        Geometry {
            heads: 2,
            dim: 7,
            kernel: 2,
            rms_eps: 1e-5,
            lower_bound: -5.0,
        },
        [&[0.0; 28]; 3],
        &[0.0; 14],
        &[1.0; 7],
        &[-1.0; 2],
    )
    .unwrap();
    assert!(!tiny.parallel_norm_admitted(executor), "unqualified KDA dimensions admitted parallel norm");
}

#[stark_test]
#[ignore]
fn kda_parallel_norm_runtime_graph_matches_selected_safe_schedule() {
    let executor = executor().unwrap();
    let reference_pso = pipeline(true);
    let mut reference = create(2, 0);
    let mut actual = create(2, 0);
    actual.transposed_state = std::env::var_os("STARK_GLM53_KDA_TRANSPOSED").is_some_and(|v| v == "1");
    assert!(
        actual.parallel_norm_admitted(executor),
        "fresh process requires KDA_PARALLEL_NORM1 KDA_FUSED_STATE1 KDA_TRANSPOSED1"
    );
    assert!(executor.parallel_norm_pipeline.is_some(), "parallel norm Safe cache missing");
    for _ in 0..2 {
        reference.reset();
        actual.reset();
        for step in 0..17 {
            stage(&reference, step);
            stage(&actual, step);
            let expected = run(&reference, &reference_pso).unwrap();
            let mut graph = DecodeGraph::new().unwrap();
            actual.encode_graph(&mut graph).unwrap();
            let output = DeviceVector {
                buffer: actual.output.clone(),
                offset: 0,
                len: actual.inner,
            };
            let values = graph.finish(&[output]).unwrap().remove(0);
            actual.check_completed().unwrap();
            assert!(
                values.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                "runtime KDA parallel norm diverged from selected Safe probe"
            );
            let a = unsafe {
                std::slice::from_raw_parts(actual.state.contents().as_ptr().cast::<f32>(), actual.state_values)
            };
            let b = unsafe {
                std::slice::from_raw_parts(reference.state.contents().as_ptr().cast::<f32>(), reference.state_values)
            };
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "runtime KDA parallel norm state/cache mismatch"
            );
        }
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        actual.reset();
        stage(&actual, 0);
        unsafe {
            actual.input.contents().as_ptr().cast::<f32>().add(4 * actual.inner).write(poison);
        }
        let mut graph = DecodeGraph::new().unwrap();
        actual.encode_graph(&mut graph).unwrap();
        let output = DeviceVector {
            buffer: actual.output.clone(),
            offset: 0,
            len: actual.inner,
        };
        assert!(graph.finish(&[output]).is_err(), "runtime KDA parallel norm masked control failure");
        actual.reset();
        stage(&actual, 0);
        assert!(run(&actual, &reference_pso).is_ok(), "runtime KDA parallel norm reset recovery");
    }
}
