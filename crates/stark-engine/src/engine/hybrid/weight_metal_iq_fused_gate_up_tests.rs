// SPDX-License-Identifier: Apache-2.0
//! Safe IQ2 gate/up fusion and independently qualified schedule controls.
use stark_engine_macros::stark_test;

use super::*;

struct Pipelines {
    bound: PipelineState,
    base: PipelineState,
    fused: PipelineState,
    activation: PipelineState,
    fused_rows_per_simd: usize,
}
fn pipelines(
    gate_offset: usize,
    up_offset: usize,
) -> Result<Pipelines> {
    pipelines_for_rows(gate_offset, up_offset, 4)
}
fn pipelines_for_rows(
    gate_offset: usize,
    up_offset: usize,
    rows_per_simd: usize,
) -> Result<Pipelines> {
    if !gate_offset.is_multiple_of(2) || !up_offset.is_multiple_of(2) {
        return Err(Error::Invalid("IQ2 gate/up typed alignment"));
    }
    if !matches!(rows_per_simd, 2 | 4) {
        return Err(Error::Invalid("IQ2 fused diagnostic row schedule"));
    }
    let e = executor()?;
    let make = |name: &str, rows: u32| -> Result<PipelineState> {
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in [(0, 17u32), (4, 256), (5, 74), (71, rows)] {
            constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
        }
        let function = e
            .library
            .new_function_with_name_constant_values(name, &constants)
            .map_err(|_| Error::Invalid("IQ2 fused diagnostic Safe function"))?;
        e.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| Error::Invalid("IQ2 fused diagnostic Safe pipeline"))
    };
    Ok(Pipelines {
        bound: make("hybrid_graph_iq_input_bound", 4)?,
        base: make("hybrid_graph_iq_indexed_gpu_bound_factor_iq2xs", 4)?,
        fused: make(
            if rows_per_simd == 2 {
                "hybrid_graph_iq2_fused_gate_up_r2"
            } else {
                "hybrid_graph_iq2_fused_gate_up"
            },
            rows_per_simd as u32,
        )?,
        activation: make("hybrid_graph_swiglu", 4)?,
        fused_rows_per_simd: rows_per_simd,
    })
}
fn codes(
    rows: usize,
    columns: usize,
    seed: u32,
) -> Vec<u8> {
    let mut state = seed;
    let mut encoded = vec![0u8; rows * columns / 256 * 74];
    for b in &mut encoded {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    for block in encoded.chunks_exact_mut(74) {
        block[..2].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
    }
    encoded
}
struct Buffers {
    gate_codes: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_codes: Retained<ProtocolObject<dyn MTLBuffer>>,
    input: Retained<ProtocolObject<dyn MTLBuffer>>,
    indices: Retained<ProtocolObject<dyn MTLBuffer>>,
    bounds: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
    failure: Retained<ProtocolObject<dyn MTLBuffer>>,
}
fn buffers(
    gate: &[u8],
    up: &[u8],
    input: &[f32],
    indices: &[u32],
    rows: usize,
    gate_offset: usize,
    up_offset: usize,
) -> Buffers {
    let e = executor().unwrap();
    let mut a = vec![0x75u8; gate_offset];
    a.extend_from_slice(gate);
    let mut b = vec![0x75u8; up_offset];
    b.extend_from_slice(up);
    let output = vec![7.25f32; rows * indices.len() + 2];
    Buffers {
        gate_codes: buffer(e, &a).unwrap(),
        up_codes: buffer(e, &b).unwrap(),
        input: buffer(e, bytemuck::cast_slice(input)).unwrap(),
        indices: buffer(e, bytemuck::cast_slice(indices)).unwrap(),
        bounds: buffer(e, bytemuck::cast_slice(&vec![12345u32; indices.len() + 2])).unwrap(),
        gate: buffer(e, bytemuck::cast_slice(&output)).unwrap(),
        up: buffer(e, bytemuck::cast_slice(&output)).unwrap(),
        output: buffer(e, bytemuck::cast_slice(&output)).unwrap(),
        failure: buffer(e, bytemuck::cast_slice(&[0u32])).unwrap(),
    }
}
fn bound(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &Pipelines,
    b: &Buffers,
    columns: usize,
    batches: usize,
    input_offset: usize,
) {
    encoder.set_compute_pipeline_state(&pso.bound);
    encoder.set_buffer(Some(&b.input), input_offset, 0);
    encoder.set_buffer(Some(&b.bounds), 4, 1);
    encoder.set_buffer(Some(&b.failure), 0, 30);
    let p = [columns as u32, batches as u32];
    encoder.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 2);
    encoder.dispatch_threadgroups(
        MTLSize {
            width: batches,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);
}
#[expect(clippy::too_many_arguments, reason = "explicit independent bank/input/output views in diagnostic")]
fn encode(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &Pipelines,
    b: &Buffers,
    p: [u32; 5],
    selected: usize,
    fused: bool,
    gate_offset: usize,
    up_offset: usize,
    input_offset: usize,
    index_offset: usize,
    output_offset: usize,
) {
    let batches = if p[3] != 0 {
        selected
    } else {
        1
    };
    if fused {
        bound(encoder, pso, b, p[1] as usize, batches, input_offset);
        encoder.set_compute_pipeline_state(&pso.fused);
        for (index, (value, offset)) in [
            (&b.gate_codes, gate_offset),
            (&b.up_codes, up_offset),
            (&b.input, input_offset),
            (&b.output, output_offset),
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(Some(value), offset, index);
        }
        encoder.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 4);
        encoder.set_buffer(Some(&b.indices), index_offset, 5);
        encoder.set_buffer(Some(&b.bounds), 4, 6);
        encoder.set_buffer(Some(&b.gate), output_offset, 7);
        encoder.set_buffer(Some(&b.up), output_offset, 8);
        encoder.set_buffer(Some(&b.failure), 0, 30);
        encoder.dispatch_threadgroups(
            MTLSize {
                width: (p[0] as usize).div_ceil(2 * pso.fused_rows_per_simd),
                height: selected,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    } else {
        for (codes, offset, output) in [(&b.gate_codes, gate_offset, &b.gate), (&b.up_codes, up_offset, &b.up)] {
            // Preserve the current two-call path's fresh per-call bound scans.
            bound(encoder, pso, b, p[1] as usize, batches, input_offset);
            encoder.set_compute_pipeline_state(&pso.base);
            encoder.set_buffer(Some(codes), offset, 0);
            encoder.set_buffer(Some(&b.input), input_offset, 1);
            encoder.set_buffer(Some(output), output_offset, 2);
            encoder.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
            encoder.set_buffer(Some(&b.indices), index_offset, 6);
            encoder.set_buffer(Some(&b.bounds), 4, 7);
            encoder.set_buffer(Some(&b.failure), 0, 30);
            encoder.dispatch_threadgroups(
                MTLSize {
                    width: (p[0] as usize).div_ceil(8),
                    height: selected,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.set_compute_pipeline_state(&pso.activation);
        for (index, value) in [&b.gate, &b.up, &b.output].into_iter().enumerate() {
            encoder.set_buffer(Some(value), output_offset, index);
        }
        let activation = [p[0] * selected as u32, p[4]];
        encoder.set_bytes(NonNull::from(&activation).cast(), std::mem::size_of_val(&activation), 3);
        encoder.set_buffer(Some(&b.failure), 0, 30);
        encoder.dispatch_threads(
            MTLSize {
                width: activation[0] as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
}
struct Run {
    values: [Vec<f32>; 3],
    failure: u32,
    seconds: f64,
}
fn read(
    b: &Buffers,
    count: usize,
    seconds: f64,
) -> Run {
    let values = [&b.gate, &b.up, &b.output].map(|buffer| {
        let all = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), count + 2) };
        assert_eq!(all[0], 7.25, "IQ2 fusion prefix guard");
        assert_eq!(all[count + 1], 7.25, "IQ2 fusion tail guard");
        all[1..count + 1].to_vec()
    });
    let flags = unsafe { *b.failure.contents().as_ptr().cast::<u32>() };
    Run {
        values,
        failure: flags,
        seconds,
    }
}
#[expect(clippy::too_many_arguments, reason = "diagnostic keeps exact banks, shape and offsets explicit")]
fn run(
    pso: &Pipelines,
    gate: &[u8],
    up: &[u8],
    input: &[f32],
    indices: &[u32],
    p: [u32; 5],
    fused: bool,
    gate_offset: usize,
    up_offset: usize,
) -> Run {
    let b = buffers(gate, up, input, indices, p[0] as usize, gate_offset, up_offset);
    let e = executor().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let command = e.queue.command_buffer().unwrap();
        let encoder = command.compute_command_encoder().unwrap();
        encode(&encoder, pso, &b, p, indices.len(), fused, gate_offset, up_offset, 0, 0, 4);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        read(&b, p[0] as usize * indices.len(), command.gpu_end_time() - command.gpu_start_time())
    })
}
fn oracle(
    codes: &[u8],
    input: &[f32],
    indices: &[u32],
    p: [u32; 5],
) -> Vec<(f64, f64)> {
    let row_bytes = p[1] as usize / 256 * 74;
    let mut result = Vec::new();
    for (batch, &expert) in indices.iter().enumerate() {
        let offset = if p[3] != 0 {
            batch * p[1] as usize
        } else {
            0
        };
        for row in 0..p[0] as usize {
            let at = (expert as usize * p[0] as usize + row) * row_bytes;
            let mut sum = 0.0f64;
            let mut absolute = 0.0f64;
            for (block, x) in
                codes[at..at + row_bytes].chunks_exact(74).zip(input[offset..offset + p[1] as usize].chunks_exact(256))
            {
                let mut decoded = [0.0f32; 256];
                weights::decode_block(17, block, &mut decoded).unwrap();
                for (&weight, &value) in decoded.iter().zip(x) {
                    let value = f64::from(weight) * f64::from(value);
                    sum += value;
                    absolute += value.abs();
                }
            }
            result.push((sum, absolute));
        }
    }
    result
}
fn rrms(
    a: &[f32],
    b: &[f32],
) -> f64 {
    let error: f64 = a.iter().zip(b).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
    let power: f64 = b.iter().map(|&v| f64::from(v).powi(2)).sum();
    (error / power.max(f64::MIN_POSITIVE)).sqrt()
}
#[stark_test]
fn iq2_fused_gate_up_matches_f64_source_tails_offsets_and_fallbacks() {
    fusion_numeric_contract(4);
}
fn fusion_numeric_contract(rows_per_simd: usize) {
    for (a, b) in [(1usize, 0usize), (0, 1), (1, 1)] {
        assert!(pipelines(a, b).is_err());
    }
    let pso = pipelines_for_rows(2, 2, rows_per_simd).unwrap();
    let (experts, rows) = (4usize, 13usize);
    let indices = [3u32, 1, 3];
    for columns in [256usize, 2048, 4096] {
        let gate = codes(experts * rows, columns, 0x9135aced);
        let up = codes(experts * rows, columns, 0x874b19d3);
        for batched in [false, true] {
            for magnitude in [1.0f32, 1e20, 1e21] {
                let input = (0..columns
                    * if batched {
                        indices.len()
                    } else {
                        1
                    })
                    .map(|i| (i as f32 * 0.017).sin() * magnitude)
                    .collect::<Vec<_>>();
                let p = [rows as u32, columns as u32, experts as u32, u32::from(batched), 11f32.to_bits()];
                let reference = run(&pso, &gate, &up, &input, &indices, p, false, 2, 2);
                let actual = run(&pso, &gate, &up, &input, &indices, p, true, 2, 2);
                assert_eq!(reference.failure, 0);
                assert_eq!(actual.failure, 0);
                for (a, b) in actual.values.iter().zip(&reference.values) {
                    assert!(a.iter().all(|v| v.is_finite()));
                    assert!(rrms(a, b) <= 2e-5, "IQ2 fused source RMS");
                    if magnitude > 1e20 {
                        assert!(
                            a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "IQ2 fused exact extreme chronology"
                        );
                    }
                }
                for (bank, values) in [(&gate, &actual.values[0]), (&up, &actual.values[1])] {
                    let wide = oracle(bank, &input, &indices, p);
                    for (row, &(expected, absolute)) in wide.iter().enumerate() {
                        assert!(
                            (f64::from(values[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "IQ2 fused F64 gate"
                        );
                    }
                }
            }
        }
    }
    let p = [rows as u32, 256, experts as u32, 0, 11f32.to_bits()];
    let gate = codes(experts * rows, 256, 1);
    let up = codes(experts * rows, 256, 2);
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut input = vec![0.0; 256];
        input[255] = poison;
        let a = run(&pso, &gate, &up, &input, &indices, p, false, 2, 2);
        let b = run(&pso, &gate, &up, &input, &indices, p, true, 2, 2);
        assert_eq!(a.failure != 0, b.failure != 0, "IQ2 fused input failure parity");
    }
    let invalid = run(&pso, &gate, &up, &vec![0.0; 256], &[u32::MAX], p, true, 2, 2);
    assert_ne!(invalid.failure, 0, "IQ2 fused invalid expert escaped");
    let zero = vec![0u8; gate.len()];
    let maximum = vec![f32::MAX; 256];
    let a = run(&pso, &zero, &zero, &maximum, &indices, p, false, 2, 2);
    let b = run(&pso, &zero, &zero, &maximum, &indices, p, true, 2, 2);
    assert_eq!(a.failure, 0);
    assert_eq!(b.failure, 0);
    for (a, b) in a.values.iter().zip(&b.values) {
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()), "IQ2 fused zero times MAX contract");
    }
}
#[stark_test]
fn iq2_fused_gate_up_retains_masked_intermediate_failures_and_recovery() {
    fusion_failure_contract(4);
}
fn fusion_failure_contract(rows_per_simd: usize) {
    let pso = pipelines_for_rows(0, 0, rows_per_simd).unwrap();
    let p = [13u32, 256, 1, 0, 1f32.to_bits()];
    let finite = vec![0u8; 13 * 74];
    let mut ordinary = finite.clone();
    for block in ordinary.chunks_exact_mut(74) {
        block[..2].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
    }
    for bits in [0x7c00u16, 0x7e00] {
        let mut invalid = ordinary.clone();
        for block in invalid.chunks_exact_mut(74) {
            block[..2].copy_from_slice(&bits.to_le_bytes());
        }
        for (gate, up) in [(&invalid, &ordinary), (&ordinary, &invalid)] {
            for fused in [false, true] {
                let actual = run(&pso, gate, up, &vec![1.0; 256], &[0], p, fused, 0, 0);
                assert_ne!(actual.failure, 0, "IQ2 fused clipped projection failure was hidden");
                if bits == 0x7c00 {
                    assert!(
                        actual.values[2].iter().all(|v| v.is_finite()),
                        "masked infinity fixture did not exercise finite activation"
                    );
                }
            }
        }
    }
    let recovered = run(&pso, &ordinary, &ordinary, &vec![1.0; 256], &[0], p, true, 0, 0);
    assert_eq!(recovered.failure, 0, "IQ2 fused fresh-command recovery");
}
#[stark_test]
fn iq2_fused_gate_up_r2_preserves_existing_oracles_and_failure_contracts() {
    fusion_numeric_contract(2);
    fusion_failure_contract(2);
}
#[stark_test]
#[ignore]
fn iq2_fused_gate_up_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let (rows, columns, experts, selected) = (2048usize, 4096usize, 64usize, 8usize);
    let groups = experts / selected;
    let gate = codes(rows * experts, columns, 0x9135aced);
    let up = codes(rows * experts, columns, 0x874b19d3);
    let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let ids = (0..experts as u32).collect::<Vec<_>>();
    let mut wide = Vec::new();
    let row_bytes = columns / 256 * 74;
    for expert in 0..experts {
        for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
            let at = (expert * rows + row) * row_bytes;
            let p = [1, columns as u32, 1, 0, 11f32.to_bits()];
            let g = oracle(&gate[at..at + row_bytes], &input, &[0], p)[0];
            let u = oracle(&up[at..at + row_bytes], &input, &[0], p)[0];
            wide.push((expert * rows + row, g, u));
        }
    }
    let pso = pipelines(0, 0).unwrap();
    let b = buffers(&gate, &up, &input, &ids, rows, 0, 0);
    let p = [rows as u32, columns as u32, experts as u32, 0, 11f32.to_bits()];
    let mut samples = [Vec::new(), Vec::new()];
    let mut reference = None;
    let mut errors = [0.0f64; 3];
    let mut f64_errors = [0.0f64; 2];
    let mut bits = [true; 3];
    for replay in 0..7 {
        for variant in if replay % 2 == 0 {
            [0usize, 1, 1, 0]
        } else {
            [1usize, 0, 0, 1]
        } {
            let result = objc2::rc::autoreleasepool(|_| {
                let e = executor().unwrap();
                let command = e.queue.command_buffer().unwrap();
                let encoder = command.compute_command_encoder().unwrap();
                for at in 0..groups {
                    let group = (at + replay) % groups;
                    encode(
                        &encoder,
                        &pso,
                        &b,
                        p,
                        selected,
                        variant == 1,
                        0,
                        0,
                        0,
                        group * selected * 4,
                        4 + group * selected * rows * 4,
                    );
                }
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                read(&b, rows * experts, (command.gpu_end_time() - command.gpu_start_time()) / groups as f64)
            });
            assert_eq!(result.failure, 0, "IQ2 fused component finite failure");
            for &(at, g, u) in &wide {
                for (i, (expected, absolute)) in [g, u].into_iter().enumerate() {
                    let error = (f64::from(result.values[i][at]) - expected).abs() / absolute.max(1.0);
                    f64_errors[i] = f64_errors[i].max(error);
                    assert!(error <= 2e-6, "IQ2 fused component F64 gate");
                }
            }

            if variant == 0 {
                reference = Some(result.values.clone());
            }
            if let Some(reference) = &reference {
                for i in 0..3 {
                    errors[i] = errors[i].max(rrms(&result.values[i], &reference[i]));
                    bits[i] &= result.values[i].iter().zip(&reference[i]).all(|(a, b)| a.to_bits() == b.to_bits());
                }
            }
            if replay > 0 {
                samples[variant].push(result.seconds);
            }
        }
    }
    for (variant, times) in samples.iter_mut().enumerate() {
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"component":"iq2_fused_gate_up_activation","fused":variant==1,
            "math_mode":"safe","activity":true,"fresh_gpu_bound":true,"bound_dispatches_per_group":if variant==0{2}else{1},
            "rows":rows,"columns":columns,"experts":experts,"selected_experts":selected,"rotating_groups":groups,
            "combined_population_bytes":gate.len()+up.len(),"threads_per_group":64,"rows_per_simd":4,
            "warm_samples":times.len(),"paired_abba":true,"intermediate_diagnostics_written":true,
            "median_gpu_microseconds_per_8_experts":(times[5]+times[6])*0.5e6,
            "minimum_gpu_microseconds_per_8_experts":times[0]*1e6,"maximum_gpu_microseconds_per_8_experts":times[times.len()-1]*1e6,
            "relative_rms_errors_gate_up_activation":errors,"maximum_f64_error_over_absolute_sum_gate_up":f64_errors,"f64_rows_per_bank":wide.len(),"bitwise_gate_up_activation":bits,
            "baseline_scope":"two_current_typed_signed_factor_gemvs_plus_existing_swiglu"})
        );
        assert!(errors.iter().all(|v| *v <= 2e-5), "IQ2 fused component source RMS");
    }
}

/// A synthetic working-set diagnostic, not an integrated-model qualification.
/// E288 matches the trunk banks; each command visits all experts once with 36
/// distinct shared inputs. The scattered permutation separates adjacent IDs,
/// retaining all 288 unique experts and exactly the same selected byte count.
/// One 1.397 GB bank pair remains smaller than the model's 41 separate pairs.
#[stark_test]
#[ignore]
fn iq2_fused_gate_up_e288_scattered_r2_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let (rows, columns, experts, selected) = (2048usize, 4096usize, 288usize, 8usize);
    let groups = experts / selected;
    let gate = codes(rows * experts, columns, 0x9135aced);
    let up = codes(rows * experts, columns, 0x874b19d3);
    let input = (0..groups * columns)
        .map(|i| ((i % columns) as f32 * 0.017 + (i / columns) as f32 * 0.13).sin())
        .collect::<Vec<_>>();
    let r4 = pipelines_for_rows(0, 0, 4).unwrap();
    let r2 = pipelines_for_rows(0, 0, 2).unwrap();
    let p = [rows as u32, columns as u32, experts as u32, 0, 11f32.to_bits()];
    for scattered in [false, true] {
        let ids = (0..experts)
            .map(|i| {
                if scattered {
                    ((i * 37 + 11) % experts) as u32
                } else {
                    i as u32
                }
            })
            .collect::<Vec<_>>();
        let unique = ids.iter().copied().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), experts, "synthetic component must cover all E288 experts");
        let mut wide = Vec::new();
        let row_bytes = columns / 256 * 74;
        for group in 0..groups {
            let expert = ids[group * selected] as usize;
            for row in [0, rows - 1] {
                let at = (expert * rows + row) * row_bytes;
                let narrow = [1, columns as u32, 1, 0, 11f32.to_bits()];
                let x = &input[group * columns..(group + 1) * columns];
                let g = oracle(&gate[at..at + row_bytes], x, &[0], narrow)[0];
                let u = oracle(&up[at..at + row_bytes], x, &[0], narrow)[0];
                wide.push((group * selected * rows + row, g, u));
            }
        }
        let b = buffers(&gate, &up, &input, &ids, rows, 0, 0);
        for (candidate, pso) in [("fused_r4", &r4), ("fused_r2", &r2)] {
            let mut samples = [Vec::new(), Vec::new()];
            let mut reference: Option<[Vec<f32>; 3]> = None;
            let mut errors = [0.0f64; 3];
            let mut f64_errors = [0.0f64; 2];
            let mut bits = [true; 3];
            for replay in 0..7 {
                for variant in if replay % 2 == 0 {
                    [0usize, 1, 1, 0]
                } else {
                    [1usize, 0, 0, 1]
                } {
                    let result = objc2::rc::autoreleasepool(|_| {
                        let e = executor().unwrap();
                        let command = e.queue.command_buffer().unwrap();
                        let encoder = command.compute_command_encoder().unwrap();
                        for at in 0..groups {
                            let group = (at + replay) % groups;
                            encode(
                                &encoder,
                                if variant == 0 {
                                    &r4
                                } else {
                                    pso
                                },
                                &b,
                                p,
                                selected,
                                variant == 1,
                                0,
                                0,
                                group * columns * 4,
                                group * selected * 4,
                                4 + group * selected * rows * 4,
                            );
                        }
                        encoder.end_encoding();
                        command.commit();
                        command.wait_until_completed();
                        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                        read(&b, rows * experts, (command.gpu_end_time() - command.gpu_start_time()) / groups as f64)
                    });
                    assert_eq!(result.failure, 0, "E288 fused component finite failure");
                    for &(at, g, u) in &wide {
                        for (i, (expected, absolute)) in [g, u].into_iter().enumerate() {
                            let error = (f64::from(result.values[i][at]) - expected).abs() / absolute.max(1.0);
                            f64_errors[i] = f64_errors[i].max(error);
                            assert!(error <= 2e-6, "E288 fused component F64 gate");
                        }
                    }
                    if variant == 0 {
                        reference = Some(result.values.clone());
                    }
                    if let Some(reference) = &reference {
                        for i in 0..3 {
                            errors[i] = errors[i].max(rrms(&result.values[i], &reference[i]));
                            bits[i] &=
                                result.values[i].iter().zip(&reference[i]).all(|(a, b)| a.to_bits() == b.to_bits());
                        }
                    }
                    if replay > 0 {
                        samples[variant].push(result.seconds);
                    }
                }
            }
            for (variant, times) in samples.iter_mut().enumerate() {
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({
                        "component":"iq2_fused_gate_up_e288", "candidate":candidate, "fused":variant==1,
                        "math_mode":"safe", "activity":true, "serial_encoder":true, "fresh_gpu_bound":true,
                        "rows":rows, "columns":columns, "experts":experts, "selected_experts":selected,
                        "id_layout":if scattered { "affine_permutation_37x11_mod288" } else { "contiguous" },
                        "rotating_groups":groups, "distinct_shared_inputs":groups, "bank_pairs":1,
                        "combined_population_bytes":gate.len()+up.len(), "selected_pair_bytes":gate.len()/groups+up.len()/groups,
                        "threads_per_group":64, "rows_per_simd":if variant==0 { 4 } else { pso.fused_rows_per_simd },
                        "warm_samples":times.len(), "paired_abba":true, "intermediate_diagnostics_written":true,
                        "median_gpu_microseconds_per_8_experts":(times[5]+times[6])*0.5e6,
                        "minimum_gpu_microseconds_per_8_experts":times[0]*1e6,
                        "maximum_gpu_microseconds_per_8_experts":times[times.len()-1]*1e6,
                        "relative_rms_errors_gate_up_activation":errors,
                        "maximum_f64_error_over_absolute_sum_gate_up":f64_errors,
                        "f64_rows_per_bank":wide.len(), "bitwise_gate_up_activation":bits,
                        "assumptions":"synthetic codes and scales; one E288 pair, not 41 model-layer pairs; component only",
                        "baseline_scope":"two_current_typed_signed_factor_gemvs_plus_existing_swiglu"
                    })
                );
                assert!(errors.iter().all(|v| *v <= 2e-5), "E288 fused component source RMS");
            }
        }
    }
}
