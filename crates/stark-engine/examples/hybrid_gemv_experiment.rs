//! Synthetic component experiment; prints aggregate timings/errors only.

#[cfg(feature = "metal")]
mod experiment {
    use std::{ptr::NonNull, time::Instant};

    use metal::{
        MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder,
        MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLDevice, MTLDeviceExt,
        MTLDispatchType, MTLFunctionConstantValues, MTLLibraryExt, MTLResourceOptions, MTLSize,
    };
    use objc2::{rc::Retained, runtime::ProtocolObject};

    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

    fn table(
        name: &str,
        metal_name: &str,
        dtype: &str,
        suffix: &str,
    ) -> Result<String, &'static str> {
        let rust = include_str!("../src/engine/hybrid/weights.rs");
        let start = rust.find(&format!("const {name}: ")).ok_or("source table missing")?;
        let rest = rust[start..].split_once("= [").ok_or("source table malformed")?.1;
        let contents = rest.split_once("]; ").or_else(|| rest.split_once("];\n")).ok_or("source table end missing")?.0;
        let values: Vec<_> = contents.split(',').map(str::trim).filter(|v| !v.is_empty()).collect();
        Ok(format!(
            "constant {dtype} {metal_name}[{}]={{ {} }};\n",
            values.len(),
            values.iter().map(|v| format!("{v}{suffix}")).collect::<Vec<_>>().join(",")
        ))
    }

    fn source() -> Result<String, &'static str> {
        let mut result = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
        result.push_str(&table("IQ2XXS_GRID", "iq2xxs_grid", "ulong", "ul")?);
        result.push_str(&table("IQ2XS_GRID", "iq2xs_grid", "ulong", "ul")?);
        result.push_str(&table("IQ3XXS_GRID", "iq3xxs_grid", "uint", "u")?);
        result.push_str(&table("KSIGNS_IQ2XS", "ksigns", "uchar", "")?);
        result.push_str(&table("KVALUES_IQ4NL", "iq4values", "short", "")?);
        result.push_str(include_str!("../src/engine/hybrid/packed_matvec.metal"));
        result.push_str(include_str!("../unit/fixtures/hybrid_affine_simd_experiment.metal"));
        result.push_str(include_str!("../unit/fixtures/hybrid_iq_simd_experiment.metal"));
        result.push_str(include_str!("../unit/fixtures/hybrid_generic_simd_experiment.metal"));
        result.push_str(include_str!("../unit/fixtures/hybrid_blocklane_experiment.metal"));
        result.push_str("kernel void copy_result(device const float*x[[buffer(0)]],device float*y[[buffer(1)]],constant uint&n[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=x[i];}\n");
        Ok(result)
    }

    fn new_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: &[u8],
    ) -> Result<Buffer, &'static str> {
        device.new_buffer_with_data(bytes, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or("allocation failed")
    }

    fn pseudo(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed
    }

    pub fn run() -> Result<(), &'static str> {
        let concurrent = std::env::args().any(|arg| arg == "--concurrent");
        let packed = std::env::args().any(|arg| arg == "--packed" || arg == "--blocklane");
        let blocklane = std::env::args().any(|arg| arg == "--blocklane");
        let device = <dyn MTLDevice>::system_default().ok_or("device missing")?;
        let queue = device.new_command_queue().ok_or("queue missing")?;
        let options = metal::MTLCompileOptions::new();
        options.set_math_mode(metal::MTLMathMode::Safe);
        let library =
            device.new_library_with_source(&source()?, Some(&options)).map_err(|_| "shader compilation failed")?;
        let copy = library.new_function_with_name("copy_result").ok_or("copy kernel missing")?;
        let copy = device.new_compute_pipeline_state_with_function(&copy).map_err(|_| "copy pipeline failed")?;
        let affine_cases = [
            ("qwen_gate4", 640usize, 2560usize, 4u32, 0u32),
            ("qwen_down4", 2560, 640, 4, 0),
            ("qwen_gate8", 640, 2560, 8, 0),
            ("qwen_down8", 2560, 640, 8, 0),
            ("glm_gate_iq2xs", 2048, 4096, 0, 17),
            ("glm_down_iq3xxs", 4096, 2048, 0, 18),
        ];
        let packed_cases = [
            ("glm_ssm_q_q5k", 8192usize, 4096usize, 0u32, 13u32),
            ("glm_ssm_kv_q6k", 8192, 4096, 0, 14),
            ("glm_ssm_out_q5k", 4096, 8192, 0, 13),
            ("glm_shared_gate_q5k", 2048, 4096, 0, 13),
            ("glm_shared_down_q6k", 4096, 2048, 0, 14),
            ("glm_mla_qb_q8", 16384, 1536, 0, 8),
            ("glm_ssm_fb_q8", 8192, 128, 0, 8),
            ("glm_ssm_fa_q8", 128, 4096, 0, 8),
            ("glm_router_f32", 288, 4096, 0, 0),
            ("dense_f16", 2048, 4096, 0, 1),
            ("dense_bf16", 2048, 4096, 0, 30),
            ("glm_readout_q4k", 154880, 4096, 0, 12),
            ("generic_q2k", 2048, 4096, 0, 10),
            ("generic_q3k", 2048, 4096, 0, 11),
            ("generic_iq2xxs", 2048, 4096, 0, 16),
            ("generic_iq4xs", 2048, 4096, 0, 23),
        ];
        let cases = if packed {
            packed_cases.as_slice()
        } else {
            affine_cases.as_slice()
        };
        for &(label, rows, cols, bits, kind) in cases {
            if blocklane && !matches!(kind, 8 | 13 | 14) {
                continue;
            }
            let affine = bits != 0;
            let (block_size, block_bytes) = match kind {
                0 => (1usize, 4usize),
                1 | 30 => (1, 2),
                8 => (32, 34),
                10 => (256, 84),
                11 => (256, 110),
                12 => (256, 144),
                13 => (256, 176),
                14 => (256, 210),
                16 => (256, 66),
                17 => (256, 74),
                18 => (256, 98),
                23 => (256, 136),
                _ => return Err("invalid packed format"),
            };
            let code_bytes = if affine {
                rows * cols * bits as usize / 8
            } else {
                rows * cols / block_size * block_bytes
            };
            let banks = if packed {
                (128 * 1024 * 1024 / code_bytes).clamp(1, 64)
            } else {
                64
            };
            let plane_bytes = if affine {
                rows * cols / 64 * 2
            } else {
                2
            };
            let mut seed = 4123u32;
            let mut codes = vec![0u8; code_bytes * banks];
            for byte in &mut codes {
                *byte = (pseudo(&mut seed) >> 24) as u8;
            }
            if !affine && kind == 0 {
                for block in codes.as_chunks_mut::<4>().0.iter_mut() {
                    let value = i32::from_le_bytes(*block) as f32 / i32::MAX as f32 * 0.125;
                    *block = value.to_le_bytes();
                }
            } else if !affine && matches!(kind, 1 | 30) {
                for block in codes.as_chunks_mut::<2>().0.iter_mut() {
                    let value = i16::from_le_bytes(*block) as f32 / i16::MAX as f32 * 0.125;
                    *block = if kind == 1 {
                        half::f16::from_f32(value).to_bits().to_le_bytes()
                    } else {
                        half::bf16::from_f32(value).to_bits().to_le_bytes()
                    };
                }
            } else if !affine {
                let offsets: &[usize] = match kind {
                    8 | 16 | 17 | 18 | 23 => &[0],
                    10 => &[80, 82],
                    11 => &[108],
                    12 | 13 => &[0, 2],
                    14 => &[208],
                    _ => return Err("invalid packed scale"),
                };
                for block in codes.chunks_exact_mut(block_bytes) {
                    for &offset in offsets {
                        block[offset..offset + 2]
                            .copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
                    }
                }
            }
            let mut scales = vec![0u8; plane_bytes * banks];
            let mut biases = vec![0u8; plane_bytes * banks];
            for (scale, bias) in scales.as_chunks_mut::<2>().0.iter_mut().zip(biases.as_chunks_mut::<2>().0.iter_mut())
            {
                let s = 0.00390625 * ((pseudo(&mut seed) % 8 + 1) as f32);
                scale.copy_from_slice(&half::bf16::from_f32(s).to_bits().to_le_bytes());
                bias.copy_from_slice(
                    &half::bf16::from_f32(
                        -s * if bits == 4 {
                            7.5
                        } else {
                            127.5
                        },
                    )
                    .to_bits()
                    .to_le_bytes(),
                );
            }
            let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin()).collect();
            let buffers = [
                new_buffer(&device, &codes)?,
                new_buffer(&device, bytemuck::cast_slice(&input))?,
                device
                    .new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                    .ok_or("output allocation failed")?,
                new_buffer(&device, &scales)?,
                new_buffer(&device, &biases)?,
            ];
            let final_output = device
                .new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or("dependent output allocation failed")?;
            let params = if affine {
                [rows as u32, cols as u32, bits, 64, 30, 0, 0, 0]
            } else {
                [rows as u32, cols as u32, kind, block_size as u32, block_bytes as u32, 0, 0, 0]
            };
            let names: &[(&str, usize)] = if blocklane && matches!(kind, 8 | 13 | 14) {
                &[
                    ("hybrid_packed_matvec", 1),
                    ("hybrid_packed_blocklane", 2),
                    ("hybrid_packed_blocklane", 4),
                    ("hybrid_packed_blocklane", 8),
                ]
            } else if affine {
                &[("affine_baseline", 1), ("affine_simd_scalar", 8), ("affine_simd_vector8", 8)]
            } else if packed {
                &[("hybrid_packed_matvec", 1), ("hybrid_packed_simd_vector8_generic", 8)]
            } else {
                &[("hybrid_packed_matvec", 1), ("hybrid_iq_simd_vector8", 8)]
            };
            let mut pipelines: Vec<(&str, usize, Pipeline)> = Vec::new();
            for &(name, tile) in names {
                let constants = MTLFunctionConstantValues::new();
                for (id, value) in [
                    (0, kind),
                    (1, bits.max(4)),
                    (2, 64),
                    (3, 30),
                    (4, block_size as u32),
                    (5, block_bytes as u32),
                    (10, bits.max(4)),
                    (11, 64),
                    (12, 30),
                    (20, tile as u32),
                ] {
                    constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, id);
                }
                let function =
                    library.new_function_with_name_constant_values(name, &constants).map_err(|_| "function failed")?;
                let pipeline =
                    device.new_compute_pipeline_state_with_function(&function).map_err(|_| "pipeline failed")?;
                pipelines.push((name, tile, pipeline));
            }
            let mut samples = vec![Vec::<f64>::new(); pipelines.len()];
            let mut baselines = Vec::new();
            let mut error = vec![0.0f64; pipelines.len()];
            let mut relative_rms = vec![0.0f64; pipelines.len()];
            let started = Instant::now();
            for replay in 0..9 {
                for index in 0..pipelines.len() {
                    let (name, tile, pipeline) = &pipelines[index];
                    let command = queue.command_buffer().ok_or("command failed")?;
                    let encoding = command
                        .compute_command_encoder_with_dispatch_type(if concurrent {
                            MTLDispatchType::CONCURRENT
                        } else {
                            MTLDispatchType::SERIAL
                        })
                        .ok_or("encoder failed")?;
                    encoding.set_compute_pipeline_state(pipeline);
                    encoding.set_buffer(Some(&buffers[1]), 0, 1);
                    encoding.set_bytes(NonNull::from(&params).cast(), std::mem::size_of_val(&params), 3);
                    let offsets = [0u32; 4];
                    encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
                    for bank in 0..banks {
                        encoding.set_buffer(Some(&buffers[2]), bank * rows * 4, 2);
                        encoding.set_buffer(Some(&buffers[0]), bank * code_bytes, 0);
                        encoding.set_buffer(Some(&buffers[3]), bank * plane_bytes, 4);
                        encoding.set_buffer(Some(&buffers[4]), bank * plane_bytes, 5);
                        encoding.dispatch_threadgroups(
                            MTLSize {
                                width: rows.div_ceil(*tile),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: if *name == "hybrid_packed_blocklane" {
                                    tile * 32
                                } else {
                                    256
                                },
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    // Independent GEMVs write disjoint ranges. The dependent
                    // readout cannot execute until every earlier write is visible.
                    encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
                    encoding.set_compute_pipeline_state(&copy);
                    encoding.set_buffer(Some(&buffers[2]), 0, 0);
                    encoding.set_buffer(Some(&final_output), 0, 1);
                    let count = (rows * banks) as u32;
                    encoding.set_bytes(NonNull::from(&count).cast(), std::mem::size_of_val(&count), 2);
                    encoding.dispatch_threads(
                        MTLSize {
                            width: rows * banks,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoding.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    if command.status() != MTLCommandBufferStatus::Completed {
                        return Err("GPU command failed");
                    }
                    // SAFETY: the completed command wrote each checked output
                    // slot once, then its buffer barrier orders the readout.
                    let values = unsafe {
                        std::slice::from_raw_parts(final_output.contents().as_ptr().cast::<f32>(), rows * banks)
                    };
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite output");
                    }
                    if index == 0 {
                        baselines = values.to_vec();
                    } else {
                        error[index] =
                            values.iter().zip(&baselines).map(|(a, b)| f64::from((*a - *b).abs())).fold(0.0, f64::max);
                        let squared_error: f64 =
                            values.iter().zip(&baselines).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                        let squared_reference: f64 = baselines.iter().map(|&v| f64::from(v).powi(2)).sum();
                        relative_rms[index] = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
                    }
                    if replay > 0 {
                        samples[index].push((command.gpu_end_time() - command.gpu_start_time()) / banks as f64);
                    }
                    let _ = name;
                }
            }
            for (index, (name, tile, _)) in pipelines.iter().enumerate() {
                samples[index].sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"case":label,"kernel":name,"rows":rows,"columns":cols,
                    "bits":bits,"kind":kind,"banks":banks,"warm_samples":samples[index].len(),
                    "median_gpu_seconds":(samples[index][3]+samples[index][4])*0.5,"max_abs_vs_baseline":error[index],
                    "relative_rms_vs_baseline":relative_rms[index],
                    "dispatch_mode":if concurrent { "concurrent" } else { "serial" },
                    "rows_per_threadgroup":tile,
                    "case_wall_seconds":started.elapsed().as_secs_f64()})
                );
            }
        }
        Ok(())
    }
}

#[cfg(feature = "metal")]
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    if std::panic::catch_unwind(|| objc2::rc::autoreleasepool(|_| experiment::run())).ok() != Some(Ok(())) {
        eprintln!("native GEMV component experiment failed");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "metal"))]
fn main() {}
