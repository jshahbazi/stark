//! Explicit private metadata-layout A/B. Synthetic values stay in memory.

use std::{env, path::PathBuf};

use half::bf16;
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::{
    super::super::kernel::{GemvKernel, GemvSpecialization},
    GemvTile, MatmulB, MatmulShape, QuantParamsLayout,
};
use crate::{
    backends::{
        common::{
            BufferRef, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, gpu_types::QuantizationMethod,
        },
        metal::{Metal, kernel::matmul::MatmulOutputWork},
    },
    data_type::DataType,
    tests::{
        cold_pool::{ColdPool, copy_count},
        helpers::buffer_to_vec,
        matmul::{QuantBuffers, QuantInput, quant_arguments},
        private_output::private_write,
    },
};

const WORKING_SET_BYTES: usize = 256 << 20;
const WARMUP_SAMPLES: usize = 3;
const MEASURED_SAMPLES: usize = 25;

fn digest_bf16(values: &[bf16]) -> String {
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

#[stark_test]
#[ignore] // Explicit GPU slot; aggregate-only output must target a fresh ignored file.
fn source_q4_cold_metadata_layout_profile() {
    let output = PathBuf::from(env::var_os("STARK_GEMV_LAYOUT_PROFILE_OUTPUT").expect("private output required"));
    let identity = PathBuf::from(
        env::var_os("STARK_GEMV_PROFILE_ENCLOSING_RECEIPT").expect("enclosing kernel/host identity receipt required"),
    );
    assert!(std::fs::metadata(&identity).unwrap().len() <= 1024 * 1024, "identity receipt too large");
    let identity_sha256 = format!("{:x}", Sha256::digest(std::fs::read(&identity).unwrap()));
    let context = crate::tests::util::shared_metal_context();
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
    let mut kernel = GemvKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    let mut cases = Vec::new();

    for (n, k, simdgroups, rows_per_simdgroup) in [(22016, 4096, 8, 2), (4096, 11008, 4, 2)] {
        let output_group = QuantInput::<bf16>::new(1, k, n, 64, 4, QuantizationMethod::ScaleBias, 109);
        let group_output = output_group.clone().with_group_output();
        assert_eq!(output_group.params_layout, QuantParamsLayout::OutputGroup);
        assert_eq!(group_output.params_layout, QuantParamsLayout::GroupOutput);
        assert!(output_group.w_packed == group_output.w_packed, "packed codes changed");
        let codes_sha256 = format!("{:x}", Sha256::digest(bytemuck::cast_slice(&output_group.w_packed)));
        let bytes_per_copy = output_group.weight_buffer_bytes();
        assert_eq!(bytes_per_copy, group_output.weight_buffer_bytes());
        let copies = copy_count(WORKING_SET_BYTES, bytes_per_copy);
        let inputs = [&group_output, &output_group];
        let mut group_pool =
            ColdPool::new(bytes_per_copy, || QuantBuffers::<Metal, bf16>::allocate(&context, &group_output));
        let mut output_pool =
            ColdPool::new(bytes_per_copy, || QuantBuffers::<Metal, bf16>::allocate(&context, &output_group));
        let mut timings = [Vec::<f64>::new(), Vec::<f64>::new()];
        let mut expected = None::<Vec<bf16>>;
        let mut output_hashes = [String::new(), String::new()];
        let mut metadata_hashes = [String::new(), String::new()];
        let mut checked_planes = None::<(Vec<bf16>, Vec<bf16>)>;

        for sample in 0..WARMUP_SAMPLES + MEASURED_SAMPLES {
            // Reverse the pair order every sample; neither layout gets a fixed thermal/order advantage.
            for layout in if sample.is_multiple_of(2) {
                [0usize, 1]
            } else {
                [1, 0]
            } {
                let input = inputs[layout];
                if sample == 0 {
                    let buffers = if layout == 0 {
                        group_pool.next_mut()
                    } else {
                        output_pool.next_mut()
                    };
                    assert_eq!(
                        format!("{:x}", Sha256::digest((&buffers.w).as_slice::<u8>())),
                        codes_sha256,
                        "uploaded packed codes changed"
                    );
                    let scales = (&buffers.scales).as_slice::<bf16>();
                    let biases = buffers.bias.as_ref().unwrap().as_slice::<bf16>();
                    let mut digest = Sha256::new();
                    digest.update(bytemuck::cast_slice(scales));
                    digest.update(bytemuck::cast_slice(biases));
                    metadata_hashes[layout] = format!("{:x}", digest.finalize());
                    let groups = (k / 64) as usize;
                    if layout == 0 {
                        checked_planes = Some((scales.to_vec(), biases.to_vec()));
                    } else {
                        let (group_scales, group_biases) = checked_planes.as_ref().expect("group-major control first");
                        for row in 0..n as usize {
                            for group in 0..groups {
                                assert!(
                                    scales[row * groups + group].to_bits()
                                        == group_scales[group * n as usize + row].to_bits(),
                                    "metadata transpose changed scale bits"
                                );
                                assert!(
                                    biases[row * groups + group].to_bits()
                                        == group_biases[group * n as usize + row].to_bits(),
                                    "metadata transpose changed bias bits"
                                );
                            }
                        }
                    }
                }

                let mut commands = context.create_command_buffer(None, None).unwrap();
                // A complete cold-bank cycle remains continuously queued on the GPU.
                for _ in 0..copies {
                    let buffers = if layout == 0 {
                        group_pool.next_mut()
                    } else {
                        output_pool.next_mut()
                    };
                    let mut arguments = quant_arguments(buffers, input);
                    let MatmulB::Quantized(quantized) = &mut arguments.b else {
                        unreachable!()
                    };
                    quantized.params = quantized.params.with_materialized_affine_input_sum(true);
                    let shape = MatmulShape::from_arguments(&arguments);
                    let specialization = GemvSpecialization::select_tile(
                        &shape,
                        DataType::BF16,
                        DataType::BF16,
                        DataType::BF16,
                        GemvTile::quantized(simdgroups, rows_per_simdgroup, 1, 32, 4),
                    )
                    .expect("same forced source tile");
                    kernel.encode(arguments, specialization, &output_work, &mut commands).unwrap();
                }
                let completed = commands.end_encoding().submit().wait_until_completed().unwrap();
                if sample >= WARMUP_SAMPLES {
                    timings[layout].push(completed.gpu_execution_time().as_secs_f64() / copies as f64);
                }
                for _ in 0..copies {
                    let buffers = if layout == 0 {
                        group_pool.next_mut()
                    } else {
                        output_pool.next_mut()
                    };
                    let result = buffer_to_vec::<Metal, bf16>(&buffers.y);
                    assert!(result.len() == n as usize, "unexpected projection output length");
                    if let Some(expected) = &expected {
                        assert!(
                            result
                                .iter()
                                .zip(expected)
                                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                            "metadata layout changed source output bits"
                        );
                    } else {
                        expected = Some(result.clone());
                    }
                    output_hashes[layout] = digest_bf16(&result);
                }
            }
        }
        for values in &mut timings {
            values.sort_by(f64::total_cmp);
        }
        let median = |layout: usize| timings[layout][MEASURED_SAMPLES / 2];
        cases.push(serde_json::json!({
            "n":n,"k":k,"m":1,"bits":4,"group_size":64,"activation_dtype":"bfloat16",
            "accumulation_policy":"existing explicit MLX BF16 source GEMV, identical fixed geometry",
            "num_simdgroups":simdgroups,"results_per_simdgroup":rows_per_simdgroup,
            "cold_target_bytes_per_layout":WORKING_SET_BYTES,"copies_per_layout":copies,
            "actual_weight_pool_bytes_per_layout":copies*bytes_per_copy,
            "dispatches_per_gpu_sample":copies,
            "warmup_samples":WARMUP_SAMPLES,"measured_samples_per_layout":MEASURED_SAMPLES,
            "packed_codes_byte_identical":true,"codes_sha256":codes_sha256,
            "metadata_transpose_bit_exact":true,"all_outputs_bit_exact":true,
            "group_output":{"median_gpu_seconds":median(0),"min_gpu_seconds":timings[0][0],"max_gpu_seconds":timings[0][MEASURED_SAMPLES-1],"metadata_sha256":metadata_hashes[0],"output_sha256":output_hashes[0]},
            "output_group":{"median_gpu_seconds":median(1),"min_gpu_seconds":timings[1][0],"max_gpu_seconds":timings[1][MEASURED_SAMPLES-1],"metadata_sha256":metadata_hashes[1],"output_sha256":output_hashes[1]},
            "output_group_speedup":median(0)/median(1),
            "group_output_gpu_samples_seconds":timings[0],"output_group_gpu_samples_seconds":timings[1],
        }));
    }
    let report = serde_json::json!({"scope":"private-source-q4-g64-cold-queued-metadata-layout-v2","passed":true,"cases":cases,
        "enclosing_identity_sha256":identity_sha256,
        "identity_contract":"Enclosing receipt must bind runtime/kernel/executable source hashes, macOS version and Rust/Metal toolchains",
        "harness_sha256":format!("{:x}",Sha256::digest(include_bytes!("source_layout_profile.rs"))),
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),"/../../Cargo.lock")))),
        "device":{"name":context.device_name,"architecture":context.gpu_architecture,"gpu_core_count":context.gpu_core_count,"family":format!("{:?}",context.apple_gpu_family)},
        "timing":"MTL4 GPU duration for one continuously queued cold-bank cycle, divided by dispatch count; samples complete serially",
        "qualification":"synthetic microbenchmark only; no production layout or serialized schema change"});
    private_write(&output, &serde_json::to_vec_pretty(&report).unwrap()).expect("fresh private report");
}
