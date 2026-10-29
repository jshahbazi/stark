use metal::MTLGPUFamily;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
        kernel::matmul::{MatmulB, MatmulShape, QuantParamsLayout},
    },
    data_type::DataType,
};

const SMALL_GPU_CORE_COUNT: u32 = 29;
const LARGE_GPU_CORE_COUNT: u32 = 30;

fn qtile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    GemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, 2)
}

#[stark_test]
fn fp_policy_cases() {
    let cases = [
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 12288, 1536, true, tile(8, 8, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 12288, 1536, false, tile(8, 1, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 256, true, tile(8, 1, 1)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 256, true, tile(8, 2, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 8, 12288, 1536, true, tile(8, 1, 4)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 12288, true, tile(8, 8, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple7, 1, 262144, 1536, true, tile(8, 1, 4)),
    ];

    for (gpu_core_count, apple_gpu_family, m, n, k, aligned, expected) in cases {
        assert_eq!(fp_tile(gpu_core_count, apple_gpu_family, m, n, k, aligned), expected);
    }
}

#[stark_test]
fn quant_policy_cases() {
    let cases = [
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 256, 1536, 4, qtile(2, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 262144, 1536, 4, qtile(8, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 256, 4, qtile(4, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 1, 1536, 256, 4, qtile(8, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple8, 1, 2048, 1536, 4, qtile(8, 2)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple7, 1, 256, 1536, 4, qtile(8, 2)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple7, 1, 1536, 256, 4, qtile(4, 8)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 2, 2048, 1536, 4, qtile(8, 4)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 4, 5120, 5120, 4, qtile(8, 4)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 2048, 1536, 8, GemvTile::quantized(8, 4, 1, 32, 4)),
    ];

    for (gpu_core_count, apple_gpu_family, m, n, k, bits, expected) in cases {
        let actual =
            quantized_tile(gpu_core_count, apple_gpu_family, bits, 32, m, n, k, true, true).expect("quantized route");
        assert_eq!(
            actual, expected,
            "m={m} n={n} k={k} bits={bits} gpu_core_count={gpu_core_count} apple_gpu_family={apple_gpu_family:?}"
        );
    }
}

#[stark_test]
fn gpu_core_count_boundary_selects_large_policy() {
    let small = quantized(SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 4, 32, 1, 256, 1536).expect("small GPU route");
    let large = quantized(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 4, 32, 1, 256, 1536).expect("large GPU route");

    assert_eq!(small, qtile(4, 2));
    assert_eq!(large, qtile(2, 1));
}

fn quantized(
    gpu_core_count: u32,
    apple_gpu_family: MTLGPUFamily,
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Option<GemvTile> {
    quantized_tile(gpu_core_count, apple_gpu_family, bits, group, m, n, k, true, true)
}

#[stark_test]
fn quantized_policy_edges() {
    assert_eq!(
        quantized(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 64, 4, 8192, 4100),
        Some(GemvTile::quantized(8, 4, 1, 32, 4))
    );
    assert!(quantized(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 4, 8, 4096).is_some());
    assert!(quantized(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 1, 8192, 4096).is_some());
    assert_eq!(quantized(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 9, 8192, 4096), None);
    assert_eq!(
        quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 4, 8192, 4096, true, true),
        Some(GemvTile::quantized(8, 4, 1, 32, 2))
    );
}

#[stark_test]
fn untuned_quantized_io_uses_only_the_generated_fallback() {
    assert_eq!(quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 4, 3, 4096, false, true), None);
    let tile = quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 4, 4096, 4096, false, true)
        .expect("mixed-IO fallback");
    assert_eq!(tile, GemvTile::quantized(8, 4, 1, 32, 2));
    assert_eq!(quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 5, 4096, 4096, false, true), None);
}

fn quant_shape(
    m: u32,
    n: u32,
    d_transform: GemmDTransform,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k: 4096,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleZeroPointDequant,
        b_bits: Some(4),
        b_group_size: Some(32),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::OutputGroup),
        d_transform,
    }
}

#[stark_test]
fn block_unaligned_quantized_k_stays_on_gemv() {
    let shape = MatmulShape {
        m: 1,
        n: 64,
        k: 1152,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleZeroPointDequant,
        b_bits: Some(4),
        b_group_size: Some(64),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::OutputGroup),
        d_transform: GemmDTransform::empty(),
    };
    assert!(
        super::super::kernel::GemvSpecialization::select_shape(
            &shape,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            LARGE_GPU_CORE_COUNT,
            MTLGPUFamily::Apple10,
        )
        .is_some()
    );
}

#[stark_test]
fn specialization_preserves_quantized_route_and_accumulate_tail() {
    let select = |n, transform| {
        super::super::kernel::GemvSpecialization::select_shape(
            &quant_shape(4, n, transform),
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            LARGE_GPU_CORE_COUNT,
            MTLGPUFamily::Apple10,
        )
    };
    let clean = select(8192, GemmDTransform::empty()).expect("quantized specialization");
    assert!(clean.output_row_tile() > DEFAULT_RESULTS_PER_SIMDGROUP);
    assert_eq!(select(8192 + clean.output_row_tile() / 2, GemmDTransform::ACCUMULATE), None);
}

#[stark_test]
fn gathered_group_major_is_a_gemv_route() {
    let mut shape = quant_shape(1, 8192, GemmDTransform::empty());
    shape.params_layout = Some(QuantParamsLayout::GroupOutput);
    shape.gathered = true;
    assert!(
        super::super::kernel::GemvSpecialization::select_tile(
            &shape,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            qtile(8, 4),
        )
        .is_some()
    );
}

#[stark_test]
fn qwen35_tile8_policy_is_limited_to_the_measured_decode_shapes() {
    let candidate = GemvTile::quantized(2, 4, 1, 32, 4);
    let previous = GemvTile::quantized(2, 1, 1, 32, 4);
    for n in [8224, 12288] {
        assert_eq!(quantized(30, MTLGPUFamily::Apple10, 4, 64, 1, n, 2048), Some(candidate));
    }

    // Each case changes one admission dimension. Nearby shapes, other devices,
    // other formats and multiple input rows keep their existing policy.
    let fallback = GemvTile::quantized(8, 4, 1, 32, 4);
    let cases = [
        (29, MTLGPUFamily::Apple10, 4, 64, 1, 8224, 2048, fallback),
        (30, MTLGPUFamily::Apple9, 4, 64, 1, 8224, 2048, previous),
        (30, MTLGPUFamily::Apple8, 4, 64, 1, 8224, 2048, previous),
        (30, MTLGPUFamily::Apple7, 4, 64, 1, 8224, 2048, previous),
        (30, MTLGPUFamily::Apple10, 8, 64, 1, 8224, 2048, GemvTile::quantized(8, 4, 1, 32, 8)),
        (30, MTLGPUFamily::Apple10, 4, 32, 1, 8224, 2048, GemvTile::quantized(2, 1, 1, 32, 2)),
        (30, MTLGPUFamily::Apple10, 4, 64, 2, 8224, 2048, fallback),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 8224, 2047, previous),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 8224, 2049, fallback),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 8223, 2048, previous),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 8225, 2048, previous),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 12287, 2048, previous),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 12289, 2048, previous),
        (30, MTLGPUFamily::Apple10, 4, 64, 1, 248320, 2048, fallback),
    ];
    for (cores, family, bits, group, m, n, k, expected) in cases {
        assert_eq!(quantized(cores, family, bits, group, m, n, k), Some(expected));
    }
    assert_eq!(quantized_tile(30, MTLGPUFamily::Apple10, 4, 64, 1, 8224, 2048, false, true), Some(fallback));
    assert_eq!(quantized_tile(30, MTLGPUFamily::Apple10, 4, 64, 1, 8224, 2048, true, false), Some(previous));
}

#[stark_test]
fn qwen35_tile8_specialization_accepts_native_group_major_affine_format() {
    for n in [8224, 12288] {
        let shape = MatmulShape {
            k: 2048,
            b_prologue: GemmBPrologueKind::ScaleBiasDequant,
            b_group_size: Some(64),
            params_layout: Some(QuantParamsLayout::GroupOutput),
            ..quant_shape(1, n, GemmDTransform::empty())
        };
        let specialization = super::super::kernel::GemvSpecialization::select_shape(
            &shape,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            30,
            MTLGPUFamily::Apple10,
        )
        .expect("native Qwen3.5 quantized specialization");
        assert_eq!(specialization.output_row_tile(), 8);
    }
}

#[stark_test]
fn qwen35_tile8_specialization_rejects_unqualified_formats() {
    let shape = MatmulShape {
        k: 2048,
        b_prologue: GemmBPrologueKind::ScaleBiasDequant,
        b_group_size: Some(64),
        params_layout: Some(QuantParamsLayout::GroupOutput),
        ..quant_shape(1, 8224, GemmDTransform::empty())
    };
    let select = |shape: &MatmulShape, weights, input, output| {
        super::super::kernel::GemvSpecialization::select_shape(shape, weights, input, output, 30, MTLGPUFamily::Apple10)
            .map(|specialization| specialization.output_row_tile())
    };
    for b_prologue in [GemmBPrologueKind::ScaleZeroPointDequant, GemmBPrologueKind::ScaleSymmetricDequant] {
        assert_eq!(
            select(
                &MatmulShape {
                    b_prologue,
                    ..shape
                },
                DataType::BF16,
                DataType::BF16,
                DataType::BF16
            ),
            Some(2)
        );
    }
    assert_eq!(
        select(
            &MatmulShape {
                params_layout: Some(QuantParamsLayout::OutputGroup),
                ..shape
            },
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
        ),
        Some(2)
    );
    assert_eq!(
        select(
            &MatmulShape {
                signed_codes: true,
                ..shape
            },
            DataType::BF16,
            DataType::BF16,
            DataType::BF16
        ),
        Some(2)
    );
    assert_eq!(select(&shape, DataType::F32, DataType::BF16, DataType::BF16), Some(2));
    assert_eq!(select(&shape, DataType::BF16, DataType::F32, DataType::BF16), Some(32));
    assert_eq!(select(&shape, DataType::BF16, DataType::BF16, DataType::F32), Some(32));
    assert_eq!(
        select(
            &MatmulShape {
                gathered: true,
                ..shape
            },
            DataType::BF16,
            DataType::BF16,
            DataType::BF16
        ),
        Some(32)
    );
}

#[stark_test]
fn qwen35_tile8_preserves_group_major_affine_outputs() {
    use half::bf16;

    use super::super::kernel::{GemvKernel, GemvSpecialization};
    use crate::{
        backends::{
            common::{
                CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
                gpu_types::QuantizationMethod,
            },
            metal::{Metal, kernel::matmul::MatmulOutputWork},
        },
        tests::{
            helpers::buffer_to_vec,
            matmul::{QuantBuffers, QuantInput, quant_arguments},
        },
    };

    let context = crate::tests::util::shared_metal_context();
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
    let mut kernel = GemvKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    for n in [8224, 12288] {
        let input = QuantInput::<bf16>::new(1, 2048, n, 64, 4, QuantizationMethod::ScaleBias, 47).with_group_output();
        let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
        let mut outputs = Vec::new();
        for results in [1, 4] {
            let mut command_buffer = context.create_command_buffer(None, None).unwrap();
            let arguments = quant_arguments(&mut buffers, &input);
            let shape = MatmulShape::from_arguments(&arguments);
            assert_eq!(shape.params_layout, Some(QuantParamsLayout::GroupOutput));
            assert_eq!(shape.b_prologue, GemmBPrologueKind::ScaleBiasDequant);
            let specialization = GemvSpecialization::select_tile(
                &shape,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
                GemvTile::quantized(2, results, 1, 32, 4),
            )
            .unwrap();
            kernel.encode(arguments, specialization, &output_work, &mut command_buffer).unwrap();
            command_buffer.end_encoding().submit().wait_until_completed().unwrap();
            outputs.push(buffer_to_vec::<Metal, bf16>(&buffers.y));
        }
        // Geometry changes only the independent output-row assignment. Compare
        // every result in memory; assertion messages retain no output values.
        assert_eq!(outputs[0].len(), n as usize);
        assert!(
            outputs[0].iter().zip(&outputs[1]).all(|(baseline, candidate)| baseline.to_bits() == candidate.to_bits())
        );
    }
}

#[stark_test]
fn source_tiles_are_scoped_to_measured_affine_projections() {
    for (n, k, simdgroups, rows) in [
        (4096, 4096, 4, 2),
        (12288, 4096, 2, 4),
        (22016, 4096, 8, 2),
        (4096, 11008, 4, 2),
        (100278, 4096, 2, 4),
        (12288, 1536, 2, 4),
    ] {
        let tuned = GemvTile::quantized(simdgroups, rows, 1, 32, 4);
        assert_eq!(source_projection_tile(40, MTLGPUFamily::Apple10, 64, n, k), Some(tuned));
        for (cores, family, group) in
            [(29, MTLGPUFamily::Apple10, 64), (40, MTLGPUFamily::Apple9, 64), (40, MTLGPUFamily::Apple10, 32)]
        {
            assert_eq!(source_projection_tile(cores, family, group, n, k), None);
        }
    }
    for (n, k) in [(4096, 4096), (12288, 4096), (22016, 4096), (4096, 11008)] {
        assert_eq!(quantized(40, MTLGPUFamily::Apple10, 4, 64, 1, n, k), Some(GemvTile::quantized(8, 4, 1, 32, 4)));
    }
    assert_eq!(source_projection_tile(40, MTLGPUFamily::Apple10, 64, 262144, 1536), None);
}

#[stark_test]
fn olmo3_tile8_preserves_every_projection_output_bit() {
    use half::bf16;

    use super::super::kernel::{GemvKernel, GemvSpecialization};
    use crate::{
        backends::{
            common::{
                CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
                gpu_types::QuantizationMethod,
            },
            metal::{Metal, kernel::matmul::MatmulOutputWork},
        },
        tests::{
            helpers::buffer_to_vec,
            matmul::{QuantBuffers, QuantInput, quant_arguments},
        },
    };
    let context = crate::tests::util::shared_metal_context();
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
    let mut kernel = GemvKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    for (n, k) in [(4096, 4096), (12288, 4096), (22016, 4096), (4096, 11008)] {
        let input = QuantInput::<bf16>::new(1, k, n, 64, 4, QuantizationMethod::ScaleBias, 89).with_group_output();
        let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
        for materialize_input_sum in [false, true] {
            let mut outputs = Vec::new();
            for groups in [8, 2] {
                let mut command_buffer = context.create_command_buffer(None, None).unwrap();
                let mut arguments = quant_arguments(&mut buffers, &input);
                if let MatmulB::Quantized(quantized) = &mut arguments.b {
                    quantized.params = quantized.params.with_materialized_affine_input_sum(materialize_input_sum);
                }
                let shape = MatmulShape::from_arguments(&arguments);
                let specialization = GemvSpecialization::select_tile(
                    &shape,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                    GemvTile::quantized(groups, 4, 1, 32, 4),
                )
                .unwrap();
                kernel.encode(arguments, specialization, &output_work, &mut command_buffer).unwrap();
                command_buffer.end_encoding().submit().wait_until_completed().unwrap();
                outputs.push(buffer_to_vec::<Metal, bf16>(&buffers.y));
            }
            assert!(
                outputs[0].iter().zip(&outputs[1]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "projection tile changed BF16 output bits"
            );
        }
    }
}

/// Development-only cold-weight sweep for the dense Q4 projection geometries.
/// Timings and hashes are aggregate-only; synthetic tensors never leave memory.
#[stark_test]
#[ignore]
fn dense_q4_cold_tile_sweep() {
    use half::bf16;

    use super::super::kernel::{GemvKernel, GemvSpecialization};
    use crate::{
        backends::{
            common::{
                CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
                gpu_types::QuantizationMethod,
            },
            metal::{Metal, kernel::matmul::MatmulOutputWork},
        },
        tests::{
            cold_pool::ColdPool,
            helpers::buffer_to_vec,
            matmul::{QuantBuffers, QuantInput, quant_arguments},
        },
    };

    let context = crate::tests::util::shared_metal_context();
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
    let mut kernel = GemvKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    let mut rows = Vec::new();
    for (n, k) in [
        (12288, 4096),
        (4096, 4096),
        (22016, 4096),
        (4096, 11008),
        (100278, 4096),
        (2048, 1536),
        (2560, 1536),
        (4096, 1536),
        (5120, 1536),
        (12288, 1536),
        (24576, 1536),
        (1536, 2048),
        (1536, 4096),
        (1536, 6144),
        (1536, 12288),
        (262144, 1536),
    ] {
        let input = QuantInput::<bf16>::new(1, k, n, 64, 4, QuantizationMethod::ScaleBias, 83).with_group_output();
        let mut pool =
            ColdPool::new(input.weight_buffer_bytes(), || QuantBuffers::<Metal, bf16>::allocate(&context, &input));
        for materialize_input_sum in [false, true] {
            let mut baseline = None::<Vec<bf16>>;
            // Use only geometries emitted by gemv.metal's specialization constraints.
            for (num_simdgroups, results) in [(2, 1), (2, 2), (2, 4), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4), (8, 8)] {
                if num_simdgroups == 8 && results == 8 && !k.is_multiple_of(512) {
                    continue;
                }
                let mut times = Vec::new();
                // Initialize the complete 256 MiB cold pool before measuring.
                // Each submission completes before the next is encoded, so
                // independent synthetic projections cannot overlap on the GPU.
                for sample in 0..18 {
                    let buffers = pool.next_mut();
                    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
                    let mut arguments = quant_arguments(buffers, &input);
                    if let MatmulB::Quantized(quantized) = &mut arguments.b {
                        quantized.params = quantized.params.with_materialized_affine_input_sum(materialize_input_sum);
                    }
                    let shape = MatmulShape::from_arguments(&arguments);
                    let specialization = GemvSpecialization::select_tile(
                        &shape,
                        DataType::BF16,
                        DataType::BF16,
                        DataType::BF16,
                        GemvTile::quantized(num_simdgroups, results, 1, 32, 4),
                    )
                    .unwrap();
                    kernel.encode(arguments, specialization, &output_work, &mut command_buffer).unwrap();
                    let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
                    if sample >= 3 {
                        times.push(completed.gpu_execution_time().as_secs_f64());
                    }
                    if sample == 0 {
                        let output = buffer_to_vec::<Metal, bf16>(&buffers.y);
                        if let Some(expected) = &baseline {
                            assert!(
                                output.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                                "tile changed synthetic output bits"
                            );
                        } else {
                            baseline = Some(output);
                        }
                    }
                }
                times.sort_by(f64::total_cmp);
                rows.push(serde_json::json!({
                    "n":n,"k":k,"num_simdgroups":num_simdgroups,"results_per_simdgroup":results,
                    "materialize_affine_input_sum":materialize_input_sum,
                    "median_gpu_seconds":times[times.len()/2],"min_gpu_seconds":times[0],
                    "max_gpu_seconds":times[times.len()-1],"measured_runs":times.len(),"bitwise_equal":true,
                }));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"scope":"synthetic-cold-q4-g64-tile-sweep-v2","working_set_bytes":256usize<<20,"cases":rows})
    );
}

#[path = "source_layout_profile.rs"]
mod source_layout_profile;
