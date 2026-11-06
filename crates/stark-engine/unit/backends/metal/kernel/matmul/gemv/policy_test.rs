use metal::MTLGPUFamily;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
        kernel::matmul::{MatmulShape, QuantParamsLayout},
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
fn dense_qwen27_affine_verify_tiles_preserve_scalar_reduction_geometry() {
    for bits in [4, 8] {
        for m in 2..=4 {
            for (n, k) in [
                (5120, 17408),
                (34816, 5120),
                (16480, 5120),
                (5120, 6144),
                (14336, 5120),
                (248320, 5120),
                (5120, 10240),
            ] {
                let actual = quantized(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, bits, 64, m, n, k).unwrap();
                let input_rows = if bits == 8 && m == 4 && n == 5120 && k == 17408 {
                    2
                } else {
                    m
                };
                assert_eq!(actual.input_row_tile, input_rows);
                assert_eq!(actual.num_simdgroups, 8);
                assert_eq!(actual.results_per_simdgroup, 4);
                assert_eq!(actual.reduction_lanes, 32);
                assert_eq!(
                    actual.group_lanes,
                    if bits == 4 {
                        4
                    } else {
                        8
                    }
                );
            }
        }
        // Older GPUs, other shapes and non-affine layouts keep their route.
        for (cores, family, m, n, k, affine) in [
            (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 4, 14336, 5120, true),
            (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 14336, 5120, true),
            (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 8192, 5120, true),
            (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 14336, 5120, false),
            (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 1, 14336, 5120, true),
        ] {
            let actual = quantized_tile(cores, family, bits, 64, m, n, k, true, affine).unwrap();
            assert_eq!(actual.input_row_tile, 1);
        }
    }
}

#[stark_test]
fn dense_qwen27_affine_verification_dispatch_stays_on_gemv() {
    use crate::backends::metal::kernel::matmul::{MatmulDispatch, MatmulMetalKernel};
    for bits in [4, 8] {
        for m in 2..=4 {
            for (n, k) in [
                (5120, 17408),
                (34816, 5120),
                (16480, 5120),
                (5120, 6144),
                (14336, 5120),
                (248320, 5120),
                (5120, 10240),
            ] {
                let shape = MatmulShape {
                    m,
                    n,
                    k,
                    b_transpose: true,
                    b_leading_dimension: None,
                    b_prologue: GemmBPrologueKind::ScaleBiasDequant,
                    b_bits: Some(bits),
                    b_group_size: Some(64),
                    signed_codes: false,
                    a_full_precision: true,
                    gathered: false,
                    params_layout: Some(QuantParamsLayout::GroupOutput),
                    d_transform: GemmDTransform::empty(),
                };
                let dispatch = MatmulMetalKernel::choose_dispatch(
                    &shape,
                    "Apple M5 Max",
                    40,
                    MTLGPUFamily::Apple10,
                    true,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                );
                assert!(
                    matches!(dispatch, MatmulDispatch::Gemv(_)),
                    "affine verification unexpectedly chose GEMM: W{bits}, m={m}, n={n}, k={k}"
                );
            }
        }
    }
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

/// Private synthetic timing probe; no model, prompt or token data is loaded.
#[stark_test]
#[ignore]
fn dense_qwen27_affine_input_row_tile_microbenchmark() {
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

    const DISPATCHES: usize = 16;
    const WARMUPS: usize = 2;
    const SAMPLES: usize = 9;
    let context = crate::tests::util::shared_metal_context();
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
    let mut kernel = GemvKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    let mut records = Vec::new();
    for bits in [4, 8] {
        for (label, n, k) in [("ffn_up", 34816, 5120), ("ffn_down", 5120, 17408), ("readout", 248320, 5120)] {
            let input =
                QuantInput::<bf16>::new(4, k, n, 64, bits, QuantizationMethod::ScaleBias, 197).with_group_output();
            // Match existing cold GEMV benches: distinct buffers cover at least
            // 256 MiB before an address is reused. Large readouts need one copy.
            let mut pool =
                ColdPool::new(input.weight_buffer_bytes(), || QuantBuffers::<Metal, bf16>::allocate(&context, &input));
            let group_lanes = if bits == 4 {
                4
            } else {
                8
            };
            let mut specializations = Vec::new();
            let mut reference = None::<Vec<bf16>>;
            for rows in [1, 2, 4] {
                let buffers = pool.next_mut();
                let mut command = context.create_command_buffer(None, None).unwrap();
                let arguments = quant_arguments(buffers, &input);
                let shape = MatmulShape::from_arguments(&arguments);
                assert!(shape.m == 4 && shape.params_layout == Some(QuantParamsLayout::GroupOutput));
                let specialization = GemvSpecialization::select_tile(
                    &shape,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                    GemvTile::quantized(8, 4, rows, 32, group_lanes),
                )
                .unwrap();
                kernel.encode(arguments, specialization, &output_work, &mut command).unwrap();
                command.end_encoding().submit().wait_until_completed().unwrap();
                let actual = buffer_to_vec::<Metal, bf16>(&buffers.y);
                assert!(actual.iter().all(|x| x.is_finite()), "synthetic affine result is nonfinite");
                if let Some(reference) = &reference {
                    assert!(
                        actual.iter().zip(reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "input-row tile changes affine arithmetic: W{bits} n={n} k={k} rows={rows}"
                    );
                } else {
                    reference = Some(actual);
                }
                specializations.push(specialization);
            }
            let mut timings: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
            for sample in 0..WARMUPS + SAMPLES {
                // Alternate traversal to reduce a consistent order/thermal bias.
                for offset in 0..3 {
                    let index = if sample % 2 == 0 {
                        offset
                    } else {
                        2 - offset
                    };
                    let mut command = context.create_command_buffer(None, None).unwrap();
                    for _ in 0..DISPATCHES {
                        kernel
                            .encode(
                                quant_arguments(pool.next_mut(), &input),
                                specializations[index],
                                &output_work,
                                &mut command,
                            )
                            .unwrap();
                    }
                    let complete = command.end_encoding().submit().wait_until_completed().unwrap();
                    let per_dispatch_us = complete.gpu_execution_time().as_secs_f64() * 1e6 / DISPATCHES as f64;
                    assert!(per_dispatch_us.is_finite() && per_dispatch_us > 0.0);
                    if sample >= WARMUPS {
                        timings[index].push(per_dispatch_us);
                    }
                }
            }
            let variants = [1, 2, 4]
                .into_iter()
                .zip(timings)
                .map(|(rows, mut values)| {
                    values.sort_by(f64::total_cmp);
                    serde_json::json!({"input_row_tile": rows, "median_gpu_us": values[SAMPLES / 2],
                    "min_gpu_us": values[0], "max_gpu_us": values[SAMPLES-1]})
                })
                .collect::<Vec<_>>();
            records.push(serde_json::json!({"label": label, "bits": bits, "m": 4, "n": n, "k": k,
                "weight_and_coefficient_bytes": input.weight_buffer_bytes(), "bit_exact": true,
                "variants": variants}));
        }
    }
    let report = serde_json::json!({"schema": "stark-affine-input-row-tile-microbenchmark-v1",
        "synthetic": true, "timer": "gpu-command-buffer", "cold_working_set_min_bytes": 256usize << 20,
        "dispatches_per_sample": DISPATCHES, "warmups": WARMUPS, "samples": SAMPLES, "records": records});
    if let Some(path) = std::env::var_os("STARK_AFFINE_TILE_BENCH_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    println!("{report}");
}
