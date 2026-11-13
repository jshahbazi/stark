use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use rstest::rstest;
use stark_engine_macros::stark_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
        matmul::{QuantBuffers, QuantInput},
    },
};
#[cfg(backend = "metal")]
use crate::{
    backends::metal::{Metal, MetalContext},
    tests::matmul::run_quant_cpu,
};

struct Input<T: ArrayElement + Float> {
    a: Box<[T]>,
    b: Box<[T]>,
    m: usize,
    k: usize,
    n: usize,
    ids: Option<Box<[u32]>>,
    soft_cap: Option<f32>,
}

#[stark_test]
fn packed_affine_cpu_matches_lossless_u8() {
    for bits in [5, 6] {
        let input =
            QuantInput::<bf16>::new(4, 256, 64, 64, bits, QuantizationMethod::ScaleBias, 107).with_group_output();
        let packed = crate::tests::matmul::run_quant_cpu(&input);
        let widened = crate::tests::matmul::run_quant_cpu(&input.widened_u8());
        assert!(packed == widened, "packed affine CPU calculation differs from widened codes");
    }
}

fn run_coarse_f32<B: Backend>(
    context: &B::Context,
    input: &QuantInput<bf16>,
) -> Vec<f32> {
    let buffers = QuantBuffers::<B, bf16>::allocate(context, input);
    let mut output = create_buffer::<B, f32>(context, (input.m * input.n) as usize);
    let mut kernel = <B::Kernels as Kernels>::MatmulKernel::new(
        context,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::F32,
    )
    .unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &buffers.x,
                    offset: 0,
                },
                b: buffers.matmul_b(input),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                gather_indices: None::<&B::GlobalBuffer>,
                m: input.m,
                n: input.n,
                k: input.k,
            },
            &mut command,
        )
        .unwrap();
    command.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<B, f32>(&output)
}

#[stark_test]
fn internal_coarse_three_bit_cpu_matches_u8_f32_scores() {
    let context = <Cpu as Backend>::Context::new().unwrap();
    let input = QuantInput::<bf16>::new(1, 256, 64, 64, 3, QuantizationMethod::ScaleBias, 131).with_group_output();
    let actual = run_coarse_f32::<Cpu>(&context, &input);
    let expected = run_coarse_f32::<Cpu>(&context, &input.widened_u8());
    assert!(
        actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "internal coarse3 CPU scores differ"
    );
}

#[cfg(backend = "metal")]
#[stark_test]
fn internal_coarse_three_bit_metal_matches_u8_f32_scores() {
    let context = MetalContext::new().unwrap();
    for k in [64, 256, 5120] {
        let input = QuantInput::<bf16>::new(1, k, 128, 64, 3, QuantizationMethod::ScaleBias, 137).with_group_output();
        let actual = run_coarse_f32::<Metal>(&context, &input);
        let expected = run_coarse_f32::<Metal>(&context, &input.widened_u8());
        assert!(
            actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
            "internal coarse3 Metal scores differ"
        );
    }
}

#[cfg(backend = "metal")]
#[stark_test]
fn packed_affine_gathered_rescore_matches_full_readout() {
    let context = MetalContext::new().unwrap();
    let (k, vocab, count) = (5120, 256, 64);
    let ids = (0..count).map(|i| (i * 73 + 11) % vocab).collect::<Vec<_>>();
    let gathered = create_buffer_with_data::<Metal, u32>(&context, &ids);
    for bits in [5, 6] {
        let input =
            QuantInput::<bf16>::new(1, k, vocab, 64, bits, QuantizationMethod::ScaleBias, 139).with_group_output();
        let buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
        let full = run_gemv::<Metal, bf16>(
            &context,
            &buffers.x,
            buffers.matmul_b(&input),
            None::<&<Metal as Backend>::GlobalBuffer>,
            1,
            vocab as usize,
            k as usize,
            None,
        );
        let selected = run_gemv::<Metal, bf16>(
            &context,
            &buffers.x,
            buffers.matmul_b(&input),
            Some(&gathered),
            1,
            count as usize,
            k as usize,
            None,
        );
        assert!(
            selected.iter().zip(&ids).all(|(&score, &id)| score.to_bits() == full[id as usize].to_bits()),
            "packed gathered rescore differs from full vocabulary scores"
        );
    }
}

#[cfg(backend = "metal")]
#[stark_test]
fn packed_affine_metal_matches_u8_and_scalar_rows() {
    let context = MetalContext::new().expect("Metal context");
    for bits in [5, 6] {
        for (k, n) in [(64, 4), (128, 4), (5120, 128), (17408, 128), (5120, 160)] {
            for m in 1..=4 {
                let input =
                    QuantInput::<bf16>::new(m, k, n, 64, bits, QuantizationMethod::ScaleBias, 109).with_group_output();
                let packed = crate::tests::matmul::quant::run_quant_metal(&context, &input, None);
                let widened =
                    crate::tests::matmul::quant::run_quant_metal_policy(&context, &input.widened_u8(), None, Some(n));
                assert!(packed == widened, "packed affine GEMV differs from widened U8 arithmetic");
                for row in 0..m as usize {
                    let mut scalar = input.clone();
                    scalar.m = 1;
                    scalar.x = input.x[row * k as usize..(row + 1) * k as usize].to_vec();
                    let reference = crate::tests::matmul::quant::run_quant_metal(&context, &scalar, None);
                    assert!(
                        packed[row * n as usize..(row + 1) * n as usize] == reference,
                        "packed affine batch differs from independent scalar rows"
                    );
                }
            }
        }
        for engine in [crate::backends::metal::GemmEngine::Simdgroup, crate::backends::metal::GemmEngine::Mxu] {
            let input = QuantInput::<bf16>::new(64, 5120, 128, 64, bits, QuantizationMethod::ScaleBias, 113)
                .with_group_output();
            let packed = crate::tests::matmul::quant::run_quant_metal(&context, &input, Some(engine));
            let widened = crate::tests::matmul::quant::run_quant_metal(&context, &input.widened_u8(), Some(engine));
            assert!(packed == widened, "packed affine prefill differs from widened U8 arithmetic");
        }
    }
}

#[cfg(backend = "metal")]
#[stark_test]
fn packed_affine_partition_retains_fused_u8_arithmetic() {
    use crate::tests::matmul::quant::{run_quant_metal, run_quant_metal_policy};

    let context = MetalContext::new().expect("Metal context");
    let (k, fused_n, part_n) = (5120, 16480, 64);
    for bits in [4, 5, 6, 8] {
        for m in [1, 2, 3, 4, 64] {
            let fused = QuantInput::<bf16>::new(m, k, fused_n, 64, bits, QuantizationMethod::ScaleBias, 127)
                .with_group_output();
            let reference = if bits == 8 {
                fused.clone()
            } else {
                fused.widened_u8()
            };
            let expected = run_quant_metal(&context, &reference, None);
            for offset in [0, 48] {
                let mut part = fused.clone();
                part.n = part_n;
                part.w_packed = fused.w_packed
                    [(offset * k * bits / 32) as usize..((offset + part_n) * k * bits / 32) as usize]
                    .to_vec();
                part.scales = fused.scales[(offset * k / 64) as usize..((offset + part_n) * k / 64) as usize].to_vec();
                part.biases = Some(
                    fused.biases.as_ref().unwrap()[(offset * k / 64) as usize..((offset + part_n) * k / 64) as usize]
                        .to_vec(),
                );
                let actual = run_quant_metal_policy(&context, &part, None, Some(fused_n));
                assert!(
                    (0..m as usize).all(|row| {
                        actual[row * part_n as usize..(row + 1) * part_n as usize]
                            == expected[row * fused_n as usize + offset as usize
                                ..row * fused_n as usize + (offset + part_n) as usize]
                    }),
                    "packed partition differs from original fused U8 arithmetic"
                );
            }
        }
    }
}

fn get_test_data<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
) -> (Input<T>, Vec<T>) {
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();

    let input = Input {
        a: a.into_boxed_slice(),
        b: b.into_boxed_slice(),
        m,
        k,
        n,
        ids: None,
        soft_cap: None,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

// Encode one GEMV (dense, or a per-row B-row gather when `gather_indices` is set) and copy out.
fn run_gemv<B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    a: impl BufferRef<Backend = B>,
    b: MatmulB<impl BufferRef<Backend = B>>,
    gather_indices: Option<impl BufferRef<Backend = B>>,
    m: usize,
    n_out: usize,
    k: usize,
    soft_cap: Option<f32>,
) -> Vec<T> {
    let mut d = create_buffer::<B, T>(context, m * n_out);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context, T::data_type(), T::data_type(), T::data_type())
            .expect("MatmulKernel");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: a,
                    offset: 0,
                },
                b,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut d,
                d_transform: MatmulDOps {
                    soft_cap,
                    ..MatmulDOps::none()
                },
                gather_indices,
                m: m as u32,
                n: n_out as u32,
                k: k as u32,
            },
            &mut command_buffer,
        )
        .expect("encode failed");
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<B, T>(&d)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let a = create_buffer_with_data::<B, T>(&context, &input.a);
    let weights = create_buffer_with_data::<B, T>(&context, &input.b);
    let ids = input.ids.as_ref().map(|ids| create_buffer_with_data::<B, u32>(&context, ids));
    run_gemv::<B, T>(
        &context,
        &a,
        MatmulB::FullPrecision {
            b: &weights,
        },
        ids.as_ref(),
        input.m,
        input.n,
        input.k,
        input.soft_cap,
    )
}

fn test<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) {
    let (input, expected) = get_test_data::<T>(m, k, n);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float(&expected, &output, eps, &format!("backend {}", std::any::type_name::<B>()));
    });
}

#[rstest]
#[test_attr(stark_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::small_n(1, 128, 3)]
fn gemv_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<bf16>(m, k, n, 0.1);
}

// F16 operands run only as an all-F16 product with F32 accumulation.
#[rstest]
#[test_attr(stark_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::prefill_gemm(64, 256, 96)]
fn gemv_f16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<f16>(m, k, n, 0.02);
}

#[cfg(backend = "metal")]
#[rstest]
#[test_attr(stark_test)]
#[case::w4_zero_point(4, QuantizationMethod::ScaleZeroPoint)]
#[case::w8_bias(8, QuantizationMethod::ScaleBias)]
fn group_major_gemv_bf16(
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(1, 256, 72, 32, bits, method, 0).with_group_output();
    let reference = run_quant_cpu::<bf16>(&input);

    let buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let actual = run_gemv::<Metal, bf16>(
        &context,
        &buffers.x,
        buffers.matmul_b(&input),
        None::<&<Metal as Backend>::GlobalBuffer>,
        1,
        input.n as usize,
        input.k as usize,
        None,
    );

    assert_eq_float(&reference, &actual, 0.05, &format!("GroupOutput GEMV W{bits} {method:?}"));
}

#[cfg(backend = "metal")]
#[stark_test]
fn dense_qwen27_affine_verification_matches_independent_scalar_rows() {
    let context = MetalContext::new().expect("Metal context");
    // Full model matrix geometries exercise all relevant loop lengths without
    // loading model weights. Scalar rows remain the explicit baseline route.
    for bits in [4, 8] {
        for (n, k) in [(5120, 17408), (34816, 5120), (16480, 5120), (5120, 6144), (14336, 5120), (5120, 10240)] {
            let input =
                QuantInput::<bf16>::new(4, k, n, 64, bits, QuantizationMethod::ScaleBias, 37).with_group_output();
            let buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
            let row_bytes = k as usize * 2;
            let mut reference = Vec::with_capacity(4 * n as usize);
            for row in 0..4 {
                reference.extend(run_gemv::<Metal, bf16>(
                    &context,
                    buffers.x.subrange(row * row_bytes..(row + 1) * row_bytes),
                    buffers.matmul_b(&input),
                    None::<&<Metal as Backend>::GlobalBuffer>,
                    1,
                    n as usize,
                    k as usize,
                    None,
                ));
            }
            for m in 2..=4 {
                let actual = run_gemv::<Metal, bf16>(
                    &context,
                    &buffers.x,
                    buffers.matmul_b(&input),
                    None::<&<Metal as Backend>::GlobalBuffer>,
                    m,
                    n as usize,
                    k as usize,
                    None,
                );
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "affine verify differs from scalar rows: W{bits}, m={m}, n={n}, k={k}"
                );
            }
        }
    }
}

#[rstest]
#[test_attr(stark_test)]
#[case::m1(1, 128, 64)]
#[case::batched(4, 128, 64)]
#[case::max_batch(8, 128, 64)]
#[case::unaligned_k(1, 33, 64)]
#[case::unaligned_n(1, 128, 11)]
#[case::large(1, 4096, 2048)]
#[case::small_n(1, 128, 3)]
fn gemv_f32(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
) {
    test::<f32>(m, k, n, 0.01);
}

fn assert_gather<T: ArrayElement + Float + Debug + Display>(
    dense: &[T],
    gather: &[T],
    ids: &[u32],
    m: usize,
    vocab: usize,
    ids_per_row: usize,
    eps: f32,
    name: &str,
) {
    let mut expected = vec![T::from(0.0).unwrap(); m * ids_per_row];
    for r in 0..m {
        for c in 0..ids_per_row {
            expected[r * ids_per_row + c] = dense[r * vocab + ids[r * ids_per_row + c] as usize];
        }
    }
    assert_eq_float(&expected, gather, eps, &format!("gather vs dense ({name})"));
}

macro_rules! check_gather {
    ($m:expr, $vocab:expr, $ids:expr, $ids_per_row:expr, $eps:expr, |$B:ident| $run:block) => {{
        {
            #[allow(non_camel_case_types)]
            type $B = Cpu;
            let (dense, gather) = $run;
            assert_gather(&dense, &gather, &$ids, $m, $vocab, $ids_per_row, $eps, "Cpu");
        }
        for_each_non_cpu_backend!(|$B| {
            let (dense, gather) = $run;
            assert_gather(&dense, &gather, &$ids, $m, $vocab, $ids_per_row, $eps, std::any::type_name::<$B>());
        });
    }};
}

fn fp_gather_case<T: ArrayElement + Float + Debug + Display>(
    soft_cap: Option<f32>,
    eps: f32,
) {
    let (m, k, vocab, ids_per_row) = (4usize, 128usize, 256usize, 8usize);
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let weights: Vec<T> = (0..vocab * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();
    let ids: Vec<u32> = (0..m * ids_per_row).map(|i| ((i * 37 + 11) % vocab) as u32).collect();

    // Dense (`n = vocab`, no ids) and gather (`n = ids_per_row`, ids) share `a`/`weights`/soft-cap.
    let make = |n: usize, ids: Option<Box<[u32]>>| Input {
        a: a.clone().into_boxed_slice(),
        b: weights.clone().into_boxed_slice(),
        m,
        k,
        n,
        ids,
        soft_cap,
    };
    let dense_input = make(vocab, None);
    let gather_input = make(ids_per_row, Some(ids.clone().into_boxed_slice()));

    check_gather!(m, vocab, ids, ids_per_row, eps, |B| {
        (get_output::<T, B>(&dense_input), get_output::<T, B>(&gather_input))
    });
}

fn quant_gather_case(
    input: QuantInput<bf16>,
    eps: f32,
) {
    let (m, k, vocab, ids_per_row) = (input.m as usize, input.k as usize, input.n as usize, 8);
    let ids: Vec<u32> = (0..m * ids_per_row).map(|i| ((i * 37 + 11) % vocab) as u32).collect();
    // K_SPLIT == 1 keeps dense and gathered accumulation order identical.
    check_gather!(m, vocab, ids, ids_per_row, eps, |B| {
        let context = <B as Backend>::Context::new().expect("context");
        let buffers = QuantBuffers::<B, bf16>::allocate(&context, &input);
        let ids_alloc = create_buffer_with_data::<B, u32>(&context, &ids);
        let b = || buffers.matmul_b(&input);
        (
            run_gemv::<B, bf16>(&context, &buffers.x, b(), None::<&<B as Backend>::GlobalBuffer>, m, vocab, k, None),
            run_gemv::<B, bf16>(&context, &buffers.x, b(), Some(&ids_alloc), m, ids_per_row, k, None),
        )
    });
}

#[stark_test]
fn gemv_gather() {
    // Full precision: one call per dtype (generic over T, so bf16/f32 can't be a runtime loop).
    for soft_cap in [None, Some(15.0)] {
        fp_gather_case::<bf16>(soft_cap, 0.1);
        fp_gather_case::<f32>(soft_cap, 0.01);
    }
    // Quantized (bf16, per bits/method) — inline, since it isn't type-generic.
    for (bits, method, signed_codes) in [
        (4, QuantizationMethod::ScaleBias, false),
        (4, QuantizationMethod::ScaleZeroPoint, false),
        (4, QuantizationMethod::ScaleZeroPoint, true),
        (4, QuantizationMethod::ScaleSymmetric, false),
        (8, QuantizationMethod::ScaleZeroPoint, false),
    ] {
        let mut input = QuantInput::new(8, 128, 64, 32, bits, method, 0x5EED);
        if signed_codes {
            input = input.with_signed_weight_codes();
        }
        quant_gather_case(input, 0.05);
    }
    quant_gather_case(
        QuantInput::new(8, 96, 66, 32, 4, QuantizationMethod::ScaleZeroPoint, 0x5EED).with_group_output(),
        0.5,
    );
}
