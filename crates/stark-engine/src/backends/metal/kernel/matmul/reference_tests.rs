use half::bf16;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{
        CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::{QuantizationMode, gemm::GemmDTransform},
        kernel::matmul::{MatmulA, MatmulB, MatmulDOps, QuantParams, QuantizedB, QuantizedCorrection},
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

#[stark_test]
fn reference_affine_accumulation_avoids_bf16_partition_rounding() {
    let context = <Metal as Backend>::Context::new().unwrap();
    let (m, n, k) = (83_u32, 128_u32, 1024_u32);
    let shape = MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleBiasDequant,
        b_bits: Some(4),
        b_group_size: Some(64),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::GroupOutput),
        d_transform: GemmDTransform::empty(),
    };
    let mut ordinary = MatmulMetalKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();
    let MatmulDispatch::Gemm(ordinary_plan) = ordinary.select_dispatch(&shape, &context) else {
        panic!("cancellation fixture must select production GEMM");
    };
    assert!(ordinary_plan.split_k > 1, "cancellation fixture must expose partition rounding");
    let mut reference = MatmulMetalKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();
    reference.set_reference_accumulation();
    let MatmulDispatch::Gemm(reference_plan) = reference.select_dispatch(&shape, &context) else {
        panic!("reference fixture must retain production GEMM");
    };
    assert_eq!(reference_plan.engine, ordinary_plan.engine);
    assert_eq!(reference_plan.tiling, ordinary_plan.tiling);
    assert_eq!(reference_plan.split_k, 1);
    assert_eq!(reference.select_activation_format(&shape, &context), ActivationFormat::Bf16);

    // Each partition casts its partial to BF16: 256+1 becomes 256 before
    // another partition contributes -256. One F32 contraction preserves 1.
    let mut values = vec![bf16::ZERO; (m * k) as usize];
    for row in values.chunks_exact_mut(k as usize) {
        row[0] = bf16::from_f32(256.0);
        row[1] = bf16::ONE;
        row[512] = bf16::from_f32(-256.0);
    }
    let values = create_buffer_with_data::<Metal, bf16>(&context, &values);
    let codes = create_buffer_with_data::<Metal, u32>(&context, &vec![0x1111_1111; (n * k / 8) as usize]);
    let scales = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ONE; (n * k / 64) as usize]);
    let biases = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; (n * k / 64) as usize]);
    let mut observed = Vec::new();
    for kernel in [&mut ordinary, &mut reference] {
        let mut output = context.create_buffer((m * n * 2) as usize).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        kernel
            .encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &values,
                        offset: 0,
                    },
                    b: MatmulB::Quantized(QuantizedB {
                        codes: &codes,
                        scales: &scales,
                        correction: QuantizedCorrection::Biases(&biases),
                        params: QuantParams::new(QuantParamsLayout::GroupOutput, n, k / 64),
                        mode: QuantizationMode::U4,
                        group_size: 64,
                        signed_codes: false,
                    }),
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
                    m,
                    n,
                    k,
                },
                &mut commands,
            )
            .unwrap();
        commands.end_encoding().submit().wait_until_completed().unwrap();
        observed.push(buffer_to_vec::<Metal, bf16>(&output));
    }
    assert!(observed[0].iter().all(|value| *value == bf16::ZERO));
    assert!(observed[1].iter().all(|value| *value == bf16::ONE));
}
