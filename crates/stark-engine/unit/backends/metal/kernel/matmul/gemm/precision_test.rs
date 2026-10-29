use half::bf16;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::{
        common::{
            CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::QuantizationMode,
            kernel::matmul::{QuantParams, QuantizedCorrection},
        },
        metal::MetalContext,
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
};

#[stark_test]
fn affine_source_split_k_preserves_bf16_small_reduction_and_legacy_cache() {
    let context = MetalContext::new().expect("Metal context");
    let (m, n, k) = (18usize, 64usize, 1536usize);
    let groups = k / 64;
    let mut input = vec![bf16::ZERO; m * k];
    for row in input.chunks_exact_mut(k) {
        row[0] = bf16::ONE;
        row[1024] = bf16::from_f32(1.0 / 256.0);
        row[128] = -bf16::ONE;
        row[1152] = bf16::from_f32(1.0 / 256.0);
    }
    let input = create_buffer_with_data::<Metal, bf16>(&context, &input);
    let codes = create_buffer_with_data::<Metal, u8>(&context, &vec![0u8; n * k / 2]);
    let scales = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; n * groups]);
    let biases = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ONE; n * groups]);
    let mut destination = create_buffer::<Metal, bf16>(&context, m * n);
    let mut kernel = GemmKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
    let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();

    // Exact FP64 dot is 2^-7. Source reduction first rounds the strided
    // partition pair 1+2^-8 to one, while -1+2^-8 is representable.
    // Combining those BF16 totals yields 2^-8. Legacy reduction sums
    // the BF16 partials in FP32. Reusing one kernel checks cache identity.
    let mut engines = vec![GemmEngine::Simdgroup];
    if context.supports_mxu {
        engines.push(GemmEngine::Mxu);
    }
    for engine in engines {
        for (source, expected) in [(false, 1.0 / 128.0f64), (true, 1.0 / 256.0), (false, 1.0 / 128.0)] {
            let mut commands = context.create_command_buffer(None, None).unwrap();
            let arguments = MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::Quantized(QuantizedB {
                    codes: &codes,
                    scales: &scales,
                    correction: QuantizedCorrection::Biases(&biases),
                    params: QuantParams::new(QuantParamsLayout::OutputGroup, n as u32, groups as u32)
                        .with_materialized_affine_input_sum(source),
                    mode: QuantizationMode::U4,
                    group_size: 64,
                    signed_codes: false,
                }),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut destination,
                d_transform: MatmulDOps::none(),
                gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
                m: m as u32,
                n: n as u32,
                k: k as u32,
            };
            let shape = MatmulShape::from_arguments(&arguments);
            let mut plan = kernel.select_plan_for_engine(&shape, engine, &context).unwrap();
            plan.split_k = 12;
            kernel.encode_plan(arguments, plan, &output_work, &mut commands).unwrap();
            commands.end_encoding().submit().wait_until_completed().unwrap();
            let actual = buffer_to_vec::<Metal, bf16>(&destination);
            assert!(actual.iter().all(|value| f64::from(value.to_f32()) == expected));
        }
    }
}
