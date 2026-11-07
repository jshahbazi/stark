use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::QuantizationMode,
        kernel::matmul::{
            MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, QuantParams, QuantParamsLayout, QuantizedB,
            QuantizedCorrection,
        },
    },
    config::weight_matrix::mlx_spec::MLXSpec,
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
};

#[derive(Clone, Copy)]
enum Scenario {
    Quad,
    Fold,
    GenericOutputTail,
    GenericInputTail,
    GroupedPartial,
}

fn output<B: Backend>(
    enabled: bool,
    invalid_dtype: bool,
    invalid_bits: bool,
    scenario: Scenario,
) -> Result<Vec<bf16>, ()> {
    let context = B::Context::new().map_err(|_| ())?;
    let mut kernel = <B::Kernels as Kernels>::MatmulKernel::new(
        &context,
        DataType::BF16,
        if invalid_dtype {
            DataType::F32
        } else {
            DataType::BF16
        },
        DataType::BF16,
    )
    .map_err(|_| ())?;
    let (k, n) = match scenario {
        Scenario::Quad => (64, 32),
        Scenario::Fold => (1024, 32),
        Scenario::GenericOutputTail => (1024, 33),
        Scenario::GenericInputTail => (768, 32),
        Scenario::GroupedPartial => (512, 33),
    };
    let groups = k / 64;
    let mut input = vec![bf16::ZERO; k];
    let mut codes = vec![0u8; k * n / 2];
    let mut biases = vec![bf16::ZERO; groups * n];
    match scenario {
        Scenario::Quad => {
            let quad = [1.0f32, 1.0 / 256.0, -1.0, 1.0 / 512.0];
            input = quad.into_iter().cycle().take(k).map(bf16::from_f32).collect();
            biases.fill(bf16::from_f32(0.5));
        },
        Scenario::Fold => {
            input[0] = bf16::ONE;
            input[512] = bf16::from_f32(33_554_432.0);
            for row in 0..n {
                codes[row * k / 2] = 1;
                codes[row * k / 2 + 512 / 2] = 1;
                biases[row * groups + 8] = -bf16::ONE;
            }
        },
        Scenario::GenericOutputTail | Scenario::GenericInputTail => {
            input[0] = bf16::from_f32(33_554_432.0);
            input[256] = bf16::from_f32(-33_554_432.0);
            input[512] = bf16::ONE;
            for row in 0..n {
                for index in [0, 256, 512] {
                    codes[row * k / 2 + index / 2] = 1;
                }
            }
        },
        Scenario::GroupedPartial => {
            input[0] = bf16::from_f32(33_554_432.0);
            input[1] = bf16::ONE;
            input[4] = bf16::from_f32(-33_554_432.0);
            input[5] = bf16::ONE;
            for row in 0..n {
                codes[row * k / 2] = 0x11;
                codes[row * k / 2 + 2] = 0x11;
            }
        },
    }
    let a = create_buffer_with_data::<B, bf16>(&context, &input);
    let codes = create_buffer_with_data::<B, u8>(&context, &codes);
    let scales = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ONE; groups * n]);
    let biases = create_buffer_with_data::<B, bf16>(&context, &biases);
    let mut destination = create_buffer::<B, bf16>(&context, n);
    let mut commands = context.create_command_buffer(None, None).map_err(|_| ())?;
    let params = QuantParams::new(QuantParamsLayout::OutputGroup, n as u32, groups as u32)
        .with_materialized_affine_input_sum(enabled);
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a,
                    offset: 0,
                },
                b: MatmulB::Quantized(QuantizedB {
                    codes: &codes,
                    scales: &scales,
                    correction: QuantizedCorrection::Biases(&biases),
                    params,
                    mode: if invalid_bits {
                        QuantizationMode::U8
                    } else {
                        QuantizationMode::U4
                    },
                    group_size: 64,
                    signed_codes: false,
                }),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut destination,
                d_transform: MatmulDOps::none(),
                gather_indices: None::<&B::GlobalBuffer>,
                m: 1,
                n: n as u32,
                k: k as u32,
            },
            &mut commands,
        )
        .map_err(|_| ())?;
    commands.end_encoding().submit().wait_until_completed().map_err(|_| ())?;
    Ok(buffer_to_vec::<B, bf16>(&destination))
}

#[stark_test]
fn materialized_affine_input_sum_has_source_bf16_quad_boundaries_and_legacy_default() {
    // Each source quad rounds 1+2^-8 to 1 before cancellation, yielding 2^-9.
    // Float accumulation yields 3 * 2^-9. The exact outputs differ by a factor of 3.
    for_each_backend!(|B| {
        let legacy = output::<B>(false, false, false, Scenario::Quad).unwrap();
        let materialized = output::<B>(true, false, false, Scenario::Quad).unwrap();
        assert!(legacy.iter().all(|value| value.to_f32() == 0.046875));
        assert!(
            materialized.iter().all(|value| value.to_f32() == 0.015625),
            "materialized first public synthetic value {}",
            materialized[0].to_f32()
        );
        assert!(output::<B>(true, true, false, Scenario::Quad).is_err());
        assert!(output::<B>(true, false, true, Scenario::Quad).is_err());
    });
}

#[stark_test]
fn materialized_affine_fold_preserves_prior_before_canceling_large_contribution() {
    // Block zero contributes one. In block one, the scaled dot and bias
    // correction are exactly +2^25 and -2^25. Source qdot cancels these before
    // adding the prior; folding the prior into the first FMA loses that one.
    for_each_backend!(|B| {
        let materialized = output::<B>(true, false, false, Scenario::Fold).unwrap();
        assert!(materialized.iter().all(|value| value.to_f32() == 1.0));
        if B::NAME == "metal" {
            let legacy = output::<B>(false, false, false, Scenario::Fold).unwrap();
            assert!(legacy.iter().all(|value| value.to_f32() == 0.0));
        }
    });
}

#[stark_test]
fn materialized_affine_generic_lane_geometry_preserves_source_reduction_order() {
    // MLX uses eight values per lane whenever N has a tail or K is not a
    // multiple of 512. Three successive lane-zero contributions cancel 2^25
    // before adding one. A sixteen-value lane distribution loses that one.
    for_each_backend!(|B| {
        for scenario in [Scenario::GenericOutputTail, Scenario::GenericInputTail] {
            let materialized = output::<B>(true, false, false, scenario).unwrap();
            assert!(materialized.iter().all(|value| value.to_f32() == 1.0));
            if B::NAME == "metal" {
                let legacy = output::<B>(false, false, false, scenario).unwrap();
                assert!(legacy.iter().all(|value| value.to_f32() == 0.0));
            }
        }
    });
}

#[stark_test]
fn materialized_affine_dot_groups_four_products_before_accumulating() {
    // Each four-product subtotal rounds away its local one. Combining those
    // subtotals gives zero; a single FMA chain across both quads yields one.
    for_each_backend!(|B| {
        let materialized = output::<B>(true, false, false, Scenario::GroupedPartial).unwrap();
        assert!(materialized.iter().all(|value| value.to_f32() == 0.0));
        if B::NAME == "metal" {
            let legacy = output::<B>(false, false, false, Scenario::GroupedPartial).unwrap();
            assert!(legacy.iter().all(|value| value.to_f32() == 1.0));
        }
    });
}

#[stark_test]
fn mlx_affine_input_sum_policy_and_quant_params_default_to_legacy() {
    let spec: MLXSpec = serde_json::from_value(serde_json::json!({"type":"MLXSpec","bits":4,
        "group_size":64,"layout":"output_input"}))
    .unwrap();
    assert!(!spec.materialize_affine_input_sum);
    let parameters = QuantParams::new(QuantParamsLayout::OutputGroup, 32, 1);
    assert!(!parameters.materializes_affine_input_sum());
    assert!(parameters.source_output_widths().is_empty());
    let materialized = parameters.with_materialized_affine_input_sum(true);
    assert!(materialized.materializes_affine_input_sum());
    assert_eq!(materialized.scale_shape(), parameters.scale_shape());
    assert_eq!(materialized.scale_strides().output_stride, parameters.scale_strides().output_stride);
}

#[stark_test]
fn source_output_widths_preserve_concat_geometry_without_changing_quantized_storage() {
    let original = QuantParams::new(QuantParamsLayout::GroupOutput, 2560, 24).with_materialized_affine_input_sum(true);
    let qkv = original.with_source_output_widths(&[2048, 256, 256]).unwrap();
    assert_eq!(qkv.source_output_widths(), &[2048, 256, 256]);
    assert_eq!(qkv.scale_shape(), original.scale_shape());
    assert_eq!(qkv.scale_strides().output_stride, original.scale_strides().output_stride);
    assert_eq!(qkv.scale_strides().group_stride, original.scale_strides().group_stride);
    assert!(qkv.materializes_affine_input_sum());
    let copied = qkv;
    assert_eq!(copied.source_output_widths(), &[2048, 256, 256]);
    let merged = copied.with_source_output_widths(&[2560]).unwrap();
    assert_eq!(merged.source_output_widths(), &[2560]);
    let up_gate =
        QuantParams::new(QuantParamsLayout::OutputGroup, 12288, 24).with_source_output_widths(&[6144, 6144]).unwrap();
    assert_eq!(up_gate.source_output_widths(), &[6144, 6144]);
}

#[stark_test]
fn source_output_widths_reject_invalid_or_unbounded_spans() {
    let parameters = QuantParams::new(QuantParamsLayout::OutputGroup, 16, 1);
    for widths in [&[][..], &[0, 16][..], &[8, 7][..], &[4, 4, 4, 2, 2][..], &[u32::MAX, 17][..]] {
        assert!(parameters.with_source_output_widths(widths).is_err());
    }
    assert!(parameters.source_output_widths().is_empty());
    let maximum = parameters.with_source_output_widths(&[4, 4, 4, 4]).unwrap();
    assert_eq!(maximum.source_output_widths(), &[4, 4, 4, 4]);
}
