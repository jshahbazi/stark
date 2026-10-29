use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use stark_engine_macros::stark_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{ActivationTransform, NormalizationKernel, ReferenceRMSBF16Kernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend, for_each_non_cpu_backend},
    },
};

fn get_output<
    B: Backend,
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
>(
    input: &[InputT],
    scales: Option<&[AffineT]>,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    full_layer: bool,
    hadamard_factors: Option<&[i32]>,
) -> Vec<OutputT> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::NormalizationKernel::new(
        &context,
        InputT::data_type(),
        AffineT::data_type(),
        OutputT::data_type(),
        DataType::F32,
        false,
        false,
        full_layer,
        false,
        false,
        hadamard_factors.is_some(),
        false,
        false,
        false,
        scales.is_some(),
        false,
    )
    .expect("Failed to create NormalizationKernel");

    let input_buffer = create_buffer_with_data::<B, InputT>(&context, input);
    let scales_buffer = scales.map(|scales| create_buffer_with_data::<B, AffineT>(&context, scales));
    let hadamard_factors_buffer =
        hadamard_factors.map(|hadamard_factors| create_buffer_with_data::<B, i32>(&context, hadamard_factors));
    let mut output_buffer = create_buffer_with_data::<B, OutputT>(&context, &vec![OutputT::zero(); input.len()]);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        Some(&input_buffer),
        scales_buffer.as_ref(),
        None::<&B::GlobalBuffer>,
        &mut output_buffer,
        None::<&mut B::GlobalBuffer>,
        hadamard_factors_buffer.as_ref(),
        batch_size,
        element_count,
        epsilon,
        0.0,
        1.0,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    buffer_to_vec::<B, OutputT>(&output_buffer)
}

fn test_internal<
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
>(
    has_scales: bool,
    full_layer: bool,
) {
    let batch_size = 2u32;
    let element_count = 64u32;
    let epsilon = 1e-6f32;

    let input: Vec<InputT> =
        (0..(batch_size * element_count)).map(|index| InputT::from(0.5f32 + (index as f32) * 0.01).unwrap()).collect();
    let scales: Vec<AffineT> =
        (0..element_count).map(|index| AffineT::from(1.0f32 + (index as f32) * 0.001).unwrap()).collect();
    let scales = has_scales.then_some(scales);

    let expected = get_output::<Cpu, InputT, AffineT, OutputT>(
        &input,
        scales.as_deref(),
        batch_size,
        element_count,
        epsilon,
        full_layer,
        None,
    );

    let eps = if matches!(InputT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(AffineT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(OutputT::data_type(), DataType::F16 | DataType::BF16)
    {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<B, InputT, AffineT, OutputT>(
            &input,
            scales.as_deref(),
            batch_size,
            element_count,
            epsilon,
            full_layer,
            None,
        );
        let message = format!(
            "Normalization kernel test failed with backend={}, has_scales={}, full_layer={}",
            std::any::type_name::<B>(),
            has_scales,
            full_layer,
        );
        assert_eq_float::<OutputT>(&expected, &actual, eps, &message);
    });
}

fn test_normalization<
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
>() {
    for has_scales in [true, false] {
        for full_layer in [true, false] {
            test_internal::<InputT, AffineT, OutputT>(has_scales, full_layer);
        }
    }
}

// The fused transform has to match plain normalization followed by the standalone input RHT
fn test_hadamard<T: ArrayElement + Float + Debug + Display>() {
    let batch_size = 2u32;
    let element_count = 64u32;
    let epsilon = 1e-6f32;

    let input: Vec<T> =
        (0..(batch_size * element_count)).map(|index| T::from(0.5f32 + (index as f32) * 0.01).unwrap()).collect();
    let scales: Vec<T> = (0..element_count).map(|index| T::from(1.0f32 + (index as f32) * 0.001).unwrap()).collect();
    let hadamard_factors: Vec<i32> = (0..element_count)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();

    let plain = get_output::<Cpu, T, T, T>(&input, Some(&scales), batch_size, element_count, epsilon, true, None);

    let context = <Cpu as Backend>::Context::new().expect("Failed to create Context");
    let input_rht = ActivationTransform::<Cpu>::input_rht(context.as_ref(), T::data_type(), false)
        .expect("Failed to create ActivationTransform");
    let plain_buffer = create_buffer_with_data::<Cpu, T>(&context, &plain);
    let hadamard_factors_buffer = create_buffer_with_data::<Cpu, i32>(&context, &hadamard_factors);
    let mut expected_buffer = create_buffer::<Cpu, T>(&context, plain.len());
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    input_rht.encode_fp(
        &plain_buffer,
        &mut expected_buffer,
        &hadamard_factors_buffer,
        batch_size,
        element_count,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    let expected = buffer_to_vec::<Cpu, T>(&expected_buffer);

    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_backend!(|B| {
        let actual = get_output::<B, T, T, T>(
            &input,
            Some(&scales),
            batch_size,
            element_count,
            epsilon,
            true,
            Some(&hadamard_factors),
        );
        let message = format!("Normalization hadamard kernel test failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps, &message);
    });
}

#[stark_test]
fn test_normalization_hadamard_f32() {
    test_hadamard::<f32>();
}

#[stark_test]
fn test_normalization_hadamard_bf16() {
    test_hadamard::<bf16>();
}

#[stark_test]
fn test_normalization_f32_f32_f32() {
    test_normalization::<f32, f32, f32>();
}

#[stark_test]
fn test_normalization_bf16_bf16_bf16() {
    test_normalization::<bf16, bf16, bf16>();
}

fn public_rms_value(index: usize) -> bf16 {
    bf16::from_bits(
        (0x3000 + ((index * 7919) % 0x1800)) as u16
            | if index % 3 == 0 {
                0x8000
            } else {
                0
            },
    )
}

fn reference_rms<B: Backend>() {
    use sha2::{Digest, Sha256};
    let context = B::Context::new().unwrap();
    for (width, packed, residual, expected) in [
        (256usize, true, false, "51eb2c10762e9c046e913a8d6e7f2b9dc8a9fc1b62fd0bb6d080d246c405967e"),
        (1024, false, false, "e1c9f60636227287f2fe8a5b8a7953e1d1d3bc70e1701bb41c07bcebb10933ba"),
        (1024, false, true, "1162e58882e9015a29dcf23bb39cd4a70c75a960687b9a710149b7c9759629b1"),
    ] {
        let values = (0..2 * width).map(public_rms_value).collect::<Vec<_>>();
        let scales = (0..width).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect::<Vec<_>>();
        let input = create_buffer_with_data::<B, bf16>(&context, &values);
        let scales = create_buffer_with_data::<B, f32>(&context, &scales);
        let initial = if packed {
            let mut data = vec![bf16::from_f32(-3.0); 4 * width];
            data[width..3 * width].copy_from_slice(&values);
            data
        } else {
            vec![bf16::ZERO; 2 * width]
        };
        let mut output = create_buffer_with_data::<B, bf16>(&context, &initial);
        let skip = (0..2 * width)
            .map(|i| {
                if residual {
                    public_rms_value(i + 13)
                } else {
                    bf16::from_f32(-7.0)
                }
            })
            .collect::<Vec<_>>();
        let mut shortcut = create_buffer_with_data::<B, bf16>(&context, &skip);
        let kernel = <B::Kernels as Kernels>::ReferenceRMSBF16Kernel::new(&context, packed, !packed, residual).unwrap();
        let mut command = context.create_command_buffer(None, None).unwrap();
        kernel.encode(
            (!packed).then_some(&input),
            &scales,
            &mut output,
            (!packed).then_some(&mut shortcut),
            if packed {
                1
            } else {
                2
            },
            width as u32,
            if packed {
                4 * width as u32
            } else {
                width as u32
            },
            1e-6,
            1.0,
            u32::from(packed),
            if packed {
                2
            } else {
                1
            },
            &mut command,
        );
        command.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&output);
        let actual = if packed {
            assert_eq!(&actual[..width], &initial[..width]);
            assert_eq!(&actual[3 * width..], &initial[3 * width..]);
            &actual[width..3 * width]
        } else {
            let copied = buffer_to_vec::<B, bf16>(&shortcut);
            let expected_skip = values
                .iter()
                .zip(&skip)
                .map(|(x, s)| {
                    if residual {
                        bf16::from_f32(x.to_f32() + s.to_f32())
                    } else {
                        *x
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(copied, expected_skip);
            &actual[..]
        };
        // Golden hashes are independent pinned Torch BF16 forward results on
        // the public integer-bit-pattern inputs, using eight CPU threads.
        let bytes = actual.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
    }
}

#[stark_test]
fn test_reference_rms_bf16_source_cascade_packed_and_residual() {
    reference_rms::<Cpu>();
    for_each_non_cpu_backend!(|B| reference_rms::<B>());
}

fn source_rms_boundary<B: Backend>() {
    // Exact public powers, not model inputs. Source FP32 hierarchy gives
    // sum=1526.50244140625; the legacy Metal hierarchy gives1526.5023193359375.
    // The probe161/128 lies across the BF16 normalization midpoint, so output
    // bits16289/16290 distinguish the reduction policies at equal epsilon.
    let powers = [
        2, 2, 2, 1, 0, -1, -1, -1, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6, -7, -8, -8, -11, -11, -11, -12, -12, -13,
        -13, -13,
    ];
    let mut input = vec![bf16::ONE; 1536];
    for i in 0..64 {
        input[(i * 37 + 102) % 1536] = bf16::ZERO;
    }
    for (i, power) in powers.into_iter().enumerate() {
        input[(i * 37 + 102) % 1536] = bf16::from_f32(2.0f32.powi(power));
    }
    input[1535] = bf16::from_f32(161.0 / 128.0);
    let context = B::Context::new().unwrap();
    let input = create_buffer_with_data::<B, bf16>(&context, &input);
    for (source, expected) in [(false, 16290u16), (true, 16289), (false, 16290)] {
        let kernel = <<B as Backend>::Kernels as Kernels>::NormalizationKernel::new(
            &context,
            DataType::BF16,
            DataType::F32,
            DataType::BF16,
            DataType::F32,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            source,
        )
        .unwrap();
        let mut output = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ZERO; 1536]);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        kernel.encode(
            Some(&input),
            None::<&B::GlobalBuffer>,
            None::<&B::GlobalBuffer>,
            &mut output,
            None::<&mut B::GlobalBuffer>,
            None::<&B::GlobalBuffer>,
            1,
            1536,
            1e-6,
            0.0,
            1.0,
            &mut commands,
        );
        commands.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(buffer_to_vec::<B, bf16>(&output)[1535].to_bits(), expected, "source RMS boundary");
    }
}

#[stark_test]
fn mlx_bf16_rms_public_midpoint_distinguishes_reduction_and_legacy_default() {
    source_rms_boundary::<Cpu>();
    for_each_non_cpu_backend!(|B| source_rms_boundary::<B>());
}
