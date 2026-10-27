use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::{
            DeltaNetConvUpdateKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel, ReferenceGDNNormGateKernel,
            ReferenceGDNQKL2Kernel,
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
};

const X: [f32; 8] = [0.1015625, 0.2109375, -0.3359375, 0.47265625, -0.6328125, 0.7734375, 1.125, -1.375];
const GAIN: [f32; 8] = [0.913, 0.997, 1.003, 1.117, 1.237, 0.887, 1.337, 1.501];
const Z: [f32; 8] = [-2.0, -1.0, -0.5, 0.125, 0.5, 1.0, 2.0, 3.0];
const KEY: usize = 128;
const VALUE: usize = 128;
const CONV: usize = 2 * KEY + VALUE;
const STRIDE: usize = CONV + VALUE + 2;

// Independent goldens from PyTorch 2.14.1 CPU, eight threads, using the pinned
// Qwen3.5 BF16 conv, sigmoid, recurrent rule and RMSNormGated dtype sequence.
// Inputs are synthetic. The non-BF16 F32 gains exercise source weight rounding.
fn assert_bits(
    actual: &[bf16],
    expected_period: &[u16],
) {
    assert!(actual.iter().enumerate().all(|(i, value)| value.to_bits() == expected_period[i % expected_period.len()]));
}

fn reference_norm<B: Backend>() {
    let context = B::Context::new().expect("context");
    let input = (0..VALUE).map(|i| bf16::from_f32(X[i % 8])).collect::<Vec<_>>();
    let gain = (0..VALUE).map(|i| GAIN[i % 8]).collect::<Vec<_>>();
    let mut controls = vec![bf16::ZERO; STRIDE];
    for i in 0..VALUE {
        controls[CONV + i] = bf16::from_f32(Z[i % 8]);
    }
    let mut output = create_buffer_with_data::<B, bf16>(&context, &input);
    let controls = create_buffer_with_data::<B, bf16>(&context, &controls);
    let gain = create_buffer_with_data::<B, f32>(&context, &gain);
    let kernel = <B::Kernels as Kernels>::ReferenceGDNNormGateKernel::new(&context).expect("norm");
    let mut command = context.create_command_buffer(None, None).expect("command");
    kernel.encode(&mut output, &controls, &gain, 1, 1, 128, CONV as u32, STRIDE as u32, 1e-6, &mut command);
    command.end_encoding().submit().wait_until_completed().expect("complete");
    assert_bits(&buffer_to_vec::<B, bf16>(&output), &[48368, 48537, 15788, 15679, 48806, 16170, 16481, 49402]);
}

fn reference_conv<B: Backend>() {
    let context = B::Context::new().expect("context");
    let input = (0..32).map(|i| bf16::from_f32(X[i % 8])).collect::<Vec<_>>();
    let old = [0.5, -0.25, 0.75, -1.0, 1.25, -1.5, 0.375, -0.625];
    let state = (0..32).map(|i| old[i % 8]).collect::<Vec<_>>();
    let weights = [0.1015625, -0.3359375].repeat(32);
    let mut input = create_buffer_with_data::<B, bf16>(&context, &input);
    let mut state = create_buffer_with_data::<B, f32>(&context, &state);
    let weights = create_buffer_with_data::<B, f32>(&context, &weights);
    let kernel = <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(&context, DataType::BF16, false).expect("conv");
    let mut command = context.create_command_buffer(None, None).expect("command");
    kernel.encode(&weights, None::<&B::GlobalBuffer>, &mut input, &mut state, 2, 32, 1, 1, &mut command);
    command.end_encoding().submit().wait_until_completed().expect("complete");
    assert_bits(&buffer_to_vec::<B, bf16>(&input), &[15369, 48444, 15828, 48616, 15947, 48680, 48657, 15988]);
}

fn reference_update<B: Backend>() {
    let context = B::Context::new().expect("context");
    let initial = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 0.125, -0.375];
    let mut input = vec![bf16::ZERO; STRIDE];
    input[0] = bf16::ONE;
    input[KEY] = bf16::ONE;
    for i in 0..VALUE {
        input[2 * KEY + i] = bf16::from_f32(X[i % 8]);
        input[CONV + i] = bf16::from_f32(Z[i % 8]);
    }
    input[CONV + VALUE] = bf16::from_f32(0.3);
    let mut state = vec![0.0; VALUE * KEY];
    for i in 0..VALUE {
        state[i * KEY] = initial[i % 8];
    }
    let gain = (0..VALUE).map(|i| GAIN[i % 8]).collect::<Vec<_>>();
    let input = create_buffer_with_data::<B, bf16>(&context, &input);
    let gain = create_buffer_with_data::<B, f32>(&context, &gain);
    let zero = create_buffer_with_data::<B, f32>(&context, &[0.0]);
    let mut state = create_buffer_with_data::<B, f32>(&context, &state);
    let mut output = create_buffer::<B, bf16>(&context, VALUE);
    let kernel =
        <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(&context, DataType::BF16, KEY as u32).expect("update");
    let mut command = context.create_command_buffer(None, None).expect("command");
    kernel.encode(&input, &zero, &zero, &gain, &mut state, &mut output, 1, 1, 128, 128, 128, 1e-6, 1, &mut command);
    command.end_encoding().submit().wait_until_completed().expect("complete");
    assert_bits(&buffer_to_vec::<B, bf16>(&output), &[48510, 48162, 15491, 15412, 48578, 15955, 16513, 49430]);
    let expected = [0.11154179, 0.014678776, -0.033233345, 0.0585171, -0.097259045, 0.12478578, 0.6726071, -0.8693845];
    let state = buffer_to_vec::<B, f32>(&state);
    for i in 0..VALUE {
        assert!((state[i * KEY] - expected[i % 8]).abs() < 2e-7, "recurrent state precision changed");
        assert!(state[i * KEY + 1..(i + 1) * KEY].iter().all(|value| *value == 0.0));
    }
}

fn reference_beta<B: Backend>() {
    let context = B::Context::new().expect("context");
    let beta_input = [-2.0, -1.0, -0.5, 0.125, 0.3, 1.0, 2.0, 3.0];
    let mut input = vec![bf16::ZERO; STRIDE * 8];
    for token in 0..8 {
        input[token * STRIDE] = bf16::ONE;
        input[token * STRIDE + KEY] = bf16::ONE;
        input[token * STRIDE + CONV + VALUE] = bf16::from_f32(beta_input[token]);
    }
    let input = create_buffer_with_data::<B, bf16>(&context, &input);
    let zero = create_buffer_with_data::<B, f32>(&context, &[0.0]);
    let mut q = create_buffer::<B, f32>(&context, 8 * KEY);
    let mut k = create_buffer::<B, f32>(&context, 8 * KEY);
    let mut beta = create_buffer::<B, f32>(&context, 8);
    let mut decay = create_buffer::<B, f32>(&context, 8);
    let kernel = <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        128,
        false,
        false,
    )
    .expect("prep");
    let mut command = context.create_command_buffer(None, None).expect("command");
    kernel.encode(
        &input,
        &input,
        &zero,
        &zero,
        &mut q,
        &mut k,
        None::<&mut B::GlobalBuffer>,
        &mut beta,
        &mut decay,
        1,
        1,
        128,
        128,
        8,
        STRIDE as u32,
        1,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().expect("complete");
    let expected = [15860, 16010, 16065, 16136, 16147, 16187, 16225, 16244];
    for (actual, expected) in buffer_to_vec::<B, f32>(&beta).into_iter().zip(expected) {
        assert_eq!(actual, bf16::from_bits(expected).to_f32());
    }
}

fn reference_l2<B: Backend>() {
    let context = B::Context::new().expect("context");
    let mut input = vec![bf16::ZERO; STRIDE];
    for i in 0..128 {
        input[i] = bf16::from_f32(X[i % 8]);
        input[128 + i] = bf16::from_f32(X[7 - i % 8]);
    }
    let input = create_buffer_with_data::<B, bf16>(&context, &input);
    let mut q = create_buffer::<B, f32>(&context, 128);
    let mut k = create_buffer::<B, f32>(&context, 128);
    let kernel = <B::Kernels as Kernels>::ReferenceGDNQKL2Kernel::new(&context).expect("reference L2");
    let mut commands = context.create_command_buffer(None, None).expect("commands");
    kernel.encode(&input, &mut q, &mut k, 128, 1, STRIDE as u32, &mut commands);
    commands.end_encoding().submit().wait_until_completed().expect("complete");
    let expected_q = [982120116, 990856478, -1151063116, 1000375284, -1143543891, 1006851249, 1010763475, -1133938146];
    let expected_k =
        [-1104865317, 1040651141, 1035582166, -1114114571, 1029898445, -1121888432, 1019911495, 1011031089];
    for (actual, expected) in [(buffer_to_vec::<B, f32>(&q), expected_q), (buffer_to_vec::<B, f32>(&k), expected_k)] {
        assert!(actual.iter().enumerate().all(|(i, value)| value.to_bits() as i32 == expected[i % 8]));
    }
}

#[stark_test]
fn delta_net_reference_bf16_matches_source_boundaries() {
    for_each_backend!(|B| {
        reference_norm::<B>();
        reference_conv::<B>();
        reference_update::<B>();
        reference_beta::<B>();
        reference_l2::<B>();
    });
}
