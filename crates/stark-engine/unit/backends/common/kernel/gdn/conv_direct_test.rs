use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::{Conv1dPackKernel, DeltaNetConvDirectKernel, DeltaNetConvScanKernel},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
};

fn compare<B: Backend>(
    suffix: usize,
    has_bias: bool,
) {
    let context = B::Context::new().expect("context");
    let channels = 64;
    let stride = 96;
    let taps = 4;
    let history = taps - 1;
    let input: Vec<bf16> = (0..suffix * stride)
        .map(|i| {
            if i % 37 == 0 {
                bf16::NEG_ZERO
            } else {
                bf16::from_f32((i % 29) as f32 / 16.0 - 0.75)
            }
        })
        .collect();
    let state: Vec<f32> = (0..channels * history).map(|i| (i % 17) as f32 / 16.0 - 0.5).collect();
    let weights: Vec<f32> = (0..channels * taps).map(|i| (i % 11) as f32 / 16.0 - 0.25).collect();
    let bias: Vec<f32> = (0..channels).map(|i| (i % 7) as f32 / 16.0 - 0.125).collect();
    let original = create_buffer_with_data::<B, bf16>(&context, &input);
    let mut legacy = create_buffer_with_data::<B, bf16>(&context, &input);
    let mut direct = create_buffer::<B, bf16>(&context, input.len());
    let mut compact = create_buffer::<B, bf16>(&context, suffix * channels);
    let mut compact_state = create_buffer::<B, f32>(&context, state.len());
    let state_in = create_buffer_with_data::<B, f32>(&context, &state);
    let mut legacy_state = create_buffer::<B, f32>(&context, state.len());
    let mut direct_state = create_buffer_with_data::<B, f32>(&context, &state);
    let mut snapshot = create_buffer::<B, f32>(&context, state.len());
    let w = create_buffer_with_data::<B, f32>(&context, &weights);
    let b = create_buffer_with_data::<B, f32>(&context, &bias);
    let mut padded = create_buffer::<B, f32>(&context, (suffix + history) * stride);
    let pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(&context, DataType::F32, DataType::BF16).expect("pack");
    let scan = <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(&context, DataType::BF16, has_bias).expect("scan");
    let direct_scan = <B::Kernels as Kernels>::DeltaNetConvDirectKernel::new(&context, DataType::BF16, has_bias)
        .expect("direct scan");
    let mut command = context.create_command_buffer(None, None).expect("command");
    pack.encode(
        &state_in,
        &legacy,
        &mut padded,
        history as u32,
        stride as u32,
        suffix as u32,
        channels as u32,
        &mut command,
    );
    scan.encode(
        &padded,
        &w,
        has_bias.then_some(&b),
        &mut legacy,
        &mut legacy_state,
        suffix as u32,
        taps as u32,
        stride as u32,
        history as u32,
        channels as u32,
        stride as u32,
        &mut command,
    );
    // Match the production snapshot path, including write-after-read tracking.
    command.encode_copy(&direct_state, &mut snapshot);
    direct_scan.encode(
        &original,
        &snapshot,
        &w,
        has_bias.then_some(&b),
        &mut direct,
        &mut direct_state,
        suffix as u32,
        taps as u32,
        stride as u32,
        stride as u32,
        history as u32,
        channels as u32,
        &mut command,
    );
    direct_scan.encode(
        &original,
        &snapshot,
        &w,
        has_bias.then_some(&b),
        &mut compact,
        &mut compact_state,
        suffix as u32,
        taps as u32,
        stride as u32,
        channels as u32,
        history as u32,
        channels as u32,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().expect("complete");
    let legacy: Vec<bf16> = buffer_to_vec(&legacy);
    let direct: Vec<bf16> = buffer_to_vec(&direct);
    let compact: Vec<bf16> = buffer_to_vec(&compact);
    let compact_state: Vec<f32> = buffer_to_vec(&compact_state);
    let retained_original: Vec<bf16> = buffer_to_vec(&original);
    assert!(input.iter().zip(&retained_original).all(|(a, b)| a.to_bits() == b.to_bits()), "original changed");
    assert!(
        direct
            .chunks_exact(stride)
            .flat_map(|row| row[..channels].iter())
            .zip(&compact)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "compact QKV bits differ"
    );
    let legacy_state: Vec<f32> = buffer_to_vec(&legacy_state);
    let direct_state: Vec<f32> = buffer_to_vec(&direct_state);
    assert!(legacy.iter().zip(&direct).all(|(a, b)| a.to_bits() == b.to_bits()), "projection bits differ");
    assert!(
        direct_state.iter().zip(&compact_state).all(|(a, b)| a.to_bits() == b.to_bits()),
        "compact state bits differ"
    );
    assert!(legacy_state.iter().zip(&direct_state).all(|(a, b)| a.to_bits() == b.to_bits()), "state bits differ");
}

#[stark_test]
fn direct_delta_convolution_preserves_packed_bits() {
    for_each_backend!(|B| {
        for suffix in [1, 2, 3, 4, 63, 64, 255, 256, 257, 1024] {
            for has_bias in [false, true] {
                compare::<B>(suffix, has_bias);
            }
        }
    });
}
