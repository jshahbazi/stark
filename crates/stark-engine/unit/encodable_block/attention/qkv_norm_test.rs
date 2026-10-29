use stark_engine_macros::stark_test;

use super::QKVNorm;
use crate::{
    backends::common::{Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_backend},
    },
};

fn run_key_value_row_stride_test<B: Backend>() {
    const BATCH_SIZE: u32 = 2;
    const NUM_KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 4;
    const ROW_WIDTH: u32 = 2 * NUM_KV_HEADS * HEAD_DIM;
    const EPSILON: f32 = 1e-6;

    let context = create_context::<B>();
    let config = NormalizationConfig {
        epsilon: EPSILON,
        scale_offset: None,
        upcast_mode: UpcastMode::OnlyNormalization,
        mlx_bf16_reduction: false,
        subtract_mean: false,
        has_scale: false,
        has_biases: false,
    };
    let key = QKVNorm::<B>::build_head(&context, DataType::F32, config, None, HEAD_DIM, false)
        .expect("failed to construct key norm");
    let norm = QKVNorm {
        query: None,
        key: Some(key),
        value: None,
        num_q_heads: 0,
        num_kv_heads: NUM_KV_HEADS,
        projection_row_stride: ROW_WIDTH,
        head_dim: HEAD_DIM,
        full_width: false,
    };

    let row_width = ROW_WIDTH as usize;
    let head_dim = HEAD_DIM as usize;
    let input = (0..BATCH_SIZE as usize * row_width).map(|index| 1.0 + index as f32 * 0.125).collect::<Vec<_>>();
    let mut expected = input.clone();
    for batch in 0..BATCH_SIZE as usize {
        for head in 0..NUM_KV_HEADS as usize {
            let start = batch * row_width + head * head_dim;
            let mean_square =
                input[start..start + head_dim].iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32;
            let inverse_rms = (mean_square + EPSILON).sqrt().recip();
            for index in start..start + head_dim {
                expected[index] *= inverse_rms;
            }
        }
    }

    let mut key_value = create_buffer_with_data::<B, f32>(&context, &input);
    let mut command_buffer = context.create_command_buffer(None, None).expect("failed to create command buffer");
    norm.encode_key_value(&mut key_value, BATCH_SIZE, &mut command_buffer).expect("failed to encode key/value norm");
    command_buffer.end_encoding().submit().wait_until_completed().expect("failed to execute key/value norm");

    let output = buffer_to_vec::<B, f32>(&key_value);
    assert_eq_float(&expected, &output, 1e-5, "key/value norm row stride mismatch");
}

#[stark_test]
fn test_key_value_norm_uses_element_row_stride() {
    for_each_backend!(|B| run_key_value_row_stride_test::<B>());
}

fn run_full_projection_norm_test<B: Backend>() {
    const Q: u32 = 4;
    const KV: u32 = 2;
    const DIM: u32 = 4;
    const BATCH: u32 = 3;
    const STRIDE: u32 = (Q + 2 * KV) * DIM + 8;
    let context = create_context::<B>();
    let config = NormalizationConfig {
        epsilon: 1e-6,
        scale_offset: None,
        upcast_mode: UpcastMode::FullLayer,
        mlx_bf16_reduction: false,
        subtract_mean: false,
        has_scale: false,
        has_biases: false,
    };
    let norm = QKVNorm {
        query: Some(QKVNorm::<B>::build_head(&context, DataType::F32, config.clone(), None, Q * DIM, false).unwrap()),
        key: Some(QKVNorm::<B>::build_head(&context, DataType::F32, config, None, KV * DIM, false).unwrap()),
        value: None,
        num_q_heads: Q,
        num_kv_heads: KV,
        projection_row_stride: STRIDE,
        head_dim: DIM,
        full_width: true,
    };
    let input = (0..BATCH * STRIDE).map(|index| (index % 47) as f32 * 0.125 - 2.0).collect::<Vec<_>>();
    let mut expected = input.clone();
    for row in 0..BATCH {
        for (offset, width) in [(0, Q * DIM), (Q * DIM, KV * DIM)] {
            let start = (row * STRIDE + offset) as usize;
            let end = start + width as usize;
            let rms =
                (input[start..end].iter().map(|value| value * value).sum::<f32>() / width as f32 + 1e-6).sqrt().recip();
            for value in &mut expected[start..end] {
                *value *= rms;
            }
        }
    }
    let mut actual = create_buffer_with_data::<B, f32>(&context, &input);
    let mut command = context.create_command_buffer(None, None).unwrap();
    norm.encode(&mut actual, BATCH, &mut command).unwrap();
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, f32>(&actual);
    assert_eq_float(&expected, &actual, 1e-5, "full-projection norm differs from independent RMS reference");
}

#[stark_test]
fn test_olmo_qk_norm_spans_projected_heads_and_preserves_value_and_padding() {
    for_each_backend!(|B| run_full_projection_norm_test::<B>());
}

#[cfg(feature = "attention-kv-only")]
fn run_projected_key_value_exact<B: Backend>() {
    use half::bf16;
    const Q_HEADS: u32 = 3;
    const KV_HEADS: u32 = 2;
    const DIM: u32 = 128;
    const STRIDE: u32 = (2 * Q_HEADS + 2 * KV_HEADS) * DIM;
    const BATCH: u32 = 9;
    let context = create_context::<B>();
    for upcast_mode in [UpcastMode::OnlyNormalization, UpcastMode::FullLayer] {
        let config = NormalizationConfig {
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode,
            subtract_mean: false,
            has_scale: false,
            has_biases: false,
        };
        let make_head = || {
            QKVNorm::<B>::build_head(&context, DataType::BF16, config.clone(), None, DIM, false)
                .expect("failed to create norm head")
        };
        let norm = QKVNorm {
            query: Some(make_head()),
            key: Some(make_head()),
            value: Some(make_head()),
            num_q_heads: Q_HEADS,
            num_kv_heads: KV_HEADS,
            projection_row_stride: STRIDE,
            head_dim: DIM,
            full_width: false,
        };
        let input = (0..BATCH * STRIDE).map(|i| bf16::from_f32((i % 127) as f32 * 0.0625 - 4.0)).collect::<Vec<_>>();
        let mut ordinary = create_buffer_with_data::<B, bf16>(&context, &input);
        let mut kv_only = create_buffer_with_data::<B, bf16>(&context, &input);
        let mut command = context.create_command_buffer(None, None).expect("failed to create command buffer");
        norm.encode(&mut ordinary, BATCH, &mut command).expect("failed to encode ordinary norm");
        norm.encode_projected_key_value(&mut kv_only, BATCH, &mut command).expect("failed to encode projected KV norm");
        command.end_encoding().submit().wait_until_completed().expect("failed to execute projected KV norm");
        let ordinary = buffer_to_vec::<B, bf16>(&ordinary);
        let kv_only = buffer_to_vec::<B, bf16>(&kv_only);
        for i in 0..input.len() {
            let channel = i as u32 % STRIDE;
            let is_kv = channel >= Q_HEADS * DIM && channel < (Q_HEADS + 2 * KV_HEADS) * DIM;
            let expected = if is_kv {
                ordinary[i]
            } else {
                input[i]
            };
            assert_eq!(kv_only[i].to_bits(), expected.to_bits(), "projected KV norm bit mismatch");
        }
    }
}

#[cfg(feature = "attention-kv-only")]
#[stark_test]
fn test_projected_key_value_norm_retains_offsets_and_dead_channels() {
    for_each_backend!(|B| run_projected_key_value_exact::<B>());
}

#[stark_test]
fn packed_qkv_rejects_mlx_reduction_for_unqualified_dtype() {
    use crate::backends::cpu::Cpu;
    let context = create_context::<Cpu>();
    let config = NormalizationConfig {
        epsilon: 1e-6,
        scale_offset: None,
        upcast_mode: UpcastMode::OnlyNormalization,
        mlx_bf16_reduction: true,
        subtract_mean: false,
        has_scale: false,
        has_biases: false,
    };
    assert!(QKVNorm::<Cpu>::build_head(&context, DataType::F32, config, None, 128, false).is_err());
}
