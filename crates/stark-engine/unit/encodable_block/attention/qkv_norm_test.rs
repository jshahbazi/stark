use stark_engine_macros::stark_test;

use super::{Head, QKVNorm};
use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        kernel::{Kernels, NormalizationKernel, QKVNormKernel},
    },
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

fn make_wide_source_head<B: Backend>(
    context: &B::Context,
    scales: &[f32],
    use_wide_source: bool,
) -> Head<B> {
    let mut config = NormalizationConfig {
        epsilon: 1e-6,
        scale_offset: None,
        upcast_mode: UpcastMode::OnlyNormalization,
        mlx_bf16_reduction: true,
        subtract_mean: false,
        has_scale: false,
        has_biases: false,
    };
    let mut head = QKVNorm::<B>::build_head(context, DataType::BF16, config.clone(), None, scales.len() as u32, false)
        .expect("failed to create source QKV head");
    head.kernel = <B::Kernels as Kernels>::QKVNormKernel::new(
        context,
        DataType::BF16,
        DataType::F32,
        DataType::BF16,
        DataType::F32,
        true,
        true,
        true,
    )
    .expect("failed to create scaled source QKV kernel");
    head.scales = Some(create_buffer_with_data::<B, f32>(context, scales));
    config.has_scale = true;
    head.config = config;
    head.wide_source = use_wide_source.then(|| {
        <B::Kernels as Kernels>::NormalizationKernel::new(
            context,
            DataType::BF16,
            DataType::F32,
            DataType::BF16,
            DataType::F32,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            true,
        )
        .expect("failed to create wide source normalization kernel")
    });
    head
}

fn run_wide_source_qkv_matches_head_kernel<B: Backend>() {
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 32;
    const HEAD_DIM: u32 = 128;
    const WIDTH: u32 = QUERY_HEADS * HEAD_DIM;
    const GATE_WIDTH: u32 = WIDTH;
    const ROW_STRIDE: u32 = 3 * WIDTH + GATE_WIDTH;
    let context = create_context::<B>();
    let make_scales = |offset: f32| {
        (0..WIDTH as usize)
            .map(|index| half::bf16::from_f32(offset + (index % 29) as f32 * 0.00390625).to_f32())
            .collect::<Vec<_>>()
    };
    let query_scales = make_scales(0.75);
    let key_scales = make_scales(0.875);
    let make_norm = |use_wide_source| QKVNorm {
        query: Some(make_wide_source_head::<B>(&context, &query_scales, use_wide_source)),
        key: Some(make_wide_source_head::<B>(&context, &key_scales, use_wide_source)),
        value: None,
        num_q_heads: QUERY_HEADS,
        num_kv_heads: KV_HEADS,
        projection_row_stride: ROW_STRIDE,
        head_dim: HEAD_DIM,
        full_width: true,
    };
    let old_norm = make_norm(false);
    let wide_norm = make_norm(true);
    for batch_size in [1u32, 2] {
        let input = (0..batch_size as usize * ROW_STRIDE as usize)
            .map(|index| {
                let channel = index % ROW_STRIDE as usize;
                let value = ((index * 37 % 257) as f32 - 128.0) * 0.0078125;
                if channel >= (3 * WIDTH) as usize {
                    half::bf16::from_f32(0.25 + (channel % 31) as f32 * 0.015625)
                } else {
                    half::bf16::from_f32(value)
                }
            })
            .collect::<Vec<_>>();
        let mut old_output = create_buffer_with_data::<B, half::bf16>(&context, &input);
        let mut wide_output = create_buffer_with_data::<B, half::bf16>(&context, &input);
        let mut commands = context.create_command_buffer(None, None).expect("failed to create command buffer");
        old_norm.encode(&mut old_output, batch_size, &mut commands).expect("failed to encode old source QKV norm");
        wide_norm.encode(&mut wide_output, batch_size, &mut commands).expect("failed to encode wide source QKV norm");
        commands.end_encoding().submit().wait_until_completed().expect("failed to execute source QKV comparison");
        let old_values = buffer_to_vec::<B, half::bf16>(&old_output);
        let wide_values = buffer_to_vec::<B, half::bf16>(&wide_output);
        let old_bits = old_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        let wide_bits = wide_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        let row_stride = ROW_STRIDE as usize;
        let qk_end = (2 * WIDTH) as usize;
        for row in 0..batch_size as usize {
            let start = row * row_stride;
            assert_eq!(
                &old_bits[start..start + qk_end],
                &wide_bits[start..start + qk_end],
                "source Q/K bits differ for batch={batch_size} on {}",
                std::any::type_name::<B>(),
            );
            assert_eq!(
                &input[start + 2 * WIDTH as usize..start + row_stride],
                &old_values[start + 2 * WIDTH as usize..start + row_stride],
                "V/gate/padding changed in old path",
            );
            assert_eq!(
                &input[start + 2 * WIDTH as usize..start + row_stride],
                &wide_values[start + 2 * WIDTH as usize..start + row_stride],
                "V/gate/padding changed in wide path",
            );
        }
    }
}

#[stark_test]
fn wide_source_qkv_matches_head_kernel_and_preserves_other_channels() {
    for_each_backend!(|B| run_wide_source_qkv_matches_head_kernel::<B>());
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

// Source small-head correctness controls, alongside the validated wide-row test.
fn run_small_source_qkv_rows<B: Backend>() {
    use half::bf16;
    const Q: u32 = 8;
    let context = create_context::<B>();
    for dim in [256u32, 512] {
        let query_scales = (0..dim).map(|i| 0.750244140625 + (i % 31) as f32 / 2048.0).collect::<Vec<_>>();
        let key_scales = (0..dim).map(|i| 0.875244140625 - (i % 23) as f32 / 2048.0).collect::<Vec<_>>();
        for batch in [1u32, 2] {
            for layout in 0..4 {
                for value_norm in [false, true] {
                    let shared = layout == 3;
                    let kv = if shared {
                        0
                    } else {
                        1
                    };
                    let stride = if layout == 2 {
                        2 * dim
                    } else {
                        (Q + 2 * kv + Q) * dim + 32
                    };
                    let make = |candidate: bool| QKVNorm {
                        query: Some(make_wide_source_head::<B>(&context, &query_scales, candidate)),
                        key: (kv != 0).then(|| make_wide_source_head::<B>(&context, &key_scales, candidate)),
                        value: (kv != 0 && value_norm).then(|| {
                            QKVNorm::<B>::build_head(
                                &context,
                                DataType::BF16,
                                NormalizationConfig {
                                    epsilon: 1e-6,
                                    scale_offset: None,
                                    upcast_mode: UpcastMode::OnlyNormalization,
                                    mlx_bf16_reduction: true,
                                    subtract_mean: false,
                                    has_scale: false,
                                    has_biases: false,
                                },
                                None,
                                dim,
                                false,
                            )
                            .unwrap()
                        }),
                        num_q_heads: Q,
                        num_kv_heads: kv,
                        projection_row_stride: stride,
                        head_dim: dim,
                        full_width: false,
                    };
                    let old = make(false);
                    let candidate = make(true);
                    let input = (0..(batch * stride + 32) as usize)
                        .map(|i| bf16::from_f32(((i * 37 % 509) as f32 - 254.0) / 128.0))
                        .collect::<Vec<_>>();
                    let mut a = create_buffer_with_data::<B, bf16>(&context, &input);
                    let mut b = create_buffer_with_data::<B, bf16>(&context, &input);
                    let mut commands = context.create_command_buffer(None, None).unwrap();
                    for (norm, buffer) in [(&old, &mut a), (&candidate, &mut b)] {
                        match layout {
                            1 => norm.encode_packed(buffer, batch, Q, stride, true, &mut commands).unwrap(),
                            2 => norm.encode_key_value(buffer, batch, &mut commands).unwrap(),
                            _ => norm.encode(buffer, batch, &mut commands).unwrap(),
                        }
                    }
                    commands.end_encoding().submit().wait_until_completed().unwrap();
                    let a = buffer_to_vec::<B, bf16>(&a);
                    let b = buffer_to_vec::<B, bf16>(&b);
                    assert_eq!(
                        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "small source QKV old/new bits differ dim={dim} batch={batch} layout={layout} value={value_norm}"
                    );
                    for row in 0..batch as usize {
                        let start = row * stride as usize;
                        let live = if layout == 2 {
                            2 * dim
                        } else {
                            (Q + 2 * kv) * dim
                        } as usize;
                        assert_eq!(
                            &b[start + live..start + stride as usize],
                            &input[start + live..start + stride as usize],
                            "gate/padding changed"
                        );
                        if layout == 1 {
                            assert_eq!(
                                &b[start..start + (Q * dim) as usize],
                                &input[start..start + (Q * dim) as usize],
                                "skipped queries changed"
                            );
                        }
                        if kv != 0 && !value_norm {
                            let value_start = start
                                + if layout == 2 {
                                    dim
                                } else {
                                    (Q + 1) * dim
                                } as usize;
                            assert_eq!(
                                &b[value_start..value_start + dim as usize],
                                &input[value_start..value_start + dim as usize],
                                "unnormalized values changed"
                            );
                        }
                    }
                    assert_eq!(
                        &b[(batch * stride) as usize..],
                        &input[(batch * stride) as usize..],
                        "trailing sentinel changed"
                    );
                }
            }
        }
        // Inject a regular source kernel into legacy heads: the host guard
        // must preserve SourceFalse, FullLayer and nonzero-offset semantics.
        for legacy in 0..3 {
            let make = |candidate: bool| {
                let mut query = make_wide_source_head::<B>(&context, &query_scales, candidate);
                let mut key = make_wide_source_head::<B>(&context, &key_scales, candidate);
                for head in [&mut query, &mut key] {
                    head.config.mlx_bf16_reduction = false;
                    head.config.upcast_mode = if legacy == 1 {
                        UpcastMode::FullLayer
                    } else {
                        UpcastMode::OnlyNormalization
                    };
                    head.config.scale_offset = (legacy == 2).then_some(1.0);
                    head.kernel = <B::Kernels as Kernels>::QKVNormKernel::new(
                        &context,
                        DataType::BF16,
                        DataType::F32,
                        DataType::BF16,
                        DataType::F32,
                        true,
                        true,
                        false,
                    )
                    .unwrap();
                }
                QKVNorm {
                    query: Some(query),
                    key: Some(key),
                    value: None,
                    num_q_heads: Q,
                    num_kv_heads: 1,
                    projection_row_stride: 10 * dim,
                    head_dim: dim,
                    full_width: false,
                }
            };
            let old = make(false);
            let candidate = make(true);
            let input = (0..(10 * dim) as usize)
                .map(|i| bf16::from_f32(((i * 13 % 251) as f32 - 125.0) / 64.0))
                .collect::<Vec<_>>();
            let mut a = create_buffer_with_data::<B, bf16>(&context, &input);
            let mut b = create_buffer_with_data::<B, bf16>(&context, &input);
            let mut commands = context.create_command_buffer(None, None).unwrap();
            old.encode(&mut a, 1, &mut commands).unwrap();
            candidate.encode(&mut b, 1, &mut commands).unwrap();
            commands.end_encoding().submit().wait_until_completed().unwrap();
            assert_eq!(
                buffer_to_vec::<B, bf16>(&a).iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                buffer_to_vec::<B, bf16>(&b).iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "legacy small-head exclusion differs dim={dim} case={legacy}"
            );
        }
    }
}

#[stark_test]
fn small_source_qkv_rows_match_packed_source_and_preserve_other_channels() {
    for_each_backend!(|B| run_small_source_qkv_rows::<B>());
}
