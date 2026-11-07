use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{AttentionPrepareKVOnlyKernel, AttentionPrepareKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
};

struct Output {
    queries: Vec<bf16>,
    keys: Vec<bf16>,
    values: Vec<bf16>,
}

fn run<B: Backend>() -> Output {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        true,
        false,
        false,
        false,
        false,
        false,
    )
    .expect("Failed to create AttentionPrepareKernel");
    let qkvg = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 101.0, 102.0, 103.0, 104.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 111.0, 112.0, 113.0, 114.0,
    ]
    .map(bf16::from_f32);
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &qkvg);
    let mut queries = create_buffer::<B, bf16>(&context, 8);
    let mut keys = create_buffer::<B, bf16>(&context, 4);
    let mut values = create_buffer::<B, bf16>(&context, 4);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");

    kernel.encode(
        &qkvg,
        &mut queries,
        Some(&mut keys),
        Some(&mut values),
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        None,
        None,
        None::<&B::GlobalBuffer>,
        2,
        Some(1),
        2,
        None,
        Some(0),
        12,
        2,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        queries: buffer_to_vec(&queries),
        keys: buffer_to_vec(&keys),
        values: buffer_to_vec(&values),
    }
}

fn assert_output(
    output: Output,
    backend: &str,
) {
    let expected_queries = [1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0].map(bf16::from_f32);
    let expected_keys = [5.0, 6.0, 15.0, 16.0].map(bf16::from_f32);
    let expected_values = [7.0, 8.0, 17.0, 18.0].map(bf16::from_f32);

    assert_eq!(output.queries, expected_queries, "query mismatch on {backend}");
    assert_eq!(output.keys, expected_keys, "key mismatch on {backend}");
    assert_eq!(output.values, expected_values, "value mismatch on {backend}");
}

fn run_query_only<B: Backend>() -> Vec<bf16> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        false,
        false,
        false,
        false,
        false,
        false,
    )
    .expect("Failed to create AttentionPrepareKernel");
    let qg = [1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0, 11.0, 12.0, 13.0, 14.0, 111.0, 112.0, 113.0, 114.0]
        .map(bf16::from_f32);
    let qg = create_buffer_with_data::<B, bf16>(&context, &qg);
    let mut queries = create_buffer::<B, bf16>(&context, 8);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");

    kernel.encode(
        &qg,
        &mut queries,
        None::<&mut B::GlobalBuffer>,
        None::<&mut B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        None,
        None,
        None::<&B::GlobalBuffer>,
        2,
        None,
        2,
        None,
        None,
        8,
        2,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    buffer_to_vec(&queries)
}

#[stark_test]
fn test_attention_prepare_uses_qkvg_row_stride() {
    assert_output(run::<Cpu>(), "CPU");
    for_each_non_cpu_backend!(|B| {
        assert_output(run::<B>(), std::any::type_name::<B>());
    });
}

#[stark_test]
fn test_attention_prepare_uses_query_gate_row_stride() {
    let expected = [1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0].map(bf16::from_f32);
    assert_eq!(run_query_only::<Cpu>(), expected, "query mismatch on CPU");
    for_each_non_cpu_backend!(|B| {
        assert_eq!(run_query_only::<B>(), expected, "query mismatch on {}", std::any::type_name::<B>());
    });
}

fn run_reference_bf16_rope<B: Backend>() {
    let context = B::Context::new().unwrap();
    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        false,
        true,
        true,
        false,
        false,
        false,
    )
    .unwrap();
    // Independent pinned Torch BF16 arithmetic on public [1.1, 2.3] and
    // FP32 sin/cos(0.3). Keeping only a final cast instead gives [16063,16417].
    let input = [16269, 16403, 49261, 16515].map(bf16::from_bits);
    let packed = create_buffer_with_data::<B, bf16>(&context, &input);
    let cosines = create_buffer_with_data::<B, f32>(&context, &[f32::from_bits(1064603887); 2]);
    let sines = create_buffer_with_data::<B, f32>(&context, &[f32::from_bits(1050103405); 2]);
    let mut queries = create_buffer::<B, bf16>(&context, 4);
    let mut command = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &packed,
        &mut queries,
        None::<&mut B::GlobalBuffer>,
        None::<&mut B::GlobalBuffer>,
        Some(&cosines),
        Some(&sines),
        None::<&B::GlobalBuffer>,
        None,
        None,
        None::<&B::GlobalBuffer>,
        1,
        None,
        4,
        Some(2),
        None,
        4,
        1,
        &mut command,
    );
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, bf16>(&queries);
    assert_eq!(actual, [16066, 16418, 49261, 16515].map(bf16::from_bits));
}

#[stark_test]
fn test_attention_prepare_reference_bf16_materializes_rotary_products() {
    run_reference_bf16_rope::<Cpu>();
    for_each_non_cpu_backend!(|B| run_reference_bf16_rope::<B>());
}

fn run_kv_only_exact<B: Backend>() {
    let context = B::Context::new().expect("failed to create context");
    // Include dispatch-family and preserved scorer-batch boundaries. Preparation
    // never changes the suffix size presented to subsequent attention consumers.
    for (q_heads, kv_heads, head_dim, batch_dim) in
        [(2, 1, 4, 2), (3, 2, 128, 9), (4, 1, 128, 65), (4, 1, 128, 1023), (4, 1, 128, 1024), (4, 1, 128, 1025)]
    {
        for rope_dim in [None, Some(head_dim / 2), Some(head_dim)] {
            let ordinary = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
                &context,
                DataType::BF16,
                DataType::F32,
                true,
                rope_dim.is_some(),
                false,
                false,
                false,
                false,
            )
            .expect("failed to create ordinary prepare");
            let kv_only = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKVOnlyKernel::new(
                &context,
                DataType::BF16,
                DataType::F32,
                rope_dim.is_some(),
                false,
                false,
                false,
                false,
            )
            .expect("failed to create KV-only prepare");
            let prefix = 3;
            let stride = (2 * q_heads + 2 * kv_heads) * head_dim;
            let input = (0..batch_dim * stride)
                .map(|i| {
                    // Poison dead Q/gate channels to detect incorrect packed offsets.
                    let channel = i % stride;
                    if channel < q_heads * head_dim || channel >= (q_heads + 2 * kv_heads) * head_dim {
                        bf16::from_bits(0x7fc1)
                    } else {
                        bf16::from_f32((i % 101) as f32 * 0.0625 - 3.0)
                    }
                })
                .collect::<Vec<_>>();
            let qkvg = create_buffer_with_data::<B, bf16>(&context, &input);
            let cos = (0..batch_dim * rope_dim.unwrap_or(0)).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>();
            let sin = (0..batch_dim * rope_dim.unwrap_or(0)).map(|i| (i as f32 * 0.013).sin()).collect::<Vec<_>>();
            let cos = rope_dim.map(|_| create_buffer_with_data::<B, f32>(&context, &cos));
            let sin = rope_dim.map(|_| create_buffer_with_data::<B, f32>(&context, &sin));
            let initial = vec![bf16::from_bits(0x7fc2); ((prefix + batch_dim) * kv_heads * head_dim) as usize];
            let mut keys = create_buffer_with_data::<B, bf16>(&context, &initial);
            let mut values = create_buffer_with_data::<B, bf16>(&context, &initial);
            let mut actual_keys = create_buffer_with_data::<B, bf16>(&context, &initial);
            let mut actual_values = create_buffer_with_data::<B, bf16>(&context, &initial);
            let mut queries = create_buffer::<B, bf16>(&context, (q_heads * batch_dim * head_dim) as usize);
            let mut command = context.create_command_buffer(None, None).expect("failed to create command buffer");
            ordinary.encode(
                &qkvg,
                &mut queries,
                Some(&mut keys),
                Some(&mut values),
                cos.as_ref(),
                sin.as_ref(),
                None::<&B::GlobalBuffer>,
                None,
                None,
                None::<&B::GlobalBuffer>,
                q_heads,
                Some(kv_heads),
                head_dim,
                rope_dim,
                Some(prefix),
                stride,
                batch_dim,
                &mut command,
            );
            kv_only.encode(
                &qkvg,
                &mut actual_keys,
                &mut actual_values,
                cos.as_ref(),
                sin.as_ref(),
                None::<&B::GlobalBuffer>,
                None,
                None,
                None::<&B::GlobalBuffer>,
                q_heads,
                kv_heads,
                head_dim,
                rope_dim,
                prefix,
                stride,
                batch_dim,
                &mut command,
            );
            command.end_encoding().submit().wait_until_completed().expect("failed to execute prepare comparison");
            let bits = |buffer: &B::GlobalBuffer| {
                buffer_to_vec::<B, bf16>(buffer).into_iter().map(bf16::to_bits).collect::<Vec<_>>()
            };
            assert_eq!(bits(&keys), bits(&actual_keys), "KV-only key bit mismatch");
            assert_eq!(bits(&values), bits(&actual_values), "KV-only value bit mismatch");
            let prefix_len = (prefix * kv_heads * head_dim) as usize;
            assert!(bits(&actual_keys)[..prefix_len].iter().all(|bits| *bits == 0x7fc2), "key prefix overwritten");
            assert!(bits(&actual_values)[..prefix_len].iter().all(|bits| *bits == 0x7fc2), "value prefix overwritten");
        }
    }
}

#[stark_test]
fn test_kv_only_prepare_preserves_packed_rope_bits_and_prefix() {
    run_kv_only_exact::<Cpu>();
    for_each_non_cpu_backend!(|B| run_kv_only_exact::<B>());
}

// Public, non-model input exercises non-contiguous absolute positions and proves
// that this explicit profile computes coefficients instead of reading the table.
fn run_source_rope_positions<B: Backend>() {
    let context = B::Context::new().unwrap();
    let positions = [0u32, 7, 19];
    let input: Vec<bf16> =
        (0..3).flat_map(|_| [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).map(bf16::from_f32).collect();
    for proportional in [false, true] {
        let kernel = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
            &context,
            DataType::BF16,
            DataType::F32,
            true,
            true,
            false,
            true,
            proportional,
            false,
        )
        .unwrap();
        let kv_only = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKVOnlyKernel::new(
            &context,
            DataType::BF16,
            DataType::F32,
            true,
            false,
            true,
            proportional,
            false,
        )
        .unwrap();
        let packed = create_buffer_with_data::<B, bf16>(&context, &input);
        let positions_buffer = create_buffer_with_data::<B, u32>(&context, &positions);
        let poison = create_buffer_with_data::<B, f32>(&context, &[f32::NAN; 12]);
        let mut queries = create_buffer::<B, bf16>(&context, 12);
        let mut keys = create_buffer::<B, bf16>(&context, 20);
        let mut values = create_buffer::<B, bf16>(&context, 20);
        let mut keys_only = create_buffer::<B, bf16>(&context, 20);
        let mut values_only = create_buffer::<B, bf16>(&context, 20);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let base = if proportional {
            4.0
        } else {
            2.0
        }; // raw base vs log2(base)
        kernel.encode(
            &packed,
            &mut queries,
            Some(&mut keys),
            Some(&mut values),
            Some(&poison),
            Some(&poison),
            Some(&positions_buffer),
            Some(base),
            Some(2),
            None::<&B::GlobalBuffer>,
            1,
            Some(1),
            4,
            Some(4),
            Some(2),
            12,
            3,
            &mut commands,
        );
        kv_only.encode(
            &packed,
            &mut keys_only,
            &mut values_only,
            Some(&poison),
            Some(&poison),
            Some(&positions_buffer),
            Some(base),
            Some(2),
            None::<&B::GlobalBuffer>,
            1,
            1,
            4,
            Some(4),
            2,
            12,
            3,
            &mut commands,
        );
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let expected: Vec<bf16> = positions
            .into_iter()
            .flat_map(|position| {
                let a = f64::from(position);
                let b = if proportional {
                    0.0
                } else {
                    a * 0.5
                };
                [
                    (a.cos() - 3.0 * a.sin()) as f32,
                    (2.0 * b.cos() - 4.0 * b.sin()) as f32,
                    (a.sin() + 3.0 * a.cos()) as f32,
                    (2.0 * b.sin() + 4.0 * b.cos()) as f32,
                ]
            })
            .map(bf16::from_f32)
            .collect();
        assert_eq!(buffer_to_vec::<B, bf16>(&queries), expected, "source coefficient profile");
        let keys = buffer_to_vec::<B, bf16>(&keys);
        assert_eq!(&keys[8..], expected.as_slice(), "cache offset");
        assert_eq!(&keys[8..], &buffer_to_vec::<B, bf16>(&keys_only)[8..], "KV-only source rotation");
        assert_eq!(&buffer_to_vec::<B, bf16>(&values)[8..], &buffer_to_vec::<B, bf16>(&values_only)[8..]);
    }
}

#[stark_test]
fn source_rope_uses_absolute_positions_and_preserves_proportional_pairs() {
    run_source_rope_positions::<Cpu>();
    for_each_non_cpu_backend!(|B| run_source_rope_positions::<B>());
}
