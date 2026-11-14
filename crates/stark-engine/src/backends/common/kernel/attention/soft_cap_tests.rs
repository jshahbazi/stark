use half::bf16;
use num_traits::Float;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    array::ArrayElement,
    backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
};

fn check<B: Backend, T: ArrayElement + Float>() {
    let context = B::Context::new().unwrap();
    for head_dim in [64, 128, 256] {
        for (rows, prefix, window, ring_offset) in [
            (1, 17, None, None),
            (9, 0, None, None),
            (9, 17, Some(4), None),
            (1, 1025, None, None),
            (9, 17, Some(4), Some(5)),
        ] {
            let keys = prefix + rows;
            let mut query = vec![T::zero(); 2 * rows * head_dim];
            let mut key = vec![T::zero(); keys * head_dim];
            let mut value = vec![T::zero(); keys * head_dim];
            for head in 0..2 {
                for row in 0..rows {
                    query[(head * rows + row) * head_dim] = T::from(if head == 0 {
                        1.0
                    } else {
                        -1.0
                    })
                    .unwrap();
                }
            }
            for at in 0..keys {
                key[at * head_dim] = T::from((at as i32 % 7 - 3) as f32 * 20.0).unwrap();
                value[at * head_dim] = T::from((at % 11) as f32 / 4.0).unwrap();
            }
            let q = create_buffer_with_data::<B, T>(&context, &query);
            let k = create_buffer_with_data::<B, T>(&context, &key);
            let v = create_buffer_with_data::<B, T>(&context, &value);
            let kernel = <B::Kernels as Kernels>::AttentionKernel::new(
                &context,
                AttentionKernelConfig {
                    head_dim: head_dim as u32,
                    num_groups: 1,
                    num_q_heads: 2,
                    has_sinks: false,
                    is_kv_cache_ring: ring_offset.is_some(),
                    is_causal: true,
                    sliding_window_size: window,
                    scale: Some(0.25),
                    logit_soft_cap: Some(2.0),
                    data_type: T::data_type(),
                    materialize_prefill_scores: false,
                    source_bf16_prefill: false,
                },
            )
            .unwrap();
            let mut commands = context.create_command_buffer(None, None).unwrap();
            let output = kernel
                .encode(
                    AttentionArguments {
                        queries: &q,
                        keys: &k,
                        values: &v,
                        suffix_length: rows as u32,
                        trie: None::<&B::GlobalBuffer>,
                        sinks: None,
                        cache: ring_offset.map_or_else(
                            || KVCacheView::full(prefix as u32),
                            |offset| KVCacheView::ring(prefix as u32, offset),
                        ),
                        image_span: None,
                    },
                    &mut commands,
                )
                .unwrap();
            let mut readback = create_buffer::<B, T>(&context, rows * 2 * head_dim);
            commands.encode_copy(&output, &mut readback);
            commands.end_encoding().submit().wait_until_completed().unwrap();
            let actual = buffer_to_vec::<B, T>(&readback);
            for row in 0..rows {
                for head in 0..2 {
                    let end = prefix + row + 1;
                    let mut norm = 0.0f64;
                    let mut weighted = 0.0f64;
                    for at in 0..end {
                        let position = if at < prefix {
                            ring_offset.map_or(at, |offset| (prefix + at - offset as usize) % prefix)
                        } else {
                            at
                        };
                        if window.is_some_and(|size| end - 1 - position >= size as usize) {
                            continue;
                        }
                        let sign = if head == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let scaled_qk = sign * 0.25 * key[at * head_dim].to_f64().unwrap();
                        let mass = (2.0 * (scaled_qk / 2.0).tanh()).exp();
                        norm += mass;
                        weighted += mass * value[at * head_dim].to_f64().unwrap();
                    }
                    let offset = (row * 2 + head) * head_dim;
                    let tolerance = if T::data_type() == DataType::F32 {
                        2e-5
                    } else {
                        0.008
                    };
                    assert!((actual[offset].to_f64().unwrap() - weighted / norm).abs() <= tolerance);
                    assert!(actual[offset + 1..offset + head_dim].iter().all(|x| *x == T::zero()));
                }
            }
        }
    }
}

#[stark_test]
fn capped_attention_matches_independent_scaled_softmax_oracle() {
    for_each_backend!(|B| {
        check::<B, f32>();
        check::<B, bf16>();
    });
}
