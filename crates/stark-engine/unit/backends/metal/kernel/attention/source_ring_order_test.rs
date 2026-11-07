//! Independent float32 cancellation oracle for the source physical cache order.
use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        },
        metal::Metal,
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
};

fn run(
    values: &[bf16],
    view: KVCacheView,
    ring: bool,
    source: bool,
) -> Vec<bf16> {
    const DIM: usize = 256;
    let context = <Metal as Backend>::Context::new().unwrap();
    let keys = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; values.len()]);
    let values = create_buffer_with_data::<Metal, bf16>(&context, values);
    let query = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; 8 * DIM]);
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(
        &context,
        AttentionKernelConfig {
            head_dim: DIM as u32,
            num_q_heads: 8,
            num_groups: 1,
            has_sinks: false,
            is_causal: true,
            is_kv_cache_ring: ring,
            sliding_window_size: ring.then_some(512),
            scale: Some(1.0),
            data_type: DataType::BF16,
            materialize_prefill_scores: source,
            source_bf16_prefill: false,
        },
    )
    .unwrap();
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let output = kernel
        .encode(
            AttentionArguments {
                queries: &query,
                keys: &keys,
                values: &values,
                suffix_length: 1,
                trie: None::<&<Metal as Backend>::ConstantBuffer>,
                sinks: None,
                cache: view,
            },
            &mut commands,
        )
        .unwrap();
    let mut copy = create_buffer::<Metal, bf16>(&context, 8 * DIM);
    commands.encode_copy(&output, &mut copy);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<Metal, bf16>(&copy)
}

#[stark_test]
fn source_ring_order_preserves_bulk_origin_and_single_token_phase_at_a_bf16_midpoint() {
    const DIM: usize = 256;
    // For uniform attention, each SIMDgroup visits every 32nd physical key.
    // Source order sees (+2^24,+1,-2^24): RN-even loses the +1, yielding 0.
    // The old appended-suffix order sees (+1,-2^24,+2^24): the first sum is
    // exactly representable, retaining 1. Uniform normalization yields 1/512.
    let large = bf16::from_f32(16_777_216.0);
    let legacy_expected = bf16::from_f32(1.0 / 512.0);
    for (origin, offset) in [(0usize, 0usize), (4, 4), (4, 5), (511, 511), (511, 0)] {
        let cursor = (offset + 512 - origin) % 512;
        let mut compact = vec![bf16::ZERO; 512 * DIM];
        compact[cursor * DIM] = large;
        compact[((cursor + 32) % 512) * DIM] = bf16::ONE;
        compact[((cursor + 64) % 512) * DIM] = -large;
        let mut physical = vec![bf16::ZERO; 513 * DIM];
        // Establish the source's independently specified physical array from
        // origin+slot timestamps, with the current key at the pending suffix.
        for slot in 0..512 {
            let destination = if slot == cursor {
                512
            } else {
                (origin + slot) % 512
            };
            physical[destination * DIM..(destination + 1) * DIM]
                .copy_from_slice(&compact[slot * DIM..(slot + 1) * DIM]);
        }
        let view = KVCacheView::ring(512, offset as u32).with_source_ring_origin(origin as u32);
        let golden = run(&compact, KVCacheView::full(511), false, true);
        assert!(golden.iter().all(|&x| x == bf16::ZERO));
        for source in [false, true, false] {
            let actual = run(&physical, view, true, source);
            for head in 0..8 {
                assert_eq!(
                    actual[head * DIM],
                    if source {
                        bf16::ZERO
                    } else {
                        legacy_expected
                    }
                );
                assert!(actual[head * DIM + 1..(head + 1) * DIM].iter().all(|&x| x == bf16::ZERO));
            }
        }
    }
    // Before the first wrap, the source-compatible path retains all 512 keys
    // in their existing order; no virtual replacement may be applied.
    let mut before_wrap = vec![bf16::ZERO; 512 * DIM];
    before_wrap[0] = large;
    before_wrap[32 * DIM] = bf16::ONE;
    before_wrap[64 * DIM] = -large;
    assert_eq!(
        run(&before_wrap, KVCacheView::ring(511, 0), true, true),
        run(&before_wrap, KVCacheView::full(511), false, true)
    );
}
