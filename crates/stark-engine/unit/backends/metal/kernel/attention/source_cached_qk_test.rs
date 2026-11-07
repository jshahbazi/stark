//! F64 dot-product oracle across source lane geometries and row tails.
use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            Kernels,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        },
        metal::Metal,
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
    tests::helpers::create_buffer_with_data,
};

#[stark_test]
fn source_cached_qk_has_independent_f64_bf16_midpoint_and_tail_oracles() {
    let context = <Metal as Backend>::Context::new().unwrap();
    if !context.supports_mxu {
        return;
    }
    const DIM: usize = 512;
    const HEADS: usize = 8;
    // Exactly representable F32 sums lie below/on/above a BF16 midpoint.
    // Rounding a partial result to BF16 loses the small correction.
    assert_ne!(
        bf16::from_f64(1.0 + 1.0 / 256.0 + 1.0 / 65536.0),
        bf16::from_f64(bf16::from_f64(1.0 + 1.0 / 256.0).to_f64() + 1.0 / 65536.0)
    );
    let mut query = vec![bf16::ZERO; HEADS * DIM];
    for head in 0..HEADS {
        query[head * DIM..(head + 1) * DIM].fill(bf16::from_f64((head + 1) as f64));
    }
    let query = create_buffer_with_data::<Metal, bf16>(&context, &query);
    for rows in [2usize, 63, 64, 65, 519, 520, 521, 1023] {
        let mut keys = vec![bf16::ZERO; rows * DIM];
        for row in 0..rows {
            keys[row * DIM] = bf16::ONE;
            keys[row * DIM + 64] = bf16::from_f64(1.0 / 256.0);
            keys[row * DIM + 128] = bf16::from_f64(match row % 3 {
                0 => 1.0 / 65536.0,
                1 => -1.0 / 65536.0,
                _ => 0.0,
            });
        }
        let keys = create_buffer_with_data::<Metal, bf16>(&context, &keys);
        let values = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; rows * DIM]);
        let config = AttentionKernelConfig {
            head_dim: DIM as u32,
            num_q_heads: HEADS as u32,
            num_groups: 1,
            has_sinks: false,
            is_kv_cache_ring: false,
            is_causal: true,
            sliding_window_size: None,
            scale: Some(1.0),
            data_type: DataType::BF16,
            materialize_prefill_scores: true,
            source_bf16_prefill: false,
        };
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let args = || AttentionArguments {
            queries: &query,
            keys: &keys,
            values: &values,
            suffix_length: 1,
            trie: None::<&<Metal as Backend>::ConstantBuffer>,
            sinks: None,
            cache: KVCacheView::full(rows as u32 - 1),
        };
        let (_, stages) = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::encode_fallback_trace_for_test(
            &context,
            config,
            args(),
            &mut commands,
        )
        .unwrap();
        let source = &stages.iter().find(|(name, _)| *name == "scores").unwrap().1;
        let mut scores = context.create_buffer(source.size()).unwrap();
        commands.encode_copy(source, &mut scores);
        // Exercise the ordinary production dispatcher as well as its trace.
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(&context, config).unwrap();
        let output = kernel.encode(args(), &mut commands).unwrap();
        let mut result = context.create_buffer(output.size()).unwrap();
        commands.encode_copy(&output, &mut result);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = scores.copyout::<bf16>();
        for head in 0..HEADS {
            for row in 0..rows {
                let delta = match row % 3 {
                    0 => 1.0 / 65536.0,
                    1 => -1.0 / 65536.0,
                    _ => 0.0,
                };
                let exact = (head + 1) as f64 * (1.0 + 1.0 / 256.0 + delta);
                assert_eq!(actual[head * rows + row], bf16::from_f64(exact), "head {head}, row {row}, length {rows}");
            }
        }
        assert!(result.copyout::<bf16>().iter().all(|&x| x == bf16::ZERO));
    }
}
