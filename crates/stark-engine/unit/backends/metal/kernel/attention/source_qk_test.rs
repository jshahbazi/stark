use half::bf16;
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            Kernels,
            kernel::{
                AttentionArguments, AttentionKernelConfig,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{GemmEngine, Metal},
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
    tests::helpers::create_buffer_with_data,
};

#[stark_test]
fn source_qk_split_keeps_f32_partitions_until_the_bf16_score_store() {
    let context = <Metal as Backend>::Context::new().unwrap();
    if !context.supports_mxu {
        return;
    }
    const ROWS: usize = 22;
    const HEADS: usize = 8;
    const DIM: usize = 512;
    let mut queries = vec![bf16::ZERO; HEADS * ROWS * DIM];
    for q in queries.chunks_exact_mut(DIM) {
        for i in [0, 1, 256, 257] {
            q[i] = bf16::ONE;
        }
    }
    let mut keys = vec![bf16::ZERO; ROWS * DIM];
    keys[0] = bf16::ONE;
    keys[1] = bf16::from_f32(1.0 / 512.0);
    keys[256] = -bf16::ONE;
    keys[257] = bf16::from_f32(1.0 / 512.0);
    let queries = create_buffer_with_data::<Metal, bf16>(&context, &queries);
    let keys = create_buffer_with_data::<Metal, bf16>(&context, &keys);
    let values = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; ROWS * DIM]);
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let (_, stages) = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::encode_fallback_trace_for_test(
        &context,
        AttentionKernelConfig {
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
        },
        AttentionArguments {
            queries: &queries,
            keys: &keys,
            values: &values,
            suffix_length: ROWS as u32,
            trie: None::<&<Metal as Backend>::ConstantBuffer>,
            sinks: None,
            cache: KVCacheView::full(0),
        },
        &mut commands,
    )
    .unwrap();
    let scores = &stages.iter().find(|(name, _)| *name == "scores").unwrap().1;
    let mut source = context.create_buffer(scores.size()).unwrap();
    commands.encode_copy(scores, &mut source);
    // Negative control: BF16 partition storage loses both half-ULP terms.
    let mut negative =
        commands.allocate_scratch_for_shape(&[(HEADS * ROWS) as u32, ROWS as u32], DataType::BF16).unwrap();
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
    )
    .unwrap();
    kernel
        .encode_with_gemm_split_for_test(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &queries,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &keys,
                },
                b_leading_dimension: Some(DIM as u32),
                b_transpose: true,
                d: &mut negative,
                d_transform: MatmulDOps::none(),
                gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                m: (HEADS * ROWS) as u32,
                n: ROWS as u32,
                k: DIM as u32,
            },
            GemmEngine::Mxu,
            2,
            &mut commands,
        )
        .unwrap();
    let mut lossy = context.create_buffer(negative.size()).unwrap();
    commands.encode_copy(&negative, &mut lossy);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let source = source.copyout::<bf16>();
    let lossy = lossy.copyout::<bf16>();
    // Independent exact rational: (1+2^-9)+(-1+2^-9)=2^-8.
    let expected = bf16::from_f32(1.0 / 256.0);
    for row in 0..HEADS * ROWS {
        assert_eq!(source[row * ROWS], expected);
        assert_eq!(lossy[row * ROWS], bf16::ZERO);
    }
}
