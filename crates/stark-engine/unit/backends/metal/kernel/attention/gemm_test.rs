use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use stark_engine_macros::stark_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
    },
};

struct Input<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    do_causal: bool,
}

fn attention_config<T: ArrayElement + Float>(input: &Input<T>) -> AttentionKernelConfig {
    super::default_attention_config::<T>(input.head_dim, input.num_heads, input.num_kv_heads, input.do_causal)
}

fn get_test_data<T: ArrayElement + Float>(
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    do_causal: bool,
) -> (Input<T>, Vec<T>) {
    let q_size = num_heads * suffix_length * head_dim;
    let mut queries = vec![T::zero(); q_size];
    for i in 0..q_size {
        queries[i] = T::from((i as f32 * 0.13 + 0.5).sin() * 0.5).unwrap();
    }

    let k_size = sequence_length * num_kv_heads * head_dim;
    let mut keys = vec![T::zero(); k_size];
    for i in 0..k_size {
        keys[i] = T::from((i as f32 * 0.07 + 1.0).cos() * 0.5).unwrap();
    }

    let v_size = sequence_length * num_kv_heads * head_dim;
    let mut values = vec![T::zero(); v_size];
    for i in 0..v_size {
        values[i] = T::from((i as f32 * 0.11 + 2.0).sin() * 0.5).unwrap();
    }

    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads,
        num_kv_heads,
        sequence_length,
        suffix_length,
        head_dim,
        do_causal,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let config = attention_config(input);

    let queries_buffer = create_buffer_with_data::<B, T>(context.as_ref(), &input.queries);
    let keys_buffer = create_buffer_with_data::<B, T>(context.as_ref(), &input.keys);
    let values_buffer = create_buffer_with_data::<B, T>(context.as_ref(), &input.values);

    let segment_prefix_length = input.sequence_length - input.suffix_length;
    let args = AttentionArguments {
        queries: &queries_buffer,
        keys: &keys_buffer,
        values: &values_buffer,
        suffix_length: input.suffix_length as u32,
        trie: None::<&B::GlobalBuffer>,
        sinks: None,
        cache: KVCacheView::full(segment_prefix_length as u32),
    };

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), config)
        .expect("Failed to create attention kernel");
    let pooled_output = kernel.encode(args, &mut command_buffer).expect("Failed to encode attention");
    let mut output_buffer =
        create_buffer::<B, T>(context.as_ref(), input.suffix_length * input.num_heads * input.head_dim);
    command_buffer.encode_copy(&pooled_output, &mut output_buffer);
    drop(pooled_output);
    let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    drop(completed);

    buffer_to_vec::<B, T>(&output_buffer)
}

fn get_gemm_output<T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    get_gemm_output_with_config(input, attention_config(input))
}

fn get_gemm_output_with_config<T: ArrayElement + Float>(
    input: &Input<T>,
    config: AttentionKernelConfig,
) -> Vec<T> {
    get_gemm_output_variant(input, config, false)
}

fn get_gemm_output_variant<T: ArrayElement + Float>(
    input: &Input<T>,
    config: AttentionKernelConfig,
    force_mxu: bool,
) -> Vec<T> {
    let context = <Metal as Backend>::Context::new().expect("Failed to create Metal context");
    let queries = create_buffer_with_data::<Metal, T>(context.as_ref(), &input.queries);
    let keys = create_buffer_with_data::<Metal, T>(context.as_ref(), &input.keys);
    let values = create_buffer_with_data::<Metal, T>(context.as_ref(), &input.values);
    let cache = KVCacheView::full((input.sequence_length - input.suffix_length) as u32);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    let arguments = AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: input.suffix_length as u32,
        trie: None::<&<Metal as Backend>::GlobalBuffer>,
        sinks: None,
        cache,
    };
    let gemm = super::gemm::AttentionGemm::new(&config);
    let pooled = if force_mxu {
        gemm.encode_forced_mxu_for_test(arguments, &mut command_buffer)
    } else {
        gemm.encode(arguments, &mut command_buffer)
    }
    .expect("Failed to encode AttentionGemm");
    let mut output =
        create_buffer::<Metal, T>(context.as_ref(), input.suffix_length * input.num_heads * input.head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    drop(pooled);
    let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    drop(completed);
    buffer_to_vec::<Metal, T>(&output)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let eps = if T::data_type() == DataType::BF16 {
        1e-2
    } else {
        1e-5
    };

    let output = get_gemm_output(input);
    let msg = format!(
        "AttentionGemm failed (heads={}, kv_heads={}, seq={}, suffix={}, head_dim={}, causal={})",
        input.num_heads,
        input.num_kv_heads,
        input.sequence_length,
        input.suffix_length,
        input.head_dim,
        input.do_causal,
    );
    assert_eq_float::<T>(expected, &output, eps, &msg);
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 1, 64, false);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, false);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 16, 1, 64, true);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, true);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(8, 2, 8, 1, 64, false);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(8, 2, 8, 4, 64, true);
    test_internal(&input, &expected);
}

fn test_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: usize) {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 2, head_dim, true);
    test_internal(&input, &expected);
}

fn test_unaligned<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 40, 7, 64, true);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 13, 4, 64, false);
    test_internal(&input, &expected);
}

#[stark_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[stark_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[stark_test]
fn test_causal_f32() {
    test_causal::<f32>();
}

#[stark_test]
fn test_causal_bf16() {
    test_causal::<bf16>();
}

#[stark_test]
fn test_gqa_f32() {
    test_gqa::<f32>();
}

#[stark_test]
fn test_gqa_bf16() {
    test_gqa::<bf16>();
}

#[stark_test]
fn test_head_dim_128_f32() {
    test_head_dim::<f32>(128);
}

#[stark_test]
fn test_head_dim_128_bf16() {
    test_head_dim::<bf16>(128);
}

#[stark_test]
fn test_unaligned_f32() {
    test_unaligned::<f32>();
}

#[stark_test]
fn test_unaligned_bf16() {
    test_unaligned::<bf16>();
}

#[stark_test]
fn test_prefill_mxu() {
    let (input, expected) = get_test_data::<bf16>(4, 4, 128, 128, 64, true);
    test_internal(&input, &expected);
    let (input, expected) = get_test_data::<bf16>(8, 2, 100, 96, 128, true);
    test_internal(&input, &expected);
}

#[stark_test]
fn source_prefill_scales_f32_score_after_cancelling_bf16_dot() {
    let context = <Metal as Backend>::Context::new().expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    const ROWS: usize = 9;
    const DIM: usize = 128;
    let mut queries = vec![bf16::ZERO; ROWS * DIM];
    let mut keys = queries.clone();
    let mut values = queries.clone();
    for row in 0..ROWS {
        queries[row * DIM] = bf16::from_f32(1.0078125);
        queries[row * DIM + 1] = bf16::ONE;
        if row == 0 {
            keys[row * DIM] = bf16::ONE;
            keys[row * DIM + 1] = bf16::from_f32(-1.0078125);
        }
        let value = if row == 0 {
            1.0
        } else {
            -0.125
        };
        for column in 0..DIM {
            values[row * DIM + column] = bf16::from_f32(value);
        }
    }
    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads: 1,
        num_kv_heads: 1,
        sequence_length: ROWS,
        suffix_length: ROWS,
        head_dim: DIM,
        do_causal: true,
    };
    let baseline_config = attention_config(&input);
    let source_config = AttentionKernelConfig {
        source_bf16_prefill: true,
        ..baseline_config
    };
    let source = get_gemm_output_with_config(&input, source_config);
    let baseline = get_gemm_output_with_config(&input, baseline_config);
    // All unscaled QK dots are exactly zero: a*1 + 1*(-a),
    // a=1+2^-7. The final row therefore has uniform attention.
    // Its exact numerator is 1 + 8*(-1/8) = 0. This also avoids
    // assumptions about MXU rounding of nontrivial probabilities.
    for column in 0..DIM {
        assert_eq!(source[(ROWS - 1) * DIM + column], bf16::ZERO);
        assert!(baseline[(ROWS - 1) * DIM + column].to_f32().abs() > 1e-8);
    }
}

#[stark_test]
fn source_prefill_mass_keeps_paired_half_ulp_contributions() {
    let context = <Metal as Backend>::Context::new().expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    const ROWS: usize = 9;
    const DIM: usize = 128;
    let mut queries = vec![bf16::ZERO; ROWS * DIM];
    let mut keys = queries.clone();
    let mut values = queries.clone();
    for row in 0..ROWS {
        queries[row * DIM] = bf16::ONE;
        keys[row * DIM] = bf16::from_f32(if row == 0 {
            0.0
        } else if row < 4 {
            -24.0
        } else {
            -128.0
        });
        let value = if row == 0 {
            1.0
        } else if row < 4 {
            65536.0
        } else {
            0.0
        };
        for column in 0..DIM {
            values[row * DIM + column] = bf16::from_f32(value);
        }
    }
    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads: 1,
        num_kv_heads: 1,
        sequence_length: ROWS,
        suffix_length: ROWS,
        head_dim: DIM,
        do_causal: true,
    };
    // ln(2)*log2(e) rounds to exactly one, so source and legacy query
    // scaling agree. Force MXU for both to isolate the reduction policy.
    let baseline = AttentionKernelConfig {
        scale: Some(std::f32::consts::LN_2),
        ..attention_config(&input)
    };
    let source = AttentionKernelConfig {
        source_bf16_prefill: true,
        ..baseline
    };
    let source_output = get_gemm_output_variant(&input, source, true);
    let legacy_output = get_gemm_output_variant(&input, baseline, true);
    // The first lane sees [1,2^-24,2^-24,2^-24]. The source pairs the
    // small terms, retaining 2^-23; the historical serial tree loses them.
    // The exact PV numerator 1+3/256 is a BF16 halfway value. Source mass
    // normalization moves below it, while the legacy halfway rounds upward.
    for column in 0..DIM {
        assert_eq!(source_output[(ROWS - 1) * DIM + column].to_f32(), 1.0078125);
        assert_eq!(legacy_output[(ROWS - 1) * DIM + column].to_f32(), 1.015625);
    }
}

#[stark_test]
fn source_prefill_finish_preserves_f32_reciprocal_boundary() {
    let context = <Metal as Backend>::Context::new().expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    const ROWS: usize = 9;
    const DIM: usize = 128;
    let mut queries = vec![bf16::ZERO; ROWS * DIM];
    let mut keys = queries.clone();
    let mut values = queries.clone();
    for row in 0..ROWS {
        queries[row * DIM] = bf16::ONE;
        keys[row * DIM] = bf16::from_f32(match row {
            0 => 0.0,
            4 => -1.0,
            8 => -2.0,
            _ => -128.0,
        });
        let value = match row {
            0 => 2.625,
            4 => 0.013671875,
            _ => 0.0,
        };
        for column in 0..DIM {
            values[row * DIM + column] = bf16::from_f32(value);
        }
    }
    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads: 1,
        num_kv_heads: 1,
        sequence_length: ROWS,
        suffix_length: ROWS,
        head_dim: DIM,
        do_causal: true,
    };
    let config = AttentionKernelConfig {
        scale: Some(std::f32::consts::LN_2),
        source_bf16_prefill: true,
        ..attention_config(&input)
    };
    let output = get_gemm_output_variant(&input, config, true);
    // The nonzero probabilities are 1, 1/2 and 1/4, giving mass 7/4.
    // Their exact PV numerator is 2695/1024. Exact division is 385/256,
    // a BF16 midpoint rounding to 1.5. The source first rounds 4/7 to
    // F32; its F32 product lies above that midpoint and rounds to 1.5078125.
    assert_eq!(bf16::from_f64(385.0 / 256.0).to_f32(), 1.5);
    for column in 0..DIM {
        assert_eq!(output[(ROWS - 1) * DIM + column].to_f32(), 1.5078125);
    }
}
