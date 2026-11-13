#[cfg(backend = "metal")]
use std::time::Instant;

use half::bf16;
use stark_engine_macros::stark_test;

#[cfg(backend = "metal")]
use crate::backends::{common::CommandBufferCompleted, metal::Metal};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::radix_top_k_small::RadixTopKSmall,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

fn radix_top_k_small_f32<B: Backend>(
    input: &[f32],
    rows: usize,
    columns: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let input = create_buffer_with_data::<B, f32>(&context, input);
    let mut ids = create_buffer::<B, u32>(&context, rows * k);
    let mut scores = create_buffer::<B, f32>(&context, rows * k);
    let kernel =
        <B::Kernels as Kernels>::RadixTopKSmall::new_with_data_type(&context, columns as u32, DataType::F32).unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&input, &mut ids, &mut scores, rows as u32, k as u32, &mut command).unwrap();
    command.end_encoding().submit().wait_until_completed().unwrap();
    (buffer_to_vec(&ids), buffer_to_vec(&scores))
}

#[stark_test]
fn radix_top_k_small_f32_retains_precision_and_total_order() {
    for shape @ (rows, columns, k) in [(1, 1, 1), (2, 17, 7), (3, 1025, 64), (2, TARGET_COLUMNS, 64)] {
        // Most positive values collapse to the same BF16 number. Ranking these
        // F32 bits therefore catches both accidental narrowing and dtype aliasing.
        let mut input = (0..rows * columns)
            .map(|index| {
                let value = f32::from_bits(0x3f80_0001 + (index * 31337 % 8191) as u32);
                if index % 7 == 0 {
                    -value
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let special = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.00001,
            1.00001,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_0001),
        ];
        let count = input.len().min(special.len());
        input[..count].copy_from_slice(&special[..count]);
        let expected = reference(&input, rows, columns, k);
        assert_output(&radix_top_k_small_f32::<Cpu>(&input, rows, columns, k), &expected, shape);
        for_each_non_cpu_backend!(|B| {
            assert_output(&radix_top_k_small_f32::<B>(&input, rows, columns, k), &expected, shape);
        });
    }
}

fn invalid_typed_buffers<B: Backend>() {
    use crate::backends::common::BufferRef;

    let context = create_context::<B>();
    assert!(<B::Kernels as Kernels>::RadixTopKSmall::new_with_data_type(&context, 16, DataType::F16).is_err());
    assert!(<B::Kernels as Kernels>::RadixTopKSmall::new(&context, 0).is_err());
    let kernel = <B::Kernels as Kernels>::RadixTopKSmall::new_with_data_type(&context, 16, DataType::F32).unwrap();
    for case in 0..6 {
        let input = create_buffer::<B, u8>(
            &context,
            if case == 0 {
                60
            } else {
                68
            },
        );
        let mut ids = create_buffer::<B, u32>(
            &context,
            if case == 1 {
                3
            } else {
                4
            },
        );
        let mut scores = create_buffer::<B, f32>(
            &context,
            if case == 2 {
                3
            } else {
                4
            },
        );
        let input = if case == 3 {
            (&input).subrange(1..65)
        } else {
            (&input).subrange(..)
        };
        let mut command = context.create_command_buffer(None, None).unwrap();
        let result = kernel.encode(
            input,
            &mut ids,
            &mut scores,
            if case == 4 {
                0
            } else {
                1
            },
            if case == 5 {
                17
            } else {
                4
            },
            &mut command,
        );
        assert!(result.is_err(), "typed radix buffer admission unexpectedly succeeded");
    }
}

#[stark_test]
fn radix_top_k_small_typed_buffers_fail_closed() {
    invalid_typed_buffers::<Cpu>();
    for_each_non_cpu_backend!(|B| {
        invalid_typed_buffers::<B>();
    });
}

const TARGET_COLUMNS: usize = 248_320;
const TARGET_K: usize = 512;

fn values(
    rows: usize,
    columns: usize,
) -> Vec<f32> {
    (0..rows * columns).map(|index| ((index * 37 % 101) as f32).sin()).collect()
}

fn radix_top_k_small<B: Backend>(
    input: &[bf16],
    rows: usize,
    columns: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let mut ids = create_buffer::<B, u32>(&context, rows * k);
    let mut scores = create_buffer::<B, f32>(&context, rows * k);
    let kernel = <B::Kernels as Kernels>::RadixTopKSmall::new(&context, columns as u32).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&input, &mut ids, &mut scores, rows as u32, k as u32, &mut command_buffer).unwrap();
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    (buffer_to_vec(&ids), buffer_to_vec(&scores))
}

fn reference(
    input: &[f32],
    rows: usize,
    columns: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(rows * k);
    let mut scores = Vec::with_capacity(rows * k);
    for values in input.chunks_exact(columns) {
        let mut row = (0..columns).collect::<Vec<_>>();
        row.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]).then_with(|| left.cmp(&right)));
        for index in row.into_iter().take(k) {
            ids.push(index as u32);
            scores.push(values[index]);
        }
    }
    (ids, scores)
}

fn assert_output(
    actual: &(Vec<u32>, Vec<f32>),
    expected: &(Vec<u32>, Vec<f32>),
    shape: (usize, usize, usize),
) {
    assert!(actual.0 == expected.0, "selector IDs differ for shape={shape:?}");
    assert!(
        actual.1.iter().map(|value| value.to_bits()).eq(expected.1.iter().map(|value| value.to_bits())),
        "shape={shape:?}",
    );
}

#[stark_test]
fn radix_top_k_small_matches_cpu() {
    for shape @ (rows, columns, k) in [
        (1, 1, 1),
        (1, 8, 8),
        (3, 1025, 1),
        (3, 1025, TARGET_K - 1),
        (3, 1025, TARGET_K),
        (15, TARGET_COLUMNS, TARGET_K),
    ] {
        let mut input = values(rows, columns).into_iter().map(bf16::from_f32).collect::<Vec<_>>();
        let special = [
            bf16::INFINITY,
            bf16::NEG_INFINITY,
            bf16::NEG_ZERO,
            bf16::ZERO,
            bf16::ONE,
            bf16::ONE,
            bf16::from_bits(0x7fc1),
            bf16::from_bits(0xffc1),
        ];
        let special_count = special.len().min(input.len());
        input[..special_count].copy_from_slice(&special[..special_count]);
        let expected = reference(&input.iter().map(|value| value.to_f32()).collect::<Vec<_>>(), rows, columns, k);
        assert_output(&radix_top_k_small::<Cpu>(&input, rows, columns, k), &expected, shape);
        for_each_non_cpu_backend!(|B| {
            let actual = radix_top_k_small::<B>(&input, rows, columns, k);
            assert_output(&actual, &expected, shape);
            if columns == TARGET_COLUMNS {
                for _ in 0..2 {
                    assert_output(&radix_top_k_small::<B>(&input, rows, columns, k), &expected, shape);
                }
            }
        });
    }
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn benchmark_radix_top_k_small() {
    const ROWS: usize = 15;
    const SAMPLES: usize = 50;
    const BATCH: u32 = 16;

    let context = create_context::<Metal>();
    let input = create_buffer_with_data::<Metal, bf16>(
        &context,
        &values(ROWS, TARGET_COLUMNS).into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
    );
    let mut ids = create_buffer::<Metal, u32>(&context, ROWS * TARGET_K);
    let mut scores = create_buffer::<Metal, f32>(&context, ROWS * TARGET_K);
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::RadixTopKSmall::new(&context, TARGET_COLUMNS as u32).unwrap();
    let mut run = || {
        let start = Instant::now();
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        for _ in 0..BATCH {
            kernel.encode(&input, &mut ids, &mut scores, ROWS as u32, TARGET_K as u32, &mut command_buffer).unwrap();
        }
        let completed = command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        (completed.gpu_execution_time().div_f64(BATCH as f64), start.elapsed().div_f64(BATCH as f64))
    };
    for _ in 0..5 {
        run();
    }
    let (mut gpu, mut wall): (Vec<_>, Vec<_>) = (0..SAMPLES).map(|_| run()).unzip();
    gpu.sort_unstable();
    wall.sort_unstable();
    eprintln!("radix_top_k_small gpu={:?} wall={:?}", gpu[SAMPLES / 2], wall[SAMPLES / 2]);
}
