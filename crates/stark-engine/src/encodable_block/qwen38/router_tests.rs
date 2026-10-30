//! Analytic controls for the parallel standard-kernel MoE router.

use stark_engine_macros::stark_test;

use crate::backends::common::{
    Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    Kernels, kernel::Qwen38MoeRouteKernel,
};

fn check<B: Backend>() {
    let context = B::Context::new().unwrap_or_else(|_| panic!("router test context unavailable"));
    let kernel = <B::Kernels as Kernels>::Qwen38MoeRouteKernel::new(context.as_ref()).unwrap();
    for (experts, selected, rows) in
        [(1, 1, 13), (3, 3, 13), (7, 7, 13), (17, 17, 13), (33, 33, 13), (257, 10, 13), (512, 10, 26), (513, 10, 26)]
    {
        let mut logits = vec![0.0_f32; rows * experts];
        for row in 0..rows {
            for expert in 0..experts {
                logits[row * experts + expert] = match row % 13 {
                    0 => {
                        if expert % 2 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    },
                    1 => {
                        if expert % 17 == 0 {
                            3.0
                        } else {
                            -2.0
                        }
                    },
                    2 => {
                        if expert == 0 || expert + 1 == experts {
                            4.0
                        } else {
                            (expert % 31) as f32 / 8.0 - 2.0
                        }
                    },
                    3 => {
                        if expert < 2 {
                            1e20
                        } else {
                            -1e20
                        }
                    },
                    4 => ((expert * 53 + row * 11) % 127) as f32 / 16.0 - 4.0,
                    5 => f32::from_bits(
                        [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000][expert % 8],
                    ),
                    6 => f32::from_bits(0x3f800000 + ((expert * 37) % 257) as u32),
                    7 => f32::from_bits(0xbf800000 + ((expert * 37) % 257) as u32),
                    8 => {
                        if expert + 1 == experts {
                            f32::MAX
                        } else {
                            -f32::MAX
                        }
                    },
                    9 => -f32::MAX,
                    10 => expert as f32,
                    11 => {
                        if expert.is_multiple_of(3) {
                            -0.0
                        } else if expert.is_multiple_of(2) {
                            0.0
                        } else {
                            -1.0
                        }
                    },
                    _ => f32::from_bits(
                        ((expert as u32).wrapping_mul(0x9e3779b9).wrapping_add((row as u32).wrapping_mul(0x85ebca6b)))
                            & 0xff7fffff,
                    ),
                };
            }
        }
        let expected: Vec<Vec<usize>> = (0..rows)
            .map(|row| {
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|&left, &right| {
                    logits[row * experts + right]
                        .partial_cmp(&logits[row * experts + left])
                        .unwrap()
                        .then(left.cmp(&right))
                });
                order
            })
            .collect();
        let mut input = context.create_buffer(logits.len() * 4).unwrap();
        input.copyin(&logits);
        for normalize in [0_u32, 1] {
            let mut indices = context.create_buffer(rows * selected * 4).unwrap();
            indices.copyin(&vec![u32::MAX; rows * selected]);
            let mut scores = context.create_buffer(rows * selected * 4).unwrap();
            scores.copyin(&vec![f32::NAN; rows * selected]);
            let mut commands = context.create_command_buffer(None, None).unwrap();
            kernel.encode(
                &input,
                &mut indices,
                &mut scores,
                experts as u32,
                selected as u32,
                normalize,
                rows as u32,
                &mut commands,
            );
            commands.end_encoding().submit().wait_until_completed().unwrap();
            let indices = (&indices).as_slice::<u32>();
            let scores = (&scores).as_slice::<f32>();
            for row in 0..rows {
                let maximum = f64::from(logits[row * experts + expected[row][0]]);
                let total = if normalize != 0 {
                    expected[row][..selected]
                        .iter()
                        .map(|&expert| (f64::from(logits[row * experts + expert]) - maximum).exp())
                        .sum::<f64>()
                } else {
                    logits[row * experts..(row + 1) * experts]
                        .iter()
                        .map(|&logit| (f64::from(logit) - maximum).exp())
                        .sum::<f64>()
                };
                for slot in 0..selected {
                    assert!(
                        indices[row * selected + slot] as usize == expected[row][slot],
                        "parallel router changed stable expert order"
                    );
                    let probability = (f64::from(logits[row * experts + expected[row][slot]]) - maximum).exp() / total;
                    let actual = f64::from(scores[row * selected + slot]);
                    assert!(
                        actual.is_finite() && (actual - probability).abs() <= 3e-6 + 3e-6 * probability.abs(),
                        "parallel router changed softmax normalization"
                    );
                }
            }
        }
    }
}

#[stark_test]
fn lexical_cutoff_matches_serial_numeric_order_over_finite_f32_categories() {
    fn key(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x7fffffff == 0 {
            0x80000000
        } else if bits & 0x80000000 != 0 {
            !bits
        } else {
            bits ^ 0x80000000
        }
    }
    let mut values = vec![
        -f32::MAX,
        -f32::MIN_POSITIVE,
        f32::from_bits(0x807fffff),
        f32::from_bits(0x80000001),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::from_bits(0x007fffff),
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    values.extend((0..1024_u32).map(|i| f32::from_bits(i.wrapping_mul(0x9e3779b9) & 0xff7fffff)));
    let mut expected = (0..values.len()).collect::<Vec<_>>();
    expected.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b)));
    // Complete selection includes the final ID and every equal-valued pair.
    // No selected-ID mask participates in this independent cutoff algorithm.
    let mut ceiling = (u32::MAX, 0_usize);
    for &expected_id in &expected {
        let winner = values
            .iter()
            .enumerate()
            .filter(|(id, value)| {
                let candidate = key(**value);
                candidate < ceiling.0 || (candidate == ceiling.0 && *id > ceiling.1)
            })
            .max_by(|(a_id, a), (b_id, b)| key(**a).cmp(&key(**b)).then(b_id.cmp(a_id)))
            .map(|(id, _)| id);
        assert!(winner == Some(expected_id), "lexical cutoff changed stable finite F32 selection");
        ceiling = (key(values[expected_id]), expected_id);
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn parallel_router_stable_ties_topk_rows_and_softmax_cpu() {
    check::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
fn parallel_router_stable_ties_topk_rows_and_softmax_metal() {
    check::<crate::backends::metal::Metal>();
}
