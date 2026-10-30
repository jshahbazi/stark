//! Synthetic 16K/64K selector, cache-address and causal-tail controls.
//! No model data or readout vectors are logged.
use half::{bf16, f16};
use stark_engine_macros::stark_test;

use crate::backends::common::{
    Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    Kernels,
    kernel::{
        Qwen38QsaAttentionKernel, Qwen38QsaBlockKeysKernel, Qwen38QsaCombineKernel, Qwen38QsaPrepareIndexKernel,
        Qwen38QsaPrepareKeyKernel, Qwen38QsaScoresKernel, Qwen38QsaSelectKernel, Qwen38QsaStoreValueKernel,
    },
};

const GUARD: usize = 32;
const D: usize = 256;
const KV: usize = 2;
const Q: usize = 24;

#[stark_test]
fn context_limit_accounts_for_all_supported_compression_ratios() {
    for ratio in [1, 2, 4, 8, 1_048_576] {
        assert_eq!(super::context_limit(65_536, ratio), 65_536);
        assert_eq!(super::context_limit(16_383, ratio), 16_383);
        assert_eq!(super::context_limit(u32::MAX, ratio), 65_536);
        assert!(65_536 / ratio <= super::MAX_VISIBLE_BLOCKS);
    }
    assert_eq!(super::context_limit(65_536, 0), 0);
}

fn selector<B: Backend>(context: &B::Context) {
    let select = <B::Kernels as Kernels>::Qwen38QsaSelectKernel::new(context).unwrap();
    // Ratios1/2 at context64K require four/two CACHE32 chunks. Include the
    // exact64K/ratio4 boundary and both sides of every cache-size transition.
    for blocks in [
        0, 1, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 16_383, 16_384, 16_385, 32_767,
        32_768, 32_769, 65_535, 65_536,
    ] {
        for pattern in 0..3 {
            let scores: Vec<f32> = (0..blocks)
                .map(|i| match pattern {
                    0 => 0.0,
                    1 => (i % 31) as f32 * 0.125,
                    _ => {
                        if i % 5 == 0 {
                            8.0
                        } else {
                            (i % 7) as f32 * 0.25
                        }
                    },
                })
                .collect();
            let mut expected: Vec<u32> = (0..blocks as u32).collect();
            expected.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]).then_with(|| a.cmp(&b)));
            expected.truncate(blocks.min(512));
            expected.sort_unstable();
            let mut input = context.create_buffer(blocks.max(1) * 4).unwrap();
            input.as_slice_mut::<f32>()[..blocks].copy_from_slice(&scores);
            let mut output = context.create_buffer((512 + 8) * 4).unwrap();
            output.as_slice_mut::<u32>().fill(u32::MAX);
            let mut count = context.create_buffer(4).unwrap();
            count.as_slice_mut::<u32>()[0] = u32::MAX;
            let mut commands = context.create_command_buffer(None, None).unwrap();
            select.encode(&input, (&mut output).subrange_mut(..512 * 4), &mut count, blocks as u32, 512, &mut commands);
            commands.end_encoding().submit().wait_until_completed().unwrap();
            assert!(count.as_slice::<u32>()[0] as usize == expected.len(), "QSA boundary selected-count mismatch");
            assert!(
                output.as_slice::<u32>()[..expected.len()] == expected[..],
                "QSA boundary stable selection mismatch"
            );
            assert!(
                output.as_slice::<u32>()[expected.len()..].iter().all(|&x| x == u32::MAX),
                "QSA selection wrote past count"
            );
        }
    }
    // Exercise the final legal ID through the selector, including a count of
    // 65,536. The 65,535 case also uses compaction near the full-row boundary;
    // the all-legal budget takes the kernel's separate direct-copy branch.
    let blocks = 65_536_usize;
    let mut scores = vec![0.0_f32; blocks];
    scores[blocks - 1] = 1.0;
    let mut input = context.create_buffer(blocks * 4).unwrap();
    input.copyin(&scores);
    for budget in [1, 512, 65_535, 65_536] {
        let mut expected: Vec<u32> = (0..blocks as u32).collect();
        expected.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]).then_with(|| a.cmp(&b)));
        expected.truncate(budget);
        expected.sort_unstable();
        let mut output = context.create_buffer(budget * 4 + GUARD * 2).unwrap();
        output.as_slice_mut::<u32>().fill(u32::MAX);
        let mut count = context.create_buffer(4).unwrap();
        count.as_slice_mut::<u32>()[0] = u32::MAX;
        let mut commands = context.create_command_buffer(None, None).unwrap();
        select.encode(
            &input,
            (&mut output).subrange_mut(GUARD..GUARD + budget * 4),
            &mut count,
            blocks as u32,
            budget as u32,
            &mut commands,
        );
        commands.end_encoding().submit().wait_until_completed().unwrap();
        assert!(count.as_slice::<u32>()[0] as usize == budget, "QSA final-ID selected-count mismatch");
        assert!(
            output.as_slice::<u32>()[GUARD / 4..GUARD / 4 + budget] == expected[..],
            "QSA final-ID stable selection mismatch"
        );
        assert!(
            output.as_slice::<u32>()[..GUARD / 4].iter().all(|&x| x == u32::MAX)
                && output.as_slice::<u32>()[GUARD / 4 + budget..].iter().all(|&x| x == u32::MAX),
            "QSA final-ID selection overwrote a guard"
        );
        assert!(expected.last() == Some(&65_535), "QSA final-ID fixture omitted the final legal ID");
    }
}

fn cache_writes<B: Backend>(context: &B::Context) {
    let key = <B::Kernels as Kernels>::Qwen38QsaPrepareKeyKernel::new(context).unwrap();
    let value = <B::Kernels as Kernels>::Qwen38QsaStoreValueKernel::new(context).unwrap();
    let index = <B::Kernels as Kernels>::Qwen38QsaPrepareIndexKernel::new(context).unwrap();
    let block = <B::Kernels as Kernels>::Qwen38QsaBlockKeysKernel::new(context).unwrap();
    for capacity in [16_384_usize, 65_536] {
        let position = capacity - 1;
        let code_bytes = KV * capacity * D;
        let scale_values = KV * capacity * (D / 32);
        let mut keys = context.create_buffer(code_bytes + GUARD * 2).unwrap();
        let mut values = context.create_buffer(code_bytes + GUARD * 2).unwrap();
        let mut ks = context.create_buffer(scale_values * 2 + GUARD * 2).unwrap();
        let mut vs = context.create_buffer(scale_values * 2 + GUARD * 2).unwrap();
        for buffer in [&mut keys, &mut values, &mut ks, &mut vs] {
            buffer.as_slice_mut::<u8>().fill(0x5a);
        }
        let mut zero_key = context.create_buffer(KV * D * 2).unwrap();
        zero_key.as_slice_mut::<bf16>().fill(bf16::ZERO);
        let mut v = context.create_buffer(KV * D * 2).unwrap();
        for (i, x) in v.as_slice_mut::<bf16>().iter_mut().enumerate() {
            *x = bf16::from_f32(if i < D {
                127.0
            } else {
                -127.0
            });
        }
        let mut norm = context.create_buffer(D * 4).unwrap();
        norm.as_slice_mut::<f32>().fill(0.0);
        let mut raw = context.create_buffer(capacity * 128 * 2 + GUARD * 2).unwrap();
        raw.as_slice_mut::<u8>().fill(0x5a);
        let mut qk = context.create_buffer(5 * 128 * 2).unwrap();
        qk.as_slice_mut::<bf16>().fill(bf16::ZERO);
        qk.as_slice_mut::<bf16>()[4 * 128..].fill(bf16::from_f32(4.0));
        let mut iq = context.create_buffer(4 * 128 * 2).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        key.encode(
            &zero_key,
            &norm,
            (&mut keys).subrange_mut(GUARD..GUARD + code_bytes),
            (&mut ks).subrange_mut(GUARD..GUARD + scale_values * 2),
            KV as u32,
            capacity as u32,
            64,
            position as u32,
            1_000_000.0,
            1e-6,
            &mut commands,
        );
        value.encode(
            &v,
            (&mut values).subrange_mut(GUARD..GUARD + code_bytes),
            (&mut vs).subrange_mut(GUARD..GUARD + scale_values * 2),
            KV as u32,
            capacity as u32,
            position as u32,
            &mut commands,
        );
        index.encode(
            &qk,
            (&norm).subrange(..128 * 4),
            &mut iq,
            (&mut raw).subrange_mut(GUARD..GUARD + capacity * 128 * 2),
            4,
            64,
            position as u32,
            1_000_000.0,
            1e-6,
            &mut commands,
        );
        commands.end_encoding().submit().wait_until_completed().unwrap();
        for i in 0..code_bytes + GUARD * 2 {
            let head = (0..KV).find(|&h| {
                (GUARD + (h * capacity + position) * D..GUARD + (h * capacity + position + 1) * D).contains(&i)
            });
            assert!(
                keys.as_slice::<u8>()[i]
                    == if head.is_some() {
                        0
                    } else {
                        0x5a
                    },
                "QSA key cache boundary overwrite"
            );
            let expected = head.map_or(0x5a, |h| {
                if h == 0 {
                    127
                } else {
                    (-127_i8) as u8
                }
            });
            assert!(values.as_slice::<u8>()[i] == expected, "QSA value cache boundary overwrite");
        }
        for i in 0..scale_values + GUARD {
            let written = (0..KV).any(|h| {
                (GUARD / 2 + (h * capacity + position) * 8..GUARD / 2 + (h * capacity + position + 1) * 8).contains(&i)
            });
            assert!(
                ks.as_slice::<f16>()[i].to_bits()
                    == if written {
                        0
                    } else {
                        0x5a5a
                    },
                "QSA key scale boundary overwrite"
            );
            assert!(
                vs.as_slice::<f16>()[i].to_bits()
                    == if written {
                        f16::ONE.to_bits()
                    } else {
                        0x5a5a
                    },
                "QSA value scale boundary overwrite"
            );
        }
        for (i, x) in raw.as_slice::<bf16>().iter().enumerate() {
            let written = (GUARD / 2 + position * 128..GUARD / 2 + (position + 1) * 128).contains(&i);
            assert!(
                x.to_bits()
                    == if written {
                        bf16::from_f32(4.0).to_bits()
                    } else {
                        0x5a5a
                    },
                "QSA raw index boundary overwrite"
            );
        }
        // KV code/scale strides above are independent of compression ratio.
        // Compressed rows have ratio-dependent capacities: at 64K, exercise
        // indices 65,535/32,767/16,383 for ratios 1/2/4 respectively. Only each
        // final raw block is zero; other blocks retain a nonzero sentinel.
        for ratio in [1, 2, 4] {
            let blocks = capacity / ratio;
            let mut compressed = context.create_buffer(blocks * 128 * 2 + GUARD * 2).unwrap();
            compressed.as_slice_mut::<u8>().fill(0x5a);
            raw.as_slice_mut::<bf16>()[GUARD / 2 + (capacity - ratio) * 128..GUARD / 2 + capacity * 128]
                .fill(bf16::ZERO);
            let mut commands = context.create_command_buffer(None, None).unwrap();
            block.encode(
                (&raw).subrange(GUARD..GUARD + capacity * 128 * 2),
                (&norm).subrange(..128 * 4),
                (&mut compressed).subrange_mut(GUARD..GUARD + blocks * 128 * 2),
                ratio as u32,
                (blocks - 1) as u32,
                64,
                1_000_000.0,
                1e-6,
                &mut commands,
            );
            commands.end_encoding().submit().wait_until_completed().unwrap();
            for (i, x) in compressed.as_slice::<bf16>().iter().enumerate() {
                let written = (GUARD / 2 + (blocks - 1) * 128..GUARD / 2 + blocks * 128).contains(&i);
                assert!(
                    if written {
                        x.to_f32() == 0.0
                    } else {
                        x.to_bits() == 0x5a5a
                    },
                    "QSA compressed index boundary overwrite"
                );
            }
        }
    }
}

fn score_grid<B: Backend>(context: &B::Context) {
    let kernel = <B::Kernels as Kernels>::Qwen38QsaScoresKernel::new(context).unwrap();
    let mut query = context.create_buffer(4 * 128 * 2).unwrap();
    query.as_slice_mut::<bf16>().fill(bf16::ONE);
    for blocks in [16_383_usize, 16_384, 65_536] {
        let mut keys = context.create_buffer(blocks * 128 * 2).unwrap();
        for (row, values) in keys.as_slice_mut::<bf16>().chunks_exact_mut(128).enumerate() {
            values.fill(bf16::from_f32((row % 5 + 1) as f32));
        }
        let mut scores = context.create_buffer(blocks * 4 + GUARD).unwrap();
        scores.as_slice_mut::<u8>().fill(0x5a);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        kernel.encode(&query, &keys, (&mut scores).subrange_mut(..blocks * 4), 4, blocks as u32, &mut commands);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        // Integer inputs keep every dot/reduction exact. Four gamma steps
        // cover the CPU sqrt/reciprocal/multiply or Metal constant/multiply;
        // the independent oracle computes the real scale in F64.
        let unit = 2_f64.powi(-24);
        let gamma = 4.0 * unit / (1.0 - 4.0 * unit);
        for (row, &value) in scores.as_slice::<f32>()[..blocks].iter().enumerate() {
            let expected = 512.0 * (row % 5 + 1) as f64 / 128_f64.sqrt();
            assert!(
                value.is_finite() && (f64::from(value) - expected).abs() <= gamma * expected,
                "QSA boundary score-grid oracle mismatch"
            );
        }
        assert!(
            scores.as_slice::<u8>()[blocks * 4..].iter().all(|&x| x == 0x5a),
            "QSA score grid wrote past final block"
        );
    }
}

fn causal_tail<B: Backend>(context: &B::Context) {
    let attention = <B::Kernels as Kernels>::Qwen38QsaAttentionKernel::new(context).unwrap();
    let combine = <B::Kernels as Kernels>::Qwen38QsaCombineKernel::new(context).unwrap();
    let capacity = 65_536_usize;
    let mut keys = context.create_buffer(KV * capacity * D).unwrap();
    let mut values = context.create_buffer(KV * capacity * D).unwrap();
    let mut ks = context.create_buffer(KV * capacity * 8 * 2).unwrap();
    let mut vs = context.create_buffer(KV * capacity * 8 * 2).unwrap();
    let mut query = context.create_buffer(Q * D * 2).unwrap();
    let mut gate = context.create_buffer(Q * D * 2).unwrap();
    query.as_slice_mut::<bf16>().fill(bf16::ZERO);
    gate.as_slice_mut::<bf16>().fill(bf16::ZERO);
    for (ratio, position) in [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 16_382),
        (4, 16_383),
        (4, 16_384),
        (4, 65_532),
        (4, 65_533),
        (4, 65_534),
        (4, 65_535),
        (1, 65_535),
        (2, 65_535),
    ] {
        let visible = (position + 1) / ratio;
        let selected: Vec<u32> = if visible == 0 {
            Vec::new()
        } else {
            (0..512).map(|i| (i * (visible - 1) / 511) as u32).collect()
        };
        let mut tokens: Vec<usize> =
            selected.iter().flat_map(|&b| b as usize * ratio..(b as usize + 1) * ratio).collect();
        tokens.extend(visible * ratio..position + 1);
        assert!(!tokens.is_empty(), "QSA attention fixture must have a nonempty causal key set");
        keys.as_slice_mut::<i8>().fill(-111);
        values.as_slice_mut::<i8>().fill(-111);
        ks.as_slice_mut::<f16>().fill(f16::NAN);
        vs.as_slice_mut::<f16>().fill(f16::NAN);
        for head in 0..KV {
            for &token in &tokens {
                let row = head * capacity + token;
                keys.as_slice_mut::<i8>()[row * D..(row + 1) * D].fill(0);
                let value = if visible == 0 {
                    // Tail-only rows must remain distinguishable if any row
                    // is omitted. Each scale group also has a distinct scale.
                    [24_i8, 60, 120][token]
                        * if head == 0 {
                            1
                        } else {
                            -1
                        }
                } else if token >= visible * ratio {
                    if head == 0 {
                        120
                    } else {
                        -120
                    }
                } else {
                    8 * (head as i8 + 1)
                };
                values.as_slice_mut::<i8>()[row * D..(row + 1) * D].fill(value);
                ks.as_slice_mut::<f16>()[row * 8..(row + 1) * 8].fill(f16::ONE);
                for (group, scale) in vs.as_slice_mut::<f16>()[row * 8..(row + 1) * 8].iter_mut().enumerate() {
                    *scale = if visible == 0 {
                        f16::from_f32([0.5, 1.0, 2.0][(token + group) % 3])
                    } else {
                        f16::ONE
                    };
                }
            }
        }
        let splits = tokens.len().div_ceil(64);
        let mut selection = context.create_buffer(512 * 4).unwrap();
        selection.as_slice_mut::<u32>().fill(u32::MAX);
        selection.as_slice_mut::<u32>()[..selected.len()].copy_from_slice(&selected);
        let mut count = context.create_buffer(4).unwrap();
        count.copyin(&[selected.len() as u32]);
        let mut partials = context.create_buffer(Q * splits * D * 4).unwrap();
        let mut stats = context.create_buffer(Q * splits * 2 * 4).unwrap();
        let mut output = context.create_buffer(Q * D * 2 + GUARD).unwrap();
        output.as_slice_mut::<u8>().fill(0x5a);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        attention.encode(
            &query,
            &keys,
            &values,
            &ks,
            &vs,
            &selection,
            &count,
            &mut partials,
            &mut stats,
            capacity as u32,
            12,
            splits as u32,
            512,
            ratio as u32,
            position as u32,
            0.0625,
            Q as u32,
            3,
            KV as u32,
            &mut commands,
        );
        combine.encode(
            &partials,
            &stats,
            &gate,
            (&mut output).subrange_mut(..Q * D * 2),
            Q as u32,
            splits as u32,
            &mut commands,
        );
        commands.end_encoding().submit().wait_until_completed().unwrap();
        // Zero queries make the independent F64 oracle a uniform mean of the
        // actual cache codes times their F16 scales, preserving raw-BF16 then
        // gate rounding. Zero selected blocks still have 1/2/3 causal tail rows;
        // their poisoned selection must not be read. Wrong old/future cache
        // rows read poisoned scales or the sentinel value.
        let codes = values.as_slice::<i8>();
        let scales = vs.as_slice::<f16>();
        let expected: Vec<bf16> = (0..KV * D)
            .map(|i| {
                let head = i / D;
                let d = i % D;
                let mean = tokens
                    .iter()
                    .map(|&token| {
                        let row = head * capacity + token;
                        f64::from(codes[row * D + d]) * scales[row * 8 + d / 32].to_f64()
                    })
                    .sum::<f64>()
                    / tokens.len() as f64;
                let raw = bf16::from_f32(mean as f32);
                bf16::from_f32(raw.to_f32() * 0.5)
            })
            .collect();
        for (i, x) in output.as_slice::<bf16>()[..Q * D].iter().enumerate() {
            let head = usize::from(i / D >= 12);
            assert!(
                x.to_bits() == expected[head * D + i % D].to_bits(),
                "QSA boundary attention read outside selected causal rows"
            );
        }
        assert!(
            output.as_slice::<u8>()[Q * D * 2..].iter().all(|&x| x == 0x5a),
            "QSA boundary attention wrote past output"
        );
    }
}

fn boundaries<B: Backend>() {
    let context = B::Context::new().unwrap();
    selector::<B>(context.as_ref());
    score_grid::<B>(context.as_ref());
    cache_writes::<B>(context.as_ref());
    causal_tail::<B>(context.as_ref());
}

#[cfg(backend = "cpu")]
#[stark_test]
fn qwen38_qsa_64k_selector_cache_and_tail_boundaries_cpu() {
    boundaries::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
fn qwen38_qsa_64k_selector_cache_and_tail_boundaries_metal() {
    boundaries::<crate::backends::metal::Metal>();
}
