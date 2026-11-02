// SPDX-License-Identifier: Apache-2.0
//! Pair-local bound reuse: unchanged IQ GEMVs, failure checks and ownership.

use stark_engine_macros::stark_test;

use super::{
    super::tests::{finite_packed_codes, packed_f64_oracle},
    *,
};

struct RestoreProfile(bool);
impl Drop for RestoreProfile {
    fn drop(&mut self) {
        profile::enable(self.0);
    }
}
fn capture_profile() -> RestoreProfile {
    let restore = RestoreProfile(profile::enabled());
    profile::reset();
    profile::enable(true);
    restore
}
fn calls(name: &str) -> u64 {
    profile::snapshot().get(name).map_or(0, |counter| counter.calls)
}
fn admitted_flags() {
    assert!(
        iq_bound_schedule(MatrixFormat::Packed(17), 2048) && iq_bound_schedule(MatrixFormat::Packed(18), 4096),
        "fresh process requires STARK_GLM53_IQ_GPU_BOUND=1"
    );
}
fn new_graph(shared: bool) -> DecodeGraph {
    let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
    graph.iq_shared_bound = shared;
    graph.batch_finite = true;
    graph.finite_selective = true;
    graph.serial_finite_order = true;
    graph
}
fn matrix(
    bytes: &[u8],
    offset: usize,
) -> MappedMatrix {
    let mut padded = vec![0x75u8; offset];
    padded.extend_from_slice(bytes);
    MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor().unwrap(), &padded).unwrap(),
            offset,
            length: bytes.len(),
            registered_length: padded.len(),
        },
        planes: None,
    }
}
fn input(values: &[f32]) -> DeviceVector {
    let allocation = DecodeGraph::zeros(values.len() + 2).unwrap();
    // SAFETY: fresh owned synthetic storage has no pending GPU users. Raw
    // writes deliberately allow adversarial NaN/Inf and integer expert IDs.
    unsafe {
        allocation
            .buffer
            .contents()
            .as_ptr()
            .cast::<f32>()
            .add(1)
            .copy_from_nonoverlapping(values.as_ptr(), values.len());
    }
    allocation.slice(1, values.len()).unwrap()
}
fn indices(ids: &[u32]) -> DeviceVector {
    input(&ids.iter().copied().map(f32::from_bits).collect::<Vec<_>>())
}
fn encode_pair(
    graph: &mut DecodeGraph,
    banks: [&MappedMatrix; 2],
    kinds: [u32; 2],
    shape: [usize; 3],
    ids: &DeviceVector,
    source: &DeviceVector,
) -> [DeviceVector; 3] {
    let [gate, up] = graph
        .indexed_pair(
            (banks[0], MatrixFormat::Packed(kinds[0]), shape),
            (banks[1], MatrixFormat::Packed(kinds[1]), shape),
            ids,
            ids.len,
            source,
        )
        .unwrap();
    let active = graph.swiglu(&gate, &up, Some(11.0)).unwrap();
    [gate, up, active]
}
fn evaluate(
    banks: [&MappedMatrix; 2],
    kinds: [u32; 2],
    shape: [usize; 3],
    values: &[f32],
    selected: &[u32],
    shared: bool,
) -> Result<Vec<Vec<f32>>> {
    let mut graph = new_graph(shared);
    let source = input(values);
    let ids = indices(selected);
    let output = if shared {
        encode_pair(&mut graph, banks, kinds, shape, &ids, &source)
    } else {
        // Reference is the original two public calls, not the new fallback.
        let gate = graph.indexed(banks[0], MatrixFormat::Packed(kinds[0]), shape, &ids, ids.len, &source, false)?;
        let up = graph.indexed(banks[1], MatrixFormat::Packed(kinds[1]), shape, &ids, ids.len, &source, false)?;
        let active = graph.swiglu(&gate, &up, Some(11.0))?;
        [gate, up, active]
    };
    graph.finish(&output)
}
fn same_bits(
    actual: &[Vec<f32>],
    expected: &[Vec<f32>],
) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| { a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()) }),
        "shared bound changed gate/up/activation bits"
    );
}
fn f64_gate(
    codes: &[u8],
    kind: u32,
    shape: [usize; 3],
    selected: &[u32],
    values: &[f32],
    actual: &[f32],
) {
    let [experts, rows, columns] = shape;
    let bytes = codes.len() / experts;
    let selected_codes = selected
        .iter()
        .flat_map(|&id| codes[id as usize * bytes..(id as usize + 1) * bytes].iter().copied())
        .collect::<Vec<_>>();
    let expected = packed_f64_oracle(kind, &selected_codes, selected.len() * rows, columns, values, 0);
    assert!(
        actual.iter().zip(expected).all(|(&actual, (expected, absolute))| {
            (f64::from(actual) - expected).abs() <= 2e-6 * absolute.max(1.0)
        }),
        "shared bound exceeded retained F64 absolute-sum gate"
    );
}

#[stark_test]
fn iq_shared_bound_selector_and_short_shape_preserve_legacy_dispatches() {
    let _profile = capture_profile();
    let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
    assert_eq!(
        graph.iq_shared_bound,
        std::env::var_os("STARK_GLM53_IQ_SHARED_BOUND").is_some_and(|value| value == "1")
    );
    graph.iq_shared_bound = true;
    let shape = [4, 13, 256];
    let bytes = finite_packed_codes(17, shape[0] * shape[1], shape[2], 7);
    let bank = matrix(&bytes, 0);
    let source = input(&vec![0.25; shape[2]]);
    let ids = indices(&[3, 1, 3]);
    let outputs = encode_pair(&mut graph, [&bank, &bank], [17, 17], shape, &ids, &source);
    graph.finish(&outputs).unwrap();
    assert_eq!(calls("native.metal.graph.indexed_matvec"), 2);
    assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 0);
    assert_eq!(calls("native.metal.graph.iq_shared_bound_pair"), 0);
}

#[stark_test]
#[ignore]
fn iq_shared_bound_pair_matches_source_f64_tails_offsets_and_exact_dispatch_counts() {
    admitted_flags();
    let _profile = capture_profile();
    let selected = [3, 1, 3];
    for columns in [2048, 4096] {
        let shape = [4, 13, columns];
        let values = (0..columns).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>();
        for kinds in [[17, 17], [18, 18], [17, 18], [18, 17]] {
            let bytes = [finite_packed_codes(kinds[0], 52, columns, 7), finite_packed_codes(kinds[1], 52, columns, 11)];
            for offsets in [[0, 2], [1, 2], [2, 1]] {
                let banks = [matrix(&bytes[0], offsets[0]), matrix(&bytes[1], offsets[1])];
                profile::reset();
                let expected = evaluate([&banks[0], &banks[1]], kinds, shape, &values, &selected, false).unwrap();
                assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 2);
                profile::reset();
                let actual = evaluate([&banks[0], &banks[1]], kinds, shape, &values, &selected, true).unwrap();
                assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 1);
                assert_eq!(calls("native.metal.graph.indexed_matvec"), 2);
                assert_eq!(calls("native.metal.graph.iq_shared_bound_pair"), 1);
                same_bits(&actual, &expected);
                for bank in 0..2 {
                    f64_gate(&bytes[bank], kinds[bank], shape, &selected, &values, &actual[bank]);
                }
            }
        }
    }
}

#[stark_test]
#[ignore]
fn iq_shared_bound_default_off_projection_and_invalid_pair_preserve_admission() {
    admitted_flags();
    let _profile = capture_profile();
    let shape = [4, 13, 2048];
    let bank = matrix(&finite_packed_codes(17, 52, 2048, 7), 0);
    let source = input(&vec![0.25; 2048]);
    let ids = indices(&[3, 1, 3]);
    for projection in [false, true] {
        profile::reset();
        let mut graph = new_graph(projection);
        if projection {
            graph.serial_projection_levels = false;
            graph.begin_projection_level().unwrap();
        }
        let [gate, up] = graph
            .indexed_pair(
                (&bank, MatrixFormat::Packed(17), shape),
                (&bank, MatrixFormat::Packed(17), shape),
                &ids,
                ids.len,
                &source,
            )
            .unwrap();
        if projection {
            graph.end_projection_level().unwrap();
        }
        let active = graph.swiglu(&gate, &up, Some(11.0)).unwrap();
        graph.finish(&[gate, up, active]).unwrap();
        assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 2);
        assert_eq!(calls("native.metal.graph.iq_shared_bound_pair"), 0);
    }
    profile::reset();
    let short_up = matrix(&finite_packed_codes(18, 20, 2048, 11), 2);
    let mut mismatch = new_graph(true);
    let outputs = mismatch
        .indexed_pair(
            (&bank, MatrixFormat::Packed(17), shape),
            (&short_up, MatrixFormat::Packed(18), [4, 5, 2048]),
            &ids,
            ids.len,
            &source,
        )
        .unwrap();
    let result = mismatch.finish(&outputs).unwrap();
    assert_eq!(result[0].len(), 39);
    assert_eq!(result[1].len(), 15);
    assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 2);
    assert_eq!(calls("native.metal.graph.iq_shared_bound_pair"), 0);
    profile::reset();
    let mut graph = new_graph(true);
    let mut invalid = bank.clone();
    invalid.codes.length -= 1;
    let before = graph.scratch_bytes;
    assert!(
        graph
            .indexed_pair(
                (&bank, MatrixFormat::Packed(17), shape),
                (&invalid, MatrixFormat::Packed(17), shape),
                &ids,
                ids.len,
                &source
            )
            .is_err()
    );
    assert_eq!(graph.scratch_bytes, before);
    assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 0);
    assert_eq!(calls("native.metal.graph.indexed_matvec"), 0);
    assert_eq!(graph.command.status(), MTLCommandBufferStatus::NotEnqueued);
}

#[stark_test]
#[ignore]
fn iq_shared_bound_extreme_nonfinite_masked_failures_and_recovery_match_source() {
    admitted_flags();
    let shape = [4, 13, 2048];
    let selected = [3, 1, 3];
    for kind in [17, 18] {
        let codes = finite_packed_codes(kind, 52, 2048, 7);
        let finite = matrix(&codes, 2);
        for scale in [1.0e20f32, f32::from_bits(1.0e20f32.to_bits() + 1), 1.0e21] {
            let mut values = (0..2048).map(|i| (i as f32 * 0.019).sin() * scale).collect::<Vec<_>>();
            values[2047] = scale;
            let a = evaluate([&finite, &finite], [kind, kind], shape, &values, &selected, false).unwrap();
            let b = evaluate([&finite, &finite], [kind, kind], shape, &values, &selected, true).unwrap();
            same_bits(&b, &a);
        }
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut values = vec![0.25; 2048];
            values[2047] = poison;
            let a = evaluate([&finite, &finite], [kind, kind], shape, &values, &selected, false);
            let b = evaluate([&finite, &finite], [kind, kind], shape, &values, &selected, true);
            assert_eq!(a.is_err(), b.is_err(), "shared bound changed extreme input failure");
            if !poison.is_finite() {
                assert!(a.is_err(), "shared bound lost nonfinite input failure");
            }
            if let (Ok(a), Ok(b)) = (a, b) {
                same_bits(&b, &a);
            }
        }
        let mut zero = codes.clone();
        let block_bytes = weights::block_layout(kind).unwrap().1;
        for block in zero.chunks_exact_mut(block_bytes) {
            block[..2].copy_from_slice(&0u16.to_le_bytes());
        }
        let zero = matrix(&zero, 0);
        let maximum = vec![f32::MAX; 2048];
        same_bits(
            &evaluate([&zero, &zero], [kind, kind], shape, &maximum, &selected, true).unwrap(),
            &evaluate([&zero, &zero], [kind, kind], shape, &maximum, &selected, false).unwrap(),
        );
        for half_poison in [0x7c00u16, 0x7e00] {
            let mut poisoned = codes.clone();
            for block in poisoned.chunks_exact_mut(block_bytes) {
                block[..2].copy_from_slice(&half_poison.to_le_bytes());
            }
            let poisoned = matrix(&poisoned, 0);
            for banks in [[&poisoned, &zero], [&zero, &poisoned]] {
                for shared in [false, true] {
                    assert!(
                        evaluate(banks, [kind, kind], shape, &vec![1.0; 2048], &selected, shared).is_err(),
                        "activation concealed an unclipped projection failure"
                    );
                }
            }
        }
        for shared in [false, true] {
            assert!(evaluate([&finite, &finite], [kind, kind], shape, &vec![1.0; 2048], &[u32::MAX], shared).is_err());
            assert!(evaluate([&finite, &finite], [kind, kind], shape, &vec![1.0; 2048], &selected, shared).is_ok());
        }
    }
}

#[stark_test]
#[ignore]
fn iq_shared_bound_rechecks_after_input_write_and_retains_unreturned_failure() {
    admitted_flags();
    let _profile = capture_profile();
    let shape = [4, 13, 2048];
    let bank = matrix(&finite_packed_codes(17, 52, 2048, 7), 0);
    let ordinary = vec![0.25; 2048];
    let large = vec![1.0e21; 2048];
    let expected = [
        evaluate([&bank, &bank], [17, 17], shape, &ordinary, &[3, 1, 3], false).unwrap(),
        evaluate([&bank, &bank], [17, 17], shape, &large, &[3, 1, 3], false).unwrap(),
    ];
    profile::reset();
    let mut graph = new_graph(true);
    let source = input(&ordinary);
    let ids = indices(&[3, 1, 3]);
    let first = encode_pair(&mut graph, [&bank, &bank], [17, 17], shape, &ids, &source);
    graph.copy_into(&input(&large), &source).unwrap();
    let second = encode_pair(&mut graph, [&bank, &bank], [17, 17], shape, &ids, &source);
    let actual = graph.finish(&[first[2].clone(), second[2].clone()]).unwrap();
    same_bits(&actual, &[expected[0][2].clone(), expected[1][2].clone()]);
    assert_eq!(calls("native.metal.graph.iq_bound_prepass"), 2);
    assert_eq!(calls("native.metal.graph.indexed_matvec"), 4);
    let mut graph = new_graph(true);
    let source = input(&vec![f32::INFINITY; 2048]);
    encode_pair(&mut graph, [&bank, &bank], [17, 17], shape, &ids, &source);
    graph.copy_into(&input(&ordinary), &source).unwrap();
    let later = encode_pair(&mut graph, [&bank, &bank], [17, 17], shape, &ids, &source);
    assert!(graph.finish(&[later[2].clone()]).is_err(), "later finite pair cleared sticky earlier failure");
}

#[stark_test]
#[ignore]
fn iq_shared_bound_unretained_submission_keeps_both_banks_and_shared_input_alive() {
    admitted_flags();
    let shape = [4, 13, 4096];
    let values = vec![0.25; 4096];
    let expected = {
        let a = matrix(&finite_packed_codes(17, 52, 4096, 7), 2);
        let b = matrix(&finite_packed_codes(18, 52, 4096, 11), 1);
        evaluate([&a, &b], [17, 18], shape, &values, &[3, 1, 3], false).unwrap()
    };
    let pending = objc2::rc::autoreleasepool(|_| {
        let a = matrix(&finite_packed_codes(17, 52, 4096, 7), 2);
        let b = matrix(&finite_packed_codes(18, 52, 4096, 11), 1);
        let mut graph = DecodeGraph::new_with_ownership(false, false, false, true).unwrap();
        graph.iq_shared_bound = true;
        let source = input(&values);
        let ids = indices(&[3, 1, 3]);
        let outputs = encode_pair(&mut graph, [&a, &b], [17, 18], shape, &ids, &source);
        for owned in [&a.codes.buffer, &b.codes.buffer, &source.buffer, &ids.buffer] {
            assert!(
                graph.resources.iter().any(|held| Retained::as_ptr(held) == Retained::as_ptr(owned)),
                "shared-bound pair omitted a raw resource keepalive"
            );
        }
        graph.resource_boundary().unwrap();
        graph.validate_queued_submission().unwrap();
        graph.submit(&outputs).unwrap()
    });
    let actual = objc2::rc::autoreleasepool(|_| pending.wait().unwrap());
    same_bits(&actual, &expected);
}

#[stark_test]
#[ignore]
fn iq_shared_bound_gpu_component_timings() {
    admitted_flags();
    let _activity = super::super::super::activity::InferenceActivity::begin().unwrap();
    let _restore = RestoreProfile(profile::enabled());
    profile::enable(false);
    let (experts, rows, columns, selected, groups) = (288usize, 2048usize, 4096usize, 8usize, 8usize);
    let shape = [experts, rows, columns];
    let values = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let source = input(&values);
    let ids = (0..groups)
        .map(|group| {
            indices(
                &(0..selected).map(|slot| (((group * selected + slot) * 37 + 19) % experts) as u32).collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    for kind in [17, 18] {
        // Generate directly into the large banks, avoiding a temporary Vec
        // allocation for each 256-weight block during component setup.
        let large_bank = |seed: usize| {
            let bytes = weights::block_layout(kind).unwrap().1;
            let mut codes = vec![0u8; experts * rows * columns / 256 * bytes];
            for (number, block) in codes.chunks_exact_mut(bytes).enumerate() {
                for (i, byte) in block.iter_mut().enumerate() {
                    *byte = ((i * 37 + number * 13 + seed) % 256) as u8;
                }
                block[..2].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
            }
            codes
        };
        let gate_bytes = large_bank(7);
        let up_bytes = large_bank(11);
        let bytes_per_expert = gate_bytes.len() / experts;
        let banks = [matrix(&gate_bytes, 2), matrix(&up_bytes, 2)];
        let mut probes = Vec::new();
        for group in 0..groups {
            let selected_ids =
                (0..selected).map(|slot| (((group * selected + slot) * 37 + 19) % experts) as u32).collect::<Vec<_>>();
            for (slot, &expert) in selected_ids.iter().enumerate() {
                for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                    let row_bytes = bytes_per_expert / rows;
                    let at = (expert as usize * rows + row) * row_bytes;
                    for (bank, codes) in [&gate_bytes, &up_bytes].into_iter().enumerate() {
                        let oracle = packed_f64_oracle(kind, &codes[at..at + row_bytes], 1, columns, &values, 0)[0];
                        probes.push((group * 3 + bank, slot * rows + row, oracle));
                    }
                }
            }
        }
        let mut samples = [Vec::new(), Vec::new()];
        let mut reference: Option<Vec<Vec<f32>>> = None;
        let mut maximum_f64_relative_absolute_sum = 0.0f64;
        for replay in 0..7 {
            for variant in if replay % 2 == 0 {
                [0usize, 1, 1, 0]
            } else {
                [1usize, 0, 0, 1]
            } {
                let (actual, seconds) = objc2::rc::autoreleasepool(|_| {
                    let mut graph = new_graph(variant == 1);
                    let command = graph.command.clone();
                    let mut outputs = Vec::new();
                    for at in 0..groups {
                        // Keep output order stable while rotating actual GPU
                        // dispatch order through scattered expert groups.
                        let group = (at + replay) % groups;
                        let row =
                            encode_pair(&mut graph, [&banks[0], &banks[1]], [kind, kind], shape, &ids[group], &source);
                        outputs.push((group, row));
                    }
                    outputs.sort_by_key(|row| row.0);
                    let views = outputs.into_iter().flat_map(|row| row.1).collect::<Vec<_>>();
                    let actual = graph.finish(&views).unwrap();
                    let seconds = (command.gpu_end_time() - command.gpu_start_time()) / groups as f64;
                    (actual, seconds)
                });
                for &(output, at, (expected, absolute)) in &probes {
                    let relative = (f64::from(actual[output][at]) - expected).abs() / absolute.max(1.0);
                    maximum_f64_relative_absolute_sum = maximum_f64_relative_absolute_sum.max(relative);
                    assert!(relative <= 2e-6, "shared-bound component F64 gate");
                }
                if let Some(reference) = &reference {
                    same_bits(&actual, reference);
                }
                if variant == 0 {
                    reference = Some(actual);
                }
                if replay > 0 {
                    samples[variant].push(seconds);
                }
            }
        }
        for (variant, times) in samples.iter_mut().enumerate() {
            times.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({
                    "component":"iq_pair_shared_bound_full_graph", "kind":kind, "shared_bound":variant==1,
                    "math_mode":"unchanged_production_safe_iq", "activity":true, "serial_encoder":true,
                    "rows":rows,"columns":columns,"experts":experts,"selected_experts":selected,
                    "rotating_scattered_groups":groups,"combined_bank_bytes":gate_bytes.len()+up_bytes.len(),
                    "rotating_population_bytes":bytes_per_expert*selected*groups*2,
                    "bound_dispatches_per_pair":if variant==0{2}else{1},"gemv_dispatches_per_pair":2,
                    "warm_samples":times.len(),"paired_abba":true,"bitwise_gate_up_activation":true,
                    "maximum_f64_error_over_absolute_sum":maximum_f64_relative_absolute_sum,
                    "median_gpu_microseconds_per_pair":(times[times.len()/2-1]+times[times.len()/2])*0.5e6,
                    "minimum_gpu_microseconds_per_pair":times[0]*1e6,
                    "maximum_gpu_microseconds_per_pair":times[times.len()-1]*1e6,
                    "timer_scope":"same_process_ABBA_serial_rotating_scattered_E288_full_pair_with_finite_checks"
                })
            );
        }
    }
}
