// SPDX-License-Identifier: Apache-2.0
//! Narrow actual-route IQ2 fusion and audited command ownership.
use stark_engine_macros::stark_test;

use super::*;

const SHAPE: [usize; 3] = [8, 2048, 4096];
fn admitted_flags() {
    assert!(
        iq_fused_gate_up_enabled() && iq_signed_factor_schedule(MatrixFormat::Packed(17), 4096, 2),
        "fresh process requires IQ_FUSED_GATE_UP1 IQ_GPU_BOUND1 IQ_TYPED161 IQ_SIGNED_FACTOR1"
    );
}

#[stark_test]
fn iq2_fused_r4_r2_safe_caches_pair_distinct_output_tiles() {
    let e = executor().unwrap();
    let mut identities = [None, None];
    for r2 in [false, true, false, true] {
        let (pipeline, tile) = e.iq_fused_gate_up_pipeline(r2).unwrap();
        assert_eq!(
            tile,
            if r2 {
                4
            } else {
                8
            },
            "IQ2 fused PSO/output-tile pairing"
        );
        assert_eq!(pipeline.thread_execution_width(), 32);
        assert!(pipeline.max_total_threads_per_threadgroup() >= 64);
        let identity = Retained::as_ptr(&pipeline) as usize;
        let previous = &mut identities[usize::from(r2)];
        if let Some(previous) = previous {
            assert_eq!(*previous, identity, "IQ2 fused schedule cache changed its PSO");
        } else {
            *previous = Some(identity);
        }
    }
    assert_ne!(identities[0], identities[1], "IQ2 R4/R2 schedules shared a format-only cache entry");
    for cache in [&e.iq_fused_gate_up_pipelines, &e.iq_fused_gate_up_r2_pipelines] {
        assert!(cache.lock().unwrap().contains_key(&MatrixFormat::Packed(17)));
    }
}
fn encoded(seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = vec![0u8; SHAPE[0] * SHAPE[1] * SHAPE[2] / 256 * 74];
    for b in &mut bytes {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    for block in bytes.chunks_exact_mut(74) {
        block[..2].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
    }
    bytes
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
    let result = DecodeGraph::zeros(values.len()).unwrap();
    unsafe {
        result.buffer.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(values.as_ptr(), values.len());
    }
    result
}
fn baseline(
    gate: &MappedMatrix,
    up: &MappedMatrix,
    values: &[f32],
    ids: &[u32],
) -> Result<Vec<f32>> {
    let mut graph = DecodeGraph::new()?;
    let source = input(values);
    let indices = DecodeGraph::upload(&ids.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
    let g = graph.indexed(gate, MatrixFormat::Packed(17), SHAPE, &indices, ids.len(), &source, false)?;
    let u = graph.indexed(up, MatrixFormat::Packed(17), SHAPE, &indices, ids.len(), &source, false)?;
    let output = graph.swiglu(&g, &u, Some(11.0))?;
    Ok(graph.finish(&[output])?.remove(0))
}
fn actual(
    gate: &MappedMatrix,
    up: &MappedMatrix,
    values: &[f32],
    ids: &[u32],
    unretained: bool,
) -> Result<(Vec<f32>, bool)> {
    let mut graph = if unretained {
        DecodeGraph::new_unretained()?
    } else {
        DecodeGraph::new()?
    };
    let source = input(values);
    let indices = DecodeGraph::upload(&ids.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
    let output = graph.try_iq2_gate_up(
        gate,
        MatrixFormat::Packed(17),
        SHAPE,
        up,
        MatrixFormat::Packed(17),
        SHAPE,
        &indices,
        ids.len(),
        &source,
        11.0,
    )?;
    let fused = output.is_some();
    let output = if let Some(output) = output {
        output
    } else {
        let g = graph.indexed(gate, MatrixFormat::Packed(17), SHAPE, &indices, ids.len(), &source, false)?;
        let u = graph.indexed(up, MatrixFormat::Packed(17), SHAPE, &indices, ids.len(), &source, false)?;
        graph.swiglu(&g, &u, Some(11.0))?
    };
    Ok((graph.finish(&[output])?.remove(0), fused))
}

#[stark_test]
fn iq2_fused_route_unsupported_descriptors_preserve_legacy_admission() {
    let tiny = matrix(&[0u8; 2], 0);
    let odd = matrix(&[0u8; 2], 1);
    let source = DecodeGraph::zeros(4096).unwrap();
    let ids = DecodeGraph::upload(&[0.0]).unwrap();
    let mut graph = DecodeGraph::new().unwrap();
    let before = graph.resources.len();
    assert!(
        graph
            .try_iq2_gate_up(
                &tiny,
                MatrixFormat::Packed(18),
                SHAPE,
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &ids,
                1,
                &source,
                11.0
            )
            .unwrap()
            .is_none()
    );
    assert!(
        graph
            .try_iq2_gate_up(
                &tiny,
                MatrixFormat::Packed(17),
                [8, 13, 4096],
                &tiny,
                MatrixFormat::Packed(17),
                [8, 13, 4096],
                &ids,
                1,
                &source,
                11.0
            )
            .unwrap()
            .is_none()
    );
    assert!(
        graph
            .try_iq2_gate_up(
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &tiny,
                MatrixFormat::Packed(17),
                [4, 2048, 4096],
                &ids,
                1,
                &source,
                11.0
            )
            .unwrap()
            .is_none()
    );
    assert!(
        graph
            .try_iq2_gate_up(
                &odd,
                MatrixFormat::Packed(17),
                SHAPE,
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &ids,
                1,
                &source,
                11.0
            )
            .unwrap()
            .is_none()
    );
    assert!(
        graph
            .try_iq2_gate_up(
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &odd,
                MatrixFormat::Packed(17),
                SHAPE,
                &ids,
                1,
                &source,
                11.0
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(graph.resources.len(), before, "unsupported fusion encoded work");
    graph.begin_projection_level().unwrap();
    assert!(
        graph
            .try_iq2_gate_up(
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &ids,
                1,
                &source,
                11.0
            )
            .unwrap()
            .is_none()
    );
    graph.end_projection_level().unwrap();
    graph.finish(&[]).unwrap();
}

#[stark_test]
#[ignore]
fn iq2_fused_actual_route_matches_two_gemvs_alignment_and_failure_recovery() {
    admitted_flags();
    let gate_bytes = encoded(0x9135aced);
    let up_bytes = encoded(0x874b19d3);
    let ids = [7u32, 0, 5, 3, 1, 7, 2, 6];
    let values = (0..4096).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    for (go, uo) in [(2usize, 2usize), (1, 2), (2, 1), (4, 2), (2, 4)] {
        let gate = matrix(&gate_bytes, go);
        let up = matrix(&up_bytes, uo);
        let expected = baseline(&gate, &up, &values, &ids).unwrap();
        for unretained in [false, true] {
            let (result, fused) = actual(&gate, &up, &values, &ids, unretained).unwrap();
            assert_eq!(fused, go.is_multiple_of(2) && uo.is_multiple_of(2), "IQ2 fusion alignment admission");
            assert!(
                result.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                "IQ2 fused actual route/source parity"
            );
        }
    }
    let gate = matrix(&gate_bytes, 2);
    let up = matrix(&up_bytes, 2);
    let large = values.iter().map(|v| v * 1e21).collect::<Vec<_>>();
    let expected = baseline(&gate, &up, &large, &ids).unwrap();
    let (actual, fused) = actual(&gate, &up, &large, &ids, true).unwrap();
    assert!(fused);
    assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()), "IQ2 fused actual extreme fallback");
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut invalid = values.clone();
        invalid[4095] = poison;
        assert_eq!(
            baseline(&gate, &up, &invalid, &ids).is_err(),
            self::actual(&gate, &up, &invalid, &ids, true).is_err(),
            "IQ2 fused actual input failure parity"
        );
    }
    assert!(self::actual(&gate, &up, &values, &[u32::MAX], true).is_err(), "IQ2 fused actual invalid expert escaped");
    let mut invalid_gate = gate_bytes.clone();
    for block in invalid_gate.chunks_exact_mut(74) {
        block[..2].copy_from_slice(&half::f16::INFINITY.to_bits().to_le_bytes());
        block[2..].fill(0);
    }
    let ordinary = matrix(&vec![0u8; gate_bytes.len()], 2);
    let invalid = matrix(&invalid_gate, 2);
    assert!(
        self::actual(&invalid, &ordinary, &vec![1.0; 4096], &ids, true).is_err(),
        "IQ2 fused gate clamp masked nonfinite projection"
    );
    assert!(
        self::actual(&ordinary, &invalid, &vec![1.0; 4096], &ids, true).is_err(),
        "IQ2 fused up clamp masked nonfinite projection"
    );
    let maximum = vec![f32::MAX; 4096];
    let (expected, _) = self::actual(&ordinary, &ordinary, &maximum, &ids, true).unwrap();
    assert!(expected.iter().all(|v| *v == 0.0), "IQ2 fused actual zero times MAX");
    assert!(self::actual(&gate, &up, &values, &ids, true).is_ok(), "IQ2 fused actual recovery");
    let e = executor().unwrap();
    let selected_cache = if iq_fused_gate_up_r2_enabled() {
        &e.iq_fused_gate_up_r2_pipelines
    } else {
        &e.iq_fused_gate_up_pipelines
    };
    assert!(selected_cache.lock().unwrap().contains_key(&MatrixFormat::Packed(17)));
    assert!(e.iq_bound_factor_indexed_pipelines.lock().unwrap().contains_key(&MatrixFormat::Packed(17)));
    assert!(e.iq_bound_indexed_pipelines.lock().unwrap().contains_key(&MatrixFormat::Packed(17)));
}

#[stark_test]
#[ignore]
fn iq2_fused_queued_unretained_graphs_retain_both_bank_owners() {
    admitted_flags();
    let gate = matrix(&encoded(1), 2);
    let up = matrix(&encoded(2), 2);
    let ids = [7u32, 0, 5, 3, 1, 7, 2, 6];
    let first = (0..4096).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let second = (0..4096).map(|i| (i as f32 * 0.019).cos()).collect::<Vec<_>>();
    let expected = [baseline(&gate, &up, &first, &ids).unwrap(), baseline(&gate, &up, &second, &ids).unwrap()];
    let gate_id = Retained::as_ptr(&gate.codes.buffer) as usize;
    let up_id = Retained::as_ptr(&up.codes.buffer) as usize;
    let mut pending = Vec::new();
    for values in [&first, &second] {
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let source = input(values);
        let indices = DecodeGraph::upload(&ids.map(f32::from_bits)).unwrap();
        let result = graph
            .try_iq2_gate_up(
                &gate,
                MatrixFormat::Packed(17),
                SHAPE,
                &up,
                MatrixFormat::Packed(17),
                SHAPE,
                &indices,
                ids.len(),
                &source,
                11.0,
            )
            .unwrap()
            .unwrap();
        for id in
            [gate_id, up_id, Retained::as_ptr(&source.buffer) as usize, Retained::as_ptr(&indices.buffer) as usize]
        {
            assert!(
                graph.resources.iter().any(|b| Retained::as_ptr(b) as usize == id),
                "IQ2 fused raw binding lacks graph ownership"
            );
        }
        graph.validate_queued_submission().unwrap();
        pending.push(graph.submit(&[result]).unwrap());
        drop(source);
        drop(indices);
    }
    drop(gate);
    drop(up);
    for (pending, expected) in pending.into_iter().zip(expected) {
        let actual = pending.wait().unwrap().remove(0);
        assert!(
            actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
            "IQ2 fused queued owner/readout mismatch"
        );
    }
}

/// Run in a fresh process with the existing fusion/typed/bound/factor flags and
/// IQ_FUSED_GATE_UP_R2=1. Reuse the full actual-route contracts for the new PSO,
/// including dropped input/bank owners in two queued unretained commands.
#[stark_test]
#[ignore]
fn iq2_fused_r2_actual_route_preserves_parity_failures_and_queued_lifetimes() {
    admitted_flags();
    assert!(iq_fused_gate_up_r2_enabled(), "fresh process requires IQ_FUSED_GATE_UP_R2=1");
    iq2_fused_actual_route_matches_two_gemvs_alignment_and_failure_recovery();
    iq2_fused_queued_unretained_graphs_retain_both_bank_owners();
}

/// R2 is a subordinate policy: requesting it must not activate fusion itself.
#[stark_test]
#[ignore]
fn iq2_fused_r2_alone_does_not_admit_fusion() {
    assert!(std::env::var_os("STARK_GLM53_IQ_FUSED_GATE_UP_R2").is_some_and(|v| v == "1"));
    assert!(!iq_fused_gate_up_enabled(), "fresh process requires IQ_FUSED_GATE_UP=0");
    assert!(!iq_fused_gate_up_r2_enabled());
    let tiny = matrix(&[0u8; 2], 2);
    let source = DecodeGraph::zeros(4096).unwrap();
    let ids = DecodeGraph::upload(&[0.0]).unwrap();
    let mut graph = DecodeGraph::new().unwrap();
    let before = graph.resources.len();
    assert!(
        graph
            .try_iq2_gate_up(
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &tiny,
                MatrixFormat::Packed(17),
                SHAPE,
                &ids,
                1,
                &source,
                11.0,
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(graph.resources.len(), before, "R2 alone encoded fused work");
    graph.finish(&[]).unwrap();
}
