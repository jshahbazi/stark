// SPDX-License-Identifier: Apache-2.0
//! Bitwise gates for the Edge0 direct-route schedule. Synthetic vectors stay in
//! memory and the grouped schedule's own kernels supply the expected F32 bits.
//! This is operator and composition coverage. Whole-model agreement comes from
//! the `STARK_EDGE0_DIRECT_ROUTES=verify` selector on the actual artifact.

use std::collections::BTreeMap;

use stark_engine_macros::stark_test;

use super::*;

fn matrix(
    bytes: &[u8],
    offset: usize,
) -> MappedMatrix {
    let mut storage = vec![0x55; offset];
    storage.extend_from_slice(bytes);
    storage.extend_from_slice(&[0xaa; 7]);
    MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor().unwrap(), &storage).unwrap(),
            offset,
            length: bytes.len(),
            registered_length: storage.len(),
        },
        planes: None,
    }
}

fn codes(
    kind: u32,
    rows: usize,
    columns: usize,
    seed: usize,
) -> Vec<u8> {
    let (block, bytes) = weights::block_layout(kind).unwrap();
    let mut codes = (0..rows * (columns / block) * bytes).map(|i| (i * 37 + seed) as u8).collect::<Vec<_>>();
    for chunk in codes.chunks_exact_mut(bytes) {
        let scale = if kind == 11 {
            108
        } else {
            0
        };
        chunk[scale..scale + 2].copy_from_slice(&0x2800u16.to_le_bytes());
    }
    codes
}

fn packed(
    kind: u32,
    rows: usize,
    columns: usize,
    seed: usize,
) -> MappedMatrix {
    matrix(&codes(kind, rows, columns, seed), 7 + seed % 5)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn pack(
    graph: &mut DecodeGraph,
    rows: &[DeviceVector],
) -> DeviceVector {
    let width = rows[0].len;
    let result = graph.scratch(width * rows.len()).unwrap();
    for (index, row) in rows.iter().enumerate() {
        graph.copy_into(row, &result.slice(index * width, width).unwrap()).unwrap();
    }
    result
}

#[stark_test]
fn edge0_direct_token_rows_match_batched_kernels_bitwise() {
    let rows = 13;
    for columns in [512, 1792] {
        for tokens in [1, 2, 3, 8] {
            let input = (0..tokens * columns).map(|i| ((i * 17 % 43) as f32 - 21.0) * 0.047).collect::<Vec<_>>();
            let source = DecodeGraph::upload(&input).unwrap();
            let scale =
                DecodeGraph::upload(&(0..rows).map(|i| 0.5 + (i % 7) as f32 * 0.37).collect::<Vec<_>>()).unwrap();
            for kind in [8, 11, 20] {
                let format = MatrixFormat::Packed(kind);
                let mapped = packed(kind, rows, columns, 19);
                let mut graph = DecodeGraph::new().unwrap();
                let mut pairs = Vec::new();
                let generic =
                    graph.edge0_packed_batch_selected(&mapped, format, rows, columns, &source, tokens, false).unwrap();
                let grouped =
                    graph.edge0_packed_batch_selected(&mapped, format, rows, columns, &source, tokens, true).unwrap();
                let split = graph.edge0_packed_split(&mapped, format, rows, columns, &source, tokens).unwrap();
                pairs.push((generic, split.clone()));
                pairs.push((grouped.clone(), split));
                let scaled = graph.edge0_column_scale(&grouped, &scale, rows, tokens).unwrap();
                let direct = graph.scratch(rows * tokens).unwrap();
                for token in 0..tokens {
                    graph
                        .edge0_packed_single_into(
                            &mapped,
                            format,
                            rows,
                            columns,
                            &source.slice(token * columns, columns).unwrap(),
                            &direct.slice(token * rows, rows).unwrap(),
                            Some(&scale),
                            false,
                        )
                        .unwrap();
                }
                pairs.push((scaled, direct));
                if kind == 11 {
                    let up = packed(kind, rows, columns, 73);
                    let fused = graph
                        .edge0_gate_up_swiglu_batch_selected(&mapped, &up, rows, columns, &source, tokens, true)
                        .unwrap();
                    for token in 0..tokens {
                        let single = graph
                            .edge0_gate_up_swiglu_single(
                                &mapped,
                                &up,
                                rows,
                                columns,
                                &source.slice(token * columns, columns).unwrap(),
                                false,
                            )
                            .unwrap();
                        pairs.push((fused.slice(token * rows, rows).unwrap(), single));
                    }
                }
                for (expected, actual) in &pairs {
                    graph.edge0_require_identical(expected, actual).unwrap();
                }
                let outputs = pairs.iter().flat_map(|(left, right)| [left.clone(), right.clone()]).collect::<Vec<_>>();
                let values = graph.finish(&outputs).unwrap();
                for pair in values.chunks_exact(2) {
                    assert_eq!(bits(&pair[0]), bits(&pair[1]), "Edge0 direct token row changed F32 result");
                }
            }
        }
    }
}

#[stark_test]
fn edge0_direct_route_composition_matches_grouped_schedule_bitwise() {
    let (dim, latent, hidden, active) = (512, 256, 32, 3);
    let labels = [0usize, 1, 0, 2, 1];
    let down_projections = (0..3).map(|group| packed(8, latent, dim, 11 + group)).collect::<Vec<_>>();
    let up_projections = (0..3).map(|group| packed(8, dim, latent, 31 + group)).collect::<Vec<_>>();
    let gates = (0..labels.len()).map(|expert| packed(11, hidden, latent, 53 + expert)).collect::<Vec<_>>();
    let ups = (0..labels.len()).map(|expert| packed(11, hidden, latent, 71 + expert)).collect::<Vec<_>>();
    let downs = (0..labels.len()).map(|expert| packed(20, latent, hidden, 97 + expert)).collect::<Vec<_>>();
    let scale = DecodeGraph::upload(&(0..latent).map(|i| 0.75 + (i % 11) as f32 * 0.21).collect::<Vec<_>>()).unwrap();
    for tokens in [1usize, 3, 8] {
        let input = (0..tokens * dim).map(|i| ((i * 29 % 61) as f32 - 30.0) * 0.011).collect::<Vec<_>>();
        let inputs = DecodeGraph::upload(&input).unwrap();
        let mut expert_jobs: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        let mut group_jobs: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for token in 0..tokens {
            for (route, expert) in [token % 5, (token + 1) % 5, (token + 3) % 5].into_iter().enumerate() {
                expert_jobs.entry(expert).or_default().push((token, route));
                group_jobs.entry(labels[expert]).or_default().push((token, route));
            }
        }
        let mut graph = DecodeGraph::new().unwrap();
        graph.set_scratch_limit(64 * 1024 * 1024).unwrap();

        // Grouped schedule: batched kernels with gathered and scattered rows.
        let mut grouped_inputs = BTreeMap::new();
        for &group in group_jobs.keys() {
            let projected = graph
                .edge0_packed_batch_selected(
                    &down_projections[group],
                    MatrixFormat::Packed(8),
                    latent,
                    dim,
                    &inputs,
                    tokens,
                    true,
                )
                .unwrap();
            grouped_inputs.insert(group, graph.edge0_column_scale(&projected, &scale, latent, tokens).unwrap());
        }
        let grouped_latent = graph.scratch(tokens * active * latent).unwrap();
        for (&expert, jobs) in &expert_jobs {
            let rows = jobs
                .iter()
                .map(|&(token, _)| grouped_inputs[&labels[expert]].slice(token * latent, latent).unwrap())
                .collect::<Vec<_>>();
            let selected = pack(&mut graph, &rows);
            let values = graph
                .edge0_gate_up_swiglu_batch_selected(
                    &gates[expert],
                    &ups[expert],
                    hidden,
                    latent,
                    &selected,
                    jobs.len(),
                    true,
                )
                .unwrap();
            let down = graph
                .edge0_packed_batch_selected(
                    &downs[expert],
                    MatrixFormat::Packed(20),
                    latent,
                    hidden,
                    &values,
                    jobs.len(),
                    true,
                )
                .unwrap();
            for (index, &(token, route)) in jobs.iter().enumerate() {
                graph
                    .copy_into(
                        &down.slice(index * latent, latent).unwrap(),
                        &grouped_latent.slice((token * active + route) * latent, latent).unwrap(),
                    )
                    .unwrap();
            }
        }
        let grouped_expanded = graph.scratch(tokens * active * dim).unwrap();
        for (&group, jobs) in &group_jobs {
            for batch in jobs.chunks(8) {
                let rows = batch
                    .iter()
                    .map(|&(token, route)| grouped_latent.slice((token * active + route) * latent, latent).unwrap())
                    .collect::<Vec<_>>();
                let selected = pack(&mut graph, &rows);
                let output = graph
                    .edge0_packed_batch_selected(
                        &up_projections[group],
                        MatrixFormat::Packed(8),
                        dim,
                        latent,
                        &selected,
                        batch.len(),
                        true,
                    )
                    .unwrap();
                for (index, &(token, route)) in batch.iter().enumerate() {
                    graph
                        .copy_into(
                            &output.slice(index * dim, dim).unwrap(),
                            &grouped_expanded.slice((token * active + route) * dim, dim).unwrap(),
                        )
                        .unwrap();
                }
            }
        }

        // Direct schedule: one-token kernels on the route's own slots.
        let mut direct_inputs = BTreeMap::new();
        for (&group, jobs) in &group_jobs {
            let projected = graph.scratch(tokens * latent).unwrap();
            let mut written = [false; 8];
            for &(token, _) in jobs {
                if std::mem::replace(&mut written[token], true) {
                    continue;
                }
                graph
                    .edge0_packed_single_into(
                        &down_projections[group],
                        MatrixFormat::Packed(8),
                        latent,
                        dim,
                        &inputs.slice(token * dim, dim).unwrap(),
                        &projected.slice(token * latent, latent).unwrap(),
                        Some(&scale),
                        false,
                    )
                    .unwrap();
            }
            direct_inputs.insert(group, projected);
        }
        let direct_latent = graph.scratch(tokens * active * latent).unwrap();
        for (&expert, jobs) in &expert_jobs {
            for &(token, route) in jobs {
                let values = graph
                    .edge0_gate_up_swiglu_single(
                        &gates[expert],
                        &ups[expert],
                        hidden,
                        latent,
                        &direct_inputs[&labels[expert]].slice(token * latent, latent).unwrap(),
                        false,
                    )
                    .unwrap();
                graph
                    .edge0_packed_single_into(
                        &downs[expert],
                        MatrixFormat::Packed(20),
                        latent,
                        hidden,
                        &values,
                        &direct_latent.slice((token * active + route) * latent, latent).unwrap(),
                        None,
                        false,
                    )
                    .unwrap();
            }
        }
        let direct_expanded = graph.scratch(tokens * active * dim).unwrap();
        for (&group, jobs) in &group_jobs {
            for &(token, route) in jobs {
                let slot = token * active + route;
                graph
                    .edge0_packed_single_into(
                        &up_projections[group],
                        MatrixFormat::Packed(8),
                        dim,
                        latent,
                        &direct_latent.slice(slot * latent, latent).unwrap(),
                        &direct_expanded.slice(slot * dim, dim).unwrap(),
                        None,
                        false,
                    )
                    .unwrap();
            }
        }
        graph.edge0_require_identical(&grouped_latent, &direct_latent).unwrap();
        graph.edge0_require_identical(&grouped_expanded, &direct_expanded).unwrap();
        let values = graph.finish(&[grouped_latent, direct_latent, grouped_expanded, direct_expanded]).unwrap();
        assert_eq!(bits(&values[0]), bits(&values[1]), "Edge0 direct route changed latent expert rows");
        assert_eq!(bits(&values[2]), bits(&values[3]), "Edge0 direct route changed expanded rows");
        assert!(values[2].iter().any(|value| *value != 0.0), "Edge0 direct route fixture is degenerate");
    }
}

#[stark_test]
fn edge0_direct_rows_reject_poison_bad_scales_and_bad_shapes() {
    let scale = DecodeGraph::upload(&[1.5]).unwrap();
    for kind in [8, 11, 20] {
        let format = MatrixFormat::Packed(kind);
        let mapped = packed(kind, 1, 256, 3);
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for scaled in [false, true] {
                let input = DecodeGraph::zeros(256).unwrap();
                // SAFETY: fresh owned storage has no pending GPU users.
                unsafe {
                    input.buffer.contents().as_ptr().cast::<f32>().add(255).write(poison);
                }
                let mut graph = DecodeGraph::new().unwrap();
                let output = graph.scratch(1).unwrap();
                // No separate scan: the row result itself must record failure.
                graph
                    .edge0_packed_single_into(&mapped, format, 1, 256, &input, &output, scaled.then_some(&scale), false)
                    .unwrap();
                assert!(graph.finish(&[output]).is_err(), "Edge0 direct row concealed nonfinite input");
            }
            if kind == 11 {
                let input = DecodeGraph::zeros(256).unwrap();
                // SAFETY: fresh owned storage has no pending GPU users.
                unsafe {
                    input.buffer.contents().as_ptr().cast::<f32>().add(255).write(poison);
                }
                let mut graph = DecodeGraph::new().unwrap();
                let output = graph.edge0_gate_up_swiglu_single(&mapped, &mapped, 1, 256, &input, false).unwrap();
                assert!(graph.finish(&[output]).is_err(), "Edge0 direct fused row concealed nonfinite input");
            }
        }
        let input = DecodeGraph::upload(&[0.25; 256]).unwrap();
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let stored = DecodeGraph::zeros(1).unwrap();
            // SAFETY: fresh owned storage has no pending GPU users.
            unsafe {
                stored.buffer.contents().as_ptr().cast::<f32>().write(bad);
            }
            let mut graph = DecodeGraph::new().unwrap();
            let output = graph.scratch(1).unwrap();
            graph.edge0_packed_single_into(&mapped, format, 1, 256, &input, &output, Some(&stored), false).unwrap();
            assert!(graph.finish(&[output]).is_err(), "Edge0 direct row accepted invalid stored scale");
        }
        let mut graph = DecodeGraph::new().unwrap();
        let output = graph.scratch(2).unwrap();
        let wide_scale = DecodeGraph::upload(&[1.0, 1.0]).unwrap();
        let mut truncated = mapped.clone();
        truncated.codes.length -= 1;
        assert!(graph.edge0_packed_single_into(&mapped, format, 1, 256, &input, &output, None, false).is_err());
        let row = output.slice(0, 1).unwrap();
        assert!(
            graph.edge0_packed_single_into(&mapped, format, 1, 256, &input, &row, Some(&wide_scale), false).is_err()
        );
        assert!(graph.edge0_packed_single_into(&truncated, format, 1, 256, &input, &row, None, false).is_err());
        assert!(
            graph
                .edge0_packed_single_into(&mapped, MatrixFormat::Packed(30), 1, 256, &input, &row, None, false)
                .is_err()
        );
        let shared = graph.scratch(512).unwrap();
        assert!(
            graph
                .edge0_packed_single_into(
                    &mapped,
                    format,
                    1,
                    256,
                    &shared.slice(0, 256).unwrap(),
                    &shared.slice(255, 1).unwrap(),
                    None,
                    false,
                )
                .is_err(),
            "Edge0 direct row accepted an aliased output"
        );
        if kind == 11 {
            assert!(graph.edge0_gate_up_swiglu_single(&mapped, &truncated, 1, 256, &input, false).is_err());
        }
        graph.finish(&[]).unwrap();
    }
}

#[stark_test]
fn edge0_schedule_verification_rejects_any_bit_difference() {
    let values = (0..300).map(|i| (i as f32 - 150.0) * 0.125).collect::<Vec<_>>();
    assert!(values[150] == 0.0, "fixture keeps a signed-zero case");
    let left = DecodeGraph::upload(&values).unwrap();
    for (index, replacement) in [(None, 0.0), (Some(299), f32::from_bits(values[299].to_bits() + 1)), (Some(150), -0.0)]
    {
        let mut changed = values.clone();
        if let Some(index) = index {
            changed[index] = replacement;
        }
        let right = DecodeGraph::upload(&changed).unwrap();
        for deferred in [false, true] {
            let mut graph = DecodeGraph::new().unwrap();
            if deferred {
                graph.begin_projection_level().unwrap();
                graph.edge0_require_identical(&left, &right).unwrap();
                graph.end_projection_level().unwrap();
                graph.edge0_flush_verification().unwrap();
            } else {
                graph.edge0_require_identical(&left, &right).unwrap();
            }
            assert!(
                graph.finish(&[]).is_ok() == index.is_none(),
                "Edge0 schedule verification missed or invented a bit difference"
            );
        }
    }
    let mut graph = DecodeGraph::new().unwrap();
    assert!(graph.edge0_require_identical(&left, &left.slice(0, 299).unwrap()).is_err());
    graph.begin_projection_level().unwrap();
    graph.edge0_require_identical(&left, &left).unwrap();
    graph.end_projection_level().unwrap();
    assert!(graph.finish(&[]).is_err(), "Edge0 schedule verification dropped a deferred comparison");
}
