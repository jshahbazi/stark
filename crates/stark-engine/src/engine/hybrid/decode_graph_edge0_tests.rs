// SPDX-License-Identifier: Apache-2.0
//! Independent numerical gates for the Phase 4 packed device operators.
//! Private synthetic vectors stay in memory. The sealed Phase 3 F64 oracle
//! supplies expected values; this is operator/composition coverage, not a
//! production mapped-owner, trunk or whole-model equivalence claim.

use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
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
            // An initialized, owned Metal copy, not a rounded Vec mapping.
            buffer: buffer(executor().unwrap(), &storage).unwrap(),
            offset,
            length: bytes.len(),
            registered_length: storage.len(),
        },
        planes: None,
    }
}

fn sealed(
    root: &Path,
    entry: &Value,
) -> Vec<u8> {
    let name = entry["path"].as_str().unwrap();
    assert!(Path::new(name).file_name().is_some_and(|part| part == name), "fixture relative path");
    let bytes = fs::read(root.join(name)).expect("sealed fixture file unavailable");
    assert!(bytes.len() <= 8 * 1024 * 1024, "fixture file bound");
    assert!(bytes.len() as u64 == entry["bytes"].as_u64().unwrap(), "fixture byte count mismatch");
    assert!(format!("{:x}", Sha256::digest(&bytes)) == entry["sha256"].as_str().unwrap(), "fixture file hash mismatch");
    bytes
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

fn f64s(bytes: &[u8]) -> Vec<f64> {
    assert!(bytes.len().is_multiple_of(8));
    bytes.chunks_exact(8).map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap())).collect()
}

fn u32s(bytes: &[u8]) -> Vec<usize> {
    assert!(bytes.len().is_multiple_of(4));
    bytes.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as usize).collect()
}

struct StoredMatrix {
    kind: u32,
    rows: usize,
    columns: usize,
    mapped: MappedMatrix,
}

struct Fixture {
    root: PathBuf,
    expected: Value,
    matrices: BTreeMap<String, StoredMatrix>,
    x: Vec<f32>,
    scales: Vec<f32>,
    labels: Vec<usize>,
    routes: Vec<usize>,
    probabilities: Vec<f32>,
}

impl Fixture {
    fn load() -> Self {
        let root = std::env::var_os("STARK_EDGE0_FORWARD_FIXTURE")
            .map(PathBuf::from)
            .expect("registered private Edge0 fixture path required");
        let input_bytes = fs::read(root.join("inputs.json")).expect("fixture manifest unavailable");
        let expected_bytes = fs::read(root.join("expected.json")).expect("fixture expected manifest unavailable");
        assert!(
            format!("{:x}", Sha256::digest(&input_bytes))
                == "17469b3bbe73570fcc259880012fd23f75996325c76e7e0385e1f07a0e97071a",
            "fixture input seal mismatch"
        );
        assert!(
            format!("{:x}", Sha256::digest(&expected_bytes))
                == "40f2b8e62b9c071eebefd0c73095c8980952b11bf293232380f4b40cb446f2d4",
            "fixture expected seal mismatch"
        );
        let inputs: Value = serde_json::from_slice(&input_bytes).unwrap();
        let expected = serde_json::from_slice(&expected_bytes).unwrap();
        let mut matrices = BTreeMap::new();
        for entry in inputs["matrices"].as_array().unwrap() {
            let bytes = sealed(&root, entry);
            matrices.insert(
                entry["name"].as_str().unwrap().to_owned(),
                StoredMatrix {
                    kind: entry["type"].as_u64().unwrap() as u32,
                    rows: entry["rows"].as_u64().unwrap() as usize,
                    columns: entry["cols"].as_u64().unwrap() as usize,
                    mapped: matrix(&bytes, 0),
                },
            );
        }
        Self {
            x: f32s(&sealed(&root, &inputs["arrays"]["x.f32"])),
            scales: f32s(&sealed(&root, &inputs["arrays"]["scales.f32"])),
            labels: u32s(&sealed(&root, &inputs["arrays"]["labels.u32"])),
            routes: u32s(&sealed(&root, &inputs["arrays"]["routes.u32"])),
            probabilities: f32s(&sealed(&root, &inputs["arrays"]["route_weights.f32"])),
            root,
            expected,
            matrices,
        }
    }

    fn expected(
        &self,
        tag: &str,
    ) -> Vec<f64> {
        f64s(&sealed(&self.root, &self.expected["arrays"][tag]))
    }

    fn project(
        &self,
        graph: &mut DecodeGraph,
        name: &str,
        input: &DeviceVector,
        tokens: usize,
    ) -> DeviceVector {
        let source = &self.matrices[name];
        graph
            .edge0_packed_batch(
                &source.mapped,
                MatrixFormat::Packed(source.kind),
                source.rows,
                source.columns,
                input,
                tokens,
            )
            .unwrap()
    }
}

enum Bound {
    Matrix(String),
    Composition(Vec<f64>),
}

struct Capture {
    output: DeviceVector,
    bound: Bound,
}

fn capture_rows(
    captures: &mut Vec<Capture>,
    values: &DeviceVector,
    width: usize,
    tags: &[String],
) {
    for (index, tag) in tags.iter().enumerate() {
        captures.push(Capture {
            output: values.slice(index * width, width).unwrap(),
            bound: Bound::Matrix(tag.clone()),
        });
    }
}

fn pack(
    graph: &mut DecodeGraph,
    rows: &[DeviceVector],
) -> DeviceVector {
    let width = rows[0].len;
    let result = graph.scratch(width * rows.len()).unwrap();
    for (index, row) in rows.iter().enumerate() {
        assert!(row.len == width);
        graph.copy_into(row, &result.slice(index * width, width).unwrap()).unwrap();
    }
    result
}

fn final_errors(
    actual: &[f32],
    expected: &[f64],
) -> (f64, f64) {
    assert!(
        actual.len() == expected.len() && actual.iter().all(|value| value.is_finite()),
        "finite composition output shape"
    );
    let (mut max, mut squared, mut norm) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&actual, &expected) in actual.iter().zip(expected) {
        let delta = f64::from(actual) - expected;
        max = max.max(delta.abs());
        squared += delta * delta;
        norm += expected * expected;
    }
    (max, (squared / norm.max(f64::MIN_POSITIVE)).sqrt())
}

#[stark_test]
#[ignore]
fn edge0_phase4_packed_composition_matches_registered_f64_for_token_batches() {
    let fixture = Fixture::load();
    let mut reports = Vec::new();
    for tokens in [1usize, 2, 3, 8] {
        let cases: Vec<usize> = (0..tokens).map(|token| token % 3).collect();
        let input =
            cases.iter().flat_map(|&case| fixture.x[case * 256..(case + 1) * 256].iter().copied()).collect::<Vec<_>>();
        let probabilities = cases
            .iter()
            .flat_map(|&case| fixture.probabilities[case * 3..(case + 1) * 3].iter().copied())
            .collect::<Vec<_>>();
        let mut graph = DecodeGraph::new().unwrap();
        graph.set_scratch_limit(64 * 1024 * 1024).unwrap();
        let input = DecodeGraph::upload(&input).unwrap();
        let scales = DecodeGraph::upload(&fixture.scales).unwrap();
        let probabilities = DecodeGraph::upload(&probabilities).unwrap();
        let mut captures = Vec::new();
        let mut down_groups = Vec::new();
        for group in 0..3 {
            let projected = fixture.project(&mut graph, &format!("d.{group}"), &input, tokens);
            let tags = cases.iter().map(|case| format!("t{case}.d.{group}")).collect::<Vec<_>>();
            capture_rows(&mut captures, &projected, 256, &tags);
            down_groups.push(graph.edge0_column_scale(&projected, &scales, 256, tokens).unwrap());
        }
        let mut expert_jobs: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        let mut group_jobs: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (token, &case) in cases.iter().enumerate() {
            for route in 0..3 {
                let expert = fixture.routes[case * 3 + route];
                expert_jobs.entry(expert).or_default().push((token, route));
                group_jobs.entry(fixture.labels[expert]).or_default().push((token, route));
            }
        }
        let latent = graph.scratch(tokens * 3 * 256).unwrap();
        for (expert, jobs) in expert_jobs {
            let group = fixture.labels[expert];
            let rows =
                jobs.iter().map(|&(token, _)| down_groups[group].slice(token * 256, 256).unwrap()).collect::<Vec<_>>();
            let selected_input = pack(&mut graph, &rows);
            let gate = fixture.project(&mut graph, &format!("gate.{expert}"), &selected_input, jobs.len());
            let up = fixture.project(&mut graph, &format!("up.{expert}"), &selected_input, jobs.len());
            for (role, values) in [("gate", &gate), ("up", &up)] {
                let tags =
                    jobs.iter().map(|&(token, route)| format!("t{}.r{route}.{role}", cases[token])).collect::<Vec<_>>();
                capture_rows(&mut captures, values, 256, &tags);
            }
            let g = &fixture.matrices[&format!("gate.{expert}")];
            let u = &fixture.matrices[&format!("up.{expert}")];
            let hidden =
                graph.edge0_gate_up_swiglu_batch(&g.mapped, &u.mapped, 256, 256, &selected_input, jobs.len()).unwrap();
            for (index, &(token, route)) in jobs.iter().enumerate() {
                let gate = fixture.expected(&format!("t{}.r{route}.gate", cases[token]));
                let up = fixture.expected(&format!("t{}.r{route}.up", cases[token]));
                let expected = gate.iter().zip(&up).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
                captures.push(Capture {
                    output: hidden.slice(index * 256, 256).unwrap(),
                    bound: Bound::Composition(expected),
                });
            }
            let down = fixture.project(&mut graph, &format!("down.{expert}"), &hidden, jobs.len());
            let tags =
                jobs.iter().map(|&(token, route)| format!("t{}.r{route}.down", cases[token])).collect::<Vec<_>>();
            capture_rows(&mut captures, &down, 256, &tags);
            for (index, &(token, route)) in jobs.iter().enumerate() {
                graph
                    .copy_into(
                        &down.slice(index * 256, 256).unwrap(),
                        &latent.slice((token * 3 + route) * 256, 256).unwrap(),
                    )
                    .unwrap();
            }
        }
        let experts = graph.scratch(tokens * 3 * 256).unwrap();
        for (group, jobs) in group_jobs {
            for batch in jobs.chunks(8) {
                let rows = batch
                    .iter()
                    .map(|&(token, route)| latent.slice((token * 3 + route) * 256, 256).unwrap())
                    .collect::<Vec<_>>();
                let input = pack(&mut graph, &rows);
                let expanded = fixture.project(&mut graph, &format!("u.{group}"), &input, batch.len());
                let tags =
                    batch.iter().map(|&(token, route)| format!("t{}.r{route}.u", cases[token])).collect::<Vec<_>>();
                capture_rows(&mut captures, &expanded, 256, &tags);
                for (index, &(token, route)) in batch.iter().enumerate() {
                    graph
                        .copy_into(
                            &expanded.slice(index * 256, 256).unwrap(),
                            &experts.slice((token * 3 + route) * 256, 256).unwrap(),
                        )
                        .unwrap();
                }
            }
        }
        let zeros = DecodeGraph::zeros(tokens * 256).unwrap();
        let zero_gates = DecodeGraph::zeros(tokens).unwrap();
        let routed = graph.edge0_weighted_sum(&experts, &probabilities, &zeros, &zero_gates, 256, 3, tokens).unwrap();
        for (token, &case) in cases.iter().enumerate() {
            captures.push(Capture {
                output: routed.slice(token * 256, 256).unwrap(),
                bound: Bound::Composition(fixture.expected(&format!("t{case}.latent"))),
            });
        }
        let shared_gate = fixture.project(&mut graph, "shared_gate", &input, tokens);
        let shared_up = fixture.project(&mut graph, "shared_up", &input, tokens);
        for (role, values) in [("shared_gate", &shared_gate), ("shared_up", &shared_up)] {
            let tags = cases.iter().map(|case| format!("t{case}.{role}")).collect::<Vec<_>>();
            capture_rows(&mut captures, values, 256, &tags);
        }
        let shared_hidden = graph.swiglu(&shared_gate, &shared_up, None).unwrap();
        let shared_down = fixture.project(&mut graph, "shared_down", &shared_hidden, tokens);
        let scalar = fixture.project(&mut graph, "shared_scalar", &input, tokens);
        for (role, values, width) in [("shared_down", &shared_down, 256), ("shared_scalar", &scalar, 1)] {
            let tags = cases.iter().map(|case| format!("t{case}.{role}")).collect::<Vec<_>>();
            capture_rows(&mut captures, values, width, &tags);
        }
        let combined =
            graph.edge0_weighted_sum(&experts, &probabilities, &shared_down, &scalar, 256, 3, tokens).unwrap();
        let readout = fixture.project(&mut graph, "readout", &combined, tokens);
        let tags = cases.iter().map(|case| format!("t{case}.readout")).collect::<Vec<_>>();
        capture_rows(&mut captures, &readout, 17, &tags);
        let high_water = graph.scratch_backing_bytes();
        let mut outputs = captures.iter().map(|capture| capture.output.clone()).collect::<Vec<_>>();
        outputs.push(readout);
        let mut actual = graph.finish(&outputs).unwrap();
        let readout = actual.pop().unwrap();
        let (mut matrix_calls, mut max_ratio, mut max_composition_abs, mut max_composition_l2) =
            (0usize, 0.0_f64, 0.0_f64, 0.0_f64);
        for (capture, values) in captures.iter().zip(actual) {
            match &capture.bound {
                Bound::Matrix(tag) => {
                    let reference = fixture.expected(tag);
                    let sumabs = fixture.expected(&format!("{tag}.sumabs"));
                    assert!(
                        values.len() == reference.len() && values.iter().all(|value| value.is_finite()),
                        "finite packed output shape"
                    );
                    for ((value, reference), products) in values.iter().zip(reference).zip(sumabs) {
                        let ratio = (f64::from(*value) - reference).abs() / (2e-5 * products.max(1.0));
                        max_ratio = max_ratio.max(ratio);
                        assert!(ratio <= 1.0, "Edge0 batched matrix exceeds preregistered F64 bound");
                    }
                    matrix_calls += 1;
                },
                Bound::Composition(reference) => {
                    let (abs, l2) = final_errors(&values, reference);
                    max_composition_abs = max_composition_abs.max(abs);
                    max_composition_l2 = max_composition_l2.max(l2);
                    assert!(abs <= 0.002 && l2 <= 0.0002, "Edge0 fused/latent component exceeds fixed F64 final bound");
                },
            }
        }
        assert!(matrix_calls == 20 * tokens, "fixture matrix coverage count");
        let expected = cases.iter().flat_map(|case| fixture.expected(&format!("t{case}.readout"))).collect::<Vec<_>>();
        let (abs, l2) = final_errors(&readout, &expected);
        assert!(abs <= 0.002 && l2 <= 0.0002, "Edge0 batched readout exceeds preregistered F64 final bound");
        reports.push(json!({"tokens":tokens,"matrix_operations":matrix_calls,"max_matrix_bound_ratio":max_ratio,
            "max_abs_readout_error":abs,"relative_l2_readout_error":l2,
            "max_abs_fused_or_latent_error":max_composition_abs,"maximum_fused_or_latent_relative_l2_error":max_composition_l2,
            "scratch_backing_bytes":high_water}));
    }
    let path = std::env::var_os("STARK_EDGE0_PHASE4_COMPONENT_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture.root.join("phase4-packed-component-report.json"));
    assert!(path.starts_with(&fixture.root), "private component report path");
    let report = json!({"attempt":"20261112-EDGE0-06","passed":true,
        "scope":"new packed batched/fused operators and synthetic latent/shared/readout composition; no production mapped-owner/trunk/model integration claim",
        "inputs_sha256":"17469b3bbe73570fcc259880012fd23f75996325c76e7e0385e1f07a0e97071a",
        "expected_sha256":"40f2b8e62b9c071eebefd0c73095c8980952b11bf293232380f4b40cb446f2d4",
        "test_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("decode_graph_edge0_tests.rs"))),
        "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("edge0_graph.metal"))),"reports":reports});
    let mut file =
        fs::OpenOptions::new().create_new(true).write(true).open(path).expect("new private aggregate report required");
    file.write_all(&serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

#[stark_test]
fn edge0_phase4_bf16_batches_and_row_tails_match_independent_f64() {
    let (rows, columns) = (13, 96);
    let words = (0..rows * columns)
        .map(|index| {
            let value = ((index * 37 % 61) as f32 - 30.0) * 0.03125;
            (value.to_bits() >> 16) as u16
        })
        .collect::<Vec<_>>();
    let bytes = words.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    let mapped = matrix(&bytes, 7);
    for tokens in [1usize, 2, 3, 8] {
        let input = (0..tokens * columns).map(|index| ((index * 17 % 43) as f32 - 21.0) * 0.047).collect::<Vec<_>>();
        let mut graph = DecodeGraph::new().unwrap();
        let source = DecodeGraph::upload(&input).unwrap();
        let output =
            graph.edge0_packed_batch(&mapped, MatrixFormat::Packed(30), rows, columns, &source, tokens).unwrap();
        let actual = graph.finish(&[output]).unwrap().remove(0);
        for token in 0..tokens {
            for row in 0..rows {
                let products = (0..columns)
                    .map(|col| {
                        f64::from(f32::from_bits(u32::from(words[row * columns + col]) << 16))
                            * f64::from(input[token * columns + col])
                    })
                    .collect::<Vec<_>>();
                let expected = products.iter().sum::<f64>();
                let bound = 2e-5 * products.iter().map(|value| value.abs()).sum::<f64>().max(1.0);
                assert!((f64::from(actual[token * rows + row]) - expected).abs() <= bound, "BF16 batched F64 bound");
            }
        }
    }
}

#[stark_test]
fn edge0_phase4_packed_bindings_reject_bad_shapes_and_nonfinite_inputs() {
    let mapped = matrix(&[0u8; 34], 0);
    let input = DecodeGraph::zeros(32).unwrap();
    let mut graph = DecodeGraph::new().unwrap();
    for tokens in [0usize, 9] {
        assert!(graph.edge0_packed_batch(&mapped, MatrixFormat::Packed(8), 1, 32, &input, tokens).is_err());
    }
    assert!(graph.edge0_packed_batch(&mapped, MatrixFormat::Packed(8), 2, 32, &input, 1).is_err());
    assert!(graph.edge0_packed_batch(&mapped, MatrixFormat::Packed(11), 1, 32, &input, 1).is_err());
    assert!(graph.edge0_gate_up_swiglu_batch(&mapped, &mapped, 1, 32, &input, 1).is_err());
    let mut invalid = mapped.clone();
    invalid.codes.length = 33;
    assert!(graph.edge0_packed_batch(&invalid, MatrixFormat::Packed(8), 1, 32, &input, 1).is_err());
    let mut invalid = mapped.clone();
    invalid.codes.offset = usize::MAX - 7;
    assert!(graph.edge0_packed_batch(&invalid, MatrixFormat::Packed(8), 1, 32, &input, 1).is_err());
    graph.finish(&[]).unwrap();
    for poison in [f32::NAN, f32::INFINITY] {
        let mut graph = DecodeGraph::new().unwrap();
        let input = DecodeGraph::zeros(32).unwrap();
        // Deliberately bypass upload's finite contract using initialized shared
        // Metal storage; sticky GPU failure must reject the completed command.
        unsafe {
            input.buffer.contents().as_ptr().cast::<f32>().write(poison);
        }
        let output = graph.edge0_packed_batch(&mapped, MatrixFormat::Packed(8), 1, 32, &input, 1).unwrap();
        assert!(graph.finish(&[output]).is_err(), "nonfinite packed input accepted");
    }
    let mut graph = DecodeGraph::new().unwrap();
    let input = DecodeGraph::upload(&[1.0; 32]).unwrap();
    let scale = DecodeGraph::zeros(32).unwrap();
    let output = graph.edge0_column_scale(&input, &scale, 32, 1).unwrap();
    assert!(graph.finish(&[output]).is_err(), "invalid stored column scale accepted");
}
