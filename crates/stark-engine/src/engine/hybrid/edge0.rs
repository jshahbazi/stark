//! Edge0's stateless latent expert stage for the F32 reference path.
//!
//! Independently implemented from Edge0-AI/Edge0
//! b7900cbb322926ccebdcd0c0c5a1280497ecff22: latent_moe.cpp, native_expert.cpp
//! and router_top10.cu. See LICENSE.strata-reference. No activation
//! quantization, grouped-U reassociation or persistent activation cache is used.

use super::{Geometry, matrix, project, projects};
use crate::engine::hybrid::{
    Error, Result,
    edge0_source::{EXPERT_WIDTH, EXPERTS_PER_LAYER, LATENT_WIDTH, MODEL_WIDTH, PROJECTION_GROUPS},
    math::{sigmoid, silu, topk},
    weights::{Edge0Matrix, Weights},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct LatentGeometry {
    pub dim: usize,
    pub latent: usize,
    pub hidden: usize,
    pub experts: usize,
    pub groups: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LatentMatrix {
    ProjectionDown(usize),
    ProjectionUp(usize),
    ExpertGate(usize),
    ExpertUp(usize),
    ExpertDown(usize),
}

/// The synthetic fixture supplies independent stored matrices through this
/// interface; production resolves each operation to a checked source span.
pub(super) trait LatentSource {
    fn geometry(&self) -> LatentGeometry;
    fn group(
        &self,
        expert: usize,
    ) -> Result<usize>;
    fn scales(&self) -> Result<&[f32]>;
    fn matvec(
        &self,
        matrix: LatentMatrix,
        input: &[f32],
    ) -> Result<Vec<f32>>;
}

struct NativeLayer<'a> {
    weights: &'a Weights,
    layer: usize,
}

impl LatentSource for NativeLayer<'_> {
    fn geometry(&self) -> LatentGeometry {
        LatentGeometry {
            dim: MODEL_WIDTH as usize,
            latent: LATENT_WIDTH as usize,
            hidden: EXPERT_WIDTH as usize,
            experts: EXPERTS_PER_LAYER,
            groups: PROJECTION_GROUPS,
        }
    }

    fn group(
        &self,
        expert: usize,
    ) -> Result<usize> {
        self.weights.edge0_group(self.layer, expert)
    }

    fn scales(&self) -> Result<&[f32]> {
        self.weights.edge0_column_scales(self.layer)
    }

    fn matvec(
        &self,
        matrix: LatentMatrix,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let layer = self.layer;
        let matrix = match matrix {
            LatentMatrix::ProjectionDown(group) => Edge0Matrix::ProjectionDown {
                layer,
                group,
            },
            LatentMatrix::ProjectionUp(group) => Edge0Matrix::ProjectionUp {
                layer,
                group,
            },
            LatentMatrix::ExpertGate(expert) => Edge0Matrix::ExpertGate {
                layer,
                expert,
            },
            LatentMatrix::ExpertUp(expert) => Edge0Matrix::ExpertUp {
                layer,
                expert,
            },
            LatentMatrix::ExpertDown(expert) => Edge0Matrix::ExpertDown {
                layer,
                expert,
            },
        };
        self.weights.edge0_matvec(matrix, input)
    }
}

/// Pinned Strata routing: full softmax with ascending F64 reductions, F32
/// reciprocal/probability boundaries, stable probability top-k and selected
/// F64 renormalization with the source's 2^-14 lower clamp.
pub(super) fn route(
    scores: &[f32],
    active: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    if scores.is_empty()
        || scores.len() > 4096
        || active == 0
        || active > scores.len()
        || scores.iter().any(|value| !value.is_finite())
    {
        return Err(Error::Invalid("Edge0 router geometry or values"));
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f64> = scores.iter().map(|&value| (f64::from(value) - f64::from(max)).exp()).collect();
    let sum = exponentials.iter().sum::<f64>();
    let inverse = (1.0 / sum) as f32;
    let probabilities: Vec<f32> = exponentials.iter().map(|value| (value * f64::from(inverse)) as f32).collect();
    let selected = topk(&probabilities, active)?;
    let selected_sum = selected.iter().map(|&expert| f64::from(probabilities[expert])).sum::<f64>();
    let denominator = selected_sum.max(0.00006103515625);
    let weights = selected.iter().map(|&expert| (f64::from(probabilities[expert]) / denominator) as f32).collect();
    Ok((selected, weights))
}

fn checked_output(
    value: Vec<f32>,
    expected: usize,
) -> Result<Vec<f32>> {
    if value.len() != expected || value.iter().any(|value| !value.is_finite()) {
        return Err(Error::Invalid("Edge0 latent matrix output"));
    }
    Ok(value)
}

pub(super) fn latent_forward(
    source: &impl LatentSource,
    input: &[f32],
    selected: &[usize],
    probabilities: &[f32],
) -> Result<Vec<f32>> {
    let g = source.geometry();
    if g.dim == 0
        || g.dim > 16384
        || g.latent == 0
        || g.latent > 16384
        || g.hidden == 0
        || g.hidden > 131072
        || g.groups == 0
        || g.groups > 512
        || g.experts == 0
        || g.experts > 4096
        || input.len() != g.dim
        || input.iter().any(|value| !value.is_finite())
        || selected.is_empty()
        || selected.len() > g.experts
        || selected.len() != probabilities.len()
        || probabilities.iter().any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(Error::Invalid("Edge0 latent geometry or routes"));
    }
    let scales = source.scales()?;
    if scales.len() != g.latent || scales.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(Error::Invalid("Edge0 latent column scales"));
    }
    let mut seen = vec![false; g.experts];
    let mut groups: Vec<Option<Vec<f32>>> = vec![None; g.groups];
    let mut result = vec![0.0; g.dim];
    for (&expert, &probability) in selected.iter().zip(probabilities) {
        if expert >= g.experts || seen[expert] {
            return Err(Error::Invalid("Edge0 latent expert route"));
        }
        seen[expert] = true;
        let group = source.group(expert)?;
        if group >= g.groups {
            return Err(Error::Invalid("Edge0 latent projection label"));
        }
        if groups[group].is_none() {
            let mut down = checked_output(source.matvec(LatentMatrix::ProjectionDown(group), input)?, g.latent)?;
            for (value, &scale) in down.iter_mut().zip(scales) {
                *value /= scale;
            }
            if down.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("Edge0 latent scaled activation"));
            }
            groups[group] = Some(down);
        }
        let activation = groups[group].as_ref().ok_or(Error::Invalid("Edge0 latent group activation"))?;
        let mut gate = checked_output(source.matvec(LatentMatrix::ExpertGate(expert), activation)?, g.hidden)?;
        let up = checked_output(source.matvec(LatentMatrix::ExpertUp(expert), activation)?, g.hidden)?;
        for (value, up) in gate.iter_mut().zip(up) {
            *value = silu(*value) * up;
        }
        let latent = checked_output(source.matvec(LatentMatrix::ExpertDown(expert), &gate)?, g.latent)?;
        // Each route gets its own U product. Summing by group first changes
        // floating-point reduction and is deliberately excluded here.
        let output = checked_output(source.matvec(LatentMatrix::ProjectionUp(group), &latent)?, g.dim)?;
        for (value, expert_output) in result.iter_mut().zip(output) {
            *value += probability * expert_output;
        }
    }
    checked_output(result, g.dim)
}

pub(super) fn validate(
    weights: &Weights,
    layer: usize,
    g: &Geometry,
    prefix: &str,
) -> Result<()> {
    let source = NativeLayer {
        weights,
        layer,
    };
    let native = source.geometry();
    if !weights.is_edge0() || g.dim != native.dim || g.experts != native.experts || g.expert_dim != native.hidden {
        return Err(Error::Invalid("Edge0 latent checkpoint geometry"));
    }
    let scales = source.scales()?;
    if scales.len() != native.latent || scales.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(Error::Invalid("Edge0 latent checkpoint scales"));
    }
    for expert in 0..g.experts {
        if source.group(expert)? >= native.groups {
            return Err(Error::Invalid("Edge0 latent checkpoint labels"));
        }
    }
    matrix(weights, &format!("{prefix}.gate"), g.experts, g.dim)?;
    matrix(weights, &format!("{prefix}.shared_expert_gate"), 1, g.dim)?;
    matrix(weights, &format!("{prefix}.shared_expert.gate_proj"), g.shared_dim, g.dim)?;
    matrix(weights, &format!("{prefix}.shared_expert.up_proj"), g.shared_dim, g.dim)?;
    matrix(weights, &format!("{prefix}.shared_expert.down_proj"), g.dim, g.shared_dim)
}

pub(super) fn moe(
    weights: &Weights,
    layer: usize,
    prefix: &str,
    input: &[f32],
    g: &Geometry,
) -> Result<Vec<f32>> {
    let [scores, shared_gate, mut shared_values, shared_up] = projects(
        weights,
        prefix,
        [
            ("gate", input),
            ("shared_expert_gate", input),
            ("shared_expert.gate_proj", input),
            ("shared_expert.up_proj", input),
        ],
    )?;
    let (selected, probabilities) = route(&scores, g.active_experts)?;
    let mut result = latent_forward(
        &NativeLayer {
            weights,
            layer,
        },
        input,
        &selected,
        &probabilities,
    )?;
    for (value, up) in shared_values.iter_mut().zip(shared_up) {
        *value = silu(*value) * up;
    }
    let shared = project(weights, &format!("{prefix}.shared_expert.down_proj"), &shared_values)?;
    if shared_gate.len() != 1 || !shared_gate[0].is_finite() {
        return Err(Error::Invalid("Edge0 shared expert gate"));
    }
    let gate = sigmoid(shared_gate[0]);
    for (value, shared) in result.iter_mut().zip(shared) {
        *value += gate * shared;
    }
    checked_output(result, g.dim)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use stark_engine_macros::stark_test;

    use super::*;

    struct Source {
        scales: [f32; 2],
        calls: RefCell<Vec<LatentMatrix>>,
    }

    impl LatentSource for Source {
        fn geometry(&self) -> LatentGeometry {
            LatentGeometry {
                dim: 2,
                latent: 2,
                hidden: 2,
                experts: 3,
                groups: 2,
            }
        }
        fn group(
            &self,
            expert: usize,
        ) -> Result<usize> {
            Ok(expert % 2)
        }
        fn scales(&self) -> Result<&[f32]> {
            Ok(&self.scales)
        }
        fn matvec(
            &self,
            matrix: LatentMatrix,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            self.calls.borrow_mut().push(matrix);
            Ok(input.to_vec())
        }
    }

    #[stark_test]
    fn edge0_latent_reuses_only_group_input_and_keeps_per_route_up_products() {
        let source = Source {
            scales: [0.5, 2.0],
            calls: RefCell::new(Vec::new()),
        };
        let result = latent_forward(&source, &[1.0, 1.0], &[2, 0], &[0.25, 0.75]).unwrap();
        for (actual, expected) in result.iter().zip([silu(2.0) * 2.0, silu(0.5) * 0.5]) {
            assert!((actual - expected).abs() < 2e-6);
        }
        assert_eq!(
            *source.calls.borrow(),
            [
                LatentMatrix::ProjectionDown(0),
                LatentMatrix::ExpertGate(2),
                LatentMatrix::ExpertUp(2),
                LatentMatrix::ExpertDown(2),
                LatentMatrix::ProjectionUp(0),
                LatentMatrix::ExpertGate(0),
                LatentMatrix::ExpertUp(0),
                LatentMatrix::ExpertDown(0),
                LatentMatrix::ProjectionUp(0),
            ]
        );
    }

    #[stark_test]
    fn edge0_latent_rejects_bad_routes_scales_and_nonfinite_inputs() {
        let mut source = Source {
            scales: [1.0, 1.0],
            calls: RefCell::new(Vec::new()),
        };
        assert!(latent_forward(&source, &[1.0, 1.0], &[3], &[1.0]).is_err());
        assert!(latent_forward(&source, &[1.0, 1.0], &[0, 0], &[0.5, 0.5]).is_err());
        assert!(latent_forward(&source, &[f32::INFINITY, 1.0], &[0], &[1.0]).is_err());
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            source.scales[0] = value;
            assert!(latent_forward(&source, &[1.0, 1.0], &[0], &[1.0]).is_err());
        }
    }

    #[stark_test]
    fn edge0_router_uses_probability_ties_after_f32_rounding() {
        assert_eq!(route(&[0.0, 0.0, 0.0], 2).unwrap(), (vec![0, 1], vec![0.5, 0.5]));
        assert_eq!(route(&[0.0, -1001.0, -1000.0], 2).unwrap(), (vec![0, 1], vec![1.0, 0.0]));
        assert!(route(&[f32::NAN], 1).is_err());
        assert!(route(&[0.0], 0).is_err());
    }

    /// Independent quantized latent/shared/readout component, not a trunk or
    /// mapped-owner parity claim. CPU uses the production scalar block decoder;
    /// Metal uses the production packed matvec kernel through its test wrapper.
    #[stark_test]
    #[ignore]
    fn edge0_latent_reference_fixture_matches_registered_f64() {
        use std::{cell::Cell, collections::BTreeMap, fs, path::Path};

        use serde_json::{Value, json};
        use sha2::{Digest, Sha256};

        fn sealed(
            root: &Path,
            info: &Value,
        ) -> Vec<u8> {
            let name = info["path"].as_str().unwrap();
            assert!(Path::new(name).file_name().is_some_and(|value| value == name));
            let bytes = fs::read(root.join(name)).expect("Edge0 fixture read failed");
            assert!(bytes.len() <= 8 * 1024 * 1024);
            assert_eq!(bytes.len() as u64, info["bytes"].as_u64().unwrap());
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), info["sha256"].as_str().unwrap());
            bytes
        }
        fn f32s(bytes: &[u8]) -> Vec<f32> {
            assert!(bytes.len().is_multiple_of(4));
            bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect()
        }
        fn u32s(bytes: &[u8]) -> Vec<usize> {
            assert!(bytes.len().is_multiple_of(4));
            bytes.chunks_exact(4).map(|value| u32::from_le_bytes(value.try_into().unwrap()) as usize).collect()
        }
        fn expected(
            root: &Path,
            reference: &Value,
            name: &str,
        ) -> Vec<f64> {
            let bytes = sealed(root, &reference["arrays"][name]);
            assert!(bytes.len().is_multiple_of(8));
            bytes.chunks_exact(8).map(|value| f64::from_le_bytes(value.try_into().unwrap())).collect()
        }
        struct Matrix {
            kind: u32,
            rows: usize,
            cols: usize,
            bytes: Vec<u8>,
        }
        struct FixtureSource<'a> {
            root: &'a Path,
            expected: &'a Value,
            matrices: &'a BTreeMap<String, Matrix>,
            labels: &'a [usize],
            scales: &'a [f32],
            selected: &'a [usize],
            token: usize,
            metal: bool,
            current_route: Cell<usize>,
            seen_down: RefCell<[bool; 3]>,
            calls: Cell<usize>,
            max_bound_ratio: Cell<f64>,
        }
        impl FixtureSource<'_> {
            fn raw(
                &self,
                name: &str,
                input: &[f32],
                tag: &str,
            ) -> Result<Vec<f32>> {
                let m = self.matrices.get(name).ok_or(Error::Invalid("Edge0 fixture matrix"))?;
                let cpu = || -> Result<Vec<f32>> {
                    let (block, block_bytes) = crate::engine::hybrid::weights::block_layout(m.kind)?;
                    if !m.cols.is_multiple_of(block) || input.len() != m.cols {
                        return Err(Error::Invalid("Edge0 fixture matrix geometry"));
                    }
                    let row_bytes = m.cols / block * block_bytes;
                    let mut output = Vec::with_capacity(m.rows);
                    for row in m.bytes.chunks_exact(row_bytes) {
                        let mut values = vec![0.0; m.cols];
                        for (source, target) in row.chunks_exact(block_bytes).zip(values.chunks_exact_mut(block)) {
                            crate::engine::hybrid::weights::decode_block(m.kind, source, target)?;
                        }
                        output.push(values.iter().zip(input).map(|(a, b)| a * b).sum::<f32>());
                    }
                    Ok(output)
                };
                #[cfg(feature = "metal")]
                let output = if self.metal {
                    crate::engine::hybrid::weight_metal::matvec(&m.bytes, m.kind, m.rows, m.cols, input)?
                } else {
                    cpu()?
                };
                #[cfg(not(feature = "metal"))]
                let output = {
                    assert!(!self.metal);
                    cpu()?
                };
                let reference = expected(self.root, self.expected, tag);
                let sumabs = expected(self.root, self.expected, &format!("{tag}.sumabs"));
                assert_eq!(output.len(), reference.len());
                assert_eq!(output.len(), sumabs.len());
                for ((&actual, &reference), &sumabs) in output.iter().zip(&reference).zip(&sumabs) {
                    assert!(actual.is_finite() && reference.is_finite() && sumabs.is_finite());
                    let ratio = (f64::from(actual) - reference).abs() / (2e-5 * sumabs.max(1.0));
                    self.max_bound_ratio.set(self.max_bound_ratio.get().max(ratio));
                    assert!(ratio <= 1.0, "Edge0 fixture per-matrix F64 bound failed");
                }
                self.calls.set(self.calls.get() + 1);
                Ok(output)
            }
        }
        impl LatentSource for FixtureSource<'_> {
            fn geometry(&self) -> LatentGeometry {
                LatentGeometry {
                    dim: 256,
                    latent: 256,
                    hidden: 256,
                    experts: 5,
                    groups: 3,
                }
            }
            fn group(
                &self,
                expert: usize,
            ) -> Result<usize> {
                self.labels.get(expert).copied().ok_or(Error::Invalid("Edge0 fixture label"))
            }
            fn scales(&self) -> Result<&[f32]> {
                Ok(self.scales)
            }
            fn matvec(
                &self,
                matrix: LatentMatrix,
                input: &[f32],
            ) -> Result<Vec<f32>> {
                let (name, tag) = match matrix {
                    LatentMatrix::ProjectionDown(group) => {
                        self.seen_down.borrow_mut()[group] = true;
                        (format!("d.{group}"), format!("t{}.d.{group}", self.token))
                    },
                    LatentMatrix::ProjectionUp(group) => {
                        (format!("u.{group}"), format!("t{}.r{}.u", self.token, self.current_route.get()))
                    },
                    LatentMatrix::ExpertGate(expert)
                    | LatentMatrix::ExpertUp(expert)
                    | LatentMatrix::ExpertDown(expert) => {
                        let route = self
                            .selected
                            .iter()
                            .position(|&value| value == expert)
                            .ok_or(Error::Invalid("Edge0 fixture selected route"))?;
                        self.current_route.set(route);
                        let role = match matrix {
                            LatentMatrix::ExpertGate(_) => "gate",
                            LatentMatrix::ExpertUp(_) => "up",
                            _ => "down",
                        };
                        (format!("{role}.{expert}"), format!("t{}.r{route}.{role}", self.token))
                    },
                };
                self.raw(&name, input, &tag)
            }
        }

        let root = std::env::var_os("STARK_EDGE0_FORWARD_FIXTURE")
            .map(std::path::PathBuf::from)
            .expect("registered Edge0 fixture path required");
        let inputs_bytes = fs::read(root.join("inputs.json")).expect("Edge0 fixture manifest required");
        let reference_bytes = fs::read(root.join("expected.json")).expect("Edge0 fixture reference required");
        assert_eq!(
            format!("{:x}", Sha256::digest(&inputs_bytes)),
            "17469b3bbe73570fcc259880012fd23f75996325c76e7e0385e1f07a0e97071a"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&reference_bytes)),
            "40f2b8e62b9c071eebefd0c73095c8980952b11bf293232380f4b40cb446f2d4"
        );
        let inputs: Value = serde_json::from_slice(&inputs_bytes).unwrap();
        let reference: Value = serde_json::from_slice(&reference_bytes).unwrap();
        let mut matrices = BTreeMap::new();
        for entry in inputs["matrices"].as_array().unwrap() {
            matrices.insert(
                entry["name"].as_str().unwrap().to_owned(),
                Matrix {
                    kind: entry["type"].as_u64().unwrap() as u32,
                    rows: entry["rows"].as_u64().unwrap() as usize,
                    cols: entry["cols"].as_u64().unwrap() as usize,
                    bytes: sealed(&root, entry),
                },
            );
        }
        let x = f32s(&sealed(&root, &inputs["arrays"]["x.f32"]));
        let scales = f32s(&sealed(&root, &inputs["arrays"]["scales.f32"]));
        let labels = u32s(&sealed(&root, &inputs["arrays"]["labels.u32"]));
        let ids = u32s(&sealed(&root, &inputs["arrays"]["routes.u32"]));
        let probabilities = f32s(&sealed(&root, &inputs["arrays"]["route_weights.f32"]));
        let logits = f32s(&sealed(&root, &inputs["arrays"]["router_logits.f32"]));
        let expected_readout = expected(&root, &reference, "expected-readout.f64");
        #[cfg(feature = "metal")]
        let modes = [false, true];
        #[cfg(not(feature = "metal"))]
        let modes = [false];
        let mut reports = Vec::new();
        for metal in modes {
            let (mut max_error, mut squared_error, mut squared_reference) = (0.0_f64, 0.0_f64, 0.0_f64);
            let (mut total_calls, mut max_bound_ratio) = (0, 0.0_f64);
            for token in 0..3 {
                let input = &x[token * 256..(token + 1) * 256];
                let selected = &ids[token * 3..(token + 1) * 3];
                let weights = &probabilities[token * 3..(token + 1) * 3];
                let (actual_ids, actual_weights) = route(&logits[token * 5..(token + 1) * 5], 3).unwrap();
                assert!(
                    actual_ids == selected
                        && actual_weights.iter().zip(weights).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "Edge0 fixture router mismatch"
                );
                let source = FixtureSource {
                    root: &root,
                    expected: &reference,
                    matrices: &matrices,
                    labels: &labels,
                    scales: &scales,
                    selected,
                    token,
                    metal,
                    current_route: Cell::new(0),
                    seen_down: RefCell::new([false; 3]),
                    calls: Cell::new(0),
                    max_bound_ratio: Cell::new(0.0),
                };
                let mut result = latent_forward(&source, input, selected, weights).unwrap();
                let latent_reference = expected(&root, &reference, &format!("t{token}.latent"));
                assert!(
                    result.iter().zip(latent_reference).all(|(&a, b)| (f64::from(a) - b).abs() <= 0.002),
                    "Edge0 latent F64 bound failed"
                );
                // Check unused D groups too, independently of the lazy forward.
                for group in 0..3 {
                    if !source.seen_down.borrow()[group] {
                        source.raw(&format!("d.{group}"), input, &format!("t{token}.d.{group}")).unwrap();
                    }
                }
                let gate = source.raw("shared_gate", input, &format!("t{token}.shared_gate")).unwrap();
                let up = source.raw("shared_up", input, &format!("t{token}.shared_up")).unwrap();
                let hidden: Vec<f32> = gate.into_iter().zip(up).map(|(a, b)| silu(a) * b).collect();
                let shared = source.raw("shared_down", &hidden, &format!("t{token}.shared_down")).unwrap();
                let scalar = source.raw("shared_scalar", input, &format!("t{token}.shared_scalar")).unwrap();
                for (value, shared) in result.iter_mut().zip(shared) {
                    *value += sigmoid(scalar[0]) * shared;
                }
                let readout = source.raw("readout", &result, &format!("t{token}.readout")).unwrap();
                for (&actual, &expected) in readout.iter().zip(&expected_readout[token * 17..(token + 1) * 17]) {
                    let difference = f64::from(actual) - expected;
                    max_error = max_error.max(difference.abs());
                    squared_error += difference * difference;
                    squared_reference += expected * expected;
                }
                total_calls += source.calls.get();
                max_bound_ratio = max_bound_ratio.max(source.max_bound_ratio.get());
            }
            let relative_l2 = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
            assert!(max_error <= 0.002 && relative_l2 <= 0.0002, "Edge0 final fixture F64 bound failed");
            assert_eq!(total_calls, 60);
            reports.push(json!({"backend":if metal {"metal-packed-kernel"} else {"cpu-scalar"},
                "matrix_operations":total_calls,"max_matrix_bound_ratio":max_bound_ratio,
                "max_abs_readout_error":max_error,"relative_l2_readout_error":relative_l2}));
        }
        fs::write(root.join("rust-component-report.json"), serde_json::to_vec_pretty(&json!({
            "attempt":"20261112-EDGE0-05","passed":true,"scope":"quantized latent/shared/readout component; no trunk or mapped-owner parity",
            "inputs_sha256":"17469b3bbe73570fcc259880012fd23f75996325c76e7e0385e1f07a0e97071a",
            "expected_sha256":"40f2b8e62b9c071eebefd0c73095c8980952b11bf293232380f4b40cb446f2d4",
            "program_sha256":format!("{:x}",Sha256::digest(include_bytes!("edge0.rs"))),"reports":reports
        })).unwrap()).expect("Edge0 aggregate fixture report failed");
    }
}
