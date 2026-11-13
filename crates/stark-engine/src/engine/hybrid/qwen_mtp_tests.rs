// SPDX-License-Identifier: Apache-2.0
//! Sealed private synthetic whole-head controls. Expected binary64 equations
//! were frozen before native evaluation. Packed input bytes are copied intact
//! into owned Metal buffers; no candidate arithmetic creates expected values.
//! Gate/up scalar projections are not observable in the fused native operator;
//! its independent original-plane operator tests qualify that separate boundary.

use std::{collections::BTreeSet, fs, path::Path};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::engine::hybrid::weight_metal::{MappedMatrix, MatrixFormat, fixture_mapped_bytes};

const MANIFEST_SHA: &str = "d9079cfb1b23492279a39393a6ed9085948084840272b3e9bd332d583f24839f";
const REGISTRATION_SHA: &str = "414d99374d6b2aa2590176ec88100981fe51ee688bbba9630da1e95e44f38f07";
const EXPECTED_SHA: &str = "5196ec1baf13d1474d57ec71861c6766c1f7fad432a87b4e95e32f90cda50513";
const GENERATOR_SHA: &str = "76e418f3488d2f150763ddd2415b6f12bc6f0866be18a5fe53f8cb67e9d8f689";
const ORACLE_TEST_SHA: &str = "2282a19259c345a7a1eba1db43861c36a67f527c1b6b839d1b285ebb03cd9e10";
const MAX_FILE: usize = 8 * 1024 * 1024;
const CELLS: usize = 4;

fn ensure(
    ok: bool,
    reason: &'static str,
) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Invalid(reason))
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn integer(value: &Value) -> Result<usize> {
    value.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or(Error::Invalid("private fixture integer"))
}

fn text(value: &Value) -> Result<&str> {
    value.as_str().ok_or(Error::Invalid("private fixture string"))
}

fn read_sealed(
    root: &Path,
    name: &str,
    identity: &Value,
) -> Result<Vec<u8>> {
    ensure(
        matches!(name, "dense.bin" | "dense.txt" | "experts.bin" | "inputs.bin" | "expected.bin"),
        "private fixture filename",
    )?;
    let path = root.join(name);
    let metadata = fs::metadata(&path).map_err(|_| Error::Invalid("private fixture metadata"))?;
    ensure(
        metadata.len() <= MAX_FILE as u64 && metadata.len() == integer(&identity["bytes"])? as u64,
        "private fixture file byte bound",
    )?;
    let bytes = fs::read(path).map_err(|_| Error::Invalid("private fixture read"))?;
    ensure(hash(&bytes) == text(&identity["sha256"])?, "private fixture raw file seal")?;
    Ok(bytes)
}

fn f32_values(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure(bytes.len().is_multiple_of(4), "private fixture F32 byte alignment")?;
    let values = bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
    ensure(values.iter().all(|x| x.is_finite()), "private fixture finite F32")?;
    Ok(values)
}

fn f64_values(bytes: &[u8]) -> Result<Vec<f64>> {
    ensure(bytes.len().is_multiple_of(8), "private fixture F64 byte alignment")?;
    let values = bytes.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
    ensure(values.iter().all(|x| x.is_finite()), "private fixture finite F64")?;
    Ok(values)
}

struct Entry {
    name: String,
    file: String,
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    sha: String,
    cells: Vec<usize>,
}

impl Entry {
    fn parse(value: &Value) -> Result<Self> {
        let shape = value["shape"]
            .as_array()
            .ok_or(Error::Invalid("private fixture shape"))?
            .iter()
            .map(integer)
            .collect::<Result<Vec<_>>>()?;
        ensure(
            !shape.is_empty() && shape.len() <= 3 && shape.iter().all(|&x| x > 0 && x <= 256),
            "private fixture shape bound",
        )?;
        let cells = value
            .get("cell_indices")
            .map(|v| {
                v.as_array()
                    .ok_or(Error::Invalid("private fixture cell indices"))?
                    .iter()
                    .map(integer)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            name: text(&value["name"])?.to_owned(),
            file: text(&value["file"])?.to_owned(),
            dtype: text(&value["dtype"])?.to_owned(),
            shape,
            offset: integer(&value["offset"])?,
            bytes: integer(&value["bytes"])?,
            sha: text(&value["sha256"])?.to_owned(),
            cells,
        })
    }

    fn raw<'a>(
        &self,
        files: &'a BTreeMap<String, Vec<u8>>,
    ) -> Result<&'a [u8]> {
        let bytes = files.get(&self.file).ok_or(Error::Invalid("private fixture missing file"))?;
        let end = self.offset.checked_add(self.bytes).ok_or(Error::Invalid("private fixture span overflow"))?;
        let raw = bytes.get(self.offset..end).ok_or(Error::Invalid("private fixture span bounds"))?;
        ensure(!raw.is_empty() && hash(raw) == self.sha, "private fixture record seal")?;
        Ok(raw)
    }
}

struct StoredMatrix {
    kind: u32,
    rows: usize,
    columns: usize,
    bytes: Vec<u8>,
    mapped: MappedMatrix,
}

struct FixtureWeights {
    matrices: BTreeMap<String, StoredMatrix>,
    experts: Vec<MappedMatrix>,
}

impl FixtureWeights {
    fn matrix(
        &self,
        name: &str,
    ) -> Result<&StoredMatrix> {
        self.matrices.get(name).ok_or(Error::Invalid("private fixture matrix name"))
    }
}

impl HeadWeights for FixtureWeights {
    fn norm(
        &self,
        name: &str,
    ) -> Result<Vec<f32>> {
        let matrix = self.matrix(name)?;
        ensure(matrix.kind == 0 && matrix.rows == 1, "private fixture norm encoding")?;
        f32_values(&matrix.bytes)
    }
    fn shape(
        &self,
        name: &str,
    ) -> Result<[usize; 2]> {
        let matrix = self.matrix(name)?;
        Ok([matrix.rows, matrix.columns])
    }
    fn project(
        &self,
        graph: &mut DecodeGraph,
        name: &str,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let matrix = self.matrix(name)?;
        graph.edge0_packed_batch(
            &matrix.mapped,
            MatrixFormat::Packed(matrix.kind),
            matrix.rows,
            matrix.columns,
            input,
            tokens,
        )
    }
    fn expert_gate_up(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let matrix = self.experts.get(expert).ok_or(Error::Invalid("private fixture expert ID"))?;
        graph.edge0_mtp_q2_gate_up(matrix, 64, 64, input, tokens)
    }
    fn expert_down(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let matrix = self.experts.get(expert).ok_or(Error::Invalid("private fixture expert ID"))?;
        graph.edge0_mtp_q2_down(matrix, 64, 64, input, tokens)
    }
    fn embedding(
        &self,
        token: u32,
    ) -> Result<Vec<f32>> {
        let matrix = self.matrix("token_embedding")?;
        ensure(
            matrix.kind == 8 && (token as usize) < matrix.rows && matrix.columns.is_multiple_of(32),
            "private fixture embedding geometry",
        )?;
        let row_bytes = matrix.columns / 32 * 34;
        let start = token as usize * row_bytes;
        let bytes =
            matrix.bytes.get(start..start + row_bytes).ok_or(Error::Invalid("private fixture embedding span"))?;
        let mut out = Vec::with_capacity(matrix.columns);
        for block in bytes.chunks_exact(34) {
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            out.extend(block[2..].iter().map(|&code| scale * f32::from(code as i8)));
        }
        ensure(out.iter().all(|x| x.is_finite()), "private fixture embedding finite")?;
        Ok(out)
    }
    fn readout(
        &self,
        graph: &mut DecodeGraph,
        input: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.project(graph, "main_head", input, 1)
    }
}

struct Expected {
    width: usize,
    cells: Vec<usize>,
    values: Vec<f64>,
}

impl Expected {
    fn row(
        &self,
        cell: usize,
    ) -> Result<&[f64]> {
        let index = self
            .cells
            .iter()
            .position(|&x| x == cell)
            .ok_or(Error::Invalid("private fixture missing expected cell"))?;
        self.values
            .get(index * self.width..(index + 1) * self.width)
            .ok_or(Error::Invalid("private fixture expected span"))
    }
}

struct Fixture {
    weights: FixtureWeights,
    residuals: Vec<f32>,
    tokens: Vec<u32>,
    expected: BTreeMap<String, Expected>,
}

fn geometry() -> HeadGeometry {
    HeadGeometry {
        hidden: 64,
        intermediate: 64,
        streams: 4,
        low_rank: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        rotary: 8,
        experts: 4,
        active: 2,
        vocabulary: 128,
        context: 16,
        rope_theta: 10000.0,
    }
}

fn stage_widths() -> BTreeMap<String, usize> {
    let mut out = [
        ("embedding_norm", 64),
        ("hidden_norm", 256),
        ("embedding_projected", 64),
        ("hidden_projected", 256),
        ("input_residual", 256),
        ("query_full", 128),
        ("query", 64),
        ("key", 32),
        ("value", 32),
        ("attention_scores", 16),
        ("attention_probabilities", 16),
        ("attention", 64),
        ("gated_attention", 64),
        ("attention_output", 64),
        ("post_attention", 256),
        ("router_logits", 4),
        ("router_weights", 2),
        ("routed_gate", 128),
        ("routed_up", 128),
        ("routed_activation", 128),
        ("routed_parts", 128),
        ("routed_gate_sumabs", 128),
        ("routed_up_sumabs", 128),
        ("routed_down_sumabs", 128),
        ("shared_gate", 64),
        ("shared_up", 64),
        ("shared_activation", 64),
        ("shared_scalar", 1),
        ("shared_output", 64),
        ("moe_output", 64),
        ("final_residual", 256),
        ("readout", 128),
        ("routed_ids", 2),
    ]
    .into_iter()
    .map(|(name, width)| (name.to_owned(), width))
    .collect::<BTreeMap<_, _>>();
    for prefix in ["attn_hc", "mlp_hc", "final_hc"] {
        for (suffix, width) in [("normalized", 256), ("low", 8), ("mixed", 64)] {
            out.insert(format!("{prefix}_{suffix}"), width);
        }
        if prefix != "final_hc" {
            out.insert(format!("{prefix}_inject"), 4);
        }
    }
    out
}

fn front_stage(name: &str) -> bool {
    matches!(
        name,
        "embedding_norm"
            | "hidden_norm"
            | "embedding_projected"
            | "hidden_projected"
            | "input_residual"
            | "key"
            | "value"
            | "attn_hc_normalized"
            | "attn_hc_low"
            | "attn_hc_mixed"
            | "attn_hc_inject"
    )
}

fn observable(name: &str) -> bool {
    !matches!(
        name,
        "routed_ids" | "routed_gate" | "routed_up" | "routed_gate_sumabs" | "routed_up_sumabs" | "routed_down_sumabs"
    )
}

impl Fixture {
    fn load() -> Result<Self> {
        let root = std::env::var_os("STARK_EDGE0_MTP_FIXTURE")
            .ok_or(Error::Invalid("registered private MTP fixture path required"))?;
        let root = Path::new(&root);
        let manifest_raw =
            fs::read(root.join("manifest.json")).map_err(|_| Error::Invalid("private fixture manifest read"))?;
        ensure(manifest_raw.len() <= MAX_FILE && hash(&manifest_raw) == MANIFEST_SHA, "private fixture manifest seal")?;
        let manifest: Value =
            serde_json::from_slice(&manifest_raw).map_err(|_| Error::Invalid("private fixture manifest JSON"))?;
        let registration_raw = fs::read(root.join("registration.json"))
            .map_err(|_| Error::Invalid("private fixture registration read"))?;
        ensure(
            registration_raw.len() <= MAX_FILE
                && hash(&registration_raw) == REGISTRATION_SHA
                && text(&manifest["registration_sha256"])? == REGISTRATION_SHA,
            "private fixture registration seal",
        )?;
        let registration: Value = serde_json::from_slice(&registration_raw)
            .map_err(|_| Error::Invalid("private fixture registration JSON"))?;
        let mut registered = manifest.clone();
        let object = registered.as_object_mut().ok_or(Error::Invalid("private fixture manifest object"))?;
        for key in ["expected", "expected_file", "registration_sha256"] {
            object.remove(key);
        }
        ensure(registered == registration, "private fixture preregistration equality")?;
        ensure(
            text(&manifest["schema"])? == "edge0-mtp-synthetic-v1"
                && text(&manifest["source_revision"])? == "b7900cbb322926ccebdcd0c0c5a1280497ecff22"
                && text(&manifest["generator_sha256"])? == GENERATOR_SHA
                && text(&manifest["tests_sha256"])? == ORACLE_TEST_SHA,
            "private fixture oracle provenance",
        )?;
        ensure(
            hash(include_bytes!("../../../../../scripts/edge0_mtp_fixture.py")) == GENERATOR_SHA
                && hash(include_bytes!("../../../../../scripts/test_edge0_mtp_fixture.py")) == ORACLE_TEST_SHA,
            "private fixture frozen helper source",
        )?;
        ensure(
            manifest["geometry"]
                == json!({"hidden":64,"intermediate":64,"streams":4,"low_rank":8,"heads":4,"kv_heads":2,
            "head_dim":16,"rotary":8,"experts":4,"active":2,"index_heads":4,"index_dim":8,"vocabulary":128,"context":16,"cells":4}),
            "private fixture fixed geometry",
        )?;
        ensure(
            manifest["gates"]["max_absolute_error"] == 3e-5
                && manifest["gates"]["max_scaled_error"] == 3e-5
                && manifest["gates"]["final_max_absolute_error"] == 0.002
                && manifest["gates"]["final_relative_l2_error"] == 0.0002,
            "private fixture fixed numerical gates",
        )?;
        let identities = manifest["files"].as_object().ok_or(Error::Invalid("private fixture raw identities"))?;
        ensure(identities.len() == 4, "private fixture raw inventory")?;
        let mut files = BTreeMap::new();
        for name in ["dense.bin", "dense.txt", "experts.bin", "inputs.bin"] {
            let identity = identities.get(name).ok_or(Error::Invalid("private fixture raw inventory"))?;
            files.insert(name.to_owned(), read_sealed(root, name, identity)?);
        }
        let expected_raw = read_sealed(root, "expected.bin", &manifest["expected_file"])?;
        ensure(hash(&expected_raw) == EXPECTED_SHA, "private fixture expected seal")?;
        files.insert("expected.bin".to_owned(), expected_raw);
        let entries = manifest["inputs"].as_array().ok_or(Error::Invalid("private fixture input inventory"))?;
        ensure(entries.len() == 33, "private fixture input count")?;
        let mut matrices = BTreeMap::new();
        let mut residuals = Vec::new();
        let mut tokens = Vec::new();
        for value in entries {
            let entry = Entry::parse(value)?;
            let raw = entry.raw(&files)?;
            if entry.name == "target_residual" {
                ensure(
                    entry.file == "inputs.bin" && entry.dtype == "f32" && entry.shape == [4, 4, 64],
                    "private fixture residual schema",
                )?;
                residuals = f32_values(raw)?;
            } else if entry.name == "next_token_ids" {
                ensure(
                    entry.file == "inputs.bin" && entry.dtype == "u32" && entry.shape == [4] && raw.len() == 16,
                    "private fixture token schema",
                )?;
                tokens = raw.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
                ensure(tokens.iter().all(|&x| x < 128), "private fixture token bounds")?;
            } else {
                ensure(entry.shape.len() == 2, "private fixture matrix rank")?;
                let kind = match entry.dtype.as_str() {
                    "f32" => 0,
                    "q8_0" => 8,
                    "bf16" => 30,
                    _ => return Err(Error::Invalid("private fixture dense encoding")),
                };
                let [nr, nc] = [entry.shape[0], entry.shape[1]];
                let bytes = match kind {
                    0 => nr * nc * 4,
                    30 => nr * nc * 2,
                    _ => {
                        ensure(nc.is_multiple_of(32), "private fixture Q8 alignment")?;
                        nr * (nc / 32) * 34
                    },
                };
                ensure(
                    bytes == raw.len() && !matrices.contains_key(&entry.name),
                    "private fixture matrix byte geometry",
                )?;
                matrices.insert(
                    entry.name,
                    StoredMatrix {
                        kind,
                        rows: nr,
                        columns: nc,
                        bytes: raw.to_vec(),
                        mapped: fixture_mapped_bytes(raw)?,
                    },
                );
            }
        }
        ensure(
            matrices.len() == 31 && residuals.len() == 1024 && tokens.len() == 4,
            "private fixture complete input coverage",
        )?;
        let expert_raw = &files["experts.bin"];
        let blob = 3 * 64 * 64 * 18 / 64;
        ensure(expert_raw.len() == 4 * blob, "private fixture expert inventory")?;
        let owner = fixture_mapped_bytes(expert_raw)?;
        let experts = (0..4)
            .map(|i| {
                Ok(MappedMatrix {
                    codes: owner.codes.subslice(i * blob, blob)?,
                    planes: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let expected_entries =
            manifest["expected"].as_array().ok_or(Error::Invalid("private fixture expected inventory"))?;
        let layout =
            manifest["expected_layout"].as_array().ok_or(Error::Invalid("private fixture registered layout"))?;
        ensure(expected_entries.len() == 132 && layout.len() == 132, "private fixture expected count")?;
        let widths = stage_widths();
        ensure(widths.len() == 44, "private fixture closed stage count")?;
        let mut expected = BTreeMap::new();
        let mut offset = 0;
        for value in expected_entries {
            let entry = Entry::parse(value)?;
            let (case, name) = entry.name.split_once('/').ok_or(Error::Invalid("private fixture expected case"))?;
            ensure(
                matches!(case, "serial" | "catchup" | "chain")
                    && widths.get(name).is_some_and(|&w| entry.shape == [entry.cells.len(), w]),
                "private fixture expected stage shape",
            )?;
            let indices = if case == "catchup" && !front_stage(name) {
                vec![2, 3]
            } else {
                vec![0, 1, 2, 3]
            };
            ensure(
                entry.cells == indices && entry.file == "expected.bin" && entry.offset == offset,
                "private fixture expected closed cell layout",
            )?;
            let dtype = if name == "routed_ids" {
                "u32"
            } else {
                "f64"
            };
            ensure(
                entry.dtype == dtype && !expected.contains_key(&entry.name),
                "private fixture expected encoding or duplicate",
            )?;
            let registered =
                json!({"name":entry.name,"dtype":entry.dtype,"shape":entry.shape,"cell_indices":entry.cells});
            ensure(layout.iter().filter(|&x| *x == registered).count() == 1, "private fixture stage preregistration")?;
            let raw = entry.raw(&files)?;
            let values = if dtype == "u32" {
                ensure(raw.len().is_multiple_of(4), "private fixture route alignment")?;
                raw.chunks_exact(4).map(|x| f64::from(u32::from_le_bytes(x.try_into().unwrap()))).collect()
            } else {
                f64_values(raw)?
            };
            ensure(values.len() == entry.shape.iter().product::<usize>(), "private fixture expected value count")?;
            offset += entry.bytes;
            expected.insert(
                entry.name,
                Expected {
                    width: entry.shape[1],
                    cells: entry.cells,
                    values,
                },
            );
        }
        ensure(offset == files["expected.bin"].len(), "private fixture expected exact coverage")?;
        Ok(Self {
            weights: FixtureWeights {
                matrices,
                experts,
            },
            residuals,
            tokens,
            expected,
        })
    }

    fn row(
        &self,
        case: &str,
        stage: &str,
        cell: usize,
    ) -> Result<&[f64]> {
        self.expected
            .get(&format!("{case}/{stage}"))
            .ok_or(Error::Invalid("private fixture missing expected stage"))?
            .row(cell)
    }
    fn input(
        &self,
        cell: usize,
    ) -> Result<DeviceVector> {
        DecodeGraph::upload(
            self.residuals.get(cell * 256..(cell + 1) * 256).ok_or(Error::Invalid("private fixture input cell"))?,
        )
    }
}

#[derive(Default)]
struct Metrics {
    stages: usize,
    routes: usize,
    max_stage_ratio: f64,
    max_final_abs: f64,
    max_final_relative_l2: f64,
    max_q2_ratio: f64,
}

fn stage_ratio(
    actual: &[f32],
    expected: &[f64],
) -> Result<f64> {
    ensure(
        actual.len() == expected.len()
            && !actual.is_empty()
            && actual.iter().all(|x| x.is_finite())
            && expected.iter().all(|x| x.is_finite()),
        "private fixture stage finite geometry",
    )?;
    let maximum = expected.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let bound = 3e-5 + 3e-5 * maximum;
    Ok(actual.iter().zip(expected).map(|(&a, &b)| (f64::from(a) - b).abs() / bound).fold(0.0, f64::max))
}

fn trace_row(
    name: &str,
    actual: &[f32],
) -> Result<Vec<f32>> {
    if matches!(name, "attention_scores" | "attention_probabilities") {
        ensure(
            actual.len() == 4 * 16 && actual.iter().all(|x| x.is_finite()),
            "private fixture attention trace stride",
        )?;
        ensure(
            actual.chunks_exact(16).all(|row| row[CELLS..].iter().all(|&x| x == 0.0)),
            "private fixture attention trace beyond fixture cells",
        )?;
        return Ok(actual.chunks_exact(16).flat_map(|row| row[..CELLS].iter().copied()).collect());
    }
    Ok(actual.to_vec())
}

impl Metrics {
    fn compare(
        &mut self,
        fixture: &Fixture,
        case: &str,
        cell: usize,
        full: bool,
        output: &HeadOutput,
    ) -> Result<()> {
        let wanted = stage_widths()
            .keys()
            .filter(|name| observable(name) && (full || front_stage(name)))
            .cloned()
            .collect::<BTreeSet<_>>();
        ensure(
            output.stages.keys().cloned().collect::<BTreeSet<_>>() == wanted,
            "private fixture missing or unexpected observed stage",
        )?;
        for (name, raw) in &output.stages {
            let actual = trace_row(name, raw)?;
            let expected = fixture.row(case, name, cell)?;
            let ratio = stage_ratio(&actual, expected)?;
            ensure(ratio <= 1.0, "private fixture stage numerical gate")?;
            self.max_stage_ratio = self.max_stage_ratio.max(ratio);
            self.stages += 1;
            if name == "routed_parts" {
                let sumabs = fixture.row(case, "routed_down_sumabs", cell)?;
                ensure(sumabs.len() == actual.len(), "private fixture Q2 sumabs geometry")?;
                for ((&a, &b), &scale) in actual.iter().zip(expected).zip(sumabs) {
                    let ratio = (f64::from(a) - b).abs() / (2e-5 * scale.max(1.0));
                    ensure(ratio <= 1.0, "private fixture Q2 down product gate")?;
                    self.max_q2_ratio = self.max_q2_ratio.max(ratio);
                }
            }
        }
        if full {
            let ids = fixture.row(case, "routed_ids", cell)?;
            ensure(
                output.routed_ids.len() == ids.len() && output.routed_ids.iter().zip(ids).all(|(&a, &b)| a as f64 == b),
                "private fixture exact routed ordering",
            )?;
            self.routes += 1;
            let actual = output.logits.as_ref().ok_or(Error::Invalid("private fixture missing readout"))?;
            let expected = fixture.row(case, "readout", cell)?;
            let (mut delta, mut norm, mut maximum) = (0.0, 0.0, 0.0_f64);
            for (&a, &b) in actual.iter().zip(expected) {
                let d = f64::from(a) - b;
                delta += d * d;
                norm += b * b;
                maximum = maximum.max(d.abs());
            }
            let relative = (delta / norm.max(f64::MIN_POSITIVE)).sqrt();
            ensure(maximum <= 0.002 && relative <= 0.0002, "private fixture final readout gates")?;
            self.max_final_abs = self.max_final_abs.max(maximum);
            self.max_final_relative_l2 = self.max_final_relative_l2.max(relative);
            ensure(output.wide.is_some(), "private fixture missing wide residual")?;
        } else {
            ensure(
                output.wide.is_none() && output.logits.is_none() && output.routed_ids.is_empty(),
                "private fixture front-only output boundary",
            )?;
        }
        Ok(())
    }
}

fn private<T>(value: Result<T>) -> T {
    value.unwrap_or_else(|_| panic!("private synthetic MTP aggregate gate failed"))
}

#[stark_test]
// Requires the sealed private synthetic MTP fixture and a physical Metal device.
#[ignore]
fn edge0_mtp_complete_head_matches_frozen_binary64_fixture() {
    let fixture = private(Fixture::load());
    let mut metrics = Metrics::default();
    for case in ["serial", "catchup", "chain"] {
        let mut head = private(Head::new(geometry(), &fixture.weights));
        let mut previous = None;
        for cell in 0..CELLS {
            let input = if case == "chain" && cell == 3 {
                previous.take().unwrap_or_else(|| panic!("private chain owner missing"))
            } else {
                private(fixture.input(cell))
            };
            let full = case != "catchup" || cell >= 2;
            let output = private(if full {
                head.step(&fixture.weights, cell, &input, fixture.tokens[cell], true)
            } else {
                head.catch_up(&fixture.weights, cell, &input, fixture.tokens[cell], true)
            });
            private(metrics.compare(&fixture, case, cell, full, &output));
            previous = output.wide;
        }
        assert!(head.next_cell() == CELLS && !head.is_fenced(), "private complete-head final boundary");
    }
    assert!(metrics.stages == 402 && metrics.routes == 10, "private observed gate coverage");
    println!(
        "{}",
        json!({"synthetic_mtp_head":true,"observed_stage_rows":metrics.stages,"exact_routed_rows":metrics.routes,
        "max_stage_bound_ratio":metrics.max_stage_ratio,"max_q2_down_bound_ratio":metrics.max_q2_ratio,
        "max_final_absolute_error":metrics.max_final_abs,"max_final_relative_l2_error":metrics.max_final_relative_l2,
        "unobserved_scalar_gate_up":true,"actual_artifact_forward":false})
    );
}

fn output_bits(output: &HeadOutput) -> Result<(Vec<u32>, Vec<u32>, Vec<usize>)> {
    let logits = output.logits.as_ref().ok_or(Error::Invalid("private replay readout"))?;
    let wide = output.wide.as_ref().ok_or(Error::Invalid("private replay wide"))?;
    let values = new_graph()?.finish(&[wide.clone()])?;
    let row = values.first().ok_or(Error::Invalid("private replay wide read"))?;
    Ok((
        logits.iter().map(|x| x.to_bits()).collect(),
        row.iter().map(|x| x.to_bits()).collect(),
        output.routed_ids.clone(),
    ))
}

#[stark_test]
// Requires the sealed private synthetic MTP fixture and a physical Metal device.
#[ignore]
fn edge0_mtp_head_checkpoint_replay_and_failure_fences() {
    let fixture = private(Fixture::load());
    let mut head = private(Head::new(geometry(), &fixture.weights));
    let zero = private(head.capture_checkpoint());
    let mut baseline = Vec::new();
    for cell in 0..CELLS {
        let x = private(fixture.input(cell));
        let out = private(head.step(&fixture.weights, cell, &x, fixture.tokens[cell], false));
        baseline.push(private(output_bits(&out)));
    }
    private(head.restore_checkpoint(&zero));
    assert!(head.next_cell() == 0, "private zero snapshot position");
    for cell in 0..2 {
        let x = private(fixture.input(cell));
        let out = private(head.step(&fixture.weights, cell, &x, fixture.tokens[cell], false));
        assert!(private(output_bits(&out)) == baseline[cell], "private zero replay bits");
    }
    let positive = private(head.capture_checkpoint());
    for _ in 0..2 {
        for cell in 2..CELLS {
            let x = private(fixture.input(cell));
            let out = private(head.step(&fixture.weights, cell, &x, fixture.tokens[cell], false));
            assert!(private(output_bits(&out)) == baseline[cell], "private positive replay bits");
        }
        private(head.restore_checkpoint(&positive));
    }
    let mut other = private(Head::new(geometry(), &fixture.weights));
    assert!(other.restore_checkpoint(&positive).is_err() && other.is_fenced(), "private cross-head snapshot refusal");
    private(other.reset());
    let stale = private(head.capture_checkpoint());
    private(head.reset());
    assert!(head.restore_checkpoint(&stale).is_err() && head.is_fenced(), "private reset stale snapshot refusal");
    private(head.reset());
    let mut malformed = private(head.capture_checkpoint());
    malformed.next_cell = geometry().context + 1;
    assert!(
        head.restore_checkpoint(&malformed).is_err() && head.is_fenced() && head.next_cell() == 0,
        "private malformed snapshot position refusal"
    );
    private(head.reset());
    let x = private(fixture.input(0));
    let out = private(head.step(&fixture.weights, 0, &x, fixture.tokens[0], false));
    assert!(private(output_bits(&out)) == baseline[0], "private failure reset fresh replay");
    let mut malformed = private(head.capture_checkpoint());
    malformed.keys[0] = f32::NAN;
    assert!(head.restore_checkpoint(&malformed).is_err() && head.is_fenced(), "private nonfinite snapshot refusal");
    println!(
        "{}",
        json!({"synthetic_mtp_head_checkpoint":true,"paired_bit_exact_rows":7,"cross_head_rejected":true,
        "reset_stale_rejected":true,"malformed_position_rejected":true,"nonfinite_prefix_rejected":true,"actual_artifact_forward":false})
    );
}

#[stark_test]
fn edge0_mtp_fixture_comparison_rejects_nonfinite_geometry_and_stride() {
    assert!(stage_ratio(&[f32::NAN], &[0.0]).is_err(), "private comparison rejects NaN");
    assert!(stage_ratio(&[0.0], &[f64::INFINITY]).is_err(), "private comparison rejects infinite reference");
    assert!(stage_ratio(&[0.0], &[0.0, 1.0]).is_err(), "private comparison rejects truncated stage");
    assert!(private(stage_ratio(&[1.0], &[0.0])) > 1.0, "private comparison detects mutation");
    assert!(trace_row("attention_scores", &[0.0; 16]).is_err(), "private comparison rejects wrong stride");
    let mut raw = vec![0.0; 64];
    raw[4] = 1.0;
    assert!(
        trace_row("attention_probabilities", &raw).is_err(),
        "private comparison rejects nonzero out-of-fixture slots"
    );
    let expected = Expected {
        width: 2,
        cells: vec![2, 3],
        values: vec![0.0; 4],
    };
    assert!(expected.row(0).is_err() && expected.row(3).is_ok(), "private comparison requires explicit cell indices");
}
