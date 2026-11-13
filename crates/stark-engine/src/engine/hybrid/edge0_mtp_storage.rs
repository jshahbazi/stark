// SPDX-License-Identifier: Apache-2.0
//! Staged original-byte MTP views; no model, GPU or MTP capability admission.
//!
//! Independently described from MIT-licensed Edge0-AI/Edge0 revision
//! b7900cbb322926ccebdcd0c0c5a1280497ecff22, tools/mtp_rt.py::blob_of and
//! include/strata/kernels/cpu/expert.hpp. See LICENSE.strata-reference.
//! Unlike a GGML Q2_0 tensor, each runtime expert has FOUR physical planes:
//! interleaved gate/up codes, down codes, gate/up F16 scales, down F16 scales.
//! The dense F32 norms already include +1 (mtp_rt.py), and are final gamma.

use std::{io::Read, sync::Arc};

use super::{Mapping, edge0_source, encoded_size};
use crate::engine::hybrid::{Error, Result};

pub(super) const DENSE_BYTES: u64 = 116_099_072;
pub(super) const EXPERT_BYTES: u64 = 707_788_800;
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge0MtpEncoding {
    F32,
    Bf16,
    Q8_0,
}

impl Edge0MtpEncoding {
    fn kind(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::Bf16 => 30,
            Self::Q8_0 => 8,
        }
    }

    fn source(self) -> edge0_source::TensorEncoding {
        match self {
            Self::F32 => edge0_source::TensorEncoding::F32,
            Self::Bf16 => edge0_source::TensorEncoding::Bf16,
            Self::Q8_0 => edge0_source::TensorEncoding::Q8_0,
        }
    }
}

struct DenseSpec {
    name: &'static str,
    encoding: Edge0MtpEncoding,
    rows: usize,
    columns: usize,
}

fn dense_specs() -> [DenseSpec; 29] {
    use Edge0MtpEncoding::{Bf16, F32, Q8_0};
    [
        ("fc_embedding.weight", Q8_0, 2560, 2560),
        ("fc_hidden.weight", Q8_0, 2560, 2560),
        ("hyper_connection_mixer.hc_norm.weight", F32, 1, 10240),
        ("hyper_connection_mixer.input_mix_weight_down.weight", Bf16, 320, 10240),
        ("hyper_connection_mixer.input_mix_weight_up.weight", Bf16, 10240, 320),
        ("attn_hyper_connection.block_inject_weight.weight", Bf16, 4, 10240),
        ("attn_hyper_connection.hc_norm.weight", F32, 1, 10240),
        ("attn_hyper_connection.input_mix_weight_down.weight", Bf16, 320, 10240),
        ("attn_hyper_connection.input_mix_weight_up.weight", Bf16, 10240, 320),
        ("mlp.gate.weight", Bf16, 512, 2560),
        ("mlp.shared_expert.down_proj.weight", Q8_0, 2560, 640),
        ("mlp.shared_expert.gate_proj.weight", Q8_0, 640, 2560),
        ("mlp.shared_expert.up_proj.weight", Q8_0, 640, 2560),
        ("mlp.shared_expert_gate.weight", Bf16, 1, 2560),
        ("mlp_hyper_connection.block_inject_weight.weight", Bf16, 4, 10240),
        ("mlp_hyper_connection.hc_norm.weight", F32, 1, 10240),
        ("mlp_hyper_connection.input_mix_weight_down.weight", Bf16, 320, 10240),
        ("mlp_hyper_connection.input_mix_weight_up.weight", Bf16, 10240, 320),
        ("self_attn.indexer.index_qk_proj.weight", Q8_0, 640, 2560),
        ("self_attn.indexer.k_layernorm.weight", F32, 1, 128),
        ("self_attn.indexer.q_layernorm.weight", F32, 1, 128),
        ("self_attn.k_norm.weight", F32, 1, 256),
        ("self_attn.k_proj.weight", Q8_0, 512, 2560),
        ("self_attn.o_proj.weight", Q8_0, 2560, 6144),
        ("self_attn.q_norm.weight", F32, 1, 256),
        ("self_attn.q_proj.weight", Q8_0, 12288, 2560),
        ("self_attn.v_proj.weight", Q8_0, 512, 2560),
        ("pre_fc_norm_embedding.weight", F32, 1, 2560),
        ("pre_fc_norm_hidden.weight", F32, 1, 10240),
    ]
    .map(|(name, encoding, rows, columns)| DenseSpec {
        name,
        encoding,
        rows,
        columns,
    })
}

fn dense_index(source: impl Read) -> Result<edge0_source::MtpDense> {
    let index = edge0_source::read_mtp_dense(source, DENSE_BYTES)?;
    let mut offset = 0_u64;
    for spec in dense_specs() {
        let tensor = index.tensors.get(spec.name).ok_or(Error::Invalid("Edge0 MTP dense tensor inventory"))?;
        let size = spec
            .rows
            .checked_mul(encoded_size(spec.encoding.kind(), spec.columns)?)
            .ok_or(Error::Invalid("Edge0 MTP dense size overflow"))? as u64;
        if tensor.encoding != spec.encoding.source()
            || tensor.shape != [spec.columns as u64, spec.rows as u64]
            || tensor.pool_offset != offset
            || tensor.payload_bytes != size
            || tensor.storage
                != edge0_source::PackStorage::Dense(edge0_source::FileSpan {
                    offset,
                    bytes: size,
                })
        {
            return Err(Error::Invalid("Edge0 MTP dense source geometry"));
        }
        offset = offset.checked_add(size).ok_or(Error::Invalid("Edge0 MTP dense offset overflow"))?;
    }
    if offset != DENSE_BYTES {
        return Err(Error::Invalid("Edge0 MTP dense file coverage"));
    }
    Ok(index)
}

/// A contiguous dense record, owning the original read-only file mapping.
/// F32 norm records are final gamma; callers must not apply another +1.
#[derive(Clone)]
pub struct Edge0MtpDenseView {
    mapping: Arc<Mapping>,
    offset: usize,
    bytes: usize,
    shape: [usize; 2],
    encoding: Edge0MtpEncoding,
}

impl Edge0MtpDenseView {
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn encoding(&self) -> Edge0MtpEncoding {
        self.encoding
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn bytes(&self) -> Result<&[u8]> {
        self.mapping.bytes(self.offset, self.bytes)
    }
}

#[derive(Clone, Copy)]
struct ExpertLayout {
    hidden: usize,
    intermediate: usize,
    experts: usize,
    gu_codes: usize,
    down_codes: usize,
    gu_scales: usize,
    record: usize,
}

impl ExpertLayout {
    fn checked(
        hidden: usize,
        intermediate: usize,
        experts: usize,
        file_bytes: usize,
    ) -> Result<Self> {
        if hidden == 0
            || intermediate == 0
            || experts == 0
            || !hidden.is_multiple_of(64)
            || !intermediate.is_multiple_of(64)
            || hidden > HIDDEN
            || intermediate > INTERMEDIATE
            || experts > EXPERTS
        {
            return Err(Error::Invalid("Edge0 MTP expert geometry"));
        }
        let elements = hidden.checked_mul(intermediate).ok_or(Error::Invalid("Edge0 MTP expert overflow"))?;
        let gu_codes =
            elements.checked_div(64).and_then(|n| n.checked_mul(32)).ok_or(Error::Invalid("Edge0 MTP GU codes"))?;
        let down_codes = elements / 64 * 16;
        let gu_scales = elements / 64 * 4;
        let record = gu_codes
            .checked_add(down_codes)
            .and_then(|n| n.checked_add(gu_scales))
            .and_then(|n| n.checked_add(elements / 64 * 2))
            .ok_or(Error::Invalid("Edge0 MTP expert record"))?;
        if record.checked_mul(experts) != Some(file_bytes) {
            return Err(Error::Invalid("Edge0 MTP expert file coverage"));
        }
        Ok(Self {
            hidden,
            intermediate,
            experts,
            gu_codes,
            down_codes,
            gu_scales,
            record,
        })
    }
}

/// One original runtime expert, not a contiguous GGML Q2_0 tensor.
#[derive(Clone)]
pub struct Edge0MtpExpertView {
    mapping: Arc<Mapping>,
    offset: usize,
    layout: ExpertLayout,
}

/// A row's disjoint code/scale planes. Each block represents 64 consecutive
/// logical columns. Gate/up rows have separate logical accessors below.
pub struct Edge0MtpQ2Row<'a> {
    codes: &'a [u8],
    scales_le: &'a [u8],
    codes_offset: usize,
    scales_offset: usize,
    columns: usize,
}

pub struct Edge0MtpQ2Block<'a> {
    pub codes: &'a [u8; 16],
    pub scale_le: &'a [u8; 2],
}

impl Edge0MtpQ2Row<'_> {
    pub fn codes(&self) -> &[u8] {
        self.codes
    }
    pub fn scales_le(&self) -> &[u8] {
        self.scales_le
    }
    pub fn codes_offset(&self) -> usize {
        self.codes_offset
    }
    pub fn scales_offset(&self) -> usize {
        self.scales_offset
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn block(
        &self,
        block: usize,
    ) -> Result<Edge0MtpQ2Block<'_>> {
        if block >= self.columns / 64 {
            return Err(Error::Invalid("Edge0 MTP Q2 block ID"));
        }
        Ok(Edge0MtpQ2Block {
            codes: self.codes[block * 16..block * 16 + 16]
                .try_into()
                .map_err(|_| Error::Invalid("Edge0 MTP Q2 codes"))?,
            scale_le: self.scales_le[block * 2..block * 2 + 2]
                .try_into()
                .map_err(|_| Error::Invalid("Edge0 MTP Q2 scale"))?,
        })
    }
}

impl Edge0MtpQ2Block<'_> {
    /// Scalar inspection only: little-endian F16 d times (two-bit code - 1).
    /// Four adjacent values occupy bits 0,2,4,6 of each code byte.
    pub fn value(
        &self,
        column: usize,
    ) -> Result<f32> {
        if column >= 64 {
            return Err(Error::Invalid("Edge0 MTP Q2 column"));
        }
        let scale = half::f16::from_bits(u16::from_le_bytes(*self.scale_le)).to_f32();
        let code = (self.codes[column / 4] >> (2 * (column % 4))) & 3;
        Ok(scale * (f32::from(code) - 1.0))
    }
}

impl Edge0MtpExpertView {
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn record_bytes(&self) -> usize {
        self.layout.record
    }
    pub fn gate_up_shape(&self) -> [usize; 2] {
        [self.layout.intermediate, self.layout.hidden]
    }
    pub fn down_shape(&self) -> [usize; 2] {
        [self.layout.hidden, self.layout.intermediate]
    }
    /// Byte strides for successive logical rows: [codes, scales].
    pub fn gate_up_row_strides(&self) -> [usize; 2] {
        [self.layout.hidden / 64 * 32, self.layout.hidden / 64 * 4]
    }
    pub fn down_row_strides(&self) -> [usize; 2] {
        [self.layout.intermediate / 64 * 16, self.layout.intermediate / 64 * 2]
    }
    pub fn plane_offsets(&self) -> [usize; 4] {
        [
            self.offset,
            self.offset + self.layout.gu_codes,
            self.offset + self.layout.gu_codes + self.layout.down_codes,
            self.offset + self.layout.gu_codes + self.layout.down_codes + self.layout.gu_scales,
        ]
    }

    pub fn gate_row(
        &self,
        row: usize,
    ) -> Result<Edge0MtpQ2Row<'_>> {
        self.gu_row(row, false)
    }

    pub fn up_row(
        &self,
        row: usize,
    ) -> Result<Edge0MtpQ2Row<'_>> {
        self.gu_row(row, true)
    }

    fn gu_row(
        &self,
        row: usize,
        up: bool,
    ) -> Result<Edge0MtpQ2Row<'_>> {
        if row >= self.layout.intermediate {
            return Err(Error::Invalid("Edge0 MTP GU row ID"));
        }
        let physical = row * 2 + usize::from(up);
        let blocks = self.layout.hidden / 64;
        self.row(
            self.offset + physical * blocks * 16,
            self.offset + self.layout.gu_codes + self.layout.down_codes + physical * blocks * 2,
            self.layout.hidden,
        )
    }

    pub fn down_row(
        &self,
        row: usize,
    ) -> Result<Edge0MtpQ2Row<'_>> {
        if row >= self.layout.hidden {
            return Err(Error::Invalid("Edge0 MTP down row ID"));
        }
        let blocks = self.layout.intermediate / 64;
        self.row(
            self.offset + self.layout.gu_codes + row * blocks * 16,
            self.offset + self.layout.gu_codes + self.layout.down_codes + self.layout.gu_scales + row * blocks * 2,
            self.layout.intermediate,
        )
    }

    fn row(
        &self,
        codes_offset: usize,
        scales_offset: usize,
        columns: usize,
    ) -> Result<Edge0MtpQ2Row<'_>> {
        Ok(Edge0MtpQ2Row {
            codes: self.mapping.bytes(codes_offset, columns / 64 * 16)?,
            scales_le: self.mapping.bytes(scales_offset, columns / 64 * 2)?,
            codes_offset,
            scales_offset,
            columns,
        })
    }
}

pub(super) struct MtpStorage {
    dense: Arc<Mapping>,
    experts: Arc<Mapping>,
    index: edge0_source::MtpDense,
    layout: ExpertLayout,
}

impl MtpStorage {
    pub(super) fn new(
        dense: Arc<Mapping>,
        experts: Arc<Mapping>,
        index: impl Read,
    ) -> Result<Self> {
        if dense.length as u64 != DENSE_BYTES || experts.length as u64 != EXPERT_BYTES {
            return Err(Error::Invalid("Edge0 MTP mapped file identity"));
        }
        let index = dense_index(index)?;
        let layout = ExpertLayout::checked(HIDDEN, INTERMEDIATE, EXPERTS, experts.length)?;
        Ok(Self {
            dense,
            experts,
            index,
            layout,
        })
    }

    pub(super) fn dense(
        &self,
        name: &str,
    ) -> Result<Edge0MtpDenseView> {
        let spec = dense_specs().into_iter().find(|s| s.name == name).ok_or(Error::Invalid("Edge0 MTP dense name"))?;
        let tensor = self.index.tensors.get(name).ok_or(Error::Invalid("Edge0 MTP dense inventory"))?;
        let offset = usize::try_from(tensor.pool_offset).map_err(|_| Error::Invalid("Edge0 MTP dense offset"))?;
        let bytes = usize::try_from(tensor.payload_bytes).map_err(|_| Error::Invalid("Edge0 MTP dense length"))?;
        self.dense.bytes(offset, bytes)?;
        Ok(Edge0MtpDenseView {
            mapping: Arc::clone(&self.dense),
            offset,
            bytes,
            shape: [spec.rows, spec.columns],
            encoding: spec.encoding,
        })
    }

    pub(super) fn expert(
        &self,
        expert: usize,
    ) -> Result<Edge0MtpExpertView> {
        if expert >= self.layout.experts {
            return Err(Error::Invalid("Edge0 MTP expert ID"));
        }
        let offset = expert.checked_mul(self.layout.record).ok_or(Error::Invalid("Edge0 MTP expert offset"))?;
        self.experts.bytes(offset, self.layout.record)?;
        Ok(Edge0MtpExpertView {
            mapping: Arc::clone(&self.experts),
            offset,
            layout: self.layout,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Seek, SeekFrom, Write},
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use stark_engine_macros::stark_test;

    use super::*;

    // Independent pinned descriptor text, SHA256
    // 991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd.
    const INDEX: &str = "fc_embedding.weight q8_0 2560 2560 0 6963200\n\
fc_hidden.weight q8_0 2560 2560 6963200 6963200\n\
hyper_connection_mixer.hc_norm.weight f32 1 10240 13926400 40960\n\
hyper_connection_mixer.input_mix_weight_down.weight bf16 320 10240 13967360 6553600\n\
hyper_connection_mixer.input_mix_weight_up.weight bf16 10240 320 20520960 6553600\n\
attn_hyper_connection.block_inject_weight.weight bf16 4 10240 27074560 81920\n\
attn_hyper_connection.hc_norm.weight f32 1 10240 27156480 40960\n\
attn_hyper_connection.input_mix_weight_down.weight bf16 320 10240 27197440 6553600\n\
attn_hyper_connection.input_mix_weight_up.weight bf16 10240 320 33751040 6553600\n\
mlp.gate.weight bf16 512 2560 40304640 2621440\n\
mlp.shared_expert.down_proj.weight q8_0 2560 640 42926080 1740800\n\
mlp.shared_expert.gate_proj.weight q8_0 640 2560 44666880 1740800\n\
mlp.shared_expert.up_proj.weight q8_0 640 2560 46407680 1740800\n\
mlp.shared_expert_gate.weight bf16 1 2560 48148480 5120\n\
mlp_hyper_connection.block_inject_weight.weight bf16 4 10240 48153600 81920\n\
mlp_hyper_connection.hc_norm.weight f32 1 10240 48235520 40960\n\
mlp_hyper_connection.input_mix_weight_down.weight bf16 320 10240 48276480 6553600\n\
mlp_hyper_connection.input_mix_weight_up.weight bf16 10240 320 54830080 6553600\n\
self_attn.indexer.index_qk_proj.weight q8_0 640 2560 61383680 1740800\n\
self_attn.indexer.k_layernorm.weight f32 1 128 63124480 512\n\
self_attn.indexer.q_layernorm.weight f32 1 128 63124992 512\n\
self_attn.k_norm.weight f32 1 256 63125504 1024\n\
self_attn.k_proj.weight q8_0 512 2560 63126528 1392640\n\
self_attn.o_proj.weight q8_0 2560 6144 64519168 16711680\n\
self_attn.q_norm.weight f32 1 256 81230848 1024\n\
self_attn.q_proj.weight q8_0 12288 2560 81231872 33423360\n\
self_attn.v_proj.weight q8_0 512 2560 114655232 1392640\n\
pre_fc_norm_embedding.weight f32 1 2560 116047872 10240\n\
pre_fc_norm_hidden.weight f32 1 10240 116058112 40960\n";

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(length: u64) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "stark-edge0-mtp-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let file = fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
            file.set_len(length).unwrap();
            Self(path)
        }
        fn mapping(&self) -> Arc<Mapping> {
            Arc::new(Mapping::new(&self.0).unwrap())
        }
        fn write(
            &self,
            offset: u64,
            bytes: &[u8],
        ) {
            let mut file = fs::OpenOptions::new().write(true).open(&self.0).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(bytes).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    #[stark_test]
    fn edge0_mtp_dense_schema_rejects_names_formats_geometry_offsets_and_truncation() {
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(INDEX.as_bytes())),
            "991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd"
        );
        assert_eq!(dense_index(INDEX.as_bytes()).unwrap().tensors.len(), 29);
        for (from, to) in [
            ("fc_embedding.weight", "other_embedding.weight"),
            ("bf16 320 10240", "bf16 640 5120"),
            ("f32 1 128", "bf16 1 256"),
            ("q8_0 2560 2560 0", "q8_0 2560 2560 1"),
            ("116058112 40960", "116058112 40959"),
            ("fc_hidden.weight", "fc_embedding.weight"),
            ("q8_0", "q2_0"),
            ("116058112", "18446744073709551615"),
        ] {
            assert!(dense_index(INDEX.replacen(from, to, 1).as_bytes()).is_err());
        }
        let swapped = INDEX
            .replacen("fc_embedding.weight", "REPLACE", 1)
            .replacen("fc_hidden.weight", "fc_embedding.weight", 1)
            .replace("REPLACE", "fc_hidden.weight");
        assert!(dense_index(swapped.as_bytes()).is_err());
        let short = Fixture::new(64);
        assert!(MtpStorage::new(short.mapping(), short.mapping(), INDEX.as_bytes()).is_err());
    }

    #[stark_test]
    fn edge0_mtp_dense_views_keep_original_gamma_and_mapping_after_owner_drop() {
        let dense = Fixture::new(DENSE_BYTES);
        dense.write(116_047_872, &1.25_f32.to_le_bytes());
        let tiny = Fixture::new(3456);
        let storage = MtpStorage {
            dense: dense.mapping(),
            experts: tiny.mapping(),
            index: dense_index(INDEX.as_bytes()).unwrap(),
            layout: ExpertLayout::checked(64, 64, 1, 3456).unwrap(),
        };
        let norm = storage.dense("pre_fc_norm_embedding.weight").unwrap();
        assert_eq!(norm.shape(), [1, 2560]);
        assert_eq!(norm.encoding(), Edge0MtpEncoding::F32);
        assert!(storage.dense("../pre_fc_norm_embedding.weight").is_err());
        assert!(storage.dense("pre_fc_norm_embedding").is_err());
        drop(storage);
        drop(dense);
        assert_eq!(&norm.bytes().unwrap()[..4], &1.25_f32.to_le_bytes());
    }

    #[stark_test]
    fn edge0_mtp_q2_four_planes_interleaved_rows_and_values_match_independent_fixture() {
        // Serialize logical GGML-like blocks independently, then emit the
        // runtime's four planes. Small synthetic geometry crosses two input
        // blocks and two experts without any model payload or GPU work.
        let (hidden, intermediate, experts) = (128, 64, 2);
        let mut bytes = Vec::new();
        let scale_bits = [0x3800_u16, 0x3e00, 0x4000]; // 0.5, 1.5, 2
        let scale_values = [0.5_f32, 1.5, 2.0];
        let code = |e: usize, role: usize, row: usize, block: usize, column: usize| {
            (e + 2 * role + 3 * row + block + column) % 4
        };
        for expert in 0..experts {
            let mut gu_codes = Vec::new();
            let mut down_codes = Vec::new();
            let mut gu_scales = Vec::new();
            let mut down_scales = Vec::new();
            for row in 0..intermediate {
                for role in 0..2 {
                    for block in 0..hidden / 64 {
                        for byte in 0..16 {
                            let packed = (0..4).fold(0_u8, |p, lane| {
                                p | ((code(expert, role, row, block, byte * 4 + lane) as u8) << (2 * lane))
                            });
                            gu_codes.push(packed);
                        }
                        gu_scales.extend(scale_bits[(expert + role + row + block) % 3].to_le_bytes());
                    }
                }
            }
            for row in 0..hidden {
                for block in 0..intermediate / 64 {
                    for byte in 0..16 {
                        let packed = (0..4).fold(0_u8, |p, lane| {
                            p | ((code(expert, 2, row, block, byte * 4 + lane) as u8) << (2 * lane))
                        });
                        down_codes.push(packed);
                    }
                    down_scales.extend(scale_bits[(expert + 2 + row + block) % 3].to_le_bytes());
                }
            }
            bytes.extend(gu_codes);
            bytes.extend(down_codes);
            bytes.extend(gu_scales);
            bytes.extend(down_scales);
        }
        assert_eq!(bytes.len(), 13_824);
        let file = Fixture::new(bytes.len() as u64);
        file.write(0, &bytes);
        let layout = ExpertLayout::checked(hidden, intermediate, experts, bytes.len()).unwrap();
        let storage = MtpStorage {
            dense: file.mapping(),
            experts: file.mapping(),
            index: dense_index(INDEX.as_bytes()).unwrap(),
            layout,
        };
        for expert in 0..experts {
            let view = storage.expert(expert).unwrap();
            assert_eq!(view.offset(), expert * 6912);
            assert_eq!(
                view.plane_offsets(),
                [expert * 6912, expert * 6912 + 4096, expert * 6912 + 6144, expert * 6912 + 6656]
            );
            assert_eq!(view.gate_up_shape(), [64, 128]);
            assert_eq!(view.down_shape(), [128, 64]);
            assert_eq!(view.gate_up_row_strides(), [64, 8]);
            assert_eq!(view.down_row_strides(), [16, 2]);
            for role in 0..3 {
                let (rows, columns) = if role == 2 {
                    (hidden, intermediate)
                } else {
                    (intermediate, hidden)
                };
                for row in 0..rows {
                    let value = match role {
                        0 => view.gate_row(row),
                        1 => view.up_row(row),
                        _ => view.down_row(row),
                    }
                    .unwrap();
                    assert_eq!(value.columns(), columns);
                    assert_eq!(value.codes().len(), columns / 64 * 16);
                    assert_eq!(value.scales_le().len(), columns / 64 * 2);
                    for block in 0..columns / 64 {
                        let decoded = value.block(block).unwrap();
                        for column in 0..64 {
                            let expected = scale_values[(expert + role + row + block) % 3]
                                * (code(expert, role, row, block, column) as f32 - 1.0);
                            assert_eq!(decoded.value(column).unwrap().to_bits(), expected.to_bits());
                        }
                        assert!(decoded.value(64).is_err());
                    }
                    assert!(value.block(columns / 64).is_err());
                    assert!(value.block(usize::MAX).is_err());
                }
            }
            assert!(view.gate_row(intermediate).is_err());
            assert!(view.up_row(usize::MAX).is_err());
            assert!(view.down_row(hidden).is_err());
        }
        assert!(storage.expert(experts).is_err());
        assert!(storage.expert(usize::MAX).is_err());
        let retained = storage.expert(1).unwrap();
        drop(storage);
        drop(file);
        assert_eq!(retained.up_row(63).unwrap().block(1).unwrap().value(63).unwrap(), -0.5);
    }

    #[stark_test]
    fn edge0_mtp_expert_geometry_caps_and_exact_final_record_are_checked() {
        let layout = ExpertLayout::checked(HIDDEN, INTERMEDIATE, EXPERTS, EXPERT_BYTES as usize).unwrap();
        assert_eq!(layout.record, 1_382_400);
        assert_eq!((EXPERTS - 1) * layout.record + layout.record, EXPERT_BYTES as usize);
        for (h, f, e, n) in [
            (0, 64, 1, 3456),
            (65, 64, 1, 3456),
            (64, 63, 1, 3456),
            (64, 64, 0, 0),
            (64, 64, 513, 3456 * 513),
            (usize::MAX, 64, 1, 3456),
            (HIDDEN, INTERMEDIATE, EXPERTS, EXPERT_BYTES as usize - 1),
        ] {
            assert!(ExpertLayout::checked(h, f, e, n).is_err());
        }
    }
}
