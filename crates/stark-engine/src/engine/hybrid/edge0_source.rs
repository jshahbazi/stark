//! Bounded, stateless metadata readers for the pinned Edge0 Strata source.
//!
//! Independent implementation of pack v3/native-expert v6 and projection
//! layouts documented by Edge0-AI/Edge0 at
//! b7900cbb322926ccebdcd0c0c5a1280497ecff22 (`edge0-strata/tools/iq_pack.py`,
//! `tools/mtp_pack.py`, and the latent projection/expert headers). Strata is
//! MIT licensed; see LICENSE.strata-reference and provenance/UPSTREAM.md. No
//! upstream implementation is copied. GGML block geometry follows llama.cpp
//! 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT).
//!
//! These readers describe byte spans; they neither map/decode model payloads
//! nor admit a model/backend. They own no recurrent, KV or request state and
//! grant no prefix-reuse capability. A future forward path must inventory and
//! qualify exact state restoration separately.

use std::{collections::BTreeMap, io::Read};

use super::{Error, Result};

pub const LAYERS: usize = 48;
pub const EXPERTS_PER_LAYER: usize = 512;
pub const PROJECTION_GROUPS: usize = 8;
pub const MODEL_WIDTH: u64 = 2560;
pub const LATENT_WIDTH: u64 = 1792;
pub const EXPERT_WIDTH: u64 = 640;
pub const PLE_PHYSICAL_ROWS: u64 = 320_001_536;
pub const PLE_LOGICAL_ROWS: u64 = 320_001_446;
pub const PLE_ROW_BYTES: u64 = 160;
const MAX_TEXT_BYTES: u64 = 256 * 1024;
const MAX_FILE_BYTES: u64 = 1 << 40;
const MAX_DIMENSION: u64 = 400_000_000;
const MAX_TENSORS: usize = 2000;
const PACK_ALIGNMENT: u64 = 64;

fn invalid(message: &'static str) -> Error {
    Error::Invalid(message)
}

fn add(
    a: u64,
    b: u64,
) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| invalid("edge0 span overflow"))
}

fn multiply(
    a: u64,
    b: u64,
) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| invalid("edge0 shape overflow"))
}

fn aligned(
    value: u64,
    alignment: u64,
) -> Result<u64> {
    Ok(add(value, alignment - 1)? / alignment * alignment)
}

fn integer(value: &str) -> Result<u64> {
    if value.is_empty() || value.len() > 20 || !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err(invalid("edge0 manifest integer"));
    }
    value.parse().map_err(|_| invalid("edge0 manifest integer"))
}

fn tensor_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'.' || c == b'_')
}

fn text(mut source: impl Read) -> Result<String> {
    let mut data = Vec::new();
    source.by_ref().take(MAX_TEXT_BYTES + 1).read_to_end(&mut data)?;
    if data.len() as u64 > MAX_TEXT_BYTES {
        return Err(invalid("edge0 metadata size"));
    }
    let value = String::from_utf8(data).map_err(|_| invalid("edge0 metadata encoding"))?;
    if value.lines().any(|line| line.len() > 4096) {
        return Err(invalid("edge0 metadata line size"));
    }
    Ok(value)
}

fn exact_bytes(
    mut source: impl Read,
    count: usize,
) -> Result<Vec<u8>> {
    let mut data = vec![0; count];
    source.read_exact(&mut data)?;
    let mut extra = [0; 1];
    if source.read(&mut extra)? != 0 {
        return Err(invalid("edge0 metadata length"));
    }
    Ok(data)
}

/// A checked byte span relative to one explicitly identified source file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileSpan {
    pub offset: u64,
    pub bytes: u64,
}

impl FileSpan {
    pub fn checked(
        offset: u64,
        bytes: u64,
        file_bytes: u64,
    ) -> Result<Self> {
        if file_bytes > MAX_FILE_BYTES || bytes == 0 || add(offset, bytes)? > file_bytes {
            return Err(invalid("edge0 file span"));
        }
        Ok(Self {
            offset,
            bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorEncoding {
    F32,
    F16,
    Bf16,
    Q8_0,
}

impl TensorEncoding {
    fn row_bytes(
        self,
        columns: u64,
    ) -> Result<u64> {
        let (block, bytes) = match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q8_0 => (32, 34),
        };
        if columns == 0 || columns > MAX_DIMENSION || !columns.is_multiple_of(block) {
            return Err(invalid("edge0 tensor row geometry"));
        }
        multiply(columns / block, bytes)
    }

    fn matrix_bytes(
        self,
        columns: u64,
        rows: u64,
    ) -> Result<u64> {
        if rows == 0 || rows > MAX_DIMENSION {
            return Err(invalid("edge0 tensor rows"));
        }
        let bytes = multiply(self.row_bytes(columns)?, rows)?;
        if bytes > MAX_FILE_BYTES {
            return Err(invalid("edge0 tensor size"));
        }
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackStorage {
    Dense(FileSpan),
    /// Kind 0 consumes no dense/pool bytes. Resolve by the same GGUF name.
    SkeletonGguf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackTensor {
    /// GGUF order: the contiguous dimension first; vectors have one dimension.
    pub shape: Vec<u64>,
    pub encoding: TensorEncoding,
    pub storage: PackStorage,
    pub pool_offset: u64,
    pub payload_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackIndex {
    pub dense_bytes: u64,
    pub tensors: BTreeMap<String, PackTensor>,
}

/// Read only the pinned unsharded, align-64 v3 dialect. Dense spans must cover
/// the file once, in order, with alignment padding and no unindexed tail.
pub fn read_pack_index(
    source: impl Read,
    dense_bytes: u64,
) -> Result<PackIndex> {
    if dense_bytes == 0 || dense_bytes > MAX_FILE_BYTES {
        return Err(invalid("edge0 dense file size"));
    }
    let source = text(source)?;
    let mut lines = source.lines();
    if !lines.next().is_some_and(|line| line.starts_with("# strata pack index v3 -- ")) {
        return Err(Error::Unsupported("edge0 pack index version"));
    }
    let header: Vec<_> = lines.next().ok_or_else(|| invalid("edge0 pack header"))?.split_ascii_whitespace().collect();
    if header.len() != 7 || header[0..4] != ["#", "align", "64", "pool"] || header[5] != "tensors" {
        return Err(invalid("edge0 pack header"));
    }
    let pool_bytes = integer(header[4])?;
    let count = integer(header[6])?;
    if pool_bytes != dense_bytes || count == 0 || count > MAX_TENSORS as u64 {
        return Err(invalid("edge0 pack pool or count"));
    }
    let mut tensors = BTreeMap::new();
    let (mut pool_end, mut dense_end) = (0, 0);
    for line in lines {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        if fields.len() != 19 || !tensor_name(fields[0]) || tensors.len() >= count as usize {
            return Err(invalid("edge0 pack record"));
        }
        let values = fields[1..].iter().map(|value| integer(value)).collect::<Result<Vec<_>>>()?;
        let (file_id, kind, offset, stored, pool_offset, expanded, columns, rows, bits, group, block) = (
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8],
            values[9], values[10],
        );
        if file_id != 0 || values[11..].iter().any(|&value| value != 0) || rows > MAX_DIMENSION {
            return Err(Error::Unsupported("edge0 pack tensor layout"));
        }
        if pool_offset != pool_end {
            return Err(invalid("edge0 pack pool offset"));
        }
        let actual_rows = rows.max(1);
        let (encoding, storage, payload_bytes) = if kind == 0 {
            if (offset, stored, expanded, bits, group, block) != (0, 0, 0, 8, 0, 32) {
                return Err(Error::Unsupported("edge0 delegated tensor layout"));
            }
            let bytes = TensorEncoding::Q8_0.matrix_bytes(columns, actual_rows)?;
            (TensorEncoding::Q8_0, PackStorage::SkeletonGguf, bytes)
        } else {
            let encoding = match kind {
                2 => TensorEncoding::F32,
                4 => TensorEncoding::Bf16,
                5 => TensorEncoding::F16,
                _ => return Err(Error::Unsupported("edge0 dense encoding")),
            };
            let bytes = encoding.matrix_bytes(columns, actual_rows)?;
            if (bits, group, block) != (0, 0, 1) || stored != bytes || expanded != bytes || offset != dense_end {
                return Err(invalid("edge0 dense record span"));
            }
            let span = FileSpan::checked(offset, bytes, dense_bytes)?;
            dense_end = aligned(add(offset, bytes)?, PACK_ALIGNMENT)?;
            (encoding, PackStorage::Dense(span), bytes)
        };
        pool_end = add(pool_end, aligned(expanded, PACK_ALIGNMENT)?)?;
        if pool_end > pool_bytes || dense_end > dense_bytes {
            return Err(invalid("edge0 pack pool bounds"));
        }
        let tensor = PackTensor {
            shape: if rows == 0 {
                vec![columns]
            } else {
                vec![columns, rows]
            },
            encoding,
            storage,
            pool_offset,
            payload_bytes,
        };
        if tensors.insert(fields[0].to_owned(), tensor).is_some() {
            return Err(invalid("edge0 duplicate tensor name"));
        }
    }
    if tensors.len() != count as usize || pool_end != pool_bytes || dense_end != dense_bytes {
        return Err(invalid("edge0 pack coverage"));
    }
    Ok(PackIndex {
        dense_bytes,
        tensors,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertRecord {
    pub gate: FileSpan,
    pub up: FileSpan,
    pub down: FileSpan,
}

/// Q3_K gate/up over the latent input; IQ4_NL down to latent width.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeExperts {
    pub layers: Vec<FileSpan>,
    pub gate_bytes: u64,
    pub down_bytes: u64,
    pub record_bytes: u64,
    pub file_bytes: u64,
}

impl NativeExperts {
    pub fn expert(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<ExpertRecord> {
        if expert >= EXPERTS_PER_LAYER {
            return Err(invalid("edge0 expert index"));
        }
        let layer = self.layers.get(layer).ok_or_else(|| invalid("edge0 expert layer"))?;
        let offset = add(layer.offset, multiply(expert as u64, self.record_bytes)?)?;
        Ok(ExpertRecord {
            gate: FileSpan::checked(offset, self.gate_bytes, self.file_bytes)?,
            up: FileSpan::checked(add(offset, self.gate_bytes)?, self.gate_bytes, self.file_bytes)?,
            down: FileSpan::checked(add(offset, multiply(2, self.gate_bytes)?)?, self.down_bytes, self.file_bytes)?,
        })
    }
}

pub fn read_native_experts(
    source: impl Read,
    file_bytes: u64,
) -> Result<NativeExperts> {
    let source = text(source)?;
    let mut lines = source.lines();
    if lines.next()
        != Some(
            "# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)",
        )
    {
        return Err(Error::Unsupported("edge0 native expert manifest version"));
    }
    let gate_bytes = EXPERT_WIDTH * (LATENT_WIDTH / 256) * 110;
    let down_bytes = LATENT_WIDTH * (EXPERT_WIDTH / 32) * 18;
    let record_bytes = 2 * gate_bytes + down_bytes;
    let layer_bytes = record_bytes * EXPERTS_PER_LAYER as u64;
    if file_bytes != layer_bytes * LAYERS as u64 {
        return Err(invalid("edge0 expert bank size"));
    }
    let mut layers = Vec::with_capacity(LAYERS);
    for line in lines {
        let fields = line.split_ascii_whitespace().map(integer).collect::<Result<Vec<_>>>()?;
        let layer = layers.len();
        if layer >= LAYERS || fields != [layer as u64, 11, 20, layer as u64 * layer_bytes, record_bytes] {
            return Err(invalid("edge0 native expert layer record"));
        }
        layers.push(FileSpan::checked(fields[3], layer_bytes, file_bytes)?);
    }
    if layers.len() != LAYERS {
        return Err(invalid("edge0 native expert layer count"));
    }
    Ok(NativeExperts {
        layers,
        gate_bytes,
        down_bytes,
        record_bytes,
        file_bytes,
    })
}

/// Projection labels are little-endian u32 on disk; validated values fit u8.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionLabels(Box<[u8]>);

impl ProjectionLabels {
    pub fn read(source: impl Read) -> Result<Self> {
        let bytes = exact_bytes(source, LAYERS * EXPERTS_PER_LAYER * 4)?;
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| {
                let value = u32::from_le_bytes(*bytes);
                if value >= PROJECTION_GROUPS as u32 {
                    Err(invalid("edge0 projection group label"))
                } else {
                    Ok(value as u8)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(values.into_boxed_slice()))
    }

    pub fn group(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<u8> {
        if layer >= LAYERS || expert >= EXPERTS_PER_LAYER {
            return Err(invalid("edge0 label index"));
        }
        Ok(self.0[layer * EXPERTS_PER_LAYER + expert])
    }
}

/// The stored positive F32 `s` vector for one layer's Q3_K gate/up products.
/// This reader does not invert or apply it. The pinned active native paths use
/// `(D_g x) / s` once, before expert activation quantization; pack `.scale` and
/// `.input_scale` records remain uninterpreted metadata for this Q3_K layout.
pub fn read_column_scales(source: impl Read) -> Result<Box<[f32]>> {
    let bytes = exact_bytes(source, LATENT_WIDTH as usize * 4)?;
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| {
            let value = f32::from_le_bytes(*bytes);
            if !value.is_finite() || value <= 0.0 {
                Err(invalid("edge0 column scale value"))
            } else {
                Ok(value)
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionMatrix {
    /// Row-major matrix, with Q8_0 blocks along columns.
    pub rows: u64,
    pub columns: u64,
    pub span: FileSpan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionGroup {
    pub down: ProjectionMatrix,
    pub up: ProjectionMatrix,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8Projections {
    pub file_bytes: u64,
    pub layer_bytes: u64,
}

impl Q8Projections {
    pub fn new(file_bytes: u64) -> Result<Self> {
        let group_bytes = TensorEncoding::Q8_0.matrix_bytes(MODEL_WIDTH, LATENT_WIDTH)?;
        let layer_bytes = group_bytes * (PROJECTION_GROUPS * 2) as u64;
        if file_bytes != layer_bytes * LAYERS as u64 {
            return Err(invalid("edge0 q8 projection file size"));
        }
        Ok(Self {
            file_bytes,
            layer_bytes,
        })
    }

    /// Each layer stores all eight D groups, then all eight U groups.
    pub fn group(
        &self,
        layer: usize,
        group: usize,
    ) -> Result<ProjectionGroup> {
        if layer >= LAYERS || group >= PROJECTION_GROUPS {
            return Err(invalid("edge0 projection index"));
        }
        let down_bytes = TensorEncoding::Q8_0.matrix_bytes(MODEL_WIDTH, LATENT_WIDTH)?;
        let up_bytes = TensorEncoding::Q8_0.matrix_bytes(LATENT_WIDTH, MODEL_WIDTH)?;
        let base = multiply(layer as u64, self.layer_bytes)?;
        let down_offset = add(base, multiply(group as u64, down_bytes)?)?;
        let up_offset =
            add(add(base, multiply(PROJECTION_GROUPS as u64, down_bytes)?)?, multiply(group as u64, up_bytes)?)?;
        Ok(ProjectionGroup {
            down: ProjectionMatrix {
                rows: LATENT_WIDTH,
                columns: MODEL_WIDTH,
                span: FileSpan::checked(down_offset, down_bytes, self.file_bytes)?,
            },
            up: ProjectionMatrix {
                rows: MODEL_WIDTH,
                columns: LATENT_WIDTH,
                span: FileSpan::checked(up_offset, up_bytes, self.file_bytes)?,
            },
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpDense {
    pub tensors: BTreeMap<String, PackTensor>,
    pub file_bytes: u64,
}

pub fn read_mtp_dense(
    source: impl Read,
    file_bytes: u64,
) -> Result<MtpDense> {
    if file_bytes == 0 || file_bytes > MAX_FILE_BYTES {
        return Err(invalid("edge0 mtp dense file size"));
    }
    let source = text(source)?;
    let mut tensors = BTreeMap::new();
    let mut end = 0;
    for line in source.lines() {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        if fields.len() != 6 || !tensor_name(fields[0]) || tensors.len() >= 29 {
            return Err(invalid("edge0 mtp dense record"));
        }
        let encoding = match fields[1] {
            "f32" => TensorEncoding::F32,
            "bf16" => TensorEncoding::Bf16,
            "q8_0" => TensorEncoding::Q8_0,
            _ => return Err(Error::Unsupported("edge0 mtp dense encoding")),
        };
        let rows = integer(fields[2])?;
        let columns = integer(fields[3])?;
        let offset = integer(fields[4])?;
        let bytes = integer(fields[5])?;
        if bytes != encoding.matrix_bytes(columns, rows)? || offset != end {
            return Err(invalid("edge0 mtp dense span"));
        }
        let span = FileSpan::checked(offset, bytes, file_bytes)?;
        end = add(offset, bytes)?;
        let tensor = PackTensor {
            shape: vec![columns, rows],
            encoding,
            storage: PackStorage::Dense(span),
            pool_offset: offset,
            payload_bytes: bytes,
        };
        if tensors.insert(fields[0].to_owned(), tensor).is_some() {
            return Err(invalid("edge0 duplicate mtp tensor name"));
        }
    }
    if tensors.len() != 29 || end != file_bytes {
        return Err(invalid("edge0 mtp dense coverage"));
    }
    Ok(MtpDense {
        tensors,
        file_bytes,
    })
}

/// PLE head offsets address logical rows; the stored tensor has 90 padding rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramTableLayout {
    pub physical_rows: u64,
    pub logical_rows: u64,
    pub payload_bytes: u64,
    heads: Vec<(u64, u64)>,
}

impl NgramTableLayout {
    pub fn new(
        physical_rows: u64,
        head_offsets: &[u64],
        head_vocab_sizes: &[u64],
    ) -> Result<Self> {
        if physical_rows != PLE_PHYSICAL_ROWS || head_offsets.len() != 16 || head_vocab_sizes.len() != 16 {
            return Err(invalid("edge0 ple stored geometry"));
        }
        let mut logical_rows = 0;
        let mut heads = Vec::with_capacity(16);
        for (&offset, &rows) in head_offsets.iter().zip(head_vocab_sizes) {
            if rows == 0 || offset != logical_rows {
                return Err(invalid("edge0 ple head range"));
            }
            logical_rows = add(offset, rows)?;
            if logical_rows > physical_rows {
                return Err(invalid("edge0 ple head bounds"));
            }
            heads.push((offset, rows));
        }
        if logical_rows != PLE_LOGICAL_ROWS {
            return Err(invalid("edge0 ple logical geometry"));
        }
        Ok(Self {
            physical_rows,
            logical_rows,
            payload_bytes: multiply(physical_rows, PLE_ROW_BYTES)?,
            heads,
        })
    }

    pub fn head_row(
        &self,
        head: usize,
        row: u64,
    ) -> Result<FileSpan> {
        let &(offset, rows) = self.heads.get(head).ok_or_else(|| invalid("edge0 ple head index"))?;
        if row >= rows {
            return Err(invalid("edge0 ple row index"));
        }
        FileSpan::checked(multiply(add(offset, row)?, PLE_ROW_BYTES)?, PLE_ROW_BYTES, self.payload_bytes)
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn pack() -> String {
        "# strata pack index v3 -- synthetic fixture\n# align 64 pool 64 tensors 2\n\
         dense.weight 0 4 0 8 0 8 4 0 0 0 1 0 0 0 0 0 0 0\n\
         native.weight 0 0 0 0 64 0 32 2 8 0 32 0 0 0 0 0 0 0\n"
            .to_owned()
    }

    #[stark_test]
    fn pack_describes_dense_and_delegated_storage() {
        let index = read_pack_index(pack().as_bytes(), 64).unwrap();
        assert_eq!(index.tensors["dense.weight"].shape, [4]);
        assert_eq!(
            index.tensors["dense.weight"].storage,
            PackStorage::Dense(FileSpan {
                offset: 0,
                bytes: 8
            })
        );
        assert_eq!(index.tensors["native.weight"].shape, [32, 2]);
        assert_eq!(index.tensors["native.weight"].storage, PackStorage::SkeletonGguf);
        assert_eq!(index.tensors["native.weight"].payload_bytes, 68);
    }

    #[stark_test]
    fn pack_rejects_versions_types_offsets_reserved_fields_and_names() {
        for (before, after) in [
            ("index v3", "index v4"),
            ("align 64", "align 32"),
            ("tensors 2", "tensors 3"),
            ("0 4 0 8", "1 4 0 8"),
            ("0 4 0 8", "0 3 0 8"),
            ("0 4 0 8", "0 4 64 8"),
            ("0 4 0 8", "0 4 0 10"),
            ("64 0 32 2", "0 0 32 2"),
            ("32 2 8 0 32", "31 2 8 0 32"),
            ("native.weight", "dense.weight"),
            ("dense.weight", "../outside"),
            ("0 4 0 8", "0 4 -1 8"),
            ("0 4 0 8", "0 4 18446744073709551616 8"),
            ("8 0 32 0 0 0 0 0 0 0", "8 0 32 1 0 0 0 0 0 0"),
        ] {
            assert!(read_pack_index(pack().replace(before, after).as_bytes(), 64).is_err());
        }
        assert!(read_pack_index(pack().as_bytes(), 128).is_err());
        assert!(read_pack_index(&vec![b'x'; MAX_TEXT_BYTES as usize + 1][..], 64).is_err());
        assert!(read_pack_index(&[0xff][..], 64).is_err());
        assert!(FileSpan::checked(u64::MAX, 2, 64).is_err());
    }

    fn experts() -> String {
        let mut source =
            "# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)\n"
                .to_owned();
        for layer in 0..LAYERS {
            source.push_str(&format!("{layer} 11 20 {} 1630720\n", layer as u64 * 834_928_640));
        }
        source
    }

    #[stark_test]
    fn expert_bank_spans_cover_q3k_gate_up_and_iq4nl_down() {
        let bank = read_native_experts(experts().as_bytes(), 40_076_574_720).unwrap();
        assert_eq!((bank.gate_bytes, bank.down_bytes, bank.record_bytes), (492800, 645120, 1630720));
        let first = bank.expert(0, 0).unwrap();
        assert_eq!(first.gate.offset, 0);
        assert_eq!(first.up.offset, 492800);
        assert_eq!(first.down.offset, 985600);
        let last = bank.expert(47, 511).unwrap();
        assert_eq!(last.down.offset + last.down.bytes, bank.file_bytes);
        assert!(bank.expert(48, 0).is_err());
        assert!(bank.expert(0, 512).is_err());
    }

    #[stark_test]
    fn expert_bank_rejects_changed_versions_records_and_truncation() {
        for (before, after) in [
            ("v6:", "v5:"),
            ("0 11 20 0", "0 11 11 0"),
            ("1 11 20 834928640", "1 11 20 0"),
            ("1630720", "1630784"),
            ("47 11 20", "46 11 20"),
        ] {
            assert!(read_native_experts(experts().replace(before, after).as_bytes(), 40_076_574_720).is_err());
        }
        assert!(read_native_experts(experts().as_bytes(), 40_076_574_719).is_err());
        let source = experts();
        assert!(read_native_experts(source[..source.rfind("47 ").unwrap()].as_bytes(), 40_076_574_720).is_err());
    }

    #[stark_test]
    fn labels_and_column_scales_are_exact_bounded_little_endian_metadata() {
        let mut labels = vec![0; LAYERS * EXPERTS_PER_LAYER * 4];
        labels[..4].copy_from_slice(&7_u32.to_le_bytes());
        assert_eq!(ProjectionLabels::read(&labels[..]).unwrap().group(0, 0).unwrap(), 7);
        for bad in [8_u32, u32::MAX] {
            labels[..4].copy_from_slice(&bad.to_le_bytes());
            assert!(ProjectionLabels::read(&labels[..]).is_err());
        }
        assert!(ProjectionLabels::read(&labels[..labels.len() - 1]).is_err());
        labels.push(0);
        assert!(ProjectionLabels::read(&labels[..]).is_err());
        let mut scales = 1.0_f32.to_le_bytes().repeat(LATENT_WIDTH as usize);
        assert_eq!(read_column_scales(&scales[..]).unwrap().len(), LATENT_WIDTH as usize);
        for value in [0.0_f32, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            scales[..4].copy_from_slice(&value.to_le_bytes());
            assert!(read_column_scales(&scales[..]).is_err());
        }
        scales[..4].copy_from_slice(&f32::from_bits(1).to_le_bytes());
        assert!(read_column_scales(&scales[..]).is_ok());
        assert!(read_column_scales(&scales[..scales.len() - 1]).is_err());
        scales.push(0);
        assert!(read_column_scales(&scales[..]).is_err());
    }

    #[stark_test]
    fn q8_projection_groups_preserve_all_down_then_all_up_order() {
        let layout = Q8Projections::new(3_743_416_320).unwrap();
        assert_eq!(layout.layer_bytes, 77_987_840);
        let first = layout.group(0, 0).unwrap();
        assert_eq!((first.down.rows, first.down.columns), (1792, 2560));
        assert_eq!((first.up.rows, first.up.columns), (2560, 1792));
        assert_eq!(
            first.down.span,
            FileSpan {
                offset: 0,
                bytes: 4_874_240
            }
        );
        assert_eq!(first.up.span.offset, 38_993_920);
        let last = layout.group(47, 7).unwrap();
        assert_eq!(last.up.span.offset + last.up.span.bytes, layout.file_bytes);
        assert!(layout.group(48, 0).is_err());
        assert!(layout.group(0, 8).is_err());
        assert!(Q8Projections::new(3_743_416_319).is_err());
        assert!(Q8Projections::new(1_981_808_640).is_err());
    }

    fn mtp() -> String {
        (0..29).map(|i| format!("norm.{i}.weight f32 1 4 {} 16\n", i * 16)).collect()
    }

    #[stark_test]
    fn mtp_dense_records_validate_row_encoding_and_complete_file_coverage() {
        let dense = read_mtp_dense(mtp().as_bytes(), 29 * 16).unwrap();
        assert_eq!(dense.tensors["norm.0.weight"].shape, [4, 1]);
        for (before, after) in [
            ("norm.1.weight", "norm.0.weight"),
            ("f32 1 4 16 16", "f32 1 4 0 16"),
            ("f32 1 4", "q8_0 1 4"),
            ("f32 1 4", "f16 1 4"),
            ("f32 1 4", "f32 0 4"),
        ] {
            assert!(read_mtp_dense(mtp().replace(before, after).as_bytes(), 29 * 16).is_err());
        }
        assert!(read_mtp_dense(mtp().as_bytes(), 29 * 16 + 1).is_err());
    }

    fn heads() -> (Vec<u64>, Vec<u64>) {
        let sizes = vec![
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093,
            20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
        ];
        let mut offsets = Vec::new();
        let mut next = 0;
        for size in &sizes {
            offsets.push(next);
            next += size;
        }
        (offsets, sizes)
    }

    #[stark_test]
    fn ple_layout_preserves_stored_padding_and_bounds_logical_heads() {
        let (offsets, sizes) = heads();
        let layout = NgramTableLayout::new(PLE_PHYSICAL_ROWS, &offsets, &sizes).unwrap();
        assert_eq!(layout.physical_rows - layout.logical_rows, 90);
        assert_eq!(layout.payload_bytes, 51_200_245_760);
        let last = layout.head_row(15, sizes[15] - 1).unwrap();
        assert_eq!(last.offset + last.bytes, PLE_LOGICAL_ROWS * PLE_ROW_BYTES);
        assert!(layout.head_row(15, sizes[15]).is_err());
        assert!(layout.head_row(16, 0).is_err());
        assert!(NgramTableLayout::new(PLE_LOGICAL_ROWS, &offsets, &sizes).is_err());
        let mut bad = offsets.clone();
        bad[1] -= 1;
        assert!(NgramTableLayout::new(PLE_PHYSICAL_ROWS, &bad, &sizes).is_err());
        let mut bad = sizes.clone();
        bad[15] += 91;
        assert!(NgramTableLayout::new(PLE_PHYSICAL_ROWS, &offsets, &bad).is_err());
        assert!(NgramTableLayout::new(PLE_PHYSICAL_ROWS, &offsets[..15], &sizes[..15]).is_err());
    }

    /// Explicit local component check: reads only manifests, labels and scales.
    /// Payload file sizes are inspected through metadata, never payload reads.
    #[stark_test]
    #[ignore]
    fn prepared_source_metadata_readers() {
        use std::{fs::File, path::PathBuf};

        let path = PathBuf::from(std::env::var_os("STARK_EDGE0_SOURCE_PATH").expect("source fixture required"));
        let size = |name: &str| std::fs::metadata(path.join(name)).unwrap().len();
        let pack = read_pack_index(File::open(path.join("index.txt")).unwrap(), size("dense.bin")).unwrap();
        assert_eq!(pack.tensors.len(), 1367);
        assert_eq!(pack.tensors.values().filter(|t| t.storage == PackStorage::SkeletonGguf).count(), 302);
        let experts =
            read_native_experts(File::open(path.join("native_experts.txt")).unwrap(), size("experts.bin")).unwrap();
        assert_eq!(experts.expert(47, 511).unwrap().down.offset + experts.down_bytes, size("experts.bin"));
        let labels = ProjectionLabels::read(File::open(path.join("labels.bin")).unwrap()).unwrap();
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS_PER_LAYER {
                assert!(labels.group(layer, expert).unwrap() < 8);
            }
            assert_eq!(
                read_column_scales(File::open(path.join(format!("scales/l{layer}.f32"))).unwrap()).unwrap().len(),
                1792
            );
        }
        let projections = Q8Projections::new(size("latent_dense_q8.bin")).unwrap();
        assert_eq!(projections.group(47, 7).unwrap().up.span.offset + 4_874_240, size("latent_dense_q8.bin"));
        let mtp = read_mtp_dense(File::open(path.join("mtp/dense.txt")).unwrap(), size("mtp/dense.bin")).unwrap();
        assert_eq!(mtp.tensors.len(), 29);
        let (offsets, sizes) = heads();
        let ple = NgramTableLayout::new(PLE_PHYSICAL_ROWS, &offsets, &sizes).unwrap();
        assert_eq!(ple.payload_bytes, 51_200_245_760);
    }
}
