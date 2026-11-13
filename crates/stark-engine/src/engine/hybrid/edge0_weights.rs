//! Explicit, original-byte views for the Edge0 Q3_K reference path.
//!
//! Format authority: Edge0-AI/Edge0 b7900cbb322926ccebdcd0c0c5a1280497ecff22,
//! Strata pack v3/native experts v6 (MIT; LICENSE.strata-reference), and
//! llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT). Transformation
//! authority is conversion/qwen.py, conversion/qwen4exp.py and
//! src/models/qwen4exp.cpp at that llama.cpp pin. No source code is copied.
//!
//! GGUF norms already contain their centered +1; ssm_a already stores
//! -exp(A_log). Recurrent V heads and their related operands use the converted
//! tiled order. Aliases preserve those bytes: the Edge0 forward path must use
//! the matching conventions, not undo/reapply these transformations.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::{
    super::edge0_source::{self, PackStorage, TensorEncoding},
    *,
};

#[cfg(feature = "metal")]
#[path = "edge0_weights_metal.rs"]
mod metal_backend;

// Staged storage inspection surface; no inference caller is enabled yet.
#[allow(dead_code)]
#[path = "edge0_mtp_storage.rs"]
mod mtp_storage;
pub use mtp_storage::{Edge0MtpDenseView, Edge0MtpEncoding, Edge0MtpExpertView, Edge0MtpQ2Block, Edge0MtpQ2Row};

const TABLE_NAME: &str = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
const TABLE_ROWS: usize = 320_001_536;
const LOGICAL_ROWS: usize = 320_001_446;
const TABLE_WIDTH: usize = 160;
// Exact source scale 0.00019931793212890625, without decimal truncation.
const TABLE_SCALE: f32 = f32::from_bits(0x3951_0000);

#[cfg(feature = "metal")]
fn remaining_edge0_weight_bytes(
    banks: usize,
    statics: usize,
) -> Option<usize> {
    43_819_991_040usize.saturating_sub(banks).checked_add(MAX_STATIC_VIEW_BYTES.saturating_sub(statics))
}

#[derive(Clone, Copy, Debug)]
pub enum Edge0Matrix {
    ProjectionDown {
        layer: usize,
        group: usize,
    },
    ProjectionUp {
        layer: usize,
        group: usize,
    },
    ExpertGate {
        layer: usize,
        expert: usize,
    },
    ExpertUp {
        layer: usize,
        expert: usize,
    },
    ExpertDown {
        layer: usize,
        expert: usize,
    },
}

pub(super) struct Edge0Data {
    expert_file: usize,
    projection_file: usize,
    table_file: usize,
    table_offset: usize,
    table_shape: Vec<usize>,
    labels: edge0_source::ProjectionLabels,
    scales: Vec<Box<[f32]>>,
    experts: edge0_source::NativeExperts,
    projections: edge0_source::Q8Projections,
    mtp: mtp_storage::MtpStorage,
    #[cfg(feature = "metal")]
    mtp_dense_file: usize,
    #[cfg(feature = "metal")]
    mtp_expert_file: usize,
    integers: HashMap<String, Vec<i64>>,
    integer_shapes: HashMap<String, Vec<usize>>,
    metal_counters: [std::sync::atomic::AtomicU64; 4],
    #[cfg(feature = "metal")]
    metal_admitted: std::sync::atomic::AtomicBool,
    #[cfg(feature = "metal")]
    metal_dynamic_allowance: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "metal")]
    metal_resident: std::sync::atomic::AtomicBool,
    #[cfg(feature = "metal")]
    mtp_admitted: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Edge0Data {
    pub(super) fn virtual_shape(
        &self,
        name: &str,
    ) -> Option<&[usize]> {
        if name == TABLE_NAME {
            Some(&self.table_shape)
        } else {
            self.integer_shapes.get(name).map(Vec::as_slice)
        }
    }

    pub(super) fn virtual_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.integers.keys().cloned().collect();
        names.push(TABLE_NAME.to_owned());
        names
    }

    pub(super) fn integers(
        &self,
        name: &str,
    ) -> Option<&[i64]> {
        self.integers.get(name).map(Vec::as_slice)
    }

    pub(super) fn is_table(
        &self,
        name: &str,
    ) -> bool {
        name == TABLE_NAME
    }

    pub(super) fn row(
        &self,
        weights: &Weights,
        row: usize,
    ) -> Result<Vec<f32>> {
        // Padded physical rows describe storage, not valid hash-table entries.
        if row >= LOGICAL_ROWS {
            return Err(Error::Invalid("Edge0 PLE logical row bound"));
        }
        let offset = row
            .checked_mul(TABLE_WIDTH)
            .and_then(|n| n.checked_add(self.table_offset))
            .ok_or(Error::Invalid("Edge0 PLE row overflow"))?;
        let bytes = weights.files[self.table_file].bytes(offset, TABLE_WIDTH)?;
        let mut output = vec![0.0; TABLE_WIDTH];
        decode_edge0_fp8_e4m3_row(bytes, TABLE_SCALE, &mut output)?;
        Ok(output)
    }

    fn matrix(
        &self,
        matrix: Edge0Matrix,
    ) -> Result<Tensor> {
        let (file, span, kind, rows, columns) = match matrix {
            Edge0Matrix::ProjectionDown {
                layer,
                group,
            } => {
                let projection = self.projections.group(layer, group)?.down;
                (self.projection_file, projection.span, 8, projection.rows, projection.columns)
            },
            Edge0Matrix::ProjectionUp {
                layer,
                group,
            } => {
                let projection = self.projections.group(layer, group)?.up;
                (self.projection_file, projection.span, 8, projection.rows, projection.columns)
            },
            Edge0Matrix::ExpertGate {
                layer,
                expert,
            } => (self.expert_file, self.experts.expert(layer, expert)?.gate, 11, 640, 1792),
            Edge0Matrix::ExpertUp {
                layer,
                expert,
            } => (self.expert_file, self.experts.expert(layer, expert)?.up, 11, 640, 1792),
            Edge0Matrix::ExpertDown {
                layer,
                expert,
            } => (self.expert_file, self.experts.expert(layer, expert)?.down, 20, 1792, 640),
        };
        let offset = usize::try_from(span.offset).map_err(|_| Error::Invalid("Edge0 matrix offset"))?;
        let size = usize::try_from(span.bytes).map_err(|_| Error::Invalid("Edge0 matrix size"))?;
        let rows = usize::try_from(rows).map_err(|_| Error::Invalid("Edge0 matrix rows"))?;
        let columns = usize::try_from(columns).map_err(|_| Error::Invalid("Edge0 matrix columns"))?;
        if rows.checked_mul(encoded_size(kind, columns)?) != Some(size) || size > MAX_SELECTED_BYTES {
            return Err(Error::Invalid("Edge0 selected matrix geometry"));
        }
        Ok(Tensor {
            file,
            offset,
            size,
            shape: vec![rows, columns],
            kind,
            affine: None,
        })
    }
}

fn source_file(
    path: &Path,
    name: &str,
) -> Result<File> {
    if name.is_empty()
        || name.contains('\\')
        || name.split('/').any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(Error::Invalid("Edge0 source relative path"));
    }
    let mut target = path.to_path_buf();
    for component in name.split('/') {
        target.push(component);
        if fs::symlink_metadata(&target)?.file_type().is_symlink() {
            return Err(Error::Invalid("Edge0 source path is linked"));
        }
    }
    let file = fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(target)?;
    if !file.metadata()?.is_file() {
        return Err(Error::Invalid("Edge0 source is not a regular file"));
    }
    Ok(file)
}

fn prepared_receipt(path: &Path) -> Result<()> {
    let mut source = source_file(path, "native-source.json")?;
    if source.metadata()?.len() > 1024 * 1024 {
        return Err(Error::Invalid("Edge0 preparation receipt size"));
    }
    let mut bytes = Vec::new();
    source.by_ref().take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024
        || format!("{:x}", Sha256::digest(&bytes)) != "8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e"
    {
        return Err(Error::Invalid("Edge0 preparation receipt identity"));
    }
    let receipt: Value = serde_json::from_slice(&bytes)?;
    verify_small_receipt_files(path, &receipt)?;
    // The receipt binds all 69 source hashes from atomic Phase 1 preparation.
    // Opening does not repeat a 103 GB hash pass. Source bytes must remain
    // immutable for this store's lifetime, like the other Weights loaders.
    Ok(())
}

fn verify_small_receipt_files(
    path: &Path,
    receipt: &Value,
) -> Result<()> {
    const SMALL_LIMIT: u64 = 16 * 1024 * 1024;
    const TOTAL_LIMIT: u64 = 64 * 1024 * 1024;
    const LARGE: [&str; 7] = [
        "experts.bin",
        "ple-fp8.gguf",
        "model-skeleton.gguf",
        "latent_dense_q8.bin",
        "dense.bin",
        "mtp/experts.bin",
        "mtp/dense.bin",
    ];
    let files = receipt
        .get("files")
        .and_then(Value::as_object)
        .filter(|files| files.len() == 72)
        .ok_or(Error::Invalid("Edge0 receipt file inventory"))?;
    let mut total = 0_u64;
    for (name, record) in files {
        let size = record.get("size").and_then(Value::as_u64).ok_or(Error::Invalid("Edge0 receipt file size"))?;
        let expected = record
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|hash| hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
            .ok_or(Error::Invalid("Edge0 receipt file hash"))?;
        let source = source_file(path, name)?;
        let before = source.metadata()?;
        if before.len() != size {
            return Err(Error::Invalid("Edge0 source receipt length"));
        }
        if LARGE.contains(&name.as_str()) {
            continue;
        }
        total = total.checked_add(size).ok_or(Error::Invalid("Edge0 small metadata size overflow"))?;
        if size > SMALL_LIMIT || total > TOTAL_LIMIT {
            return Err(Error::Invalid("Edge0 small metadata bound"));
        }
        let mut stream = source.take(size + 1);
        let mut data = Vec::new();
        stream.read_to_end(&mut data)?;
        if data.len() as u64 != size || format!("{:x}", Sha256::digest(&data)) != expected {
            return Err(Error::Invalid("Edge0 small metadata identity"));
        }
        let after = stream.get_ref().metadata()?;
        use std::os::unix::fs::MetadataExt;
        if (
            before.dev(),
            before.ino(),
            before.len(),
            before.mtime(),
            before.mtime_nsec(),
            before.ctime(),
            before.ctime_nsec(),
        ) != (
            after.dev(),
            after.ino(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec(),
        ) {
            return Err(Error::Invalid("Edge0 small metadata changed"));
        }
    }
    Ok(())
}

fn map_file(
    weights: &mut Weights,
    path: &Path,
    name: &str,
    expected: u64,
) -> Result<usize> {
    let file = source_file(path, name)?;
    if file.metadata()?.len() != expected {
        return Err(Error::Invalid("Edge0 pinned payload size"));
    }
    // One virtual mapping per file; selected GPU wrappers retain this mapping.
    let mapping = Arc::new(Mapping::new(&path.join(name))?);
    if mapping.length as u64 != expected {
        return Err(Error::Invalid("Edge0 payload changed while opening"));
    }
    let index = weights.files.len();
    weights.files.push(mapping);
    Ok(index)
}

fn expected_skeleton() -> BTreeMap<String, (u32, Vec<usize>)> {
    let mut expected = BTreeMap::new();
    let mut add = |name: String, kind, shape: &[usize]| {
        expected.insert(name, (kind, shape.to_vec()));
    };
    for name in ["token_embd.weight", "output.weight"] {
        add(name.to_owned(), 8, &[248320, 2560]);
    }
    add("output_hc_norm.weight".to_owned(), 0, &[10240]);
    add("output_hc_down.weight".to_owned(), 30, &[320, 10240]);
    add("output_hc_up.weight".to_owned(), 30, &[10240, 320]);
    for layer in 0..48 {
        let prefix = format!("blk.{layer}");
        for (suffix, kind, shape) in [
            ("ffn_gate_inp.weight", 30, vec![512, 2560]),
            ("ffn_gate_inp_shexp.weight", 30, vec![1, 2560]),
            ("ffn_gate_shexp.weight", 8, vec![640, 2560]),
            ("ffn_up_shexp.weight", 8, vec![640, 2560]),
            ("ffn_down_shexp.weight", 8, vec![2560, 640]),
        ] {
            add(format!("{prefix}.{suffix}"), kind, &shape);
        }
        for block in ["attn", "ffn"] {
            for (suffix, kind, shape) in [
                ("norm", 0, vec![10240]),
                ("down", 30, vec![320, 10240]),
                ("up", 30, vec![10240, 320]),
                ("inject", 30, vec![4, 10240]),
            ] {
                add(format!("{prefix}.hc_{block}_{suffix}.weight"), kind, &shape);
            }
        }
        let specifications = if layer % 4 == 3 {
            vec![
                ("attn_q.weight", 8, vec![12288, 2560]),
                ("attn_k.weight", 8, vec![512, 2560]),
                ("attn_v.weight", 8, vec![512, 2560]),
                ("attn_output.weight", 8, vec![2560, 6144]),
                ("attn_q_norm.weight", 0, vec![256]),
                ("attn_k_norm.weight", 0, vec![256]),
                ("indexer.q_proj.weight", 30, vec![512, 2560]),
                ("indexer.k_proj.weight", 30, vec![128, 2560]),
                ("indexer.q_norm.weight", 0, vec![128]),
                ("indexer.k_norm.weight", 0, vec![128]),
            ]
        } else {
            vec![
                ("attn_qkv.weight", 8, vec![10240, 2560]),
                ("attn_gate.weight", 8, vec![6144, 2560]),
                ("ssm_out.weight", 8, vec![2560, 6144]),
                ("ssm_alpha.weight", 30, vec![48, 2560]),
                ("ssm_beta.weight", 30, vec![48, 2560]),
                ("ssm_a", 0, vec![48]),
                ("ssm_dt.bias", 0, vec![48]),
                ("ssm_conv1d.weight", 0, vec![10240, 4]),
                ("ssm_norm.weight", 0, vec![128]),
            ]
        };
        for (suffix, kind, shape) in specifications {
            add(format!("{prefix}.{suffix}"), kind, &shape);
        }
        if layer == 1 {
            for (suffix, kind, shape) in [
                ("ple_key.weight", 30, vec![10240, 2560]),
                ("ple_value.weight", 30, vec![2560, 2560]),
                ("ple_conv1d.weight", 1, vec![10240, 4]),
                ("ple_norm_key.weight", 0, vec![10240]),
                ("ple_norm_query.weight", 0, vec![10240]),
                ("ple_norm_conv.weight", 0, vec![10240]),
            ] {
                add(format!("{prefix}.{suffix}"), kind, &shape);
            }
        }
    }
    expected
}

fn validate_skeleton(weights: &Weights) -> Result<()> {
    super::super::edge0_config::validate_metadata(weights.metadata())?;
    let expected = expected_skeleton();
    if expected.len() != 1079 || weights.tensors.len() != expected.len() {
        return Err(Error::Invalid("Edge0 skeleton tensor inventory"));
    }
    for (name, (kind, shape)) in expected {
        let tensor = weights.tensor(&name)?;
        if tensor.kind != kind
            || tensor.shape != shape
            || tensor.affine.is_some()
            || tensor.size != encoded_size(kind, product(&shape)?)?
        {
            return Err(Error::Invalid("Edge0 skeleton tensor format or shape"));
        }
    }
    Ok(())
}

fn encoding_kind(encoding: TensorEncoding) -> u32 {
    match encoding {
        TensorEncoding::F32 => 0,
        TensorEncoding::F16 => 1,
        TensorEncoding::Bf16 => 30,
        TensorEncoding::Q8_0 => 8,
    }
}

fn bind_pool(
    weights: &mut Weights,
    pool: edge0_source::PackIndex,
    file: usize,
) -> Result<()> {
    let mut expected: BTreeSet<_> = weights.tensors.keys().cloned().collect();
    for layer in 0..48 {
        for role in ["gate", "up", "down"] {
            for suffix in ["scale", "input_scale"] {
                expected.insert(format!("blk.{layer}.ffn_{role}_exps.{suffix}"));
            }
        }
    }
    if pool.tensors.len() != 1367 || pool.tensors.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return Err(Error::Invalid("Edge0 dense pool tensor inventory"));
    }
    for (name, record) in pool.tensors {
        let mut shape: Vec<_> = record
            .shape
            .into_iter()
            .map(|d| usize::try_from(d).map_err(|_| Error::Invalid("Edge0 pool shape")))
            .collect::<Result<_>>()?;
        shape.reverse();
        let kind = encoding_kind(record.encoding);
        if let Some(original) = weights.tensors.get(&name) {
            if original.shape != shape || original.kind != kind || original.size as u64 != record.payload_bytes {
                return Err(Error::Invalid("Edge0 dense pool and skeleton disagree"));
            }
        } else if shape != [512] || kind != 0 || record.payload_bytes != 2048 {
            return Err(Error::Invalid("Edge0 expert scale metadata shape"));
        }
        match record.storage {
            PackStorage::SkeletonGguf => {
                if kind != 8 || !weights.tensors.contains_key(&name) {
                    return Err(Error::Invalid("Edge0 delegated pool tensor"));
                }
            },
            PackStorage::Dense(span) => {
                if kind == 8 {
                    return Err(Error::Invalid("Edge0 q8 pool storage"));
                }
                let offset = usize::try_from(span.offset).map_err(|_| Error::Invalid("Edge0 pool offset"))?;
                let size = usize::try_from(span.bytes).map_err(|_| Error::Invalid("Edge0 pool size"))?;
                weights.files[file].bytes(offset, size)?;
                weights.tensors.insert(
                    name,
                    Tensor {
                        file,
                        offset,
                        size,
                        shape,
                        kind,
                        affine: None,
                    },
                );
            },
        }
    }
    Ok(())
}

fn alias(
    weights: &mut Weights,
    target: String,
    source: &str,
    shape: Option<&[usize]>,
) -> Result<()> {
    let mut tensor = weights.tensor(source)?.clone();
    if let Some(shape) = shape {
        if product(shape)? != product(&tensor.shape)? {
            return Err(Error::Invalid("Edge0 alias reshape"));
        }
        tensor.shape = shape.to_vec();
    }
    if weights.tensors.insert(target, tensor).is_some() {
        return Err(Error::Invalid("Edge0 duplicate alias"));
    }
    Ok(())
}

fn alias_rows(
    weights: &mut Weights,
    target: String,
    source: &str,
    start: usize,
    rows: usize,
) -> Result<()> {
    let mut tensor = weights.tensor(source)?.clone();
    if tensor.shape.len() != 2
        || tensor.affine.is_some()
        || rows == 0
        || start.checked_add(rows).is_none_or(|end| end > tensor.shape[0])
    {
        return Err(Error::Invalid("Edge0 alias row range"));
    }
    let row_bytes = encoded_size(tensor.kind, tensor.shape[1])?;
    let begin = start
        .checked_mul(row_bytes)
        .and_then(|n| n.checked_add(tensor.offset))
        .ok_or(Error::Invalid("Edge0 alias row overflow"))?;
    let size = rows.checked_mul(row_bytes).ok_or(Error::Invalid("Edge0 alias row size"))?;
    weights.files[tensor.file].bytes(begin, size)?;
    tensor.offset = begin;
    tensor.size = size;
    tensor.shape[0] = rows;
    if weights.tensors.insert(target, tensor).is_some() {
        return Err(Error::Invalid("Edge0 duplicate alias"));
    }
    Ok(())
}

fn concat_rows(
    weights: &mut Weights,
    target: String,
    first: &str,
    second: &str,
) -> Result<()> {
    let mut tensor = weights.tensor(first)?.clone();
    let right = weights.tensor(second)?;
    if tensor.shape.len() != 2
        || right.shape.len() != 2
        || tensor.file != right.file
        || tensor.kind != right.kind
        || tensor.shape[1] != right.shape[1]
        || tensor.offset.checked_add(tensor.size) != Some(right.offset)
        || tensor.affine.is_some()
        || right.affine.is_some()
    {
        return Err(Error::Invalid("Edge0 contiguous row concatenation"));
    }
    tensor.shape[0] =
        tensor.shape[0].checked_add(right.shape[0]).ok_or(Error::Invalid("Edge0 alias shape overflow"))?;
    tensor.size = tensor.size.checked_add(right.size).ok_or(Error::Invalid("Edge0 alias size overflow"))?;
    weights.files[tensor.file].bytes(tensor.offset, tensor.size)?;
    if weights.tensors.insert(target, tensor).is_some() {
        return Err(Error::Invalid("Edge0 duplicate alias"));
    }
    Ok(())
}

fn bind_aliases(weights: &mut Weights) -> Result<()> {
    alias_rows(weights, "model.embed_tokens.weight".to_owned(), "token_embd.weight", 0, 248320)?;
    alias_rows(weights, "lm_head.weight".to_owned(), "output.weight", 0, 248320)?;
    for (target, source) in [
        ("model.hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight"),
        ("model.hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_down.weight"),
        ("model.hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight"),
    ] {
        alias(weights, target.to_owned(), source, None)?;
    }
    for layer in 0..48 {
        let source = format!("blk.{layer}");
        let target = format!("model.layers.{layer}");
        for (dst, src) in [
            ("mlp.gate.weight", "ffn_gate_inp.weight"),
            ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
            ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
            ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
            ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
        ] {
            alias(weights, format!("{target}.{dst}"), &format!("{source}.{src}"), None)?;
        }
        for (dst, src) in [("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")] {
            for (dst_role, src_role) in [
                ("hc_norm", "norm"),
                ("input_mix_weight_down", "down"),
                ("input_mix_weight_up", "up"),
                ("block_inject_weight", "inject"),
            ] {
                alias(
                    weights,
                    format!("{target}.{dst}.{dst_role}.weight"),
                    &format!("{source}.{src}_{src_role}.weight"),
                    None,
                )?;
            }
        }
        if layer % 4 == 3 {
            for (dst, src) in [
                ("q_proj", "attn_q"),
                ("k_proj", "attn_k"),
                ("v_proj", "attn_v"),
                ("o_proj", "attn_output"),
                ("q_norm", "attn_q_norm"),
                ("k_norm", "attn_k_norm"),
                ("indexer.q_layernorm", "indexer.q_norm"),
                ("indexer.k_layernorm", "indexer.k_norm"),
            ] {
                alias(weights, format!("{target}.self_attn.{dst}.weight"), &format!("{source}.{src}.weight"), None)?;
            }
            concat_rows(
                weights,
                format!("{target}.self_attn.indexer.index_qk_proj.weight"),
                &format!("{source}.indexer.q_proj.weight"),
                &format!("{source}.indexer.k_proj.weight"),
            )?;
        } else {
            for (dst, src) in [
                ("in_proj_qkv.weight", "attn_qkv.weight"),
                ("in_proj_z.weight", "attn_gate.weight"),
                ("in_proj_a.weight", "ssm_alpha.weight"),
                ("in_proj_b.weight", "ssm_beta.weight"),
                ("out_proj.weight", "ssm_out.weight"),
                ("A_log", "ssm_a"),
                ("dt_bias", "ssm_dt.bias"),
                ("norm.weight", "ssm_norm.weight"),
            ] {
                alias(weights, format!("{target}.linear_attn.{dst}"), &format!("{source}.{src}"), None)?;
            }
            alias(
                weights,
                format!("{target}.linear_attn.conv1d.weight"),
                &format!("{source}.ssm_conv1d.weight"),
                Some(&[10240, 4, 1]),
            )?;
        }
        if layer == 1 {
            for (dst, src) in [
                ("key_proj", "ple_key"),
                ("value_proj", "ple_value"),
                ("norm_key", "ple_norm_key"),
                ("norm_query", "ple_norm_query"),
                ("norm_conv", "ple_norm_conv"),
            ] {
                alias(weights, format!("{target}.ple.{dst}.weight"), &format!("{source}.{src}.weight"), None)?;
            }
            alias(
                weights,
                format!("{target}.ple.conv1d.weight"),
                &format!("{source}.ple_conv1d.weight"),
                Some(&[10240, 4, 1]),
            )?;
        }
    }
    Ok(())
}

fn table_header(
    file: &File,
    length: usize,
) -> Result<usize> {
    let mut reader = HeaderReader {
        reader: BufReader::new(file.try_clone()?),
        consumed: 0,
    };
    if &reader.bytes::<4>()? != b"GGUF" || reader.u32()? != 3 || reader.u64()? != 1 {
        return Err(Error::Invalid("Edge0 FP8 table header"));
    }
    let count = reader.count()?;
    if count != 5 {
        return Err(Error::Invalid("Edge0 FP8 table metadata count"));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        let value = reader.value(kind)?;
        if metadata.insert(key, value).is_some() {
            return Err(Error::Invalid("Edge0 FP8 duplicate metadata"));
        }
    }
    let expected = serde_json::json!({"general.architecture":"strata-ple","general.name":"PLE n-gram table, FP8 E4M3 as shipped","strata.ple.format":"f8_e4m3","strata.ple.scale":0.00019931793212890625_f64,"strata.ple.source":"nvfp4-src"});
    if serde_json::to_value(metadata)? != expected
        || reader.string()? != "per_layer_token_embd.weight"
        || reader.u32()? != 2
        || reader.u64()? != 160
        || reader.u64()? != TABLE_ROWS as u64
        || reader.u32()? != 24
        || reader.u64()? != 0
    {
        return Err(Error::Invalid("Edge0 FP8 table storage or geometry"));
    }
    let header = usize::try_from(reader.reader.stream_position()?).map_err(|_| Error::Invalid("Edge0 table offset"))?;
    let start = header.checked_add(31).map(|n| n & !31).ok_or(Error::Invalid("Edge0 table offset"))?;
    if start.checked_add(TABLE_ROWS * TABLE_WIDTH) != Some(length) {
        return Err(Error::Invalid("Edge0 FP8 table size"));
    }
    Ok(start)
}

impl Weights {
    /// Explicit reference adapter. This does not mutate preparation metadata,
    /// requantize payloads, load a model, or discover arbitrary GGUF folders.
    pub fn open_edge0(path: &Path) -> Result<Self> {
        prepared_receipt(path)?;
        let skeleton = source_file(path, "model-skeleton.gguf")?;
        if skeleton.metadata()?.len() != 5_991_569_248 {
            return Err(Error::Invalid("Edge0 skeleton size"));
        }
        let mut weights = Self::open(&path.join("model-skeleton.gguf"))?;
        validate_skeleton(&weights)?;
        let dense_file = map_file(&mut weights, path, "dense.bin", 1_538_625_024)?;
        let pool = edge0_source::read_pack_index(source_file(path, "index.txt")?, 1_538_625_024)?;
        bind_pool(&mut weights, pool, dense_file)?;
        let expert_file = map_file(&mut weights, path, "experts.bin", 40_076_574_720)?;
        let experts = edge0_source::read_native_experts(source_file(path, "native_experts.txt")?, 40_076_574_720)?;
        let projection_file = map_file(&mut weights, path, "latent_dense_q8.bin", 3_743_416_320)?;
        let projections = edge0_source::Q8Projections::new(3_743_416_320)?;
        let table_file = map_file(&mut weights, path, "ple-fp8.gguf", 51_200_246_112)?;
        let table_offset = table_header(&weights.files[table_file]._file, weights.files[table_file].length)?;
        let labels = edge0_source::ProjectionLabels::read(source_file(path, "labels.bin")?)?;
        let scales = (0..48)
            .map(|layer| edge0_source::read_column_scales(source_file(path, &format!("scales/l{layer}.f32"))?))
            .collect::<Result<Vec<_>>>()?;
        let mtp_dense = map_file(&mut weights, path, "mtp/dense.bin", mtp_storage::DENSE_BYTES)?;
        let mtp_experts = map_file(&mut weights, path, "mtp/experts.bin", mtp_storage::EXPERT_BYTES)?;
        let mtp = mtp_storage::MtpStorage::new(
            Arc::clone(&weights.files[mtp_dense]),
            Arc::clone(&weights.files[mtp_experts]),
            source_file(path, "mtp/dense.txt")?,
        )?;
        let mut integers = HashMap::new();
        let mut integer_shapes = HashMap::new();
        for (suffix, key, count) in [
            ("layer_multipliers", "qwen4exp.ple.layer_multipliers", 3),
            ("ngram_heads_vocab_sizes", "qwen4exp.ple.head_vocab_sizes", 16),
            ("ngram_heads_offsets", "qwen4exp.ple.head_offsets", 16),
        ] {
            let values = weights
                .metadata
                .get(key)
                .and_then(Value::as_array)
                .ok_or(Error::Invalid("Edge0 PLE integer metadata"))?
                .iter()
                .map(|v| v.as_i64().ok_or(Error::Invalid("Edge0 PLE exact integer")))
                .collect::<Result<Vec<_>>>()?;
            if values.len() != count {
                return Err(Error::Invalid("Edge0 PLE integer shape"));
            }
            let name = format!("model.layers.1.ple.ple_embedding.{suffix}");
            integer_shapes.insert(name.clone(), vec![count]);
            integers.insert(name, values);
        }
        bind_aliases(&mut weights)?;
        weights.edge0 = Some(Edge0Data {
            expert_file,
            projection_file,
            table_file,
            table_offset,
            table_shape: vec![TABLE_ROWS, TABLE_WIDTH],
            labels,
            scales,
            experts,
            projections,
            mtp,
            #[cfg(feature = "metal")]
            mtp_dense_file: mtp_dense,
            #[cfg(feature = "metal")]
            mtp_expert_file: mtp_experts,
            integers,
            integer_shapes,
            metal_counters: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)),
            #[cfg(feature = "metal")]
            metal_admitted: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "metal")]
            metal_dynamic_allowance: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "metal")]
            metal_resident: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "metal")]
            mtp_admitted: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        });
        #[cfg(feature = "metal")]
        {
            weights.indexed_bank_geometry_admitted = false;
        }
        Ok(weights)
    }

    pub fn is_edge0(&self) -> bool {
        self.edge0.is_some()
    }

    /// Staged storage view only. Does not enable draft execution/capabilities.
    pub fn edge0_mtp_dense(
        &self,
        name: &str,
    ) -> Result<Edge0MtpDenseView> {
        self.edge0.as_ref().ok_or(Error::Invalid("Edge0 MTP storage unavailable"))?.mtp.dense(name)
    }

    /// Original four-plane Q2_0 expert, retaining the mapped file owner.
    pub fn edge0_mtp_expert(
        &self,
        expert: usize,
    ) -> Result<Edge0MtpExpertView> {
        self.edge0.as_ref().ok_or(Error::Invalid("Edge0 MTP storage unavailable"))?.mtp.expert(expert)
    }

    /// Aggregate diagnostics only; these counters never retain route IDs,
    /// tokens, activations or numerical readouts.
    pub fn edge0_metal_metadata(&self) -> Value {
        let counters = self
            .edge0
            .as_ref()
            .map(|data| data.metal_counters.each_ref().map(|value| value.load(std::sync::atomic::Ordering::Relaxed)))
            .unwrap_or([0; 4]);
        #[cfg(feature = "metal")]
        let resident =
            self.encoder_residency.lock().is_ok_and(|guard| guard.as_ref().is_some_and(|state| !state.failed));
        #[cfg(not(feature = "metal"))]
        let resident = false;
        #[cfg(feature = "metal")]
        let virtual_bytes = self.encoder_bank_views.lock().map(|cache| cache.virtual_bytes).unwrap_or(0);
        #[cfg(not(feature = "metal"))]
        let virtual_bytes = 0;
        serde_json::json!({"resident_enabled":resident,"registered_bank_bytes":virtual_bytes,
            "graph_scratch_high_water_bytes":counters[3],
            "counters":{"batched_matrix_calls":counters[0],"batched_q8_calls":counters[1],"batched_expert_calls":counters[2]}})
    }

    #[cfg(feature = "metal")]
    pub(in crate::engine::hybrid) fn edge0_record_graph_scratch(
        &self,
        bytes: usize,
    ) {
        if let Some(data) = &self.edge0 {
            data.metal_counters[3].fetch_max(bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Charge future retained GPU weight resources in either registration
    /// policy. Registration is an allocation allowance, not a measurement of
    /// physical residency; the Qwen owner separately accounts dynamic state.
    pub(in crate::engine::hybrid) fn edge0_additional_weight_bytes(&self) -> Option<usize> {
        #[cfg(feature = "metal")]
        {
            let Some(data) = self.edge0.as_ref() else {
                return Some(0);
            };
            // Constructor admission asks for dynamic state before registering
            // any bank policy. Do not feed weights back into that allowance.
            if !data.metal_admitted.load(Ordering::Acquire) {
                return Some(0);
            }
            let registered = {
                let guard = self.encoder_residency.lock().ok()?;
                match guard.as_ref() {
                    Some(state) if state.failed => return None,
                    Some(state) => Some((state.bank_bytes, state.static_bytes)),
                    None => None,
                }
            };
            // Never acquire a cache lock while holding residency: dispatch
            // registration uses the opposite (cache -> residency) order.
            let (banks, statics) = if let Some(registered) = registered {
                registered
            } else {
                let banks = self.encoder_bank_views.lock().ok()?.virtual_bytes;
                let statics = self.static_views.lock().ok()?.registered_bytes;
                (banks, statics)
            };
            remaining_edge0_weight_bytes(banks, statics)
        }
        #[cfg(not(feature = "metal"))]
        Some(0)
    }

    pub fn edge0_group(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<usize> {
        Ok(usize::from(
            self.edge0.as_ref().ok_or(Error::Invalid("Edge0 layout required"))?.labels.group(layer, expert)?,
        ))
    }

    /// Return stored `s`, not `1/s`; the forward applies it exactly once.
    pub fn edge0_column_scales(
        &self,
        layer: usize,
    ) -> Result<&[f32]> {
        self.edge0
            .as_ref()
            .ok_or(Error::Invalid("Edge0 layout required"))?
            .scales
            .get(layer)
            .map(|values| values.as_ref())
            .ok_or(Error::Invalid("Edge0 scale layer"))
    }

    pub fn edge0_matvec(
        &self,
        matrix: Edge0Matrix,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let tensor = self.edge0.as_ref().ok_or(Error::Invalid("Edge0 layout required"))?.matrix(matrix)?;
        if input.len() != tensor.shape[1] || input.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("Edge0 matrix input"));
        }
        #[cfg(feature = "metal")]
        {
            let codes = self.files[tensor.file].bounded_metal_slice(tensor.offset, tensor.size, MAX_SELECTED_BYTES)?;
            let operation = super::super::weight_metal::MappedOperation {
                matrices: vec![super::super::weight_metal::MappedMatrix {
                    codes,
                    planes: None,
                }],
                inputs: input,
                rows: tensor.shape[0],
                columns: tensor.shape[1],
                rows_per_batch: 0,
                format: super::super::weight_metal::MatrixFormat::Packed(tensor.kind),
            };
            super::super::weight_metal::mapped_many(&[operation]).map(|mut output| output.remove(0))
        }
        #[cfg(not(feature = "metal"))]
        {
            self.matrix_slice(&tensor, 0, tensor.shape[0], tensor.shape[1], input)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use stark_engine_macros::stark_test;

    use super::*;

    #[cfg(feature = "metal")]
    #[stark_test]
    fn edge0_retained_resource_allowance_counts_cold_partial_and_complete_banks() {
        assert_eq!(remaining_edge0_weight_bytes(0, 0), Some(52_409_925_632));
        assert_eq!(remaining_edge0_weight_bytes(20_000_000_000, 1_000_000_000), Some(31_409_925_632));
        assert_eq!(remaining_edge0_weight_bytes(43_819_991_040, MAX_STATIC_VIEW_BYTES), Some(0));
        assert_eq!(remaining_edge0_weight_bytes(44_000_000_000, MAX_STATIC_VIEW_BYTES + 1), Some(0));
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static SEQUENCE: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "stark-edge0-weights-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn weights(&self) -> Weights {
            fn string(
                output: &mut Vec<u8>,
                value: &str,
            ) {
                output.extend((value.len() as u64).to_le_bytes());
                output.extend(value.as_bytes());
            }
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3_u32.to_le_bytes());
            bytes.extend(2_u64.to_le_bytes());
            bytes.extend(0_u64.to_le_bytes());
            for (name, offset) in [("left", 0_u64), ("right", 32)] {
                string(&mut bytes, name);
                bytes.extend(2_u32.to_le_bytes());
                bytes.extend(4_u64.to_le_bytes());
                bytes.extend(4_u64.to_le_bytes());
                bytes.extend(30_u32.to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            bytes.resize(bytes.len().next_multiple_of(32), 0);
            for value in 1..=32 {
                bytes.extend(bf16::from_f32(value as f32).to_bits().to_le_bytes());
            }
            let path = self.0.join("fixture.gguf");
            fs::write(&path, bytes).unwrap();
            Weights::open(&path).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[stark_test]
    fn edge0_alias_rows_reshape_and_concat_retain_one_source_mapping() {
        let fixture = Fixture::new();
        let mut weights = fixture.weights();
        alias_rows(&mut weights, "middle".to_owned(), "left", 1, 2).unwrap();
        assert_eq!(weights.shape("middle").unwrap(), [2, 4]);
        assert_eq!(weights.row("middle", 0).unwrap(), [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(weights.row("middle", 1).unwrap(), [9.0, 10.0, 11.0, 12.0]);
        alias(&mut weights, "reshaped".to_owned(), "left", Some(&[2, 8])).unwrap();
        assert_eq!(weights.vector("reshaped").unwrap(), (1..=16).map(|v| v as f32).collect::<Vec<_>>());
        concat_rows(&mut weights, "combined".to_owned(), "left", "right").unwrap();
        assert_eq!(weights.shape("combined").unwrap(), [8, 4]);
        assert_eq!(weights.row("combined", 4).unwrap(), [17.0, 18.0, 19.0, 20.0]);
        assert_eq!(weights.files.len(), 1);
        assert!(alias_rows(&mut weights, "bad".to_owned(), "left", 3, 2).is_err());
        assert!(alias_rows(&mut weights, "bad".to_owned(), "left", usize::MAX, 2).is_err());
        assert!(alias(&mut weights, "bad".to_owned(), "left", Some(&[3, 8])).is_err());
        assert!(concat_rows(&mut weights, "bad".to_owned(), "right", "left").is_err());
    }

    #[stark_test]
    fn edge0_complete_skeleton_shape_format_and_name_inventory_is_exact() {
        let fixture = Fixture::new();
        let mut weights = fixture.weights();
        weights.metadata = serde_json::from_str(include_str!("edge0_geometry.json")).unwrap();
        weights.metadata.insert("general.architecture".to_owned(), Value::String("qwen4exp".to_owned()));
        weights.tensors.clear();
        for (name, (kind, shape)) in expected_skeleton() {
            let size = encoded_size(kind, product(&shape).unwrap()).unwrap();
            weights.tensors.insert(
                name,
                Tensor {
                    file: 0,
                    offset: 0,
                    size,
                    shape,
                    kind,
                    affine: None,
                },
            );
        }
        validate_skeleton(&weights).unwrap();
        let original = weights.tensors["blk.0.ssm_a"].clone();
        weights.tensors.get_mut("blk.0.ssm_a").unwrap().kind = 30;
        assert!(validate_skeleton(&weights).is_err());
        weights.tensors.insert("blk.0.ssm_a".to_owned(), original.clone());
        weights.tensors.get_mut("blk.0.ssm_a").unwrap().shape = vec![47];
        assert!(validate_skeleton(&weights).is_err());
        weights.tensors.insert("blk.0.ssm_a".to_owned(), original.clone());
        weights.tensors.insert("unexpected".to_owned(), original);
        assert!(validate_skeleton(&weights).is_err());
    }

    #[stark_test]
    fn edge0_wrong_preparation_receipt_fails_before_payload_open() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("native-source.json"), b"{}").unwrap();
        assert!(matches!(Weights::open_edge0(&fixture.0), Err(Error::Invalid("Edge0 preparation receipt identity"))));
        fs::write(fixture.0.join("native-source.json"), vec![0; 1024 * 1024 + 1]).unwrap();
        assert!(matches!(Weights::open_edge0(&fixture.0), Err(Error::Invalid("Edge0 preparation receipt size"))));
    }

    #[stark_test]
    fn edge0_small_receipt_bindings_reject_drift_links_and_unsafe_paths() {
        let fixture = Fixture::new();
        let mut files = serde_json::Map::new();
        for index in 0..72 {
            let name = format!("metadata-{index:02}.json");
            fs::write(fixture.0.join(&name), b"{}").unwrap();
            files.insert(name, serde_json::json!({"size":2,"sha256":format!("{:x}",Sha256::digest(b"{}"))}));
        }
        let receipt = serde_json::json!({"files":files});
        verify_small_receipt_files(&fixture.0, &receipt).unwrap();
        fs::write(fixture.0.join("metadata-00.json"), b"[]").unwrap();
        assert!(verify_small_receipt_files(&fixture.0, &receipt).is_err());
        fs::remove_file(fixture.0.join("metadata-00.json")).unwrap();
        std::os::unix::fs::symlink("metadata-01.json", fixture.0.join("metadata-00.json")).unwrap();
        assert!(verify_small_receipt_files(&fixture.0, &receipt).is_err());
        assert!(source_file(&fixture.0, "../metadata-00.json").is_err());
        assert!(source_file(&fixture.0, "/metadata-00.json").is_err());
        let mut wrong = receipt.clone();
        wrong["files"].as_object_mut().unwrap().remove("metadata-00.json");
        assert!(verify_small_receipt_files(&fixture.0, &wrong).is_err());
    }

    /// Opens lazy file mappings and small metadata only; no model construction,
    /// forward, payload scan, dequantization or full content hashing occurs.
    #[stark_test]
    #[ignore]
    fn edge0_actual_adapter_metadata_aliases_and_virtual_ple_are_admitted() {
        let path = PathBuf::from(std::env::var_os("STARK_EDGE0_SOURCE_PATH").expect("source fixture required"));
        let weights = Weights::open_edge0(&path).unwrap();
        assert!(weights.is_edge0());
        assert_eq!(weights.files.len(), 7);
        let draft = weights.edge0_mtp_dense("self_attn.q_proj.weight").unwrap();
        assert_eq!(draft.shape(), [12288, 2560]);
        assert_eq!(draft.encoding(), Edge0MtpEncoding::Q8_0);
        assert_eq!(draft.offset(), 81_231_872);
        let final_draft_expert = weights.edge0_mtp_expert(511).unwrap();
        assert_eq!(final_draft_expert.offset() + final_draft_expert.record_bytes(), mtp_storage::EXPERT_BYTES as usize);
        assert!(weights.edge0_mtp_expert(512).is_err());
        assert!(weights.edge0_mtp_dense("unexpected.weight").is_err());
        assert_eq!(weights.shape("model.embed_tokens.weight").unwrap(), [248320, 2560]);
        assert_eq!(weights.shape("model.layers.3.self_attn.indexer.index_qk_proj.weight").unwrap(), [640, 2560]);
        assert_eq!(weights.shape("model.layers.0.linear_attn.conv1d.weight").unwrap(), [10240, 4, 1]);
        assert_eq!(weights.shape(TABLE_NAME).unwrap(), [TABLE_ROWS, TABLE_WIDTH]);
        assert!(weights.row(TABLE_NAME, LOGICAL_ROWS).is_err());
        assert_eq!(
            weights.integers("model.layers.1.ple.ple_embedding.layer_multipliers").unwrap(),
            [23703573157769, 20109073645365, 8052911324071]
        );
        for layer in 0..48 {
            assert_eq!(weights.edge0_column_scales(layer).unwrap().len(), 1792);
            assert!(weights.edge0_group(layer, 511).unwrap() < 8);
        }
        let data = weights.edge0.as_ref().unwrap();
        assert_eq!(
            data.matrix(Edge0Matrix::ExpertDown {
                layer: 47,
                expert: 511
            })
            .unwrap()
            .offset
                + 645120,
            weights.files[data.expert_file].length
        );
        assert_eq!(
            data.matrix(Edge0Matrix::ProjectionUp {
                layer: 47,
                group: 7
            })
            .unwrap()
            .offset
                + 4_874_240,
            weights.files[data.projection_file].length
        );
        assert!(
            data.matrix(Edge0Matrix::ExpertGate {
                layer: 48,
                expert: 0
            })
            .is_err()
        );
        assert!(
            data.matrix(Edge0Matrix::ProjectionDown {
                layer: 0,
                group: 8
            })
            .is_err()
        );
    }
}
