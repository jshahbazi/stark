// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Local safetensors checkpoint reading: index and shard headers plus byte-range reads.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;

pub mod uzu;

/// The index file name a checkpoint directory is expected to contain.
const INDEX_NAME: &str = "model.safetensors.index.json";
/// The single-file checkpoint name used when no index is present.
const SINGLE_FILE_NAME: &str = "model.safetensors";

/// Safetensors dtypes lily can load. Anything else stays `Other` so unrelated
/// tensors (e.g. vision tower) never fail `open`; reading one is an error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SafetensorsDType {
    BF16,
    F32,
    /// Packed quantized weight codes in mlx checkpoints.
    U32,
    Other(String),
}

impl SafetensorsDType {
    fn from_str(s: &str) -> Self {
        match s {
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "U32" => Self::U32,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn size(&self) -> Option<usize> {
        match self {
            Self::BF16 => Some(2),
            Self::F32 | Self::U32 => Some(4),
            Self::Other(name) if name == "I64" || name == "U64" => Some(8),
            Self::Other(_) => None,
        }
    }
}

/// Where one tensor lives: its shard file plus absolute byte offsets.
#[derive(Clone)]
pub struct TensorMeta {
    pub dtype: SafetensorsDType,
    pub shape: Vec<usize>,
    shard: PathBuf,
    start: u64,
    end: u64,
    // Converted group-major affine parameters are restored bit-for-bit at
    // load. Native checkpoint tensors and paged embeddings remain direct.
    group_output: Option<(usize, usize, usize)>,
}

impl TensorMeta {
    pub fn byte_len(&self) -> usize {
        self.group_output
            .map_or((self.end - self.start) as usize, |(rows, groups, _)| {
                rows * groups * 2
            })
    }

    /// The shard file holding this tensor.
    pub fn shard(&self) -> &Path {
        &self.shard
    }

    /// Absolute byte offset of the tensor data within its shard file.
    pub fn start(&self) -> u64 {
        self.start
    }
}

/// A single tensor's entry in a safetensors JSON header.
#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Deserialize)]
struct IndexFile {
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

/// A parsed checkpoint directory: every tensor name mapped to its location.
pub struct Checkpoint {
    tensors: HashMap<String, TensorMeta>,
}

impl Checkpoint {
    /// Opens the checkpoint under `dir`, via `model.safetensors.index.json`
    /// when present or a single `model.safetensors` otherwise.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join(INDEX_NAME);
        let shards: Vec<PathBuf> = if index_path.exists() {
            let bytes = std::fs::read(&index_path)
                .with_context(|| format!("reading {}", index_path.display()))?;
            let index: IndexFile =
                serde_json::from_slice(&bytes).context("parsing safetensors index")?;
            let mut names: Vec<&String> = index.weight_map.values().collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|n| dir.join(n)).collect()
        } else {
            let single = dir.join(SINGLE_FILE_NAME);
            ensure!(
                single.exists(),
                "no {INDEX_NAME} or {SINGLE_FILE_NAME} under {}",
                dir.display()
            );
            vec![single]
        };

        let mut tensors = HashMap::new();
        for shard in shards {
            for (name, meta) in parse_shard_header(&shard)? {
                if tensors.insert(name.clone(), meta).is_some() {
                    bail!("tensor {name} appears in multiple shards");
                }
            }
        }
        Ok(Self { tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// Reads one tensor's raw bytes (its native dtype layout).
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.read_with(name, |meta| {
            bytes = vec![0u8; meta.byte_len()];
            Ok(&mut bytes)
        })?;
        Ok(bytes)
    }

    /// Reads one tensor's raw bytes into a caller-provided buffer, letting the
    /// caller size/place the destination (e.g. directly in an MTLBuffer).
    pub fn read_with<'a>(
        &self,
        name: &str,
        dest: impl FnOnce(&TensorMeta) -> Result<&'a mut [u8]>,
    ) -> Result<()> {
        let meta = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))?;
        ensure!(
            meta.dtype.size().is_some(),
            "tensor {name} has unsupported dtype {:?}",
            meta.dtype
        );
        let buf = dest(meta)?;
        ensure!(
            buf.len() == meta.byte_len(),
            "destination for {name} is {} bytes, tensor is {}",
            buf.len(),
            meta.byte_len()
        );
        let mut file = File::open(&meta.shard)
            .with_context(|| format!("opening {}", meta.shard.display()))?;
        // Weights read once into GPU buffers must not evict the page cache:
        // the paged n-gram table lives there and would go cold on every load.
        set_nocache(&file);
        file.seek(SeekFrom::Start(meta.start))
            .with_context(|| format!("seeking to {name}"))?;
        if let Some((rows, groups, stride)) = meta.group_output {
            let mut physical = vec![0u8; (meta.end - meta.start) as usize];
            file.read_exact(&mut physical)
                .with_context(|| format!("reading {name}"))?;
            restore_group_output(&physical, buf, rows, groups, stride)?;
        } else {
            file.read_exact(buf).with_context(|| format!("reading {name}"))?;
        }
        Ok(())
    }
}

/// Restore canonical group-major affine parameters without interpreting their
/// BF16 bits. NEON transposes 8x8 word tiles; byte tails/fallback retain all bits.
fn restore_group_output(
    physical: &[u8],
    output: &mut [u8],
    rows: usize,
    groups: usize,
    stride: usize,
) -> Result<()> {
    for group in 0..groups {
        let padding =
            &physical[2 * (group * stride + rows)..2 * ((group + 1) * stride)];
        if padding.iter().any(|&byte| byte != 0) {
            // Preserve original failure-side writes: only complete earlier
            // groups were copied before this invalid padding was inspected.
            for previous in 0..group {
                for row in 0..rows {
                    let from = 2 * (previous * stride + row);
                    let to = 2 * (row * groups + previous);
                    output[to..to + 2].copy_from_slice(&physical[from..from + 2]);
                }
            }
            anyhow::bail!("nonzero Uzu affine padding");
        }
    }
    #[cfg(target_arch = "aarch64")]
    if physical.as_ptr().align_offset(std::mem::align_of::<u16>()) == 0
        && output.as_ptr().align_offset(std::mem::align_of::<u16>()) == 0
        && std::arch::is_aarch64_feature_detected!("neon")
    {
        let row_blocks = rows / 8 * 8;
        let group_blocks = groups / 8 * 8;
        for start in (0..rows).step_by(64) {
            let end = rows.min(start + 64);
            let vector_end = end.min(row_blocks);
            for group in (0..group_blocks).step_by(8) {
                for row in (start..vector_end).step_by(8) {
                    // Both strided 8x8 spans are checked before any vector
                    // access; each load/store is exactly eight u16 values.
                    ensure!(
                        transpose_u16_tile_bounds(
                            physical.len(),
                            output.len(),
                            row,
                            group,
                            groups,
                            stride
                        ),
                        "invalid NEON transpose tile bounds"
                    );
                    let source = 2 * (group * stride + row);
                    let destination = 2 * (row * groups + group);
                    // SAFETY: alignment/CPU feature checked above; the helper
                    // bounds include the final row/group and all eight lanes.
                    unsafe {
                        transpose_u16_8x8_neon(
                            physical.as_ptr().add(source).cast(),
                            output.as_mut_ptr().add(destination).cast(),
                            stride,
                            groups,
                        );
                    }
                }
                for g in group..group + 8 {
                    for row in vector_end.max(start)..end {
                        let from = 2 * (g * stride + row);
                        let to = 2 * (row * groups + g);
                        output[to..to + 2].copy_from_slice(&physical[from..from + 2]);
                    }
                }
            }
            for group in group_blocks..groups {
                for row in start..end {
                    let from = 2 * (group * stride + row);
                    let to = 2 * (row * groups + group);
                    output[to..to + 2].copy_from_slice(&physical[from..from + 2]);
                }
            }
        }
        return Ok(());
    }
    restore_group_output_blocked(physical, output, rows, groups, stride);
    Ok(())
}

fn restore_group_output_blocked(
    physical: &[u8],
    output: &mut [u8],
    rows: usize,
    groups: usize,
    stride: usize,
) {
    for start in (0..rows).step_by(64) {
        let end = rows.min(start + 64);
        for group in 0..groups {
            for row in start..end {
                let from = 2 * (group * stride + row);
                let to = 2 * (row * groups + group);
                output[to..to + 2].copy_from_slice(&physical[from..from + 2]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn transpose_u16_tile_bounds(
    source_bytes: usize,
    dest_bytes: usize,
    row: usize,
    group: usize,
    groups: usize,
    stride: usize,
) -> bool {
    let Some(last_row) = row.checked_add(7) else { return false };
    let Some(last_group) = group.checked_add(7) else { return false };
    if last_row >= stride || last_group >= groups {
        return false;
    }
    let end = |major: usize, pitch: usize, minor: usize| {
        major
            .checked_mul(pitch)
            .and_then(|n| n.checked_add(minor))
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(2))
    };
    end(last_group, stride, last_row).is_some_and(|end| end <= source_bytes)
        && end(last_row, groups, last_group).is_some_and(|end| end <= dest_bytes)
}

/// Transpose eight contiguous source rows with independent source/dest
/// strides. Caller must validate both 8x8 spans and u16 pointer alignment.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transpose_u16_8x8_neon(
    source: *const u16,
    dest: *mut u16,
    source_stride: usize,
    dest_stride: usize,
) {
    use std::arch::aarch64::*;
    // SAFETY: caller proved each of the eight loads/stores is in bounds and
    // both bases have u16 alignment. NEON permits unaligned 16-byte vectors.
    unsafe {
        let a0 = vld1q_u16(source);
        let a1 = vld1q_u16(source.add(source_stride));
        let a2 = vld1q_u16(source.add(2 * source_stride));
        let a3 = vld1q_u16(source.add(3 * source_stride));
        let a4 = vld1q_u16(source.add(4 * source_stride));
        let a5 = vld1q_u16(source.add(5 * source_stride));
        let a6 = vld1q_u16(source.add(6 * source_stride));
        let a7 = vld1q_u16(source.add(7 * source_stride));
        let t0 = vreinterpretq_u32_u16(vtrn1q_u16(a0, a1));
        let t1 = vreinterpretq_u32_u16(vtrn2q_u16(a0, a1));
        let t2 = vreinterpretq_u32_u16(vtrn1q_u16(a2, a3));
        let t3 = vreinterpretq_u32_u16(vtrn2q_u16(a2, a3));
        let t4 = vreinterpretq_u32_u16(vtrn1q_u16(a4, a5));
        let t5 = vreinterpretq_u32_u16(vtrn2q_u16(a4, a5));
        let t6 = vreinterpretq_u32_u16(vtrn1q_u16(a6, a7));
        let t7 = vreinterpretq_u32_u16(vtrn2q_u16(a6, a7));
        let b0 = vreinterpretq_u64_u32(vtrn1q_u32(t0, t2));
        let b1 = vreinterpretq_u64_u32(vtrn1q_u32(t1, t3));
        let b2 = vreinterpretq_u64_u32(vtrn2q_u32(t0, t2));
        let b3 = vreinterpretq_u64_u32(vtrn2q_u32(t1, t3));
        let b4 = vreinterpretq_u64_u32(vtrn1q_u32(t4, t6));
        let b5 = vreinterpretq_u64_u32(vtrn1q_u32(t5, t7));
        let b6 = vreinterpretq_u64_u32(vtrn2q_u32(t4, t6));
        let b7 = vreinterpretq_u64_u32(vtrn2q_u32(t5, t7));
        vst1q_u16(dest, vreinterpretq_u16_u64(vtrn1q_u64(b0, b4)));
        vst1q_u16(dest.add(dest_stride), vreinterpretq_u16_u64(vtrn1q_u64(b1, b5)));
        vst1q_u16(dest.add(2 * dest_stride), vreinterpretq_u16_u64(vtrn1q_u64(b2, b6)));
        vst1q_u16(dest.add(3 * dest_stride), vreinterpretq_u16_u64(vtrn1q_u64(b3, b7)));
        vst1q_u16(dest.add(4 * dest_stride), vreinterpretq_u16_u64(vtrn2q_u64(b0, b4)));
        vst1q_u16(dest.add(5 * dest_stride), vreinterpretq_u16_u64(vtrn2q_u64(b1, b5)));
        vst1q_u16(dest.add(6 * dest_stride), vreinterpretq_u16_u64(vtrn2q_u64(b2, b6)));
        vst1q_u16(dest.add(7 * dest_stride), vreinterpretq_u16_u64(vtrn2q_u64(b3, b7)));
    }
}

#[cfg(test)]
mod transpose_tests;

/// Asks the kernel not to keep this file's pages in the unified buffer cache
/// (`F_NOCACHE`). Best effort; reads still work if it fails.
#[cfg(target_os = "macos")]
fn set_nocache(file: &File) {
    use std::os::unix::io::AsRawFd as _;
    unsafe extern "C" {
        fn fcntl(fd: std::ffi::c_int, cmd: std::ffi::c_int, ...) -> std::ffi::c_int;
    }
    const F_NOCACHE: std::ffi::c_int = 48;
    // SAFETY: plain fcntl on a valid descriptor; the return value is ignored.
    unsafe {
        fcntl(file.as_raw_fd(), F_NOCACHE, 1 as std::ffi::c_int);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_nocache(_file: &File) {}

/// Parses one shard's header: 8-byte little-endian header length, then the
/// JSON header; `data_offsets` are relative to the data section that follows.
fn parse_shard_header(path: &Path) -> Result<HashMap<String, TensorMeta>> {
    let mut file =
        File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("reading header length of {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("reading header of {}", path.display()))?;
    let data_section = 8 + header_len;

    // `__metadata__` is an optional free-form object, not a tensor entry.
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).context("parsing safetensors header")?;
    let mut out = HashMap::with_capacity(raw.len());
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: HeaderEntry = serde_json::from_value(value)
            .with_context(|| format!("parsing header entry for {name}"))?;
        let [start, end] = entry.data_offsets;
        ensure!(end >= start, "tensor {name} has reversed data offsets");
        out.insert(
            name,
            TensorMeta {
                dtype: SafetensorsDType::from_str(&entry.dtype),
                shape: entry.shape,
                shard: path.to_path_buf(),
                start: data_section + start,
                end: data_section + end,
                group_output: None,
            },
        );
    }
    Ok(out)
}
