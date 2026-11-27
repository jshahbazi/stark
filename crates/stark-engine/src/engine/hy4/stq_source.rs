// SPDX-License-Identifier: Apache-2.0
//! Read-only source views for the pinned Sushi Hy4 STQ pack.

use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    fs::{self, File, Metadata, OpenOptions},
    io,
    ops::Range,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    ptr::NonNull,
};

use serde::{Deserialize, Deserializer, de};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const REPOSITORY: &str = "beamster/Hy4-Flash-Sushi-STQ-4bpw";
pub const REVISION: &str = "61528ba635a623f127c5a7b1e1f4ac2c2b0d4885";
pub const PACK_REVISION: &str = "capture-2a0815359e97";

const CONFIG: &str = "config.json";
const CONFIG_SHA256: &str = "c1c5af8449a8e7db0dc5c5e8aa768c80673c94604997cfe13c6ee6aa98e9a3e3";
const INDEX: &str = "model.safetensors.index.json";
const INDEX_SHA256: &str = "6a3d938ac63427349ff8e647edd894cdfa00d3075c45cda4d1e7b662a99359ca";
// Length prefixes and JSON headers, concatenated in filename order. The full
// payload verification of the original download is separate evidence.
const HEADERS_SHA256: &str = "7c1f7ce2a958c153daea4372915616b58454da8dc36f014c6725fdd7fdf11c76";
const SOURCE_BYTES: u64 = 208_397_806_219;
const TENSOR_BYTES: u64 = 208_387_139_446;
const TENSOR_COUNT: usize = 3179;
const MAX_HEADER_BYTES: usize = 64 * 1024;

const SIDECARS: [(&str, u64); 10] = [
    (".gitattributes", 1519),
    ("LICENSE", 13942),
    ("README.md", 2578),
    ("chat_template.jinja", 17250),
    (CONFIG, 11615),
    ("generation_config.json", 189),
    (INDEX, 254344),
    ("processor_config.json", 1634),
    ("tokenizer.json", 9359191),
    ("tokenizer_config.json", 1277),
];

const DFLY_SHARD: &str = "model-dfly.safetensors";
const GLOBAL_SHARD: &str = "model-globals.safetensors";

#[derive(Debug, Error)]
pub enum StqSourceError {
    #[error("invalid Hy4 STQ source: {0}")]
    Invalid(&'static str),
    #[error("invalid Hy4 STQ metadata JSON")]
    Json(#[from] serde_json::Error),
    #[error("cannot access Hy4 STQ source: {0}")]
    Io(#[from] io::Error),
}

type Result<T> = std::result::Result<T, StqSourceError>;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum StqDtype {
    BF16,
    F16,
    F32,
    U16,
    I32,
}

impl StqDtype {
    fn item_size(self) -> usize {
        match self {
            Self::BF16 | Self::F16 | Self::U16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }
}

pub struct StqTensorInfo {
    pub dtype: StqDtype,
    pub shape: Vec<u32>,
    shard: usize,
    bytes: Range<usize>,
}

impl StqTensorInfo {
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

pub struct StqTensorLocation<'a> {
    pub file: &'a File,
    pub shard_name: &'a str,
    pub offset: u64,
    pub byte_len: usize,
}

pub struct StqTensorView<'a> {
    pub dtype: StqDtype,
    pub shape: &'a [u32],
    pub bytes: &'a [u8],
}

#[derive(PartialEq, Eq)]
struct Fingerprint {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl From<&Metadata> for Fingerprint {
    fn from(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        }
    }
}

struct SourceFile {
    name: String,
    file: File,
    fingerprint: Fingerprint,
}

impl SourceFile {
    fn open(
        directory: &File,
        name: String,
    ) -> Result<Self> {
        if name.is_empty() || name == "." || name == ".." || name.contains('/') {
            return Err(StqSourceError::Invalid("invalid component name"));
        }
        let component = CString::new(name.as_bytes()).map_err(|_| StqSourceError::Invalid("invalid component name"))?;
        // SAFETY: the descriptor and the terminated filename stay live through
        // openat. No creation flags are passed. Nonblocking open rejects a
        // substituted FIFO at the descriptor metadata check below.
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                component.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: openat transferred ownership of a new file descriptor.
        let file = unsafe { File::from_raw_fd(fd) };
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(StqSourceError::Invalid("source components must be regular files"));
        }
        Ok(Self {
            name,
            file,
            fingerprint: Fingerprint::from(&metadata),
        })
    }

    fn read_pinned(
        &self,
        sha256: &str,
    ) -> Result<Vec<u8>> {
        // Only used after verifying a sidecar length against its pinned size.
        let mut bytes = vec![0; self.fingerprint.size as usize];
        self.file.read_exact_at(&mut bytes, 0)?;
        if format!("{:x}", Sha256::digest(&bytes)) != sha256 {
            return Err(StqSourceError::Invalid("sidecar SHA-256 mismatch"));
        }
        Ok(bytes)
    }

    fn verify_unchanged(
        &self,
        root: &Path,
    ) -> Result<()> {
        let descriptor = self.file.metadata()?;
        let named_path = fs::symlink_metadata(root.join(&self.name))?;
        if !descriptor.is_file()
            || !named_path.is_file()
            || Fingerprint::from(&descriptor) != self.fingerprint
            || Fingerprint::from(&named_path) != self.fingerprint
        {
            return Err(StqSourceError::Invalid("source component changed"));
        }
        Ok(())
    }
}

/// Header-validated pack with retained descriptors. This checks metadata
/// identity, tensor spans and ten alignment bytes. It never hashes weights.
pub struct StqSource {
    root: PathBuf,
    directory: File,
    fingerprint: Fingerprint,
    sidecars: Vec<SourceFile>,
    shards: Vec<SourceFile>,
    config: Vec<u8>,
    tensors: HashMap<String, StqTensorInfo>,
}

impl StqSource {
    pub fn open(root: &Path) -> Result<Self> {
        let root: PathBuf = std::path::absolute(root)?.components().collect();
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&root)?;
        let fingerprint = Fingerprint::from(&directory.metadata()?);
        let shard_names = shard_names();
        check_inventory(&root, &shard_names)?;

        let mut sidecars = Vec::with_capacity(SIDECARS.len());
        for (name, size) in SIDECARS {
            let sidecar = SourceFile::open(&directory, name.to_owned())?;
            if sidecar.fingerprint.size != size {
                return Err(StqSourceError::Invalid("sidecar size mismatch"));
            }
            sidecars.push(sidecar);
        }
        let read_sidecar = |name, hash| {
            sidecars
                .iter()
                .find(|file| file.name == name)
                .ok_or(StqSourceError::Invalid("missing sidecar"))?
                .read_pinned(hash)
        };
        let config = read_sidecar(CONFIG, CONFIG_SHA256)?;
        let mut index: Index = serde_json::from_slice(&read_sidecar(INDEX, INDEX_SHA256)?)?;
        if index.metadata.total_size != TENSOR_BYTES || index.weight_map.0.len() != TENSOR_COUNT {
            return Err(StqSourceError::Invalid("index inventory mismatch"));
        }

        let mut source = Self {
            root,
            directory,
            fingerprint,
            sidecars,
            shards: Vec::with_capacity(shard_names.len()),
            config,
            tensors: HashMap::with_capacity(TENSOR_COUNT),
        };
        let mut source_bytes_remaining = SOURCE_BYTES - SIDECARS.iter().map(|(_, size)| size).sum::<u64>();
        let mut tensor_bytes_remaining = TENSOR_BYTES;
        let mut headers = Sha256::new();
        for name in shard_names {
            let shard = SourceFile::open(&source.directory, name)?;
            source_bytes_remaining = source_bytes_remaining
                .checked_sub(shard.fingerprint.size)
                .ok_or(StqSourceError::Invalid("source exceeds pinned byte count"))?;
            let header = read_shard(&shard, &mut headers, source.shards.len())?;
            for (tensor_name, info) in header {
                if index.weight_map.0.remove(&tensor_name).as_deref() != Some(shard.name.as_str()) {
                    return Err(StqSourceError::Invalid("tensor missing from index or assigned to the wrong shard"));
                }
                tensor_bytes_remaining = tensor_bytes_remaining
                    .checked_sub(info.byte_len() as u64)
                    .ok_or(StqSourceError::Invalid("tensors exceed pinned byte count"))?;
                if source.tensors.insert(tensor_name, info).is_some() {
                    return Err(StqSourceError::Invalid("duplicate tensor across shards"));
                }
            }
            source.shards.push(shard);
        }
        if !index.weight_map.0.is_empty()
            || format!("{:x}", headers.finalize()) != HEADERS_SHA256
            || source_bytes_remaining != 0
            || tensor_bytes_remaining != 0
        {
            return Err(StqSourceError::Invalid("pinned shard inventory mismatch"));
        }
        source.verify_unchanged()?;
        Ok(source)
    }

    pub fn config_bytes(&self) -> &[u8] {
        &self.config
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor(
        &self,
        name: &str,
    ) -> Option<&StqTensorInfo> {
        self.tensors.get(name)
    }

    /// Access the already opened descriptor for a bounded positional read.
    pub fn tensor_location(
        &self,
        name: &str,
    ) -> Option<StqTensorLocation<'_>> {
        let info = self.tensor(name)?;
        let shard = &self.shards[info.shard];
        Some(StqTensorLocation {
            file: &shard.file,
            shard_name: &shard.name,
            offset: info.bytes.start as u64,
            byte_len: info.byte_len(),
        })
    }

    pub fn verify_unchanged(&self) -> Result<()> {
        let path = fs::symlink_metadata(&self.root)?;
        if !path.is_dir()
            || Fingerprint::from(&path) != self.fingerprint
            || Fingerprint::from(&self.directory.metadata()?) != self.fingerprint
        {
            return Err(StqSourceError::Invalid("source directory changed"));
        }
        for file in self.shards.iter().chain(&self.sidecars) {
            file.verify_unchanged(&self.root)?;
        }
        Ok(())
    }

    /// Maps the validated shards on demand without prefetching any weights.
    ///
    /// # Safety
    /// The caller must keep the opened files immutable until the returned
    /// mapping and all its byte views have been dropped. External writes or
    /// truncation violate the safety of slices backed by file mappings.
    pub unsafe fn map(&self) -> Result<StqMappedSource<'_>> {
        self.verify_unchanged()?;
        // SAFETY: the caller assumes responsibility for source immutability.
        let shards = self.shards.iter().map(|file| unsafe { Mapping::new(file) }).collect::<Result<_>>()?;
        self.verify_unchanged()?;
        Ok(StqMappedSource {
            source: self,
            shards,
        })
    }
}

fn shard_names() -> Vec<String> {
    let mut names: Vec<_> = (1..=4).map(|n| format!("engram-model-{n:05}-of-00148.safetensors")).collect();
    names.extend([DFLY_SHARD.to_owned(), GLOBAL_SHARD.to_owned()]);
    names.extend((1..42).map(|n| format!("model-layer-{n:02}.safetensors")));
    names.extend((0..42).map(|n| format!("model-trunk-{n:02}.safetensors")));
    names
}

fn check_inventory(
    root: &Path,
    shard_names: &[String],
) -> Result<()> {
    let mut expected: HashSet<_> = shard_names.iter().map(String::as_str).collect();
    expected.extend(SIDECARS.iter().map(|&(name, _)| name));
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        if file_type.is_dir() && !name.to_str().is_some_and(|name| expected.contains(name)) {
            continue;
        }
        if !file_type.is_file() || !name.to_str().is_some_and(|name| expected.remove(name)) {
            return Err(StqSourceError::Invalid("unexpected top-level source entry"));
        }
    }
    if !expected.is_empty() {
        return Err(StqSourceError::Invalid("missing source component"));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    metadata: IndexMetadata,
    weight_map: StringMap,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: u64,
}

struct StringMap(HashMap<String, String>);

impl<'de> Deserialize<'de> for StringMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Strings;

        impl<'de> de::Visitor<'de> for Strings {
            type Value = StringMap;

            fn expecting(
                &self,
                formatter: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                formatter.write_str("an object of unique string pairs")
            }

            fn visit_map<A: de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = HashMap::new();
                while let Some((key, value)) = map.next_entry()? {
                    if values.insert(key, value).is_some() {
                        return Err(de::Error::custom("duplicate string map entry"));
                    }
                }
                Ok(StringMap(values))
            }
        }

        deserializer.deserialize_map(Strings)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: StqDtype,
    shape: Vec<u32>,
    data_offsets: [usize; 2],
}

struct Header {
    metadata: Option<StringMap>,
    tensors: HashMap<String, RawTensor>,
}

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Entries;

        impl<'de> de::Visitor<'de> for Entries {
            type Value = Header;

            fn expecting(
                &self,
                formatter: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                formatter.write_str("a Safetensors header with unique tensor names")
            }

            fn visit_map<A: de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut header = Header {
                    metadata: None,
                    tensors: HashMap::new(),
                };
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if header.metadata.replace(map.next_value()?).is_some() {
                            return Err(de::Error::custom("duplicate shard metadata"));
                        }
                    } else if name.is_empty() || header.tensors.insert(name, map.next_value()?).is_some() {
                        return Err(de::Error::custom("empty or duplicate tensor name"));
                    }
                }
                Ok(header)
            }
        }

        deserializer.deserialize_map(Entries)
    }
}

fn read_shard(
    shard: &SourceFile,
    headers: &mut Sha256,
    shard_number: usize,
) -> Result<Vec<(String, StqTensorInfo)>> {
    let length = usize::try_from(shard.fingerprint.size)
        .ok()
        .filter(|&length| length <= isize::MAX as usize)
        .ok_or(StqSourceError::Invalid("shard extent exceeds addressable range"))?;
    let mut prefix = [0; size_of::<u64>()];
    shard.file.read_exact_at(&mut prefix, 0)?;
    let header_len = usize::try_from(u64::from_le_bytes(prefix))
        .ok()
        .filter(|&length| (2..=MAX_HEADER_BYTES).contains(&length))
        .ok_or(StqSourceError::Invalid("invalid shard header size"))?;
    let data_start = prefix.len() + header_len;
    let data_len = length.checked_sub(data_start).ok_or(StqSourceError::Invalid("truncated shard header"))?;
    let mut json = vec![0_u8; header_len];
    shard.file.read_exact_at(&mut json, prefix.len() as u64)?;
    if json.first() != Some(&b'{') {
        return Err(StqSourceError::Invalid("invalid shard header start"));
    }
    headers.update(prefix);
    headers.update(&json);
    let header: Header = serde_json::from_slice(&json)?;
    check_metadata(&shard.name, header.metadata.as_ref())?;

    let alignment_gap = match shard.name.as_str() {
        DFLY_SHARD => Some(404_029_762..404_029_768),
        GLOBAL_SHARD => Some(557_486_100..557_486_104),
        _ => None,
    };
    let tensors = check_spans(header.tensors, shard_number, data_start, data_len, alignment_gap.clone())?;
    if let Some(gap) = alignment_gap {
        let mut padding = [0_u8; 6];
        shard.file.read_exact_at(&mut padding[..gap.len()], (data_start + gap.start) as u64)?;
        if padding != [0; 6] {
            return Err(StqSourceError::Invalid("nonzero shard alignment padding"));
        }
    }
    Ok(tensors)
}

fn check_metadata(
    name: &str,
    metadata: Option<&StringMap>,
) -> Result<()> {
    let common = [("format", "stq"), ("pack_revision", PACK_REVISION)];
    let spec1_trunk = [("k", "6"), ("codebook", "spec1"), ("window", "15")];
    let spec1_dfly = [("k", "4"), ("codebook", "spec1"), ("window", "15")];
    let pair1 =
        [("k", "4"), ("codebook", "pair1"), ("window", "16"), ("quantizer", "ldlq"), ("tile_order", "pair_major")];
    if name.starts_with("engram-model-") {
        return if metadata.is_none_or(|metadata| metadata.0.is_empty()) {
            Ok(())
        } else {
            Err(StqSourceError::Invalid("unexpected Engram shard metadata"))
        };
    }
    let quant: &[(&str, &str)] = if name.starts_with("model-layer-") {
        &pair1
    } else if name == DFLY_SHARD {
        &spec1_dfly
    } else {
        &spec1_trunk
    };
    if metadata.is_none_or(|metadata| {
        metadata.0.len() != common.len() + quant.len()
            || common.iter().chain(quant).any(|&(key, value)| metadata.0.get(key).map(String::as_str) != Some(value))
    }) {
        return Err(StqSourceError::Invalid("shard quantization metadata mismatch"));
    }
    Ok(())
}

fn check_spans(
    tensors: HashMap<String, RawTensor>,
    shard: usize,
    data_start: usize,
    data_len: usize,
    mut alignment_gap: Option<Range<usize>>,
) -> Result<Vec<(String, StqTensorInfo)>> {
    let mut tensors: Vec<_> = tensors.into_iter().collect();
    tensors.sort_unstable_by_key(|(_, raw)| raw.data_offsets);
    let mut checked = Vec::with_capacity(tensors.len());
    let mut cursor = 0;
    for (name, raw) in tensors {
        let [start, end] = raw.data_offsets;
        let item_size = raw.dtype.item_size();
        if !(1..=5).contains(&raw.shape.len()) || raw.shape.contains(&0) {
            return Err(StqSourceError::Invalid("invalid tensor shape"));
        }
        let size = raw.shape.iter().try_fold(item_size, |size, &dimension| size.checked_mul(dimension as usize));
        if start >= end
            || end > data_len
            || !(data_start + start).is_multiple_of(item_size)
            || size != Some(end - start)
        {
            return Err(StqSourceError::Invalid("invalid tensor extent"));
        }
        if start != cursor && alignment_gap.take() != Some(cursor..start) {
            return Err(StqSourceError::Invalid("overlapping tensors or unsupported shard gap"));
        }
        cursor = end;
        checked.push((
            name,
            StqTensorInfo {
                dtype: raw.dtype,
                shape: raw.shape,
                shard,
                bytes: data_start + start..data_start + end,
            },
        ));
    }
    if cursor != data_len || alignment_gap.is_some() || checked.is_empty() {
        return Err(StqSourceError::Invalid("incomplete shard extent"));
    }
    Ok(checked)
}

pub struct StqMappedSource<'a> {
    source: &'a StqSource,
    shards: Vec<Mapping>,
}

impl StqMappedSource<'_> {
    pub fn tensor(
        &self,
        name: &str,
    ) -> Option<StqTensorView<'_>> {
        let info = self.source.tensor(name)?;
        Some(StqTensorView {
            dtype: info.dtype,
            shape: &info.shape,
            bytes: self.shards[info.shard].bytes(info.bytes.clone()),
        })
    }
}

struct Mapping {
    pointer: NonNull<u8>,
    length: usize,
}

impl Mapping {
    unsafe fn new(source: &SourceFile) -> Result<Self> {
        let length = usize::try_from(source.fingerprint.size)
            .ok()
            .filter(|&length| (1..=isize::MAX as usize).contains(&length))
            .ok_or(StqSourceError::Invalid("invalid mapped extent"))?;
        // SAFETY: map a bounded extent of the validated read-only descriptor.
        // Its owner remains live, and the caller ensures file immutability.
        let raw = unsafe {
            libc::mmap(std::ptr::null_mut(), length, libc::PROT_READ, libc::MAP_PRIVATE, source.file.as_raw_fd(), 0)
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().into());
        }
        let Some(pointer) = NonNull::new(raw.cast()) else {
            // SAFETY: a successful mapping must be released even when it
            // cannot back a Rust slice because the pointer is null.
            unsafe { libc::munmap(raw, length) };
            return Err(StqSourceError::Invalid("null mapped address"));
        };
        Ok(Self {
            pointer,
            length,
        })
    }

    fn bytes(
        &self,
        bytes: Range<usize>,
    ) -> &[u8] {
        assert!(bytes.start <= bytes.end && bytes.end <= self.length);
        // SAFETY: the checked range belongs to the live read-only mapping,
        // and the slice cannot outlive it.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().add(bytes.start), bytes.len()) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this object exclusively owns this mapped extent.
        unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.length) };
    }
}

#[cfg(test)]
mod tests {
    use std::{
        os::unix::fs::symlink,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    const TENSOR: &str = r#"{"dtype":"U16","shape":[2],"data_offsets":[0,4]}"#;

    #[stark_engine_macros::stark_test]
    fn rejects_duplicate_tensor_names_and_metadata_fields() {
        for json in [
            format!(r#"{{"tensor":{TENSOR},"tensor":{TENSOR}}}"#),
            r#"{"__metadata__":{},"__metadata__":{}}"#.to_owned(),
            r#"{"__metadata__":{"k":"4","k":"6"}}"#.to_owned(),
            r#"{"tensor":{"dtype":"U16","dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_owned(),
        ] {
            assert!(serde_json::from_str::<Header>(&json).is_err());
        }
        assert!(serde_json::from_str::<StringMap>(r#"{"tensor":"one","tensor":"two"}"#).is_err());
    }

    #[stark_engine_macros::stark_test]
    fn constrains_padding_and_rejects_bad_tensor_extents() {
        let json = format!(r#"{{"first":{TENSOR},"second":{{"dtype":"BF16","shape":[1],"data_offsets":[8,10]}}}}"#);
        let spans = |len, gap| {
            let header: Header = serde_json::from_str(&json).unwrap();
            check_spans(header.tensors, 0, 16, len, gap)
        };
        let tensors = spans(10, Some(4..8)).unwrap();
        assert_eq!(tensors[1].1.bytes, 24..26);
        assert!(spans(10, None).is_err());
        assert!(spans(10, Some(2..8)).is_err());
        assert!(spans(9, Some(4..8)).is_err());
        assert!(spans(11, Some(4..8)).is_err());
        for raw in [
            r#"{"dtype":"U16","shape":[0],"data_offsets":[0,2]}"#,
            r#"{"dtype":"U16","shape":[3],"data_offsets":[0,4]}"#,
            r#"{"dtype":"U16","shape":[2],"data_offsets":[2,6]}"#,
            r#"{"dtype":"F32","shape":[1],"data_offsets":[2,6]}"#,
            r#"{"dtype":"U16","shape":[4294967295,4294967295,2],"data_offsets":[0,4]}"#,
        ] {
            let json = format!(r#"{{"first":{TENSOR},"bad":{raw}}}"#);
            let header: Header = serde_json::from_str(&json).unwrap();
            assert!(check_spans(header.tensors, 0, 16, 6, None).is_err());
        }
    }

    struct FixtureDirectory(PathBuf);

    impl FixtureDirectory {
        fn new() -> Self {
            static ID: AtomicUsize = AtomicUsize::new(0);
            let id = ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("stark-stq-source-{}-{id}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for FixtureDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[stark_engine_macros::stark_test]
    fn descriptor_and_mmap_view_match_the_tensor_payload() {
        let fixture = FixtureDirectory::new();
        let directory = File::open(&fixture.0).unwrap();
        let name = "engram-model-00001-of-00148.safetensors";
        let mut json = format!(r#"{{"tensor":{TENSOR}}}"#).into_bytes();
        json.resize(json.len().next_multiple_of(8), b' ');
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        let offset = bytes.len();
        let expected = [0x12, 0x34, 0xAB, 0xCD];
        bytes.extend_from_slice(&expected);
        fs::write(fixture.0.join(name), bytes).unwrap();

        let shard = SourceFile::open(&directory, name.to_owned()).unwrap();
        let tensors: HashMap<_, _> = read_shard(&shard, &mut Sha256::new(), 0).unwrap().into_iter().collect();
        let source = StqSource {
            root: fixture.0.clone(),
            fingerprint: Fingerprint::from(&directory.metadata().unwrap()),
            directory,
            sidecars: Vec::new(),
            shards: vec![shard],
            config: Vec::new(),
            tensors,
        };
        let location = source.tensor_location("tensor").unwrap();
        assert_eq!(location.offset, offset as u64);
        assert_eq!(location.shard_name, name);
        assert_eq!(location.byte_len, expected.len());
        let mut payload = [0; 4];
        location.file.read_exact_at(&mut payload, location.offset).unwrap();
        assert_eq!(payload, expected);

        // SAFETY: the fixture is private and stays untouched until drop.
        let mapped = unsafe { source.map() }.unwrap();
        let view = mapped.tensor("tensor").unwrap();
        assert_eq!(view.dtype, StqDtype::U16);
        assert_eq!(view.shape, &[2]);
        assert_eq!(view.bytes, &expected);
        assert!(mapped.tensor("missing").is_none());
    }

    #[stark_engine_macros::stark_test]
    fn rejects_unsafe_component_paths_and_a_replaced_inode() {
        let fixture = FixtureDirectory::new();
        let directory = File::open(&fixture.0).unwrap();
        let path = fixture.0.join("shard");
        fs::write(&path, [1]).unwrap();
        symlink(&path, fixture.0.join("alias")).unwrap();
        assert!(SourceFile::open(&directory, "alias".to_owned()).is_err());
        assert!(SourceFile::open(&directory, "../shard".to_owned()).is_err());
        let source = SourceFile::open(&directory, "shard".to_owned()).unwrap();
        source.verify_unchanged(&fixture.0).unwrap();
        fs::rename(&path, fixture.0.join("original")).unwrap();
        fs::write(path, [1]).unwrap();
        assert!(source.verify_unchanged(&fixture.0).is_err());
    }
}
