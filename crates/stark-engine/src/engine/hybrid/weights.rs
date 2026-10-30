//! Bounded native GGUF / MLX weight access without an expanded model copy.
//!
//! GGUF block layouts and the IQ lookup tables are adapted from ggml, revision
//! 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552 (MIT).
//! Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
//! Scalar decoding supplies the reference for the native Metal matvec path.

#[cfg(feature = "metal")]
use std::{collections::VecDeque, sync::Mutex};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufReader, Read, Seek},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

use half::{bf16, f16};
use serde_json::Value;

use super::{Error, Result};

const MAX_HEADER: usize = 128 * 1024 * 1024;
const MAX_COUNT: usize = 2_000_000;
const MAX_VECTOR: usize = 8_000_000;
const MAX_SELECTED_BYTES: usize = 256 * 1024 * 1024;
#[cfg(feature = "metal")]
const MAX_BATCH_WEIGHT_BYTES: usize = 2 * 1024 * 1024 * 1024;
#[cfg(feature = "metal")]
const MAX_STATIC_VIEW_BYTES: usize = 8 * 1024 * 1024 * 1024;
const U32_KIND: u32 = 101;
const I64_KIND: u32 = 102;
const I32_KIND: u32 = 103;
const U8_KIND: u32 = 104;

/// Mapping stays alive for every borrowed tensor byte range. Source files must
/// remain immutable for the loaded model's lifetime, like the existing loader.
struct Mapping {
    _file: File,
    pointer: NonNull<u8>,
    length: usize,
}

impl Mapping {
    fn new(path: &Path) -> Result<Self> {
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file() {
            return Err(Error::Invalid("weight source must be a regular file"));
        }
        let file = fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path)?;
        if !file.metadata()?.is_file() {
            return Err(Error::Invalid("weight source must be a regular file"));
        }
        let length = usize::try_from(file.metadata()?.len()).map_err(|_| Error::Invalid("weight file is too large"))?;
        if length == 0 {
            return Err(Error::Invalid("empty weight file"));
        }
        // SAFETY: this is an owned read-only mapping of a regular file. No
        // references outlive Mapping, and Drop unmaps exactly this range.
        let raw = unsafe {
            libc::mmap(std::ptr::null_mut(), length, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0)
        };
        if raw == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let pointer = NonNull::new(raw.cast()).ok_or(Error::Invalid("invalid weight mapping"))?;
        Ok(Self {
            _file: file,
            pointer,
            length,
        })
    }

    fn bytes(
        &self,
        offset: usize,
        size: usize,
    ) -> Result<&[u8]> {
        let end = offset.checked_add(size).ok_or(Error::Invalid("weight span overflow"))?;
        if end > self.length {
            return Err(Error::Invalid("truncated weight payload"));
        }
        // SAFETY: the checked span belongs to the live immutable mapping.
        Ok(unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().add(offset), size) })
    }

    #[cfg(feature = "metal")]
    fn metal_span(
        &self,
        offset: usize,
        size: usize,
    ) -> Result<(usize, usize)> {
        // Validate the logical file span before rounding only the OS mapping
        // extent. mmap owns complete pages; the last partial file page is
        // zero-filled by the OS. GPU kernels only address the checked span.
        self.bytes(offset, size)?;
        static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let page = *PAGE_SIZE.get_or_init(|| {
            // SAFETY: sysconf has no pointer arguments or ownership effects.
            usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(0)
        });
        if !page.is_power_of_two() {
            return Err(Error::Invalid("unsupported mapping page size"));
        }
        let start = offset & !(page - 1);
        let end = offset
            .checked_add(size)
            .and_then(|n| n.checked_add(page - 1))
            .ok_or(Error::Invalid("mapped Metal span overflow"))?
            & !(page - 1);
        let owned_end =
            self.length.checked_add(page - 1).ok_or(Error::Invalid("mapped file span overflow"))? & !(page - 1);
        if end > owned_end || end <= start {
            return Err(Error::Invalid("invalid mapped Metal page span"));
        }
        Ok((start, end - start))
    }

    #[cfg(feature = "metal")]
    fn metal_slice(
        self: &Arc<Self>,
        offset: usize,
        size: usize,
    ) -> Result<super::weight_metal::MappedSlice> {
        let (start, registered_length) = self.metal_span(offset, size)?;
        // SAFETY: start/end lie in the complete pages owned by mmap. Capturing
        // this Arc in Metal's deallocator keeps the allocation and file alive,
        // including when the Weights store is dropped before the GPU buffer.
        objc2::rc::autoreleasepool(|_| unsafe {
            super::weight_metal::mapped_slice(
                self.pointer.add(start).cast(),
                registered_length,
                offset - start,
                size,
                Arc::clone(self),
            )
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the address and length are exactly those returned by mmap.
        unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.length) };
    }
}

// The mapping is immutable and retained by the store for every shared access.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

#[derive(Clone)]
struct Tensor {
    file: usize,
    offset: usize,
    size: usize,
    shape: Vec<usize>,
    kind: u32,
    affine: Option<Affine>,
}

#[derive(Clone)]
struct Affine {
    bits: usize,
    group: usize,
    scales: String,
    biases: String,
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StaticViewKey {
    file: usize,
    offset: usize,
    length: usize,
}

#[cfg(feature = "metal")]
struct StaticViewCache {
    limit: usize,
    registered_bytes: usize,
    entries: HashMap<StaticViewKey, super::weight_metal::MappedSlice>,
    order: VecDeque<StaticViewKey>,
}

#[cfg(feature = "metal")]
impl StaticViewCache {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            registered_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn reserve(
        &mut self,
        bytes: usize,
        eviction_label: &'static str,
    ) {
        while self.registered_bytes > self.limit - bytes {
            let key = self.order.pop_front().unwrap();
            let old = self.entries.remove(&key).unwrap();
            let released = old.registered_bytes();
            self.registered_bytes -= released;
            super::profile::record(eviction_label, 0.0, released);
        }
    }
}

pub struct Weights {
    files: Vec<Arc<Mapping>>,
    tensors: HashMap<String, Tensor>,
    metadata: BTreeMap<String, Value>,
    #[cfg(feature = "metal")]
    static_views: Mutex<StaticViewCache>,
    #[cfg(feature = "metal")]
    bank_views: Mutex<StaticViewCache>,
    // Tensor geometry is immutable after open; avoid rescanning every bank
    // for each routed projection. Memory-budget admission remains live below.
    #[cfg(all(test, feature = "metal"))]
    indexed_bank_geometry_admitted: bool,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = if path.is_dir() {
            fs::read_dir(path)?.map(|entry| entry.map(|entry| entry.path())).collect::<std::io::Result<_>>()?
        } else {
            vec![path.to_path_buf()]
        };
        paths.retain(|path| matches!(path.extension().and_then(|v| v.to_str()), Some("gguf" | "safetensors")));
        paths.sort();
        if paths.is_empty() || paths.len() > 512 {
            return Err(Error::Invalid("missing or excessive weight shards"));
        }
        let is_gguf = paths[0].extension().and_then(|v| v.to_str()) == Some("gguf");
        if paths.iter().any(|path| (path.extension().and_then(|v| v.to_str()) == Some("gguf")) != is_gguf) {
            return Err(Error::Invalid("mixed weight containers"));
        }
        let mut result = Self {
            files: Vec::new(),
            tensors: HashMap::new(),
            metadata: BTreeMap::new(),
            #[cfg(feature = "metal")]
            static_views: Mutex::new(StaticViewCache::new(MAX_STATIC_VIEW_BYTES)),
            #[cfg(feature = "metal")]
            bank_views: Mutex::new(StaticViewCache::new(0)),
            #[cfg(all(test, feature = "metal"))]
            indexed_bank_geometry_admitted: false,
        };
        let mut splits = BTreeMap::new();
        for path in &paths {
            let file = result.files.len();
            let mapping = Arc::new(Mapping::new(path)?);
            let (tensors, metadata) = if is_gguf {
                read_gguf(&mapping._file, file, mapping.length)?
            } else {
                read_safetensors(&mapping, file)?
            };
            if is_gguf {
                let count = metadata.get("split.count").and_then(Value::as_u64).unwrap_or(1);
                let index = metadata.get("split.no").and_then(Value::as_u64).unwrap_or(0);
                if count == 0 || count > 512 || index >= count || splits.insert(index, count).is_some() {
                    return Err(Error::Invalid("invalid or duplicate GGUF split"));
                }
                if index == 0 {
                    result.metadata = metadata;
                }
            }
            for (name, tensor) in tensors {
                if result.tensors.insert(name, tensor).is_some() {
                    return Err(Error::Invalid("duplicate tensor across weight shards"));
                }
            }
            result.files.push(mapping);
        }
        if is_gguf {
            let count = paths.len() as u64;
            if splits.len() != paths.len() || splits.iter().any(|(&index, &total)| index >= count || total != count) {
                return Err(Error::Invalid("incomplete GGUF split set"));
            }
            if let Some(total) = result.metadata.get("split.tensors.count").and_then(Value::as_u64)
                && total != result.tensors.len() as u64
            {
                return Err(Error::Invalid("incomplete GGUF tensor inventory"));
            }
        } else {
            let config_path = if path.is_dir() {
                path.join("config.json")
            } else {
                path.parent().unwrap_or(Path::new(".")).join("config.json")
            };
            let config_file = File::open(config_path)?;
            if config_file.metadata()?.len() > MAX_HEADER as u64 {
                return Err(Error::Invalid("weight config is too large"));
            }
            let config: Value = serde_json::from_reader(BufReader::new(config_file))?;
            result.configure_affine(&config)?;
        }
        #[cfg(feature = "metal")]
        result.configure_bank_views()?;
        Ok(result)
    }

    /// Retain complete rank-three weight-bank views only when the entire
    /// inventory fits below three quarters of Metal's recommended working set,
    /// including the separate static-view allowance. PLE/embedding tables stay
    /// CPU-mapped. These are immutable buffer wrappers, not a page-lock promise.
    #[cfg(feature = "metal")]
    fn configure_bank_views(&mut self) -> Result<()> {
        #[cfg(test)]
        {
            self.indexed_bank_geometry_admitted =
                self.tensors.values().filter(|tensor| tensor.shape.len() == 3).all(|tensor| {
                    tensor.shape[0].checked_mul(tensor.shape[1]).is_some_and(|rows| rows <= MAX_VECTOR)
                        && tensor.shape[2] <= MAX_VECTOR
                });
        }
        let mut spans = HashMap::new();
        for tensor in self.tensors.values().filter(|tensor| tensor.shape.len() == 3) {
            let mut banks = vec![tensor];
            if let Some(affine) = &tensor.affine {
                banks.push(self.tensor(&affine.scales)?);
                banks.push(self.tensor(&affine.biases)?);
            }
            for bank in banks {
                let key = StaticViewKey {
                    file: bank.file,
                    offset: bank.offset,
                    length: bank.size,
                };
                spans.insert(key, self.files[bank.file].metal_span(bank.offset, bank.size)?.1);
            }
        }
        let bytes = spans
            .values()
            .try_fold(0usize, |total, &bytes| total.checked_add(bytes))
            .ok_or(Error::Invalid("Metal bank inventory overflow"))?;
        let budget = (super::weight_metal::recommended_working_set()? / 4 * 3).saturating_sub(MAX_STATIC_VIEW_BYTES);
        // Kernel view offsets are u32 even on devices whose buffer length
        // limit is larger. Larger banks keep the per-selected-range path.
        let max_buffer_length = super::weight_metal::max_buffer_length()?.min(u32::MAX as usize);
        if bytes <= budget && spans.values().all(|&bytes| bytes <= max_buffer_length) {
            let cache = self.bank_views.get_mut().map_err(|_| Error::Invalid("Metal bank view cache failed"))?;
            cache.limit = bytes;
        }
        Ok(())
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
    pub fn has(
        &self,
        name: &str,
    ) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    fn tensor(
        &self,
        name: &str,
    ) -> Result<&Tensor> {
        self.tensors.get(name).ok_or(Error::Invalid("required weight tensor is missing"))
    }

    pub fn shape(
        &self,
        name: &str,
    ) -> Result<&[usize]> {
        Ok(&self.tensor(name)?.shape)
    }

    fn raw(
        &self,
        tensor: &Tensor,
    ) -> Result<&[u8]> {
        self.files[tensor.file].bytes(tensor.offset, tensor.size)
    }

    fn configure_affine(
        &mut self,
        config: &Value,
    ) -> Result<()> {
        let quant = config.get("quantization").or_else(|| config.get("quantization_config"));
        let names: Vec<_> = self.tensors.keys().filter(|name| name.ends_with(".weight")).cloned().collect();
        for name in names {
            if !matches!(self.tensor(&name)?.kind, U32_KIND | U8_KIND) {
                continue;
            }
            let stem = name.strip_suffix(".weight").unwrap();
            let scales_name = format!("{stem}.scales");
            let biases_name = format!("{stem}.biases");
            let scales = self.tensor(&scales_name)?;
            let biases = self.tensor(&biases_name)?;
            let tensor = self.tensor(&name)?;
            let description = quant
                .and_then(|q| q.get(stem).or_else(|| stem.strip_prefix("language_model.").and_then(|name| q.get(name))))
                .or(quant)
                .ok_or(Error::Invalid("missing affine quantization config"))?;
            if description.get("mode").and_then(Value::as_str).is_some_and(|mode| mode != "affine") {
                return Err(Error::Unsupported("unsupported matrix quantization mode"));
            }
            // Recursive's official INT4 export stores the same little-endian
            // MLX words as byte arrays. Its embedding alone uses eight bits,
            // explicitly declared by mlx_export rather than the global width.
            let bits =
                if config.get("model_type").and_then(Value::as_str) == Some("flint") && stem == "model.embed_tokens" {
                    config.get("mlx_export").and_then(|v| v.get("embedding_bits")).or_else(|| description.get("bits"))
                } else {
                    description.get("bits")
                }
                .and_then(Value::as_u64)
                .ok_or(Error::Invalid("missing affine width"))? as usize;
            let group = description
                .get("group_size")
                .and_then(Value::as_u64)
                .ok_or(Error::Invalid("missing affine group size"))? as usize;
            let rank = tensor.shape.len();
            let stored_bits = if tensor.kind == U8_KIND {
                8
            } else {
                32
            };
            if !matches!(bits, 4 | 8)
                || !matches!(group, 32 | 64 | 128)
                || rank < 2
                || scales.shape.len() != rank
                || scales.shape != biases.shape
                || scales.kind != biases.kind
                || !matches!(scales.kind, 0 | 1 | 30)
                || tensor.shape[..rank - 1] != scales.shape[..rank - 1]
                || (tensor.kind == U8_KIND && !tensor.shape[rank - 1].is_multiple_of(4))
            {
                return Err(Error::Unsupported("unsupported affine matrix layout"));
            }
            let columns = scales.shape[rank - 1].checked_mul(group).ok_or(Error::Invalid("affine shape overflow"))?;
            if tensor.shape[rank - 1].checked_mul(stored_bits) != columns.checked_mul(bits) {
                return Err(Error::Invalid("inconsistent affine planes"));
            }
            let tensor = self.tensors.get_mut(&name).unwrap();
            tensor.kind = U32_KIND;
            tensor.shape[rank - 1] = columns;
            tensor.affine = Some(Affine {
                bits,
                group,
                scales: scales_name,
                biases: biases_name,
            });
        }
        Ok(())
    }

    /// Flatten a small floating tensor, including channel-major convolutions.
    pub fn vector(
        &self,
        name: &str,
    ) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        let count = product(&tensor.shape)?;
        if count > MAX_VECTOR || tensor.affine.is_some() {
            return Err(Error::Invalid("vector access exceeds bounded floating tensor size"));
        }
        decode_values(tensor.kind, self.raw(tensor)?, count)
    }

    /// Preserve exact integer hash tables; float roundtrips are not permitted.
    pub fn integers(
        &self,
        name: &str,
    ) -> Result<Vec<i64>> {
        let tensor = self.tensor(name)?;
        let count = product(&tensor.shape)?;
        if count > MAX_VECTOR {
            return Err(Error::Invalid("integer tensor is too large"));
        }
        let data = self.raw(tensor)?;
        match tensor.kind {
            I64_KIND => Ok(data.as_chunks::<8>().0.iter().map(|v| i64::from_le_bytes(*v)).collect()),
            I32_KIND => Ok(data.as_chunks::<4>().0.iter().map(|v| i64::from(i32::from_le_bytes(*v))).collect()),
            U32_KIND => Ok(data.as_chunks::<4>().0.iter().map(|v| i64::from(u32::from_le_bytes(*v))).collect()),
            _ => Err(Error::Invalid("integer tensor dtype is required")),
        }
    }

    pub fn row(
        &self,
        name: &str,
        row: usize,
    ) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() < 2 {
            return Err(Error::Invalid("matrix row access requires rank two or higher"));
        }
        let columns = *tensor.shape.last().unwrap();
        let rows = product(&tensor.shape[..tensor.shape.len() - 1])?;
        if row >= rows || columns > MAX_VECTOR {
            return Err(Error::Invalid("matrix row is out of range"));
        }
        if let Some(affine) = &tensor.affine {
            return self.affine_row(tensor, affine, row, columns);
        }
        let row_bytes = encoded_size(tensor.kind, columns)?;
        let bytes = self.raw(tensor)?;
        decode_values(tensor.kind, &bytes[row * row_bytes..(row + 1) * row_bytes], columns)
    }

    fn affine_row(
        &self,
        tensor: &Tensor,
        affine: &Affine,
        row: usize,
        columns: usize,
    ) -> Result<Vec<f32>> {
        let codes = self.raw(tensor)?;
        let scales = self.tensor(&affine.scales)?;
        let biases = self.tensor(&affine.biases)?;
        let scale_data = self.raw(scales)?;
        let bias_data = self.raw(biases)?;
        let group_count = columns / affine.group;
        let scale_size = encoded_size(scales.kind, 1)?;
        let packed_row = columns * affine.bits / 8;
        let mut output = vec![0.0; columns];
        let codes = &codes[row * packed_row..(row + 1) * packed_row];
        for (column, value) in output.iter_mut().enumerate() {
            let index = row * group_count + column / affine.group;
            let scale = scalar(scales.kind, &scale_data[index * scale_size..])?;
            let bias = scalar(biases.kind, &bias_data[index * scale_size..])?;
            let q = (codes[column * affine.bits / 8] >> (column * affine.bits % 8)) & ((1u16 << affine.bits) - 1) as u8;
            *value = scale * f32::from(q) + bias;
        }
        Ok(output)
    }

    pub fn matvec(
        &self,
        name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 2 {
            return Err(Error::Invalid("matvec requires a matrix"));
        }
        self.matrix_slice(tensor, 0, tensor.shape[0], tensor.shape[1], input)
    }

    /// Execute independent matrix products in order with one native Metal
    /// submission. Inputs, parameters and outputs remain distinct for every
    /// job until that submission completes; heterogeneous formats are allowed.
    pub fn matvec_many(
        &self,
        jobs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>> {
        if jobs.is_empty() || jobs.len() > 64 {
            return Err(Error::Invalid("invalid matrix job count"));
        }
        #[cfg(feature = "metal")]
        {
            let mut total_weights = 0usize;
            let mut total_inputs = 0usize;
            let mut total_outputs = 0usize;
            for &(name, inputs) in jobs {
                let tensor = self.tensor(name)?;
                if tensor.shape.len() != 2
                    || tensor.shape[1] != inputs.len()
                    || inputs.iter().any(|value| !value.is_finite())
                {
                    return Err(Error::Invalid("invalid matrix job shape or values"));
                }
                let plane_bytes = if let Some(affine) = &tensor.affine {
                    self.tensor(&affine.scales)?
                        .size
                        .checked_add(self.tensor(&affine.biases)?.size)
                        .ok_or(Error::Invalid("matrix job size overflow"))?
                } else {
                    0
                };
                total_weights = total_weights
                    .checked_add(tensor.size)
                    .and_then(|n| n.checked_add(plane_bytes))
                    .ok_or(Error::Invalid("matrix job size overflow"))?;
                total_inputs =
                    total_inputs.checked_add(inputs.len()).ok_or(Error::Invalid("matrix job size overflow"))?;
                total_outputs =
                    total_outputs.checked_add(tensor.shape[0]).ok_or(Error::Invalid("matrix job size overflow"))?;
                if total_weights > MAX_BATCH_WEIGHT_BYTES || total_inputs > MAX_VECTOR || total_outputs > MAX_VECTOR {
                    return Err(Error::Invalid("matrix batch exceeds workspace bound"));
                }
            }
            let mut operations = Vec::with_capacity(jobs.len());
            for &(name, inputs) in jobs {
                let tensor = self.tensor(name)?;
                if tensor.shape.len() != 2 {
                    return Err(Error::Invalid("matvec requires a matrix"));
                }
                operations.push(super::weight_metal::MappedOperation {
                    matrices: vec![self.mapped_matrix(tensor, 0, tensor.shape[0], tensor.shape[1], true)?],
                    inputs,
                    rows: tensor.shape[0],
                    columns: tensor.shape[1],
                    rows_per_batch: 0,
                    format: self.matrix_format(tensor)?,
                });
            }
            super::weight_metal::mapped_many(&operations)
        }
        #[cfg(not(feature = "metal"))]
        jobs.iter().map(|(name, inputs)| self.matvec(name, inputs)).collect()
    }

    #[cfg(feature = "metal")]
    fn matrix_format(
        &self,
        tensor: &Tensor,
    ) -> Result<super::weight_metal::MatrixFormat> {
        Ok(if let Some(affine) = &tensor.affine {
            super::weight_metal::MatrixFormat::Affine {
                scale_dtype: self.tensor(&affine.scales)?.kind,
                bits: affine.bits,
                group: affine.group,
            }
        } else {
            super::weight_metal::MatrixFormat::Packed(tensor.kind)
        })
    }

    #[cfg(feature = "metal")]
    fn metal_view(
        &self,
        file: usize,
        offset: usize,
        length: usize,
        cacheable: bool,
    ) -> Result<super::weight_metal::MappedSlice> {
        let mapping = &self.files[file];
        if !cacheable {
            return mapping.metal_slice(offset, length);
        }
        let key = StaticViewKey {
            file,
            offset,
            length,
        };
        let mut cache = self.static_views.lock().map_err(|_| Error::Invalid("static Metal view cache failed"))?;
        if let Some(view) = cache.entries.get(&key) {
            let result = view.clone();
            super::profile::record("native.metal.static_cache.hit", 0.0, result.registered_bytes());
            super::profile::record("native.metal.static_cache.registered_bytes", 0.0, cache.registered_bytes);
            return Ok(result);
        }
        let registered = mapping.metal_span(offset, length)?.1;
        super::profile::record("native.metal.static_cache.miss", 0.0, registered);
        if registered > cache.limit {
            drop(cache);
            return mapping.metal_slice(offset, length);
        }
        // Evict before allocation; creating while holding the mutex also
        // prevents duplicate FIFO keys and accounting races across callers.
        cache.reserve(registered, "native.metal.static_cache.eviction");
        let view = mapping.metal_slice(offset, length)?;
        cache.registered_bytes += view.registered_bytes();
        cache.order.push_back(key);
        cache.entries.insert(key, view.clone());
        super::profile::record("native.metal.static_cache.registered_bytes", 0.0, cache.registered_bytes);
        Ok(view)
    }

    #[cfg(feature = "metal")]
    fn bank_view(
        &self,
        tensor: &Tensor,
        start: usize,
        length: usize,
    ) -> Result<super::weight_metal::MappedSlice> {
        let mut cache = self.bank_views.lock().map_err(|_| Error::Invalid("Metal bank view cache failed"))?;
        if cache.limit == 0 {
            drop(cache);
            return self.metal_view(tensor.file, tensor.offset + start, length, false);
        }
        let key = StaticViewKey {
            file: tensor.file,
            offset: tensor.offset,
            length: tensor.size,
        };
        if let Some(view) = cache.entries.get(&key) {
            let result = view.subslice(start, length)?;
            super::profile::record("native.metal.bank_cache.hit", 0.0, 0);
            return Ok(result);
        }
        let mapping = &self.files[tensor.file];
        let registered = mapping.metal_span(tensor.offset, tensor.size)?.1;
        if registered > cache.limit {
            drop(cache);
            return self.metal_view(tensor.file, tensor.offset + start, length, false);
        }
        cache.reserve(registered, "native.metal.bank_cache.eviction");
        let view = mapping.metal_slice(tensor.offset, tensor.size)?;
        let result = view.subslice(start, length)?;
        cache.registered_bytes += registered;
        cache.order.push_back(key);
        cache.entries.insert(key, view);
        super::profile::record("native.metal.bank_cache.miss", 0.0, registered);
        super::profile::record("native.metal.bank_cache.registered_bytes", 0.0, cache.registered_bytes);
        Ok(result)
    }

    #[cfg(feature = "metal")]
    fn mapped_matrix(
        &self,
        tensor: &Tensor,
        index: usize,
        rows: usize,
        columns: usize,
        cacheable: bool,
    ) -> Result<super::weight_metal::MappedMatrix> {
        let matrix_bytes = if let Some(affine) = &tensor.affine {
            rows.checked_mul(columns).and_then(|n| n.checked_mul(affine.bits)).map(|n| n / 8)
        } else {
            rows.checked_mul(encoded_size(tensor.kind, columns)?)
        }
        .ok_or(Error::Invalid("mapped matrix size overflow"))?;
        let start = index.checked_mul(matrix_bytes).ok_or(Error::Invalid("mapped matrix offset overflow"))?;
        if start.checked_add(matrix_bytes).is_none_or(|end| end > tensor.size) {
            return Err(Error::Invalid("mapped matrix exceeds tensor span"));
        }
        let bank = tensor.shape.len() == 3 && !cacheable;
        let codes = if bank {
            self.bank_view(tensor, start, matrix_bytes)?
        } else {
            self.metal_view(tensor.file, tensor.offset + start, matrix_bytes, cacheable)?
        };
        let planes = if let Some(affine) = &tensor.affine {
            let scales = self.tensor(&affine.scales)?;
            let biases = self.tensor(&affine.biases)?;
            let item_size = encoded_size(scales.kind, 1)?;
            let plane_bytes = rows
                .checked_mul(columns / affine.group)
                .and_then(|n| n.checked_mul(item_size))
                .ok_or(Error::Invalid("mapped affine plane overflow"))?;
            let start = index.checked_mul(plane_bytes).ok_or(Error::Invalid("mapped plane offset overflow"))?;
            if start.checked_add(plane_bytes).is_none_or(|end| end > scales.size || end > biases.size) {
                return Err(Error::Invalid("mapped affine plane exceeds tensor span"));
            }
            Some(if bank {
                (self.bank_view(scales, start, plane_bytes)?, self.bank_view(biases, start, plane_bytes)?)
            } else {
                (
                    self.metal_view(scales.file, scales.offset + start, plane_bytes, cacheable)?,
                    self.metal_view(biases.file, biases.offset + start, plane_bytes, cacheable)?,
                )
            })
        } else {
            None
        };
        Ok(super::weight_metal::MappedMatrix {
            codes,
            planes,
        })
    }

    pub fn expert_matvec(
        &self,
        name: &str,
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 3 || expert >= tensor.shape[0] {
            return Err(Error::Invalid("expert matrix index is out of range"));
        }
        self.matrix_slice(tensor, expert, tensor.shape[1], tensor.shape[2], input)
    }

    /// Execute each matrix of a rank-three tensor against its own input row.
    /// Shapes are [batch, output, input], [batch, input], [batch, output].
    /// The Metal path submits every head in one command without expanding the
    /// compressed matrices or changing their row reduction order.
    pub fn batched_matvec(
        &self,
        name: &str,
        inputs: &[f32],
    ) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 3 {
            return Err(Error::Invalid("batched matvec requires rank three"));
        }
        let (batches, rows, columns) = (tensor.shape[0], tensor.shape[1], tensor.shape[2]);
        if batches.checked_mul(columns) != Some(inputs.len())
            || inputs.len() > MAX_VECTOR
            || batches.checked_mul(rows).is_none_or(|count| count > MAX_VECTOR)
            || inputs.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Invalid("invalid batched matvec input shape or values"));
        }
        #[cfg(feature = "metal")]
        {
            let operation = super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(
                    tensor,
                    0,
                    batches * rows,
                    columns,
                    name.ends_with(".attn_k_b.weight") || name.ends_with(".attn_v_b.weight"),
                )?],
                inputs,
                rows: batches * rows,
                columns,
                rows_per_batch: rows,
                format: self.matrix_format(tensor)?,
            };
            super::weight_metal::mapped_many(&[operation]).map(|mut outputs| outputs.remove(0))
        }
        #[cfg(not(feature = "metal"))]
        {
            let mut output = Vec::with_capacity(batches * rows);
            for batch in 0..batches {
                output.extend(self.matrix_slice(
                    tensor,
                    batch,
                    rows,
                    columns,
                    &inputs[batch * columns..(batch + 1) * columns],
                )?);
            }
            Ok(output)
        }
    }

    /// Apply nonconsecutive expert matrices in selection order. The only weight
    /// workspace contains the selected compressed slices (plus affine planes),
    /// bounded to 256 MiB; no full-precision expert weights are materialized.
    pub fn selected_matvec(
        &self,
        name: &str,
        experts: &[usize],
        inputs: &[f32],
    ) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 3 || experts.is_empty() {
            return Err(Error::Invalid("selected matvec requires rank three and nonempty experts"));
        }
        let (count, rows, columns) = (tensor.shape[0], tensor.shape[1], tensor.shape[2]);
        if experts.iter().any(|&expert| expert >= count)
            || experts.len().checked_mul(columns) != Some(inputs.len())
            || inputs.len() > MAX_VECTOR
            || experts.len().checked_mul(rows).is_none_or(|total| total > MAX_VECTOR)
            || inputs.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Invalid("invalid selected expert geometry or inputs"));
        }
        let matrix_bytes = if let Some(affine) = &tensor.affine {
            rows.checked_mul(columns).and_then(|n| n.checked_mul(affine.bits)).map(|n| n / 8)
        } else {
            rows.checked_mul(encoded_size(tensor.kind, columns)?)
        }
        .ok_or(Error::Invalid("selected expert size overflow"))?;
        let plane_bytes = if let Some(affine) = &tensor.affine {
            let item_size = encoded_size(self.tensor(&affine.scales)?.kind, 1)?;
            rows.checked_mul(columns / affine.group)
                .and_then(|n| n.checked_mul(item_size))
                .ok_or(Error::Invalid("selected affine plane size overflow"))?
        } else {
            0
        };
        let selected_bytes = plane_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(matrix_bytes))
            .and_then(|n| n.checked_mul(experts.len()))
            .ok_or(Error::Invalid("selected expert workspace overflow"))?;
        if selected_bytes > MAX_SELECTED_BYTES {
            return Err(Error::Invalid("selected expert workspace exceeds bound"));
        }
        #[cfg(feature = "metal")]
        {
            // Map only selected compressed ranges. Each expert dispatch shares
            // one command buffer; unused experts are never registered as GPU
            // resources and there is no gathered weight copy.
            let matrices = {
                let _views = super::profile::Scope::bytes("native.weights.views", selected_bytes);
                experts
                    .iter()
                    .map(|&expert| self.mapped_matrix(tensor, expert, rows, columns, false))
                    .collect::<Result<Vec<_>>>()?
            };
            let operation = super::weight_metal::MappedOperation {
                matrices,
                inputs,
                rows,
                columns,
                rows_per_batch: 0,
                format: self.matrix_format(tensor)?,
            };
            super::weight_metal::mapped_many(&[operation]).map(|mut outputs| outputs.remove(0))
        }
        #[cfg(not(feature = "metal"))]
        {
            let mut output = Vec::with_capacity(experts.len() * rows);
            for (index, &expert) in experts.iter().enumerate() {
                output.extend(self.matrix_slice(
                    tensor,
                    expert,
                    rows,
                    columns,
                    &inputs[index * columns..(index + 1) * columns],
                )?);
            }
            Ok(output)
        }
    }

    /// Only a complete admitted rank-three inventory permits device-side bank
    /// indexing. Partial admission must retain the bounded CPU selection path.
    #[cfg(all(test, feature = "metal"))]
    pub(super) fn indexed_banks_admitted(&self) -> bool {
        self.bank_views.lock().is_ok_and(|cache| cache.limit > 0) && self.indexed_bank_geometry_admitted
    }

    #[cfg(feature = "metal")]
    pub(super) fn encode_matvec(
        &self,
        graph: &mut super::weight_metal::DecodeGraph,
        name: &str,
        input: &super::weight_metal::DeviceVector,
    ) -> Result<super::weight_metal::DeviceVector> {
        let t = self.tensor(name)?;
        if t.shape.len() != 2 || t.shape[1] != input.len || t.shape.iter().any(|&n| n > MAX_VECTOR) {
            return Err(Error::Invalid("decode matrix shape mismatch"));
        }
        graph.mapped(
            &super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(t, 0, t.shape[0], t.shape[1], true)?],
                inputs: &[],
                rows: t.shape[0],
                columns: t.shape[1],
                rows_per_batch: 0,
                format: self.matrix_format(t)?,
            },
            input,
        )
    }

    #[cfg(feature = "metal")]
    pub(super) fn encode_batched_matvec(
        &self,
        graph: &mut super::weight_metal::DecodeGraph,
        name: &str,
        input: &super::weight_metal::DeviceVector,
    ) -> Result<super::weight_metal::DeviceVector> {
        let t = self.tensor(name)?;
        if t.shape.len() != 3
            || t.shape[0].checked_mul(t.shape[2]) != Some(input.len)
            || t.shape[0].checked_mul(t.shape[1]).is_none_or(|n| n > MAX_VECTOR)
        {
            return Err(Error::Invalid("decode batched matrix shape mismatch"));
        }
        graph.mapped(
            &super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(t, 0, t.shape[0] * t.shape[1], t.shape[2], true)?],
                inputs: &[],
                rows: t.shape[0] * t.shape[1],
                columns: t.shape[2],
                rows_per_batch: t.shape[1],
                format: self.matrix_format(t)?,
            },
            input,
        )
    }

    #[cfg(feature = "metal")]
    pub(super) fn encode_selected_matvec(
        &self,
        graph: &mut super::weight_metal::DecodeGraph,
        name: &str,
        experts: &[usize],
        input: &super::weight_metal::DeviceVector,
    ) -> Result<super::weight_metal::DeviceVector> {
        let t = self.tensor(name)?;
        if t.shape.len() != 3 || experts.is_empty() || experts.len() > 64 || experts.iter().any(|&i| i >= t.shape[0]) {
            return Err(Error::Invalid("decode selected matrix geometry"));
        }
        let columns = t.shape[2];
        let repeated;
        let input = if input.len == columns && experts.len() > 1 {
            repeated = graph.repeat(input, experts.len())?;
            &repeated
        } else {
            input
        };
        if columns.checked_mul(experts.len()) != Some(input.len) {
            return Err(Error::Invalid("decode selected matrix input"));
        }
        let planes = if let Some(a) = &t.affine {
            self.tensor(&a.scales)?.size.checked_add(self.tensor(&a.biases)?.size)
        } else {
            Some(0)
        };
        if planes
            .and_then(|n| n.checked_add(t.size))
            .map(|n| n / t.shape[0])
            .and_then(|n| n.checked_mul(experts.len()))
            .is_none_or(|n| n > MAX_SELECTED_BYTES)
        {
            return Err(Error::Invalid("decode selected weight bound exceeded"));
        }
        graph.mapped(
            &super::weight_metal::MappedOperation {
                matrices: experts
                    .iter()
                    .map(|&i| self.mapped_matrix(t, i, t.shape[1], columns, false))
                    .collect::<Result<_>>()?,
                inputs: &[],
                rows: t.shape[1],
                columns,
                rows_per_batch: 0,
                format: self.matrix_format(t)?,
            },
            input,
        )
    }

    #[cfg(all(test, feature = "metal"))]
    pub(super) fn encode_indexed_matvec(
        &self,
        graph: &mut super::weight_metal::DecodeGraph,
        name: &str,
        indices: &super::weight_metal::DeviceVector,
        selected: usize,
        input: &super::weight_metal::DeviceVector,
        input_batched: bool,
    ) -> Result<super::weight_metal::DeviceVector> {
        if !self.indexed_banks_admitted() {
            return Err(Error::Invalid("complete expert inventory is not admitted"));
        }
        let t = self.tensor(name)?;
        if t.shape.len() != 3 || t.shape.iter().any(|&n| n == 0 || n > MAX_VECTOR) {
            return Err(Error::Invalid("decode indexed matrix requires rank three"));
        }
        let planes = if let Some(a) = &t.affine {
            self.tensor(&a.scales)?.size.checked_add(self.tensor(&a.biases)?.size)
        } else {
            Some(0)
        };
        if planes
            .and_then(|n| n.checked_add(t.size))
            .map(|n| n / t.shape[0])
            .and_then(|n| n.checked_mul(selected))
            .is_none_or(|n| n > MAX_SELECTED_BYTES)
        {
            return Err(Error::Invalid("decode indexed selected weight bound exceeded"));
        }
        let rows = t.shape[0]
            .checked_mul(t.shape[1])
            .filter(|&n| n <= MAX_VECTOR)
            .ok_or(Error::Invalid("decode indexed bank shape bound"))?;
        let matrix = self.mapped_matrix(t, 0, rows, t.shape[2], false)?;
        graph.indexed(
            &matrix,
            self.matrix_format(t)?,
            [t.shape[0], t.shape[1], t.shape[2]],
            indices,
            selected,
            input,
            input_batched,
        )
    }

    /// GLM KDA projections, recurrence and readout share one submission.
    #[cfg(feature = "metal")]
    pub(super) fn glm_kda(
        &self,
        prefix: &str,
        input: &[f32],
        state: &mut super::glm_metal::Kda,
    ) -> Result<Vec<f32>> {
        let names =
            ["attn_q", "attn_k", "attn_v", "ssm_f_a", "ssm_g_a", "ssm_beta", "ssm_f_b", "ssm_g_b", "attn_output"]
                .map(|suffix| format!("{prefix}.{suffix}.weight"));
        let tensors = names.each_ref().map(|name| self.tensor(name)).into_iter().collect::<Result<Vec<_>>>()?;
        if input.is_empty()
            || input.len() > MAX_VECTOR
            || input.iter().any(|v| !v.is_finite())
            || tensors.iter().any(|tensor| tensor.shape.len() != 2 || tensor.shape.iter().any(|&n| n > MAX_VECTOR))
            || tensors[..6].iter().any(|tensor| tensor.shape[1] != input.len())
            || tensors[6].shape[1] != tensors[3].shape[0]
            || tensors[7].shape[1] != tensors[4].shape[0]
            || tensors[8].shape[1] != tensors[0].shape[0]
            || tensors[8].shape[0] != input.len()
        {
            return Err(Error::Invalid("invalid GLM KDA projection geometry"));
        }
        let placeholders = tensors[6..].iter().map(|tensor| vec![0.0; tensor.shape[1]]).collect::<Vec<_>>();
        let operations = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                Ok(super::weight_metal::MappedOperation {
                    matrices: vec![self.mapped_matrix(tensor, 0, tensor.shape[0], tensor.shape[1], true)?],
                    inputs: if index < 6 {
                        input
                    } else {
                        &placeholders[index - 6]
                    },
                    rows: tensor.shape[0],
                    columns: tensor.shape[1],
                    rows_per_batch: 0,
                    format: self.matrix_format(tensor)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let operations = operations.try_into().map_err(|_| Error::Invalid("invalid GLM KDA projection count"))?;
        super::weight_metal::mapped_glm_kda(&operations, state)
    }

    /// Qwen DeltaNet's projections, recurrent update and output projection
    /// share one submission and keep the recurrent matrix on the GPU.
    #[cfg(all(test, feature = "metal"))]
    pub(super) fn qwen_gdn(
        &self,
        prefix: &str,
        input: &[f32],
        state: &mut super::weight_metal::QwenGdnState,
    ) -> Result<Vec<f32>> {
        let names = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            .map(|suffix| format!("{prefix}.{suffix}.weight"));
        let tensors = names.each_ref().map(|name| self.tensor(name));
        let tensors = tensors.into_iter().collect::<Result<Vec<_>>>()?;
        if input.is_empty()
            || input.len() > MAX_VECTOR
            || input.iter().any(|v| !v.is_finite())
            || tensors.iter().any(|tensor| tensor.shape.len() != 2 || tensor.shape.iter().any(|&n| n > MAX_VECTOR))
            || tensors[..4].iter().any(|tensor| tensor.shape[1] != input.len())
            || tensors[4].shape[0] != input.len()
        {
            return Err(Error::Invalid("invalid Qwen DeltaNet projection geometry"));
        }
        let output_input = vec![0.0; tensors[4].shape[1]];
        let operations = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                Ok(super::weight_metal::MappedOperation {
                    matrices: vec![self.mapped_matrix(tensor, 0, tensor.shape[0], tensor.shape[1], true)?],
                    inputs: if index < 4 {
                        input
                    } else {
                        &output_input
                    },
                    rows: tensor.shape[0],
                    columns: tensor.shape[1],
                    rows_per_batch: 0,
                    format: self.matrix_format(tensor)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let operations = operations.try_into().map_err(|_| Error::Invalid("invalid Qwen DeltaNet projection count"))?;
        super::weight_metal::mapped_qwen_gdn(&operations, state)
    }

    /// Qwen hyperconnection's dependent projections and stream mixture share
    /// one submission. Normalization stays in the source scalar path.
    #[cfg(all(test, feature = "metal"))]
    pub(super) fn qwen_hyperconnection(
        &self,
        down_name: &str,
        up_name: &str,
        inject_name: Option<&str>,
        normalized: &[f32],
        streams: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let down = self.tensor(down_name)?;
        let up = self.tensor(up_name)?;
        let inject = inject_name.map(|name| self.tensor(name)).transpose()?;
        if streams == 0
            || streams > 64
            || !normalized.len().is_multiple_of(streams)
            || down.shape.len() != 2
            || up.shape.len() != 2
            || down.shape[1] != normalized.len()
            || up.shape[1] != down.shape[0]
            || up.shape[0] != normalized.len()
            || normalized.len() > MAX_VECTOR
            || down.shape[0] > MAX_VECTOR
            || normalized.iter().any(|v| !v.is_finite())
            || inject.is_some_and(|tensor| tensor.shape != [streams, normalized.len()])
        {
            return Err(Error::Invalid("invalid Qwen hyperconnection projection geometry"));
        }
        let up_input = vec![0.0; down.shape[0]];
        let mut operations = vec![
            super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(down, 0, down.shape[0], down.shape[1], true)?],
                inputs: normalized,
                rows: down.shape[0],
                columns: down.shape[1],
                rows_per_batch: 0,
                format: self.matrix_format(down)?,
            },
            super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(up, 0, up.shape[0], up.shape[1], true)?],
                inputs: &up_input,
                rows: up.shape[0],
                columns: up.shape[1],
                rows_per_batch: 0,
                format: self.matrix_format(up)?,
            },
        ];
        if let Some(tensor) = inject {
            operations.push(super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(tensor, 0, tensor.shape[0], tensor.shape[1], true)?],
                inputs: normalized,
                rows: tensor.shape[0],
                columns: tensor.shape[1],
                rows_per_batch: 0,
                format: self.matrix_format(tensor)?,
            });
        }
        super::weight_metal::mapped_qwen_hyperconnection(&operations, normalized, streams)
    }

    /// Feed-forward projections, activations and ordered expert reduction in
    /// one Metal submission after CPU routing. Temporary storage contains only
    /// activation vectors; compressed weights stay in immutable mapped views.
    #[cfg(feature = "metal")]
    #[expect(clippy::too_many_arguments, reason = "explicit tensor names and independent routed/shared FFN inputs")]
    pub(super) fn selected_ffn(
        &self,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
        shared_down_name: &str,
        experts: &[usize],
        input: &[f32],
        probabilities: &[f32],
        shared_gate: f32,
        shared_values: &[f32],
        shared_up: &[f32],
        clamp: Option<(f32, f32)>,
    ) -> Result<Vec<f32>> {
        let gate = self.tensor(gate_name)?;
        let up = self.tensor(up_name)?;
        let down = self.tensor(down_name)?;
        let shared = self.tensor(shared_down_name)?;
        if gate.shape.len() != 3
            || gate.shape != up.shape
            || down.shape.len() != 3
            || shared.shape.len() != 2
            || gate.shape[0] != down.shape[0]
            || gate.shape[1] != down.shape[2]
            || gate.shape[2] != down.shape[1]
            || gate.shape[2] != input.len()
            || shared.shape[0] != down.shape[1]
            || shared.shape[1] != shared_values.len()
            || shared_values.len() != shared_up.len()
            || experts.is_empty()
            || experts.len() > 64
            || experts.iter().any(|&expert| expert >= gate.shape[0])
            || probabilities.len() != experts.len()
            || !shared_gate.is_finite()
            || probabilities.iter().any(|v| !v.is_finite() || *v < 0.0)
            || [input, shared_values, shared_up].iter().any(|v| v.iter().any(|x| !x.is_finite()))
            || clamp.is_some_and(|(r, s)| !r.is_finite() || !s.is_finite() || r <= 0.0 || s <= 0.0)
        {
            return Err(Error::Invalid("invalid fused feed-forward geometry or values"));
        }
        for tensor in [gate, up, down] {
            if tensor.shape[1].checked_mul(experts.len()).is_none_or(|n| n > MAX_VECTOR)
                || tensor.shape[2].checked_mul(experts.len()).is_none_or(|n| n > MAX_VECTOR)
            {
                return Err(Error::Invalid("fused feed-forward activation bound exceeded"));
            }
            let planes = if let Some(affine) = &tensor.affine {
                self.tensor(&affine.scales)?.size.checked_add(self.tensor(&affine.biases)?.size)
            } else {
                Some(0)
            };
            let selected = planes
                .and_then(|n| n.checked_add(tensor.size))
                .map(|n| n / tensor.shape[0])
                .and_then(|n| n.checked_mul(experts.len()));
            if selected.is_none_or(|n| n > MAX_SELECTED_BYTES) {
                return Err(Error::Invalid("fused feed-forward selected weight bound exceeded"));
            }
        }
        let repeated = input.repeat(experts.len());
        let down_inputs = vec![0.0; experts.len() * down.shape[2]];
        let shared_inputs = vec![0.0; shared_values.len()];
        let selected = |tensor: &Tensor| {
            experts
                .iter()
                .map(|&index| self.mapped_matrix(tensor, index, tensor.shape[1], tensor.shape[2], false))
                .collect::<Result<Vec<_>>>()
        };
        let operations = [
            super::weight_metal::MappedOperation {
                matrices: selected(gate)?,
                inputs: &repeated,
                rows: gate.shape[1],
                columns: gate.shape[2],
                rows_per_batch: 0,
                format: self.matrix_format(gate)?,
            },
            super::weight_metal::MappedOperation {
                matrices: selected(up)?,
                inputs: &repeated,
                rows: up.shape[1],
                columns: up.shape[2],
                rows_per_batch: 0,
                format: self.matrix_format(up)?,
            },
            super::weight_metal::MappedOperation {
                matrices: selected(down)?,
                inputs: &down_inputs,
                rows: down.shape[1],
                columns: down.shape[2],
                rows_per_batch: 0,
                format: self.matrix_format(down)?,
            },
            super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(shared, 0, shared.shape[0], shared.shape[1], true)?],
                inputs: &shared_inputs,
                rows: shared.shape[0],
                columns: shared.shape[1],
                rows_per_batch: 0,
                format: self.matrix_format(shared)?,
            },
        ];
        super::weight_metal::mapped_ffn(&operations, probabilities, shared_gate, shared_values, shared_up, clamp)
    }

    fn matrix_slice(
        &self,
        tensor: &Tensor,
        index: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != columns || rows > MAX_VECTOR || input.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("invalid matvec input shape or values"));
        }
        #[cfg(feature = "metal")]
        {
            let operation = super::weight_metal::MappedOperation {
                matrices: vec![self.mapped_matrix(tensor, index, rows, columns, tensor.shape.len() == 2)?],
                inputs: input,
                rows,
                columns,
                rows_per_batch: 0,
                format: self.matrix_format(tensor)?,
            };
            super::weight_metal::mapped_many(&[operation]).map(|mut outputs| outputs.remove(0))
        }
        #[cfg(not(feature = "metal"))]
        {
            let mut output = Vec::with_capacity(rows);
            let row_size = if tensor.affine.is_none() {
                encoded_size(tensor.kind, columns)?
            } else {
                0
            };
            for row in index * rows..(index + 1) * rows {
                let values = if let Some(affine) = &tensor.affine {
                    self.affine_row(tensor, affine, row, columns)?
                } else {
                    let bytes = self.raw(tensor)?;
                    decode_values(tensor.kind, &bytes[row * row_size..(row + 1) * row_size], columns)?
                };
                output.push(values.iter().zip(input).map(|(&weight, &value)| weight * value).sum());
            }
            Ok(output)
        }
    }
}

fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, &dimension| {
        if dimension == 0 {
            return Err(Error::Invalid("empty tensor dimension"));
        }
        total.checked_mul(dimension).ok_or(Error::Invalid("tensor size overflow"))
    })
}

pub(super) fn block_layout(kind: u32) -> Result<(usize, usize)> {
    Ok(match kind {
        0 => (1, 4),
        1 | 30 => (1, 2),
        8 => (32, 34),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        23 => (256, 136),
        U32_KIND | I32_KIND => (1, 4),
        U8_KIND => (1, 1),
        I64_KIND => (1, 8),
        _ => return Err(Error::Unsupported("unsupported weight block format")),
    })
}

fn encoded_size(
    kind: u32,
    count: usize,
) -> Result<usize> {
    let (elements, bytes) = block_layout(kind)?;
    if !count.is_multiple_of(elements) {
        return Err(Error::Invalid("weight dimension is not block aligned"));
    }
    (count / elements).checked_mul(bytes).ok_or(Error::Invalid("encoded weight size overflow"))
}

fn read_safetensors(
    mapping: &Mapping,
    file: usize,
) -> Result<(HashMap<String, Tensor>, BTreeMap<String, Value>)> {
    let size = usize::try_from(u64::from_le_bytes(mapping.bytes(0, 8)?.try_into().unwrap()))
        .map_err(|_| Error::Invalid("weight header size overflow"))?;
    if size == 0 || size > MAX_HEADER {
        return Err(Error::Invalid("invalid weight header size"));
    }
    let header: Value = serde_json::from_slice(mapping.bytes(8, size)?)?;
    let entries = header.as_object().ok_or(Error::Invalid("invalid Safetensors header"))?;
    if entries.len() > MAX_COUNT {
        return Err(Error::Invalid("too many weight tensors"));
    }
    let data_start = 8 + size;
    let mut tensors = HashMap::new();
    let mut spans = Vec::new();
    for (name, value) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value.get("dtype").and_then(Value::as_str).ok_or(Error::Invalid("missing weight dtype"))?;
        let kind = match dtype {
            "F32" => 0,
            "F16" => 1,
            "BF16" => 30,
            "U32" => U32_KIND,
            "U8" => U8_KIND,
            "I32" => I32_KIND,
            "I64" => I64_KIND,
            _ => return Err(Error::Unsupported("unsupported Safetensors dtype")),
        };
        let shape: Vec<usize> = value
            .get("shape")
            .and_then(Value::as_array)
            .ok_or(Error::Invalid("missing weight shape"))?
            .iter()
            .map(|v| v.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(Error::Invalid("invalid weight dimension")))
            .collect::<Result<_>>()?;
        // Text exports can carry an unused 3D vision convolution as a rank-five
        // tensor. Admission validates its complete span without enabling that
        // operation; matrix execution independently requires rank two/three.
        if shape.is_empty() || shape.len() > 8 {
            return Err(Error::Invalid("unsupported weight rank"));
        }
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|v| v.len() == 2)
            .ok_or(Error::Invalid("missing weight offsets"))?;
        let begin =
            offsets[0].as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(Error::Invalid("invalid weight offset"))?;
        let end =
            offsets[1].as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(Error::Invalid("invalid weight offset"))?;
        let count = product(&shape)?;
        let expected = encoded_size(kind, count)?;
        if end.checked_sub(begin) != Some(expected) {
            return Err(Error::Invalid("invalid weight payload size"));
        }
        let offset = data_start.checked_add(begin).ok_or(Error::Invalid("weight offset overflow"))?;
        mapping.bytes(offset, expected)?;
        spans.push((begin, end));
        tensors.insert(
            name.clone(),
            Tensor {
                file,
                offset,
                size: expected,
                shape,
                kind,
                affine: None,
            },
        );
    }
    spans.sort_unstable();
    let mut previous = 0;
    for (begin, end) in spans {
        if begin != previous {
            return Err(Error::Invalid("weight data has gaps or overlapping tensors"));
        }
        previous = end;
    }
    if data_start.checked_add(previous) != Some(mapping.length) {
        return Err(Error::Invalid("unindexed weight payload"));
    }
    Ok((tensors, BTreeMap::new()))
}

struct HeaderReader {
    reader: BufReader<File>,
    consumed: usize,
}
impl HeaderReader {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.consumed = self.consumed.checked_add(N).ok_or(Error::Invalid("GGUF header overflow"))?;
        if self.consumed > MAX_HEADER {
            return Err(Error::Invalid("GGUF header too large"));
        }
        let mut bytes = [0u8; N];
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn count(&mut self) -> Result<usize> {
        let count = usize::try_from(self.u64()?).map_err(|_| Error::Invalid("GGUF count overflow"))?;
        if count > MAX_COUNT {
            return Err(Error::Invalid("GGUF count exceeds bounds"));
        }
        Ok(count)
    }
    fn string(&mut self) -> Result<String> {
        let count = self.count()?;
        self.consumed = self.consumed.checked_add(count).ok_or(Error::Invalid("GGUF header overflow"))?;
        if self.consumed > MAX_HEADER {
            return Err(Error::Invalid("GGUF header too large"));
        }
        let mut bytes = vec![0u8; count];
        self.reader.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| Error::Invalid("invalid GGUF text encoding"))
    }
    fn value(
        &mut self,
        kind: u32,
    ) -> Result<Value> {
        Ok(match kind {
            0 => Value::from(self.bytes::<1>()?[0]),
            1 => Value::from(self.bytes::<1>()?[0] as i8),
            2 => Value::from(u16::from_le_bytes(self.bytes()?)),
            3 => Value::from(i16::from_le_bytes(self.bytes()?)),
            4 => Value::from(self.u32()?),
            5 => Value::from(i32::from_le_bytes(self.bytes()?)),
            6 => {
                let value = f32::from_le_bytes(self.bytes()?);
                if !value.is_finite() {
                    return Err(Error::Invalid("nonfinite GGUF metadata"));
                }
                Value::from(value)
            },
            7 => {
                let value = self.bytes::<1>()?[0];
                if value > 1 {
                    return Err(Error::Invalid("invalid GGUF boolean"));
                }
                Value::from(value == 1)
            },
            8 => Value::from(self.string()?),
            9 => {
                let subtype = self.u32()?;
                if subtype == 9 {
                    return Err(Error::Unsupported("nested GGUF arrays"));
                }
                let count = self.count()?;
                Value::Array((0..count).map(|_| self.value(subtype)).collect::<Result<_>>()?)
            },
            10 => Value::from(self.u64()?),
            11 => Value::from(i64::from_le_bytes(self.bytes()?)),
            12 => {
                let value = f64::from_le_bytes(self.bytes()?);
                if !value.is_finite() {
                    return Err(Error::Invalid("nonfinite GGUF metadata"));
                }
                Value::from(value)
            },
            _ => return Err(Error::Unsupported("unsupported GGUF metadata type")),
        })
    }
}

fn read_gguf(
    source: &File,
    file: usize,
    length: usize,
) -> Result<(HashMap<String, Tensor>, BTreeMap<String, Value>)> {
    let mut reader = HeaderReader {
        reader: BufReader::new(source.try_clone()?),
        consumed: 0,
    };
    if &reader.bytes::<4>()? != b"GGUF" || reader.u32()? != 3 {
        return Err(Error::Unsupported("GGUF version three is required"));
    }
    let tensor_count = reader.count()?;
    let metadata_count = reader.count()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let name = reader.string()?;
        let kind = reader.u32()?;
        let value = reader.value(kind)?;
        if name.is_empty() || metadata.insert(name, value).is_some() {
            return Err(Error::Invalid("duplicate GGUF metadata"));
        }
    }
    let alignment = metadata.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
    if !alignment.is_power_of_two() || alignment > 4096 {
        return Err(Error::Invalid("invalid GGUF alignment"));
    }
    let mut tensors = HashMap::new();
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let rank = reader.u32()? as usize;
        if name.is_empty() || !(1..=4).contains(&rank) {
            return Err(Error::Invalid("invalid GGUF tensor identity or rank"));
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(usize::try_from(reader.u64()?).map_err(|_| Error::Invalid("GGUF dimension overflow"))?);
        }
        let kind = reader.u32()?;
        // Safetensors integer tags are private to this loader. They must never
        // be interpreted as GGML enum values or admitted from a GGUF header.
        if !matches!(kind, 0 | 1 | 30 | 8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 23) {
            return Err(Error::Unsupported("unsupported GGUF tensor type"));
        }
        let offset = usize::try_from(reader.u64()?).map_err(|_| Error::Invalid("GGUF offset overflow"))?;
        let size = encoded_size(kind, product(&shape)?)?;
        encoded_size(kind, shape[0])?;
        if !offset.is_multiple_of(alignment) {
            return Err(Error::Invalid("unaligned GGUF tensor"));
        }
        shape.reverse();
        if tensors
            .insert(
                name,
                Tensor {
                    file,
                    offset,
                    size,
                    shape,
                    kind,
                    affine: None,
                },
            )
            .is_some()
        {
            return Err(Error::Invalid("duplicate GGUF tensor"));
        }
    }
    let position =
        usize::try_from(reader.reader.stream_position()?).map_err(|_| Error::Invalid("GGUF header overflow"))?;
    let start = position
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or(Error::Invalid("GGUF offset overflow"))?;
    let mut spans = Vec::new();
    for tensor in tensors.values_mut() {
        tensor.offset = tensor.offset.checked_add(start).ok_or(Error::Invalid("GGUF offset overflow"))?;
        let end = tensor
            .offset
            .checked_add(tensor.size)
            .filter(|&end| end <= length)
            .ok_or(Error::Invalid("truncated GGUF tensor"))?;
        spans.push((tensor.offset, end));
    }
    spans.sort_unstable();
    let mut end = start;
    for (begin, stop) in spans {
        if begin < end {
            return Err(Error::Invalid("overlapping GGUF tensors"));
        }
        end = stop;
    }
    if start > length {
        return Err(Error::Invalid("truncated GGUF header"));
    }
    Ok((tensors, metadata))
}

fn scalar(
    kind: u32,
    bytes: &[u8],
) -> Result<f32> {
    Ok(match kind {
        0 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
        1 => f16::from_bits(u16::from_le_bytes(bytes[..2].try_into().unwrap())).to_f32(),
        30 => bf16::from_bits(u16::from_le_bytes(bytes[..2].try_into().unwrap())).to_f32(),
        _ => return Err(Error::Invalid("floating scalar dtype is required")),
    })
}

fn decode_values(
    kind: u32,
    bytes: &[u8],
    count: usize,
) -> Result<Vec<f32>> {
    let (block, block_bytes) = block_layout(kind)?;
    if encoded_size(kind, count)? != bytes.len() {
        return Err(Error::Invalid("invalid encoded row size"));
    }
    let mut values = vec![0.0; count];
    for (source, target) in bytes.chunks_exact(block_bytes).zip(values.chunks_exact_mut(block)) {
        decode_block(kind, source, target)?;
    }
    Ok(values)
}

fn half_at(
    bytes: &[u8],
    at: usize,
) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
}
fn word(
    bytes: &[u8],
    at: usize,
) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn scale_min(
    scales: &[u8],
    index: usize,
) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

/// Decode exactly one ggml block. Used as the CPU oracle by Metal tests.
pub(super) fn decode_block(
    kind: u32,
    bytes: &[u8],
    output: &mut [f32],
) -> Result<()> {
    let (count, size) = block_layout(kind)?;
    if bytes.len() != size || output.len() != count {
        return Err(Error::Invalid("invalid quantization block size"));
    }
    match kind {
        0 | 1 | 30 => {
            output[0] = scalar(kind, bytes)?;
        },
        8 => {
            let d = half_at(bytes, 0);
            for i in 0..32 {
                output[i] = d * (bytes[2 + i] as i8) as f32;
            }
        },
        10 => {
            let d = half_at(bytes, 80);
            let minimum = half_at(bytes, 82);
            for n in 0..2 {
                for j in 0..4 {
                    for half in 0..2 {
                        let sc = bytes[n * 8 + j * 2 + half];
                        let at = n * 128 + j * 32 + half * 16;
                        for l in 0..16 {
                            output[at + l] =
                                d * f32::from(sc & 15) * f32::from((bytes[16 + n * 32 + half * 16 + l] >> (2 * j)) & 3)
                                    - minimum * f32::from(sc >> 4);
                        }
                    }
                }
            }
        },
        11 => {
            let d = half_at(bytes, 108);
            let a = word(bytes, 96);
            let b = word(bytes, 100);
            let c = word(bytes, 104);
            let packed = [
                (a & 0x0f0f0f0f) | ((c & 0x03030303) << 4),
                (b & 0x0f0f0f0f) | (((c >> 2) & 0x03030303) << 4),
                ((a >> 4) & 0x0f0f0f0f) | (((c >> 4) & 0x03030303) << 4),
                ((b >> 4) & 0x0f0f0f0f) | (((c >> 6) & 0x03030303) << 4),
            ];
            let mut scales = [0u8; 16];
            for (i, p) in packed.iter().enumerate() {
                scales[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
            for n in 0..2 {
                for j in 0..4 {
                    for half in 0..2 {
                        let sc = i32::from(scales[n * 8 + j * 2 + half]) - 32;
                        let at = n * 128 + j * 32 + half * 16;
                        for l in 0..16 {
                            let hm = bytes[half * 16 + l] & (1 << (n * 4 + j));
                            let q = i32::from((bytes[32 + n * 32 + half * 16 + l] >> (2 * j)) & 3)
                                - if hm != 0 {
                                    0
                                } else {
                                    4
                                };
                            output[at + l] = d * sc as f32 * q as f32;
                        }
                    }
                }
            }
        },
        12 | 13 => {
            let d = half_at(bytes, 0);
            let minimum = half_at(bytes, 2);
            let scales = &bytes[4..16];
            let qstart = if kind == 13 {
                48
            } else {
                16
            };
            for part in 0..4 {
                for half in 0..2 {
                    let (scale, min) = scale_min(scales, part * 2 + half);
                    let at = part * 64 + half * 32;
                    for l in 0..32 {
                        let ql = (bytes[qstart + part * 32 + l] >> (half * 4)) & 15;
                        let high = if kind == 13 && bytes[16 + l] & (1 << (part * 2 + half)) != 0 {
                            16
                        } else {
                            0
                        };
                        output[at + l] = d * f32::from(scale) * f32::from(ql + high) - minimum * f32::from(min);
                    }
                }
            }
        },
        14 => {
            let d = half_at(bytes, 208);
            for n in 0..2 {
                for l in 0..32 {
                    for part in 0..4 {
                        let ql = bytes[n * 64 + (part % 2) * 32 + l];
                        let qh = bytes[128 + n * 32 + l];
                        let low =
                            (ql >> if part >= 2 {
                                4
                            } else {
                                0
                            }) & 15;
                        let high = ((qh >> (part * 2)) & 3) << 4;
                        let q = i32::from(low | high) - 32;
                        let sc = bytes[192 + n * 8 + l / 16 + part * 2] as i8;
                        output[n * 128 + part * 32 + l] = d * f32::from(sc) * q as f32;
                    }
                }
            }
        },
        16 => {
            let d = half_at(bytes, 0);
            for part in 0..8 {
                let at = 2 + part * 8;
                let signs_scale = word(bytes, at + 4);
                let scale = d * (0.5 + (signs_scale >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[usize::from(bytes[at + l])].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((signs_scale >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        output[part * 32 + l * 8 + j] = scale
                            * f32::from(grid[j])
                            * if signs & (1 << j) != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                    }
                }
            }
        },
        17 => {
            let d = half_at(bytes, 0);
            for part in 0..8 {
                let scales = bytes[66 + part];
                for l in 0..4 {
                    let at = 2 + part * 8 + l * 2;
                    let q = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                    let grid = IQ2XS_GRID[usize::from(q & 511)].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
                    let scale = d
                        * (0.5
                            + f32::from(
                                (scales
                                    >> (if l < 2 {
                                        0
                                    } else {
                                        4
                                    }))
                                    & 15,
                            ))
                        * 0.25;
                    for j in 0..8 {
                        output[part * 32 + l * 8 + j] = scale
                            * f32::from(grid[j])
                            * if signs & (1 << j) != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                    }
                }
            }
        },
        18 => {
            let d = half_at(bytes, 0);
            for part in 0..8 {
                let signs_scale = word(bytes, 66 + part * 4);
                let scale = d * (0.5 + (signs_scale >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let at = 2 + part * 8 + l * 2;
                    let grid1 = IQ3XXS_GRID[usize::from(bytes[at])].to_le_bytes();
                    let grid2 = IQ3XXS_GRID[usize::from(bytes[at + 1])].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((signs_scale >> (7 * l)) & 127) as usize];
                    for j in 0..4 {
                        output[part * 32 + l * 8 + j] = scale
                            * f32::from(grid1[j])
                            * if signs & (1 << j) != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                        output[part * 32 + l * 8 + j + 4] = scale
                            * f32::from(grid2[j])
                            * if signs & (1 << (j + 4)) != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                    }
                }
            }
        },
        23 => {
            let d = half_at(bytes, 0);
            let high = u16::from_le_bytes([bytes[2], bytes[3]]);
            for part in 0..8 {
                let low = (bytes[4 + part / 2] >> (4 * (part % 2))) & 15;
                let sc = i32::from(low | (((high >> (part * 2)) as u8 & 3) << 4)) - 32;
                let scale = d * sc as f32;
                for j in 0..16 {
                    let q = bytes[8 + part * 16 + j];
                    output[part * 32 + j] = scale * f32::from(KVALUES_IQ4NL[usize::from(q & 15)]);
                    output[part * 32 + j + 16] = scale * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
                }
            }
        },
        _ => return Err(Error::Unsupported("unsupported floating weight block")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn gguf_type_ids_and_sizes_match_independently_compiled_upstream_headers() {
        // This fixture was produced by compiling the pinned ggml.h enum and
        // ggml-common.h sizeof(block_*) declarations, independently of Stark's
        // decoder and synthetic fixture generators.
        let reference: Value =
            serde_json::from_str(include_str!("../../../unit/fixtures/hybrid_ggml_formats.json")).unwrap();
        assert_eq!(reference["revision"], "8ca4060b7e30e76baaa5670ee1ec1b760b7f8552");
        for format in reference["formats"].as_array().unwrap() {
            let kind = format["kind"].as_u64().unwrap() as u32;
            let expected =
                (format["block_elements"].as_u64().unwrap() as usize, format["block_bytes"].as_u64().unwrap() as usize);
            assert_eq!(block_layout(kind).unwrap(), expected, "upstream format {}", format["name"]);
        }
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "stark-hybrid-weights-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            ));
            fs::create_dir(&directory).unwrap();
            Self(directory)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[stark_test]
    fn all_glm_dynamic_quant_blocks_match_pinned_ggml_c_reference_hashes() {
        // Independently generated by ggml's C dequantize_row_* at revision
        // 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552, clang -ffp-contract=off.
        // Fixtures are deterministic synthetic blocks, not evaluation data.
        let cases: &[(u32, &[usize], &str)] = &[
            (8, &[0], "533450ea45e9c66a46e18fb8aa8547810190fd2cc1b0549fdba579956e411d2b"),
            (10, &[80, 82], "a3b88e197eb3e0acf34a6dc96af7cd602b64c7beb88a0e1b9af4aa36616e25a5"),
            (11, &[108], "033f1230f572d79e5d519654f868452d7f23823f0fcd1ec19f9e84934b2a29c9"),
            (12, &[0, 2], "82d3b0f7dd2ccf1b2c4e1fc4b1f1e01110092877dc483c72069be101cae9d679"),
            (13, &[0, 2], "023cc32c0ce3e38ef5646ca5ee4cb1aa7fc05c3130c0c5f11af691f7e863c72b"),
            (14, &[208], "806661ad6a7cdc976e3aa91cfc8a1f0e52be2407daa14bbd027478550d8b2822"),
            (16, &[0], "41f51728da1a81bc286ec99b07fd883ace31ec28cf41d21cf3237a2560914f61"),
            (17, &[0], "052aee89b2aac55f6944ad7341ad1fe94443d0e271f79232d70f9eac177fb6ea"),
            (18, &[0], "fd07b83b9cc10065cedbdd3ecc9394ecdcc0063c465de0c9095925b8c7c329fb"),
            (23, &[0], "3f1a6953fe6368e2971dc92e501e5f7523252432bebfedc60ab6acf295cb64fb"),
        ];
        for &(kind, scales, expected) in cases {
            let (count, size) = block_layout(kind).unwrap();
            let mut bytes: Vec<u8> = (0..size).map(|i| ((i * 73 + 41) % 256) as u8).collect();
            for (i, &at) in scales.iter().enumerate() {
                bytes[at..at + 2].copy_from_slice(
                    &(if i == 0 {
                        0x3400u16
                    } else {
                        0x3000u16
                    })
                    .to_le_bytes(),
                );
            }
            let mut output = vec![0.0; count];
            decode_block(kind, &bytes, &mut output).unwrap();
            let mut digest = Sha256::new();
            for value in output {
                digest.update(value.to_le_bytes());
            }
            assert_eq!(format!("{:x}", digest.finalize()), expected, "format {kind}");
            assert!(decode_block(kind, &bytes[..size - 1], &mut vec![0.0; count]).is_err());
        }
    }

    fn safetensors(
        path: &Path,
        entries: &[(&str, &str, Vec<usize>, Vec<u8>)],
    ) {
        let mut metadata = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, payload) in entries {
            let start = data.len();
            data.extend_from_slice(payload);
            metadata.insert(
                (*name).to_string(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&metadata).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        fs::write(path, bytes).unwrap();
    }

    #[cfg(feature = "metal")]
    #[stark_test]
    fn mapped_partial_page_retains_its_owner_after_store_drop() {
        let fixture = Fixture::new();
        let path = fixture.0.join("partial.bin");
        let mut bytes = vec![9u8; 3];
        bytes.extend([1.0f32, 2.0, -1.0, 0.5, -0.25, 4.0].iter().flat_map(|value| value.to_le_bytes()));
        fs::write(&path, bytes).unwrap();
        let mapping = Arc::new(Mapping::new(&path).unwrap());
        let weak = Arc::downgrade(&mapping);
        let codes = mapping.metal_slice(3, 24).unwrap();
        assert!(mapping.metal_slice(3, 25).is_err());
        drop(mapping);
        assert!(weak.upgrade().is_some());
        let inputs = [1.0f32, 0.5, -2.0];
        let operation = super::super::weight_metal::MappedOperation {
            matrices: vec![super::super::weight_metal::MappedMatrix {
                codes,
                planes: None,
            }],
            inputs: &inputs,
            rows: 2,
            columns: 3,
            rows_per_batch: 0,
            format: super::super::weight_metal::MatrixFormat::Packed(0),
        };
        let outputs = super::super::weight_metal::mapped_many(std::slice::from_ref(&operation)).unwrap();
        assert_eq!(outputs, vec![vec![4.0, -7.625]]);
        drop(operation);
        for _ in 0..100 {
            if weak.upgrade().is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(weak.upgrade().is_none(), "completed mapped resources retained their allocation");
    }

    #[cfg(feature = "metal")]
    #[stark_test]
    fn repeated_mapped_submissions_release_registered_ranges() {
        let fixture = Fixture::new();
        let path = fixture.0.join("model.safetensors");
        safetensors(&path, &[("matrix", "F32", vec![1024, 1024], vec![0; 4 * 1024 * 1024])]);
        fs::write(fixture.0.join("config.json"), b"{}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let inputs = vec![1.0; 1024];
        assert_eq!(weights.matvec("matrix", &inputs).unwrap().len(), 1024);
        let before = super::super::weight_metal::allocated_bytes().unwrap();
        for _ in 0..32 {
            assert_eq!(weights.matvec("matrix", &inputs).unwrap().len(), 1024);
        }
        let after = super::super::weight_metal::allocated_bytes().unwrap();
        // The source mapping stays alive, but completed Metal views and their
        // input/output allocations must not accumulate one 4 MiB resource per
        // call. Allow modest driver scratch/reuse after the warm submission.
        assert!(after <= before + 16 * 1024 * 1024);
    }

    #[cfg(feature = "metal")]
    #[stark_test]
    fn static_view_cache_reuses_weights_excludes_experts_and_drops_ownership() {
        struct DisableProfile;
        impl Drop for DisableProfile {
            fn drop(&mut self) {
                super::super::profile::enable(false);
            }
        }
        let _disable = DisableProfile;
        super::super::profile::reset();
        super::super::profile::enable(true);
        let fixture = Fixture::new();
        let data = |value: f32, count: usize| (0..count).flat_map(|_| value.to_le_bytes()).collect::<Vec<_>>();
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("matrix", "F32", vec![2, 32], data(1.0, 64)),
                ("experts", "F32", vec![3, 2, 32], data(2.0, 192)),
                ("block.attn_k_b.weight", "F32", vec![3, 2, 32], data(3.0, 192)),
            ],
        );
        fs::write(fixture.0.join("config.json"), b"{}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let weak = Arc::downgrade(&weights.files[0]);
        let input = vec![1.0; 32];
        assert_eq!(weights.matvec("matrix", &input).unwrap(), vec![32.0; 2]);
        assert_eq!(weights.matvec("matrix", &input).unwrap(), vec![32.0; 2]);
        assert_eq!(weights.static_views.lock().unwrap().entries.len(), 1);
        weights.expert_matvec("experts", 1, &input).unwrap();
        weights.selected_matvec("experts", &[2, 0], &input.repeat(2)).unwrap();
        weights.batched_matvec("experts", &input.repeat(3)).unwrap();
        assert_eq!(weights.static_views.lock().unwrap().entries.len(), 1);
        weights.batched_matvec("block.attn_k_b.weight", &input.repeat(3)).unwrap();
        {
            let cache = weights.static_views.lock().unwrap();
            assert_eq!(cache.entries.len(), 2);
            assert_eq!(
                cache.registered_bytes,
                cache.entries.values().map(|view| view.registered_bytes()).sum::<usize>()
            );
            assert!(cache.registered_bytes <= cache.limit);
        }
        let counters = super::super::profile::snapshot();
        assert_eq!(counters["native.metal.static_cache.hit"].calls, 1);
        assert_eq!(counters["native.metal.static_cache.miss"].calls, 2);
        drop(weights);
        for _ in 0..100 {
            if weak.upgrade().is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(weak.upgrade().is_none(), "dropped static cache retained its source mapping");
    }

    #[cfg(feature = "metal")]
    #[stark_test]
    fn whole_bank_views_reuse_nonconsecutive_experts_and_release_mapping() {
        let fixture = Fixture::new();
        let data: Vec<_> = (1..=4).flat_map(|expert| (expert as f32).to_le_bytes().repeat(64)).collect();
        safetensors(&fixture.0.join("model.safetensors"), &[("experts", "F32", vec![4, 2, 32], data)]);
        fs::write(fixture.0.join("config.json"), b"{}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let weak = Arc::downgrade(&weights.files[0]);
        let input = vec![1.0; 32];
        assert!(weights.indexed_banks_admitted());
        assert_eq!(weights.expert_matvec("experts", 2, &input).unwrap(), vec![96.0; 2]);
        assert_eq!(
            weights.selected_matvec("experts", &[3, 1, 3], &input.repeat(3)).unwrap(),
            vec![128.0, 128.0, 64.0, 64.0, 128.0, 128.0]
        );
        assert!(weights.selected_matvec("experts", &[4], &input).is_err());
        assert!(weights.static_views.lock().unwrap().entries.is_empty());
        {
            let cache = weights.bank_views.lock().unwrap();
            assert_eq!(cache.entries.len(), 1);
            assert_eq!(cache.order.len(), 1);
            assert_eq!(cache.registered_bytes, cache.limit);
            let bank = cache.entries.values().next().unwrap();
            assert!(bank.subslice(0, 0).is_err());
            assert!(bank.subslice(1024, 1).is_err());
            assert!(bank.subslice(usize::MAX, 1).is_err());
        }
        // Exercise the same numerical path when complete-bank retention was
        // rejected by the memory-budget admission check.
        *weights.bank_views.lock().unwrap() = StaticViewCache::new(0);
        assert!(!weights.indexed_banks_admitted());
        assert_eq!(
            weights.selected_matvec("experts", &[3, 1, 3], &input.repeat(3)).unwrap(),
            vec![128.0, 128.0, 64.0, 64.0, 128.0, 128.0]
        );
        assert!(weights.bank_views.lock().unwrap().entries.is_empty());
        drop(weights);
        for _ in 0..100 {
            if weak.upgrade().is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(weak.upgrade().is_none(), "whole-bank views retained their source mapping after drop");
    }

    #[cfg(feature = "metal")]
    #[stark_test]
    fn static_view_cache_bounds_rounded_spans_and_preserves_evicted_active_views() {
        let fixture = Fixture::new();
        let data = |value: f32| value.to_le_bytes().repeat(32);
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("a", "F32", vec![1, 32], data(1.0)),
                ("b", "F32", vec![1, 32], data(2.0)),
                ("c", "F32", vec![1, 32], data(3.0)),
            ],
        );
        fs::write(fixture.0.join("config.json"), b"{}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let tensor = weights.tensor("a").unwrap();
        let page_span = weights.files[tensor.file].metal_span(tensor.offset, tensor.size).unwrap().1;
        // This override exists only in the test; production has a fixed 8 GiB
        // retained-cache bound. Logical spans are only 128 bytes each.
        *weights.static_views.lock().unwrap() = StaticViewCache::new(page_span);
        let inputs = vec![1.0; 32];
        let outputs = weights.matvec_many(&[("a", &inputs), ("b", &inputs)]).unwrap();
        assert_eq!(outputs, vec![vec![32.0], vec![64.0]]);
        let b = weights.tensor("b").unwrap();
        {
            let cache = weights.static_views.lock().unwrap();
            assert_eq!(cache.entries.len(), 1);
            assert_eq!(cache.order.len(), 1);
            assert_eq!(cache.registered_bytes, page_span);
            assert!(cache.entries.contains_key(&StaticViewKey {
                file: b.file,
                offset: b.offset,
                length: b.size
            }));
        }
        let invalid_offset = weights.files[0].length;
        assert!(weights.metal_view(0, invalid_offset, 1, true).is_err());
        assert_eq!(weights.static_views.lock().unwrap().registered_bytes, page_span);
        assert_eq!(weights.static_views.lock().unwrap().entries.len(), 1);
        // An over-cap view stays transient instead of evicting everything and
        // entering an insertion loop that can never satisfy the bound.
        *weights.static_views.lock().unwrap() = StaticViewCache::new(page_span - 1);
        assert_eq!(weights.matvec("c", &inputs).unwrap(), vec![96.0]);
        let cache = weights.static_views.lock().unwrap();
        assert_eq!(cache.registered_bytes, 0);
        assert!(cache.entries.is_empty() && cache.order.is_empty());
        drop(cache);
        *weights.static_views.lock().unwrap() = StaticViewCache::new(0);
        assert_eq!(weights.matvec("a", &inputs).unwrap(), vec![32.0]);
        assert!(weights.static_views.lock().unwrap().entries.is_empty());
    }

    #[stark_test]
    fn byte_packed_flint_planes_preserve_words_and_embedding_width() {
        let fixture = Fixture::new();
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", "U8", vec![2, 32], vec![0x42; 64]),
                (
                    "model.embed_tokens.scales",
                    "F32",
                    vec![2, 1],
                    [0.25f32; 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                (
                    "model.embed_tokens.biases",
                    "F32",
                    vec![2, 1],
                    [0.5f32; 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                ("byte.weight", "U8", vec![2, 16], vec![0x42; 32]),
                ("word.weight", "U32", vec![2, 4], vec![0x42; 32]),
                ("byte.scales", "F32", vec![2, 1], [0.25f32; 2].iter().flat_map(|v| v.to_le_bytes()).collect()),
                ("word.scales", "F32", vec![2, 1], [0.25f32; 2].iter().flat_map(|v| v.to_le_bytes()).collect()),
                ("byte.biases", "F32", vec![2, 1], [0.5f32; 2].iter().flat_map(|v| v.to_le_bytes()).collect()),
                ("word.biases", "F32", vec![2, 1], [0.5f32; 2].iter().flat_map(|v| v.to_le_bytes()).collect()),
            ],
        );
        fs::write(
            fixture.0.join("config.json"),
            br#"{"model_type":"flint","quantization":{"bits":4,"group_size":32},"mlx_export":{"embedding_bits":8}}"#,
        )
        .unwrap();
        // MLX may leave the JSON header unpadded. Packed kernels load bytes,
        // so a word's absolute file address need not be aligned to four bytes.
        let path = fixture.0.join("model.safetensors");
        let mut bytes = fs::read(&path).unwrap();
        let header_bytes = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        bytes.insert(8 + header_bytes, b' ');
        bytes[..8].copy_from_slice(&((header_bytes + 1) as u64).to_le_bytes());
        fs::write(path, bytes).unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        assert_eq!(weights.shape("model.embed_tokens.weight").unwrap(), [2, 32]);
        assert_eq!(weights.row("model.embed_tokens.weight", 1).unwrap(), vec![17.0; 32]);
        assert_eq!(weights.shape("byte.weight").unwrap(), [2, 32]);
        assert_eq!(weights.row("byte.weight", 1).unwrap(), weights.row("word.weight", 1).unwrap());
        let input: Vec<_> = (0..32).map(|i| i as f32 * 0.01).collect();
        assert_eq!(weights.matvec("byte.weight", &input).unwrap(), weights.matvec("word.weight", &input).unwrap());
        // A missing embedding override must reject the inconsistent plane width.
        fs::write(
            fixture.0.join("config.json"),
            br#"{"model_type":"flint","quantization":{"bits":4,"group_size":32}}"#,
        )
        .unwrap();
        assert!(Weights::open(&fixture.0).is_err());
    }

    #[stark_test]
    fn independent_matrix_jobs_preserve_mixed_formats_and_distinct_inputs() {
        let fixture = Fixture::new();
        let path = fixture.0.join("model.safetensors");
        let a = [1.0f32, 2.0, -1.0, 0.5, -0.25, 4.0];
        let b = [0.5f32, 1.0, -2.0, 3.0, 4.0, -1.0];
        safetensors(
            &path,
            &[
                ("a", "F32", vec![2, 3], a.iter().flat_map(|value| value.to_le_bytes()).collect()),
                (
                    "b",
                    "BF16",
                    vec![3, 2],
                    b.iter().flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes()).collect(),
                ),
                ("quant.weight", "U32", vec![1, 4], vec![0x21; 16]),
                ("quant.scales", "F32", vec![1, 1], 0.125f32.to_le_bytes().to_vec()),
                ("quant.biases", "F32", vec![1, 1], (-0.25f32).to_le_bytes().to_vec()),
            ],
        );
        fs::write(fixture.0.join("config.json"), b"{\"quantization\":{\"bits\":4,\"group_size\":32}}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let input_a = [1.0f32, 0.5, -2.0];
        let input_b = [2.0f32, -1.0];
        let input_quant: Vec<_> = (0..32).map(|index| index as f32 * 0.125 - 1.0).collect();
        let jobs: &[(&str, &[f32])] = &[("a", &input_a), ("b", &input_b), ("quant.weight", &input_quant)];
        let outputs = weights.matvec_many(jobs).unwrap();
        assert_eq!(outputs[0], vec![4.0, -7.625]);
        assert_eq!(outputs[1], vec![0.0, -7.0, 9.0]);
        let quant_expected: f32 = input_quant
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input
                    * if index % 2 == 0 {
                        -0.125
                    } else {
                        0.0
                    }
            })
            .sum();
        assert!((outputs[2][0] - quant_expected).abs() < 1e-6);
        for (index, &(name, inputs)) in jobs.iter().enumerate() {
            assert_eq!(outputs[index], weights.matvec(name, inputs).unwrap());
        }
        assert!(weights.matvec_many(&[]).is_err());
        assert!(weights.matvec_many(&[("a", &input_b)]).is_err());
    }

    #[stark_test]
    fn safetensors_admits_unused_rank_five_vision_and_bounds_other_ranks_and_sizes() {
        let fixture = Fixture::new();
        let path = fixture.0.join("model.safetensors");
        fs::write(fixture.0.join("config.json"), b"{}").unwrap();
        safetensors(
            &path,
            &[
                (
                    "language_model.matrix.weight",
                    "F32",
                    vec![2, 2],
                    [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|value| value.to_le_bytes()).collect(),
                ),
                ("vision_tower.patch_embed.proj.weight", "BF16", vec![2, 3, 2, 2, 2], vec![0; 96]),
                ("unused.rank_eight", "BF16", vec![1; 8], vec![0; 2]),
            ],
        );
        {
            let weights = Weights::open(&fixture.0).unwrap();
            assert_eq!(weights.shape("vision_tower.patch_embed.proj.weight").unwrap(), &[2, 3, 2, 2, 2]);
            assert_eq!(weights.shape("unused.rank_eight").unwrap(), &[1; 8]);
            assert_eq!(weights.matvec("language_model.matrix.weight", &[1.0, 2.0]).unwrap(), [5.0, 11.0]);
            assert!(weights.matvec("vision_tower.patch_embed.proj.weight", &[0.0; 2]).is_err());
            assert!(weights.expert_matvec("vision_tower.patch_embed.proj.weight", 0, &[0.0; 2]).is_err());
        }
        let original_size = fs::metadata(&path).unwrap().len();
        File::options().write(true).open(&path).unwrap().set_len(original_size - 1).unwrap();
        assert!(Weights::open(&fixture.0).is_err());
        safetensors(&path, &[("unused.rank_nine", "BF16", vec![1; 9], vec![0; 2])]);
        assert!(Weights::open(&fixture.0).is_err());
        safetensors(&path, &[("overflow", "BF16", vec![usize::MAX, 2], vec![0; 2])]);
        assert!(matches!(Weights::open(&fixture.0), Err(Error::Invalid("tensor size overflow"))));
        safetensors(&path, &[("empty_dimension", "BF16", vec![2, 0, 2, 2, 2], vec![])]);
        assert!(Weights::open(&fixture.0).is_err());
    }

    #[stark_test]
    fn mlx_planes_keep_logical_rows_and_integer_hash_constants() {
        let fixture = Fixture::new();
        let integers = [(1i64 << 40) + 123, -9i64];
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("matrix.weight", "U32", vec![2, 4], vec![0x21; 32]),
                ("matrix.scales", "F32", vec![2, 1], [0.5f32, 0.25].iter().flat_map(|x| x.to_le_bytes()).collect()),
                ("matrix.biases", "F32", vec![2, 1], [-1.0f32, 1.0].iter().flat_map(|x| x.to_le_bytes()).collect()),
                ("hashes", "I64", vec![2], integers.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ],
        );
        fs::write(fixture.0.join("config.json"), b"{\"quantization\":{\"bits\":4,\"group_size\":32}}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        assert_eq!(weights.shape("matrix.weight").unwrap(), &[2, 32]);
        let first = weights.row("matrix.weight", 0).unwrap();
        let second = weights.row("matrix.weight", 1).unwrap();
        for i in 0..32 {
            assert_eq!(
                first[i],
                if i.is_multiple_of(2) {
                    -0.5
                } else {
                    0.0
                }
            );
            assert_eq!(
                second[i],
                if i.is_multiple_of(2) {
                    1.25
                } else {
                    1.5
                }
            );
        }
        assert_eq!(weights.integers("hashes").unwrap(), integers);
        assert!(weights.row("matrix.weight", 2).is_err());
        assert!(weights.vector("matrix.weight").is_err());
        assert!(weights.expert_matvec("matrix.weight", 0, &[0.0; 32]).is_err());
    }

    #[stark_test]
    fn mlx_wrapped_names_use_matching_per_module_precision() {
        // The pinned Qwen export retains language_model.* in tensor names,
        // while config.quantization uses unwrapped module names.
        let fixture = Fixture::new();
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("language_model.matrix.weight", "U32", vec![1, 8], vec![0x21; 32]),
                ("language_model.matrix.scales", "F32", vec![1, 1], 0.5f32.to_le_bytes().to_vec()),
                ("language_model.matrix.biases", "F32", vec![1, 1], 0.0f32.to_le_bytes().to_vec()),
            ],
        );
        fs::write(
            fixture.0.join("config.json"),
            b"{\"quantization\":{\"bits\":4,\"group_size\":64,\"matrix\":{\"bits\":8,\"group_size\":32}}}",
        )
        .unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        assert_eq!(weights.shape("language_model.matrix.weight").unwrap(), &[1, 32]);
        assert_eq!(weights.row("language_model.matrix.weight", 0).unwrap(), vec![16.5; 32]);
    }

    #[stark_test]
    fn batched_head_matvec_preserves_each_input_and_plane() {
        let fixture = Fixture::new();
        let full: Vec<f32> = (0..3 * 2 * 32).map(|i| 0.125 * ((i % 11) as f32 - 5.0)).collect();
        let scales: Vec<f32> = (0..6).map(|i| 0.125 * (i + 1) as f32).collect();
        let biases: Vec<f32> = (0..6).map(|i| -0.25 * i as f32).collect();
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("full.weight", "F32", vec![3, 2, 32], full.iter().flat_map(|x| x.to_le_bytes()).collect()),
                ("affine.weight", "U32", vec![3, 2, 4], vec![0x42; 3 * 2 * 16]),
                ("affine.scales", "F32", vec![3, 2, 1], scales.iter().flat_map(|x| x.to_le_bytes()).collect()),
                ("affine.biases", "F32", vec![3, 2, 1], biases.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ],
        );
        fs::write(fixture.0.join("config.json"), b"{\"quantization\":{\"bits\":4,\"group_size\":32}}").unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let inputs: Vec<f32> = (0..3 * 32).map(|i| 0.25 * ((i % 7) as f32 - 3.0)).collect();
        for name in ["full.weight", "affine.weight"] {
            let actual = weights.batched_matvec(name, &inputs).unwrap();
            assert_eq!(actual.len(), 6);
            for batch in 0..3 {
                for row in 0..2 {
                    let expected: f32 = (0..32)
                        .map(|column: usize| {
                            let weight = if name == "full.weight" {
                                full[(batch * 2 + row) * 32 + column]
                            } else {
                                scales[batch * 2 + row]
                                    * if column.is_multiple_of(2) {
                                        2.0
                                    } else {
                                        4.0
                                    }
                                    + biases[batch * 2 + row]
                            };
                            weight * inputs[batch * 32 + column]
                        })
                        .sum();
                    assert!((actual[batch * 2 + row] - expected).abs() < 1e-5);
                }
            }
            assert!(weights.batched_matvec(name, &inputs[..32]).is_err());
        }
        let mut invalid = inputs;
        invalid[32] = f32::NAN;
        assert!(weights.batched_matvec("full.weight", &invalid).is_err());
    }

    #[stark_test]
    fn selected_nonconsecutive_affine_experts_keep_codes_scales_and_inputs() {
        let fixture = Fixture::new();
        let scales: Vec<f32> = (0..8).map(|i| 0.125 * (i + 1) as f32).collect();
        let biases: Vec<f32> = (0..8).map(|i| -0.25 * i as f32).collect();
        safetensors(
            &fixture.0.join("model.safetensors"),
            &[
                ("four.weight", "U32", vec![4, 2, 4], vec![0x42; 4 * 2 * 16]),
                ("four.scales", "F32", vec![4, 2, 1], scales.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ("four.biases", "F32", vec![4, 2, 1], biases.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ("eight.weight", "U32", vec![4, 2, 8], vec![0x42; 4 * 2 * 32]),
                (
                    "eight.scales",
                    "BF16",
                    vec![4, 2, 1],
                    scales.iter().flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes()).collect(),
                ),
                (
                    "eight.biases",
                    "BF16",
                    vec![4, 2, 1],
                    biases.iter().flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes()).collect(),
                ),
            ],
        );
        fs::write(
            fixture.0.join("config.json"),
            b"{\"quantization\":{\"bits\":4,\"group_size\":32,\"eight\":{\"bits\":8,\"group_size\":32}}}",
        )
        .unwrap();
        let weights = Weights::open(&fixture.0).unwrap();
        let experts = [3, 1];
        let inputs: Vec<f32> = (0..64).map(|i| 0.25 * ((i % 11) as f32 - 5.0)).collect();
        for name in ["four.weight", "eight.weight"] {
            let actual = weights.selected_matvec(name, &experts, &inputs).unwrap();
            assert_eq!(actual.len(), 4);
            for (selected, &expert) in experts.iter().enumerate() {
                for row in 0..2 {
                    let expected: f32 = (0..32)
                        .map(|column: usize| {
                            let q = if name == "eight.weight" {
                                66.0
                            } else if column.is_multiple_of(2) {
                                2.0
                            } else {
                                4.0
                            };
                            (scales[expert * 2 + row] * q + biases[expert * 2 + row]) * inputs[selected * 32 + column]
                        })
                        .sum();
                    assert!((actual[selected * 2 + row] - expected).abs() < 1e-5);
                }
            }
            assert!(weights.selected_matvec(name, &[], &[]).is_err());
            assert!(weights.selected_matvec(name, &[4], &inputs[..32]).is_err());
            assert!(weights.selected_matvec(name, &experts, &inputs[..32]).is_err());
        }
    }

    #[stark_test]
    fn selected_nonconsecutive_gguf_experts_match_scalar_for_all_checkpoint_formats() {
        fn string(
            bytes: &mut Vec<u8>,
            value: &str,
        ) {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        for kind in [8, 10, 11, 12, 13, 14, 16, 17, 18, 23] {
            let fixture = Fixture::new();
            let (block, size) = block_layout(kind).unwrap();
            let mut payload = Vec::new();
            // Three rows make IQ2_XXS expert strides 198 bytes, deliberately
            // misaligned for a four-byte Metal buffer binding offset.
            for row in 0..12 {
                for number in 0..256 / block {
                    let mut bytes: Vec<u8> =
                        (0..size).map(|i| ((i * 73 + row * 19 + number * 7) % 256) as u8).collect();
                    let offsets: &[usize] = match kind {
                        8 | 16 | 17 | 18 | 23 => &[0],
                        10 => &[80, 82],
                        11 => &[108],
                        12 | 13 => &[0, 2],
                        14 => &[208],
                        _ => unreachable!(),
                    };
                    for (i, &at) in offsets.iter().enumerate() {
                        bytes[at..at + 2].copy_from_slice(
                            &f16::from_f32(if i == 0 {
                                0.125 * (row + 1) as f32
                            } else {
                                0.0625
                            })
                            .to_bits()
                            .to_le_bytes(),
                        );
                    }
                    payload.extend(bytes);
                }
            }
            let mut header = b"GGUF".to_vec();
            header.extend(3u32.to_le_bytes());
            header.extend(1u64.to_le_bytes());
            header.extend(0u64.to_le_bytes());
            string(&mut header, "experts");
            header.extend(3u32.to_le_bytes());
            for dimension in [256u64, 3, 4] {
                header.extend(dimension.to_le_bytes());
            }
            header.extend(kind.to_le_bytes());
            header.extend(0u64.to_le_bytes());
            while !header.len().is_multiple_of(32) {
                header.push(0);
            }
            header.extend(&payload);
            fs::write(fixture.0.join("model.gguf"), header).unwrap();
            let weights = Weights::open(&fixture.0).unwrap();
            let experts = [3, 1];
            let inputs: Vec<f32> = (0..512).map(|i| (i as f32 * 0.17).sin()).collect();
            let actual = weights.selected_matvec("experts", &experts, &inputs).unwrap();
            let row_bytes = 256 / block * size;
            for (selected, &expert) in experts.iter().enumerate() {
                for row in 0..3 {
                    let at = (expert * 3 + row) * row_bytes;
                    let decoded = decode_values(kind, &payload[at..at + row_bytes], 256).unwrap();
                    let products: Vec<f64> = decoded
                        .iter()
                        .zip(&inputs[selected * 256..(selected + 1) * 256])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .collect();
                    let expected: f64 = products.iter().sum();
                    let absolute: f64 = products.iter().map(|x| x.abs()).sum();
                    assert!(
                        (f64::from(actual[selected * 3 + row]) - expected).abs() <= 2e-5 * absolute.max(1.0),
                        "selected format {kind}"
                    );
                }
            }
        }
    }

    fn gguf_file(
        path: &Path,
        splits: u16,
        truncate: bool,
    ) {
        fn string(
            bytes: &mut Vec<u8>,
            value: &str,
        ) {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        string(&mut bytes, "split.no");
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        string(&mut bytes, "split.count");
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(splits.to_le_bytes());
        string(&mut bytes, "matrix");
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(32u64.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        for scale in [0x3c00u16, 0x3800u16] {
            bytes.extend(scale.to_le_bytes());
            bytes.extend((0..32).map(|x| x as u8));
        }
        if truncate {
            bytes.pop();
        }
        fs::write(path, bytes).unwrap();
    }

    #[stark_test]
    fn gguf_rows_validate_blocks_truncation_and_split_completeness() {
        let fixture = Fixture::new();
        let path = fixture.0.join("model.gguf");
        gguf_file(&path, 1, false);
        let weights = Weights::open(&fixture.0).unwrap();
        assert_eq!(weights.shape("matrix").unwrap(), &[2, 32]);
        assert_eq!(weights.row("matrix", 0).unwrap(), (0..32).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(weights.row("matrix", 1).unwrap(), (0..32).map(|x| 0.5 * x as f32).collect::<Vec<_>>());
        drop(weights);
        gguf_file(&path, 2, false);
        assert!(Weights::open(&fixture.0).is_err());
        gguf_file(&path, 1, true);
        assert!(Weights::open(&fixture.0).is_err());
    }

    #[stark_test]
    fn gguf_rejects_safetensors_private_integer_tags() {
        for kind in [U32_KIND, I64_KIND, I32_KIND, U8_KIND] {
            let fixture = Fixture::new();
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.push(b't');
            bytes.extend(1u32.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.extend(kind.to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            while !bytes.len().is_multiple_of(32) {
                bytes.push(0);
            }
            bytes.extend(vec![
                0u8;
                if kind == I64_KIND {
                    8
                } else {
                    4
                }
            ]);
            fs::write(fixture.0.join("model.gguf"), bytes).unwrap();
            assert!(matches!(Weights::open(&fixture.0), Err(Error::Unsupported("unsupported GGUF tensor type"))));
        }
    }
}

pub(super) const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

pub(super) const IQ2XS_GRID: [u64; 512] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b192b19,
    0x080808082b2b0808,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x0808081908192b2b,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x08080819192b0808,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b082b2b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908190819,
    0x0808191908191908,
    0x08081919082b0808,
    0x0808191919080819,
    0x0808191919081908,
    0x0808191919190808,
    0x08081919192b0819,
    0x080819192b080808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b08190808,
    0x0808192b082b192b,
    0x0808192b19080808,
    0x0808192b1908082b,
    0x0808192b2b081908,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808082b2b,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b0819192b08,
    0x08082b082b080808,
    0x08082b082b2b0808,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b1919080808,
    0x08082b192b080819,
    0x08082b192b082b19,
    0x08082b2b08080808,
    0x08082b2b082b0808,
    0x08082b2b082b2b08,
    0x08082b2b2b19192b,
    0x08082b2b2b2b0808,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x08190808192b0808,
    0x08190808192b2b2b,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b190808,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908190819,
    0x0819081908191908,
    0x08190819082b0808,
    0x0819081919080819,
    0x0819081919081908,
    0x0819081919190808,
    0x081908192b080808,
    0x081908192b191908,
    0x081908192b19192b,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b0808192b,
    0x0819082b08190808,
    0x0819082b19080808,
    0x0819082b192b0808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x08191908082b0808,
    0x0819190819080819,
    0x0819190819081908,
    0x0819190819082b19,
    0x0819190819190808,
    0x08191908192b1908,
    0x081919082b080808,
    0x0819191908080819,
    0x0819191908081908,
    0x0819191908190808,
    0x0819191919080808,
    0x0819192b08080808,
    0x0819192b08191908,
    0x0819192b19082b19,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b080819082b,
    0x08192b0819080808,
    0x08192b0819191908,
    0x08192b082b08192b,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b19192b192b,
    0x08192b2b19190819,
    0x08192b2b2b2b2b19,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808082b2b,
    0x082b080808190819,
    0x082b080808191908,
    0x082b0808082b0808,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08082b080808,
    0x082b08082b2b0808,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b081919080808,
    0x082b081919082b08,
    0x082b0819192b1919,
    0x082b082b08080808,
    0x082b082b082b082b,
    0x082b082b2b080808,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b1908082b2b19,
    0x082b190819080808,
    0x082b191908080808,
    0x082b191919080819,
    0x082b19191919082b,
    0x082b19192b192b19,
    0x082b192b08080819,
    0x082b192b08192b2b,
    0x082b192b2b2b192b,
    0x082b2b0808080808,
    0x082b2b0808082b08,
    0x082b2b0808082b2b,
    0x082b2b08082b0808,
    0x082b2b0819191919,
    0x082b2b082b082b08,
    0x082b2b082b2b082b,
    0x082b2b19192b2b08,
    0x082b2b192b190808,
    0x082b2b2b08082b08,
    0x082b2b2b082b0808,
    0x082b2b2b2b08082b,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x19080808192b0808,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x19080819082b0808,
    0x1908081919080819,
    0x1908081919081908,
    0x1908081919190808,
    0x190808192b080808,
    0x190808192b081919,
    0x190808192b2b082b,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b082b2b19,
    0x1908082b19080808,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808190819,
    0x1908190808191908,
    0x1908190808192b19,
    0x19081908082b0808,
    0x1908190819080819,
    0x1908190819081908,
    0x1908190819190808,
    0x190819082b080808,
    0x190819082b191908,
    0x1908191908080819,
    0x1908191908081908,
    0x1908191908190808,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819192b192b2b,
    0x1908192b08080808,
    0x1908192b08082b2b,
    0x1908192b19081908,
    0x1908192b19190808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b0819191908,
    0x19082b08192b082b,
    0x19082b1908080808,
    0x19082b1908190819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b19192b2b19,
    0x19082b2b08081908,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x19190808082b0808,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x1919080819190808,
    0x191908082b080808,
    0x1919081908080819,
    0x1919081908081908,
    0x1919081908190808,
    0x1919081908191919,
    0x1919081919080808,
    0x191908191908082b,
    0x1919082b08080808,
    0x1919082b19081908,
    0x1919082b2b2b2b2b,
    0x1919190808080819,
    0x1919190808081908,
    0x1919190808190808,
    0x19191908082b0819,
    0x1919190819080808,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b2b0819,
    0x1919191908080808,
    0x1919191908082b08,
    0x191919192b080808,
    0x191919192b082b08,
    0x1919192b082b0819,
    0x1919192b192b2b08,
    0x1919192b2b2b0819,
    0x19192b0808080808,
    0x19192b0808191908,
    0x19192b0819080819,
    0x19192b0819190808,
    0x19192b082b192b19,
    0x19192b1908192b2b,
    0x19192b1919080808,
    0x19192b191908082b,
    0x19192b2b2b081919,
    0x192b080808080819,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b080819191908,
    0x192b0808192b082b,
    0x192b08082b08192b,
    0x192b08082b2b2b19,
    0x192b081908080808,
    0x192b082b082b1908,
    0x192b082b19082b2b,
    0x192b082b2b19082b,
    0x192b190808080808,
    0x192b19080819192b,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919081919,
    0x192b19192b2b1908,
    0x192b2b0808080819,
    0x192b2b08192b2b2b,
    0x192b2b19082b1919,
    0x192b2b2b0808192b,
    0x192b2b2b19191908,
    0x192b2b2b192b082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b080808082b0808,
    0x2b080808082b2b2b,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808082b080808,
    0x2b0808082b08082b,
    0x2b0808082b2b2b08,
    0x2b0808082b2b2b2b,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b0808190808192b,
    0x2b08081908190808,
    0x2b08081919080808,
    0x2b08081919190819,
    0x2b08081919192b19,
    0x2b08082b08080808,
    0x2b08082b082b0808,
    0x2b08082b2b080808,
    0x2b08082b2b08082b,
    0x2b08082b2b2b0808,
    0x2b08082b2b2b2b08,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b0819082b082b19,
    0x2b08191908080808,
    0x2b08191919081908,
    0x2b0819192b2b1919,
    0x2b08192b08192b08,
    0x2b08192b192b2b2b,
    0x2b082b0808080808,
    0x2b082b0808082b08,
    0x2b082b08082b1919,
    0x2b082b0819192b2b,
    0x2b082b082b080808,
    0x2b082b082b08082b,
    0x2b082b082b2b2b08,
    0x2b082b190808192b,
    0x2b082b2b082b082b,
    0x2b082b2b2b080808,
    0x2b082b2b2b082b08,
    0x2b082b2b2b19192b,
    0x2b082b2b2b2b2b08,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b19080819080808,
    0x2b1908081919192b,
    0x2b1908082b081908,
    0x2b19081908080808,
    0x2b190819082b082b,
    0x2b190819192b1908,
    0x2b19082b1919192b,
    0x2b19082b2b082b19,
    0x2b19190808080808,
    0x2b19190808081919,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b19190819192b08,
    0x2b191919082b2b19,
    0x2b1919192b190808,
    0x2b1919192b19082b,
    0x2b19192b19080819,
    0x2b192b0819190819,
    0x2b192b082b2b192b,
    0x2b192b1919082b19,
    0x2b192b2b08191919,
    0x2b192b2b192b0808,
    0x2b2b080808080808,
    0x2b2b08080808082b,
    0x2b2b080808082b08,
    0x2b2b080808082b2b,
    0x2b2b0808082b0808,
    0x2b2b0808082b2b2b,
    0x2b2b08082b2b0808,
    0x2b2b081919190819,
    0x2b2b081919192b19,
    0x2b2b08192b2b192b,
    0x2b2b082b08080808,
    0x2b2b082b0808082b,
    0x2b2b082b08082b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b080808,
    0x2b2b082b2b2b0808,
    0x2b2b190819080808,
    0x2b2b19082b191919,
    0x2b2b192b192b1919,
    0x2b2b192b2b192b08,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808,
    0x2b2b2b08082b082b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08,
    0x2b2b2b1908081908,
    0x2b2b2b192b081908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b082b2b2b,
    0x2b2b2b2b2b190819,
    0x2b2b2b2b2b2b2b2b,
];

pub(super) const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153,
    154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178,
    51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75,
    204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228,
    101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

pub(super) const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

pub(super) const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414, 0x04041c0c,
    0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c,
    0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404, 0x04140414, 0x04140424, 0x04140c0c,
    0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c,
    0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e,
    0x04243e1c, 0x04243e2c, 0x042c040c, 0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04,
    0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c,
    0x0c04141c, 0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04, 0x0c143e14,
    0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04,
    0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414, 0x0c3e2404, 0x14040404, 0x14040414,
    0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c,
    0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c, 0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404,
    0x14140414, 0x14140c0c, 0x14140c3e, 0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,
    0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c,
    0x142c3e24, 0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424,
    0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c, 0x1c1c1c1c,
    0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414, 0x1c2c2c2c, 0x1c340c24, 0x1c341c34,
    0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e,
    0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424,
    0x24242c0c, 0x24243424, 0x242c142c, 0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04,
    0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14,
    0x2c143e14, 0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c, 0x340c340c,
    0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c,
    0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14, 0x3e042c14, 0x3e0c1434, 0x3e0c2404,
    0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
    0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];
