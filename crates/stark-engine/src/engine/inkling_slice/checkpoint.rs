// SPDX-License-Identifier: Apache-2.0
//! A scoped SafeTensors reader for the exact original mixed-format slice.
//! Keep E4M3 bytes packed. Validate requested dtype and geometry before upload.

use std::{collections::HashMap, fs::File, io::Read, os::unix::fs::FileExt, path::Path};

use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use serde::Deserialize;

use super::{Buffer, Device, Result};

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(super) struct Tensors {
    file: File,
    base: u64,
    size: u64,
    tensors: HashMap<String, Tensor>,
}

impl Tensors {
    pub(super) fn contains(
        &self,
        name: &str,
    ) -> bool {
        self.tensors.contains_key(name)
    }

    pub(super) fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|_| "cannot open SafeTensors")?;
        let size = file.metadata().map_err(|_| "cannot inspect SafeTensors")?.len();
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix).map_err(|_| "missing SafeTensors header")?;
        let header = u64::from_le_bytes(prefix);
        if header > 4 * 1024 * 1024 || header + 8 > size {
            return Err("invalid SafeTensors header size");
        }
        let mut data = vec![0; header as usize];
        file.read_exact(&mut data).map_err(|_| "truncated SafeTensors header")?;
        let mut values = serde_json::from_slice::<HashMap<String, serde_json::Value>>(&data)
            .map_err(|_| "invalid SafeTensors header")?;
        values.remove("__metadata__");
        let tensors = values
            .into_iter()
            .map(|(key, value)| {
                let tensor = serde_json::from_value(value).map_err(|_| "invalid SafeTensors entry")?;
                Ok((key, tensor))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            file,
            base: header + 8,
            size,
            tensors,
        })
    }

    fn range(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<(u64, usize)> {
        let tensor = self.tensors.get(name).ok_or("missing requested tensor")?;
        let element = match dtype {
            "U8" | "F8_E4M3" => 1,
            "BF16" => 2,
            "U32" | "F32" => 4,
            _ => return Err("unsupported requested dtype"),
        };
        if tensor.dtype != dtype || tensor.shape != shape {
            return Err("unexpected tensor dtype or shape");
        }
        let bytes = shape
            .iter()
            .try_fold(element, |count: usize, &dim| count.checked_mul(dim))
            .ok_or("tensor size overflow")?;
        let [start, end] = tensor.data_offsets;
        if end.checked_sub(start) != Some(bytes) || end as u64 > self.size - self.base {
            return Err("invalid tensor offsets");
        }
        Ok((self.base + start as u64, bytes))
    }

    fn read(
        &self,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<()> {
        self.file.read_exact_at(destination, offset).map_err(|_| "cannot read tensor contents")
    }

    pub(super) fn f32(
        &self,
        name: &str,
    ) -> Result<f32> {
        let (offset, _) = self.range(name, "F32", &[1])?;
        let mut data = [0; 4];
        self.read(offset, &mut data)?;
        Ok(f32::from_le_bytes(data))
    }

    pub(super) fn values<T: bytemuck::Pod>(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Vec<T>> {
        let (offset, bytes) = self.range(name, dtype, shape)?;
        if bytes % size_of::<T>() != 0 {
            return Err("unexpected tensor element width");
        }
        let mut data = vec![T::zeroed(); bytes / size_of::<T>()];
        self.read(offset, bytemuck::cast_slice_mut(&mut data))?;
        Ok(data)
    }

    pub(super) fn buffer(
        &self,
        device: &Device,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Buffer> {
        let (offset, bytes) = self.range(name, dtype, shape)?;
        let buffer = device
            .new_buffer(bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or("Metal tensor allocation failed")?;
        self.read_into(offset, &buffer, 0, bytes)?;
        Ok(buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_to(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        source: usize,
        dest: &Buffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let (start, len) = self.range(name, dtype, shape)?;
        if source > len || bytes > len - source || offset > dest.length() || bytes > dest.length() - offset {
            return Err("invalid tensor view range");
        }
        self.read_into(start + source as u64, dest, offset, bytes)
    }

    fn read_into(
        &self,
        start: u64,
        dest: &Buffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        // The buffer is shared and is exclusively initialized before any Metal
        // dispatch reads it. Reading directly avoids a duplicate weight bank.
        let out = unsafe { std::slice::from_raw_parts_mut(dest.contents().as_ptr().cast::<u8>().add(offset), bytes) };
        self.read(start, out)
    }
}
