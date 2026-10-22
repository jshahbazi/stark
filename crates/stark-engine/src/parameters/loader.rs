#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_family = "wasm")]
use std::os::wasi::fs::FileExt;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
};

use super::{
    error::ParameterLoaderError,
    safetensors_metadata::{HeaderLoadingError, read_metadata as read_st_metadata},
};
use crate::{
    array::{ArrayElement, size_for_shape},
    backends::common::{Backend, BufferMut, Context},
    data_type::DataType,
    utils::strict_serde::DeserializeStrictOwned,
};

struct ParameterMetadata {
    shape: Box<[u32]>,
    data_type: DataType,
    offset: usize,
    size: usize,
}

pub struct ParameterLoader<'a, B: Backend> {
    context: &'a B::Context,
    index: HashMap<String, ParameterMetadata>,
    metadata: HashMap<String, String>,
    validated_tensors: RefCell<HashSet<String>>,
    file: &'a File,
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub fn new(
        file: &'a File,
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let (global_offset, st_metadata) = read_st_metadata(file)?;
        let file_length = file.metadata()?.len();
        let index = st_metadata
            .tensors
            .into_iter()
            .map(|(key, value)| {
                let (local_begin, local_end) = value.data_offsets;
                let size =
                    local_end.checked_sub(local_begin).ok_or_else(|| HeaderLoadingError::InvalidTensorOffsets {
                        key: key.clone().into_boxed_str(),
                        begin: local_begin,
                        end: local_end,
                    })?;
                let offset =
                    global_offset.checked_add(local_begin).ok_or_else(|| HeaderLoadingError::TensorOffsetOverflow {
                        key: key.clone().into_boxed_str(),
                        global_offset,
                        local_begin,
                    })?;
                let absolute_end = global_offset.checked_add(local_end).ok_or(HeaderLoadingError::TruncatedTensor)?;
                if u64::try_from(absolute_end).map_or(true, |end| end > file_length) {
                    return Err(HeaderLoadingError::TruncatedTensor);
                }
                let weight_metadata = ParameterMetadata {
                    shape: value.shape.into(),
                    data_type: value.dtype.data_type()?,
                    offset,
                    size,
                };
                Ok((key, weight_metadata))
            })
            .collect::<Result<HashMap<_, _>, HeaderLoadingError>>()?;
        let metadata = st_metadata.metadata.unwrap_or_default();
        Ok(ParameterLoader {
            context,
            index,
            metadata,
            validated_tensors: RefCell::new(HashSet::new()),
            file,
        })
    }

    pub fn tree(&self) -> ParameterTree<'_, B> {
        ParameterTree {
            loader: self,
            prefix: None,
        }
    }
}

pub struct ParameterLeaf<'a, 'leaf, B: Backend, const VALIDATED: bool> {
    key: &'leaf str,
    metadata: &'leaf ParameterMetadata,
    loader: &'leaf ParameterLoader<'a, B>,
}

impl<'a, 'leaf, B: Backend> ParameterLeaf<'a, 'leaf, B, false> {
    pub fn validate(
        self,
        expected_shape: &[u32],
        expected_data_type: DataType,
    ) -> Result<ParameterLeaf<'a, 'leaf, B, true>, ParameterLoaderError<B>> {
        let shape = self.metadata.shape.as_ref();
        let data_type = self.metadata.data_type;
        if (shape, data_type) != (expected_shape, expected_data_type) {
            return Err(ParameterLoaderError::InvalidTensor {
                shape: shape.into(),
                data_type,
                expected_shape: expected_shape.into(),
                expected_data_type,
            });
        }
        let expected_size = size_for_shape(expected_shape, expected_data_type);
        if self.metadata.size != expected_size {
            return Err(ParameterLoaderError::InvalidTensorSize {
                shape: shape.into(),
                data_type,
                size: self.metadata.size,
                expected_size,
            });
        }
        self.loader.validated_tensors.borrow_mut().insert(self.key.to_string());
        Ok(ParameterLeaf {
            key: self.key,
            metadata: self.metadata,
            loader: self.loader,
        })
    }
}

impl<'a, 'leaf, B: Backend> ParameterLeaf<'a, 'leaf, B, true> {
    pub fn read_slice<T: ArrayElement>(&self) -> Result<Box<[T]>, ParameterLoaderError<B>> {
        let element_count = self.metadata.size / std::mem::size_of::<T>();
        let mut data = vec![T::zeroed(); element_count];
        let destination = bytemuck::cast_slice_mut(&mut data);
        self.loader.file.read_exact_at(destination, self.metadata.offset as u64)?;
        Ok(data.into_boxed_slice())
    }

    pub fn read_buffer(&self) -> Result<B::GlobalBuffer, ParameterLoaderError<B>> {
        let mut buffer =
            self.loader.context.create_buffer(self.metadata.size).map_err(ParameterLoaderError::BackendError)?;
        self.loader.file.read_exact_at(buffer.as_slice_mut::<u8>(), self.metadata.offset as u64)?;
        Ok(buffer)
    }
}

pub struct ParameterTree<'loader, B: Backend> {
    loader: &'loader ParameterLoader<'loader, B>,
    prefix: Option<String>,
}

impl<'loader, B: Backend> ParameterTree<'loader, B> {
    fn join_prefix(
        &self,
        name: &str,
    ) -> String {
        self.prefix.as_ref().map_or_else(|| name.to_string(), |p| format!("{p}.{name}"))
    }

    pub fn subtree(
        &self,
        name: &str,
    ) -> Self {
        Self {
            loader: self.loader,
            prefix: Some(self.join_prefix(name)),
        }
    }

    pub fn leaf<'leaf>(
        &'leaf self,
        name: &str,
    ) -> Result<ParameterLeaf<'loader, 'leaf, B, false>, ParameterLoaderError<B>> {
        let key = self.join_prefix(name);
        let Some((key, metadata)) = self.loader.index.get_key_value(&key) else {
            return Err(ParameterLoaderError::KeyNotFound(key));
        };
        Ok(ParameterLeaf {
            key,
            metadata,
            loader: self.loader,
        })
    }

    pub fn metadata<T: DeserializeStrictOwned>(
        &self,
        name: &str,
    ) -> Result<T, ParameterLoaderError<B>> {
        let new_prefix = self.join_prefix(name);

        Ok(serde_json::from_str(
            self.loader.metadata.get(&new_prefix).ok_or(ParameterLoaderError::KeyNotFound(new_prefix))?,
        )?)
    }

    pub fn assert_all_tensors_validated(&self) -> Result<(), ParameterLoaderError<B>> {
        let subtree_prefix = self.prefix.as_ref().map(|prefix| format!("{prefix}."));
        let validated_tensors = self.loader.validated_tensors.borrow();
        let mut unvalidated_tensors = self
            .loader
            .index
            .keys()
            .filter(|key| subtree_prefix.as_ref().is_none_or(|prefix| key.starts_with(prefix)))
            .filter(|key| !validated_tensors.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        unvalidated_tensors.sort();
        if unvalidated_tensors.is_empty() {
            Ok(())
        } else {
            Err(ParameterLoaderError::UnvalidatedTensors {
                prefix: self.prefix.clone(),
                keys: unvalidated_tensors.into_boxed_slice(),
            })
        }
    }
}

#[cfg(all(test, backend = "cpu"))]
mod payload_tests {
    use std::{
        fs::OpenOptions,
        io::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    use stark_engine_macros::stark_test;

    use super::ParameterLoader;
    use crate::{
        backends::{
            common::{Backend, BufferRef, Context},
            cpu::Cpu,
        },
        data_type::DataType,
        parameters::{ParameterLoaderError, safetensors_metadata::HeaderLoadingError},
    };

    fn tensor_file(
        end: usize,
        payload: &[u8],
    ) -> std::fs::File {
        let path = std::env::temp_dir().join(format!(
            "stark-payload-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let shape = end.min(u32::MAX as usize);
        let header = format!(r#"{{"weight":{{"dtype":"U8","shape":[{shape}],"data_offsets":[0,{end}]}}}}"#);
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
        file
    }

    #[stark_test]
    fn tensor_payload_is_checked_before_any_tensor_allocation() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        for end in [1, 1 << 30, usize::MAX] {
            let file = tensor_file(end, &[]);
            assert!(matches!(
                ParameterLoader::<Cpu>::new(&file, &context),
                Err(ParameterLoaderError::HeaderLoadingError(HeaderLoadingError::TruncatedTensor))
            ));
        }
    }

    #[stark_test]
    fn exact_tensor_payload_still_loads() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let file = tensor_file(3, &[4, 5, 6]);
        let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
        let tree = loader.tree();
        let leaf = tree.leaf("weight").unwrap().validate(&[3], DataType::U8).unwrap();
        assert_eq!(&*leaf.read_slice::<u8>().unwrap(), &[4, 5, 6]);
        assert_eq!(leaf.read_buffer().unwrap().copyout::<u8>(), [4, 5, 6]);
    }
}
