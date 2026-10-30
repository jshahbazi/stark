#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_family = "wasm")]
use std::os::wasi::fs::FileExt;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
    io,
    sync::Arc,
};

use super::{
    error::ParameterLoaderError,
    safetensors_metadata::{
        HashMetadata, HeaderLoadingError, read_metadata as read_st_metadata,
        read_metadata_with_limit as read_st_metadata_with_limit,
    },
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

/// An internal arithmetic policy selected by an admitting loader, never by
/// checkpoint metadata. Ordinary text loading retains the established policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum NumericalProfile {
    #[default]
    Legacy,
    Qwen35ReferenceBF16,
}

pub struct ParameterLoader<'a, B: Backend> {
    context: &'a B::Context,
    index: HashMap<String, ParameterMetadata>,
    metadata: HashMap<String, String>,
    validated_tensors: RefCell<HashSet<String>>,
    file: &'a File,
    region_file: RefCell<Option<Arc<File>>>,
    numerical_profile: NumericalProfile,
}

impl<'a, B: Backend> ParameterLoader<'a, B> {
    pub fn new(
        file: &'a File,
        context: &'a B::Context,
    ) -> Result<Self, ParameterLoaderError<B>> {
        Self::from_metadata(file, context, read_st_metadata(file)?)
    }

    /// Reject an advertised safetensors header length before allocating or
    /// reading its JSON buffer. The limit can tighten the standard ceiling of
    /// 100,000,000 bytes, and cannot increase it.
    pub fn new_with_header_limit(
        file: &'a File,
        context: &'a B::Context,
        maximum_header_bytes: usize,
    ) -> Result<Self, ParameterLoaderError<B>> {
        Self::from_metadata(file, context, read_st_metadata_with_limit(file, maximum_header_bytes)?)
    }

    fn from_metadata(
        file: &'a File,
        context: &'a B::Context,
        (global_offset, st_metadata): (usize, HashMetadata),
    ) -> Result<Self, ParameterLoaderError<B>> {
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
            region_file: RefCell::new(None),
            numerical_profile: NumericalProfile::Legacy,
        })
    }

    pub(crate) fn with_numerical_profile(
        mut self,
        numerical_profile: NumericalProfile,
    ) -> Self {
        self.numerical_profile = numerical_profile;
        self
    }

    pub fn tree(&self) -> ParameterTree<'_, B> {
        ParameterTree {
            loader: self,
            prefix: None,
        }
    }

    /// Bounded original header bytes for architecture-specific strict admission.
    /// This does not mark any tensor consumed or read a tensor payload.
    pub(crate) fn bounded_header(
        &self,
        maximum: usize,
    ) -> io::Result<(u64, Vec<u8>, u64)> {
        let file_len = self.file.metadata()?.len();
        let mut length = [0_u8; 8];
        self.file.read_exact_at(&mut length, 0)?;
        let count = usize::try_from(u64::from_le_bytes(length))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid parameter header"))?;
        let start = 8_u64
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid parameter header"))?;
        if count == 0 || count > maximum || start > file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid parameter header"));
        }
        let mut bytes = vec![0_u8; count];
        self.file.read_exact_at(&mut bytes, 8)?;
        Ok((start, bytes, file_len))
    }

    pub(crate) fn read_file_exact_at(
        &self,
        offset: u64,
        destination: &mut [u8],
    ) -> io::Result<()> {
        let end = offset
            .checked_add(destination.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "parameter region overflow"))?;
        if end > self.file.metadata()?.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "parameter region exceeds file"));
        }
        self.file.read_exact_at(destination, offset)
    }
}

/// An owned, validated tensor span. Paging survives the construction loader and
/// cannot read another tensor's bytes through an unchecked relative offset.
#[derive(Clone)]
pub struct ValidatedFileRegion {
    file: Arc<File>,
    offset: u64,
    length: u64,
    shape: Box<[u32]>,
    data_type: DataType,
}

impl ValidatedFileRegion {
    pub fn len(&self) -> u64 {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
    pub fn read_exact_into(
        &self,
        offset: u64,
        destination: &mut [u8],
    ) -> io::Result<()> {
        let end = offset
            .checked_add(destination.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "parameter region overflow"))?;
        if end > self.length {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "read exceeds validated tensor region"));
        }
        let absolute = self
            .offset
            .checked_add(offset)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "parameter region overflow"))?;
        self.file.read_exact_at(destination, absolute)
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
    pub fn file_region(&self) -> Result<ValidatedFileRegion, ParameterLoaderError<B>> {
        let shared = self.loader.region_file.borrow().as_ref().cloned();
        let file = match shared {
            Some(file) => file,
            None => {
                let file = Arc::new(self.loader.file.try_clone()?);
                *self.loader.region_file.borrow_mut() = Some(file.clone());
                file
            },
        };
        Ok(ValidatedFileRegion {
            file,
            offset: self.metadata.offset as u64,
            length: self.metadata.size as u64,
            shape: self.metadata.shape.clone(),
            data_type: self.metadata.data_type,
        })
    }

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
    pub(crate) fn numerical_profile(&self) -> NumericalProfile {
        self.loader.numerical_profile
    }

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

    use super::{FileExt, NumericalProfile, ParameterLoader};
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

    #[stark_test]
    fn configured_header_limit_rejects_prefix_before_allocation_or_parse() {
        const LIMIT: usize = 8 * 1024 * 1024;
        let context = <Cpu as Backend>::Context::new().unwrap();
        let file = tensor_file(0, &[]);
        file.set_len(8).unwrap();
        file.write_all_at(&((LIMIT + 1) as u64).to_le_bytes(), 0).unwrap();
        assert!(matches!(
            ParameterLoader::<Cpu>::new_with_header_limit(&file, context.as_ref(), LIMIT),
            Err(ParameterLoaderError::HeaderLoadingError(HeaderLoadingError::HeaderTooLarge))
        ));
        assert!(file.metadata().unwrap().len() == 8, "oversized-header fixture allocated a payload");
        let valid = tensor_file(3, &[4, 5, 6]);
        let loader = ParameterLoader::<Cpu>::new_with_header_limit(&valid, context.as_ref(), LIMIT).unwrap();
        let tree = loader.tree();
        let leaf = tree.leaf("weight").unwrap().validate(&[3], DataType::U8).unwrap();
        assert!(leaf.read_slice::<u8>().unwrap().as_ref() == [4, 5, 6]);
    }

    #[stark_test]
    fn validated_file_region_survives_loader_drop_and_enforces_its_span() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let region = {
            let file = tensor_file(3, &[4, 5, 6]);
            let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
            assert!(loader.bounded_header(1).is_err());
            loader.tree().leaf("weight").unwrap().validate(&[3], DataType::U8).unwrap().file_region().unwrap()
        };
        let mut bytes = [0_u8; 3];
        region.read_exact_into(0, &mut bytes).unwrap();
        assert!(bytes == [4, 5, 6], "owned parameter region changed bytes");
        assert!(region.read_exact_into(3, &mut [0]).is_err());
        assert!(region.read_exact_into(u64::MAX, &mut [0, 0]).is_err());
        region.read_exact_into(3, &mut []).unwrap();
        assert!(region.shape() == [3] && region.data_type() == DataType::U8 && region.len() == 3);
    }

    #[stark_test]
    fn numerical_profile_defaults_to_legacy_and_reaches_subtrees() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let file = tensor_file(1, &[0]);
        let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
        assert_eq!(loader.tree().numerical_profile(), NumericalProfile::Legacy);
        let loader = loader.with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
        assert_eq!(
            loader.tree().subtree("decoder").subtree("mixer").numerical_profile(),
            NumericalProfile::Qwen35ReferenceBF16
        );
    }
}
