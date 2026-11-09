#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_family = "wasm")]
use std::os::wasi::fs::FileExt;
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
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

/// Parameter storage only, excluding request state, scratch and allocator slack.
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct ParameterStorageStatistics {
    pub resident_bytes: usize,
    pub peak_loading_bytes: usize,
    pub shared_matrix_bytes: usize,
    /// Subset of shared_matrix_bytes, not an additional allocation.
    pub shared_expert_matrix_bytes: usize,
    pub representation_copy_bytes: usize,
    pub unshared_parameter_bytes: usize,
}

/// An internal arithmetic policy selected by an admitting loader, never by
/// checkpoint metadata. Ordinary text loading retains the established policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum NumericalProfile {
    #[default]
    Legacy,
    Qwen35ReferenceBF16,
    /// BF16 activation boundaries with canonical packed affine matrices.
    Qwen35AffineBF16,
    /// Test-only causal experiment: full BF16 linears with all other operators unchanged.
    #[cfg(test)]
    QwenMtpReferenceLinears,
}

impl NumericalProfile {
    pub(crate) fn qwen35_bf16_activations(self) -> bool {
        matches!(self, Self::Qwen35ReferenceBF16 | Self::Qwen35AffineBF16)
    }
}

pub struct ParameterLoader<'a, B: Backend> {
    context: &'a B::Context,
    index: HashMap<String, ParameterMetadata>,
    metadata: HashMap<String, String>,
    validated_tensors: RefCell<HashSet<String>>,
    file: &'a File,
    numerical_profile: NumericalProfile,
    shared_weights: Option<RefCell<HashMap<String, Arc<B::GlobalBuffer>>>>,
    representation_copy_bytes: Arc<AtomicUsize>,
    allocated_buffer_bytes: Cell<usize>,
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
            numerical_profile: NumericalProfile::Legacy,
            shared_weights: None,
            representation_copy_bytes: Arc::new(AtomicUsize::new(0)),
            allocated_buffer_bytes: Cell::new(0),
        })
    }

    pub(crate) fn with_numerical_profile(
        mut self,
        numerical_profile: NumericalProfile,
    ) -> Self {
        self.numerical_profile = numerical_profile;
        self
    }

    pub(crate) fn enable_weight_sharing(&mut self) {
        self.shared_weights = Some(RefCell::new(HashMap::new()));
    }

    pub(crate) fn set_numerical_profile(
        &mut self,
        profile: NumericalProfile,
    ) {
        self.numerical_profile = profile;
    }

    /// Upper bound for a second Qwen graph: immutable matrices share storage;
    /// signed A8 codes may require a different representation, and small F32
    /// controls are independently constructed. Keep kernel/allocator margin.
    pub(crate) fn qwen_text_graph_extra_bytes(&self) -> Option<usize> {
        let mut codes = 0usize;
        let mut controls = 0usize;
        for (name, tensor) in &self.index {
            if !name.starts_with("decoder.") && !name.starts_with("mtp.") {
                continue;
            }
            if tensor.data_type == DataType::U8 && name.ends_with(".weights") {
                codes = codes.checked_add(tensor.size)?;
            } else if tensor.data_type == DataType::F32 {
                controls = controls.checked_add(tensor.size)?;
            }
        }
        codes.checked_add(controls.checked_mul(4)?)?.checked_add(256 * 1024 * 1024)
    }

    pub(crate) fn storage_statistics(&self) -> ParameterStorageStatistics {
        let mut cache_only = 0usize;
        let mut shared = 0usize;
        let mut shared_experts = 0usize;
        if let Some(cache) = &self.shared_weights {
            for (name, buffer) in cache.borrow().iter() {
                use crate::backends::common::Buffer;
                let bytes = buffer.size();
                match Arc::strong_count(buffer) {
                    1 => cache_only += bytes,
                    2 => {},
                    _ => {
                        shared += bytes;
                        if name.contains(".mlp.experts.") {
                            shared_experts += bytes;
                        }
                    },
                }
            }
        }
        let copies = self.representation_copy_bytes.load(Ordering::Relaxed);
        let allocated = self.allocated_buffer_bytes.get();
        ParameterStorageStatistics {
            resident_bytes: allocated + copies - cache_only,
            peak_loading_bytes: allocated + copies,
            shared_matrix_bytes: shared,
            shared_expert_matrix_bytes: shared_experts,
            representation_copy_bytes: copies,
            unshared_parameter_bytes: allocated - cache_only - shared,
        }
    }

    pub fn tree(&self) -> ParameterTree<'_, B> {
        ParameterTree {
            loader: self,
            prefix: None,
            #[cfg(test)]
            validation_profile: None,
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
        self.loader.allocated_buffer_bytes.set(self.loader.allocated_buffer_bytes.get() + self.metadata.size);
        Ok(buffer)
    }

    /// Only raw validated parameter bytes enter this cache. Callers receive
    /// immutable ownership; representation changes must allocate private bytes.
    pub(crate) fn read_shared_weight(&self) -> Result<Arc<B::GlobalBuffer>, ParameterLoaderError<B>> {
        if let Some(cache) = &self.loader.shared_weights {
            if let Some(buffer) = cache.borrow().get(self.key) {
                return Ok(buffer.clone());
            }
            let buffer = Arc::new(self.read_buffer()?);
            cache.borrow_mut().insert(self.key.into(), buffer.clone());
            Ok(buffer)
        } else {
            Ok(Arc::new(self.read_buffer()?))
        }
    }
}

pub struct ParameterTree<'loader, B: Backend> {
    loader: &'loader ParameterLoader<'loader, B>,
    prefix: Option<String>,
    #[cfg(test)]
    validation_profile: Option<NumericalProfile>,
}

impl<'loader, B: Backend> ParameterTree<'loader, B> {
    pub(crate) fn representation_copy_counter(&self) -> Arc<AtomicUsize> {
        self.loader.representation_copy_bytes.clone()
    }
    pub(crate) fn numerical_profile(&self) -> NumericalProfile {
        #[cfg(test)]
        if let Some(profile) = self.validation_profile {
            return profile;
        }
        self.loader.numerical_profile
    }

    #[cfg(test)]
    pub(crate) fn validation_with_numerical_profile(
        &self,
        profile: NumericalProfile,
    ) -> Self {
        Self {
            loader: self.loader,
            prefix: self.prefix.clone(),
            validation_profile: Some(profile),
        }
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
            #[cfg(test)]
            validation_profile: self.validation_profile,
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

    use super::{NumericalProfile, ParameterLoader};
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

    #[stark_test]
    fn validation_profile_is_scoped_and_does_not_select_reference_activations() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let file = tensor_file(1, &[0]);
        let loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
        let root = loader.tree();
        let head = root.subtree("mtp").validation_with_numerical_profile(NumericalProfile::QwenMtpReferenceLinears);
        assert_eq!(
            head.subtree("transformer").subtree("mixer").numerical_profile(),
            NumericalProfile::QwenMtpReferenceLinears
        );
        assert!(!head.numerical_profile().qwen35_bf16_activations());
        assert_eq!(root.numerical_profile(), NumericalProfile::Legacy);
        assert_eq!(root.subtree("decoder").numerical_profile(), NumericalProfile::Legacy);
        assert_eq!(loader.tree().subtree("mtp").numerical_profile(), NumericalProfile::Legacy);
    }

    #[stark_test]
    fn shared_raw_parameter_survives_profile_change_without_duplicate_allocation() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let file = tensor_file(3, &[4, 5, 6]);
        let mut loader = ParameterLoader::<Cpu>::new(&file, &context).unwrap();
        loader.enable_weight_sharing();
        let first =
            loader.tree().leaf("weight").unwrap().validate(&[3], DataType::U8).unwrap().read_shared_weight().unwrap();
        loader.set_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
        let second =
            loader.tree().leaf("weight").unwrap().validate(&[3], DataType::U8).unwrap().read_shared_weight().unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_ref().copyout::<u8>(), [4, 5, 6]);
        let stats = loader.storage_statistics();
        assert_eq!(stats.resident_bytes, 3);
        assert_eq!(stats.shared_matrix_bytes, 3);
        assert_eq!(stats.representation_copy_bytes, 0);
    }
}
