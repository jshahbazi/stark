use std::{fs::File, io::BufReader, path::Path, sync::Arc};

use derive_more::Debug;
use thiserror::Error;

#[cfg(feature = "metal")]
use crate::backends::metal::Metal;
use crate::{
    backends::common::{Backend, Context},
    config::model::AnyModelConfig,
    engine::capture::CaptureManager,
};

pub mod classifier_model;
#[cfg(unix)]
pub mod hybrid;
pub mod language_model;
pub mod vision;

mod capture;

pub struct Engine<B: Backend> {
    context: Arc<B::Context>,
    capture_manager: Option<CaptureManager<B>>,
}

#[derive(Debug, Error)]
pub enum EngineNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
}

impl<B: Backend> Engine<B> {
    pub fn new() -> Result<Arc<Self>, EngineNewError<B>> {
        let capture_enabled = CaptureManager::<B>::pre_load_enable();

        let context = <B::Context as Context>::new().map_err(EngineNewError::Backend)?;

        let capture_manager = capture_enabled.then(|| CaptureManager::new(context.clone()));

        Ok(Arc::new(Self {
            context,
            capture_manager,
        }))
    }

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.context.peak_memory_usage()
    }
}

#[cfg(feature = "metal")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceIdentity {
    pub backend: &'static str,
    pub device_name: String,
    pub apple_gpu_family: u32,
    pub gpu_core_count: u32,
    pub supports_mxu: bool,
}

#[cfg(feature = "metal")]
impl Engine<Metal> {
    /// Stable, non-unique device properties for binding runtime evidence to a
    /// tested Metal device class. This deliberately excludes host identifiers.
    pub fn metal_device_identity(&self) -> MetalDeviceIdentity {
        MetalDeviceIdentity {
            backend: Metal::NAME,
            device_name: self.context.device_name.clone(),
            apple_gpu_family: self.context.apple_gpu_family as isize as u32,
            gpu_core_count: self.context.gpu_core_count,
            supports_mxu: self.context.supports_mxu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    LanguageModel,
    Classifier,
}

#[derive(Debug, Error)]
pub enum ResolveModelTypeError {
    #[error("Unable to open model configuration: {0}")]
    UnableToOpenConfig(#[from] std::io::Error),
    #[error("Unable to deserialize model configuration: {0}")]
    UnableToDeserializeConfig(#[from] serde_json::Error),
}

pub fn resolve_model_type(model_path: &Path) -> Result<ModelType, ResolveModelTypeError> {
    let config_path = model_path.join("config.json");
    let file = File::open(&config_path)?;
    let config: AnyModelConfig = serde_json::from_reader(BufReader::new(file))?;
    Ok(match config {
        AnyModelConfig::LanguageModelConfig(_) => ModelType::LanguageModel,
        AnyModelConfig::ClassifierModelConfig(_) => ModelType::Classifier,
    })
}
