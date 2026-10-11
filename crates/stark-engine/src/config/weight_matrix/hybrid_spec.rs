use stark_engine_macros::stark_config;

use crate::config::weight_matrix::AnyWeightMatrixSpec;

#[stark_config]
#[serde(rename_all = "snake_case")]
pub enum IncoherenceProcessingMode {
    Input,
    Output,
    InputOutput,
}

#[stark_config(super::WeightMatrixSpec)]
pub struct HybridSpec {
    pub quantization_spec: Box<AnyWeightMatrixSpec>,
    pub adapter_spec: Option<Box<AnyWeightMatrixSpec>>,
    pub incoherence_block_size: Option<u32>,
    pub incoherence_processing_mode: IncoherenceProcessingMode,
}
