use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransformError {
    #[error("Transformation execution limit exceeded")]
    ExecutionLimit,
    #[error("Cyclic transformation pipeline")]
    CyclicPipeline,
    #[error("Undefined pipeline: {name}")]
    UndefinedPipeline {
        name: String,
    },
    #[error("Invalid regex pattern: {pattern}")]
    InvalidRegex {
        pattern: String,
    },
    #[error("Invalid JSON: {message}")]
    InvalidJson {
        message: String,
    },
}
