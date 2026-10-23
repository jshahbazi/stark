use thiserror::Error;

/// A content-free validation failure before native allocation or dispatch.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("Invalid native input: {0}")]
pub struct InvalidInput(pub &'static str);
