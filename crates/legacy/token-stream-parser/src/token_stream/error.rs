use crate::reduction::ReductionParserError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TokenStreamParserError {
    #[error(transparent)]
    Transform(#[from] json_transform::TransformError),
    #[error(transparent)]
    Reduction(#[from] ReductionParserError),
}
