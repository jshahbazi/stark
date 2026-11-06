use stark_engine_macros::stark_config;

use crate::config::weight_matrix::Layout;

/// Contiguous output rows retain their source affine widths independently.
#[stark_config]
pub struct MLXRowSegment {
    pub rows: u32,
    pub bits: u32,
    pub group_size: u32,
}

#[stark_config(super::WeightMatrixSpec)]
pub struct MLXRowPartitionedSpec {
    pub layout: Layout,
    pub segments: Box<[MLXRowSegment]>,
}
