use stark_engine_macros::stark_config;

use crate::config::weight_matrix::Layout;

#[stark_config(super::WeightMatrixSpec)]
pub struct MLXSpec {
    pub bits: u32,
    pub group_size: u32,
    pub layout: Layout,
    /// Match selected W4 sources' BF16 four-value bias sums and F32 affine fold.
    #[serde(default)]
    pub materialize_affine_input_sum: bool,
}

impl MLXSpec {
    pub fn new(
        bits: u32,
        group_size: u32,
        layout: Layout,
    ) -> Self {
        Self {
            ty: Default::default(),
            bits,
            group_size,
            materialize_affine_input_sum: false,
            layout,
        }
    }
}
