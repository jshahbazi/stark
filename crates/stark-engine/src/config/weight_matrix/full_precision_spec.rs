use stark_engine_macros::stark_config;

use crate::config::weight_matrix::Layout;

#[stark_config(super::WeightMatrixSpec)]
pub struct FullPrecisionSpec {
    pub layout: Layout,
}
