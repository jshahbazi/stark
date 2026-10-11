use stark_engine_macros::stark_config;

use crate::config::weight_matrix::Layout;

/// Lookup-only embedding table: one D4 lattice code per 4 columns and a ladder scale per 64 columns.
#[stark_config(super::WeightMatrixSpec)]
pub struct D4S4Spec {
    pub layout: Layout,
}
