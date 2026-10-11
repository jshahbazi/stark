use stark_engine_macros::{stark_config, stark_config_abstract};

pub mod d4s4_spec;
pub mod full_precision_spec;
pub mod hybrid_spec;
pub mod int_spec;
pub mod low_rank_spec;
pub mod mlx_spec;

#[stark_config]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    OutputInput,
    InputOutput,
}

#[stark_config_abstract(
    full_precision_spec::FullPrecisionSpec,
    low_rank_spec::LowRankSpec,
    hybrid_spec::HybridSpec,
    int_spec::IntSpec,
    mlx_spec::MLXSpec,
    d4s4_spec::D4S4Spec
)]
pub struct WeightMatrixSpec;
