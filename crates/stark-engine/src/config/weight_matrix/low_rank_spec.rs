use stark_engine_macros::stark_config;

#[stark_config(super::WeightMatrixSpec)]
pub struct LowRankSpec {
    pub rank: u32,
}
