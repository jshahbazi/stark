use stark_engine_macros::stark_config_abstract;

pub mod dense_mlp;
pub mod mixture_of_experts;
pub mod routing_function;

#[stark_config_abstract(dense_mlp::DenseMLPConfig, mixture_of_experts::MixtureOfExpertsConfig)]
pub struct MLPConfig;
