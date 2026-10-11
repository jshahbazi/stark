use stark_engine_macros::stark_config_abstract;

pub mod softmax_routing;

#[stark_config_abstract(softmax_routing::SoftmaxRouting)]
pub struct RoutingFunction;
