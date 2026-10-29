#[cfg(not(backend = "cpu"))]
compile_error!("stark tests require cpu backend");

pub mod assert;
pub mod cold_pool;
pub mod env_vars;
pub mod harness;
pub mod helpers;
pub mod matmul;
pub mod path;
pub mod perf;
#[cfg(unix)]
pub mod private_output;
pub mod proptest;
pub mod util;

#[path = "../bench_cold_pool_test.rs"]
mod bench_cold_pool_test;
#[path = "../session/model_loading_bench.rs"]
mod model_loading_bench;

#[cfg(backend = "metal")]
#[path = "../engine/gemma4_precision.rs"]
mod gemma4_precision;
#[cfg(backend = "metal")]
#[path = "../engine/olmo3_precision.rs"]
mod olmo3_precision;

#[cfg(backend = "metal")]
#[path = "../engine/gemma4_ple_precision.rs"]
mod gemma4_ple_precision;

#[cfg(backend = "metal")]
#[path = "../engine/rope_frequency_precision.rs"]
mod rope_frequency_precision;

#[cfg(backend = "metal")]
#[path = "../engine/olmo3_prepared_precision.rs"]
mod olmo3_prepared_precision;

#[path = "../backends/common/kernel/matmul/affine_input_sum_test.rs"]
mod affine_input_sum_test;
