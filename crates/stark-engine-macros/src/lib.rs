mod dsl;
mod stark_config;
mod stark_test;

use proc_macro::TokenStream;

// CPU kernel DSL
#[proc_macro_attribute]
pub fn kernel(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    dsl::kernel(args, input)
}

// Config DSL
#[proc_macro_attribute]
pub fn stark_config_abstract(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    stark_config::stark_config_abstract(args, input)
}

#[proc_macro_attribute]
pub fn stark_config(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    stark_config::stark_config(args, input)
}

// Test DSL
#[proc_macro_attribute]
pub fn stark_test(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    stark_test::stark_test(args, input)
}

#[proc_macro_attribute]
pub fn stark_bench(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    stark_test::stark_bench(args, input)
}
