mod stark_tool;

use proc_macro::TokenStream;

// Tool calls
#[proc_macro]
pub fn stark_tool_closure(input: TokenStream) -> TokenStream {
    stark_tool::stark_tool_closure(input)
}

#[proc_macro_attribute]
pub fn stark_tool_function(
    args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    stark_tool::stark_tool_function(args, input)
}
