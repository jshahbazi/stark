// The CXX fork compiles its builtins in XGrammar's bridge. Link the engine so
// these tests exercise that same native runtime and its deferred build path.
extern crate stark_engine as _;

#[path = "../../../vendor/cxx/tests/cxx_string.rs"]
mod upstream_cxx_string;
