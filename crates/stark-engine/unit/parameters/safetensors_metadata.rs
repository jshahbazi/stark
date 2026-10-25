use std::fs::File;

use stark_engine_macros::stark_test;

use crate::{parameters::safetensors_metadata::read_metadata, tests::path::get_test_weights_path};

#[stark_test]
#[ignore] // needs STARK_TEST_MODEL_PATH
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}
