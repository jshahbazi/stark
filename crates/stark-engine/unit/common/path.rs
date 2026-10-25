use std::{path::PathBuf, sync::OnceLock};

use crate::tests::env_vars::STARK_TEST_MODEL_PATH;

const MODEL_FILE_NAME: &str = "model.safetensors";

pub fn get_test_model_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(resolve_test_model_path).clone()
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

// Model-bound tests read a model converted with tools/converter; nothing is
// downloaded.
fn resolve_test_model_path() -> PathBuf {
    let path = std::env::var_os(STARK_TEST_MODEL_PATH)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {STARK_TEST_MODEL_PATH} to a converted model directory"));
    assert!(
        path.join(MODEL_FILE_NAME).is_file(),
        "{STARK_TEST_MODEL_PATH} has no {MODEL_FILE_NAME}: {}",
        path.display(),
    );
    path
}
