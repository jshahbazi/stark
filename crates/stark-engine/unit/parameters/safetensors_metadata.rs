use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

use stark_engine_macros::stark_test;

use crate::{
    parameters::safetensors_metadata::{HeaderLoadingError, read_metadata, read_metadata_with_limit},
    tests::path::get_test_weights_path,
};

fn header_file(
    advertised: u64,
    header: &[u8],
) -> File {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "stark-bounded-header-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    file.write_all(&advertised.to_le_bytes()).unwrap();
    file.write_all(header).unwrap();
    file
}

#[stark_test]
fn explicit_header_limit_precedes_allocation_and_json_read() {
    const LIMIT: usize = 8 * 1024 * 1024;
    let file = header_file((LIMIT + 1) as u64, &[]);
    assert!(file.metadata().unwrap().len() == 8);
    assert!(matches!(read_metadata_with_limit(&file, LIMIT), Err(HeaderLoadingError::HeaderTooLarge)));
}

#[stark_test]
fn explicit_header_limit_keeps_exact_boundary_and_standard_ceiling() {
    let empty = header_file(2, b"{}");
    let (offset, metadata) = read_metadata_with_limit(&empty, 2).unwrap();
    assert!(offset == 10 && metadata.tensors.is_empty() && metadata.metadata.is_none());
    assert!(read_metadata(&empty).unwrap().0 == offset);
    for limit in [0, 1] {
        assert!(matches!(read_metadata_with_limit(&empty, limit), Err(HeaderLoadingError::HeaderTooLarge)));
    }
    let oversized = header_file(100_000_001, &[]);
    assert!(matches!(read_metadata_with_limit(&oversized, usize::MAX), Err(HeaderLoadingError::HeaderTooLarge)));
    assert!(matches!(read_metadata(&oversized), Err(HeaderLoadingError::HeaderTooLarge)));
}

#[stark_test]
#[ignore] // needs STARK_TEST_MODEL_PATH
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}
