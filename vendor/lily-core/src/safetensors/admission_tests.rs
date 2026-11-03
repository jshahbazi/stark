use super::*;
use serde_json::{Value, json};
use std::{
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Shard(PathBuf);

impl Shard {
    fn bytes(bytes: &[u8]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-shard-admission-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }

    fn tensor(dtype: &str, shape: Value, offsets: [u64; 2], payload: usize) -> Self {
        let header = serde_json::to_vec(&json!({
            "weight": {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        }))
        .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + payload, 7);
        Self::bytes(&bytes)
    }
}

impl Drop for Shard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn shape_small_payload_large_is_rejected_before_any_tensor_read() {
    let shard = Shard::tensor("U32", json!([1, 320]), [0, 65_536], 65_536);
    let error = parse_shard_header(&shard.0).err().unwrap();
    assert!(error.to_string().contains("shape and byte length disagree"));
}

#[test]
fn payload_must_match_shape_in_both_directions() {
    for length in [4, 12] {
        let shard = Shard::tensor("BF16", json!([2, 2]), [0, length], length as usize);
        assert!(parse_shard_header(&shard.0).is_err());
    }
}

#[test]
fn header_length_is_bounded_before_allocation() {
    for length in [MAX_HEADER_BYTES + 1, i64::MAX as u64 - 1, u64::MAX] {
        let shard = Shard::bytes(&length.to_le_bytes());
        let error = parse_shard_header(&shard.0).err().unwrap();
        assert!(error.to_string().contains("header too large"));
    }
    let shard = Shard::bytes(&64u64.to_le_bytes());
    assert!(
        parse_shard_header(&shard.0).err().unwrap().to_string().contains("truncated")
    );
}

#[test]
fn malformed_offsets_and_shape_arithmetic_fail_closed() {
    for offsets in [[8, 4], [0, 12], [u64::MAX - 1, u64::MAX]] {
        let shard = Shard::tensor("U32", json!([1]), offsets, 4);
        assert!(parse_shard_header(&shard.0).is_err());
    }
    for shape in [json!([usize::MAX, 2]), json!([usize::MAX])] {
        let shard = Shard::tensor("U32", shape, [0, 4], 4);
        assert!(parse_shard_header(&shard.0).is_err());
    }
}

#[test]
fn valid_native_dtypes_scalars_empty_tensors_and_unused_dtypes_remain_supported() {
    for (dtype, width) in [("BF16", 2), ("F32", 4), ("U32", 4), ("I64", 8), ("U64", 8)]
    {
        let shard =
            Shard::tensor(dtype, json!([2, 3]), [0, 6 * width], (6 * width) as usize);
        let tensors = parse_shard_header(&shard.0).unwrap();
        let ckpt = Checkpoint { tensors };
        assert_eq!(ckpt.read("weight").unwrap(), vec![7; (6 * width) as usize]);
    }
    for (shape, bytes) in [(json!([]), 4), (json!([usize::MAX, 2, 0]), 0)] {
        let shard = Shard::tensor("U32", shape, [0, bytes], bytes as usize);
        assert!(parse_shard_header(&shard.0).is_ok());
    }
    let shard = Shard::tensor("F8_E4M3", json!([4]), [0, 4], 4);
    let ckpt = Checkpoint { tensors: parse_shard_header(&shard.0).unwrap() };
    assert!(ckpt.read("weight").is_err());
}

#[test]
fn unused_dtype_still_requires_a_file_bounded_interval() {
    let shard = Shard::tensor("F8_E4M3", json!([4]), [0, 8], 4);
    assert!(parse_shard_header(&shard.0).is_err());
}

#[test]
#[ignore = "requires an operator-selected local native Lily checkpoint"]
fn local_native_checkpoint_headers_remain_admissible() {
    let path = std::env::var_os("STARK_TEST_LILY_CHECKPOINT")
        .expect("set STARK_TEST_LILY_CHECKPOINT for this opt-in check");
    let checkpoint = Checkpoint::open(path);
    assert!(checkpoint.is_ok(), "native checkpoint metadata admission failed");
    assert!(checkpoint.unwrap().names().next().is_some());
}

#[cfg(unix)]
#[test]
fn regular_cached_shards_can_be_symlinked() {
    let shard = Shard::tensor("U32", json!([1]), [0, 4], 4);
    let link = Shard(shard.0.with_extension("symlink"));
    std::os::unix::fs::symlink(&shard.0, &link.0).unwrap();
    assert!(parse_shard_header(&link.0).is_ok());
    assert!(parse_shard_header(std::env::temp_dir().as_path()).is_err());
}
