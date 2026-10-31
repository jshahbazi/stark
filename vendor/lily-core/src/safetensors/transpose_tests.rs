//! Exact bit controls for canonical Uzu affine-parameter loading.
use super::*;

// Original group-outer byte implementation from 66b29b16, retained only as a
// test oracle. It also defines padding failure-side partial destination writes.
fn reference(
    physical: &[u8],
    output: &mut [u8],
    rows: usize,
    groups: usize,
    stride: usize,
) -> Result<()> {
    for group in 0..groups {
        let padding =
            &physical[2 * (group * stride + rows)..2 * ((group + 1) * stride)];
        ensure!(padding.iter().all(|&byte| byte == 0), "nonzero Uzu affine padding");
        for row in 0..rows {
            let from = 2 * (group * stride + row);
            let to = 2 * (row * groups + group);
            output[to..to + 2].copy_from_slice(&physical[from..from + 2]);
        }
    }
    Ok(())
}

fn plane(rows: usize, groups: usize, seed: u16) -> (Vec<u8>, Vec<u8>, usize) {
    let stride = rows.div_ceil(4) * 4;
    let mut physical = vec![0; groups * stride * 2];
    let mut logical = vec![0; rows * groups * 2];
    for row in 0..rows {
        for group in 0..groups {
            let value = ((row * groups + group) as u16)
                .wrapping_mul(40_503)
                .wrapping_add(seed)
                .to_le_bytes();
            let from = 2 * (row * groups + group);
            let to = 2 * (group * stride + row);
            logical[from..from + 2].copy_from_slice(&value);
            physical[to..to + 2].copy_from_slice(&value);
        }
    }
    (physical, logical, stride)
}

#[test]
fn restore_preserves_row_group_tails_and_zero_padding() {
    for rows in
        [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257]
    {
        for groups in [1, 2, 3, 7, 8, 9, 10, 15, 16, 17, 40, 160] {
            let (physical, expected, stride) = plane(rows, groups, 0x8001);
            let mut original = vec![0xa5; expected.len()];
            let mut actual = vec![0x5a; expected.len()];
            reference(&physical, &mut original, rows, groups, stride).unwrap();
            restore_group_output(&physical, &mut actual, rows, groups, stride).unwrap();
            assert!(
                original == expected && actual == expected,
                "transpose changed source bits"
            );
        }
    }
}

#[test]
fn restore_preserves_every_u16_encoding() {
    for groups in [1, 3, 8, 9, 10, 16, 17] {
        let rows = 65_537;
        let (physical, expected, stride) = plane(rows, groups, 0);
        let mut seen = vec![false; 65_536];
        for bytes in expected.chunks_exact(2) {
            seen[u16::from_le_bytes(bytes.try_into().unwrap()) as usize] = true;
        }
        assert!(seen.iter().all(|value| *value), "u16 control missed an encoding");
        let mut actual = vec![0; expected.len()];
        restore_group_output(&physical, &mut actual, rows, groups, stride).unwrap();
        assert!(actual == expected, "ordinary or exceptional BF16 encoding changed");
    }
}

#[test]
fn malformed_padding_preserves_error_and_partial_destination() {
    for rows in [1, 3, 5, 63, 65] {
        for groups in [1, 3, 8, 10, 17] {
            let (valid, expected, stride) = plane(rows, groups, 0x7fc1);
            for invalid_group in 0..groups {
                for byte in 0..2 * (stride - rows) {
                    let mut physical = valid.clone();
                    physical[2 * (invalid_group * stride + rows) + byte] = 0xff;
                    for shift in [0, 1] {
                        let mut source = vec![0x33; physical.len() + shift + 8];
                        source[shift..shift + physical.len()]
                            .copy_from_slice(&physical);
                        let mut original = vec![0xa5; expected.len() + shift + 8];
                        let mut actual = original.clone();
                        let error = reference(
                            &source[shift..shift + physical.len()],
                            &mut original[shift..shift + expected.len()],
                            rows,
                            groups,
                            stride,
                        )
                        .unwrap_err()
                        .to_string();
                        let actual_error = restore_group_output(
                            &source[shift..shift + physical.len()],
                            &mut actual[shift..shift + expected.len()],
                            rows,
                            groups,
                            stride,
                        )
                        .unwrap_err()
                        .to_string();
                        assert!(
                            error == "nonzero Uzu affine padding"
                                && actual_error == error,
                            "padding failure diagnostic changed"
                        );
                        assert!(
                            actual == original,
                            "padding failure changed partial writes or canaries"
                        );
                        restore_group_output(
                            &valid,
                            &mut actual[shift..shift + expected.len()],
                            rows,
                            groups,
                            stride,
                        )
                        .unwrap();
                        assert!(
                            actual[shift..shift + expected.len()] == expected,
                            "valid recovery changed bits"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn restore_misaligned_source_destination_and_canaries() {
    for rows in [1, 7, 8, 9, 63, 64, 65, 129] {
        for groups in [1, 7, 8, 9, 10, 15, 16, 17, 40] {
            let (physical, expected, stride) = plane(rows, groups, 0x7fff);
            for source_shift in [0, 1, 2, 15] {
                let mut source = vec![0x33; physical.len() + source_shift + 16];
                source[source_shift..source_shift + physical.len()]
                    .copy_from_slice(&physical);
                for destination_shift in [0, 1, 2, 15] {
                    let mut destination =
                        vec![0xa5; expected.len() + destination_shift + 16];
                    restore_group_output(
                        &source[source_shift..source_shift + physical.len()],
                        &mut destination
                            [destination_shift..destination_shift + expected.len()],
                        rows,
                        groups,
                        stride,
                    )
                    .unwrap();
                    assert!(
                        destination
                            [destination_shift..destination_shift + expected.len()]
                            == expected,
                        "misalignment changed transpose bits"
                    );
                    assert!(
                        destination[..destination_shift]
                            .iter()
                            .chain(
                                destination[destination_shift + expected.len()..]
                                    .iter()
                            )
                            .all(|&byte| byte == 0xa5),
                        "destination canary changed"
                    );
                }
                assert!(
                    source[..source_shift]
                        .iter()
                        .chain(source[source_shift + physical.len()..].iter())
                        .all(|&byte| byte == 0x33),
                    "source canary changed"
                );
            }
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_tile_bounds_cover_all_lanes_and_strides() {
    assert!(transpose_u16_tile_bounds(128, 128, 0, 0, 8, 8), "valid tile rejected");
    for (source, dest, row, group, groups, stride) in [
        (127, 128, 0, 0, 8, 8),
        (128, 127, 0, 0, 8, 8),
        (128, 128, 1, 0, 8, 8),
        (128, 128, 0, 1, 8, 8),
        (128, 128, usize::MAX, 0, 8, 8),
        (128, 128, 0, usize::MAX, 8, 8),
        (usize::MAX, usize::MAX, 0, 0, usize::MAX, usize::MAX),
    ] {
        assert!(
            !transpose_u16_tile_bounds(source, dest, row, group, groups, stride),
            "invalid tile bounds admitted"
        );
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-transpose-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn read_with_preserves_destination_and_read_failure_order() {
    let directory = Directory::new();
    let path = directory.0.join("plane.bin");
    let (physical, expected, stride) = plane(65, 10, 0xff81);
    std::fs::write(&path, &physical).unwrap();
    let checkpoint = Checkpoint {
        tensors: HashMap::from([(
            "plane".into(),
            TensorMeta {
                dtype: SafetensorsDType::BF16,
                shape: vec![65, 10],
                shard: path.clone(),
                start: 0,
                end: physical.len() as u64,
                group_output: Some((65, 10, stride)),
            },
        )]),
    };
    let mut actual = vec![0xa5; expected.len()];
    checkpoint.read_with("plane", |_| Ok(actual.as_mut_slice())).unwrap();
    assert!(actual == expected, "read_with changed exact output");
    std::fs::remove_file(&path).unwrap();
    let mut short = vec![0x5a; expected.len() - 1];
    let error = checkpoint
        .read_with("plane", |_| Ok(short.as_mut_slice()))
        .unwrap_err()
        .to_string();
    assert!(
        error.starts_with("destination for plane is "),
        "destination check moved after file open"
    );
    assert!(
        short.iter().all(|&byte| byte == 0x5a),
        "destination-size failure changed output"
    );
    std::fs::write(&path, &physical[..physical.len() - 1]).unwrap();
    actual.fill(0xa5);
    let error = checkpoint
        .read_with("plane", |_| Ok(actual.as_mut_slice()))
        .unwrap_err()
        .to_string();
    assert!(error == "reading plane", "truncated source context changed");
    assert!(
        actual.iter().all(|&byte| byte == 0xa5),
        "truncated source changed destination"
    );
}
