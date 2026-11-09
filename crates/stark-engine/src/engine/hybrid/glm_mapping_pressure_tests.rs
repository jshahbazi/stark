// SPDX-License-Identifier: Apache-2.0
//! Private capped-fixture discriminator. The original GGUF is never opened.
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;

const SPAN: usize = 23_085_056;
const OFFSET: usize = 9_408;
const BYTES: usize = 23_068_672;
const ROWS: usize = 8192;
const COLUMNS: usize = 4096;
const CAP: usize = 512 * 1024 * 1024;
static LIVE_MAPS: AtomicUsize = AtomicUsize::new(0);
static UNMAP_FAILURES: AtomicUsize = AtomicUsize::new(0);

struct FixtureMapping {
    pointer: NonNull<c_void>,
    length: usize,
    _file: File,
}
// SAFETY: this owns an immutable read-only mmap and its descriptor. No writes
// or unmapping occur until every Arc, including the Metal deallocator, drops.
unsafe impl Send for FixtureMapping {}
unsafe impl Sync for FixtureMapping {}

impl FixtureMapping {
    fn new(
        file: &File,
        length: usize,
    ) -> Arc<Self> {
        use std::os::fd::AsRawFd;
        assert!((SPAN..=CAP).contains(&length));
        let retained_file = file.try_clone().unwrap();
        // SAFETY: the caller already verified a regular private fixture of
        // exactly CAP bytes. The owner uniquely unmaps this read-only range.
        let raw = unsafe {
            libc::mmap(std::ptr::null_mut(), length, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0)
        };
        assert!(raw != libc::MAP_FAILED, "private mapping failed");
        let pointer = NonNull::new(raw).unwrap_or_else(|| {
            // SAFETY: release the successful mapping on an unexpected null.
            unsafe {
                libc::munmap(raw, length);
            }
            panic!("null private mapping")
        });
        LIVE_MAPS.fetch_add(1, Ordering::AcqRel);
        Arc::new(Self {
            pointer,
            length,
            _file: retained_file,
        })
    }
}

impl Drop for FixtureMapping {
    fn drop(&mut self) {
        // SAFETY: pointer/length identify precisely this live owned mapping.
        if unsafe { libc::munmap(self.pointer.as_ptr(), self.length) } == 0 {
            LIVE_MAPS.fetch_sub(1, Ordering::AcqRel);
        } else {
            UNMAP_FAILURES.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn mark(
    started: Instant,
    phase: &'static str,
    initialized: bool,
    details: serde_json::Value,
) {
    println!(
        "STARK_GLM_MAPPING16 {}",
        serde_json::json!({"protocol":"glm-capped-q5-mapping-v16","phase":phase,
            "pid":std::process::id(),"elapsed_ns":started.elapsed().as_nanos() as u64,
            "metal_allocated_bytes":initialized.then(|| allocated_bytes().unwrap()),
            "live_cpu_mappings":LIVE_MAPS.load(Ordering::Acquire),
            "unmap_failures":UNMAP_FAILURES.load(Ordering::Acquire),"details":details})
    );
    std::io::stdout().flush().unwrap();
}

#[stark_test]
#[ignore]
fn glm_actual_q5_capped_mapping_pressure_control() {
    assert_eq!(std::env::var("STARK_TEST_GLM_MAPPING16").as_deref(), Ok("1"));
    assert!(std::env::vars_os().all(|(key, _)| {
        let key = key.to_string_lossy();
        !key.starts_with("STARK_GLM53_") && !key.starts_with("STARK_HYBRID_") && !key.starts_with("STARK_QWEN38_")
    }));
    let arm = std::env::var("STARK_TEST_GLM_MAPPING16_ARM").unwrap();
    let (fixture_mib, map_length) = match arm.as_str() {
        "owned" => (512, None),
        "full512" => (512, Some(CAP)),
        "dedicated512" => (512, Some(SPAN)),
        _ => panic!("unregistered mapping arm"),
    };
    let path = PathBuf::from(std::env::var_os("STARK_TEST_GLM_MAPPING16_FIXTURE").unwrap());
    // Reject original GGUF/model paths and every large file before reads/maps.
    assert_eq!(path.file_name().unwrap().to_str().unwrap(), format!("actual-q5-private-{fixture_mib}mib.bin"));
    assert_eq!(path.parent().unwrap().file_name().unwrap(), "mapping16-private-fixtures02");
    let expected_hash = std::env::var("STARK_TEST_GLM_MAPPING16_SPAN_SHA256").unwrap();
    assert!(
        expected_hash.len() == 64 && expected_hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
    let format = MatrixFormat::Packed(13);
    assert!(block_schedule(format) && !factored_schedule(format) && !q5_grouped32_shape(format, COLUMNS));
    assert_eq!(matrix_threads(format, ROWS, COLUMNS), 64);
    assert_eq!(matrix_groups(format, ROWS, COLUMNS), 4096);
    let started = Instant::now();
    mark(started, "waiting_for_owner_start", false, serde_json::json!({"arm":arm}));
    let mut gate = [0u8; 1];
    std::io::stdin().read_exact(&mut gate).unwrap();
    assert!(gate == *b"G", "invalid mapping owner gate");
    mark(started, "before_weight_library", false, serde_json::json!({}));
    initialize().unwrap();
    mark(started, "weight_library_ready", true, serde_json::json!({}));
    objc2::rc::autoreleasepool(|_| {
        let file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(&path).unwrap();
        let before = file.metadata().unwrap();
        assert!(before.is_file() && before.len() == fixture_mib * 1024 * 1024 && before.len() <= CAP as u64);
        assert!(std::fs::symlink_metadata(&path).unwrap().is_file());
        let mut bytes = vec![0u8; SPAN];
        file.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected_hash, "fixture view hash changed");
        mark(
            started,
            "private_fixture_validated",
            true,
            serde_json::json!({
            "file_bytes":before.len(),"logical_bytes":BYTES,"registered_bytes":SPAN,"logical_offset":OFFSET,
                "vm_mapping_bytes":map_length.unwrap_or(0),
                "background_cpu_mapping_bytes":if arm == "dedicated512" { CAP } else { 0 }}),
        );
        mark(started, "before_gpu_buffer", true, serde_json::json!({}));
        // Production keeps its CPU full-file Mapping alive while a Metal view
        // exists. Retain that same bounded background mapping in the dedicated
        // arm, so the direct per-view ownership choice is tested faithfully.
        let _cpu_owner = (arm == "dedicated512").then(|| FixtureMapping::new(&file, CAP));
        let codes = if let Some(length) = map_length {
            let owner = FixtureMapping::new(&file, length);
            // SAFETY: the fixed registered span is inside the initialized
            // mapped pages; the retained Arc prevents premature munmap.
            unsafe { mapped_slice(owner.pointer, SPAN, OFFSET, BYTES, owner).unwrap() }
        } else {
            MappedSlice {
                buffer: buffer(executor().unwrap(), &bytes).unwrap(),
                offset: OFFSET,
                length: BYTES,
                registered_length: SPAN,
            }
        };
        let operation = MappedOperation {
            matrices: vec![MappedMatrix {
                codes,
                planes: None,
            }],
            inputs: &[],
            rows: ROWS,
            columns: COLUMNS,
            rows_per_batch: 0,
            format,
        };
        mark(started, "gpu_buffer_ready", true, serde_json::json!({}));
        let values: Vec<_> = (0..COLUMNS).map(|i| (((i * 7919) % 65521) as f32 - 32760.0) / 16384.0).collect();
        let input = DecodeGraph::upload(&values).unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        let output = graph.mapped(&operation, &input).unwrap();
        mark(started, "q5_encoded", true, serde_json::json!({}));
        let output = graph.finish(&[output]).unwrap().remove(0);
        mark(started, "q5_completed", true, serde_json::json!({}));
        assert_eq!(output.len(), ROWS);
        assert!(output.iter().all(|x| x.is_finite()));
        let oracle = tests::packed_f64_oracle(13, &bytes[OFFSET..OFFSET + BYTES], ROWS, COLUMNS, &values, 0);
        assert!(oracle.iter().any(|(x, _)| x.abs() > 1e-6), "empty actual Q5 control");
        for (&actual, &(expected, absolute)) in output.iter().zip(&oracle) {
            assert!((f64::from(actual) - expected).abs() <= 2e-5 * absolute.max(1.0), "actual Q5 F64 projection error");
        }
        let after = file.metadata().unwrap();
        assert!(
            before.dev() == after.dev()
                && before.ino() == after.ino()
                && before.len() == after.len()
                && before.mtime() == after.mtime()
                && before.mtime_nsec() == after.mtime_nsec(),
            "fixture descriptor changed"
        );
        mark(
            started,
            "oracle_passed",
            true,
            serde_json::json!({
            "output_sha256":format!("{:x}", Sha256::digest(bytemuck::cast_slice::<f32, u8>(&output))),
            "output_rows":ROWS,"finite":true}),
        );
    });
    assert_eq!(LIVE_MAPS.load(Ordering::Acquire), 0, "private mapping owner retained");
    assert_eq!(UNMAP_FAILURES.load(Ordering::Acquire), 0, "private unmap failed");
    mark(started, "test_buffer_owners_drained", true, serde_json::json!({}));
    mark(started, "control_passed", true, serde_json::json!({}));
}
