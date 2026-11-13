// SPDX-License-Identifier: Apache-2.0
//! Small original-byte window transport and completion ownership checks.

use metal::MTLBuffer;
use stark_engine_macros::stark_test;

use super::*;

struct WindowFixture(PathBuf);

impl WindowFixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-edge0-window-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for WindowFixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[stark_test]
fn edge0_expert_window_transport_preserves_original_q8_bytes_and_graph_ownership() {
    // SAFETY: sysconf has no pointer arguments or ownership effects.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
    assert!(page.is_power_of_two() && page >= 1024 && 3 * page < 1024 * 1024);
    // Include an unaligned record, a record crossing an OS page boundary and
    // a final partial-file page. The synthetic file is always under 1 MiB.
    for (prefix, tail) in [(0, 0), (7, 19), (page - 80, 31)] {
        let weak = objc2::rc::autoreleasepool(|_| {
            let fixture = WindowFixture::new();
            let path = fixture.0.join("synthetic.bin");
            let file_bytes = 3 * page - tail;
            let record_offset = page + prefix;
            let record_bytes = 192;
            let matrix_offset = record_offset + 32;
            let mut source = vec![0xa5; file_bytes];
            let mut expected = Vec::new();
            for row in 0..3 {
                let at = matrix_offset + row * 34;
                source[at..at + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                let mut sum = 0i32;
                for column in 0..32 {
                    let value = ((column + 3 * row) % 17) as i8 - 8;
                    source[at + 2 + column] = value as u8;
                    sum += i32::from(value);
                }
                expected.push(sum as f32 * 0.5);
            }
            fs::write(&path, &source).unwrap();
            drop(source);
            let mapping = Arc::new(Mapping::new(&path).unwrap());
            let weak = Arc::downgrade(&mapping);
            let whole = mapping.metal_slice(0, file_bytes).unwrap();
            let record = mapping.metal_slice(record_offset, record_bytes).unwrap();
            assert!(whole.registered_bytes() > record.registered_bytes());
            assert!(record.subslice(record_bytes, 1).is_err());
            assert!(mapping.metal_slice(file_bytes, 1).is_err());
            let bank = Bank {
                name: "synthetic.expert".to_owned(),
                file: 0,
                offset: record_offset,
                bytes: record_bytes,
            };
            let tensor = Tensor {
                file: 0,
                offset: matrix_offset,
                size: 3 * 34,
                shape: vec![3, 32],
                kind: 8,
                affine: None,
            };
            let selected_offset = bank_matrix_offset(&bank, &tensor).unwrap();
            let (_, record_span_bytes) = mapping.metal_span(record_offset, record_bytes).unwrap();
            assert!(record.registered_bytes() == record_span_bytes);
            let record_page_start = record_offset & !(page - 1);
            let wide_codes = whole.subslice(matrix_offset, tensor.size).unwrap();
            let narrow_codes = record.subslice(selected_offset, tensor.size).unwrap();
            let original = mapping.bytes(matrix_offset, tensor.size).unwrap().as_ptr().cast_mut();
            // Compare addresses as booleans so a failed assertion never emits
            // address values. Neither transport allocates a weight copy.
            assert!(unsafe { wide_codes.buffer().contents().as_ptr().cast::<u8>().add(matrix_offset) } == original);
            assert!(
                unsafe {
                    narrow_codes.buffer().contents().as_ptr().cast::<u8>().add(matrix_offset - record_page_start)
                } == original
            );
            let wide = MappedMatrix {
                codes: wide_codes,
                planes: None,
            };
            let narrow = MappedMatrix {
                codes: narrow_codes,
                planes: None,
            };
            drop(whole);
            drop(record);
            let mut first = DecodeGraph::new().unwrap();
            let input = first.upload_into(&[1.0; 32]).unwrap();
            let wide_output = first.edge0_packed_batch(&wide, MatrixFormat::Packed(8), 3, 32, &input, 1).unwrap();
            let mut second = DecodeGraph::new().unwrap();
            let input = second.upload_into(&[1.0; 32]).unwrap();
            let narrow_output = second.edge0_packed_batch(&narrow, MatrixFormat::Packed(8), 3, 32, &input, 1).unwrap();
            drop(wide);
            drop(narrow);
            drop(mapping);
            drop(fixture);
            assert!(weak.upgrade().is_some());
            let first = first.submit(&[wide_output]).unwrap();
            let second = second.submit(&[narrow_output]).unwrap();
            let wide_values = first.wait().unwrap();
            assert!(weak.upgrade().is_some(), "the second graph lost its mapped owner");
            let narrow_values = second.wait().unwrap();
            assert!(wide_values.len() == 1 && narrow_values.len() == 1);
            assert!(wide_values[0].len() == 3 && narrow_values[0].len() == 3);
            assert!(
                wide_values[0]
                    .iter()
                    .zip(&narrow_values[0])
                    .zip(&expected)
                    .all(|((wide, narrow), expected)| wide.to_bits() == narrow.to_bits()
                        && wide.to_bits() == expected.to_bits()),
                "original-byte window transport changed the Q8 product"
            );
            weak
        });
        for _ in 0..100 {
            if weak.upgrade().is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(weak.upgrade().is_none(), "completed graphs retained the synthetic mapping");
    }
}
