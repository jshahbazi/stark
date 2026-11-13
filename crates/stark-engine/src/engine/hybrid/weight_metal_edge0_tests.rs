//! Edge0's additional packed formats, without a model or backend admission.

use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn edge0_metal_block_values_match_qualified_scalar_values_bitwise() {
    objc2::rc::autoreleasepool(|_| {
        let executor = executor().unwrap();
        let pipelines = std::iter::once(&executor.library)
            .chain(executor.packed_fast_library.as_ref())
            .map(|library| {
                let function = library.new_function_with_name("hybrid_edge0_block_decode_reference").unwrap();
                executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
            })
            .collect::<Vec<_>>();
        // Every nibble/pair value reaches every block position. Scales include
        // both zeros, FP16 subnormal boundaries, normals and the finite maximum.
        for (variant, pipeline) in pipelines.iter().enumerate() {
            for kind in [2, 20, 42] {
                let (block, size) = weights::block_layout(kind).unwrap();
                let mut codes = Vec::new();
                for scale in [0u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3400, 0xb800, 0x7bff] {
                    for phase in 0..16 {
                        codes.extend_from_slice(&scale.to_le_bytes());
                        for i in 0..16 {
                            let value = if kind == 42 {
                                (0..4).fold(0u8, |byte, pair| byte | (((i + pair + phase) & 3) << (2 * pair)))
                            } else {
                                ((i + phase) & 15) | (((i * 7 + phase) & 15) << 4)
                            };
                            codes.push(value);
                        }
                    }
                }
                let blocks = codes.len() / size;
                let count = blocks * block;
                let source = buffer(executor, &codes).unwrap();
                let output = executor.device.new_buffer(count * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
                let command = executor.queue.command_buffer().unwrap();
                let encoder = command.compute_command_encoder().unwrap();
                let parameters = [kind, blocks as u32, block as u32, size as u32];
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(Some(&source), 0, 0);
                encoder.set_buffer(Some(&output), 0, 1);
                encoder.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 2);
                encoder.dispatch_threadgroups(
                    MTLSize {
                        width: count.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                // SAFETY: the completed dispatch wrote count F32 values to this
                // shared allocation, with an explicit tail bound in the kernel.
                let actual = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count) };
                for (index, encoded) in codes.chunks_exact(size).enumerate() {
                    let mut expected = vec![0.0; block];
                    weights::decode_block(kind, encoded, &mut expected).unwrap();
                    for (within, expected) in expected.iter().enumerate() {
                        assert!(
                            actual[index * block + within].to_bits() == expected.to_bits(),
                            "packed decoder bits differ for library={variant} kind={kind} block={index} position={within}"
                        );
                    }
                }
            }
        }
    });
}
