//! Arithmetic-preserving prefill for explicitly declared MLX affine profiles.
//!
//! Dispatch thresholds and the wide-vector schedule are adapted from MLX
//! v0.32.3, revision 64ea011cb65f14d9ce2737e60db9a4ae91ed7441.
//! Copyright (c) 2023-2026 Apple Inc. MIT license: LICENSE.mlx-reference.

use std::collections::{HashMap, hash_map::Entry};

use crate::backends::{
    common::{
        BufferMut, BufferRef, CommandBufferEncoding,
        kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError},
    },
    metal::{
        Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, kernel::SourceWideQmvMetalKernel,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourcePrefillRoute {
    Wide,
    Qmm,
    Rows,
}

fn architecture(architecture: &str) -> (u32, u8) {
    let bytes = architecture.as_bytes();
    if bytes.len() < 3 {
        return (0, bytes.last().copied().unwrap_or_default());
    }
    let digit = |byte: u8| u32::from(byte.checked_sub(b'0').filter(|&n| n <= 9).unwrap_or_default());
    (10 * digit(bytes[bytes.len() - 3]) + digit(bytes[bytes.len() - 2]), bytes[bytes.len() - 1])
}

fn qmv_batch_limit(
    n: u32,
    k: u32,
    generation: u32,
    size: u8,
) -> u32 {
    let bucket = if n <= 2048 && k <= 2048 {
        0
    } else if n <= 4096 && k <= 4096 {
        1
    } else {
        2
    };
    let limits = if generation >= 17 && size != b'd' {
        [33, 25, 13]
    } else if generation >= 15 && size != b'd' {
        [13, 15, 13]
    } else if size == b'd' {
        [32, 18, 12]
    } else if generation >= 13 {
        [14, 10, 6]
    } else {
        [18, 12, 10]
    };
    limits[bucket]
}

pub fn source_prefill_route(
    m: u32,
    n: u32,
    k: u32,
    context: &MetalContext,
) -> SourcePrefillRoute {
    let (generation, size) = architecture(&context.gpu_architecture);
    if m >= qmv_batch_limit(n, k, generation, size) {
        SourcePrefillRoute::Qmm
    } else if m >= 2 && generation >= 15 {
        SourcePrefillRoute::Wide
    } else {
        SourcePrefillRoute::Rows
    }
}

pub struct SourceWideQmv {
    pipelines: HashMap<(u32, u32), SourceWideQmvMetalKernel>,
}

impl SourceWideQmv {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    pub fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulA::FullPrecision {
            values: input,
            offset,
        } = arguments.a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "SourceWideQmv",
                reason: "BF16 input required",
            });
        };
        let MatmulB::Quantized(weights) = arguments.b else {
            return Err(MatmulError::UnsupportedLayout {
                path: "SourceWideQmv quantized weights",
            });
        };
        let Some(biases) = weights.biases() else {
            return Err(MatmulError::UnsupportedLayout {
                path: "SourceWideQmv affine biases",
            });
        };
        if !weights.params.materializes_affine_input_sum()
            || weights.bits() != 4
            || weights.signed_codes
            || !arguments.b_transpose
            || !matches!(weights.group_size, 16 | 32 | 64 | 128)
            || !arguments.k.is_multiple_of(weights.group_size)
            || arguments.m < 2
            || arguments.n == 0
            || !arguments.d_transform.mask().is_empty()
            || arguments.gather_indices.is_some()
        {
            return Err(MatmulError::UnsupportedLayout {
                path: "SourceWideQmv BF16 affine profile",
            });
        }
        // MLX fills vector tiles evenly, up to five vectors per group.
        let vectors = arguments.m.div_ceil(arguments.m.div_ceil(5));
        let kernel = match self.pipelines.entry((weights.group_size, vectors)) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(
                SourceWideQmvMetalKernel::new(command_buffer.context(), weights.group_size, vectors)
                    .map_err(MatmulError::BackendError)?,
            ),
        };
        let strides = weights.params.scale_strides();
        kernel.encode(
            weights.codes,
            weights.scales,
            biases,
            input.subrange(offset..),
            arguments.d,
            arguments.k,
            arguments.n,
            arguments.m,
            arguments.b_leading_dimension.unwrap_or(arguments.k) / 2,
            strides.output_stride,
            strides.group_stride,
            command_buffer,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::{architecture, qmv_batch_limit};

    #[stark_test]
    fn source_architecture_and_projection_limits() {
        assert_eq!(architecture("applegpu_g17s"), (17, b's'));
        assert_eq!(architecture("applegpu_g15d"), (15, b'd'));
        assert_eq!(architecture(""), (0, 0));
        assert_eq!(architecture("gxs"), (0, b's'));
        assert_eq!(qmv_batch_limit(2048, 1536, 17, b's'), 33);
        assert_eq!(qmv_batch_limit(2560, 1536, 17, b's'), 25);
        assert_eq!(qmv_batch_limit(4096, 4096, 17, b's'), 25);
        assert_eq!(qmv_batch_limit(12288, 4096, 17, b's'), 13);
        assert_eq!(qmv_batch_limit(256, 1536, 15, b's'), 13);
        assert_eq!(qmv_batch_limit(2560, 1536, 16, b's'), 15);
        assert_eq!(qmv_batch_limit(8192, 4096, 15, b'd'), 12);
        assert_eq!(qmv_batch_limit(8192, 4096, 14, b's'), 6);
        assert_eq!(qmv_batch_limit(8192, 4096, 12, b's'), 10);
    }

    #[stark_test]
    fn source_wide_retains_f32_weights_before_the_final_bf16_store() {
        use half::bf16;

        use crate::{
            backends::{
                common::{
                    Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
                    gpu_types::QuantizationMode,
                    kernel::matmul::{
                        MatmulA, MatmulArguments, MatmulB, MatmulDOps, QuantParams, QuantParamsLayout, QuantizedB,
                        QuantizedCorrection,
                    },
                },
                metal::{Metal, MetalContext},
            },
            tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        };
        let context = MetalContext::new().expect("Metal context");
        let mut kernel = super::SourceWideQmv::new();
        let (n, k) = (9usize, 256usize);
        for group_size in [16usize, 32, 64, 128] {
            let groups = k / group_size;
            let mut codes = vec![0u8; n * k / 2];
            for row in 0..n {
                codes[row * k / 2] = 1;
            }
            let codes = create_buffer_with_data::<Metal, u8>(&context, &codes);
            let scales =
                create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::from_f32(1.0 / 512.0); n * groups]);
            let biases = create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ONE; n * groups]);
            for m in [2usize, 9, 18] {
                let mut input = vec![bf16::ZERO; m * k];
                for row in input.chunks_exact_mut(k) {
                    row[0] = bf16::ONE;
                    row[1] = -bf16::ONE;
                }
                let input = create_buffer_with_data::<Metal, bf16>(&context, &input);
                for layout in [QuantParamsLayout::OutputGroup, QuantParamsLayout::GroupOutput] {
                    let mut output = create_buffer::<Metal, bf16>(&context, m * n);
                    let mut commands = context.create_command_buffer(None, None).unwrap();
                    kernel
                        .encode(
                            MatmulArguments {
                                a: MatmulA::FullPrecision {
                                    values: &input,
                                    offset: 0,
                                },
                                b: MatmulB::Quantized(QuantizedB {
                                    codes: &codes,
                                    scales: &scales,
                                    correction: QuantizedCorrection::Biases(&biases),
                                    params: QuantParams::new(layout, n as u32, groups as u32)
                                        .with_materialized_affine_input_sum(true),
                                    mode: QuantizationMode::U4,
                                    group_size: group_size as u32,
                                    signed_codes: false,
                                }),
                                b_leading_dimension: None,
                                b_transpose: true,
                                d: &mut output,
                                d_transform: MatmulDOps::none(),
                                gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
                                m: m as u32,
                                n: n as u32,
                                k: k as u32,
                            },
                            &mut commands,
                        )
                        .unwrap();
                    commands.end_encoding().submit().wait_until_completed().unwrap();
                    // Independent exact dot: (1 + 2^-9) - 1 = 2^-9.
                    // BF16 dequantization would erase the difference entirely.
                    assert!(buffer_to_vec::<Metal, bf16>(&output).iter().all(|x| x.to_f32() == 1.0 / 512.0));
                }
            }
        }
    }
}
