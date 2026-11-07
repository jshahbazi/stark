//! Preserve logical projection shapes in explicitly declared affine profiles.

use super::{
    source_prefill::{SourcePrefillRoute, source_prefill_route},
    *,
};
use crate::backends::{
    common::kernel::matmul::{MatmulA, MatmulB, MatmulDOps, QuantizedB, QuantizedCorrection},
    metal::kernel::SourceScatterRowsMetalKernel,
};

impl MatmulMetalKernel {
    pub(super) fn encode_source_segments(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        commands: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let invalid = || MatmulError::<Metal>::UnsupportedLayout {
            path: "source prefill requires contiguous BF16 affine projections",
        };
        if self.weights_data_type != DataType::BF16
            || self.input_data_type != DataType::BF16
            || self.output_data_type != DataType::BF16
            || arguments.m <= 1
            || !arguments.b_transpose
            || arguments.b_leading_dimension.is_some()
            || !arguments.d_transform.mask().is_empty()
            || arguments.gather_indices.is_some()
        {
            return Err(invalid().into());
        }
        let MatmulA::FullPrecision {
            values,
            offset,
        } = arguments.a
        else {
            return Err(invalid().into());
        };
        let MatmulB::Quantized(weights) = arguments.b else {
            return Err(invalid().into());
        };
        let Some(biases) = weights.biases() else {
            return Err(invalid().into());
        };
        if !weights.params.materializes_affine_input_sum()
            || weights.bits() != 4
            || weights.group_size != 64
            || weights.signed_codes
            || !arguments.k.is_multiple_of(weights.group_size)
        {
            return Err(invalid().into());
        }
        let widths = weights.params.source_output_widths();
        let widths = if widths.is_empty() {
            vec![arguments.n]
        } else {
            widths.to_vec()
        };
        if widths.iter().try_fold(0u32, |sum, &width| sum.checked_add(width)) != Some(arguments.n)
            || widths.iter().any(|&width| width == 0)
        {
            return Err(invalid().into());
        }
        let (m, n, k) = (arguments.m, arguments.n, arguments.k);
        let mut destination = arguments.d;
        let mut start = 0u32;
        for width in widths {
            let codes_offset = start as usize * k as usize / 2;
            let params_offset = start as usize * weights.params.scale_strides().output_stride as usize * 2;
            let codes = weights.codes.subrange(codes_offset..codes_offset + width as usize * k as usize / 2);
            // Preserve the physical group stride when the planes are group-major.
            // Inner primitives use logical n=width and do not recurse here.
            let scales = weights.scales.subrange(params_offset..);
            let biases = biases.subrange(params_offset..);
            let params = weights.params;
            let b = || {
                MatmulB::Quantized(QuantizedB {
                    codes,
                    scales,
                    correction: QuantizedCorrection::Biases(biases),
                    params,
                    mode: weights.mode,
                    group_size: weights.group_size,
                    signed_codes: false,
                })
            };
            let mut temporary = commands.allocate_scratch_for_shape(&[m, width], DataType::BF16)?;
            self.encode_source_contiguous(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values,
                        offset,
                    },
                    b: b(),
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut temporary,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
                    m,
                    n: width,
                    k,
                },
                commands,
            )?;
            if width == n {
                commands.encode_copy(&temporary, destination.reborrow());
            } else {
                if self.source_scatter.is_none() {
                    self.source_scatter = Some(SourceScatterRowsMetalKernel::new(commands.context())?);
                }
                self.source_scatter.as_ref().unwrap().encode(
                    &temporary,
                    destination.reborrow(),
                    m,
                    width,
                    n,
                    start,
                    commands,
                );
            }
            start += width;
        }
        Ok(())
    }

    fn encode_source_contiguous(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        commands: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        match source_prefill_route(arguments.m, arguments.n, arguments.k, commands.context()) {
            SourcePrefillRoute::Wide => self.source_wide.encode(arguments, commands).map_err(MetalError::from),
            SourcePrefillRoute::Qmm => self.gemm.encode_source_qmm(arguments, &self.output_work, commands),
            SourcePrefillRoute::Rows => {
                let MatmulA::FullPrecision {
                    values,
                    offset,
                } = arguments.a
                else {
                    return Err(MatmulError::<Metal>::UnsupportedLayout {
                        path: "source rows require full precision input",
                    }
                    .into());
                };
                let MatmulB::Quantized(weights) = arguments.b else {
                    return Err(MatmulError::<Metal>::UnsupportedLayout {
                        path: "source rows require affine weights",
                    }
                    .into());
                };
                let mut output = arguments.d;
                for row in 0..arguments.m {
                    let begin = row as usize * arguments.n as usize * 2;
                    let row_arguments = MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values,
                            offset: offset + row as usize * arguments.k as usize * 2,
                        },
                        b: MatmulB::Quantized(QuantizedB {
                            codes: weights.codes,
                            scales: weights.scales,
                            correction: weights.correction,
                            params: weights.params,
                            mode: weights.mode,
                            group_size: weights.group_size,
                            signed_codes: weights.signed_codes,
                        }),
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: output.reborrow().subrange_mut(begin..begin + arguments.n as usize * 2),
                        d_transform: MatmulDOps::none(),
                        gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
                        m: 1,
                        n: arguments.n,
                        k: arguments.k,
                    };
                    let shape = MatmulShape::from_arguments(&row_arguments);
                    let MatmulDispatch::Gemv(plan) = self.select_dispatch(&shape, commands.context()) else {
                        return Err(MatmulError::<Metal>::UnsupportedLayout {
                            path: "source row requires GEMV",
                        }
                        .into());
                    };
                    self.gemv.encode_selected(row_arguments, plan, &self.output_work, commands)?;
                }
                Ok(())
            },
        }
    }
}
