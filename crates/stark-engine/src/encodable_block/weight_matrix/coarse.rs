//! A draft-only approximation. Target matrix storage and readout stay intact.
use half::bf16;

use super::*;
use crate::backends::common::{Buffer, BufferRef, Context};

impl<B: Backend> WeightMatrix<B> {
    /// Drop low integer bits within each existing affine group. Midpoint bias
    /// minimizes the interval error; this defines a new proposal distribution,
    /// never an alternative target-weight representation.
    pub(crate) fn coarse_affine(
        &self,
        context: &B::Context,
        output_dim: u32,
        input_dim: u32,
        coarse_bits: u32,
    ) -> Result<Option<Self>, WeightMatrixError<B>> {
        let Some(q) = &self.quantized else {
            return Ok(None);
        };
        let bits = q.info.mode.bits();
        let Some(source_biases) = q.correction.biases() else {
            return Ok(None);
        };
        if !matches!(coarse_bits, 3 | 4)
            || !(5..=8).contains(&bits)
            || q.signed_codes
            || q.info.group_size != 64
            || q.params.layout() != QuantParamsLayout::GroupOutput
            || !input_dim.is_multiple_of(64)
        {
            return Ok(None);
        }
        let source = self.values.as_ref().as_slice::<u8>();
        let expected = output_dim as usize * input_dim as usize * bits as usize / 8;
        let parameter_count = (input_dim / 64) as usize * output_dim.next_multiple_of(4) as usize;
        if source.len() != expected
            || q.scales.as_ref().size() != parameter_count * 2
            || source_biases.as_ref().size() != parameter_count * 2
        {
            return Ok(None);
        }
        let backend = WeightMatrixError::ParameterError;
        let allocate =
            |bytes| context.create_buffer(bytes).map_err(|error| backend(ParameterLoaderError::BackendError(error)));
        let mut values = allocate(output_dim as usize * input_dim as usize * coarse_bits as usize / 8)?;
        let codes = (&mut values).as_slice_mut::<u8>();
        let shift = bits - coarse_bits;
        // Complete rows/groups guarantee byte-aligned eight-code packs.
        for (pack, out) in source.chunks_exact(bits as usize).zip(codes.chunks_exact_mut(coarse_bits as usize)) {
            out.fill(0);
            for index in 0..8usize {
                let bit = index * bits as usize;
                let byte = bit / 8;
                let offset = bit % 8;
                let mut word = u16::from(pack[byte]) >> offset;
                if offset + bits as usize > 8 {
                    word |= u16::from(pack[byte + 1]) << (8 - offset);
                }
                let code = ((word & ((1u16 << bits) - 1)) >> shift) as u8;
                let destination = index * coarse_bits as usize;
                out[destination / 8] |= code << (destination % 8);
                if destination % 8 + coarse_bits as usize > 8 {
                    out[destination / 8 + 1] |= code >> (8 - destination % 8);
                }
            }
        }
        let mut scales = allocate(parameter_count * 2)?;
        let mut biases = allocate(parameter_count * 2)?;
        let factor = (1u32 << shift) as f32;
        let midpoint = (factor - 1.0) * 0.5;
        for (((out_scale, out_bias), &scale), &bias) in (&mut scales)
            .as_slice_mut::<bf16>()
            .iter_mut()
            .zip((&mut biases).as_slice_mut::<bf16>())
            .zip(q.scales.as_ref().as_slice::<bf16>())
            .zip(source_biases.as_ref().as_slice::<bf16>())
        {
            let scale = f32::from(scale);
            let bias = f32::from(bias);
            if !scale.is_finite() || !bias.is_finite() {
                return Ok(None);
            }
            *out_scale = bf16::from_f32(scale * factor);
            *out_bias = bf16::from_f32(bias + scale * midpoint);
            if !out_scale.is_finite() || !out_bias.is_finite() {
                return Ok(None);
            }
        }
        self.representation_copy_bytes.fetch_add(values.size() + scales.size() + biases.size(), Ordering::Relaxed);
        Ok(Some(Self {
            values: Arc::new(values),
            representation_copy_bytes: self.representation_copy_bytes.clone(),
            quantized: Some(Quantized {
                scales: Arc::new(scales),
                correction: QuantizedCorrection::Biases(Arc::new(biases)),
                params: q.params,
                info: QuantizationInfo {
                    mode: if coarse_bits == 3 {
                        QuantizationMode::U3
                    } else {
                        QuantizationMode::U4
                    },
                    method: QuantizationMethod::ScaleBias,
                    group_size: 64,
                },
                signed_codes: false,
            }),
        }))
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{backends::cpu::Cpu, tests::helpers::create_buffer_with_data};

    #[stark_test]
    fn shared_raw_codes_are_isolated_from_a8_sign_conversion() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        for mode in [QuantizationMode::U8, QuantizationMode::I8, QuantizationMode::U4] {
            let raw = Arc::new(create_buffer_with_data::<Cpu, u8>(&context, &[0, 127, 128, 255]));
            let mut matrix = WeightMatrix::<Cpu> {
                values: raw.clone(),
                representation_copy_bytes: Arc::new(AtomicUsize::new(0)),
                quantized: Some(Quantized {
                    scales: Arc::new(create_buffer_with_data::<Cpu, bf16>(&context, &[bf16::ONE; 4])),
                    correction: QuantizedCorrection::Symmetric,
                    params: QuantParams::new(QuantParamsLayout::GroupOutput, 4, 1),
                    info: QuantizationInfo {
                        mode,
                        method: QuantizationMethod::ScaleSymmetric,
                        group_size: 64,
                    },
                    signed_codes: false,
                }),
            };
            assert!(matrix.try_prepare_a8_storage(&context).unwrap());
            assert_eq!(raw.as_ref().as_slice::<u8>(), &[0, 127, 128, 255]);
            if mode == QuantizationMode::U8 {
                assert!(!Arc::ptr_eq(&raw, &matrix.values));
                assert_eq!(matrix.values.as_ref().as_slice::<u8>(), &[128, 255, 0, 127]);
                assert_eq!(matrix.representation_copy_bytes.load(Ordering::Relaxed), 4);
            } else {
                assert!(Arc::ptr_eq(&raw, &matrix.values));
                assert_eq!(matrix.representation_copy_bytes.load(Ordering::Relaxed), 0);
            }
        }
    }

    #[stark_test]
    fn coarse_head_uses_midpoint_proposal_weights_without_mutating_target() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        for (mode, bits) in [(QuantizationMode::U5, 5), (QuantizationMode::U6, 6), (QuantizationMode::U8, 8)] {
            let codes = (0..512).map(|i| ((i * 73 + 19) & ((1 << bits) - 1)) as u8).collect::<Vec<_>>();
            let mut packed = vec![0u8; codes.len() * bits / 8];
            for (index, &code) in codes.iter().enumerate() {
                let bit = index * bits;
                packed[bit / 8] |= code << (bit % 8);
                if bit % 8 + bits > 8 {
                    packed[bit / 8 + 1] |= code >> (8 - bit % 8);
                }
            }
            let matrix = WeightMatrix::<Cpu> {
                values: Arc::new(create_buffer_with_data::<Cpu, u8>(&context, &packed)),
                representation_copy_bytes: Arc::new(AtomicUsize::new(0)),
                quantized: Some(Quantized {
                    scales: Arc::new(create_buffer_with_data::<Cpu, bf16>(&context, &[bf16::from_f32(0.125); 8])),
                    correction: QuantizedCorrection::Biases(Arc::new(create_buffer_with_data::<Cpu, bf16>(
                        &context,
                        &[bf16::from_f32(-0.25); 8],
                    ))),
                    params: QuantParams::new(QuantParamsLayout::GroupOutput, 4, 2),
                    info: QuantizationInfo {
                        mode,
                        method: QuantizationMethod::ScaleBias,
                        group_size: 64,
                    },
                    signed_codes: false,
                }),
            };
            for coarse_bits in [3, 4] {
                let coarse = matrix.coarse_affine(&context, 4, 128, coarse_bits).unwrap().unwrap();
                let actual = coarse.values.as_ref().as_slice::<u8>();
                for (index, &code) in codes.iter().enumerate() {
                    let bit = index * coarse_bits as usize;
                    let mut word = u16::from(actual[bit / 8]) >> (bit % 8);
                    if bit % 8 + coarse_bits as usize > 8 {
                        word |= u16::from(actual[bit / 8 + 1]) << (8 - bit % 8);
                    }
                    assert_eq!((word & ((1 << coarse_bits) - 1)) as u8, code >> (bits - coarse_bits as usize));
                }
                let factor = (1 << (bits - coarse_bits as usize)) as f32;
                assert!(
                    coarse
                        .quantized
                        .as_ref()
                        .unwrap()
                        .scales
                        .as_ref()
                        .as_slice::<bf16>()
                        .iter()
                        .all(|&s| f32::from(s) == 0.125 * factor)
                );
                assert!(
                    coarse
                        .biases()
                        .unwrap()
                        .as_slice::<bf16>()
                        .iter()
                        .all(|&b| f32::from(b) == -0.25 + 0.125 * (factor - 1.0) * 0.5)
                );
                assert_eq!(matrix.values.as_ref().as_slice::<u8>(), packed);
            }
        }
    }
}
