use half::{bf16, f16};

use crate::{
    backends::{
        common::{
            BufferRef,
            kernel::matmul::{MatmulB, MatmulError, QuantParamsStrides},
        },
        cpu::{Cpu, buffer::CpuBufferExt},
    },
    data_type::DataType,
    utils::pointers::SendPtr,
};

pub(super) enum WeightData {
    FullPrecision {
        ptr: SendPtr<u8>,
        leading_dimension: usize,
        transpose: bool,
    },
    Quantized {
        weights: SendPtr<u8>,
        weights_size: usize,
        scales: SendPtr<u8>,
        zero_points: Option<(SendPtr<u8>, QuantParamsStrides)>,
        biases: Option<SendPtr<u8>>,
        scale_strides: QuantParamsStrides,
        bits: usize,
        group_size: usize,
        signed_codes: bool,
    },
}

impl WeightData {
    pub(super) fn from_b(
        b: MatmulB<impl BufferRef<Backend = Cpu>>,
        b_leading_dimension: Option<u32>,
        b_transpose: bool,
        k: usize,
        n: usize,
    ) -> Result<Self, MatmulError<Cpu>> {
        fn buffer_ptr(view: impl BufferRef<Backend = Cpu>) -> SendPtr<u8> {
            let (buffer, range) = view.parts();
            SendPtr(buffer.cpu_address().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(range.start))
        }
        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let leading_dimension = b_leading_dimension.map(|ld| ld as usize).unwrap_or(if b_transpose {
                    k
                } else {
                    n
                });
                Ok(WeightData::FullPrecision {
                    ptr: buffer_ptr(weights),
                    leading_dimension,
                    transpose: b_transpose,
                })
            },
            MatmulB::Quantized(quantized) => {
                let weights_size = quantized.codes.size();
                let required_bytes = n
                    .checked_mul(k)
                    .and_then(|count| count.checked_mul(quantized.bits() as usize))
                    .map(|bits| bits.div_ceil(8));
                if required_bytes.is_none_or(|size| size > weights_size) {
                    return Err(MatmulError::UnsupportedLayout {
                        path: "CpuMatmul codes buffer",
                    });
                }
                Ok(WeightData::Quantized {
                    weights: buffer_ptr(quantized.codes),
                    weights_size,
                    scales: buffer_ptr(quantized.scales),
                    zero_points: quantized
                        .zero_points()
                        .map(|values| (buffer_ptr(values), quantized.zero_point_strides())),
                    biases: quantized.biases().map(buffer_ptr),
                    scale_strides: quantized.params.scale_strides(),
                    bits: quantized.bits() as usize,
                    group_size: quantized.group_size as usize,
                    signed_codes: quantized.signed_codes,
                })
            },
        }
    }
}

#[inline]
pub(super) fn packed_code(
    codes: &[u8],
    bits: usize,
    index: usize,
) -> u8 {
    let codes_per_byte = 8 / bits;
    let byte = codes[index / codes_per_byte];
    (byte >> ((index % codes_per_byte) * bits)) & ((1u16 << bits) - 1) as u8
}

/// Source-compatible single-row W4 correction. Dot products and accumulation
/// stay F32; each four-value input sum materializes BF16 additions, and each
/// affine contribution is evaluated before adding it to the F32 prior.
pub(super) unsafe fn materialized_affine_dot(
    weights: &WeightData,
    input: *const u8,
    row: usize,
    column: usize,
    k: usize,
) -> f32 {
    let WeightData::Quantized {
        weights,
        weights_size,
        scales,
        biases: Some(biases),
        scale_strides,
        group_size,
        ..
    } = weights
    else {
        unreachable!();
    };
    let codes = unsafe { std::slice::from_raw_parts(weights.as_ptr(), *weights_size) };
    let mut result = 0.0f32;
    for begin in (0..k).step_by(16) {
        let end = (begin + 16).min(k);
        let mut dot = 0.0f32;
        let mut sum = 0.0f32;
        for quad in (begin..end).step_by(4) {
            let mut values = [bf16::ZERO; 4];
            let mut subtotal = 0.0f32;
            for (index, value) in values.iter_mut().enumerate() {
                if quad + index < end {
                    *value = unsafe { *(input as *const bf16).add(row * k + quad + index) };
                    let code = packed_code(codes, 4, column * k + quad + index);
                    subtotal += value.to_f32() * f32::from(code);
                }
            }
            dot += subtotal;
            sum += (((values[0] + values[1]) + values[2]) + values[3]).to_f32();
        }
        let metadata =
            column * scale_strides.output_stride as usize + (begin / group_size) * scale_strides.group_stride as usize;
        let scale = unsafe { read_f32(scales.as_ptr(), DataType::BF16, metadata) };
        let bias = unsafe { read_f32(biases.as_ptr(), DataType::BF16, metadata) };
        result += scale * dot + bias * sum;
    }
    result
}

#[inline]
pub(super) unsafe fn read_f32(
    base: *const u8,
    data_type: DataType,
    index: usize,
) -> f32 {
    unsafe {
        match data_type {
            DataType::F32 => *(base as *const f32).add(index),
            DataType::F16 => (*(base as *const f16).add(index)).to_f32(),
            DataType::BF16 => (*(base as *const bf16).add(index)).to_f32(),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub(super) unsafe fn write_f32(
    base: *mut u8,
    data_type: DataType,
    index: usize,
    value: f32,
) {
    unsafe {
        match data_type {
            DataType::F32 => *(base as *mut f32).add(index) = value,
            DataType::F16 => *(base as *mut f16).add(index) = f16::from_f32(value),
            DataType::BF16 => *(base as *mut bf16).add(index) = bf16::from_f32(value),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod packed_tail_tests {
    use stark_engine_macros::stark_test;

    use super::packed_code;

    #[stark_test]
    fn packed_codes_only_require_their_exact_byte_tail() {
        assert_eq!(packed_code(&[0xd3], 8, 0), 0xd3);
        assert_eq!(packed_code(&[0xd3], 4, 0), 3);
        assert_eq!(packed_code(&[0xd3], 4, 1), 13);
        let codes = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba];
        for index in 0..12 {
            assert_eq!(packed_code(&codes, 4, index), index as u8);
        }
        let aligned = [0x12, 0x34, 0x56, 0x78];
        for (index, expected) in aligned.into_iter().enumerate() {
            assert_eq!(packed_code(&aligned, 8, index), expected);
        }
    }

    #[stark_test]
    fn cpu_matmul_dispatches_exact_packed_tails() {
        use crate::{
            backends::{
                common::{
                    Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable,
                    CommandBufferPending, Context,
                    gpu_types::QuantizationMode,
                    kernel::matmul::{
                        MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, QuantParams, QuantParamsLayout,
                        QuantizedB, QuantizedCorrection,
                    },
                },
                cpu::{Cpu, kernel::matmul::MatmulCpuKernel},
            },
            data_type::DataType,
        };

        let context = <Cpu as Backend>::Context::new().unwrap();
        for (mode, k, n, codes, expected) in [
            (QuantizationMode::U8, 1, 1, vec![129u8], vec![1.0f32]),
            (QuantizationMode::U4, 2, 1, vec![0x21], vec![-13.0]),
            (QuantizationMode::U4, 6, 2, vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba], vec![-33.0, 3.0]),
            (QuantizationMode::U8, 4, 1, vec![129, 130, 131, 132], vec![10.0]),
        ] {
            let mut input = context.create_buffer(k * 4).unwrap();
            input.copyin(&vec![1.0f32; k]);
            let mut weights = context.create_buffer(codes.len()).unwrap();
            weights.copyin(&codes);
            let mut scales = context.create_buffer(n * 4).unwrap();
            scales.copyin(&vec![1.0f32; n]);
            let mut output = context.create_buffer(n * 4).unwrap();
            let mut commands = context.create_command_buffer(None, None).unwrap();
            let mut kernel = MatmulCpuKernel::new(&context, DataType::F32, DataType::F32, DataType::F32).unwrap();
            kernel
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &input,
                            offset: 0,
                        },
                        b: MatmulB::Quantized(QuantizedB {
                            codes: &weights,
                            scales: &scales,
                            correction: QuantizedCorrection::Symmetric,
                            params: QuantParams::new(QuantParamsLayout::OutputGroup, n as u32, 1),
                            mode,
                            group_size: k as u32,
                            signed_codes: false,
                        }),
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut output,
                        d_transform: MatmulDOps::none(),
                        gather_indices: None::<&<Cpu as Backend>::GlobalBuffer>,
                        m: 1,
                        n: n as u32,
                        k: k as u32,
                    },
                    &mut commands,
                )
                .unwrap();
            commands.end_encoding().submit().wait_until_completed().unwrap();
            assert_eq!(output.copyout::<f32>(), expected);
        }
        let codes = context.create_buffer(1).unwrap();
        let scales = context.create_buffer(4).unwrap();
        let undersized = MatmulB::Quantized(QuantizedB {
            codes: &codes,
            scales: &scales,
            correction: QuantizedCorrection::Symmetric,
            params: QuantParams::new(QuantParamsLayout::OutputGroup, 1, 1),
            mode: QuantizationMode::U8,
            group_size: 2,
            signed_codes: false,
        });
        assert!(super::WeightData::from_b(undersized, None, true, 2, 1).is_err());
    }
}
