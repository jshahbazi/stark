//! Internal contiguous BF16 linear for the admitted Qwen3.5 reference profile.

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{VisionCastBF16Kernel, VisionMatmulBF16Kernel},
    },
    data_type::DataType,
};

pub(crate) struct ReferenceBF16Matmul<B: Backend> {
    dot: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    cast: <B::Kernels as Kernels>::VisionCastBF16Kernel,
}

impl<B: Backend> ReferenceBF16Matmul<B> {
    pub(crate) fn new(context: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            dot: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context)?,
            cast: <B::Kernels as Kernels>::VisionCastBF16Kernel::new(context)?,
        })
    }

    /// A is contiguous [m,k], with a byte offset; B is full row-major [n,k].
    /// The exact source dot schedule is eight four-lane F32 accumulators and a
    /// fixed binary reduction, followed by the public BF16 output boundary.
    pub(crate) fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        input_offset: usize,
        weights: impl BufferRef<Backend = B>,
        m: u32,
        n: u32,
        k: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let invalid = || InvalidInput("unsupported reference BF16 linear geometry");
        if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(32) || !input_offset.is_multiple_of(2) {
            return Err(invalid().into());
        }
        let input_bytes = (m as usize).checked_mul(k as usize).and_then(|x| x.checked_mul(2)).ok_or_else(invalid)?;
        let weight_bytes = (n as usize).checked_mul(k as usize).and_then(|x| x.checked_mul(2)).ok_or_else(invalid)?;
        let input_end = input_offset.checked_add(input_bytes).ok_or_else(invalid)?;
        let elements = m.checked_mul(n).ok_or_else(invalid)?;
        if input_end > input.size() || weights.size() != weight_bytes {
            return Err(invalid().into());
        }
        let mut accumulated = commands.allocate_scratch_for_shape(&[m, n], DataType::F32)?;
        let mut output = commands.allocate_scratch_for_shape(&[m, n], DataType::BF16)?;
        self.dot.encode(
            input.subrange(input_offset..input_end),
            weights,
            &mut accumulated,
            m,
            n,
            k,
            k,
            1,
            32,
            commands,
        );
        self.cast.encode(&accumulated, &mut output, elements, commands);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{
        backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
        tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
    };

    fn offset_and_geometry<B: Backend>() {
        let context = B::Context::new().unwrap();
        let mut input = vec![bf16::from_f32(-99.0); 2];
        input.extend((1..=64).map(|value| bf16::from_f32(value as f32)));
        let input = create_buffer_with_data::<B, bf16>(&context, &input);
        let mut weights = vec![bf16::ZERO; 64];
        weights[0] = bf16::ONE;
        weights[63] = bf16::ONE;
        let weights = create_buffer_with_data::<B, bf16>(&context, &weights);
        let kernel = ReferenceBF16Matmul::<B>::new(&context).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        for (offset, m, n, k) in [(1, 2, 2, 32), (6, 2, 2, 32), (4, 0, 2, 32), (4, 2, 1, 32), (4, 2, 2, 31)] {
            assert!(kernel.encode(&input, offset, &weights, m, n, k, &mut commands).is_err());
        }
        let output = kernel.encode(&input, 4, &weights, 2, 2, 32, &mut commands).unwrap();
        let mut readback = context.create_buffer(4 * 2).unwrap();
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let output = buffer_to_vec::<B, bf16>(&readback);
        assert!(output.iter().copied().map(f32::from).eq([1.0, 32.0, 33.0, 64.0]));
    }

    #[stark_test]
    fn reference_bf16_linear_checks_geometry_and_honors_byte_offset() {
        for_each_backend!(|B| {
            offset_and_geometry::<B>();
        });
    }
}
