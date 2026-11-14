//! Internal contiguous BF16 linear for the admitted Qwen3.5 reference profile.

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            MlxHeadMatmulBF16Kernel, ReferenceGatherMatmulBF16Kernel, ReferenceRoundBF16F32Kernel,
            VisionCastBF16Kernel, VisionMatmulBF16Kernel,
        },
    },
    data_type::DataType,
};

pub(crate) struct ReferenceBF16Matmul<B: Backend> {
    dot: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    cast: <B::Kernels as Kernels>::VisionCastBF16Kernel,
    round_f32: <B::Kernels as Kernels>::ReferenceRoundBF16F32Kernel,
    gather: <B::Kernels as Kernels>::ReferenceGatherMatmulBF16Kernel,
    mlx_head: Option<<B::Kernels as Kernels>::MlxHeadMatmulBF16Kernel>,
}

impl<B: Backend> ReferenceBF16Matmul<B> {
    pub(crate) fn new(context: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            dot: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context)?,
            cast: <B::Kernels as Kernels>::VisionCastBF16Kernel::new(context)?,
            round_f32: <B::Kernels as Kernels>::ReferenceRoundBF16F32Kernel::new(context)?,
            gather: <B::Kernels as Kernels>::ReferenceGatherMatmulBF16Kernel::new(context)?,
            mlx_head: None,
        })
    }

    pub(crate) fn with_mlx_wide(
        mut self,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        self.mlx_head = Some(<B::Kernels as Kernels>::MlxHeadMatmulBF16Kernel::new(context)?);
        Ok(self)
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
        self.encode_output(input, input_offset, weights, m, n, k, DataType::BF16, commands)
    }

    /// Router logits use F32 storage after the same BF16 output boundary.
    pub(crate) fn encode_output(
        &self,
        input: impl BufferRef<Backend = B>,
        input_offset: usize,
        weights: impl BufferRef<Backend = B>,
        m: u32,
        n: u32,
        k: u32,
        output_type: DataType,
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
        let source_wide = self.mlx_head.as_ref().filter(|_| (2..=5).contains(&m) && n > 1);
        if source_wide.is_some() {
            // BufferRef can already describe a subview. Check the effective
            // byte offsets before reinterpreting either operand as bfloat4.
            let input_start = input.parts().1.start.checked_add(input_offset).ok_or_else(invalid)?;
            if !input_start.is_multiple_of(8) || !weights.parts().1.start.is_multiple_of(8) {
                return Err(invalid().into());
            }
            elements.checked_mul(32).ok_or_else(invalid)?;
        }
        let mut accumulated = commands.allocate_scratch_for_shape(&[m, n], DataType::F32)?;
        if !matches!(output_type, DataType::BF16 | DataType::F32) {
            return Err(invalid().into());
        }
        let mut output = commands.allocate_scratch_for_shape(&[m, n], output_type)?;
        if let Some(source) = source_wide {
            source.encode(input.subrange(input_offset..input_end), weights, &mut accumulated, m, n, k, commands);
        } else {
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
        }
        if output_type == DataType::BF16 {
            self.cast.encode(&accumulated, &mut output, elements, commands);
        } else {
            self.round_f32.encode(&accumulated, &mut output, elements, commands);
        }
        Ok(output)
    }

    /// Indexed reference dots read bank rows directly. Scratch is only the
    /// F32 accumulator plus output, independent of the bank size or K width.
    pub(crate) fn encode_gathered(
        &self,
        input: impl BufferRef<Backend = B>,
        input_offset: usize,
        weights: impl BufferRef<Backend = B>,
        indices: impl BufferRef<Backend = B>,
        m: u32,
        n: u32,
        bank_rows: u32,
        k: u32,
        output_type: DataType,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let invalid = || InvalidInput("unsupported reference BF16 gathered linear geometry");
        let elements = m.checked_mul(n).ok_or_else(invalid)?;
        let bank_elements = bank_rows.checked_mul(k).ok_or_else(invalid)?;
        let input_bytes = (m as usize).checked_mul(k as usize).and_then(|n| n.checked_mul(2)).ok_or_else(invalid)?;
        let input_end = input_offset.checked_add(input_bytes).ok_or_else(invalid)?;
        if m == 0
            || n == 0
            || k == 0
            || !k.is_multiple_of(32)
            || !input_offset.is_multiple_of(2)
            || indices.size() != elements as usize * 4
            || input_end > input.size()
            || weights.size() != bank_elements as usize * 2
            || !matches!(output_type, DataType::BF16 | DataType::F32)
        {
            return Err(invalid().into());
        }
        let mut accumulated = commands.allocate_scratch_for_shape(&[m, n], DataType::F32)?;
        let mut output = commands.allocate_scratch_for_shape(&[m, n], output_type)?;
        self.gather.encode(
            input.subrange(input_offset..input_end),
            weights,
            indices,
            &mut accumulated,
            m,
            n,
            k,
            bank_rows,
            commands,
        );
        if output_type == DataType::BF16 {
            self.cast.encode(&accumulated, &mut output, elements, commands);
        } else {
            self.round_f32.encode(&accumulated, &mut output, elements, commands);
        }
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

    fn gathered_reference<B: Backend>() {
        let context = B::Context::new().unwrap();
        let (m, n, k, bank_rows) = (4u32, 3u32, 64u32, 8u32);
        let input = create_buffer_with_data::<B, bf16>(
            &context,
            &(0..m * k).map(|i| bf16::from_f32((i as i32 % 29 - 14) as f32 / 7.0)).collect::<Vec<_>>(),
        );
        let weights = create_buffer_with_data::<B, bf16>(
            &context,
            &(0..bank_rows * k).map(|i| bf16::from_f32((i as i32 % 37 - 18) as f32 / 13.0)).collect::<Vec<_>>(),
        );
        let ids = [7u32, 0, 3, 1, 6, 2, 5, 3, 0, 2, 7, 1];
        let indices = create_buffer_with_data::<B, u32>(&context, &ids);
        let kernel = ReferenceBF16Matmul::<B>::new(&context).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let full = kernel.encode(&input, 0, &weights, m, bank_rows, k, &mut commands).unwrap();
        let gathered = kernel
            .encode_gathered(&input, 0, &weights, &indices, m, n, bank_rows, k, DataType::BF16, &mut commands)
            .unwrap();
        let widened = kernel
            .encode_gathered(&input, 0, &weights, &indices, m, n, bank_rows, k, DataType::F32, &mut commands)
            .unwrap();
        let invalid_ids = create_buffer_with_data::<B, u32>(&context, &[u32::MAX; 12]);
        let invalid = kernel
            .encode_gathered(&input, 0, &weights, &invalid_ids, m, n, bank_rows, k, DataType::BF16, &mut commands)
            .unwrap();
        assert!(
            kernel
                .encode_gathered(&input, 0, &weights, &indices, m, n + 1, bank_rows, k, DataType::BF16, &mut commands)
                .is_err()
        );
        assert!(
            kernel
                .encode_gathered(&input, 0, &weights, &indices, m, n, bank_rows, k - 1, DataType::BF16, &mut commands)
                .is_err()
        );
        let mut full_result = context.create_buffer((m * bank_rows * 2) as usize).unwrap();
        let mut gathered_result = context.create_buffer((m * n * 2) as usize).unwrap();
        let mut widened_result = context.create_buffer((m * n * 4) as usize).unwrap();
        let mut invalid_result = context.create_buffer((m * n * 2) as usize).unwrap();
        commands.encode_copy(&full, &mut full_result);
        commands.encode_copy(&gathered, &mut gathered_result);
        commands.encode_copy(&widened, &mut widened_result);
        commands.encode_copy(&invalid, &mut invalid_result);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let full = buffer_to_vec::<B, bf16>(&full_result);
        let gathered = buffer_to_vec::<B, bf16>(&gathered_result);
        let widened = buffer_to_vec::<B, f32>(&widened_result);
        for (index, selected) in ids.iter().enumerate() {
            let expected = full[index / n as usize * bank_rows as usize + *selected as usize];
            assert_eq!(gathered[index], expected, "indexed reference dot differs");
            assert_eq!(widened[index], expected.to_f32(), "router boundary differs");
        }
        assert!(buffer_to_vec::<B, bf16>(&invalid_result).iter().all(|value| value.is_nan()));
    }

    #[stark_test]
    fn gathered_reference_matches_contiguous_dots_and_preserves_bf16_router_boundary() {
        for_each_backend!(|B| gathered_reference::<B>());
    }
}
