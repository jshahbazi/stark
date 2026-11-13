use crate::{
    backends::common::{
        Backend, BufferMut, CommandBuffer, CommandBufferEncoding, InvalidInput,
        kernel::{CopyRowsKernel, Kernels},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout, mlx_row_partitioned_spec::MLXRowPartitionedSpec, mlx_spec::MLXSpec,
    },
    data_type::DataType,
    encodable_block::linear::{Gather, Linear, LinearBlockError, LinearInput, LinearMatmul},
    parameters::ParameterTree,
};

pub(super) struct RowPartitionedLinear<B: Backend> {
    parts: Vec<(u32, LinearMatmul<B>)>,
    output_dim: u32,
    output_data_type: DataType,
    input_dim: u32,
    reference_affine: bool,
    copy: <B::Kernels as Kernels>::CopyRowsKernel,
}

impl<B: Backend> RowPartitionedLinear<B> {
    pub fn load(
        context: &B::Context,
        spec: MLXRowPartitionedSpec,
        input_dim: u32,
        output_dim: u32,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        tree: &ParameterTree<B>,
    ) -> Result<Self, LinearBlockError<B>> {
        let rows = spec.segments.iter().try_fold(0u32, |sum, part| sum.checked_add(part.rows));
        if spec.layout != Layout::OutputInput
            || has_biases
            || [weights_data_type, input_data_type, output_data_type].iter().any(|&dtype| dtype != DataType::BF16)
            || spec.segments.len() < 2
            || spec.segments.len() > 128
            || spec.segments.iter().any(|part| part.rows == 0)
            || rows != Some(output_dim)
        {
            return Err(LinearBlockError::UnsupportedConfiguration("invalid affine row partition".into()));
        }
        let mut parts = Vec::with_capacity(spec.segments.len());
        for (index, part) in spec.segments.into_vec().into_iter().enumerate() {
            let child = tree.subtree(&format!("segment_{index}"));
            let mut matrix = LinearMatmul::load(
                context,
                AnyWeightMatrixSpec::MLXSpec(MLXSpec::new(part.bits, part.group_size, Layout::OutputInput)),
                input_dim,
                part.rows,
                weights_data_type,
                input_data_type,
                output_data_type,
                &child,
                None,
                None,
            )?;
            matrix.preserve_w8_row_arithmetic(output_dim)?;
            parts.push((part.rows, matrix));
        }
        let copy = <B::Kernels as Kernels>::CopyRowsKernel::new(context)
            .map_err(|error| LinearBlockError::LinearMatmulError(super::LinearMatmulError::BackendError(error)))?;
        Ok(Self {
            parts,
            output_dim,
            output_data_type,
            input_dim,
            reference_affine: tree.numerical_profile() == crate::parameters::NumericalProfile::Qwen35AffineBF16,
            copy,
        })
    }
}

impl<B: Backend> Linear<B> for RowPartitionedLinear<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode_input(LinearInput::FullPrecision(input), batch_dim, command_buffer)
    }

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        if self.reference_affine {
            use super::reference_affine::{MAX_REFERENCE_DISPATCHES, ReferenceAffinePlan};
            ReferenceAffinePlan::new(batch_dim, self.output_dim, self.input_dim)?;
            let mut dispatches = 0_u32;
            for (rows, _) in &self.parts {
                dispatches = dispatches
                    .checked_add(ReferenceAffinePlan::new(batch_dim, *rows, self.input_dim)?.dispatches)
                    .and_then(|count| count.checked_add(1)) // Existing row-copy dispatch.
                    .ok_or(InvalidInput("affine row-partition dispatch count overflow"))?;
            }
            if dispatches > MAX_REFERENCE_DISPATCHES {
                return Err(InvalidInput("affine row-partition dispatch count exceeds bound").into());
            }
        }
        let element_bytes = self.output_data_type.size_in_bits() as u32 / 8;
        let output_row_bytes = self
            .output_dim
            .checked_mul(element_bytes)
            .ok_or(InvalidInput("row-partitioned output row exceeds kernel indexing"))?;
        output_row_bytes
            .checked_mul(batch_dim)
            .ok_or(InvalidInput("row-partitioned output exceeds kernel indexing"))?;
        let mut output =
            command_buffer.allocate_scratch_for_shape(&[batch_dim, self.output_dim], self.output_data_type)?;
        let mut offset = 0;
        for (rows, matrix) in &self.parts {
            let part = matrix.encode_with_a(
                input.as_matmul_a(),
                batch_dim,
                None::<Gather<&B::ScratchBuffer>>,
                command_buffer,
            )?;
            self.copy.encode(
                &part,
                (&mut output).reborrow(),
                rows * element_bytes,
                output_row_bytes,
                offset * element_bytes,
                batch_dim * rows * element_bytes,
                command_buffer,
            );
            offset += rows;
        }
        Ok(output)
    }
}
