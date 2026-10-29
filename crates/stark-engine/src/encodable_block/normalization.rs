use derive_more::Debug;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{Kernels, NormalizationKernel, ReferenceRMSBF16Kernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    parameters::{NumericalProfile, ParameterLoaderError, ParameterTree},
};

// TODO: clean up

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PostLayerScalar {
    None,
    ScaleResidualSum(f32),
    ScaleOutput(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortcutMode {
    None,
    Copy,
    Add,
}

#[derive(Debug, Error)]
pub enum NormalizationNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
}

pub struct Normalization<B: Backend> {
    epsilon: f32,
    scale_offset: Option<f32>,
    scales: Option<B::GlobalBuffer>,
    biases: Option<B::GlobalBuffer>,
    element_count: u32,
    hadamard_factors: Option<B::GlobalBuffer>,
    post_layer_scalar_value: f32,
    data_type: DataType,
    kernel: <B::Kernels as Kernels>::NormalizationKernel,
    reference: Option<<B::Kernels as Kernels>::ReferenceRMSBF16Kernel>,
}

impl<B: Backend> Normalization<B> {
    pub fn new(
        element_count: u32,
        hadamard_factors: Option<B::GlobalBuffer>,
        shortcut_mode: ShortcutMode,
        post_layer_scalar: PostLayerScalar,
        data_type: DataType,
        config: &NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, NormalizationNewError<B>> {
        config
            .validate_mlx_bf16_reduction(data_type, element_count, hadamard_factors.is_some())
            .map_err(|error| NormalizationNewError::Backend(error.into()))?;
        let (copy_to_shortcut, residual_add) = match shortcut_mode {
            ShortcutMode::None => (false, false),
            ShortcutMode::Copy => (true, false),
            ShortcutMode::Add => (true, true),
        };
        let reference = if parameter_tree.numerical_profile() == NumericalProfile::Qwen35ReferenceBF16 {
            if data_type != DataType::BF16
                || config.subtract_mean
                || config.has_biases
                || !config.has_scale
                || config.upcast_mode != UpcastMode::FullLayer
                || hadamard_factors.is_some()
                || post_layer_scalar != PostLayerScalar::None
                || element_count < 16
                || !element_count.is_multiple_of(16)
            {
                return Err(NormalizationNewError::Backend(
                    crate::backends::common::InvalidInput("unsupported reference BF16 RMS configuration").into(),
                ));
            }
            Some(
                <B::Kernels as Kernels>::ReferenceRMSBF16Kernel::new(context, false, copy_to_shortcut, residual_add)
                    .map_err(NormalizationNewError::Backend)?,
            )
        } else {
            None
        };

        let scales = config
            .has_scale
            .then(|| parameter_tree.leaf("scales")?.validate(&[element_count], DataType::F32)?.read_buffer())
            .transpose()?;
        let biases = config
            .has_biases
            .then(|| parameter_tree.leaf("biases")?.validate(&[element_count], DataType::F32)?.read_buffer())
            .transpose()?;

        let (scale_residual_sum, scale_output, post_layer_scalar_value) = match post_layer_scalar {
            PostLayerScalar::None => (false, false, 1.0),
            PostLayerScalar::ScaleResidualSum(value) => (true, false, value),
            PostLayerScalar::ScaleOutput(value) => (false, true, value),
        };

        let kernel = <B::Kernels as Kernels>::NormalizationKernel::new(
            context,
            data_type,
            DataType::F32,
            data_type,
            DataType::F32,
            false,
            config.subtract_mean,
            config.upcast_mode == UpcastMode::FullLayer,
            copy_to_shortcut,
            residual_add,
            hadamard_factors.is_some(),
            scale_residual_sum,
            scale_output,
            biases.is_some(),
            scales.is_some(),
            config.mlx_bf16_reduction,
        )
        .map_err(NormalizationNewError::Backend)?;

        Ok(Self {
            epsilon: config.epsilon,
            scale_offset: config.scale_offset,
            scales,
            biases,
            element_count,
            hadamard_factors,
            post_layer_scalar_value,
            data_type,
            kernel,
            reference,
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        row_offset: u32,
        row_count: u32,
        shortcut: Option<impl BufferMut<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("normalization");

        let row_size = size_for_shape(&[self.element_count], self.data_type);
        let row_offset_bytes = row_offset as usize * row_size;
        let shortcut = shortcut.map(|shortcut| shortcut.subrange_mut(row_offset_bytes..));
        let mut output = command_buffer.allocate_scratch_for_shape(&[row_count, self.element_count], self.data_type)?;
        if let Some(reference) = &self.reference {
            reference.encode(
                Some(input.subrange(row_offset_bytes..)),
                self.scales.as_ref().unwrap(),
                &mut output,
                shortcut,
                row_count,
                self.element_count,
                self.element_count,
                self.epsilon,
                self.scale_offset.unwrap_or(0.0),
                0,
                1,
                command_buffer,
            );
            command_buffer.pop_debug_group();
            return Ok(output);
        }
        self.kernel.encode(
            Some(input.subrange(row_offset_bytes..)),
            self.scales.as_ref(),
            self.biases.as_ref(),
            &mut output,
            shortcut,
            self.hadamard_factors.as_ref(),
            row_count,
            self.element_count,
            self.epsilon,
            self.scale_offset.unwrap_or(0.0),
            self.post_layer_scalar_value,
            command_buffer,
        );

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
