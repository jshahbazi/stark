use derive_more::Debug;
use parking_lot::Mutex;
use thiserror::Error;

mod coarse;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput,
        kernel::{
            ActivationQuantization, Kernels,
            matmul::{ActivationFormat, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulShape},
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::{
        linear::{
            Gather, Linear, LinearInput, ReferenceAffineBF16Matmul, ReferenceBF16Matmul, has_source_affine_policy,
            is_affine_dot_witness_active,
        },
        weight_matrix::{WeightMatrix, WeightMatrixError},
    },
    parameters::{NumericalProfile, ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported linear matmul configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct LinearMatmul<B: Backend> {
    kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    matrix: WeightMatrix<B>,
    biases: Option<B::GlobalBuffer>,
    output_hadamard_factors: Option<B::GlobalBuffer>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
    reference_matmul: Option<ReferenceBF16Matmul<B>>,
    reference_affine: Option<ReferenceAffineBF16Matmul<B>>,
}

fn load_biases<B: Backend>(
    weights_data_type: DataType,
    output_data_type: DataType,
    output_dim: u32,
    parameter_tree: Option<&ParameterTree<B>>,
) -> Result<Option<B::GlobalBuffer>, LinearMatmulError<B>> {
    if parameter_tree.is_some() && weights_data_type != output_data_type {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "mixed precision linear with biases is not supported: weights={weights_data_type:?}, output={output_data_type:?}",
        )));
    }
    Ok(parameter_tree
        .map(|tree| tree.leaf("biases")?.validate(&[output_dim], weights_data_type)?.read_buffer())
        .transpose()?)
}

impl<B: Backend> LinearMatmul<B> {
    #[cfg(test)]
    pub(super) fn uses_reference_affine(&self) -> bool {
        self.reference_affine.is_some()
    }
    /// A row partition inherits its original fused matrix's eight-bit lane
    /// arithmetic and prefill K partitions, while retaining packed storage.
    pub(super) fn preserve_w8_row_arithmetic(
        &mut self,
        reference_output_dim: u32,
    ) -> Result<(), LinearMatmulError<B>> {
        let valid = self.matrix.quantization().is_some_and(|info| {
            info.method == crate::backends::common::gpu_types::QuantizationMethod::ScaleBias
                && info.group_size == 64
                && matches!(info.mode.bits(), 4 | 5 | 6 | 8)
        });
        if !valid || self.output_dim < 4 || reference_output_dim < self.output_dim {
            return Err(LinearMatmulError::UnsupportedConfiguration(
                "row arithmetic preservation requires group64 affine rows and their original output dimension".into(),
            ));
        }
        self.kernel.lock().set_row_arithmetic_reference(reference_output_dim);
        Ok(())
    }

    pub(super) fn set_source_output_widths(
        &mut self,
        widths: &[u32],
    ) -> Result<(), LinearMatmulError<B>> {
        self.matrix.set_source_output_widths(widths)?;
        Ok(())
    }

    /// Loads a linear over any parsed spec — full precision, MLX or Int. Hybrid
    /// compositions live in the wrappers, not here.
    pub fn load(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
        output_hadamard_factors: Option<B::GlobalBuffer>,
    ) -> Result<Self, LinearMatmulError<B>> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(LinearMatmulError::UnsupportedDataType(data_type));
            }
        }

        let matrix =
            WeightMatrix::load(weights_tree, spec, Layout::OutputInput, output_dim, input_dim, weights_data_type)?;
        if output_hadamard_factors.is_some() && matrix.quantization().is_none() {
            return Err(LinearMatmulError::UnsupportedConfiguration(
                "fused output-hadamard factors require quantized weights".into(),
            ));
        }

        let biases = load_biases(weights_data_type, output_data_type, output_dim, bias_tree)?;

        let reference_matmul = if weights_tree.numerical_profile() == NumericalProfile::Qwen35ReferenceBF16 {
            if weights_data_type != DataType::BF16
                || input_data_type != DataType::BF16
                || !matches!(output_data_type, DataType::BF16 | DataType::F32)
                || matrix.quantization().is_some()
                || biases.is_some()
            {
                return Err(LinearMatmulError::UnsupportedConfiguration(
                    "reference BF16 linear requires full BF16 weights/activations without biases".into(),
                ));
            }
            Some(ReferenceBF16Matmul::new(context).map_err(LinearMatmulError::BackendError)?)
        } else {
            None
        };
        let use_reference_affine = weights_tree.numerical_profile() == NumericalProfile::Qwen35AffineBF16
            && !has_source_affine_policy::<B>(&matrix.matmul_b());
        let reference_affine = if use_reference_affine {
            if weights_data_type != DataType::BF16
                || input_data_type != DataType::BF16
                || !matches!(output_data_type, DataType::BF16 | DataType::F32)
                || biases.is_some()
                || output_hadamard_factors.is_some()
            {
                return Err(LinearMatmulError::UnsupportedConfiguration(
                    "affine reference requires BF16 inputs/weights and no output transforms".into(),
                ));
            }
            if matrix.quantization().is_some_and(|info| {
                info.method != crate::backends::common::gpu_types::QuantizationMethod::ScaleBias
                    || info.group_size != 64
                    || !matches!(info.mode.bits(), 4 | 5 | 6 | 8)
            }) {
                return Err(LinearMatmulError::UnsupportedConfiguration(
                    "affine reference requires unsigned group64 affine storage or full BF16 controls".into(),
                ));
            }
            Some(ReferenceAffineBF16Matmul::new(context, output_data_type).map_err(LinearMatmulError::BackendError)?)
        } else {
            None
        };
        let mut kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context, weights_data_type, input_data_type, output_data_type)
                .map_err(LinearMatmulError::BackendError)?;
        if use_reference_affine {
            kernel.set_reference_accumulation();
        }

        Ok(Self {
            kernel: Mutex::new(kernel),
            matrix,
            biases,
            output_hadamard_factors,
            input_dim,
            output_dim,
            output_data_type,
            reference_matmul,
            reference_affine,
        })
    }

    pub(super) fn prepare_a8(
        &mut self,
        context: &B::Context,
    ) -> Option<ActivationQuantization> {
        let mut candidate = self.matmul_shape(1, false);
        candidate.signed_codes = self.matrix.a8_signed_codes()?;
        let quantization = self.kernel.lock().select_activation_quantization(&candidate, context)?;
        self.matrix.try_prepare_a8_storage().then_some(quantization)
    }

    pub(crate) fn encode_with_a(
        &self,
        a: MatmulA<impl BufferRef<Backend = B>>,
        batch_dim: u32,
        gather: Option<Gather<impl BufferRef<Backend = B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let (output_dim, gather_indices) =
            gather.map_or((self.output_dim, None), |gather| (gather.output_dim, Some(gather.indices)));
        if let Some(reference) = &self.reference_matmul {
            let MatmulA::FullPrecision {
                values,
                offset,
            } = a
            else {
                return Err(InvalidInput("reference BF16 linear requires full precision activations").into());
            };
            let MatmulB::FullPrecision {
                b,
            } = self.matmul_b()
            else {
                return Err(InvalidInput("reference BF16 linear requires full precision weights").into());
            };
            return if let Some(indices) = gather_indices {
                reference.encode_gathered(
                    values,
                    offset,
                    b,
                    indices,
                    batch_dim,
                    output_dim,
                    self.output_dim,
                    self.input_dim,
                    self.output_data_type,
                    command_buffer,
                )
            } else if self.output_data_type == DataType::BF16 {
                reference.encode(values, offset, b, batch_dim, output_dim, self.input_dim, command_buffer)
            } else {
                reference.encode_output(
                    values,
                    offset,
                    b,
                    batch_dim,
                    output_dim,
                    self.input_dim,
                    self.output_data_type,
                    command_buffer,
                )
            };
        }
        if let Some(reference) = &self.reference_affine
            && !is_affine_dot_witness_active::<B>()
            && !has_source_affine_policy::<B>(&self.matmul_b())
        {
            let MatmulA::FullPrecision {
                values,
                offset,
            } = a
            else {
                return Err(InvalidInput("affine reference requires BF16 activations").into());
            };
            return reference.encode(
                values,
                offset,
                self.matmul_b(),
                gather_indices,
                batch_dim,
                output_dim,
                self.input_dim,
                command_buffer,
            );
        }
        let mut output = command_buffer.allocate_scratch_for_shape(&[batch_dim, output_dim], self.output_data_type)?;
        self.kernel.lock().encode(
            MatmulArguments {
                a,
                b: self.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: self.d_ops(),
                gather_indices,
                m: batch_dim,
                n: output_dim,
                k: self.input_dim,
            },
            command_buffer,
        )?;

        Ok(output)
    }

    fn matmul_shape(
        &self,
        batch_dim: u32,
        a_full_precision: bool,
    ) -> MatmulShape {
        let b = self.matmul_b();
        MatmulShape {
            m: batch_dim,
            n: self.output_dim,
            k: self.input_dim,
            b_transpose: true,
            b_leading_dimension: None,
            b_prologue: b.b_prologue(),
            b_bits: b.bits_per_b(),
            b_group_size: b.group_size(),
            signed_codes: b.signed_codes(),
            a_full_precision,
            gathered: false,
            params_layout: b.quant_params_layout(),
            d_transform: self.d_ops().mask(),
        }
    }

    fn matmul_b(&self) -> MatmulB<&B::GlobalBuffer> {
        self.matrix.matmul_b()
    }

    fn d_ops(&self) -> MatmulDOps<'_, B> {
        MatmulDOps {
            bias: self.biases.as_ref(),
            rht_factors: self.output_hadamard_factors.as_ref(),
            ..MatmulDOps::none()
        }
    }
}

impl<B: Backend> Linear<B> for LinearMatmul<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("matmul");

        let output = self.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            batch_dim,
            None::<Gather<&B::ScratchBuffer>>,
            command_buffer,
        )?;

        command_buffer.pop_debug_group();

        Ok(output)
    }

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode_with_a(input.as_matmul_a(), batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer)
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        context: &B::Context,
    ) -> ActivationFormat {
        let bf16_shape = self.matmul_shape(batch_dim, true);
        self.kernel.lock().select_activation_format(&bf16_shape, context)
    }
}
