use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        gpu_types::ActivationType,
        kernel::{
            ActivationQuantization, GatedActMul, GatedActMulSettings, ReferenceGatedSiLUBF16Kernel,
            matmul::ActivationFormat,
        },
    },
    config::{activation::AnyActivation, clipping::ClippingBounds},
    data_type::DataType,
    encodable_block::linear::{LinearInput, LinearInputPreparation},
};

pub struct MlpGateActMulEncodable<B: Backend> {
    fp_kernel: GatedActMul<B>,
    activation: AnyActivation,
    hidden_dim: u32,
    data_type: DataType,
    hadamard_factors: Option<B::GlobalBuffer>,
    activation_quantization: Option<ActivationQuantization>,
    quantized_kernel: Option<GatedActMul<B>>,
    reference: Option<<B::Kernels as Kernels>::ReferenceGatedSiLUBF16Kernel>,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: AnyActivation,
        gate_clipping: ClippingBounds,
        value_clipping: ClippingBounds,
        hidden_dim: u32,
        input_preparation: Option<LinearInputPreparation<B>>,
        reference_bf16: bool,
    ) -> Result<Self, B::Error> {
        let (hadamard_factors, activation_quantization) = input_preparation
            .map_or((None, None), |preparation| (Some(preparation.rht_signs), preparation.activation_quantization));
        let reference = if reference_bf16 {
            if data_type != DataType::BF16
                || activation.act_type() != ActivationType::SILU
                || activation.alpha() != 1.0
                || hidden_dim == 0
                || gate_clipping.into_pair().is_some()
                || value_clipping.into_pair().is_some()
                || hadamard_factors.is_some()
                || activation_quantization.is_some()
            {
                return Err(InvalidInput("unsupported reference BF16 MLP gate configuration").into());
            }
            Some(<B::Kernels as Kernels>::ReferenceGatedSiLUBF16Kernel::new(context)?)
        } else {
            None
        };
        let settings = GatedActMulSettings {
            activation_alpha: activation.custom_alpha(),
            gate_clipping,
            value_clipping,
        };
        let fp_kernel =
            GatedActMul::<B>::full_precision(context, data_type, true, hadamard_factors.is_some(), settings)?;
        let quantized_kernel = activation_quantization
            .map(|quantization| GatedActMul::<B>::quantized(context, data_type, quantization, settings))
            .transpose()?;
        Ok(Self {
            fp_kernel,
            activation,
            hidden_dim,
            data_type,
            hadamard_factors,
            activation_quantization,
            quantized_kernel,
            reference,
        })
    }

    pub fn encode_for_linear(
        &self,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        fused_up: impl BufferRef<Backend = B>,
        batch_dim: u32,
        act_format: ActivationFormat,
    ) -> Result<LinearInput<B>, B::Error> {
        command_buffer.push_debug_group("gate act mul");
        if let Some(reference) = &self.reference {
            if act_format != ActivationFormat::Bf16
                || fused_up.size() != batch_dim as usize * self.hidden_dim as usize * 2 * size_of::<half::bf16>()
            {
                return Err(InvalidInput("invalid reference BF16 MLP gate input").into());
            }
            let mut hidden =
                command_buffer.allocate_scratch_for_shape(&[batch_dim, self.hidden_dim], DataType::BF16)?;
            reference.encode(fused_up, &mut hidden, self.hidden_dim, batch_dim, command_buffer);
            command_buffer.pop_debug_group();
            return Ok(LinearInput::FullPrecision(hidden));
        }

        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        let input = if act_format == ActivationFormat::Int8
            && let Some(quantization) = self.activation_quantization
        {
            let kernel = self.quantized_kernel.as_ref().expect("INT8 input requires a quantized gate kernel");
            let mut values = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.hidden_dim], DataType::I8)?;
            let mut scales = command_buffer.allocate_scratch_for_shape(
                &[batch_dim, self.hidden_dim.div_ceil(quantization.scale_group_size())],
                DataType::F32,
            )?;
            let mut group_sums = quantization
                .sum_group_size()
                .map(|group_size| {
                    command_buffer
                        .allocate_scratch_for_shape(&[batch_dim, self.hidden_dim.div_ceil(group_size)], DataType::I32)
                })
                .transpose()?;
            kernel.encode_quantized(
                fused_up,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                self.hadamard_factors.as_ref().expect("INT8 input requires RHT factors"),
                self.hidden_dim,
                batch_dim,
                self.activation.act_type(),
                command_buffer,
            );
            LinearInput::Int8Symmetric {
                values,
                scales,
                group_sums,
                scale_group_size: quantization.scale_group_size(),
                code_layout: quantization.code_layout(),
            }
        } else {
            let mut hidden =
                command_buffer.allocate_scratch_for_shape(&[batch_dim, self.hidden_dim], self.data_type)?;
            self.fp_kernel.encode_fp(
                fused_up,
                None::<&B::ScratchBuffer>,
                &mut hidden,
                self.hadamard_factors.as_ref(),
                self.hidden_dim,
                batch_dim,
                0,
                0,
                self.activation.act_type(),
                command_buffer,
            );
            LinearInput::FullPrecision(hidden)
        };

        command_buffer.pop_debug_group();

        Ok(input)
    }
}
