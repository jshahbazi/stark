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
    #[cfg(test)]
    pub(crate) fn validation_activation(
        &self,
        fused_up: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        use crate::backends::common::{BufferMut, kernel::ActivationKernel};
        if rows == 0
            || rows > 16
            || self.data_type != DataType::BF16
            || self.reference.is_some()
            || self.activation.custom_alpha().is_some()
            || fused_up.size() != rows as usize * self.hidden_dim as usize * 4
        {
            return Err(InvalidInput("small unit-slope BF16 activation trace required").into());
        }
        let row_bytes = self.hidden_dim as usize * 2;
        let mut activated = commands.allocate_scratch(rows as usize * row_bytes)?;
        for row in 0..rows as usize {
            commands.encode_copy(
                fused_up.subrange((row * 2 + 1) * row_bytes..(row * 2 + 2) * row_bytes),
                activated.subrange_mut(row * row_bytes..(row + 1) * row_bytes),
            );
        }
        let kernel = <B::Kernels as Kernels>::ActivationKernel::new(commands.context(), self.data_type, true)?;
        kernel.encode(
            None::<&B::ScratchBuffer>,
            &mut activated,
            rows * self.hidden_dim,
            self.activation.act_type(),
            commands,
        );
        Ok(activated)
    }

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
        if activation.act_type() == ActivationType::SILUMaterializedSigmoid
            && (data_type != DataType::BF16 || activation.alpha() != 1.0 || reference_bf16)
        {
            return Err(InvalidInput("materialized sigmoid requires BF16 unit-slope SiLU").into());
        }
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
            // The following matrix may prefer A8 on MXU devices. This policy
            // owns an explicit BF16 activation boundary, so emit full BF16
            // input regardless of that advisory preference. Linear dispatch
            // then selects its ordinary BF16-input packed-weight path.
            if fused_up.size() != batch_dim as usize * self.hidden_dim as usize * 2 * size_of::<half::bf16>() {
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

#[cfg(all(test, backend = "metal"))]
mod reference_tests {
    use half::bf16;
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{
        backends::{
            common::{
                CommandBufferExecutable, CommandBufferPending, Context,
                gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
                kernel::matmul::{MatmulKernel, MatmulShape, QuantParamsLayout},
            },
            metal::Metal,
        },
        tests::helpers::{buffer_to_vec, create_buffer_with_data},
    };

    #[stark_test]
    fn reference_bf16_gate_preserves_multirow_boundary_when_mxu_prefers_a8() {
        let context = <Metal as Backend>::Context::new().unwrap();
        if !context.supports_mxu {
            return;
        }
        // Real dense27 Q4 down-projection shape at the automatic text check.
        // No model weights are allocated; the gate uses less than 2 MiB.
        let shape = MatmulShape {
            m: 14,
            n: 5120,
            k: 17408,
            b_transpose: true,
            b_leading_dimension: None,
            b_prologue: GemmBPrologueKind::ScaleBiasDequant,
            b_bits: Some(4),
            b_group_size: Some(64),
            signed_codes: false,
            a_full_precision: true,
            gathered: false,
            params_layout: Some(QuantParamsLayout::GroupOutput),
            d_transform: GemmDTransform::empty(),
        };
        type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;
        let matmul = MetalMatmul::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();
        let requested = matmul.select_activation_format(&shape, &context);
        assert_eq!(requested, ActivationFormat::Int8, "regression fixture must exercise the MXU A8 preference");
        let gate = MlpGateActMulEncodable::<Metal>::new(
            &context,
            DataType::BF16,
            serde_json::from_value(serde_json::json!({"type":"SiLU","alpha":1.0})).unwrap(),
            ClippingBounds::default(),
            ClippingBounds::default(),
            shape.k,
            None,
            true,
        )
        .unwrap();
        let input =
            create_buffer_with_data::<Metal, bf16>(&context, &vec![bf16::ONE; (shape.m * shape.k * 2) as usize]);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let output = gate.encode_for_linear(&mut commands, &input, shape.m, requested).unwrap();
        let LinearInput::FullPrecision(output) = output else {
            panic!("reference BF16 gate emitted quantized activations");
        };
        let mut readback = context.create_buffer((shape.m * shape.k) as usize * size_of::<bf16>()).unwrap();
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let values = buffer_to_vec::<Metal, bf16>(&readback);
        let expected = bf16::from_f32(1.0 / (1.0 + (-1.0f32).exp()));
        assert!(values.iter().all(|value| value.to_bits() == expected.to_bits()), "reference gate result differs");
    }
}
