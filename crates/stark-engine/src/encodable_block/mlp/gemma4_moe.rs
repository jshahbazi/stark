//! Gemma 4 dense and routed expert branches with source-specific normalization.

use super::{
    Mlp, MlpBlockError,
    routed::{RoutedExperts, RoutingPrecision},
};
use crate::{
    backends::common::{
        Backend, Buffer, CommandBuffer, CommandBufferEncoding, InvalidInput,
        kernel::{GemmaMoeRouterScaleBF16Kernel, Kernels, TensorAddScaleKernel, matmul::MatmulA},
    },
    config::mlp::{AnyMLPConfig, gemma4_moe::Gemma4MoeConfig},
    data_type::DataType,
    encodable_block::{
        linear::{Gather, LinearBlockError, LinearMatmul},
        normalization::{Normalization, PostLayerScalar, ShortcutMode},
    },
    parameters::ParameterTree,
};

pub(super) struct Gemma4Moe<B: Backend> {
    dense: Box<dyn Mlp<B>>,
    dense_norm: Normalization<B>,
    expert_input_norm: Normalization<B>,
    expert_output_norm: Normalization<B>,
    router_norm: Normalization<B>,
    router_scale: B::GlobalBuffer,
    expert_scales: B::GlobalBuffer,
    router: LinearMatmul<B>,
    experts: RoutedExperts<B>,
    scale: <B::Kernels as Kernels>::GemmaMoeRouterScaleBF16Kernel,
    add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    model: u32,
    root_scale: f32,
}

impl<B: Backend> Gemma4Moe<B> {
    pub fn new(
        config: &Gemma4MoeConfig,
        model: u32,
        dense_hidden: u32,
        context: &B::Context,
        tree: &ParameterTree<B>,
        dtype: DataType,
    ) -> Result<Self, MlpBlockError<B>> {
        if dtype != DataType::BF16
            || !config.router_root_scale.is_finite()
            || config.router_root_scale <= 0.0
            || config.norm_config.epsilon != 1e-6
            || config.norm_config.mlx_bf16_reduction
            || !config.norm_config.has_scale
            || config.norm_config.has_biases
            || config.norm_config.subtract_mean
        {
            return Err(MlpBlockError::BackendError(InvalidInput("unsupported canonical Gemma expert profile").into()));
        }
        let normalize = |name: &str, norm: &crate::config::normalization::NormalizationConfig| {
            Normalization::new(
                model,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                dtype,
                norm,
                &tree.subtree(name),
                context,
            )
        };
        let (dense, _) = <dyn Mlp<B>>::new(
            &AnyMLPConfig::DenseMLPConfig(config.expert_config.clone()),
            model,
            dense_hidden,
            context,
            &tree.subtree("dense"),
            dtype,
        )?;
        let mut weightless = config.norm_config.clone();
        weightless.has_scale = false;
        let router_tree = tree.subtree("router").subtree("weights");
        let router = LinearMatmul::load(
            context,
            router_tree.metadata("spec").map_err(LinearBlockError::ParameterError)?,
            model,
            config.num_experts,
            dtype,
            dtype,
            DataType::F32,
            &router_tree,
            None,
            None,
        )
        .map_err(LinearBlockError::LinearMatmulError)?;
        Ok(Self {
            dense,
            dense_norm: normalize("dense_output_norm", &config.norm_config)?,
            expert_input_norm: normalize("expert_input_norm", &config.norm_config)?,
            expert_output_norm: normalize("expert_output_norm", &config.norm_config)?,
            router_norm: normalize("router_norm", &weightless)?,
            router_scale: tree
                .leaf("router_scale")
                .map_err(LinearBlockError::ParameterError)?
                .validate(&[model], dtype)
                .map_err(LinearBlockError::ParameterError)?
                .read_buffer()
                .map_err(LinearBlockError::ParameterError)?,
            expert_scales: tree
                .leaf("expert_scales")
                .map_err(LinearBlockError::ParameterError)?
                .validate(&[config.num_experts], DataType::F32)
                .map_err(LinearBlockError::ParameterError)?
                .read_buffer()
                .map_err(LinearBlockError::ParameterError)?,
            router,
            experts: RoutedExperts::new(
                context,
                model,
                config.expert_hidden_dim,
                config.num_experts,
                config.active_experts,
                &config.expert_config,
                &tree.subtree("experts"),
                false,
                RoutingPrecision {
                    round_logits_bf16: true,
                    round_weights_bf16: false,
                    round_products_bf16: true,
                    round_accumulator_bf16: true,
                    sort_expert_ids: true,
                },
            )?,
            scale: <B::Kernels as Kernels>::GemmaMoeRouterScaleBF16Kernel::new(context)
                .map_err(MlpBlockError::BackendError)?,
            add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, dtype, false)
                .map_err(MlpBlockError::BackendError)?,
            model,
            root_scale: config.router_root_scale,
        })
    }
}

impl<B: Backend> Mlp<B> for Gemma4Moe<B> {
    fn requires_residual_input(&self) -> bool {
        true
    }

    fn encode(
        &self,
        _input: B::ScratchBuffer,
        _rows: u32,
        _commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        Err(InvalidInput("Gemma experts require the original residual input").into())
    }

    fn encode_with_residual(
        &self,
        input: B::ScratchBuffer,
        residual: Option<B::ScratchBuffer>,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let residual = residual.ok_or(InvalidInput("Gemma expert residual missing"))?;
        let length = rows.checked_mul(self.model).ok_or(InvalidInput("Gemma expert row overflow"))?;
        if residual.size() != length as usize * 2 || input.size() != residual.size() {
            return Err(InvalidInput("Gemma expert residual geometry").into());
        }
        let router_input = self.router_norm.encode(&residual, 0, rows, None::<&mut B::ScratchBuffer>, commands)?;
        let mut scaled = commands.allocate_scratch_for_shape(&[rows, self.model], DataType::BF16)?;
        self.scale.encode(
            &router_input,
            &self.router_scale,
            &mut scaled,
            self.model,
            length,
            self.root_scale,
            commands,
        );
        let logits = self.router.encode_with_a(
            MatmulA::FullPrecision {
                values: &scaled,
                offset: 0,
            },
            rows,
            None::<Gather<&B::ScratchBuffer>>,
            commands,
        )?;
        let expert_input =
            self.expert_input_norm.encode(&residual, 0, rows, None::<&mut B::ScratchBuffer>, commands)?;
        let expert = self.experts.encode_scaled(&expert_input, &logits, Some(&self.expert_scales), rows, commands)?;
        let expert = self.expert_output_norm.encode(&expert, 0, rows, None::<&mut B::ScratchBuffer>, commands)?;
        let dense = self.dense.encode(input, rows, commands)?;
        let dense = self.dense_norm.encode(&dense, 0, rows, None::<&mut B::ScratchBuffer>, commands)?;
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.model], DataType::BF16)?;
        self.add.encode(Some(&dense), &expert, &mut output, length, length, 1.0, commands);
        Ok(output)
    }
}
