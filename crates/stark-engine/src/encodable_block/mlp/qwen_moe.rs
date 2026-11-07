//! Qwen3.5 normalized top-k experts plus its sigmoid-gated shared expert.

use super::{
    Mlp, MlpBlockError,
    routed::{RoutedExperts, RoutingPrecision},
};
use crate::{
    backends::common::{
        Backend, Buffer, CommandBuffer, CommandBufferEncoding, InvalidInput,
        kernel::{Kernels, MoeSharedAddKernel, matmul::MatmulA},
    },
    config::mlp::{AnyMLPConfig, mixture_of_experts::MixtureOfExpertsConfig},
    data_type::DataType,
    encodable_block::linear::{Gather, LinearBlockError, LinearMatmul},
    parameters::ParameterTree,
};

pub(super) struct QwenMoe<B: Backend> {
    router: LinearMatmul<B>,
    experts: RoutedExperts<B>,
    shared: Box<dyn Mlp<B>>,
    shared_gate: LinearMatmul<B>,
    add: <B::Kernels as Kernels>::MoeSharedAddKernel,
    model: u32,
}

impl<B: Backend> QwenMoe<B> {
    pub fn new(
        config: &MixtureOfExpertsConfig,
        model: u32,
        shared_hidden: u32,
        context: &B::Context,
        tree: &ParameterTree<B>,
        dtype: DataType,
    ) -> Result<Self, MlpBlockError<B>> {
        if dtype != DataType::BF16
            || config.router_has_biases
            || config.num_shared_experts != 1
            || config.gate_config.is_none()
            || !tree.numerical_profile().qwen35_bf16_activations()
        {
            return Err(MlpBlockError::UnsupportedMixtureOfExperts);
        }
        let load = |name: &str, output, output_type| {
            let weights = tree.subtree(name).subtree("weights");
            let spec = weights.metadata("spec").map_err(LinearBlockError::ParameterError)?;
            LinearMatmul::load(
                context,
                spec,
                model,
                output,
                DataType::BF16,
                DataType::BF16,
                output_type,
                &weights,
                None,
                None,
            )
            .map_err(LinearBlockError::LinearMatmulError)
        };
        let router = load("router", config.num_routed_experts, DataType::F32)?;
        let shared_gate = load("shared_gate", 1, DataType::BF16)?;
        let experts = RoutedExperts::new(
            context,
            model,
            config.expert_hidden_dim,
            config.num_routed_experts,
            config.num_active_routed_experts,
            &config.expert_config,
            &tree.subtree("experts"),
            true,
            RoutingPrecision {
                round_logits_bf16: true,
                round_weights_bf16: true,
                round_products_bf16: true,
                ..Default::default()
            },
        )?;
        let dense = AnyMLPConfig::DenseMLPConfig(config.expert_config.clone());
        let (shared, _) =
            <dyn Mlp<B>>::new(&dense, model, shared_hidden, context, &tree.subtree("shared_expert"), dtype)?;
        Ok(Self {
            router,
            experts,
            shared,
            shared_gate,
            model,
            add: <B::Kernels as Kernels>::MoeSharedAddKernel::new(context).map_err(MlpBlockError::BackendError)?,
        })
    }
}

impl<B: Backend> Mlp<B> for QwenMoe<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let router = self.router.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            rows,
            None::<Gather<&B::ScratchBuffer>>,
            commands,
        )?;
        let expert = self.experts.encode(&input, &router, rows, commands)?;
        let gate = self.shared_gate.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            rows,
            None::<Gather<&B::ScratchBuffer>>,
            commands,
        )?;
        let shared = self.shared.encode(input, rows, commands)?;
        let length = rows.checked_mul(self.model).ok_or(InvalidInput("shared expert output overflow"))?;
        if expert.size() != length as usize * 2
            || shared.size() != length as usize * 2
            || gate.size() != rows as usize * 2
        {
            return Err(InvalidInput("shared expert output geometry").into());
        }
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.model], DataType::BF16)?;
        self.add.encode(&expert, &shared, &gate, &mut output, self.model, length, commands);
        Ok(output)
    }
}
