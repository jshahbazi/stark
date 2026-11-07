//! Sparse expert execution using canonical matrix banks and device gathers.
//!
//! Banks flatten `[experts, output, input]` to one logical matrix. Routing IDs
//! stay on device. Each token/active-expert pair dispatches only its selected
//! rows through the existing full-precision or packed affine matrix kernel.

use super::{MlpBlockError, gate_act_mul::MlpGateActMulEncodable};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput,
        kernel::{
            Kernels, MoeAccumulateKernel, MoeGatherRowsKernel, MoeRouteKernel,
            matmul::{ActivationFormat, MatmulA},
        },
    },
    config::mlp::dense_mlp::DenseMLPConfig,
    data_type::DataType,
    encodable_block::linear::{Gather, LinearBlockError, LinearInput, LinearMatmul},
    parameters::ParameterTree,
};

#[derive(Clone, Copy)]
pub(crate) struct RoutingPrecision {
    pub round_logits_bf16: bool,
    pub round_weights_bf16: bool,
    pub round_products_bf16: bool,
}

#[cfg(test)]
#[path = "routed_tests.rs"]
mod tests;

pub(crate) struct RoutedExperts<B: Backend> {
    up: LinearMatmul<B>,
    down: LinearMatmul<B>,
    gate: MlpGateActMulEncodable<B>,
    route: <B::Kernels as Kernels>::MoeRouteKernel,
    gather: <B::Kernels as Kernels>::MoeGatherRowsKernel,
    accumulate: <B::Kernels as Kernels>::MoeAccumulateKernel,
    model: u32,
    hidden: u32,
    experts: u32,
    active: u32,
    precision: RoutingPrecision,
}

impl<B: Backend> RoutedExperts<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &B::Context,
        model: u32,
        hidden: u32,
        experts: u32,
        active: u32,
        config: &DenseMLPConfig,
        tree: &ParameterTree<B>,
        reference_bf16: bool,
        precision: RoutingPrecision,
    ) -> Result<Self, MlpBlockError<B>> {
        if !(1..=1024).contains(&experts)
            || !(1..=32).contains(&active)
            || active > experts
            || model == 0
            || model > 16384
            || hidden == 0
            || hidden > 131072
            || config.has_up_biases
            || config.has_down_biases
        {
            return Err(MlpBlockError::BackendError(InvalidInput("unsupported routed expert geometry").into()));
        }
        let up_rows = experts
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| MlpBlockError::BackendError(InvalidInput("expert bank row overflow").into()))?;
        let down_rows = experts
            .checked_mul(model)
            .ok_or_else(|| MlpBlockError::BackendError(InvalidInput("expert bank row overflow").into()))?;
        if up_rows.checked_mul(model).is_none() || down_rows.checked_mul(hidden).is_none() {
            return Err(MlpBlockError::BackendError(InvalidInput("expert bank element indexing overflow").into()));
        }
        let up_tree = tree.subtree("up_projection").subtree("weights");
        let down_tree = tree.subtree("down_projection").subtree("weights");
        let up = LinearMatmul::load(
            context,
            up_tree.metadata("spec").map_err(LinearBlockError::ParameterError)?,
            model,
            up_rows,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            &up_tree,
            None,
            None,
        )
        .map_err(LinearBlockError::LinearMatmulError)?;
        let down = LinearMatmul::load(
            context,
            down_tree.metadata("spec").map_err(LinearBlockError::ParameterError)?,
            hidden,
            down_rows,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            &down_tree,
            None,
            None,
        )
        .map_err(LinearBlockError::LinearMatmulError)?;
        let gate = MlpGateActMulEncodable::new(
            context,
            DataType::BF16,
            config.activation.clone(),
            config.gate_clipping,
            config.up_clipping,
            hidden,
            None,
            reference_bf16,
        )
        .map_err(MlpBlockError::BackendError)?;
        Ok(Self {
            up,
            down,
            gate,
            model,
            hidden,
            experts,
            active,
            precision,
            route: <B::Kernels as Kernels>::MoeRouteKernel::new(context).map_err(MlpBlockError::BackendError)?,
            gather: <B::Kernels as Kernels>::MoeGatherRowsKernel::new(context).map_err(MlpBlockError::BackendError)?,
            accumulate: <B::Kernels as Kernels>::MoeAccumulateKernel::new(context)
                .map_err(MlpBlockError::BackendError)?,
        })
    }

    /// `input` may be normalized separately from the router's own input.
    /// Router logits must already be F32; no CPU readback occurs here.
    pub(crate) fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        router_logits: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let elements = rows.checked_mul(self.model).ok_or(InvalidInput("expert output indexing overflow"))?;
        if rows == 0
            || rows > 1024
            || input.size() != elements as usize * 2
            || router_logits.size() != rows as usize * self.experts as usize * 4
        {
            return Err(InvalidInput("routed expert buffer geometry").into());
        }
        let mut ids = commands.allocate_scratch_for_shape(&[rows, self.active], DataType::U32)?;
        let mut weights = commands.allocate_scratch_for_shape(&[rows, self.active], DataType::F32)?;
        self.route.encode(
            router_logits,
            &mut ids,
            &mut weights,
            rows,
            self.experts,
            self.active,
            u32::from(self.precision.round_weights_bf16),
            u32::from(self.precision.round_logits_bf16),
            commands,
        );
        let mut accumulator = commands.allocate_scratch_for_shape(&[rows, self.model], DataType::F32)?;
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.model], DataType::BF16)?;
        for row in 0..rows {
            for rank in 0..self.active {
                let route = row * self.active + rank;
                let mut up_ids = commands.allocate_scratch_for_shape(&[2 * self.hidden], DataType::U32)?;
                self.gather.encode(&ids, &mut up_ids, route, 2 * self.hidden, self.experts, 2 * self.hidden, commands);
                let projected = self.up.encode_with_a(
                    MatmulA::FullPrecision {
                        values: input.subrange(
                            row as usize * self.model as usize * 2..(row + 1) as usize * self.model as usize * 2,
                        ),
                        offset: 0,
                    },
                    1,
                    Some(Gather {
                        indices: &up_ids,
                        output_dim: 2 * self.hidden,
                    }),
                    commands,
                )?;
                let activated = self.gate.encode_for_linear(commands, &projected, 1, ActivationFormat::Bf16)?;
                let LinearInput::FullPrecision(activated) = activated else {
                    return Err(InvalidInput("routed expert activation format").into());
                };
                let mut down_ids = commands.allocate_scratch_for_shape(&[self.model], DataType::U32)?;
                self.gather.encode(&ids, &mut down_ids, route, self.model, self.experts, self.model, commands);
                let projected = self.down.encode_with_a(
                    MatmulA::FullPrecision {
                        values: &activated,
                        offset: 0,
                    },
                    1,
                    Some(Gather {
                        indices: &down_ids,
                        output_dim: self.model,
                    }),
                    commands,
                )?;
                self.accumulate.encode(
                    &projected,
                    &weights,
                    &mut accumulator,
                    &mut output,
                    self.model,
                    route,
                    row * self.model,
                    u32::from(rank == 0),
                    u32::from(rank + 1 == self.active),
                    u32::from(self.precision.round_products_bf16),
                    commands,
                );
            }
        }
        Ok(output)
    }
}
