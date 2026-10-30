//! Routed and shared experts using Uzu's standard gathered matmul on full banks.
use parking_lot::Mutex;

use super::{Encoding, Qwen38Error, project_batch, weights::Qwen38WeightLoader};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding, Kernels,
        kernel::{
            Qwen38MoeCombineKernel, Qwen38MoeIndicesKernel, Qwen38MoeRouteKernel, Qwen38SiluMultiplyKernel,
            matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulKernel},
        },
    },
    config::model::qwen38_language_model::Qwen38ArchitectureConfig,
    data_type::DataType,
    encodable_block::{linear::LinearMatmul, weight_matrix::WeightMatrix},
};

pub(super) struct Moe<B: Backend> {
    router: LinearMatmul<B>,
    gate: WeightMatrix<B>,
    up: WeightMatrix<B>,
    down: WeightMatrix<B>,
    shared_gate: LinearMatmul<B>,
    shared_up: LinearMatmul<B>,
    shared_down: LinearMatmul<B>,
    shared_router: LinearMatmul<B>,
    matmul: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    route: <B::Kernels as Kernels>::Qwen38MoeRouteKernel,
    indices: <B::Kernels as Kernels>::Qwen38MoeIndicesKernel,
    activation: <B::Kernels as Kernels>::Qwen38SiluMultiplyKernel,
    combine: <B::Kernels as Kernels>::Qwen38MoeCombineKernel,
    hidden: u32,
    intermediate: u32,
    shared: u32,
    experts: u32,
    selected: u32,
    normalize: u32,
}

impl<B: Backend> Moe<B> {
    pub fn new(
        context: &B::Context,
        cfg: &Qwen38ArchitectureConfig,
        layer: u32,
        loader: &Qwen38WeightLoader<B>,
    ) -> Result<Self, Qwen38Error<B>> {
        // The production model ranks raw logits and normalizes only its top-k
        // experts. A full-softmax router ranks rounded probabilities, whose
        // ties can differ; that policy needs separate implementation/validation.
        if !cfg.norm_topk_prob {
            return Err(Qwen38Error::Invalid("Qwen routing requires norm_topk_prob=true"));
        }
        if cfg.num_experts_per_tok == 0
            || cfg.num_experts_per_tok > cfg.num_experts
            || cfg.moe_intermediate_size == 0
            || cfg.shared_expert_intermediate_size == 0
        {
            return Err(Qwen38Error::Invalid("Qwen expert geometry"));
        }
        let p = format!("model.language_model.layers.{layer}.mlp");
        let linear = |name: &str, input, output, dtype| {
            loader.linear(&format!("{p}.{name}"), input, output, DataType::BF16, dtype)
        };
        Ok(Self {
            router: linear("gate", cfg.hidden_size, cfg.num_experts, DataType::F32)?,
            gate: loader.bank(
                &format!("{p}.experts.gate_proj"),
                cfg.num_experts,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
            )?,
            up: loader.bank(
                &format!("{p}.experts.up_proj"),
                cfg.num_experts,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
            )?,
            down: loader.bank(
                &format!("{p}.experts.down_proj"),
                cfg.num_experts,
                cfg.hidden_size,
                cfg.moe_intermediate_size,
            )?,
            shared_gate: linear(
                "shared_expert.gate_proj",
                cfg.hidden_size,
                cfg.shared_expert_intermediate_size,
                DataType::BF16,
            )?,
            shared_up: linear(
                "shared_expert.up_proj",
                cfg.hidden_size,
                cfg.shared_expert_intermediate_size,
                DataType::BF16,
            )?,
            shared_down: linear(
                "shared_expert.down_proj",
                cfg.shared_expert_intermediate_size,
                cfg.hidden_size,
                DataType::BF16,
            )?,
            shared_router: linear("shared_expert_gate", cfg.hidden_size, 1, DataType::F32)?,
            matmul: Mutex::new(
                <B::Kernels as Kernels>::MatmulKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
                    .map_err(Qwen38Error::Backend)?,
            ),
            route: <B::Kernels as Kernels>::Qwen38MoeRouteKernel::new(context).map_err(Qwen38Error::Backend)?,
            indices: <B::Kernels as Kernels>::Qwen38MoeIndicesKernel::new(context).map_err(Qwen38Error::Backend)?,
            activation: <B::Kernels as Kernels>::Qwen38SiluMultiplyKernel::new(context)
                .map_err(Qwen38Error::Backend)?,
            combine: <B::Kernels as Kernels>::Qwen38MoeCombineKernel::new(context).map_err(Qwen38Error::Backend)?,
            hidden: cfg.hidden_size,
            intermediate: cfg.moe_intermediate_size,
            shared: cfg.shared_expert_intermediate_size,
            experts: cfg.num_experts,
            selected: cfg.num_experts_per_tok,
            normalize: u32::from(cfg.norm_topk_prob),
        })
    }

    fn bank_into(
        &self,
        matrix: &WeightMatrix<B>,
        input: impl BufferRef<Backend = B>,
        indices: impl BufferRef<Backend = B>,
        result: impl BufferMut<Backend = B>,
        rows: u32,
        output: u32,
        reduction: u32,
        commands: &mut Encoding<B>,
    ) -> Result<(), B::Error> {
        self.matmul.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: input,
                    offset: 0,
                },
                b: matrix.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: result,
                d_transform: MatmulDOps::none(),
                gather_indices: Some(indices),
                m: rows,
                n: output,
                k: reduction,
            },
            commands,
        )
    }

    /// Gathered GEMV supports up to eight independent input rows. Splitting
    /// only M keeps every expert bank and its global affine strides intact.
    fn bank_batches(
        &self,
        matrix: &WeightMatrix<B>,
        input: impl BufferRef<Backend = B>,
        indices: impl BufferRef<Backend = B>,
        rows: u32,
        output: u32,
        reduction: u32,
        commands: &mut Encoding<B>,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut result = commands.allocate_scratch_for_shape(&[rows, output], DataType::BF16)?;
        for first in (0..rows).step_by(8) {
            let count = (rows - first).min(8);
            let input_span = first as usize * reduction as usize * 2..(first + count) as usize * reduction as usize * 2;
            let index_span = first as usize * output as usize * 4..(first + count) as usize * output as usize * 4;
            let output_span = first as usize * output as usize * 2..(first + count) as usize * output as usize * 2;
            self.bank_into(
                matrix,
                input.subrange(input_span),
                indices.subrange(index_span),
                (&mut result).subrange_mut(output_span),
                count,
                output,
                reduction,
                commands,
            )?;
        }
        Ok(result)
    }

    pub fn encode_batch(
        &self,
        input: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut Encoding<B>,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let logits = project_batch(&self.router, input, rows, commands)?;
        let mut selected = commands.allocate_scratch_for_shape(&[rows, self.selected], DataType::U32)?;
        let mut scores = commands.allocate_scratch_for_shape(&[rows, self.selected], DataType::F32)?;
        self.route.encode(
            &logits,
            &mut selected,
            &mut scores,
            self.experts,
            self.selected,
            self.normalize,
            rows,
            commands,
        );
        let mut gate_indices =
            commands.allocate_scratch_for_shape(&[rows, self.selected, self.intermediate], DataType::U32)?;
        let mut down_indices =
            commands.allocate_scratch_for_shape(&[rows, self.selected, self.hidden], DataType::U32)?;
        self.indices.encode(
            &selected,
            &mut gate_indices,
            &mut down_indices,
            self.intermediate,
            self.hidden,
            self.selected,
            rows,
            rows * self.selected * self.intermediate.max(self.hidden),
            commands,
        );
        let gate = self.bank_batches(
            &self.gate,
            input,
            &gate_indices,
            rows,
            self.selected * self.intermediate,
            self.hidden,
            commands,
        )?;
        let up = self.bank_batches(
            &self.up,
            input,
            &gate_indices,
            rows,
            self.selected * self.intermediate,
            self.hidden,
            commands,
        )?;
        let mut activated =
            commands.allocate_scratch_for_shape(&[rows, self.selected, self.intermediate], DataType::BF16)?;
        self.activation.encode(&gate, &up, &mut activated, rows * self.selected * self.intermediate, commands);
        // Flatten token-major (token, selected-slot) rows, so the final partial
        // group can cross a token boundary without changing expert IDs/order.
        let experts = self.bank_batches(
            &self.down,
            &activated,
            &down_indices,
            rows * self.selected,
            self.hidden,
            self.intermediate,
            commands,
        )?;
        let shared_gate = project_batch(&self.shared_gate, input, rows, commands)?;
        let shared_up = project_batch(&self.shared_up, input, rows, commands)?;
        let mut shared_activation = commands.allocate_scratch_for_shape(&[rows, self.shared], DataType::BF16)?;
        self.activation.encode(&shared_gate, &shared_up, &mut shared_activation, rows * self.shared, commands);
        let shared_output = project_batch(&self.shared_down, &shared_activation, rows, commands)?;
        let shared_router = project_batch(&self.shared_router, input, rows, commands)?;
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.hidden], DataType::BF16)?;
        self.combine.encode(
            &experts,
            &scores,
            &shared_output,
            &shared_router,
            &mut output,
            self.hidden,
            self.selected,
            rows,
            commands,
        );
        Ok(output)
    }
}
