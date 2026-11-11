//! MLP block encodable.

use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding},
    encodable_block::{
        linear::Linear,
        mlp::{Mlp, gate_act_mul::MlpGateActMulEncodable},
    },
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
}

impl<B: Backend> DenseMlp<B> {
    pub fn new(
        up: Box<dyn Linear<B>>,
        gate: MlpGateActMulEncodable<B>,
        down: Box<dyn Linear<B>>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
        }
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    #[cfg(test)]
    fn trace_dense(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<Vec<(String, B::ScratchBuffer)>>, B::Error> {
        use crate::{
            backends::common::{Buffer, InvalidInput},
            encodable_block::linear::LinearInput,
        };
        let fused = self.up.encode(input, batch_dim, command)?;
        let mut stages = self.gate.trace_gelu(&fused, batch_dim, command)?;
        let format = self.down.select_activation_format(batch_dim, command.context());
        let LinearInput::FullPrecision(product) = self.gate.encode_for_linear(command, &fused, batch_dim, format)?
        else {
            return Err(InvalidInput("MLP diagnostic requires BF16 product").into());
        };
        let mut product_copy = command.allocate_scratch(product.size())?;
        command.encode_copy(&product, &mut product_copy);
        let down = self.down.encode(product, batch_dim, command)?;
        let candidate_product = stages
            .iter()
            .find(|(name, _)| name == "mlp.reference_product_candidate")
            .ok_or(InvalidInput("MLP diagnostic candidate missing"))?;
        let mut candidate = command.allocate_scratch(candidate_product.1.size())?;
        command.encode_copy(&candidate_product.1, &mut candidate);
        let candidate_down = self.down.encode(candidate, batch_dim, command)?;
        stages.insert(0, ("mlp.fused_up".into(), fused));
        stages.push(("mlp.product".into(), product_copy));
        stages.push(("mlp.down".into(), down));
        stages.push(("mlp.reference_down_candidate".into(), candidate_down));
        Ok(Some(stages))
    }

    #[cfg(test)]
    fn validation_trace_dense(
        &self,
        input: B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<Vec<(&'static str, B::ScratchBuffer)>>, B::Error> {
        use crate::{backends::common::Buffer, encodable_block::linear::LinearInput};
        if rows == 0 || rows > 16 {
            return Err(crate::backends::common::InvalidInput("small dense trace required").into());
        }
        let fused_up = self.up.encode(input, rows, commands)?;
        let activated = self.gate.validation_activation(&fused_up, rows, commands)?;
        let format = self.down.select_activation_format(rows, commands.context());
        let product = self.gate.encode_for_linear(commands, &fused_up, rows, format)?;
        let LinearInput::FullPrecision(product) = product else {
            return Err(crate::backends::common::InvalidInput("full precision dense trace required").into());
        };
        let mut saved_product = commands.allocate_scratch(product.size())?;
        commands.encode_copy(&product, &mut saved_product);
        let down = self.down.encode_input(LinearInput::FullPrecision(product), rows, commands)?;
        Ok(Some(vec![
            ("mlp_fused_up", fused_up),
            ("mlp_activation", activated),
            ("mlp_product", saved_product),
            ("mlp_down", down),
        ]))
    }

    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("mlp (dense)");

        let fused_up = self.up.encode(input, batch_dim, command_buffer)?;
        let act_format = self.down.select_activation_format(batch_dim, command_buffer.context());
        let down_input = self.gate.encode_for_linear(command_buffer, &fused_up, batch_dim, act_format)?;
        let output = self.down.encode_input(down_input, batch_dim, command_buffer)?;

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
