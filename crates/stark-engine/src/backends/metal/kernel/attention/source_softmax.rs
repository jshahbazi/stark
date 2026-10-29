use std::collections::{HashMap, hash_map::Entry};

use parking_lot::Mutex;

use crate::backends::{
    common::{BufferMut, CommandBufferEncoding},
    metal::{
        Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError,
        kernel::SourceBF16SoftmaxMetalKernel,
    },
};

pub(super) struct SourceBF16Softmax {
    pipelines: Mutex<HashMap<u32, SourceBF16SoftmaxMetalKernel>>,
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{
        backends::common::{CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        tests::helpers::{buffer_to_vec, create_buffer_with_data},
    };

    #[stark_test]
    fn source_bf16_softmax_has_independent_masked_probability_oracles_and_partial_rows() {
        let context = MetalContext::new().unwrap();
        let kernel = SourceBF16Softmax::new(&context).unwrap();
        for width in [16usize, 511, 512, 1024] {
            let mut input = vec![bf16::NEG_INFINITY; 3 * width];
            input[..4].fill(bf16::from_f32(-77.0));
            input[2 * width - 1] = bf16::from_f32(3.0);
            input[2 * width] = bf16::ONE;
            input[2 * width + 1] = bf16::ZERO;
            input[2 * width + 2] = -bf16::ONE;
            let mut values = create_buffer_with_data::<Metal, bf16>(&context, &input);
            let mut commands = context.create_command_buffer(None, None).unwrap();
            kernel.encode(&mut values, width as u32, 3, &mut commands).unwrap();
            commands.end_encoding().submit().wait_until_completed().unwrap();
            let actual = buffer_to_vec::<Metal, bf16>(&values);
            let denominator = 1.0f64.exp() + 1.0 + (-1.0f64).exp();
            for row in 0..3 {
                for column in 0..width {
                    let probability = match (row, column) {
                        (0, 0..=3) => 0.25,
                        (1, column) if column == width - 1 => 1.0,
                        (2, 0) => 1.0f64.exp() / denominator,
                        (2, 1) => 1.0 / denominator,
                        (2, 2) => (-1.0f64).exp() / denominator,
                        _ => 0.0,
                    };
                    assert!(actual[row * width + column].to_bits() == bf16::from_f64(probability).to_bits());
                }
            }
        }
    }
}

impl SourceBF16Softmax {
    pub(super) fn new(context: &MetalContext) -> Result<Self, MetalError> {
        let kernel = SourceBF16SoftmaxMetalKernel::new(context, 128)?;
        Ok(Self {
            pipelines: Mutex::new(HashMap::from([(128, kernel)])),
        })
    }

    pub(super) fn encode(
        &self,
        values: impl BufferMut<Backend = Metal>,
        row_width: u32,
        row_count: u32,
        commands: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        assert!(row_width > 0 && row_count > 0);
        let threads = row_width.div_ceil(4).div_ceil(32).saturating_mul(32).clamp(32, 1024);
        let mut pipelines = self.pipelines.lock();
        let kernel = match pipelines.entry(threads) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(SourceBF16SoftmaxMetalKernel::new(commands.context(), threads)?),
        };
        kernel.encode(values, row_width, row_count, commands);
        Ok(())
    }
}
