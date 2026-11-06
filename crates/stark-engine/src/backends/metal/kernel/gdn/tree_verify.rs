use metal::MTLGPUFamily;

use super::super::DeltaNetShortChainVerifyMetalKernel;
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Kernels,
            kernel::{
                BuildTreeGramKernel, BuildTreeOutKernel, BuildTreePrefixKernel, TreeUpdateSolveKernel,
                delta_net_tree_verify::DeltaNetTreeVerify,
            },
        },
        metal::{
            Metal, MetalContext, command_buffer::MetalCommandBufferEncoding, context::LARGE_MIN_GPU_CORES,
            error::MetalError, kernel::MetalKernels,
        },
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const TOKEN_BLOCK: u32 = 16;
const BLOCK_PAIR_WIDTH: u32 = 2 * TOKEN_BLOCK;
const INNER_DATA_TYPE: DataType = DataType::F32;

fn supports_short_chain(
    arguments: &TreeVerifyNewArguments,
    use_mxu: bool,
) -> bool {
    use_mxu
        && arguments.data_type == DataType::BF16
        && arguments.head_k_dim == 128
        && arguments.head_v_dim == 128
        && arguments.num_k_heads == 16
        && arguments.num_v_heads == 48
}

fn short_chain_size_supported(size: u32) -> bool {
    (2..=4).contains(&size)
}

struct Layout {
    tree_size: u32,
    num_blocks: u32,
    num_block_pairs: u32,
    num_v_heads: u32,
    head_v_dim: u32,
}

impl Layout {
    const fn new(
        tree_size: u32,
        arguments: &TreeVerifyNewArguments,
    ) -> Self {
        let num_blocks = tree_size.div_ceil(TOKEN_BLOCK);
        Self {
            tree_size,
            num_blocks,
            num_block_pairs: num_blocks.div_ceil(2),
            num_v_heads: arguments.num_v_heads,
            head_v_dim: arguments.head_v_dim,
        }
    }

    const fn a_packed_shape(&self) -> [u32; 5] {
        [self.num_v_heads, self.num_blocks, self.num_block_pairs, TOKEN_BLOCK, BLOCK_PAIR_WIDTH]
    }

    const fn a_inverse_shape(&self) -> [u32; 4] {
        [self.num_v_heads, self.num_blocks, TOKEN_BLOCK, TOKEN_BLOCK]
    }
}

pub struct MetalDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    prefix: <MetalKernels as Kernels>::BuildTreePrefixKernel,
    gram: <MetalKernels as Kernels>::BuildTreeGramKernel,
    solve: <MetalKernels as Kernels>::TreeUpdateSolveKernel,
    out: <MetalKernels as Kernels>::BuildTreeOutKernel,
    short_chain: Option<DeltaNetShortChainVerifyMetalKernel>,
}

impl DeltaNetTreeVerify for MetalDeltaNetTreeVerify {
    type Backend = Metal;

    fn is_supported(_context: &MetalContext) -> bool {
        true
    }

    fn new(
        context: &MetalContext,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, MetalError> {
        let use_mxu = arguments.data_type == DataType::BF16 && context.supports_mxu;
        let transposed_h0 = !use_mxu
            && context.gpu_core_count < LARGE_MIN_GPU_CORES
            && context.apple_gpu_family <= MTLGPUFamily::Apple8;
        Ok(Self {
            arguments: *arguments,
            prefix: <MetalKernels as Kernels>::BuildTreePrefixKernel::new(context)?,
            gram: <MetalKernels as Kernels>::BuildTreeGramKernel::new(context, arguments.data_type, use_mxu, true)?,
            solve: <MetalKernels as Kernels>::TreeUpdateSolveKernel::new(context, arguments.data_type, 32, true)?,
            out: <MetalKernels as Kernels>::BuildTreeOutKernel::new(
                context,
                arguments.data_type,
                arguments.data_type,
                use_mxu,
                transposed_h0,
                true,
            )?,
            short_chain: supports_short_chain(arguments, use_mxu)
                .then(|| DeltaNetShortChainVerifyMetalKernel::new(context))
                .transpose()?,
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Metal>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let layout = Layout::new(arguments.tree_size, &self.arguments);
        let mut h0_indices = command_buffer.allocate_scratch(DataType::I32.size_in_bytes())?;
        command_buffer.encode_fill(&mut h0_indices, 0);

        let mut prefix =
            command_buffer.allocate_scratch_for_shape(&[layout.tree_size, layout.num_v_heads], INNER_DATA_TYPE)?;
        let mut a_packed = command_buffer.allocate_scratch_for_shape(&layout.a_packed_shape(), INNER_DATA_TYPE)?;
        let mut qkd = command_buffer
            .allocate_scratch_for_shape(&[layout.num_v_heads, layout.tree_size, layout.tree_size], INNER_DATA_TYPE)?;
        let mut a_inverse = command_buffer.allocate_scratch_for_shape(&layout.a_inverse_shape(), INNER_DATA_TYPE)?;
        let mut kh0 = command_buffer
            .allocate_scratch_for_shape(&[layout.tree_size, layout.num_v_heads, layout.head_v_dim], INNER_DATA_TYPE)?;
        let mut u = command_buffer
            .allocate_scratch_for_shape(&[layout.num_v_heads, layout.tree_size, layout.head_v_dim], INNER_DATA_TYPE)?;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[layout.tree_size, layout.num_v_heads, layout.head_v_dim],
            self.arguments.data_type,
        )?;

        self.prefix.encode(
            arguments.trie,
            arguments.log_decay,
            &mut prefix,
            1,
            arguments.tree_size,
            self.arguments.num_v_heads,
            command_buffer,
        );
        self.gram.encode(
            arguments.q,
            arguments.k,
            arguments.trie,
            &prefix,
            arguments.beta,
            Some(arguments.h0),
            Some(&h0_indices),
            &mut a_packed,
            &mut qkd,
            &mut a_inverse,
            Some(&mut kh0),
            1.0,
            1,
            arguments.tree_size,
            self.arguments.num_k_heads,
            self.arguments.num_v_heads,
            self.arguments.head_k_dim,
            self.arguments.head_v_dim,
            command_buffer,
        );
        self.solve.encode(
            Some(&kh0),
            arguments.v,
            &prefix,
            arguments.beta,
            &a_packed,
            &a_inverse,
            Some(&h0_indices),
            &mut u,
            1,
            arguments.tree_size,
            self.arguments.num_v_heads,
            self.arguments.head_v_dim,
            command_buffer,
        );
        self.out.encode(
            arguments.q,
            &prefix,
            &qkd,
            &u,
            Some(arguments.h0),
            Some(&h0_indices),
            &mut output,
            1.0,
            1,
            arguments.tree_size,
            self.arguments.num_k_heads,
            self.arguments.num_v_heads,
            self.arguments.head_k_dim,
            self.arguments.head_v_dim,
            command_buffer,
        );
        Ok(output)
    }

    fn encode_short_chain(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Metal>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<Option<<Metal as Backend>::ScratchBuffer>, MetalError> {
        let Some(kernel) = &self.short_chain else {
            return Ok(None);
        };
        if !short_chain_size_supported(arguments.tree_size) {
            return Ok(None);
        }
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.tree_size, self.arguments.num_v_heads, self.arguments.head_v_dim],
            self.arguments.data_type,
        )?;
        kernel.encode(
            arguments.q,
            arguments.k,
            arguments.v,
            arguments.log_decay,
            arguments.beta,
            arguments.h0,
            &mut output,
            arguments.tree_size,
            self.arguments.num_v_heads,
            self.arguments.num_k_heads,
            command_buffer,
        );
        Ok(Some(output))
    }
}

#[cfg(test)]
mod short_chain_policy_tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn short_chain_fusion_requires_qualified_geometry() {
        let args = TreeVerifyNewArguments {
            data_type: DataType::BF16,
            num_k_heads: 16,
            num_v_heads: 48,
            head_k_dim: 128,
            head_v_dim: 128,
        };
        assert!(supports_short_chain(&args, true));
        assert!(!supports_short_chain(&args, false));
        for changed in [
            TreeVerifyNewArguments {
                data_type: DataType::F32,
                ..args
            },
            TreeVerifyNewArguments {
                num_k_heads: 8,
                ..args
            },
            TreeVerifyNewArguments {
                num_v_heads: 32,
                ..args
            },
            TreeVerifyNewArguments {
                head_k_dim: 64,
                ..args
            },
            TreeVerifyNewArguments {
                head_v_dim: 64,
                ..args
            },
        ] {
            assert!(!supports_short_chain(&changed, true));
        }
        for size in [0, 1, 5, 16, 32] {
            assert!(!short_chain_size_supported(size));
        }
        for size in 2..=4 {
            assert!(short_chain_size_supported(size));
        }
    }
}
