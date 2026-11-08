//! Fixed-memory larger-tower source attention. No quadratic score plane.

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            Gemma4VisionFlashExpBF16Kernel, Gemma4VisionFlashFinishBF16Kernel, Gemma4VisionFlashPVF32Kernel,
            Gemma4VisionFlashQKKernel,
        },
    },
    data_type::DataType,
};

pub(super) struct FlashAttention<B: Backend> {
    qk: <B::Kernels as Kernels>::Gemma4VisionFlashQKKernel,
    exp: <B::Kernels as Kernels>::Gemma4VisionFlashExpBF16Kernel,
    pv: <B::Kernels as Kernels>::Gemma4VisionFlashPVF32Kernel,
    finish: <B::Kernels as Kernels>::Gemma4VisionFlashFinishBF16Kernel,
}

impl<B: Backend> FlashAttention<B> {
    pub(super) fn new(context: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            qk: <B::Kernels as Kernels>::Gemma4VisionFlashQKKernel::new(context)?,
            exp: <B::Kernels as Kernels>::Gemma4VisionFlashExpBF16Kernel::new(context)?,
            pv: <B::Kernels as Kernels>::Gemma4VisionFlashPVF32Kernel::new(context)?,
            finish: <B::Kernels as Kernels>::Gemma4VisionFlashFinishBF16Kernel::new(context)?,
        })
    }

    pub(super) fn encode(
        &self,
        q: &B::ScratchBuffer,
        k: &B::ScratchBuffer,
        v: &B::ScratchBuffer,
        query_rows: u32,
        key_rows: u32,
        heads: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode_with_observer(q, k, v, query_rows, key_rows, heads, commands, &mut |_, _, _, _| Ok(()))
    }

    pub(super) fn encode_with_observer<F>(
        &self,
        q: &B::ScratchBuffer,
        k: &B::ScratchBuffer,
        v: &B::ScratchBuffer,
        query_rows: u32,
        key_rows: u32,
        heads: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<B::ScratchBuffer, B::Error>
    where
        F: FnMut(
            &'static str,
            u32,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        if query_rows == 0 || query_rows > 2520 || key_rows == 0 || key_rows > 2520 || heads == 0 || heads > 16 {
            return Err(InvalidInput("larger Gemma source attention geometry outside bounds").into());
        }
        if q.size() < (heads * query_rows * 128 * 2) as usize
            || k.size() < (heads * key_rows * 128 * 2) as usize
            || v.size() < (heads * key_rows * 128 * 2) as usize
        {
            return Err(InvalidInput("larger Gemma source attention buffer outside bounds").into());
        }
        // Four fixed allocations, independent of key-tile count. PoolAllocation
        // drops return pages for reuse by later layers; Metal's access tracker
        // inserts device dispatch barriers on every read/write reuse.
        // At 2520x16, allocator-rounded scratch is 208.5MiB, plus 8MiB output.
        let mut scores = commands.allocate_scratch_for_shape(&[heads, query_rows, 512], DataType::F32)?;
        let mut exponents = commands.allocate_scratch_for_shape(&[heads, query_rows, 512], DataType::BF16)?;
        let mut state = commands.allocate_scratch_for_shape(&[heads, query_rows, 3], DataType::F32)?;
        let mut accumulator = commands.allocate_scratch_for_shape(&[heads, query_rows, 72], DataType::F32)?;
        for key_start in (0..key_rows).step_by(512) {
            self.qk.encode(q, k, &mut scores, query_rows, key_rows, heads, key_start, commands);
            observer("qk", key_start, &scores, commands)?;
            self.exp.encode(&scores, &mut exponents, &mut state, query_rows, heads, key_start, commands);
            observer("exp", key_start, &exponents, commands)?;
            observer("state", key_start, &state, commands)?;
            self.pv.encode(&exponents, v, &state, &mut accumulator, query_rows, key_rows, heads, key_start, commands);
            observer("pv", key_start, &accumulator, commands)?;
        }
        let mut output = commands.allocate_scratch_for_shape(&[query_rows, heads, 72], DataType::BF16)?;
        self.finish.encode(&accumulator, &state, &mut output, query_rows, heads, commands);
        Ok(output)
    }
}
