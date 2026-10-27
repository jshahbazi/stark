//! Private, full-F32 port of the pinned HF 64-token chunk rule.
use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{
            ReferenceGDNChunkGramKernel, ReferenceGDNChunkNewValuesKernel, ReferenceGDNChunkOutputKernel,
            ReferenceGDNChunkPrepareKernel, ReferenceGDNChunkSolveKernel, ReferenceGDNChunkStateKernel,
        },
    },
    data_type::DataType,
};

pub(super) struct ReferenceChunk<B: Backend> {
    prepare: <B::Kernels as Kernels>::ReferenceGDNChunkPrepareKernel,
    gram: <B::Kernels as Kernels>::ReferenceGDNChunkGramKernel,
    solve: <B::Kernels as Kernels>::ReferenceGDNChunkSolveKernel,
    vnew: <B::Kernels as Kernels>::ReferenceGDNChunkNewValuesKernel,
    output: <B::Kernels as Kernels>::ReferenceGDNChunkOutputKernel,
    state: <B::Kernels as Kernels>::ReferenceGDNChunkStateKernel,
}

impl<B: Backend> ReferenceChunk<B> {
    pub(super) fn new(context: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            prepare: <B::Kernels as Kernels>::ReferenceGDNChunkPrepareKernel::new(context)?,
            gram: <B::Kernels as Kernels>::ReferenceGDNChunkGramKernel::new(context)?,
            solve: <B::Kernels as Kernels>::ReferenceGDNChunkSolveKernel::new(context)?,
            vnew: <B::Kernels as Kernels>::ReferenceGDNChunkNewValuesKernel::new(context)?,
            output: <B::Kernels as Kernels>::ReferenceGDNChunkOutputKernel::new(context)?,
            state: <B::Kernels as Kernels>::ReferenceGDNChunkStateKernel::new(context)?,
        })
    }

    pub(super) fn encode(
        &self,
        q_input: &B::ScratchBuffer,
        k_input: &B::ScratchBuffer,
        beta: &B::ScratchBuffer,
        log_decay: &B::ScratchBuffer,
        values: &B::ScratchBuffer,
        state: &mut B::GlobalBuffer,
        output: &mut B::ScratchBuffer,
        rows: u32,
        heads: u32,
        groups: u32,
        key_dim: u32,
        value_dim: u32,
        row_stride: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let chunks = rows.div_ceil(64);
        let planes = chunks * heads;
        let mut q = commands.allocate_scratch_for_shape(&[planes, 64, 128], DataType::F32)?;
        let mut k = commands.allocate_scratch_for_shape(&[planes, 64, 128], DataType::F32)?;
        let mut kb = commands.allocate_scratch_for_shape(&[planes, 64, 128], DataType::F32)?;
        let mut vb = commands.allocate_scratch_for_shape(&[planes, 64, 128], DataType::F32)?;
        let mut cumulative = commands.allocate_scratch_for_shape(&[planes, 64], DataType::F32)?;
        let mut triangular = commands.allocate_scratch_for_shape(&[planes, 64, 64], DataType::F32)?;
        let mut attention = commands.allocate_scratch_for_shape(&[planes, 64, 64], DataType::F32)?;
        let mut u = commands.allocate_scratch_for_shape(&[planes, 64, 128], DataType::F32)?;
        let mut w = commands.allocate_scratch_for_shape(&[planes, 64, 128], DataType::F32)?;
        let mut vnew = commands.allocate_scratch_for_shape(&[heads, 64, 128], DataType::F32)?;
        self.prepare.encode(
            q_input,
            k_input,
            beta,
            log_decay,
            values,
            &mut q,
            &mut k,
            &mut kb,
            &mut vb,
            &mut cumulative,
            rows,
            heads,
            groups,
            key_dim,
            value_dim,
            row_stride,
            commands,
        );
        self.gram.encode(&q, &k, &kb, &cumulative, &mut triangular, &mut attention, planes, commands);
        self.solve.encode(&triangular, &kb, &vb, &cumulative, &mut u, &mut w, planes, commands);
        for chunk in 0..chunks {
            self.vnew.encode(&u, &w, &*state, &mut vnew, heads, chunk, commands);
            self.output.encode(&q, &attention, &cumulative, &*state, &vnew, &mut *output, heads, chunk, rows, commands);
            self.state.encode(&k, &cumulative, &vnew, &mut *state, heads, chunk, commands);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "reference_chunk_tests.rs"]
mod tests;
