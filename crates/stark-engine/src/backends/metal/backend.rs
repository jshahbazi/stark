use crate::backends::{
    common::{
        Backend,
        allocator::{
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    metal::{
        buffer::{global::MetalGlobalBuffer, sparse::MetalSparseBuffer},
        command_buffer::MetalCommandBuffer,
        context::MetalContext,
        error::MetalError,
        kernel::MetalKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type GlobalBuffer = MetalGlobalBuffer;
    type ConstantBuffer = BumpAllocation<Self::GlobalBuffer>;
    type ScratchBuffer = PoolAllocation<Self::GlobalBuffer>;
    type SparseBuffer = MetalSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalBuffer>;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const NAME: &'static str = "metal";

    // Scratch allocation never touches buffer contents. Every encoder begins
    // and ends with Device queue barriers, and constants have their own bump
    // allocator, so GPU-only scratch can alias across queued command buffers.
    const QUEUED_SCRATCH_REUSE_SAFE: bool = true;
}
