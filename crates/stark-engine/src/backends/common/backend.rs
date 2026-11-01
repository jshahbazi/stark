use std::error::Error;

use super::{CommandBuffer, ConstantBuffer, Context, GlobalBuffer, Kernels, ScratchBuffer, SparseBuffer};

pub trait Backend: Clone + Send + Sync + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type GlobalBuffer: GlobalBuffer<Backend = Self>;
    type ConstantBuffer: ConstantBuffer<Backend = Self>;
    type ScratchBuffer: ScratchBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type AllocationPool: Send + Sync;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error + From<super::InvalidInput>;

    const NAME: &'static str;

    /// Scratch pages can be recycled while earlier submissions still use them.
    /// Enabling this requires allocation/reuse to avoid all host reads/writes,
    /// queue barriers to order prior and subsequent device access with memory
    /// visibility, and CPU-written constants to use separate per-buffer storage.
    const QUEUED_SCRATCH_REUSE_SAFE: bool = false;
}
