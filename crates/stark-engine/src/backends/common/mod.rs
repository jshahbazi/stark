//! Backend selection for the engine's validated model APIs.
//!
//! Native resource construction and dispatch are internal implementation details.
//! External callers cannot create unchecked command buffers through a generic backend.
//!
//! ```compile_fail
//! use stark_engine::backends::common::Backend;
//! fn unchecked_dispatch<B: Backend>(context: &B::Context) {
//!     let _ = context.create_command_buffer(None, None);
//! }
//! ```
//!
//! Generated kernel dispatch is internal as well.
//!
//! ```compile_fail
//! use stark_engine::backends::common::kernel::TensorAddBiasKernel;
//! ```
//!
//! The public engine constructor remains available to generic backend callers.
//!
//! ```no_run
//! use stark_engine::{backends::common::Backend, engine::Engine};
//! fn create_engine<B: Backend>() {
//!     let _ = Engine::<B>::new();
//! }
//! ```

pub(super) mod allocator;
mod backend;
mod buffer;
mod command_buffer;
mod context;
mod device_capabilities;
// These layouts are also consumed by generated shader headers.
#[allow(dead_code)]
pub(crate) mod gpu_types;
mod input_error;
pub(crate) mod kernel;
pub(crate) mod vision_math;

pub use backend::Backend;
pub(crate) use buffer::{
    Buffer,
    constant::ConstantBuffer,
    cpu_accessible::BufferCpuAccessible,
    global::GlobalBuffer,
    reference::{BufferMut, BufferRef},
    scratch::ScratchBuffer,
    sparse::SparseBuffer,
};
pub(crate) use command_buffer::{
    CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
};
pub(crate) use context::Context;
pub(crate) use device_capabilities::DeviceCapabilities;
pub use input_error::InvalidInput;
pub(crate) use kernel::Kernels;
