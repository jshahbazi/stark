use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
            CommandBufferExecutable, CommandBufferPending, Context, allocator::bump::BumpAllocator,
        },
        cpu::{Cpu, buffer::CpuBufferExt, context::CpuContext, error::CpuError},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct CpuCommandBuffer;

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    type Encoding = CpuCommandBufferEncoding;
    type Executable = CpuCommandBufferExecutable;
    type Pending = CpuCommandBufferPending;
    type Completed = CpuCommandBufferCompleted;
}

pub struct CpuCommandBufferEncoding {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    context: Arc<CpuContext>,
}

impl CpuCommandBufferEncoding {
    pub fn new(
        constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
        allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
        context: Arc<CpuContext>,
    ) -> CpuCommandBufferEncoding {
        CpuCommandBufferEncoding {
            commands: Vec::new(),
            constant_allocator,
            allocation_pool,
            context,
        }
    }

    pub fn push_command(
        &mut self,
        command: impl FnOnce() + Send + 'static,
    ) {
        self.commands.push(Box::new(command))
    }
}

impl CommandBufferEncoding for CpuCommandBufferEncoding {
    type CommandBuffer = CpuCommandBuffer;

    fn context(&self) -> &CpuContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Cpu as Backend>::ConstantBuffer, CpuError> {
        self.constant_allocator.allocate(size, |size| self.context.create_buffer(size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Cpu as Backend>::ScratchBuffer, CpuError> {
        let buffer = self.allocation_pool.allocate(size, |size| self.context.create_buffer(size))?;
        // Recycled pages remain readable through the public buffer API.
        // Clear the full page because PoolAllocation::page also exposes its padding.
        unsafe {
            buffer.page().cpu_address().as_ptr().cast::<u8>().write_bytes(0, Buffer::size(buffer.page()));
        }
        Ok(buffer)
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Cpu>,
        dst: impl BufferMut<Backend = Cpu>,
    ) {
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        assert_eq!(src_range.iter().len(), dst_range.iter().len());

        let src_ptr = SendPtr(unsafe { src.cpu_address().as_ptr().cast::<u8>().add(src_range.start) });
        let dst_ptr = SendPtrMut(unsafe { dst.cpu_address().as_ptr().cast::<u8>().add(dst_range.start) });

        self.push_command(move || unsafe {
            std::ptr::copy(src_ptr.as_ptr(), dst_ptr.as_ptr(), src_range.iter().len());
        });
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Cpu>,
        value: u8,
    ) {
        let (dst, range) = dst.parts();
        let size = range.iter().len();
        let dst = SendPtrMut(unsafe { dst.cpu_address().as_ptr().cast::<u8>().add(range.start) });
        self.push_command(move || unsafe {
            dst.as_ptr().write_bytes(value, size);
        });
    }

    fn push_debug_group(
        &mut self,
        _name: &str,
    ) {
    }

    fn pop_debug_group(&mut self) {}

    fn end_encoding(self) -> CpuCommandBufferExecutable {
        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");
        CpuCommandBufferExecutable {
            commands: self.commands,
            constant_allocator: self.constant_allocator,
            allocation_pool: self.allocation_pool,
            context: self.context,
        }
    }
}

pub struct CpuCommandBufferExecutable {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    context: Arc<CpuContext>,
}

impl CommandBufferExecutable for CpuCommandBufferExecutable {
    type CommandBuffer = CpuCommandBuffer;

    fn submit(self) -> CpuCommandBufferPending {
        let (return_sender, return_receiver) = mpsc::channel();
        let allocation_pool = self.allocation_pool.clone();

        self.context
            .command_queue
            .send(Box::new(move || {
                let start = Instant::now();

                for command in self.commands {
                    command()
                }

                let gpu_execution_time = start.elapsed();

                let completed = CpuCommandBufferCompleted {
                    gpu_execution_time,
                    _allocation_pool: self.allocation_pool,
                };

                let _ = return_sender.send(completed);
                drop(self.constant_allocator);
            }))
            .unwrap();

        CpuCommandBufferPending {
            return_receiver,
            _allocation_pool: allocation_pool,
        }
    }
}

pub struct CpuCommandBufferPending {
    return_receiver: mpsc::Receiver<CpuCommandBufferCompleted>,
    _allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferPending for CpuCommandBufferPending {
    type CommandBuffer = CpuCommandBuffer;

    fn wait_until_completed(self) -> Result<CpuCommandBufferCompleted, CpuError> {
        Ok(self.return_receiver.recv()?)
    }
}

pub struct CpuCommandBufferCompleted {
    gpu_execution_time: Duration,
    _allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for CpuCommandBufferCompleted {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}

#[cfg(test)]
mod initialized_scratch_tests {
    use stark_engine_macros::stark_test;

    use super::CpuContext;
    use crate::backends::common::{BufferMut, BufferRef, CommandBufferEncoding, Context};

    #[stark_test]
    fn recycled_scratch_clears_readable_page_padding() {
        let context = CpuContext::new().unwrap();
        let pool = context.create_allocation_pool();
        let mut commands = context.create_command_buffer(None, Some(pool)).unwrap();
        let mut previous = commands.allocate_scratch(64).unwrap();
        previous.as_slice_mut::<u8>().fill(0xa5);
        drop(previous);
        let next = commands.allocate_scratch(33).unwrap();
        assert!(next.as_slice::<u8>().iter().all(|&byte| byte == 0));
        assert!(next.page().as_slice::<u8>().iter().all(|&byte| byte == 0));
    }

    #[stark_test]
    fn queued_cpu_storage_survives_forgotten_wait_and_owner_drop() {
        use std::sync::mpsc;

        use crate::backends::common::{CommandBufferExecutable, CommandBufferPending};

        let context = CpuContext::new().unwrap();
        let (release, gate) = mpsc::channel();
        context
            .command_queue
            .send(Box::new(move || {
                let _ = gate.recv();
            }))
            .unwrap();
        struct Release(Option<mpsc::Sender<()>>);
        impl Drop for Release {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let release = Release(Some(release));

        let mut source = context.create_buffer(4).unwrap();
        source.copyin(&[42u32]);
        let source_allocation = source.allocation_weak();
        let mut state = context.create_buffer(4).unwrap();
        let mut first = context.create_command_buffer(None, None).unwrap();
        first.encode_copy(&source, &mut state);
        let pending = first.end_encoding().submit();
        std::mem::forget(pending);
        drop(source);
        assert!(source_allocation.upgrade().is_some(), "queued source allocation was freed before execution");

        // A fresh caller can safely reuse the state after forgetting the old
        // borrowed operation: its commands still run after the prior submission.
        let mut original = context.create_buffer(4).unwrap();
        let mut reused = context.create_buffer(4).unwrap();
        let mut second = context.create_command_buffer(None, None).unwrap();
        second.encode_copy(&state, &mut original);
        second.encode_fill(&mut state, 7);
        second.encode_copy(&state, &mut reused);
        let pending = second.end_encoding().submit();
        drop(release);
        pending.wait_until_completed().unwrap();
        assert_eq!(original.copyout::<u32>(), [42]);
        assert!(reused.copyout::<u8>().iter().all(|&value| value == 7));
        assert!(source_allocation.upgrade().is_none(), "retirement did not release completed storage");
    }
}
