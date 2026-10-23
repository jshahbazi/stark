use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    os::raw::c_void,
    ptr::NonNull,
    sync::{Arc, mpsc::Sender},
};

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, GlobalBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuBuffer {
    allocation: Option<Arc<CpuAllocation>>,
    command_queue: Sender<Box<dyn FnOnce() + Send>>,
}

#[derive(Debug)]
pub(in crate::backends::cpu) struct CpuAllocation {
    ptr: NonNull<c_void>,
    layout: Layout,
}

impl CpuBuffer {
    pub(in crate::backends::cpu) fn new(
        size: usize,
        command_queue: Sender<Box<dyn FnOnce() + Send>>,
    ) -> Self {
        assert!(size > 0, "CPU buffer size must be nonzero");
        let layout = Layout::from_size_align(size, 64).expect("invalid CPU buffer layout");
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap_or_else(|| handle_alloc_error(layout)).cast();
        Self {
            allocation: Some(Arc::new(CpuAllocation {
                ptr,
                layout,
            })),
            command_queue,
        }
    }

    #[cfg(test)]
    pub(in crate::backends::cpu) fn allocation_weak(&self) -> std::sync::Weak<CpuAllocation> {
        Arc::downgrade(self.allocation.as_ref().unwrap())
    }
}

// The allocation has a stable address; callers synchronize access through buffer borrows and command buffers.
unsafe impl Send for CpuAllocation {}
unsafe impl Sync for CpuAllocation {}

impl Drop for CpuAllocation {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) };
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        // Supported callers retain the buffer through submission. Queue its
        // storage release after all submitted raw-pointer uses, including when
        // a borrowed stream or pending handle is deliberately forgotten.
        let allocation = self.allocation.take().unwrap();
        if let Err(unsent) = self.command_queue.send(Box::new(move || drop(allocation))) {
            (unsent.0)();
        }
    }
}

impl Buffer for CpuBuffer {
    type Backend = Cpu;

    fn size(&self) -> usize {
        self.allocation.as_ref().unwrap().layout.size()
    }
}

impl BufferCpuAccessible for CpuBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.allocation.as_ref().unwrap().ptr
    }
}

impl GlobalBuffer for CpuBuffer {}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use crate::backends::{
        common::{Backend, BufferMut, BufferRef, Context},
        cpu::Cpu,
    };

    #[stark_test]
    fn fresh_cpu_buffers_are_initialized() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        for size in [1, 3, 64, 65, 4096] {
            let mut previous = context.create_buffer(size).unwrap();
            previous.as_slice_mut::<u8>().fill(0xa5);
            drop(previous);
            let fresh = context.create_buffer(size).unwrap();
            assert!(fresh.as_slice::<u8>().iter().all(|&byte| byte == 0));
        }
    }
}
