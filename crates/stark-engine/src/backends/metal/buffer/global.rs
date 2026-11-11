use std::{os::raw::c_void, ptr::NonNull, range::Range, sync::Arc};

use metal::{MTL4CommandQueue, MTLBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, GlobalBuffer, allocator::block::BlockAllocation},
    metal::{
        Metal,
        buffer::{
            dense::MetalDenseBuffer,
            retirement::{RetirementTracker, retire_after_queue},
        },
    },
};

pub struct MetalGlobalBuffer {
    allocation: Option<BlockAllocation<MetalDenseBuffer>>,
    command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    retirements: Arc<RetirementTracker>,
}

impl MetalGlobalBuffer {
    pub(in crate::backends::metal) fn new(
        allocation: BlockAllocation<MetalDenseBuffer>,
        command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
        retirements: Arc<RetirementTracker>,
    ) -> Self {
        Self {
            allocation: Some(allocation),
            command_queue,
            retirements,
        }
    }

    pub(super) fn page(&self) -> &MetalDenseBuffer {
        self.allocation.as_ref().unwrap().page()
    }

    pub(super) fn range(&self) -> Range<usize> {
        self.allocation.as_ref().unwrap().range()
    }
}

impl Drop for MetalGlobalBuffer {
    fn drop(&mut self) {
        // Do not return this page to the allocation cache before submitted uses
        // finish, even if a safe caller forgets a borrowed stream or wait handle.
        retire_after_queue(&self.command_queue, &self.retirements, self.allocation.take().unwrap());
    }
}

impl Buffer for MetalGlobalBuffer {
    type Backend = Metal;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for MetalGlobalBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().mtl_buffer().contents().byte_add(self.range().start) }
    }
}

impl GlobalBuffer for MetalGlobalBuffer {}

#[cfg(test)]
mod tests {
    use metal::{MTL4CommandQueue, MTLDeviceExt, MTLSharedEvent};
    use stark_engine_macros::stark_test;

    use crate::backends::{
        common::{BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        metal::{MetalContext, buffer::MetalBufferExt},
    };

    #[stark_test]
    fn queued_metal_storage_cannot_return_to_cache_before_completion() {
        let context = MetalContext::new().unwrap();
        let gate = context.device.new_shared_event().unwrap();
        context.command_queue.wait_for_event_value(gate.as_ref(), 1);
        struct Release(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn metal::MTLSharedEvent>>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.set_signaled_value(1);
            }
        }
        let release = Release(gate);

        let mut source = context.create_buffer(64).unwrap();
        let source_address = source.gpu_address();
        let mut state = context.create_buffer(64).unwrap();
        let mut first = context.create_command_buffer(None, None).unwrap();
        first.encode_fill(&mut source, 42);
        first.encode_copy(&source, &mut state);
        std::mem::forget(first.end_encoding().submit());
        drop(source);
        let mut replacement = context.create_buffer(64).unwrap();
        assert_ne!(replacement.gpu_address(), source_address, "queued allocation returned to cache early");
        replacement.as_slice_mut::<u8>().fill(13);

        let mut original = context.create_buffer(64).unwrap();
        let mut reused = context.create_buffer(64).unwrap();
        let mut second = context.create_command_buffer(None, None).unwrap();
        second.encode_copy(&state, &mut original);
        second.encode_fill(&mut state, 7);
        second.encode_copy(&state, &mut reused);
        let pending = second.end_encoding().submit();
        drop(release);
        pending.wait_until_completed().unwrap();
        assert!(original.copyout::<u8>().iter().all(|&value| value == 42));
        assert!(reused.copyout::<u8>().iter().all(|&value| value == 7));
        assert!(replacement.copyout::<u8>().iter().all(|&value| value == 13));
    }

    #[stark_test]
    fn pressure_reclamation_waits_for_queue_retirement_and_preserves_live_storage() {
        use std::{sync::mpsc, time::Duration};

        let context = MetalContext::new().unwrap();
        let gate = context.device.new_shared_event().unwrap();
        context.command_queue.wait_for_event_value(gate.as_ref(), 1);
        struct Release(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn metal::MTLSharedEvent>>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.set_signaled_value(1);
            }
        }
        let release = Release(gate);
        let mut queued = context.create_buffer(64).unwrap();
        let mut command = context.create_command_buffer(None, None).unwrap();
        command.encode_fill(&mut queued, 42);
        std::mem::forget(command.end_encoding().submit());
        drop(queued);
        assert_eq!(context.cached_memory_bytes(), Some(0));
        let mut live = context.create_buffer(128).unwrap();
        live.as_slice_mut::<u8>().fill(19);
        let worker_context = context.clone();
        let (sent, received) = mpsc::channel();
        let (started, begin) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            started.send(()).unwrap();
            sent.send(worker_context.reclaim_cached_memory()).unwrap();
        });
        begin.recv().unwrap();
        assert!(received.recv_timeout(Duration::from_millis(20)).is_err());
        drop(release);
        let reclaimed = received.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        thread.join().unwrap();
        assert!(reclaimed >= 16 * 1024);
        assert_eq!(context.cached_memory_bytes(), Some(0));
        assert!(live.copyout::<u8>().iter().all(|&value| value == 19));
        drop(live);
        assert!(context.reclaim_cached_memory().unwrap() >= 16 * 1024);
    }

    #[stark_test]
    fn sparse_resident_bytes_follow_successful_mapping_not_virtual_capacity() {
        use crate::backends::common::{Buffer, SparseBuffer};
        let context = MetalContext::new().unwrap();
        let page = context.sparse_page_size().unwrap();
        let mut cache = context.create_sparse_buffer(3 * page).unwrap();
        assert_eq!(cache.size(), 3 * page);
        assert_eq!(cache.mapped_size(), 0);
        cache.map(1).unwrap();
        assert_eq!(cache.mapped_size(), page);
        cache.map(page + 1).unwrap();
        assert_eq!(cache.mapped_size(), 2 * page);
        assert!(cache.map(3 * page + 1).is_err());
        assert_eq!(cache.mapped_size(), 2 * page);
        cache.map(1).unwrap();
        assert_eq!(cache.mapped_size(), 2 * page);
        drop(cache);
        context.reclaim_cached_memory().unwrap();
    }
}
