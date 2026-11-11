use std::{collections::VecDeque, mem::ManuallyDrop, range::Range, sync::Arc};

use parking_lot::Mutex;

pub struct PoolAllocator<T> {
    pages: [Mutex<VecDeque<T>>; 32],
    #[cfg(test)]
    requested_bytes: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    reserved_bytes: std::sync::atomic::AtomicUsize,
}

impl<T> PoolAllocator<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            pages: Default::default(),
            #[cfg(test)]
            requested_bytes: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            reserved_bytes: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn allocate<E>(
        self: &Arc<Self>,
        size: usize,
        upstream: impl FnOnce(usize) -> Result<T, E>,
    ) -> Result<PoolAllocation<T>, E> {
        assert!(size > 0 && size <= 2 * 1024 * 1024 * 1024);
        #[cfg(test)]
        self.requested_bytes.fetch_add(size, std::sync::atomic::Ordering::Relaxed);

        let page = if let Some(page) = self.pages[(size - 1).bit_width() as usize].lock().pop_front() {
            page
        } else {
            let page = upstream(size.next_power_of_two())?;
            #[cfg(test)]
            self.reserved_bytes.fetch_add(size.next_power_of_two(), std::sync::atomic::Ordering::Relaxed);
            page
        };

        Ok(PoolAllocation {
            allocator: self.clone(),
            page: ManuallyDrop::new(page),
            range: (0..size).into(),
        })
    }

    /// Cumulative requested scratch, an upper bound on simultaneous requested
    /// payload even when pages recycle. Excludes rounding and allocator overhead.
    #[cfg(test)]
    pub(crate) fn requested_bytes(&self) -> usize {
        self.requested_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Upstream payload retained by this pool, including page rounding. Does
    /// not include Metal command heaps, argument tables or constant pages.
    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reserved_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn alias(
        self: &Arc<Self>,
        page: T,
        range: Range<usize>,
    ) {
        self.pages[(range.iter().len() - 1).bit_width() as usize].lock().push_back(page);
    }
}

pub struct PoolAllocation<T> {
    allocator: Arc<PoolAllocator<T>>,
    page: ManuallyDrop<T>,
    range: Range<usize>,
}

impl<T> PoolAllocation<T> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl<T> Drop for PoolAllocation<T> {
    fn drop(&mut self) {
        self.allocator.alias(unsafe { ManuallyDrop::take(&mut self.page) }, self.range);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use stark_engine_macros::stark_test;

    use super::PoolAllocator;

    #[stark_test]
    fn live_scratch_handles_keep_their_pages_out_of_the_reuse_queue() {
        let pool = PoolAllocator::new();
        let next_page = Cell::new(0);
        let allocate = || {
            pool.allocate(4, |_| {
                let id = next_page.get();
                next_page.set(id + 1);
                Ok::<_, ()>(id)
            })
            .unwrap()
        };
        let retained = allocate();
        let temporary = allocate();
        assert_ne!(retained.page(), temporary.page());
        let temporary_page = *temporary.page();
        drop(temporary);
        let reused = allocate();
        assert_eq!(*reused.page(), temporary_page);
        assert_ne!(retained.page(), reused.page());
        let retained_page = *retained.page();
        drop(retained);
        assert_eq!(*allocate().page(), retained_page);
        assert_eq!(next_page.get(), 2);
        assert_eq!(pool.reserved_bytes(), 8);
        assert!(pool.allocate(16, |_| Err::<usize, _>(())).is_err());
        assert_eq!(pool.reserved_bytes(), 8);
    }
}
