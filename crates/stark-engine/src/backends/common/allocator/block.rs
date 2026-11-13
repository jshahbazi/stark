use std::{collections::HashMap, mem::ManuallyDrop, range::Range, sync::Arc};

use parking_lot::Mutex;

// TODO: this is a stub, real block allocator should suballocate from large pages

pub struct BlockAllocator<T> {
    min_page_size: usize,
    cache: Mutex<HashMap<usize, Vec<T>>>,
}

impl<T> BlockAllocator<T> {
    /// Bytes held by completely free cached pages. Live allocations are never
    /// present here, including pages still held by deferred GPU retirement.
    pub fn cached_bytes(&self) -> Option<usize> {
        self.cache
            .lock()
            .iter()
            .try_fold(0usize, |bytes, (&size, pages)| bytes.checked_add(size.checked_mul(pages.len())?))
    }

    /// Release free owners to the upstream allocator. Taking the map before
    /// dropping keeps resource destructors outside the allocation lock.
    pub fn trim_cached(&self) -> Option<usize> {
        let pages = std::mem::take(&mut *self.cache.lock());
        let bytes =
            pages.iter().try_fold(0usize, |bytes, (&size, pages)| bytes.checked_add(size.checked_mul(pages.len())?));
        drop(pages);
        bytes
    }

    pub fn new(min_page_size: usize) -> Arc<Self> {
        Arc::new(Self {
            min_page_size,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn allocate<E>(
        self: &Arc<Self>,
        size: usize,
        upstream: impl FnOnce(usize) -> Result<T, E>,
    ) -> Result<BlockAllocation<T>, E> {
        assert!(size > 0);

        let page = if let mut cache_locked = self.cache.lock()
            && let Some(cache_entries) = cache_locked.get_mut(&size.max(self.min_page_size))
            && let Some(cache_entry) = cache_entries.pop()
        {
            cache_entry
        } else {
            upstream(size.max(self.min_page_size))?
        };

        Ok(BlockAllocation {
            allocator: self.clone(),
            page: ManuallyDrop::new(page),
            range: (0..size).into(),
        })
    }

    fn free(
        self: &Arc<Self>,
        page: T,
        range: Range<usize>,
    ) {
        self.cache.lock().entry(range.iter().len().max(self.min_page_size)).or_default().push(page);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use stark_engine_macros::stark_test;

    use super::BlockAllocator;

    struct Page(Arc<AtomicUsize>);
    impl Drop for Page {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[stark_test]
    fn trimming_releases_only_free_cached_owners_and_preserves_live_allocations() {
        let drops = Arc::new(AtomicUsize::new(0));
        let allocator = BlockAllocator::new(16);
        let first = allocator.allocate(4, |_| Ok::<_, ()>(Page(drops.clone()))).unwrap();
        let live = allocator.allocate(32, |_| Ok::<_, ()>(Page(drops.clone()))).unwrap();
        assert_eq!(allocator.cached_bytes(), Some(0));
        drop(first);
        assert_eq!(allocator.cached_bytes(), Some(16));
        assert_eq!(allocator.trim_cached(), Some(16));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(live.range().iter().len(), 32);
        assert_eq!(allocator.cached_bytes(), Some(0));
        drop(live);
        assert_eq!(allocator.cached_bytes(), Some(32));
        assert_eq!(allocator.trim_cached(), Some(32));
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(allocator.trim_cached(), Some(0));
    }
}

pub struct BlockAllocation<T> {
    allocator: Arc<BlockAllocator<T>>,
    page: ManuallyDrop<T>,
    range: Range<usize>,
}

impl<T> BlockAllocation<T> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl<T> Drop for BlockAllocation<T> {
    fn drop(&mut self) {
        self.allocator.free(unsafe { ManuallyDrop::take(&mut self.page) }, self.range);
    }
}
