//! Process-wide limits include JSON parsing, compilation and intermediate values.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use stark_resource_budget::Budget;

static MAX_CUMULATIVE_ALLOCATION: AtomicUsize = AtomicUsize::new(Budget::DEFAULT.template_cumulative_bytes);
static MAX_LIVE_ALLOCATION: AtomicUsize = AtomicUsize::new(Budget::DEFAULT.template_live_bytes);
static INPUT_BYTES: AtomicUsize = AtomicUsize::new(Budget::DEFAULT.input_bytes);
static WIRE_BYTES: AtomicUsize = AtomicUsize::new(Budget::DEFAULT.wire_bytes);
const CPU_SECONDS: libc::rlim_t = 2;

static CUMULATIVE: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);

pub fn input_bytes() -> usize {
    INPUT_BYTES.load(Ordering::Relaxed)
}

pub fn wire_bytes() -> usize {
    WIRE_BYTES.load(Ordering::Relaxed)
}

/// Configure the disposable helper before reading or compiling any model data.
pub fn configure() -> Result<(), &'static str> {
    let budget = Budget::from_environment()?;
    INPUT_BYTES.store(budget.input_bytes, Ordering::Relaxed);
    WIRE_BYTES.store(budget.wire_bytes, Ordering::Relaxed);
    MAX_CUMULATIVE_ALLOCATION.store(budget.template_cumulative_bytes, Ordering::Relaxed);
    MAX_LIVE_ALLOCATION.store(budget.template_live_bytes, Ordering::Relaxed);
    Ok(())
}

/// Ends a disposable helper without allocation, diagnostics, destructors or core dumps.
pub fn terminate() -> ! {
    unsafe {
        libc::write(libc::STDOUT_FILENO, b"E".as_ptr().cast(), 1);
        libc::_exit(1);
    }
}

fn charge(
    counter: &AtomicUsize,
    amount: usize,
    limit: usize,
) {
    if counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| next_charge(used, amount, limit)).is_err() {
        terminate();
    }
}

fn next_charge(
    used: usize,
    amount: usize,
    limit: usize,
) -> Option<usize> {
    used.checked_add(amount).filter(|next| *next <= limit)
}

pub struct BoundedAllocator;

unsafe impl GlobalAlloc for BoundedAllocator {
    unsafe fn alloc(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        charge(&CUMULATIVE, layout.size(), MAX_CUMULATIVE_ALLOCATION.load(Ordering::Relaxed));
        charge(&LIVE, layout.size(), MAX_LIVE_ALLOCATION.load(Ordering::Relaxed));
        let result = unsafe { System.alloc(layout) };
        if result.is_null() {
            terminate();
        }
        result
    }

    unsafe fn alloc_zeroed(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        charge(&CUMULATIVE, layout.size(), MAX_CUMULATIVE_ALLOCATION.load(Ordering::Relaxed));
        charge(&LIVE, layout.size(), MAX_LIVE_ALLOCATION.load(Ordering::Relaxed));
        let result = unsafe { System.alloc_zeroed(layout) };
        if result.is_null() {
            terminate();
        }
        result
    }

    unsafe fn dealloc(
        &self,
        pointer: *mut u8,
        layout: Layout,
    ) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(
        &self,
        pointer: *mut u8,
        layout: Layout,
        new_size: usize,
    ) -> *mut u8 {
        // Charge the entire replacement conservatively, including shrink/copy paths.
        charge(&CUMULATIVE, new_size, MAX_CUMULATIVE_ALLOCATION.load(Ordering::Relaxed));
        if new_size > layout.size() {
            charge(&LIVE, new_size - layout.size(), MAX_LIVE_ALLOCATION.load(Ordering::Relaxed));
        }
        let result = unsafe { System.realloc(pointer, layout, new_size) };
        if result.is_null() {
            terminate();
        }
        if new_size < layout.size() {
            LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        result
    }
}

pub fn install_process_limits() -> bool {
    for (resource, value) in [(libc::RLIMIT_CORE, 0), (libc::RLIMIT_CPU, CPU_SECONDS), (libc::RLIMIT_FSIZE, 0)] {
        let limit = libc::rlimit {
            rlim_cur: value,
            rlim_max: value,
        };
        if unsafe { libc::setrlimit(resource, &limit) } != 0 {
            return false;
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let limit = libc::rlimit {
            rlim_cur: MAX_LIVE_ALLOCATION.load(Ordering::Relaxed) as libc::rlim_t,
            rlim_max: MAX_LIVE_ALLOCATION.load(Ordering::Relaxed) as libc::rlim_t,
        };
        if unsafe { libc::setrlimit(libc::RLIMIT_AS, &limit) } != 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_charges_reject_aggregate_overflow_and_limit_breaches() {
        assert_eq!(next_charge(0, 16, 32), Some(16));
        assert_eq!(next_charge(16, 16, 32), Some(32));
        assert_eq!(next_charge(32, 1, 32), None);
        assert_eq!(next_charge(usize::MAX, 1, usize::MAX), None);
        assert_eq!(next_charge(1, usize::MAX, usize::MAX), None);
    }

    #[test]
    fn resource_limits_are_installed() {
        assert!(install_process_limits());
        for (resource, expected) in [(libc::RLIMIT_CORE, 0), (libc::RLIMIT_CPU, 2), (libc::RLIMIT_FSIZE, 0)] {
            let mut limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            assert_eq!(unsafe { libc::getrlimit(resource, &mut limit) }, 0);
            assert_eq!(limit.rlim_cur, expected);
            assert_eq!(limit.rlim_max, expected);
        }
    }
}
