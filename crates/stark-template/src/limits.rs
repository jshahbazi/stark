//! Process-wide limits include JSON parsing, compilation and intermediate values.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

pub const MAX_CUMULATIVE_ALLOCATION: usize = 16 * 1024 * 1024;
pub const MAX_LIVE_ALLOCATION: usize = 256 * 1024 * 1024;
const CPU_SECONDS: libc::rlim_t = 2;

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeLimits {
    pub request_bytes: usize,
    pub value_bytes: usize,
    pub output_bytes: usize,
    pub sequence_items: usize,
    pub value_nodes: usize,
    pub fuel: u64,
    pub cumulative_allocation_bytes: usize,
    pub live_allocation_bytes: usize,
    pub cpu_seconds: usize,
    pub wall_seconds: f64,
}

static REQUEST_LIMIT: AtomicUsize = AtomicUsize::new(2 * 1024 * 1024);
static VALUE_LIMIT: AtomicUsize = AtomicUsize::new(1024 * 1024);
static OUTPUT_LIMIT: AtomicUsize = AtomicUsize::new(1024 * 1024);
static ITEM_LIMIT: AtomicUsize = AtomicUsize::new(16384);
static NODE_LIMIT: AtomicUsize = AtomicUsize::new(65536);
static FUEL_LIMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1_000_000);
static CUMULATIVE_LIMIT: AtomicUsize = AtomicUsize::new(MAX_CUMULATIVE_ALLOCATION);
static LIVE_LIMIT: AtomicUsize = AtomicUsize::new(MAX_LIVE_ALLOCATION);
static CPU_LIMIT: AtomicUsize = AtomicUsize::new(CPU_SECONDS as usize);

pub fn request_bytes() -> usize {
    REQUEST_LIMIT.load(Ordering::Relaxed)
}
pub fn value_bytes() -> usize {
    VALUE_LIMIT.load(Ordering::Relaxed)
}
pub fn output_bytes() -> usize {
    OUTPUT_LIMIT.load(Ordering::Relaxed)
}
pub fn sequence_items() -> usize {
    ITEM_LIMIT.load(Ordering::Relaxed)
}
pub fn value_nodes() -> usize {
    NODE_LIMIT.load(Ordering::Relaxed)
}
pub fn fuel() -> u64 {
    FUEL_LIMIT.load(Ordering::Relaxed)
}

/// Called once by trusted process arguments, before reading untrusted JSON.
pub fn configure(value: RuntimeLimits) -> bool {
    let numbers = [
        value.request_bytes,
        value.value_bytes,
        value.output_bytes,
        value.sequence_items,
        value.value_nodes,
        value.live_allocation_bytes,
        value.cpu_seconds,
    ];
    if numbers.iter().any(|&n| n == 0 || n > isize::MAX as usize)
        || value.cumulative_allocation_bytes == 0
        || value.fuel == 0
        || !value.wall_seconds.is_finite()
        || value.wall_seconds <= 0.0
        || CUMULATIVE.load(Ordering::Relaxed) > value.cumulative_allocation_bytes
        || LIVE.load(Ordering::Relaxed) > value.live_allocation_bytes
    {
        return false;
    }
    REQUEST_LIMIT.store(value.request_bytes, Ordering::Relaxed);
    VALUE_LIMIT.store(value.value_bytes, Ordering::Relaxed);
    OUTPUT_LIMIT.store(value.output_bytes, Ordering::Relaxed);
    ITEM_LIMIT.store(value.sequence_items, Ordering::Relaxed);
    NODE_LIMIT.store(value.value_nodes, Ordering::Relaxed);
    FUEL_LIMIT.store(value.fuel, Ordering::Relaxed);
    CUMULATIVE_LIMIT.store(value.cumulative_allocation_bytes, Ordering::Relaxed);
    LIVE_LIMIT.store(value.live_allocation_bytes, Ordering::Relaxed);
    CPU_LIMIT.store(value.cpu_seconds, Ordering::Relaxed);
    true
}

/// Expanded trusted render budgets must not outlive their owning host.
pub fn guard_parent() -> bool {
    let parent = unsafe { libc::getppid() };
    if parent <= 1 {
        return false;
    }
    std::thread::Builder::new()
        .name("stark-template-owner".to_owned())
        .spawn(move || {
            loop {
                if unsafe { libc::getppid() } != parent {
                    terminate();
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        })
        .is_ok()
}

static CUMULATIVE: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);

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
        charge(&CUMULATIVE, layout.size(), CUMULATIVE_LIMIT.load(Ordering::Relaxed));
        charge(&LIVE, layout.size(), LIVE_LIMIT.load(Ordering::Relaxed));
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
        charge(&CUMULATIVE, layout.size(), CUMULATIVE_LIMIT.load(Ordering::Relaxed));
        charge(&LIVE, layout.size(), LIVE_LIMIT.load(Ordering::Relaxed));
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
        charge(&CUMULATIVE, new_size, CUMULATIVE_LIMIT.load(Ordering::Relaxed));
        if new_size > layout.size() {
            charge(&LIVE, new_size - layout.size(), LIVE_LIMIT.load(Ordering::Relaxed));
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
    for (resource, value) in [
        (libc::RLIMIT_CORE, 0),
        (libc::RLIMIT_CPU, CPU_LIMIT.load(Ordering::Relaxed) as libc::rlim_t),
        (libc::RLIMIT_FSIZE, 0),
    ] {
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
            rlim_cur: LIVE_LIMIT.load(Ordering::Relaxed) as libc::rlim_t,
            rlim_max: LIVE_LIMIT.load(Ordering::Relaxed) as libc::rlim_t,
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
        assert_eq!(next_charge(isize::MAX as usize, 1, usize::MAX), Some(isize::MAX as usize + 1));
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
