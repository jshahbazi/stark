// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Diagnostics behind the per-request timings: where a prefill's wall time
//! went, how the paged n-gram table's gathers fared, and what the system's
//! virtual memory did meanwhile.
//!
//! The engine counters are cumulative and per thread: the code that does the
//! work adds to them ([`record`]), and the server reads them before and after
//! a phase and subtracts ([`Counters::since`]). Only the engine thread runs a
//! request, so its counters are exactly the request's, with no locking and no
//! plumbing through the model API. Updating them costs a thread-local read
//! and write.
//!
//! The system side ([`vm_counters`], [`pressure_level`], [`task_memory`]) is
//! a handful of Mach and sysctl calls, a few microseconds each, sampled at
//! phase boundaries only. The dashboard's machine figures ([`cpu_ticks`],
//! [`memory_use`], [`swap_used`]) are more of the same, read on the HTTP
//! thread that answers `/v1/lily/status` and on the dashboard recorder's
//! thread, which samples them (with [`pressure_level`] and [`task_memory`])
//! for the 7-day history (`serve::recorder`). They are declared here rather than
//! taken from the `libc` crate, whose Mach bindings are deprecated.
//! [`own_energy_nj`] is lily's own CPU energy, which the machine sampler
//! (`serve::machine`) turns into lily's CPU power; it needs a newer
//! `rusage_info` revision than `libc` declares.

use std::cell::Cell;

/// What the paged n-gram table's gathers did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gather {
    /// Staged batches (one per decode step, verify step or prefill chunk).
    pub batches: u64,
    pub rows: u64,
    /// Rows whose pages were checked with `mincore` right before the copy:
    /// all of them in decode and verify batches, every 16th in prefill
    /// batches, where checking every page would cost more than the copy.
    pub checked_rows: u64,
    /// Pages the checked rows' bytes lie on, counted per row (a row spans
    /// three to six pages across its codes, scales and biases; a page two
    /// rows share counts twice).
    pub pages: u64,
    /// Of those, the ones that were not resident: each is a read from the
    /// SSD that the copy then waits for.
    pub cold_pages: u64,
    /// Prefill rows outside the 1-in-16 sample whose pages were checked and
    /// hinted as well, because the sample found their batch cold (at least
    /// 1 page in 32); their pages are not in `pages` or `cold_pages`.
    pub hinted_rows: u64,
    /// Wall time inside `PagedTable::gather`, residency checks included.
    pub secs: f64,
    /// Of the prefill's staging time (hashing plus the gathers), the part
    /// that ran while the GPU was still busy with the previous chunk, so it
    /// cost the prefill nothing; the rest is `Prefill::ngram_secs`.
    pub hidden_secs: f64,
}

/// Where the prefill loop's time went, summed over its chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prefill {
    pub chunks: u64,
    /// Growing the state's caches and the prefill scratch (allocation and
    /// zeroing, plus the copy of the old caches when the state grows).
    pub alloc_secs: f64,
    /// Hashing the chunks' n-gram ids and staging their rows, as far as the
    /// GPU waited for it: the first chunk's whole staging, and whatever of a
    /// later chunk's (staged while the GPU ran the chunk before it) outlasted
    /// that chunk. The overlapped rest is `Gather::hidden_secs`.
    pub ngram_secs: f64,
    /// Encoding the chunk's pass on the host.
    pub encode_secs: f64,
    /// The passes' GPU execution, from the commit feedback's timestamps.
    pub gpu_secs: f64,
    /// Commit to completion minus the GPU span: submission latency, the GPU
    /// busy with other work, residency being established. A pass without
    /// usable timestamps counts here whole.
    pub wait_secs: f64,
}

/// The engine thread's cumulative counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counters {
    pub gather: Gather,
    pub prefill: Prefill,
}

const ZERO: Counters = Counters {
    gather: Gather {
        batches: 0,
        rows: 0,
        checked_rows: 0,
        pages: 0,
        cold_pages: 0,
        hinted_rows: 0,
        secs: 0.0,
        hidden_secs: 0.0,
    },
    prefill: Prefill {
        chunks: 0,
        alloc_secs: 0.0,
        ngram_secs: 0.0,
        encode_secs: 0.0,
        gpu_secs: 0.0,
        wait_secs: 0.0,
    },
};

thread_local! {
    static COUNTERS: Cell<Counters> = const { Cell::new(ZERO) };
}

/// Adds to this thread's counters.
pub fn record(f: impl FnOnce(&mut Counters)) {
    COUNTERS.with(|cell| {
        let mut counters = cell.get();
        f(&mut counters);
        cell.set(counters);
    });
}
