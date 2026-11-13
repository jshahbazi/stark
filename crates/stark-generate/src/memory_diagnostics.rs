//! Explicit, model-free aggregate diagnostics for the worker's own task.
//!
//! ABI provenance: Apple's macOS 26.4 SDK `usr/include/mach/task_info.h`
//! (`#pragma pack(push, 4)`), `mach/task.h`, and `mach/mach_init.h`, checked
//! with Clang's record-layout dump. TASK_VM_INFO rev5 is 87 natural_t words;
//! decompressions is a signed integer_t at byte 344. The full SDK rev7 is
//! 372 bytes, alignment 4. Requesting only rev5 avoids trailing ABI drift.
//! TASK_EVENTS_INFO is eight integer_t words, with pageins at byte 4.
//!
//! No task port supplied by a caller is accepted. No addresses, identifiers,
//! model state, text, or raw kernel errors are serialized. Sampling performs
//! two task_info calls only when the caller explicitly invokes `snapshot`.

use serde::Serialize;

const VM_WORDS: usize = 87;
const VM_BASE_WORDS: usize = 38;
const EVENTS_WORDS: usize = 8;

/// Fixed diagnostic codes; native failures never become exception dumps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotStatus {
    Ok,
    #[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
    UnsupportedPlatform,
    TaskInfoFailed,
    ShortReply,
    InvalidReplyCount,
    InvalidPageSize,
    CounterOutOfRange,
    CounterSaturated,
    CounterRegressed,
}

/// Instantaneous byte gauges and cumulative, nonnegative task counters.
///
/// `compressed_bytes` is the task's internal-compressed ledger balance, not
/// physical compressor storage. Decompressions count execution charged to
/// this task's threads, not all decompression of model-owned or GPU pages.
/// `pageins_total` is the task-events actual-pagein count. These sources do
/// not establish complete model memory attribution or whole-system churn.
///
/// Byte gauges can decrease. Cumulative counters that reach the signed SDK
/// ceiling, become negative, or regress are unavailable for the remainder of
/// this sampler's lifetime. They must never establish a zero-churn interval.
#[derive(Debug, Serialize)]
pub struct MemorySnapshot {
    pub supported: bool,
    pub source: &'static str,
    pub vm_status: SnapshotStatus,
    pub decompressions_status: SnapshotStatus,
    pub pageins_status: SnapshotStatus,
    pub vm_reply_words: Option<u32>,
    pub events_reply_words: Option<u32>,
    pub resident_bytes: Option<u64>,
    pub compressed_bytes: Option<u64>,
    pub physical_footprint_bytes: Option<u64>,
    pub virtual_bytes: Option<u64>,
    pub page_size_bytes: Option<u64>,
    pub decompressions_total_pages: Option<u64>,
    pub pageins_total: Option<u64>,
}

impl MemorySnapshot {
    fn unavailable(
        supported: bool,
        status: SnapshotStatus,
    ) -> Self {
        Self {
            supported,
            source: "darwin_task_info_self",
            vm_status: status,
            decompressions_status: status,
            pageins_status: status,
            vm_reply_words: None,
            events_reply_words: None,
            resident_bytes: None,
            compressed_bytes: None,
            physical_footprint_bytes: None,
            virtual_bytes: None,
            page_size_bytes: None,
            decompressions_total_pages: None,
            pageins_total: None,
        }
    }
}

#[derive(Default)]
struct Counter {
    last: Option<u64>,
    failure: Option<SnapshotStatus>,
}

impl Counter {
    fn observe(
        &mut self,
        raw: i32,
    ) -> (Option<u64>, SnapshotStatus) {
        if let Some(failure) = self.failure {
            return (None, failure);
        }
        let failure = if raw < 0 {
            Some(SnapshotStatus::CounterOutOfRange)
        } else if raw == i32::MAX {
            // XNU caps task_info's signed event/decompression counters here.
            Some(SnapshotStatus::CounterSaturated)
        } else if self.last.is_some_and(|last| (raw as u64) < last) {
            Some(SnapshotStatus::CounterRegressed)
        } else {
            None
        };
        if let Some(failure) = failure {
            self.failure = Some(failure);
            return (None, failure);
        }
        let value = raw as u64;
        self.last = Some(value);
        (Some(value), SnapshotStatus::Ok)
    }
}

/// Keep one sampler for the worker lifetime to detect cumulative regressions.
#[derive(Default)]
pub struct MemoryDiagnostics {
    decompressions: Counter,
    pageins: Counter,
}

impl MemoryDiagnostics {
    pub fn snapshot(&mut self) -> MemorySnapshot {
        #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
        {
            self.observe_replies(darwin::vm_info(), darwin::events_info())
        }
        #[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
        {
            MemorySnapshot::unavailable(false, SnapshotStatus::UnsupportedPlatform)
        }
    }

    fn observe_replies(
        &mut self,
        vm: Result<Reply<VM_WORDS>, SnapshotStatus>,
        events: Result<Reply<EVENTS_WORDS>, SnapshotStatus>,
    ) -> MemorySnapshot {
        let mut result = MemorySnapshot::unavailable(true, SnapshotStatus::TaskInfoFailed);
        result.vm_reply_words = vm.as_ref().ok().map(|reply| reply.count);
        result.events_reply_words = events.as_ref().ok().map(|reply| reply.count);
        match vm.and_then(|reply| reply.require(VM_BASE_WORDS)) {
            Ok(vm) => {
                result.vm_status = SnapshotStatus::Ok;
                result.virtual_bytes = Some(vm.u64_at(0));
                result.resident_bytes = Some(vm.u64_at(16));
                result.compressed_bytes = Some(vm.u64_at(120));
                result.physical_footprint_bytes = Some(vm.u64_at(144));
                let page_size = vm.i32_at(12);
                if page_size > 0 && (page_size as u32).is_power_of_two() {
                    result.page_size_bytes = Some(page_size as u64);
                } else {
                    result.vm_status = SnapshotStatus::InvalidPageSize;
                }
                match vm.require(VM_WORDS) {
                    Ok(vm) => {
                        (result.decompressions_total_pages, result.decompressions_status) =
                            self.decompressions.observe(vm.i32_at(344));
                    },
                    Err(status) => result.decompressions_status = status,
                }
            },
            Err(status) => {
                result.vm_status = status;
                result.decompressions_status = status;
            },
        }
        match events.and_then(|reply| reply.require(EVENTS_WORDS)) {
            Ok(events) => {
                (result.pageins_total, result.pageins_status) = self.pageins.observe(events.i32_at(4));
            },
            Err(status) => result.pageins_status = status,
        }
        result
    }
}

// The kernel's packed ABI is an array of four-byte natural_t words. Decoding
// words avoids creating references to unaligned u64 fields in a packed struct.
struct Reply<const N: usize> {
    words: [u32; N],
    count: u32,
}

impl<const N: usize> Reply<N> {
    fn require(
        self,
        minimum: usize,
    ) -> Result<Self, SnapshotStatus> {
        if self.count as usize > N {
            Err(SnapshotStatus::InvalidReplyCount)
        } else if (self.count as usize) < minimum {
            Err(SnapshotStatus::ShortReply)
        } else {
            Ok(self)
        }
    }

    fn i32_at(
        &self,
        byte_offset: usize,
    ) -> i32 {
        i32::from_ne_bytes(self.words[byte_offset / 4].to_ne_bytes())
    }

    fn u64_at(
        &self,
        byte_offset: usize,
    ) -> u64 {
        let index = byte_offset / 4;
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&self.words[index].to_ne_bytes());
        bytes[4..].copy_from_slice(&self.words[index + 1].to_ne_bytes());
        u64::from_ne_bytes(bytes)
    }
}

#[cfg(all(target_os = "macos", target_pointer_width = "64"))]
mod darwin {
    use super::{EVENTS_WORDS, Reply, SnapshotStatus, VM_WORDS};

    const TASK_EVENTS_INFO: libc::task_flavor_t = 2;
    const TASK_VM_INFO: libc::task_flavor_t = 22;

    // Keep the locked libc SDK declaration; no new Mach dependency is needed.
    #[allow(deprecated)]
    fn read_self<const N: usize>(flavor: libc::task_flavor_t) -> Result<Reply<N>, SnapshotStatus> {
        let mut reply = Reply {
            words: [0; N],
            count: N as u32,
        };
        // SAFETY: this is the current task's port, not a foreign task port.
        // The writable buffer contains exactly `count` four-byte words, and
        // remains alive for the synchronous call. No uninitialized bytes are
        // inspected. Returned counts are validated before reading any field.
        let status = unsafe {
            libc::task_info(libc::mach_task_self(), flavor, reply.words.as_mut_ptr().cast(), &mut reply.count)
        };
        if status != libc::KERN_SUCCESS {
            return Err(SnapshotStatus::TaskInfoFailed);
        }
        reply.require(0)
    }

    pub(super) fn vm_info() -> Result<Reply<VM_WORDS>, SnapshotStatus> {
        read_self(TASK_VM_INFO)
    }

    pub(super) fn events_info() -> Result<Reply<EVENTS_WORDS>, SnapshotStatus> {
        read_self(TASK_EVENTS_INFO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
    #[ignore = "requires a local Darwin kernel supporting task VM rev5"]
    fn live_self_task_contract_has_available_monotonic_counters() {
        let mut sampler = MemoryDiagnostics::default();
        let before = sampler.snapshot();
        let after = sampler.snapshot();
        for value in [&before, &after] {
            assert!(value.supported);
            assert_eq!(value.vm_status, SnapshotStatus::Ok);
            assert_eq!(value.decompressions_status, SnapshotStatus::Ok);
            assert_eq!(value.pageins_status, SnapshotStatus::Ok);
            assert_eq!(value.vm_reply_words, Some(87));
            assert_eq!(value.events_reply_words, Some(8));
            assert!(value.page_size_bytes.is_some_and(u64::is_power_of_two));
            assert!(value.resident_bytes.is_some());
            assert!(value.compressed_bytes.is_some());
            assert!(value.physical_footprint_bytes.is_some());
        }
        assert!(after.decompressions_total_pages.unwrap() >= before.decompressions_total_pages.unwrap());
        assert!(after.pageins_total.unwrap() >= before.pageins_total.unwrap());
    }

    fn vm_reply(
        count: u32,
        decompressions: i32,
    ) -> Reply<VM_WORDS> {
        let mut words = [0; VM_WORDS];
        // SDK byte offsets, with distinct 64-bit upper words to catch narrow
        // reads, field swaps, and accidental byte/word unit conversions.
        for (offset, value) in
            [(0, 0x1234_0000_0001_u64), (16, 0x2345_0000_0002), (120, 0x3456_0000_0003), (144, 0x4567_0000_0004)]
        {
            let bytes = value.to_ne_bytes();
            words[offset / 4] = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
            words[offset / 4 + 1] = u32::from_ne_bytes(bytes[4..].try_into().unwrap());
        }
        words[3] = 16384;
        words[86] = u32::from_ne_bytes(decompressions.to_ne_bytes());
        Reply {
            words,
            count,
        }
    }

    fn events_reply(pageins: i32) -> Reply<EVENTS_WORDS> {
        let mut words = [0; EVENTS_WORDS];
        words[1] = u32::from_ne_bytes(pageins.to_ne_bytes());
        Reply {
            words,
            count: EVENTS_WORDS as u32,
        }
    }

    #[test]
    fn sdk_revision_boundaries_do_not_invent_missing_counters() {
        for count in [0, 36, 37] {
            let snapshot = MemoryDiagnostics::default().observe_replies(Ok(vm_reply(count, 500)), Ok(events_reply(7)));
            assert_eq!(snapshot.vm_status, SnapshotStatus::ShortReply);
            assert_eq!(snapshot.physical_footprint_bytes, None);
            assert_eq!(snapshot.decompressions_total_pages, None);
            assert_eq!(snapshot.pageins_total, Some(7));
        }
        for count in [38, 42, 84, 86] {
            let snapshot = MemoryDiagnostics::default().observe_replies(Ok(vm_reply(count, 500)), Ok(events_reply(7)));
            assert_eq!(snapshot.physical_footprint_bytes, Some(0x4567_0000_0004));
            assert_eq!(snapshot.decompressions_status, SnapshotStatus::ShortReply);
            assert_eq!(snapshot.decompressions_total_pages, None);
        }
        let snapshot = MemoryDiagnostics::default().observe_replies(Ok(vm_reply(87, 500)), Ok(events_reply(7)));
        assert_eq!(snapshot.decompressions_total_pages, Some(500));
        assert_eq!(snapshot.virtual_bytes, Some(0x1234_0000_0001));
        assert_eq!(snapshot.resident_bytes, Some(0x2345_0000_0002));
        assert_eq!(snapshot.compressed_bytes, Some(0x3456_0000_0003));
        assert_eq!(snapshot.page_size_bytes, Some(16384));
    }

    #[test]
    fn malformed_counts_and_failures_do_not_look_like_zero_churn() {
        let snapshot = MemoryDiagnostics::default().observe_replies(
            Ok(vm_reply(88, 0)),
            Ok(Reply {
                words: [0; EVENTS_WORDS],
                count: 7,
            }),
        );
        assert_eq!(snapshot.vm_status, SnapshotStatus::InvalidReplyCount);
        assert_eq!(snapshot.decompressions_total_pages, None);
        assert_eq!(snapshot.pageins_status, SnapshotStatus::ShortReply);
        assert_eq!(snapshot.pageins_total, None);
        let snapshot = MemoryDiagnostics::default()
            .observe_replies(Err(SnapshotStatus::TaskInfoFailed), Err(SnapshotStatus::TaskInfoFailed));
        let json = serde_json::to_value(snapshot).unwrap();
        assert_eq!(json["decompressions_total_pages"], serde_json::Value::Null);
        assert_eq!(json["vm_status"], "task_info_failed");
    }

    #[test]
    fn invalid_or_saturated_counters_stay_unavailable_after_recovery() {
        for (invalid, status) in [(-1, SnapshotStatus::CounterOutOfRange), (i32::MAX, SnapshotStatus::CounterSaturated)]
        {
            let mut sampler = MemoryDiagnostics::default();
            for value in [invalid, 1, 50] {
                let snapshot = sampler.observe_replies(Ok(vm_reply(87, value)), Ok(events_reply(value)));
                assert_eq!(snapshot.decompressions_status, status);
                assert_eq!(snapshot.pageins_status, status);
                assert_eq!(snapshot.decompressions_total_pages, None);
                assert_eq!(snapshot.pageins_total, None);
            }
        }
    }

    #[test]
    fn cumulative_regressions_latch_while_byte_gauges_may_decrease() {
        let mut sampler = MemoryDiagnostics::default();
        for value in [0, 2, 2, 500] {
            let snapshot = sampler.observe_replies(Ok(vm_reply(87, value)), Ok(events_reply(value)));
            assert_eq!(snapshot.decompressions_total_pages, Some(value as u64));
        }
        for value in [499, 600] {
            let mut vm = vm_reply(87, value);
            vm.words[4] = 0;
            vm.words[5] = 0;
            let snapshot = sampler.observe_replies(Ok(vm), Ok(events_reply(value)));
            assert_eq!(snapshot.decompressions_status, SnapshotStatus::CounterRegressed);
            assert_eq!(snapshot.pageins_status, SnapshotStatus::CounterRegressed);
            assert_eq!(snapshot.decompressions_total_pages, None);
            assert_eq!(snapshot.resident_bytes, Some(0));
        }
    }

    #[test]
    fn unavailable_reply_preserves_counter_history() {
        let mut sampler = MemoryDiagnostics::default();
        let baseline = sampler.observe_replies(Ok(vm_reply(87, 10)), Ok(events_reply(10)));
        assert_eq!(baseline.decompressions_total_pages, Some(10));
        let unavailable =
            sampler.observe_replies(Err(SnapshotStatus::TaskInfoFailed), Err(SnapshotStatus::TaskInfoFailed));
        assert_eq!(unavailable.decompressions_total_pages, None);
        let regressed = sampler.observe_replies(Ok(vm_reply(87, 9)), Ok(events_reply(9)));
        assert_eq!(regressed.decompressions_status, SnapshotStatus::CounterRegressed);
        assert_eq!(regressed.pageins_status, SnapshotStatus::CounterRegressed);
    }

    #[test]
    fn invalid_page_size_is_explicit_and_output_contains_only_aggregate_fields() {
        let mut vm = vm_reply(87, 23);
        vm.words[3] = 0;
        let snapshot = MemoryDiagnostics::default().observe_replies(Ok(vm), Ok(events_reply(7)));
        assert_eq!(snapshot.vm_status, SnapshotStatus::InvalidPageSize);
        assert_eq!(snapshot.page_size_bytes, None);
        let json = serde_json::to_value(snapshot).unwrap();
        let mut keys: Vec<_> = json.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "compressed_bytes",
                "decompressions_status",
                "decompressions_total_pages",
                "events_reply_words",
                "page_size_bytes",
                "pageins_status",
                "pageins_total",
                "physical_footprint_bytes",
                "resident_bytes",
                "source",
                "supported",
                "virtual_bytes",
                "vm_reply_words",
                "vm_status",
            ]
        );
        assert_eq!(json["decompressions_total_pages"], 23);
    }

    #[test]
    #[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
    fn unsupported_platform_is_explicit() {
        let snapshot = MemoryDiagnostics::default().snapshot();
        assert!(!snapshot.supported);
        assert_eq!(snapshot.vm_status, SnapshotStatus::UnsupportedPlatform);
        assert!(snapshot.decompressions_total_pages.is_none());
    }
}
