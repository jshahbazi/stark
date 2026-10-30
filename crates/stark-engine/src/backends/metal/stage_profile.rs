//! Best-effort, bounded, opt-in attribution of single-token GPU passes.
//! Precise timestamps perturb execution; these are diagnostic timings only.

use std::{
    fs::{File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use metal::{
    MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapExt, MTL4CounterHeapType,
    MTL4TimestampGranularity, MTLDevice, MTLDeviceExt,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::StageProfileCategory;

const MAX_PASSES: usize = 4;
// Admission allows at most 256 layers: seven checkpoints per layer plus
// embedding, final HC, head and sampling fit without a model-sized allocation.
const MAX_MARKERS: usize = 2048;
const LABELS: [&str; 10] =
    ["embedding", "hc_read", "hc_inject", "gdn", "qsa", "moe", "ple", "final_hc", "head", "sampling"];

struct Sink {
    file: Mutex<File>,
    attempts: AtomicUsize,
    disabled: AtomicBool,
}

fn sink() -> Option<&'static Arc<Sink>> {
    static SINK: OnceLock<Option<Arc<Sink>>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = std::env::var_os("STARK_UZU_STAGE_PROFILE_OUTPUT")?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .ok()?;
        Some(Arc::new(Sink {
            file: Mutex::new(file),
            attempts: AtomicUsize::new(0),
            disabled: AtomicBool::new(false),
        }))
    })
    .as_ref()
}

struct Heap(Retained<ProtocolObject<dyn MTL4CounterHeap>>);

// SAFETY: a Metal counter heap is a device-owned resource with no thread
// affinity. Encoding mutates its entries only through ordered GPU commands;
// CPU resolution occurs only after the matching completion. This wrapper is
// immutable after submission and merely retains the resource across that
// completion, including an abandoned Pending. The binding omits Send/Sync.
unsafe impl Send for Heap {}
// SAFETY: shared owners only retain this immutable handle; the sole CPU
// resolver runs after GPU completion, as described above.
unsafe impl Sync for Heap {}

pub(super) struct StageProfile {
    heap: Heap,
    sink: Arc<Sink>,
    pass: usize,
    frequency: u64,
    categories: Vec<StageProfileCategory>,
    cpu_seconds: Vec<f64>,
    last_cpu: Instant,
    valid: bool,
}

struct Totals {
    calls: [u64; LABELS.len()],
    ticks: [u64; LABELS.len()],
    cpu: [f64; LABELS.len()],
}

fn aggregate(
    raw: &[u8],
    categories: &[StageProfileCategory],
    cpu_seconds: &[f64],
    frequency: u64,
) -> Option<Totals> {
    if frequency == 0 || raw.len() != (categories.len() + 1) * 8 || cpu_seconds.len() != categories.len() {
        return None;
    }
    let mut totals = Totals {
        calls: [0; LABELS.len()],
        ticks: [0; LABELS.len()],
        cpu: [0.0; LABELS.len()],
    };
    let read = |index: usize| u64::from_ne_bytes(raw[index * 8..index * 8 + 8].try_into().unwrap());
    let mut previous = read(0);
    if previous == 0 {
        return None;
    }
    for (index, category) in categories.iter().enumerate() {
        let next = read(index + 1);
        let elapsed = next.checked_sub(previous)?;
        previous = next;
        let cpu = cpu_seconds[index];
        if !cpu.is_finite() || cpu < 0.0 {
            return None;
        }
        let category = *category as usize;
        totals.calls[category] += 1;
        totals.ticks[category] = totals.ticks[category].checked_add(elapsed)?;
        totals.cpu[category] += cpu;
    }
    Some(totals)
}

impl StageProfile {
    pub(super) fn begin(
        device: &ProtocolObject<dyn MTLDevice>,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) -> Option<Self> {
        let sink = sink()?;
        if sink.disabled.load(Ordering::Relaxed) {
            return None;
        }
        let mut attempts = sink.attempts.load(Ordering::Relaxed);
        let pass = loop {
            if attempts >= MAX_PASSES {
                return None;
            }
            match sink.attempts.compare_exchange_weak(attempts, attempts + 1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break attempts + 1,
                Err(current) => attempts = current,
            }
        };
        if device.size_of_counter_heap_entry(MTL4CounterHeapType::TIMESTAMP) != 8 {
            sink.disabled.store(true, Ordering::Relaxed);
            return None;
        }
        let frequency = device.query_timestamp_frequency();
        if frequency == 0 {
            sink.disabled.store(true, Ordering::Relaxed);
            return None;
        }
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.set_type(MTL4CounterHeapType::TIMESTAMP);
        descriptor.set_count(MAX_MARKERS);
        let heap = match device.new_counter_heap_with_descriptor(&descriptor) {
            Ok(heap) => heap,
            Err(_) => {
                sink.disabled.store(true, Ordering::Relaxed);
                return None;
            },
        };
        heap.set_label(Some("stark-fixed-stage-profile"));
        encoder.write_timestamp_with_granularity_into_heap_at_index(MTL4TimestampGranularity::PRECISE, &heap, 0);
        Some(Self {
            heap: Heap(heap),
            sink: sink.clone(),
            pass,
            frequency,
            categories: Vec::with_capacity(MAX_MARKERS - 1),
            cpu_seconds: Vec::with_capacity(MAX_MARKERS - 1),
            last_cpu: Instant::now(),
            valid: true,
        })
    }

    pub(super) fn checkpoint(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        category: StageProfileCategory,
    ) {
        if !self.valid {
            return;
        }
        let index = self.categories.len() + 1;
        if index >= MAX_MARKERS {
            self.valid = false;
            self.sink.disabled.store(true, Ordering::Relaxed);
            return;
        }
        let now = Instant::now();
        self.cpu_seconds.push(now.duration_since(self.last_cpu).as_secs_f64());
        self.categories.push(category);
        encoder.write_timestamp_with_granularity_into_heap_at_index(
            MTL4TimestampGranularity::PRECISE,
            &self.heap.0,
            index,
        );
        self.last_cpu = Instant::now();
    }

    /// The caller has completed the matching normal GPU wait. Never invoked
    /// from a callback or while any producer can still write these entries.
    pub(super) fn complete(
        &self,
        command_gpu_time: Duration,
    ) {
        if !self.valid || self.categories.is_empty() || self.sink.disabled.load(Ordering::Relaxed) {
            return;
        }
        if self.write_aggregate(command_gpu_time).is_none() {
            self.sink.disabled.store(true, Ordering::Relaxed);
        }
    }

    fn write_aggregate(
        &self,
        command_gpu_time: Duration,
    ) -> Option<()> {
        let count = self.categories.len() + 1;
        let raw = self.heap.0.resolve_counter_range(0..count)?;
        let Totals {
            calls,
            ticks,
            cpu,
        } = aggregate(&raw, &self.categories, &self.cpu_seconds, self.frequency)?;
        let stages = LABELS
            .iter()
            .enumerate()
            .filter(|(index, _)| calls[*index] > 0)
            .map(|(index, label)| {
                serde_json::json!({"stage":label,"calls":calls[index],
                "gpu_seconds":ticks[index] as f64 / self.frequency as f64,"cpu_encode_seconds":cpu[index]})
            })
            .collect::<Vec<_>>();
        let record = serde_json::json!({"schema":"stark-uzu-stage-profile-v1","pass":self.pass,
            "single_token_pass":true,"timestamp_granularity":"precise","marker_count":count,
            "gpu_command_seconds":command_gpu_time.as_secs_f64(),"timestamp_frequency_hz":self.frequency,
            "cpu_encode_seconds":cpu.iter().sum::<f64>(),"stages":stages});
        let mut encoded = serde_json::to_vec(&record).ok()?;
        encoded.push(b'\n');
        let mut file = self.sink.file.lock().ok()?;
        file.write_all(&encoded).ok()?;
        file.flush().ok()?;
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn bytes(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    #[stark_test]
    fn fixed_stage_aggregation_keeps_categories_and_rejects_invalid_samples() {
        let categories = [StageProfileCategory::HcRead, StageProfileCategory::Moe, StageProfileCategory::HcRead];
        let totals = aggregate(&bytes(&[100, 140, 145, 170]), &categories, &[0.1, 0.2, 0.3], 1_000).unwrap();
        let hc = StageProfileCategory::HcRead as usize;
        let moe = StageProfileCategory::Moe as usize;
        assert!(totals.calls[hc] == 2 && totals.ticks[hc] == 65 && (totals.cpu[hc] - 0.4).abs() < f64::EPSILON);
        assert!(totals.calls[moe] == 1 && totals.ticks[moe] == 5 && totals.cpu[moe] == 0.2);
        assert!(totals.calls.iter().sum::<u64>() == 3 && totals.ticks.iter().sum::<u64>() == 70);
        for raw in [bytes(&[100, 140]), bytes(&[0, 140, 145, 170]), bytes(&[100, 140, 139, 170])] {
            assert!(aggregate(&raw, &categories, &[0.1, 0.2, 0.3], 1_000).is_none());
        }
        let valid = bytes(&[100, 140, 145, 170]);
        assert!(aggregate(&valid, &categories, &[0.1, 0.2, 0.3], 0).is_none());
        assert!(aggregate(&valid, &categories, &[0.1, 0.2], 1_000).is_none());
        assert!(aggregate(&valid, &categories, &[0.1, f64::NAN, 0.3], 1_000).is_none());
    }
}
