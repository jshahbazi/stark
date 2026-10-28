//! Opt-in aggregate profiling for native execution. No model data is recorded.
//! Names are fixed call-site categories; nested scopes report inclusive time.

use std::{
    collections::BTreeMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use serde::Serialize;

static ENABLED: AtomicBool = AtomicBool::new(false);
static COUNTERS: OnceLock<Mutex<BTreeMap<&'static str, Counter>>> = OnceLock::new();

#[derive(Clone, Default, Serialize)]
pub struct Counter {
    pub calls: u64,
    pub seconds: f64,
    pub bytes: u64,
    pub max_bytes: u64,
}

/// Profiling is process-local and disabled by default. Enable only while the
/// caller owns the serialized native model execution interval.
pub fn enable(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn reset() {
    COUNTERS.get_or_init(Mutex::default).lock().unwrap().clear();
}

pub fn snapshot() -> BTreeMap<&'static str, Counter> {
    COUNTERS.get_or_init(Mutex::default).lock().unwrap().clone()
}

pub fn record(
    name: &'static str,
    seconds: f64,
    bytes: usize,
) {
    if !ENABLED.load(Ordering::Relaxed) || !seconds.is_finite() || seconds < 0.0 {
        return;
    }
    let mut counters = COUNTERS.get_or_init(Mutex::default).lock().unwrap();
    let counter = counters.entry(name).or_default();
    counter.calls += 1;
    counter.seconds += seconds;
    counter.bytes = counter.bytes.saturating_add(bytes as u64);
    counter.max_bytes = counter.max_bytes.max(bytes as u64);
}

pub struct Scope {
    name: &'static str,
    started: Option<Instant>,
    bytes: usize,
}

impl Scope {
    pub fn new(name: &'static str) -> Self {
        Self::bytes(name, 0)
    }

    pub fn bytes(
        name: &'static str,
        bytes: usize,
    ) -> Self {
        Self {
            name,
            started: ENABLED.load(Ordering::Relaxed).then(Instant::now),
            bytes,
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            record(self.name, started.elapsed().as_secs_f64(), self.bytes);
        }
    }
}
