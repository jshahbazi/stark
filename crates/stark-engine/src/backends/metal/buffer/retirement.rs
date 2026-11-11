use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use metal::{
    MTL4CommandQueue, MTLDeviceExt, MTLSharedEventExt, MTLSharedEventListener, MTLSharedEventNotificationBlock,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::{Condvar, Mutex};

#[derive(Default)]
struct RetirementState {
    pending: usize,
    unavailable: bool,
}

#[derive(Default)]
pub(in crate::backends::metal) struct RetirementTracker {
    state: Mutex<RetirementState>,
    changed: Condvar,
}

impl RetirementTracker {
    fn begin(&self) {
        self.state.lock().pending += 1;
    }

    fn finish(
        &self,
        unavailable: bool,
    ) {
        let mut state = self.state.lock();
        state.pending -= 1;
        state.unavailable |= unavailable;
        self.changed.notify_all();
    }

    pub(in crate::backends::metal) fn wait(
        &self,
        timeout: Duration,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock();
        while state.pending != 0 && !state.unavailable {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if self.changed.wait_for(&mut state, remaining).timed_out() && state.pending != 0 {
                return false;
            }
        }
        !state.unavailable && state.pending == 0
    }
}

pub(super) fn retire_after_queue<T: Send + Sync + 'static>(
    queue: &Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    tracker: &Arc<RetirementTracker>,
    resource: T,
) {
    tracker.begin();
    let Some(event) = queue.device().new_shared_event() else {
        // Drop cannot report allocation failure. Retaining storage is safer than
        // recycling an address whose submitted GPU uses may still be running.
        std::mem::forget(resource);
        tracker.finish(true);
        return;
    };
    let resource = Mutex::new(Some(resource));
    let retain = (event.clone(), queue.clone(), tracker.clone());
    event.notify_listener_at_value(
        &MTLSharedEventListener::shared_listener(),
        1,
        &MTLSharedEventNotificationBlock::new(move |_, _| {
            let _ = &retain;
            // Drop the resource explicitly before publishing retirement. A
            // completed GPU event alone does not prove its listener callback
            // has returned storage to the allocator's free cache.
            drop(resource.lock().take());
            retain.2.finish(false);
        }),
    );
    queue.signal_event_value(event.as_ref(), 1);
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use stark_engine_macros::stark_test;

    use super::RetirementTracker;

    #[stark_test]
    fn reclamation_waits_for_resource_retirement_and_fails_closed_when_unavailable() {
        let tracker = Arc::new(RetirementTracker::default());
        tracker.begin();
        assert!(!tracker.wait(Duration::ZERO));
        let retired = Arc::new(AtomicBool::new(false));
        let worker_tracker = tracker.clone();
        let worker_retired = retired.clone();
        std::thread::spawn(move || {
            worker_retired.store(true, Ordering::SeqCst);
            worker_tracker.finish(false);
        })
        .join()
        .unwrap();
        assert!(tracker.wait(Duration::ZERO));
        assert!(retired.load(Ordering::SeqCst));
        tracker.begin();
        tracker.finish(true);
        assert!(!tracker.wait(Duration::ZERO));
    }
}
