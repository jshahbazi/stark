use metal::{
    MTL4CommandQueue, MTLDeviceExt, MTLSharedEventExt, MTLSharedEventListener, MTLSharedEventNotificationBlock,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

pub(super) fn retire_after_queue<T: Send + Sync + 'static>(
    queue: &Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    resource: T,
) {
    let Some(event) = queue.device().new_shared_event() else {
        // Drop cannot report allocation failure. Retaining storage is safer than
        // recycling an address whose submitted GPU uses may still be running.
        std::mem::forget(resource);
        return;
    };
    let retain = (resource, event.clone(), queue.clone());
    event.notify_listener_at_value(
        &MTLSharedEventListener::shared_listener(),
        1,
        &MTLSharedEventNotificationBlock::new(move |_, _| {
            let _ = &retain;
        }),
    );
    queue.signal_event_value(event.as_ref(), 1);
}
