//! Request-scoped macOS activity for latency-sensitive inference.
//!
//! The activity keeps a windowless worker's request classified as work on the
//! user's behalf. It ends when the request returns or unwinds, allowing an idle
//! worker to sleep normally. It does not change model precision or scheduling
//! dependencies. Reference: Lily's `src/activity.rs` at revision
//! `e22d58fa7cc488493fb1b5d810a198bd29730aeb` (Apache-2.0); the ABI and options
//! follow Foundation's `NSProcessInfo.h`. This implementation reuses Stark's
//! existing Objective-C dependency.

use std::{marker::PhantomData, rc::Rc};

use super::Result;

#[cfg(all(target_os = "macos", feature = "metal"))]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

/// Keeps a single inference request active until dropped on its calling thread.
///
/// This guard is inert outside macOS Metal builds. The thread affinity applies
/// on every platform so callers cannot accidentally move its lifetime between
/// threads when changing build targets.
pub struct InferenceActivity {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    process: objc2::rc::Retained<objc2::runtime::AnyObject>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    token: objc2::rc::Retained<objc2::runtime::AnyObject>,
    same_thread: PhantomData<Rc<()>>,
}

impl InferenceActivity {
    /// Begins an activity with a fixed diagnostic reason containing no request
    /// data. Hold it through completion and GPU cleanup, then let it drop.
    pub fn begin() -> Result<Self> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            objc2::rc::autoreleasepool(|_| {
                use objc2::{
                    msg_send,
                    rc::Retained,
                    runtime::{AnyClass, AnyObject},
                };

                use super::Error;

                let unavailable = || Error::Invalid("inference activity unavailable");
                let process_class = AnyClass::get(c"NSProcessInfo").ok_or_else(unavailable)?;
                let string_class = AnyClass::get(c"NSString").ok_or_else(unavailable)?;
                // SAFETY: Foundation's class methods return Objective-C objects;
                // Retained handles the autoreleased return convention and nil.
                let process: Option<Retained<AnyObject>> = unsafe { msg_send![process_class, processInfo] };
                let reason: Option<Retained<AnyObject>> =
                    unsafe { msg_send![string_class, stringWithUTF8String: c"Stark inference request".as_ptr()] };
                let process = process.ok_or_else(unavailable)?;
                let reason = reason.ok_or_else(unavailable)?;
                // NSActivityUserInitiated | NSActivityLatencyCritical. Foundation
                // defines NSActivityOptions as a 64-bit unsigned integer on macOS.
                let options = 0x00ff_ffffu64 | 0xff_0000_0000u64;
                // SAFETY: options has the declared ABI, reason is a live NSString,
                // and the returned activity object is retained until endActivity.
                let token: Option<Retained<AnyObject>> =
                    unsafe { msg_send![&*process, beginActivityWithOptions: options, reason: &*reason] };
                Ok(Self {
                    process,
                    token: token.ok_or_else(unavailable)?,
                    same_thread: PhantomData,
                })
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        Ok(Self {
            same_thread: PhantomData,
        })
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Drop for InferenceActivity {
    fn drop(&mut self) {
        // SAFETY: this is the exact activity token returned by this process.
        // PhantomData<Rc<()>> keeps begin/end on the same calling thread.
        objc2::rc::autoreleasepool(|_| unsafe {
            let _: () = objc2::msg_send![&*self.process, endActivity: &*self.token];
        });
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::InferenceActivity;

    #[stark_test]
    fn activity_supports_nested_request_scopes_and_unwinding() {
        let outer = InferenceActivity::begin().expect("request activity unavailable");
        {
            let _inner = InferenceActivity::begin().expect("nested activity unavailable");
        }
        assert!(
            std::panic::catch_unwind(|| {
                let _activity = InferenceActivity::begin().expect("unwinding activity unavailable");
                panic!("test request unwind");
            })
            .is_err()
        );
        drop(outer);
        // A fresh begin/end after unwind exercises the same Foundation token
        // ownership as a worker recovering from a failed request.
        drop(InferenceActivity::begin().expect("recovery activity unavailable"));
    }
}
