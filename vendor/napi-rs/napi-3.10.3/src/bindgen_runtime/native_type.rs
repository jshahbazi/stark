//! Stark patch: check Node-API's non-inherited native identity before reading a payload.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::{check_status, sys, Error, Result, Status};

// A leaked allocation gives each Rust TypeId a distinct process-lifetime identity.
// The registry address also separates this runtime from other loaded addon copies.
#[cfg(feature = "napi8")]
static TAGS: LazyLock<Mutex<HashMap<TypeId, &'static sys::napi_type_tag>>> =
  LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "napi8")]
fn tag<T: 'static>() -> &'static sys::napi_type_tag {
  let mut tags = TAGS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
  tags.entry(TypeId::of::<T>()).or_insert_with(|| {
    let value = Box::leak(Box::new(sys::napi_type_tag {
      lower: 0,
      upper: 0,
    }));
    value.lower = value as *const _ as usize as u64;
    value.upper = &TAGS as *const _ as usize as u64;
    value
  })
}

/// Tag an object immediately after successfully attaching the matching storage.
/// Never tag before `napi_wrap`: a failed wrap could otherwise label foreign storage.
///
/// # Safety
/// `env` and `value` must be live Node-API handles on the current thread.
#[doc(hidden)]
pub unsafe fn tag_native_type<T: 'static>(
  env: sys::napi_env,
  value: sys::napi_value,
) -> Result<()> {
  #[cfg(feature = "napi8")]
  {
    let mut matches = false;
    check_status!(
      unsafe { sys::napi_check_object_type_tag(env, value, tag::<T>(), &mut matches) },
      "Failed to inspect native object type"
    )?;
    if matches {
      return Ok(());
    }
    check_status!(unsafe { sys::napi_type_tag_object(env, value, tag::<T>()) }, "Failed to tag native object")?;
    Ok(())
  }
  #[cfg(not(feature = "napi8"))]
  {
    let _ = (env, value);
    Err(Error::new(Status::InvalidArg, "Native classes require Node-API 8 type tags"))
  }
}

/// Validate native storage identity without reading caller-controlled native memory.
/// JavaScript prototypes and Symbol.hasInstance have no effect on this check.
///
/// # Safety
/// `env` and `value` must be live Node-API handles on the current thread.
#[doc(hidden)]
pub unsafe fn validate_native_type<T: 'static>(
  env: sys::napi_env,
  value: sys::napi_value,
) -> Result<()> {
  #[cfg(feature = "napi8")]
  {
    let mut matches = false;
    check_status!(
      unsafe { sys::napi_check_object_type_tag(env, value, tag::<T>(), &mut matches) },
      "Failed to validate native object type"
    )?;
    if matches {
      Ok(())
    } else {
      Err(Error::new(Status::InvalidArg, "Native object type mismatch"))
    }
  }
  #[cfg(not(feature = "napi8"))]
  {
    let _ = (env, value);
    Err(Error::new(Status::InvalidArg, "Native classes require Node-API 8 type tags"))
  }
}
