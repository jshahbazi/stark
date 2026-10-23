use napi::bindgen_prelude::*;
use napi_derive::napi;

// Exercise the repository's owned-class and enum conversion generators.
#[bindings::export(Structure(Class))]
#[derive(Clone)]
pub struct Payload {
    pub value: String,
}

#[bindings::export(Structure(Class))]
#[derive(Clone)]
pub struct Seed {}

#[bindings::export(Structure(Class))]
#[derive(Clone)]
pub struct ForeignPayload {
    pub value: String,
}

#[bindings::export(Enumeration)]
#[derive(Clone)]
pub enum Content {
    Text {
        value: String,
    },
    Empty {},
}

#[napi]
pub fn read_owned(value: Payload) -> String {
    value.value
}

#[napi]
pub fn read_ref(value: &Payload) -> String {
    value.value.clone()
}

#[napi]
pub fn read_mut(value: &mut Payload) -> String {
    value.value.clone()
}

#[napi]
pub fn read_instance(value: ClassInstance<Payload>) -> String {
    value.as_ref().value.clone()
}

#[napi]
pub fn read_instances(values: Vec<ClassInstance<Payload>>) -> String {
    values.iter().map(|value| value.as_ref().value.as_str()).collect()
}

#[napi]
pub fn read_reference(
    env: Env,
    value: Unknown,
) -> Result<String> {
    let value = unsafe { Reference::<Payload>::from_napi_value(env.raw(), value.raw())? };
    Ok(value.value.clone())
}

#[napi]
pub fn read_enum(value: Content) -> String {
    match value {
        Content::Text {
            value,
        } => value,
        Content::Empty {} => "empty".to_owned(),
    }
}

#[napi]
impl Payload {
    #[napi]
    pub fn read(&self) -> String {
        self.value.clone()
    }

    #[napi]
    pub fn with_reference(
        &self,
        reference: Reference<Payload>,
    ) -> String {
        reference.value.clone()
    }
}

// No JS constructor: returned values go through runtime new_instance instead.
#[napi]
pub struct Dynamic {
    pub value: String,
}

#[napi]
impl Dynamic {
    #[napi(factory)]
    pub async fn async_factory(value: String) -> Self {
        Self {
            value,
        }
    }

    #[napi(factory)]
    pub fn factory(value: String) -> Self {
        Self {
            value,
        }
    }

    #[napi]
    pub fn read(&self) -> String {
        self.value.clone()
    }
}

#[napi]
pub fn make_dynamic() -> Dynamic {
    Dynamic {
        value: "dynamic".to_owned(),
    }
}

#[napi]
pub fn read_dynamic(value: ClassInstance<Dynamic>) -> String {
    value.as_ref().value.clone()
}

#[napi]
pub fn generic_wrap<'env>(mut object: Object<'env>) -> Result<Object<'env>> {
    object.wrap("wrapped".to_owned(), None)?;
    Ok(object)
}

#[napi]
pub fn generic_unwrap(object: Object) -> Result<String> {
    Ok(object.unwrap::<String>()?.clone())
}

#[napi]
pub fn generic_remove(mut object: Object) -> Result<()> {
    object.remove_wrapped::<String>()
}

#[napi]
pub fn signal(value: AbortSignal) {
    value.on_abort(|| {});
}

#[napi]
pub fn make_external(value: String) -> External<String> {
    External::new(value)
}

#[napi]
pub fn make_external_ref(
    env: Env,
    value: String,
) -> Result<ExternalRef<String>> {
    ExternalRef::new(&env, value)
}

#[napi]
pub fn make_wrong_external() -> External<u32> {
    External::new(42)
}

#[napi]
#[allow(deprecated)]
pub fn make_compat_external(
    env: Env,
    value: String,
    size_hint: i32,
) -> Result<napi::JsExternal<'static>> {
    let value = env.create_external(value, Some(i64::from(size_hint)))?;
    unsafe { napi::JsExternal::from_napi_value(env.raw(), value.raw()) }
}

#[napi]
pub fn read_external(value: &External<String>) -> String {
    value.as_ref().clone()
}

#[napi]
pub fn mutate_external(value: &mut External<String>) -> String {
    value.as_mut().push('!');
    value.as_ref().clone()
}

#[napi]
pub fn read_external_ref(value: ExternalRef<String>) -> String {
    (*value).clone()
}

#[napi]
pub fn read_js_external(value: napi::JsExternal) -> Result<String> {
    value.get_value::<String>().cloned()
}

#[napi]
pub fn read_js_external_ref(value: napi::JsExternal) -> Result<String> {
    value.create_ref::<String>().map(|value| (*value).clone())
}

#[napi]
#[allow(deprecated)]
pub fn read_compat_external(
    env: Env,
    value: napi::JsExternal,
) -> Result<String> {
    env.get_value_external::<String>(&value).cloned()
}

#[napi]
pub fn transport_external(value: napi::JsExternal) -> napi::JsExternal {
    value
}

// Emulate externals supplied by an independent native addon. Typed extraction
// must reject them before inspecting the allocation, including a forged header.
#[napi]
pub fn make_foreign_external(
    env: Env,
    kind: u32,
) -> Result<napi::JsExternal<'static>> {
    unsafe extern "C" fn free_byte(
        _: sys::napi_env,
        data: *mut std::ffi::c_void,
        _: *mut std::ffi::c_void,
    ) {
        unsafe { drop(Box::from_raw(data.cast::<u8>())) };
    }
    unsafe extern "C" fn free_header(
        _: sys::napi_env,
        data: *mut std::ffi::c_void,
        _: *mut std::ffi::c_void,
    ) {
        unsafe { drop(Box::from_raw(data.cast::<std::any::TypeId>())) };
    }
    let (data, finalize): (_, sys::napi_finalize) = match kind {
        0 => (std::ptr::null_mut(), None),
        1 => (Box::into_raw(Box::new(0u8)).cast(), Some(free_byte)),
        _ => (Box::into_raw(Box::new(std::any::TypeId::of::<String>())).cast(), Some(free_header)),
    };
    let mut value = std::ptr::null_mut();
    let status = unsafe { sys::napi_create_external(env.raw(), data, finalize, std::ptr::null_mut(), &mut value) };
    if status != sys::Status::napi_ok {
        if let Some(finalize) = finalize {
            unsafe { finalize(env.raw(), data, std::ptr::null_mut()) };
        }
        return Err(napi::Error::new(napi::Status::GenericFailure, "external_fixture_creation_failed"));
    }
    unsafe { napi::JsExternal::from_napi_value(env.raw(), value) }
}

pub struct TrackedExternal(String);

static EXTERNAL_DROPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
thread_local! {
    static HELD_EXTERNAL: std::cell::RefCell<Option<ExternalRef<TrackedExternal>>> = const { std::cell::RefCell::new(None) };
}

impl Drop for TrackedExternal {
    fn drop(&mut self) {
        EXTERNAL_DROPS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[napi]
pub fn make_tracked_external(value: String) -> External<TrackedExternal> {
    External::new(TrackedExternal(value))
}

#[napi]
pub fn hold_external(value: ExternalRef<TrackedExternal>) {
    HELD_EXTERNAL.with(|held| *held.borrow_mut() = Some(value));
}

#[napi]
pub fn held_external_value() -> String {
    HELD_EXTERNAL.with(|held| held.borrow().as_ref().map(|value| value.0.clone()).unwrap_or_default())
}

#[napi]
pub fn release_external() {
    HELD_EXTERNAL.with(|held| *held.borrow_mut() = None);
}

#[napi]
pub fn external_drop_count() -> u32 {
    EXTERNAL_DROPS.load(std::sync::atomic::Ordering::SeqCst)
}
