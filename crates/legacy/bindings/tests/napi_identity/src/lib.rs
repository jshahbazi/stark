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
    pub fn with_reference(&self, reference: Reference<Payload>) -> String {
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
        Self { value }
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
