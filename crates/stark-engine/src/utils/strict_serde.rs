use monostate::{ConstStr, MustBeBool, MustBeStr};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum Unsupported {}

pub trait DeserializeStrict<'de>: Deserialize<'de> {}
pub trait DeserializeStrictOwned: for<'de> DeserializeStrict<'de> {}
impl<T: for<'de> DeserializeStrict<'de>> DeserializeStrictOwned for T {}

macro_rules! impl_strict {
  ($($ty:ty),* $(,)?) => {
      $(impl<'de> DeserializeStrict<'de> for $ty {})*
  };
}

impl_strict!(Unsupported, String, f32, u32, u64, usize, bool, serde_json::Map<String, serde_json::Value>);

impl<'de, T: DeserializeStrict<'de>> DeserializeStrict<'de> for Box<T> {}
impl<'de, T: DeserializeStrict<'de>> DeserializeStrict<'de> for Box<[T]> {}
impl<'de, T: DeserializeStrict<'de>> DeserializeStrict<'de> for Option<T> {}
impl<'de, T: DeserializeStrict<'de>, const N: usize> DeserializeStrict<'de> for [T; N] where [T; N]: Deserialize<'de> {}
impl<'de, A: DeserializeStrict<'de>, B: DeserializeStrict<'de>> DeserializeStrict<'de> for (A, B) {}
impl<'de, const V: bool> DeserializeStrict<'de> for MustBeBool<V> {}
impl<'de, V: ConstStr> DeserializeStrict<'de> for MustBeStr<V> {}

pub fn required<'de, D: Deserializer<'de>, T: DeserializeStrict<'de>>(deserializer: D) -> Result<T, D::Error> {
    T::deserialize(deserializer)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixedArray {
        #[serde(deserialize_with = "required")]
        sections: [u32; 3],
    }

    #[stark_test]
    fn strict_fixed_arrays_require_exact_length_and_strict_elements() {
        let valid: FixedArray = serde_json::from_value(serde_json::json!({"sections":[11,11,10]})).unwrap();
        assert_eq!(valid.sections, [11, 11, 10]);
        for invalid in [
            serde_json::json!({}),
            serde_json::json!({"sections":null}),
            serde_json::json!({"sections":[]}),
            serde_json::json!({"sections":[11,11]}),
            serde_json::json!({"sections":[11,11,10,0]}),
            serde_json::json!({"sections":[11,11,"10"]}),
            serde_json::json!({"sections":[11,11,-1]}),
            serde_json::json!({"sections":[11,11,10.5]}),
            serde_json::json!({"sections":[11,11,10],"unexpected":true}),
        ] {
            assert!(serde_json::from_value::<FixedArray>(invalid).is_err());
        }
    }
}
