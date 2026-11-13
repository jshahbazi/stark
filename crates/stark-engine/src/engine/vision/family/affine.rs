//! Canonical packed decoder storage admission for the v3 image profile.
//!
//! The operation graph still comes from bounded dense geometry. Storage specs
//! can only replace its exact logical matrices, never add operations or tensors.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{TensorMap, exact_keys, invalid, validate_tensor_header_with_specs};
use crate::backends::common::InvalidInput;

fn integer(
    value: &Value,
    maximum: u32,
) -> Result<u32, InvalidInput> {
    value.as_u64().and_then(|n| u32::try_from(n).ok()).filter(|n| *n > 0 && *n <= maximum).ok_or_else(invalid)
}

fn packed(
    tensors: &mut TensorMap,
    prefix: &str,
    rows: u32,
    columns: u32,
    bits: u32,
    layout: &str,
) -> Result<(), InvalidInput> {
    if ![4, 5, 6, 8].contains(&bits) || !columns.is_multiple_of(64) {
        return Err(invalid());
    }
    let bytes = columns.checked_mul(bits).filter(|n| n.is_multiple_of(8)).ok_or_else(invalid)? / 8;
    tensors.insert(format!("{prefix}.weights"), ("U8", vec![rows, bytes]));
    let groups = columns / 64;
    let planes = if layout == "input_output" {
        vec![rows, groups]
    } else {
        vec![groups, rows.checked_add(3).ok_or_else(invalid)? / 4 * 4]
    };
    for name in ["scales", "biases"] {
        tensors.insert(format!("{prefix}.{name}"), ("BF16", planes.clone()));
    }
    Ok(())
}

pub(super) fn validate_header(
    header: &Value,
    file_length: u64,
    data_start: u64,
    logical: &TensorMap,
) -> Result<(), InvalidInput> {
    let metadata = header["__metadata__"].as_object().ok_or_else(invalid)?;
    let mut tensors = logical.clone();
    let mut specs = BTreeMap::new();
    let mut packed_count = 0;
    for (name, (dtype, shape)) in logical {
        let Some(prefix) = name.strip_suffix(".weights") else {
            continue;
        };
        if *dtype != "BF16" || !(prefix.starts_with("decoder.") || prefix.starts_with("mtp.")) {
            continue;
        }
        let layout = if matches!(prefix, "decoder.embedding.embedding" | "decoder.embedding.input_embedding") {
            "input_output"
        } else {
            "output_input"
        };
        let key = format!("{prefix}.spec");
        let spec: Value = serde_json::from_str(metadata.get(&key).and_then(Value::as_str).ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
        match spec["type"].as_str() {
            Some("FullPrecisionSpec") if spec == json!({"type":"FullPrecisionSpec", "layout":layout}) => {},
            Some("MLXSpec") => {
                if !exact_keys(&spec, &["type", "bits", "group_size", "layout"])
                    || spec["layout"] != layout
                    || spec["group_size"] != 64
                    || shape.len() != 2
                {
                    return Err(invalid());
                }
                let bits = integer(&spec["bits"], 8)?;
                tensors.remove(name);
                packed(&mut tensors, prefix, shape[0], shape[1], bits, layout)?;
                packed_count += 1;
            },
            Some("MLXRowPartitionedSpec") => {
                if layout != "output_input"
                    || !exact_keys(&spec, &["type", "layout", "segments"])
                    || spec["layout"] != layout
                    || shape.len() != 2
                {
                    return Err(invalid());
                }
                let segments =
                    spec["segments"].as_array().filter(|s| !s.is_empty() && s.len() <= 1024).ok_or_else(invalid)?;
                let mut total = 0u32;
                for (index, segment) in segments.iter().enumerate() {
                    if !exact_keys(segment, &["rows", "bits", "group_size"]) || segment["group_size"] != 64 {
                        return Err(invalid());
                    }
                    let rows = integer(&segment["rows"], shape[0])?;
                    total = total.checked_add(rows).filter(|total| *total <= shape[0]).ok_or_else(invalid)?;
                    packed(
                        &mut tensors,
                        &format!("{prefix}.segment_{index}"),
                        rows,
                        shape[1],
                        integer(&segment["bits"], 8)?,
                        layout,
                    )?;
                }
                if total != shape[0] {
                    return Err(invalid());
                }
                tensors.remove(name);
                packed_count += 1;
            },
            _ => return Err(invalid()),
        }
        specs.insert(key, spec);
    }
    if packed_count == 0 {
        return Err(invalid());
    }
    let mut bytes = 0u64;
    for (dtype, shape) in tensors.values() {
        let count = shape.iter().try_fold(1u32, |n, width| n.checked_mul(*width)).ok_or_else(invalid)?;
        let width = match *dtype {
            "U8" => 1,
            "BF16" => 2,
            "F32" => 4,
            _ => return Err(invalid()),
        };
        bytes = bytes.checked_add(u64::from(count) * width).ok_or_else(invalid)?;
    }
    if bytes > super::dynamic::MAX_WEIGHT_BYTES {
        return Err(invalid());
    }
    validate_tensor_header_with_specs(header, file_length, data_start, &tensors, Some(&specs))
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn fixture(partitioned: bool) -> (Value, u64, TensorMap) {
        let prefix = "decoder.transformer.layers.0.mlp.up_projection.weights";
        let logical = BTreeMap::from([(format!("{prefix}.weights"), ("BF16", vec![3, 64]))]);
        let mut physical = TensorMap::new();
        let spec = if partitioned {
            packed(&mut physical, &format!("{prefix}.segment_0"), 1, 64, 4, "output_input").unwrap();
            packed(&mut physical, &format!("{prefix}.segment_1"), 2, 64, 8, "output_input").unwrap();
            json!({"type":"MLXRowPartitionedSpec","layout":"output_input",
                "segments":[{"rows":1,"bits":4,"group_size":64},{"rows":2,"bits":8,"group_size":64}]})
        } else {
            packed(&mut physical, prefix, 3, 64, 4, "output_input").unwrap();
            json!({"type":"MLXSpec","bits":4,"group_size":64,"layout":"output_input"})
        };
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, (dtype, shape)) in physical {
            let begin = end;
            end += shape.iter().map(|n| u64::from(*n)).product::<u64>()
                * if dtype == "U8" {
                    1
                } else {
                    2
                };
            header.insert(name, json!({"dtype":dtype,"shape":shape,"data_offsets":[begin,end]}));
        }
        header.insert("__metadata__".into(), json!({format!("{prefix}.spec"):spec.to_string()}));
        (Value::Object(header), end + 8, logical)
    }

    #[stark_test]
    fn affine_header_requires_exact_physical_planes_and_safe_specs() {
        for partitioned in [false, true] {
            let (header, size, logical) = fixture(partitioned);
            validate_header(&header, size, 8, &logical).unwrap();
            for corruption in ["dtype", "shape", "offset", "extra", "missing", "spec"] {
                let mut invalid = header.clone();
                let name = invalid.as_object().unwrap().keys().find(|name| *name != "__metadata__").unwrap().clone();
                match corruption {
                    "dtype" => invalid[&name]["dtype"] = json!("F32"),
                    "shape" => invalid[&name]["shape"] = json!([u32::MAX, u32::MAX]),
                    "offset" => invalid[&name]["data_offsets"] = json!([0, size]),
                    "extra" => {
                        invalid["unexpected"] = invalid[&name].clone();
                    },
                    "missing" => {
                        invalid.as_object_mut().unwrap().remove(&name);
                    },
                    "spec" => {
                        let meta = invalid["__metadata__"].as_object_mut().unwrap();
                        let spec = meta.values_mut().next().unwrap();
                        let mut value: Value = serde_json::from_str(spec.as_str().unwrap()).unwrap();
                        value["layout"] = json!("input_output");
                        *spec = json!(value.to_string());
                    },
                    _ => unreachable!(),
                }
                assert!(validate_header(&invalid, size, 8, &logical).is_err(), "{corruption}");
            }
        }
    }
}
