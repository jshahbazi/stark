//! Strict, self-contained Uzu MLX storage admission for the Qwen architecture.
//! Only canonical converted planes are opened; logical aliases are in memory.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    path::Path,
};

use anyhow::{Context as _, Result, ensure};
use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, SeqAccess, Visitor},
};

use super::{Checkpoint, SafetensorsDType, TensorMeta};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum DType {
    BF16,
    F32,
    U32,
    I64,
    U64,
}

impl DType {
    fn size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::F32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::U32 => "U32",
            Self::I64 => "I64",
            Self::U64 => "U64",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Direct,
    PackedU32,
    GroupOutput,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub layout: Layout,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    #[serde(rename = "type")]
    ty: String,
    bits: usize,
    group_size: usize,
    layout: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OmittedPart {
    bytes: u64,
    tensors: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Omitted {
    vision: OmittedPart,
    mtp: OmittedPart,
}

// Unlike serde_json::Value's ordinary map visitor, preserve the failure on
// duplicate keys at every depth. Header entries and metadata are converted
// into their typed forms only after this check, so no earlier value is lost.
struct UniqueValue(serde_json::Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value with unique object keys")
            }
            fn visit_bool<E: serde::de::Error>(
                self,
                value: bool,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::Bool(value)))
            }
            fn visit_i64<E: serde::de::Error>(
                self,
                value: i64,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::from(value)))
            }
            fn visit_u64<E: serde::de::Error>(
                self,
                value: u64,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::from(value)))
            }
            fn visit_f64<E: serde::de::Error>(
                self,
                value: f64,
            ) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueValue(serde_json::Value::Number(number)))
                    .ok_or_else(|| E::custom("invalid Uzu JSON number"))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                value: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::String(value.to_owned())))
            }
            fn visit_string<E: serde::de::Error>(
                self,
                value: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::String(value)))
            }
            fn visit_unit<E: serde::de::Error>(
                self,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(serde_json::Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((name, value)) =
                    map.next_entry::<String, UniqueValue>()?
                {
                    if name.len() > 512
                        || values.len() >= 16_385
                        || values.insert(name, value.0).is_some()
                    {
                        return Err(serde::de::Error::custom(
                            "duplicate or excessive nested Uzu header entries",
                        ));
                    }
                }
                Ok(UniqueValue(serde_json::Value::Object(values)))
            }
        }
        deserializer.deserialize_any(ValueVisitor)
    }
}

// JSON object duplicates must fail before maps can overwrite them.
struct UniqueObject(BTreeMap<String, serde_json::Value>);
impl<'de> Deserialize<'de> for UniqueObject {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct ObjectVisitor;
        impl<'de> Visitor<'de> for ObjectVisitor {
            type Value = UniqueObject;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a unique tensor header object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut entries = BTreeMap::new();
                while let Some((name, value)) =
                    map.next_entry::<String, UniqueValue>()?
                {
                    if name.len() > 512
                        || entries.len() >= 16_385
                        || entries.insert(name, value.0).is_some()
                    {
                        return Err(serde::de::Error::custom(
                            "duplicate or excessive Uzu header entries",
                        ));
                    }
                }
                Ok(UniqueObject(entries))
            }
        }
        deserializer.deserialize_map(ObjectVisitor)
    }
}

fn elements(shape: &[usize]) -> Result<usize> {
    ensure!(!shape.is_empty() && shape.len() <= 4, "invalid Uzu tensor rank");
    shape.iter().try_fold(1usize, |n, &dim| {
        ensure!(dim > 0, "empty Uzu tensor dimension");
        n.checked_mul(dim).context("Uzu tensor dimensions overflow")
    })
}

impl Checkpoint {
    /// Opens exactly `model.safetensors` and binds canonical Uzu storage to
    /// the specialized architecture's logical tensor names. The aliases do
    /// not refer to another checkpoint or change any quantized weight bits.
    pub fn open_uzu(
        dir: impl AsRef<Path>,
        bindings: &BTreeMap<String, Binding>,
    ) -> Result<Self> {
        ensure!(
            !bindings.is_empty() && bindings.len() <= 16_384,
            "invalid Uzu tensor bindings"
        );
        let path = dir.as_ref().join("model.safetensors");
        let metadata =
            std::fs::symlink_metadata(&path).context("Uzu model file metadata")?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "Uzu model must be a regular file"
        );
        let mut file = File::open(&path).context("opening Uzu model")?;
        let mut length = [0u8; 8];
        file.read_exact(&mut length).context("reading Uzu header length")?;
        let header_len = u64::from_le_bytes(length);
        ensure!(
            header_len > 0 && header_len <= 8 * 1024 * 1024 && header_len % 8 == 0,
            "invalid Uzu header length"
        );
        let data_start = 8 + header_len;
        ensure!(data_start <= metadata.len(), "truncated Uzu header");
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header).context("reading Uzu header")?;
        let UniqueObject(mut raw) =
            serde_json::from_slice(&header).context("parsing Uzu header")?;
        let raw_metadata =
            raw.remove("__metadata__").context("missing Uzu matrix metadata")?;
        let metadata: BTreeMap<String, String> =
            serde_json::from_value(raw_metadata)
                .context("invalid Uzu matrix metadata")?;
        ensure!(
            metadata.get("stark.qwen38.schema_version").map(String::as_str)
                == Some("1"),
            "unsupported Uzu Qwen schema version"
        );
        let omitted: Omitted = serde_json::from_str(
            metadata
                .get("stark.qwen38.omitted")
                .context("missing Uzu omission provenance")?,
        )
        .context("invalid Uzu omission provenance")?;
        ensure!(
            omitted.vision.tensors <= 16_384
                && omitted.mtp.tensors <= 16_384
                && omitted.vision.bytes <= 1 << 44
                && omitted.mtp.bytes <= 1 << 44,
            "invalid Uzu omission provenance bounds"
        );
        ensure!(raw.len() == bindings.len(), "Uzu tensor inventory mismatch");
        let mut physical = BTreeMap::new();
        let mut spans = Vec::with_capacity(raw.len());
        for (name, value) in raw {
            ensure!(
                name.starts_with("language_model.") && name.len() <= 512,
                "noncanonical Uzu tensor name"
            );
            let entry: Entry =
                serde_json::from_value(value).context("invalid Uzu tensor entry")?;
            let size = match entry.dtype.as_str() {
                "U8" => 1,
                "BF16" => 2,
                "F32" | "U32" => 4,
                "I64" | "U64" => 8,
                _ => anyhow::bail!("unsupported Uzu tensor dtype"),
            };
            let byte_len = elements(&entry.shape)?
                .checked_mul(size)
                .context("Uzu tensor size overflow")?;
            let [start, end] = entry.data_offsets;
            ensure!(
                start <= end
                    && end - start == byte_len as u64
                    && end <= file.metadata()?.len() - data_start,
                "invalid Uzu tensor span"
            );
            spans.push((start, end));
            physical.insert(name, entry);
        }
        spans.sort_unstable();
        let mut end = 0;
        for (start, next_end) in spans {
            ensure!(start == end, "Uzu tensor spans overlap or have gaps");
            end = next_end;
        }
        ensure!(
            data_start + end == file.metadata()?.len(),
            "Uzu model has trailing data"
        );

        let mut tensors = HashMap::with_capacity(bindings.len());
        let mut seen = HashSet::with_capacity(bindings.len());
        let mut specs = HashSet::new();
        for (logical, binding) in bindings {
            ensure!(
                logical.len() <= 512
                    && (logical.starts_with("model.language_model.")
                        || logical.starts_with("lm_head.")),
                "invalid Qwen logical tensor name"
            );
            ensure!(seen.insert(&binding.name), "duplicate Uzu physical binding");
            let entry =
                physical.get(&binding.name).context("missing Uzu bound tensor")?;
            let count = elements(&binding.shape)?;
            let native_size = count
                .checked_mul(binding.dtype.size())
                .context("Uzu logical tensor size overflow")?;
            let mut group_output = None;
            match binding.layout {
                Layout::Direct => ensure!(
                    binding.dtype != DType::U32
                        && entry.dtype == binding.dtype.name()
                        && entry.shape == binding.shape,
                    "direct Uzu tensor mismatch"
                ),
                Layout::PackedU32 => {
                    ensure!(
                        binding.dtype == DType::U32
                            && (binding.shape.len() == 2 || binding.shape.len() == 3),
                        "invalid packed Uzu logical tensor"
                    );
                    let columns = *binding.shape.last().unwrap();
                    ensure!(
                        entry.dtype == "U8"
                            && entry.shape == [count / columns, columns * 4],
                        "packed Uzu tensor mismatch"
                    );
                    let logical_base = logical
                        .strip_suffix(".weight")
                        .context("packed Uzu tensor name")?;
                    let base = binding
                        .name
                        .strip_suffix(".weights")
                        .context("packed Uzu physical name")?;
                    let spec_key = format!("{base}.spec");
                    let spec: Spec = serde_json::from_str(
                        metadata.get(&spec_key).context("missing Uzu MLXSpec")?,
                    )
                    .context("invalid Uzu MLXSpec")?;
                    let ngram =
                        logical_base.contains(".ple_embedding.ngram_embedding.shard_");
                    ensure!(
                        spec.ty == "MLXSpec"
                            && matches!(spec.bits, 4 | 8)
                            && if ngram {
                                spec.bits == 4 && spec.group_size == 32
                            } else {
                                spec.group_size == 64
                            },
                        "unsupported Uzu MLXSpec"
                    );
                    let rows = count / columns;
                    let input =
                        columns.checked_mul(32).context("packed width overflow")?
                            / spec.bits;
                    ensure!(
                        columns * 32 % spec.bits == 0 && input % spec.group_size == 0,
                        "invalid Uzu group geometry"
                    );
                    let groups = input / spec.group_size;
                    let embedding = logical_base.ends_with("embed_tokens") || ngram;
                    ensure!(
                        spec.layout
                            == if embedding { "input_output" } else { "output_input" },
                        "Uzu matrix layout mismatch"
                    );
                    for suffix in ["scales", "biases"] {
                        let param = bindings
                            .get(&format!("{logical_base}.{suffix}"))
                            .context("missing Uzu affine parameter binding")?;
                        ensure!(
                            param.name == format!("{base}.{suffix}")
                                && param.dtype == DType::BF16
                                && param.shape.last() == Some(&groups)
                                && elements(&param.shape)? == rows * groups,
                            "invalid Uzu affine parameter geometry"
                        );
                        ensure!(
                            param.layout
                                == if embedding {
                                    Layout::Direct
                                } else {
                                    Layout::GroupOutput
                                },
                            "Uzu affine parameter layout mismatch"
                        );
                    }
                    specs.insert(spec_key);
                }
                Layout::GroupOutput => {
                    ensure!(
                        binding.dtype == DType::BF16
                            && (binding.shape.len() == 2 || binding.shape.len() == 3),
                        "invalid group-major Uzu parameter"
                    );
                    let groups = *binding.shape.last().unwrap();
                    let rows = count / groups;
                    let stride =
                        rows.checked_add(3).context("Uzu row padding overflow")? / 4
                            * 4;
                    ensure!(
                        entry.dtype == "BF16" && entry.shape == [groups, stride],
                        "group-major Uzu tensor mismatch"
                    );
                    // Affine parameters are small; keep malicious configuration
                    // from turning the load-time transpose into a weight copy.
                    ensure!(
                        native_size <= 256 * 1024 * 1024,
                        "Uzu affine parameter is too large"
                    );
                    group_output = Some((rows, groups, stride));
                }
            }
            if group_output.is_none() {
                ensure!(
                    entry.data_offsets[1] - entry.data_offsets[0] == native_size as u64,
                    "Uzu logical byte span mismatch"
                );
            }
            tensors.insert(
                logical.clone(),
                TensorMeta {
                    dtype: SafetensorsDType::from_str(binding.dtype.name()),
                    shape: binding.shape.clone(),
                    shard: path.clone(),
                    start: data_start + entry.data_offsets[0],
                    end: data_start + entry.data_offsets[1],
                    group_output,
                },
            );
        }
        ensure!(
            metadata.keys().all(|key| specs.contains(key)
                || matches!(
                    key.as_str(),
                    "stark.qwen38.schema_version" | "stark.qwen38.omitted"
                )),
            "unbound Uzu matrix metadata"
        );
        Ok(Self { tensors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture {
        dir: std::path::PathBuf,
        bindings: BTreeMap<String, Binding>,
        native: BTreeMap<String, Vec<u8>>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
    fn fixture(
        rows: usize,
        expert: bool,
        embedding: bool,
        bits: usize,
        ngram: bool,
    ) -> Fixture {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "stark-uzu-checkpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let base = if ngram {
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
        } else if embedding {
            "model.language_model.embed_tokens"
        } else {
            "lm_head"
        };
        let physical_base = if ngram {
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
        } else if embedding {
            "language_model.model.embed_tokens"
        } else {
            "language_model.lm_head"
        };
        let groups = 2;
        let group_size = if ngram { 32 } else { 64 };
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({
            (format!("{physical_base}.spec")): serde_json::json!({"type":"MLXSpec","bits":bits,"group_size":group_size,"layout":if embedding {"input_output"} else {"output_input"}}).to_string(),
            "stark.qwen38.schema_version":"1", "stark.qwen38.omitted":"{\"vision\":{\"bytes\":0,\"tensors\":0},\"mtp\":{\"bytes\":0,\"tensors\":0}}"
        }));
        let mut payload = Vec::new();
        let mut bindings = BTreeMap::new();
        let mut native = BTreeMap::new();
        for suffix in ["weight", "scales", "biases"] {
            let weight = suffix == "weight";
            let logical = format!("{base}.{suffix}");
            let physical =
                format!("{physical_base}.{}", if weight { "weights" } else { suffix });
            let columns = if weight { groups * group_size * bits / 32 } else { groups };
            let shape =
                if expert { vec![2, rows / 2, columns] } else { vec![rows, columns] };
            let bytes: Vec<u8> = (0..rows * columns * if weight { 4 } else { 2 })
                .map(|n| (n * 31 + 7) as u8)
                .collect();
            let start = payload.len();
            let physical_shape;
            let layout;
            if weight {
                physical_shape = vec![rows, columns * 4];
                layout = Layout::PackedU32;
                payload.extend_from_slice(&bytes);
            } else if embedding {
                physical_shape = shape.clone();
                layout = Layout::Direct;
                payload.extend_from_slice(&bytes);
            } else {
                let stride = (rows + 3) / 4 * 4;
                physical_shape = vec![groups, stride];
                layout = Layout::GroupOutput;
                let mut transposed = vec![0u8; groups * stride * 2];
                for row in 0..rows {
                    for group in 0..groups {
                        transposed[2 * (group * stride + row)
                            ..2 * (group * stride + row) + 2]
                            .copy_from_slice(
                                &bytes[2 * (row * groups + group)
                                    ..2 * (row * groups + group) + 2],
                            );
                    }
                }
                payload.extend_from_slice(&transposed);
            }
            header.insert(physical.clone(), serde_json::json!({"dtype":if weight {"U8"} else {"BF16"},"shape":physical_shape,"data_offsets":[start,payload.len()]}));
            bindings.insert(
                logical.clone(),
                Binding {
                    name: physical,
                    dtype: if weight { DType::U32 } else { DType::BF16 },
                    shape,
                    layout,
                },
            );
            native.insert(logical, bytes);
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(json);
        file.extend(payload);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        Fixture { dir, bindings, native }
    }

    #[test]
    fn canonical_q4_planes_restore_exact_bits_for_linear_expert_and_embedding() {
        for (rows, expert, embedding, bits, ngram) in [
            (3, false, false, 4, false),
            (6, true, false, 4, false),
            (3, false, true, 4, false),
            (3, false, false, 8, false),
            (3, false, true, 8, false),
            (3, false, true, 4, true),
        ] {
            let fixture = fixture(rows, expert, embedding, bits, ngram);
            let checkpoint =
                Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).unwrap();
            for (name, bytes) in &fixture.native {
                assert_eq!(&checkpoint.read(name).unwrap(), bytes);
                assert_eq!(
                    checkpoint.meta(name).unwrap().shape,
                    fixture.bindings[name].shape
                );
            }
        }
    }

    #[test]
    fn duplicate_physical_bindings_and_native_u32_storage_are_rejected() {
        let mut fixture = fixture(3, false, false, 4, false);
        let original = fixture.bindings.clone();
        fixture.bindings.get_mut("lm_head.biases").unwrap().name =
            fixture.bindings["lm_head.scales"].name.clone();
        assert!(Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).is_err());
        fixture.bindings = original;
        fixture.bindings.get_mut("lm_head.weight").unwrap().layout = Layout::Direct;
        assert!(Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).is_err());
    }

    #[test]
    fn wrong_shapes_and_nonzero_affine_padding_fail() {
        let mut fixture = fixture(3, false, false, 4, false);
        fixture.bindings.get_mut("lm_head.scales").unwrap().shape[1] = 1;
        assert!(Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).is_err());
        fixture.bindings.get_mut("lm_head.scales").unwrap().shape[1] = 2;
        let checkpoint = Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).unwrap();
        let meta = checkpoint.meta("lm_head.scales").unwrap();
        let mut file = std::fs::read(fixture.dir.join("model.safetensors")).unwrap();
        file[meta.start() as usize + 6] = 1;
        std::fs::write(fixture.dir.join("model.safetensors"), file).unwrap();
        assert!(checkpoint.read("lm_head.scales").is_err());
    }

    #[test]
    fn trailing_bytes_and_unbound_inventory_fail() {
        let mut fixture = fixture(3, false, false, 4, false);
        let mut bytes = std::fs::read(fixture.dir.join("model.safetensors")).unwrap();
        bytes.push(0);
        std::fs::write(fixture.dir.join("model.safetensors"), bytes).unwrap();
        assert!(Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).is_err());
        fixture.bindings.remove("lm_head.scales");
        assert!(Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).is_err());
    }

    #[test]
    fn declared_mlx_group_policy_cannot_alias_another_quantization() {
        let linear = fixture(3, false, false, 4, false);
        let path = linear.dir.join("model.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let spec = &mut header["__metadata__"]["language_model.lm_head.spec"];
        let mut changed: serde_json::Value =
            serde_json::from_str(spec.as_str().unwrap()).unwrap();
        // Both descriptions have the same packed width and affine-group
        // count. Only the architecture's fixed group policy rejects this
        // metadata-only reinterpretation of Q4/group64 as Q8/group32.
        changed["bits"] = serde_json::json!(8);
        changed["group_size"] = serde_json::json!(32);
        *spec = serde_json::Value::String(changed.to_string());
        let mut json = serde_json::to_vec(&header).unwrap();
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let mut rewritten = (json.len() as u64).to_le_bytes().to_vec();
        rewritten.extend(json);
        rewritten.extend_from_slice(&bytes[8 + header_len..]);
        std::fs::write(path, rewritten).unwrap();
        assert!(Checkpoint::open_uzu(&linear.dir, &linear.bindings).is_err());

        // A self-consistent Q8/group32 ngram description also disagrees with
        // the only supported paged embedding policy, Q4/group32.
        let ngram = fixture(3, false, true, 8, true);
        assert!(Checkpoint::open_uzu(&ngram.dir, &ngram.bindings).is_err());
    }

    #[test]
    fn duplicate_nested_tensor_fields_and_metadata_are_rejected() {
        let fixture = fixture(3, false, false, 4, false);
        let path = fixture.dir.join("model.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let original = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
        for changed in [
            original.replacen("\"dtype\":\"U8\"", "\"dtype\":\"F32\",\"dtype\":\"U8\"", 1),
            original.replacen("\"data_offsets\":", "\"data_offsets\":[0,0],\"data_offsets\":", 1),
            original.replacen("\"stark.qwen38.schema_version\":", "\"stark.qwen38.schema_version\":\"0\",\"stark.qwen38.schema_version\":", 1),
        ] {
            assert_ne!(changed, original);
            let mut header = changed.into_bytes();
            while header.len() % 8 != 0 { header.push(b' '); }
            let mut rewritten = (header.len() as u64).to_le_bytes().to_vec();
            rewritten.extend(header);
            rewritten.extend_from_slice(&bytes[8 + header_len..]);
            std::fs::write(&path, rewritten).unwrap();
            assert!(Checkpoint::open_uzu(&fixture.dir, &fixture.bindings).is_err());
        }
    }
}
