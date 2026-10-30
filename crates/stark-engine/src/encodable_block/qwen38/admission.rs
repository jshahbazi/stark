//! Strict native admission for Stark's Qwen3.8 Uzu storage schema v1.
//! The immutable storage contract and unique-JSON checks originate in Stark
//! revision 5a082f7; this implementation consumes standard ParameterLoader data.

use std::{
    collections::{BTreeMap, HashSet},
    fmt,
};

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, SeqAccess, Visitor},
};

use super::Qwen38Error;
use crate::{
    backends::common::Backend,
    config::model::qwen38_language_model::{
        Qwen38DecoderConfig, Qwen38LayerType, Qwen38TensorDType, Qwen38TensorLayout,
    },
    data_type::DataType,
    parameters::ParameterLoader,
};

pub(super) struct Matrix {
    pub physical: String,
    pub rows: u32,
    pub columns: u32,
    pub bits: u32,
    pub group_size: u32,
    pub embedding: bool,
    pub ngram: bool,
}

struct Plane {
    physical: String,
    logical_shape: Vec<u32>,
    physical_shape: Vec<u32>,
    logical_dtype: Qwen38TensorDType,
    physical_dtype: DataType,
    layout: Qwen38TensorLayout,
    padding: Option<(u32, u32, u32)>,
    hash: Option<Vec<u64>>,
}

struct Inventory {
    planes: BTreeMap<String, Plane>,
    matrices: BTreeMap<String, Matrix>,
}

fn invalid<B: Backend>(message: &'static str) -> Qwen38Error<B> {
    Qwen38Error::Invalid(message)
}

fn canonical(name: &str) -> String {
    let name = if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("language_model.model.{rest}")
    } else if name == "lm_head" || name.starts_with("lm_head.") {
        format!("language_model.{name}")
    } else {
        name.to_owned()
    };
    name.replace(".mlp.experts.", ".mlp.switch_mlp.")
}

fn product<B: Backend>(shape: &[u32]) -> Result<u32, Qwen38Error<B>> {
    if shape.is_empty() || shape.len() > 4 {
        return Err(invalid("invalid Qwen3.8 tensor rank"));
    }
    shape.iter().try_fold(1_u32, |total, &dim| {
        if dim == 0 {
            Err(invalid("empty Qwen3.8 tensor dimension"))
        } else {
            total.checked_mul(dim).ok_or(invalid("Qwen3.8 tensor dimensions overflow"))
        }
    })
}

impl Inventory {
    fn insert<B: Backend>(
        &mut self,
        name: String,
        plane: Plane,
    ) -> Result<(), Qwen38Error<B>> {
        if self.planes.len() >= 16_384 || self.planes.insert(name, plane).is_some() {
            return Err(invalid("duplicate or excessive Qwen3.8 role inventory"));
        }
        Ok(())
    }
    fn matrix<B: Backend>(
        &mut self,
        cfg: &Qwen38DecoderConfig,
        name: &str,
        shape: &[u32],
        group: Option<&str>,
        fixed: bool,
        embedding: bool,
        ngram: bool,
    ) -> Result<(), Qwen38Error<B>> {
        if !(2..=3).contains(&shape.len()) {
            return Err(invalid("invalid Qwen3.8 matrix rank"));
        }
        let columns = *shape.last().ok_or(invalid("missing Qwen3.8 matrix width"))?;
        let rows = product::<B>(&shape[..shape.len() - 1])?;
        let bits = if ngram {
            4
        } else if fixed
            || group.is_some_and(|group| cfg.architecture.quantization.q8_groups.iter().any(|value| value == group))
        {
            8
        } else {
            4
        };
        let group_size = if ngram {
            32
        } else {
            64
        };
        if columns == 0 || !columns.is_multiple_of(group_size) {
            return Err(invalid("invalid Qwen3.8 quantized matrix groups"));
        }
        let groups = columns / group_size;
        let packed = columns.checked_mul(bits).ok_or(invalid("Qwen3.8 packed width overflow"))?;
        let mut logical_codes = shape.to_vec();
        *logical_codes.last_mut().unwrap() = packed / 32;
        let mut logical_params = shape.to_vec();
        *logical_params.last_mut().unwrap() = groups;
        let physical = canonical(name);
        self.insert::<B>(
            format!("{name}.weight"),
            Plane {
                physical: format!("{physical}.weights"),
                logical_shape: logical_codes,
                physical_shape: vec![rows, packed / 8],
                logical_dtype: Qwen38TensorDType::U32,
                physical_dtype: DataType::U8,
                layout: Qwen38TensorLayout::PackedU32,
                padding: None,
                hash: None,
            },
        )?;
        let stride = rows.checked_add(3).ok_or(invalid("Qwen3.8 row padding overflow"))? / 4 * 4;
        for suffix in ["scales", "biases"] {
            self.insert::<B>(
                format!("{name}.{suffix}"),
                Plane {
                    physical: format!("{physical}.{suffix}"),
                    logical_shape: logical_params.clone(),
                    physical_shape: if embedding {
                        vec![rows, groups]
                    } else {
                        vec![groups, stride]
                    },
                    logical_dtype: Qwen38TensorDType::BF16,
                    physical_dtype: DataType::BF16,
                    layout: if embedding {
                        Qwen38TensorLayout::Direct
                    } else {
                        Qwen38TensorLayout::GroupOutput
                    },
                    padding: (!embedding && stride != rows).then_some((rows, groups, stride)),
                    hash: None,
                },
            )?;
        }
        self.matrices.insert(
            name.to_owned(),
            Matrix {
                physical,
                rows,
                columns,
                bits,
                group_size,
                embedding,
                ngram,
            },
        );
        Ok(())
    }
    fn control<B: Backend>(
        &mut self,
        cfg: &Qwen38DecoderConfig,
        name: &str,
        shape: &[u32],
        hash: Option<Vec<u64>>,
    ) -> Result<(), Qwen38Error<B>> {
        product::<B>(shape)?;
        let logical_dtype =
            cfg.tensor_bindings.0.get(name).map(|binding| binding.dtype.clone()).unwrap_or(if hash.is_some() {
                Qwen38TensorDType::U64
            } else {
                Qwen38TensorDType::BF16
            });
        let physical_dtype = match (&logical_dtype, hash.is_some()) {
            (Qwen38TensorDType::BF16, false) => DataType::BF16,
            (Qwen38TensorDType::F32, false) => DataType::F32,
            (Qwen38TensorDType::I64, true) => DataType::I64,
            (Qwen38TensorDType::U64, true) => DataType::U64,
            _ => return Err(invalid("invalid Qwen3.8 control dtype")),
        };
        self.insert::<B>(
            name.to_owned(),
            Plane {
                physical: canonical(name),
                logical_shape: shape.to_vec(),
                physical_shape: shape.to_vec(),
                logical_dtype,
                physical_dtype,
                layout: Qwen38TensorLayout::Direct,
                padding: None,
                hash,
            },
        )
    }
    fn hyper<B: Backend>(
        &mut self,
        cfg: &Qwen38DecoderConfig,
        name: &str,
        inject: bool,
    ) -> Result<(), Qwen38Error<B>> {
        let a = &cfg.architecture;
        let wide = a.hidden_size.checked_mul(a.hc_count).ok_or(invalid("Qwen3.8 HC width overflow"))?;
        self.matrix::<B>(
            cfg,
            &format!("{name}.input_mix_weight_down"),
            &[a.hc_lowrank, wide],
            None,
            true,
            false,
            false,
        )?;
        self.matrix::<B>(cfg, &format!("{name}.input_mix_weight_up"), &[wide, a.hc_lowrank], None, true, false, false)?;
        if inject {
            self.matrix::<B>(
                cfg,
                &format!("{name}.block_inject_weight"),
                &[a.hc_count, wide],
                None,
                true,
                false,
                false,
            )?;
        }
        self.control::<B>(cfg, &format!("{name}.hc_norm.weight"), &[wide], None)
    }
}

fn inventory<B: Backend>(cfg: &Qwen38DecoderConfig) -> Result<Inventory, Qwen38Error<B>> {
    let a = &cfg.architecture;
    let h = a.hidden_size;
    let wide = h.checked_mul(a.hc_count).ok_or(invalid("Qwen3.8 HC width overflow"))?;
    let mut inv = Inventory {
        planes: BTreeMap::new(),
        matrices: BTreeMap::new(),
    };
    let base = "model.language_model";
    inv.matrix::<B>(cfg, &format!("{base}.embed_tokens"), &[a.vocab_size, h], Some("embed"), false, true, false)?;
    inv.matrix::<B>(cfg, "lm_head", &[a.vocab_size, h], Some("head"), false, false, false)?;
    inv.hyper::<B>(cfg, &format!("{base}.hyper_connection_mixer"), false)?;
    for (index, kind) in a.layer_types.iter().enumerate() {
        let stem = format!("{base}.layers.{index}");
        inv.hyper::<B>(cfg, &format!("{stem}.attn_hyper_connection"), true)?;
        inv.hyper::<B>(cfg, &format!("{stem}.mlp_hyper_connection"), true)?;
        match kind {
            Qwen38LayerType::LinearAttention => {
                let p = format!("{stem}.linear_attn");
                let kw = a
                    .linear_num_key_heads
                    .checked_mul(a.linear_key_head_dim)
                    .ok_or(invalid("Qwen3.8 key width overflow"))?;
                let vw = a
                    .linear_num_value_heads
                    .checked_mul(a.linear_value_head_dim)
                    .ok_or(invalid("Qwen3.8 value width overflow"))?;
                let channels = kw
                    .checked_mul(2)
                    .and_then(|width| width.checked_add(vw))
                    .ok_or(invalid("Qwen3.8 GDN width overflow"))?;
                for (name, rows) in [
                    ("in_proj_qkv", channels),
                    ("in_proj_z", vw),
                    ("in_proj_a", a.linear_num_value_heads),
                    ("in_proj_b", a.linear_num_value_heads),
                ] {
                    inv.matrix::<B>(cfg, &format!("{p}.{name}"), &[rows, h], Some("gdn"), false, false, false)?;
                }
                inv.matrix::<B>(cfg, &format!("{p}.out_proj"), &[h, vw], Some("gdn"), false, false, false)?;
                for (name, shape) in [
                    ("conv1d.weight", vec![channels, 1, a.linear_conv_kernel_dim]),
                    ("A_log", vec![a.linear_num_value_heads]),
                    ("dt_bias", vec![a.linear_num_value_heads]),
                    ("norm.weight", vec![a.linear_value_head_dim]),
                ] {
                    inv.control::<B>(cfg, &format!("{p}.{name}"), &shape, None)?;
                }
            },
            Qwen38LayerType::FullAttention => {
                let p = format!("{stem}.self_attn");
                let q =
                    a.num_attention_heads.checked_mul(a.head_dim).ok_or(invalid("Qwen3.8 attention width overflow"))?;
                let kv =
                    a.num_key_value_heads.checked_mul(a.head_dim).ok_or(invalid("Qwen3.8 attention width overflow"))?;
                for (name, rows) in [
                    ("q_proj", q.checked_mul(2).ok_or(invalid("Qwen3.8 gated query overflow"))?),
                    ("k_proj", kv),
                    ("v_proj", kv),
                ] {
                    inv.matrix::<B>(cfg, &format!("{p}.{name}"), &[rows, h], Some("attn"), false, false, false)?;
                }
                inv.matrix::<B>(cfg, &format!("{p}.o_proj"), &[h, q], Some("attn"), false, false, false)?;
                let rows = a
                    .indexer_n_heads
                    .checked_add(1)
                    .and_then(|heads| heads.checked_mul(a.indexer_head_dim))
                    .ok_or(invalid("Qwen3.8 index width overflow"))?;
                inv.matrix::<B>(cfg, &format!("{p}.indexer.index_qk_proj"), &[rows, h], None, true, false, false)?;
                for (name, width) in [
                    ("q_norm", a.head_dim),
                    ("k_norm", a.head_dim),
                    ("indexer.q_layernorm", a.indexer_head_dim),
                    ("indexer.k_layernorm", a.indexer_head_dim),
                ] {
                    inv.control::<B>(cfg, &format!("{p}.{name}.weight"), &[width], None)?;
                }
            },
        }
        let p = format!("{stem}.mlp");
        inv.matrix::<B>(cfg, &format!("{p}.gate"), &[a.num_experts, h], None, true, false, false)?;
        inv.matrix::<B>(cfg, &format!("{p}.shared_expert_gate"), &[1, h], None, true, false, false)?;
        for name in ["gate_proj", "up_proj"] {
            inv.matrix::<B>(
                cfg,
                &format!("{p}.experts.{name}"),
                &[a.num_experts, a.moe_intermediate_size, h],
                None,
                false,
                false,
                false,
            )?;
            inv.matrix::<B>(
                cfg,
                &format!("{p}.shared_expert.{name}"),
                &[a.shared_expert_intermediate_size, h],
                Some("shared"),
                false,
                false,
                false,
            )?;
        }
        inv.matrix::<B>(
            cfg,
            &format!("{p}.experts.down_proj"),
            &[a.num_experts, h, a.moe_intermediate_size],
            None,
            false,
            false,
            false,
        )?;
        inv.matrix::<B>(
            cfg,
            &format!("{p}.shared_expert.down_proj"),
            &[h, a.shared_expert_intermediate_size],
            Some("shared"),
            false,
            false,
            false,
        )?;
        if a.ple_layer_ids.contains(&(index as u32 + 1)) {
            let p = format!("{stem}.ple");
            let pe = a.ple_embed_dim.unwrap_or(h);
            let heads = (a.ngram_size - 1) * a.heads_per_ngram;
            let shard_rows = a.ple.padded_vocab_size / a.split_ngram_parts;
            for shard in 0..a.split_ngram_parts {
                inv.matrix::<B>(
                    cfg,
                    &format!("{p}.ple_embedding.ngram_embedding.shard_{shard}"),
                    &[shard_rows, pe / heads],
                    None,
                    false,
                    true,
                    true,
                )?;
            }
            inv.matrix::<B>(cfg, &format!("{p}.key_proj"), &[wide, pe], None, true, false, false)?;
            inv.matrix::<B>(cfg, &format!("{p}.value_proj"), &[h, pe], None, true, false, false)?;
            for name in ["norm_key", "norm_query", "norm_conv"] {
                inv.control::<B>(cfg, &format!("{p}.{name}.weight"), &[wide], None)?;
            }
            inv.control::<B>(cfg, &format!("{p}.conv1d.weight"), &[wide, 1, a.ple_conv_kernel_size], None)?;
            for (name, values) in [
                ("layer_multipliers", a.ple.layer_multipliers.as_ref()),
                ("ngram_heads_vocab_sizes", a.ple.ngram_heads_vocab_sizes.as_ref()),
                ("ngram_heads_offsets", a.ple.ngram_heads_offsets.as_ref()),
            ] {
                let key = format!("{p}.ple_embedding.{name}");
                if cfg.tensor_bindings.0.contains_key(&key) {
                    inv.control::<B>(cfg, &key, &[values.len() as u32], Some(values.to_vec()))?;
                }
            }
        }
    }
    Ok(inv)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    dtype: String,
    shape: Vec<u32>,
    data_offsets: [u64; 2],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    #[serde(rename = "type")]
    ty: String,
    bits: u32,
    group_size: u32,
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

struct UniqueValue(serde_json::Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = UniqueValue;
            fn expecting(
                &self,
                f: &mut fmt::Formatter,
            ) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: serde::de::Error>(
                self,
                v: bool,
            ) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_i64<E: serde::de::Error>(
                self,
                v: i64,
            ) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(
                self,
                v: u64,
            ) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(
                self,
                v: f64,
            ) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|v| UniqueValue(v.into()))
                    .ok_or_else(|| E::custom("invalid JSON number"))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_string<E: serde::de::Error>(
                self,
                v: String,
            ) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                while let Some(item) = seq.next_element::<UniqueValue>()? {
                    v.push(item.0);
                }
                Ok(UniqueValue(v.into()))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut entries = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, UniqueValue>()? {
                    if key.len() > 512 || entries.len() >= 16_385 || entries.insert(key, value.0).is_some() {
                        return Err(serde::de::Error::custom("duplicate or excessive Qwen header entries"));
                    }
                }
                Ok(UniqueValue(entries.into()))
            }
        }
        deserializer.deserialize_any(V)
    }
}

fn dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::U8 => "U8",
        DataType::BF16 => "BF16",
        DataType::F32 => "F32",
        DataType::I64 => "I64",
        DataType::U64 => "U64",
        _ => "invalid",
    }
}

pub(super) fn admit<B: Backend>(
    loader: &ParameterLoader<B>,
    cfg: &Qwen38DecoderConfig,
) -> Result<BTreeMap<String, Matrix>, Qwen38Error<B>> {
    let expected = inventory::<B>(cfg)?;
    if cfg.tensor_bindings.0.len() != expected.planes.len() {
        return Err(invalid("Qwen3.8 logical inventory mismatch"));
    }
    let (data_start, bytes, file_len) = loader.bounded_header(8 * 1024 * 1024)?;
    if bytes.len() % 8 != 0 {
        return Err(invalid("unaligned Qwen3.8 safetensors header"));
    }
    let unique: UniqueValue =
        serde_json::from_slice(&bytes).map_err(|_| invalid("invalid or duplicate Qwen3.8 header"))?;
    let mut raw = unique.0.as_object().cloned().ok_or(invalid("invalid Qwen3.8 header object"))?;
    let metadata: BTreeMap<String, String> =
        serde_json::from_value(raw.remove("__metadata__").ok_or(invalid("missing Qwen3.8 matrix metadata"))?)
            .map_err(|_| invalid("invalid Qwen3.8 matrix metadata"))?;
    if metadata.get("stark.qwen38.schema_version").map(String::as_str) != Some("1") {
        return Err(invalid("unsupported Qwen3.8 storage schema"));
    }
    let omitted: Omitted = serde_json::from_str(
        metadata.get("stark.qwen38.omitted").ok_or(invalid("missing Qwen3.8 omission provenance"))?,
    )
    .map_err(|_| invalid("invalid Qwen3.8 omission provenance"))?;
    if [omitted.vision, omitted.mtp].iter().any(|part| part.tensors > 16_384 || part.bytes > 1 << 44) {
        return Err(invalid("invalid Qwen3.8 omission bounds"));
    }
    if raw.len() != expected.planes.len() {
        return Err(invalid("Qwen3.8 physical inventory mismatch"));
    }
    let mut physical = BTreeMap::new();
    let mut spans = Vec::new();
    for (name, value) in raw {
        if !name.starts_with("language_model.") || name.len() > 512 {
            return Err(invalid("noncanonical Qwen3.8 physical tensor"));
        }
        let entry: Entry = serde_json::from_value(value).map_err(|_| invalid("invalid Qwen3.8 tensor entry"))?;
        let size = match entry.dtype.as_str() {
            "U8" => 1,
            "BF16" => 2,
            "F32" => 4,
            "I64" | "U64" => 8,
            _ => return Err(invalid("unsupported Qwen3.8 physical dtype")),
        };
        let elements = entry
            .shape
            .iter()
            .try_fold(1_u64, |n, &dim| {
                if dim == 0 {
                    None
                } else {
                    n.checked_mul(dim as u64)
                }
            })
            .ok_or(invalid("Qwen3.8 tensor size overflow"))?;
        let length = elements.checked_mul(size).ok_or(invalid("Qwen3.8 tensor size overflow"))?;
        let [start, end] = entry.data_offsets;
        if entry.shape.is_empty()
            || entry.shape.len() > 4
            || start > end
            || end - start != length
            || end > file_len - data_start
        {
            return Err(invalid("invalid Qwen3.8 tensor span"));
        }
        spans.push((start, end));
        physical.insert(name, entry);
    }
    spans.sort_unstable();
    let mut end = 0;
    for (start, next) in spans {
        if start != end {
            return Err(invalid("Qwen3.8 tensor spans overlap or have gaps"));
        }
        end = next;
    }
    if data_start.checked_add(end) != Some(file_len) {
        return Err(invalid("Qwen3.8 tensor file has trailing data"));
    }
    let mut seen = HashSet::new();
    for (logical, plane) in &expected.planes {
        let binding = cfg.tensor_bindings.0.get(logical).ok_or(invalid("missing Qwen3.8 logical binding"))?;
        if binding.name != plane.physical
            || binding.shape.as_ref() != plane.logical_shape
            || binding.dtype != plane.logical_dtype
            || binding.layout != plane.layout
            || !seen.insert(&binding.name)
        {
            return Err(invalid("Qwen3.8 binding disagrees with architecture role"));
        }
        let entry = physical.get(&plane.physical).ok_or(invalid("missing Qwen3.8 physical plane"))?;
        if entry.shape != plane.physical_shape || entry.dtype != dtype_name(plane.physical_dtype) {
            return Err(invalid("Qwen3.8 physical shape or dtype mismatch"));
        }
        if let Some((rows, groups, stride)) = plane.padding {
            let count = (stride - rows) as usize * 2;
            let mut padding = [0_u8; 6];
            for group in 0..groups {
                let relative = (group as u64 * stride as u64 + rows as u64) * 2;
                loader.read_file_exact_at(data_start + entry.data_offsets[0] + relative, &mut padding[..count])?;
                if padding[..count].iter().any(|&byte| byte != 0) {
                    return Err(invalid("nonzero Qwen3.8 affine padding"));
                }
            }
        }
        if let Some(values) = &plane.hash {
            let tree = loader.tree();
            let leaf = tree.leaf(&plane.physical)?.validate(&plane.physical_shape, plane.physical_dtype)?;
            let region = leaf.file_region()?;
            let mut raw = [0_u8; 8];
            for (index, &value) in values.iter().enumerate() {
                region.read_exact_into(index as u64 * 8, &mut raw)?;
                if u64::from_le_bytes(raw) != value {
                    return Err(invalid("Qwen3.8 integer hash control mismatch"));
                }
            }
        }
    }
    let mut allowed = HashSet::from(["stark.qwen38.schema_version".to_owned(), "stark.qwen38.omitted".to_owned()]);
    for matrix in expected.matrices.values() {
        let key = format!("{}.spec", matrix.physical);
        let spec: Spec = serde_json::from_str(metadata.get(&key).ok_or(invalid("missing Qwen3.8 MLXSpec"))?)
            .map_err(|_| invalid("invalid Qwen3.8 MLXSpec"))?;
        if spec.ty != "MLXSpec"
            || spec.bits != matrix.bits
            || spec.group_size != matrix.group_size
            || spec.layout
                != if matrix.embedding {
                    "input_output"
                } else {
                    "output_input"
                }
        {
            return Err(invalid("Qwen3.8 MLXSpec disagrees with storage policy"));
        }
        allowed.insert(key);
    }
    if metadata.keys().any(|key| !allowed.contains(key)) {
        return Err(invalid("unbound Qwen3.8 matrix metadata"));
    }
    Ok(expected.matrices)
}

#[cfg(test)]
#[path = "weight_tests.rs"]
pub(super) mod tests;
