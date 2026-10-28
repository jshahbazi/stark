//! Versioned structural admission for merged BF16 Qwen3.5 families.
//!
//! This profile describes implemented operations and geometry. Its hashes bind
//! the selected files; neither source names nor changed weights grant reference
//! parity or hardware qualification.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
};

use serde_json::{Value, json};

use super::{ImageTokenIds, VisionTowerConfig};
use crate::backends::common::InvalidInput;

mod dynamic;

pub(super) const PROFILE: &str = "qwen35-dense-bf16-merged-v1";
pub(super) const VERSION: u32 = 1;
pub(super) const DYNAMIC_PROFILE: &str = "qwen35-dense-bf16-merged-v2";
pub(super) const DYNAMIC_VERSION: u32 = 2;
type TensorMap = BTreeMap<String, (&'static str, Vec<u32>)>;

pub(super) struct TensorContract {
    tensors: TensorMap,
}

#[derive(Clone, Copy)]
pub(super) enum FamilyProfile {
    FixedV1,
    DynamicV2,
}

impl FamilyProfile {
    pub fn parse(value: &Value) -> Option<Self> {
        if *value == json!({"profile":PROFILE,"version":VERSION}) {
            Some(Self::FixedV1)
        } else if *value == json!({"profile":DYNAMIC_PROFILE,"version":DYNAMIC_VERSION}) {
            Some(Self::DynamicV2)
        } else {
            None
        }
    }

    pub fn identity(self) -> (&'static str, u32) {
        match self {
            Self::FixedV1 => (PROFILE, VERSION),
            Self::DynamicV2 => (DYNAMIC_PROFILE, DYNAMIC_VERSION),
        }
    }

    pub fn max_weight_bytes(self) -> u64 {
        match self {
            Self::FixedV1 => 2_000_000_000,
            Self::DynamicV2 => dynamic::MAX_WEIGHT_BYTES,
        }
    }

    pub fn accepts_tokenizer(
        self,
        digest: &str,
    ) -> bool {
        digest == super::ARTIFACT_FILES[2].1
            || matches!(self, Self::DynamicV2)
                && digest == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    }
}
const MAX_HEADER_BYTES: u64 = 1_048_576;
const COMPONENTS: [&str; 4] =
    ["model.safetensors", "tokenizer.json", "preprocessor_config.json", "chat_template.jinja"];
// Reviewed Qwen templates differ only in the mapping guard/iteration of tool
// call arguments. Both preserve the same image placeholder program.
const TEMPLATE_HASHES: [&str; 2] =
    [super::ARTIFACT_FILES[4].1, "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67"];
const DYNAMIC_STOCK_TEMPLATE: &str = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715";

pub(super) struct FamilyAdmission {
    pub tower: VisionTowerConfig,
    pub tokens: ImageTokenIds,
    pub source: String,
    pub revision: Option<String>,
    pub profile: FamilyProfile,
    pub tensors: Option<TensorContract>,
}

fn invalid() -> InvalidInput {
    InvalidInput("unsupported native vision family contract")
}

pub(super) fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn exact_keys(
    value: &Value,
    keys: &[&str],
) -> bool {
    value.as_object().is_some_and(|map| map.len() == keys.len() && keys.iter().all(|key| map.contains_key(*key)))
}

fn norm(
    offset: Value,
    upcast: &str,
) -> Value {
    json!({"epsilon":1e-6,"scale_offset":offset,"upcast_mode":upcast,"subtract_mean":false,"has_scale":true,"has_biases":false})
}

/// Exact semantic geometry for this profile, independent of checkpoint bytes.
pub(super) fn decoder_profile() -> Value {
    let norm = norm(json!(1.0), "full_layer");
    let gated_norm = self::norm(Value::Null, "only_normalization");
    let mut layers = Vec::with_capacity(24);
    for index in 0..24 {
        let attention = index % 4 == 3;
        let mixer = if attention {
            json!({
                "type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},
                "query_norm_config":norm,"key_norm_config":norm,"logit_soft_cap":null,"has_sinks":false,
                "has_qkvg_biases":false,"has_out_biases":false,"num_heads":8,"num_groups":2,"head_dim":256,
                "is_causal":true,"scale":null,"sliding_window_size":null,"has_gate":true,"normalize_values":false,
                "is_kv_sharing":false,"reference_bf16":true
            })
        } else {
            json!({
                "type":"DeltaNetConfig","in_proj_config":{},"conv_config":{"has_biases":false},"out_proj_config":{},
                "norm_config":gated_norm,"num_heads":16,"num_groups":16,"head_dim":128,"value_head_dim":128,
                "kernel_size":4,"reference_bf16":true
            })
        };
        layers.push(json!({
            "pre_mixer_norm_config":norm,"mixer_conv_config":null,"mixer_config":mixer,"post_mixer_norm_config":null,
            "pre_mlp_norm_config":norm,"mlp_conv_config":null,
            "mlp_config":{"type":"DenseMLPConfig","linear_config":{},"activation":{"type":"SiLU","alpha":1.0},
                "has_up_biases":false,"has_down_biases":false,"up_clipping":null,"gate_clipping":null},
            "post_mlp_norm_config":null,"hidden_dim":null,"ple_config":null,"has_post_layer_scalar":false,
            "kv_source_layer_index":null,
            "rope_config":if attention {json!({"type":"Qwen35MRoPEConfig","base":10_000_000.0,
                "max_sequence_length":262144,"head_dim":64,"mrope_section":[11,11,10],"interleaved":true})} else {Value::Null}
        }));
    }
    json!({
        "embedding_config":{"type":"TiedEmbeddingConfig","input_scale":null,"logit_soft_cap":null,"logit_scale":null},
        "transformer_config":{"layer_configs":layers,"output_norm_config":norm,"model_dim":1024,"hidden_dim":3584},
        "vocab_size":248320,"ple_model_config":null,"embedding_norm_config":null
    })
}

pub(super) fn processor_profile() -> Value {
    json!({
        "size":{"longest_edge":16777216,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"processor_class":"Qwen3VLProcessor",
        "image_processor_type":"Qwen2VLImageProcessorFast"
    })
}

fn vision_profile() -> Value {
    json!({
        "deepstack_visual_indexes":[],"depth":12,"hidden_act":"gelu_pytorch_tanh","hidden_size":768,"in_channels":3,
        "initializer_range":0.02,"intermediate_size":3072,"model_type":"qwen3_5","num_heads":12,
        "num_position_embeddings":2304,"out_hidden_size":1024,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2
    })
}

fn vision_semantics(value: &Value) -> bool {
    let Some(mut actual) = value.as_object().cloned() else {
        return false;
    };
    if let Some(initializer) = actual.remove("initializer_range")
        && !initializer.as_f64().is_some_and(|number| number.is_finite() && number >= 0.0)
    {
        return false;
    }
    let mut expected = vision_profile().as_object().cloned().expect("profile is an object");
    expected.remove("initializer_range");
    actual == expected
}

fn logical_source(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && !value.starts_with('/')
        && !value.contains('\\')
        && !value.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        && value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-/:".contains(&byte))
}

fn source_valid(
    value: &Value,
    profile: FamilyProfile,
) -> bool {
    if !exact_keys(value, &["repo", "revision", "hashes", "weight_files"])
        || !value["repo"].as_str().is_some_and(logical_source)
        || !(value["revision"].is_null()
            || value["revision"].as_str().is_some_and(|revision| {
                !revision.is_empty()
                    && revision.len() <= 128
                    && revision.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
            }))
    {
        return false;
    }
    let Some(hashes) = value["hashes"].as_object() else {
        return false;
    };
    let Some(weights) = value["weight_files"].as_object() else {
        return false;
    };
    let required = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"];
    let processors = usize::from(hashes.contains_key("preprocessor_config.json"))
        + usize::from(hashes.contains_key("processor_config.json"));
    let processor_count_supported = match profile {
        FamilyProfile::FixedV1 => processors == 1,
        FamilyProfile::DynamicV2 => (1..=2).contains(&processors),
    };
    hashes.len() == required.len() + processors + usize::from(hashes.contains_key("generation_config.json"))
        && required.iter().all(|name| hashes.contains_key(*name))
        && processor_count_supported
        && !weights.is_empty()
        && weights.len() <= 256
        && weights.values().any(|weight| weight["role"] == "weights")
        && hashes
            .iter()
            .all(|(name, hash)| logical_source(name) && !name.contains('/') && hash.as_str().is_some_and(valid_hash))
        && weights.iter().all(|(name, weight)| {
            logical_source(name)
                && exact_keys(weight, &["role", "size", "sha256"])
                && matches!(weight["role"].as_str(), Some("weights" | "weight_index"))
                && weight["size"].as_u64().is_some_and(|size| size > 0)
                && weight["sha256"].as_str().is_some_and(valid_hash)
        })
}

pub(super) fn admit(
    config: &Value,
    hashes: &BTreeMap<&str, String>,
    processor_bytes: &[u8],
    template_bytes: &[u8],
    tokenizer: &tokenizers::Tokenizer,
) -> Result<FamilyAdmission, InvalidInput> {
    let contract = &config["artifact_contract"];
    let profile = FamilyProfile::parse(&contract["family"]).ok_or_else(invalid)?;
    if !exact_keys(
        contract,
        &[
            "type",
            "schema_version",
            "input_modalities",
            "dtype",
            "source",
            "special_token_ids",
            "vision_config",
            "processor_config",
            "position_contract",
            "tensor_layout",
            "transport",
            "qualification",
            "runtime_supported",
            "family",
            "component_hashes",
        ],
    ) || contract["type"] != "Qwen35VisionLanguageArtifact"
        || contract["schema_version"] != 3
        || contract["input_modalities"] != json!(["text", "image"])
        || contract["dtype"] != "bfloat16"
        || contract["tensor_layout"] != "hf-qwen35-vision-v1"
        || contract["transport"] != "mediaref-v1"
        || contract["qualification"] != "unverified"
        || contract["runtime_supported"] != true
        || config["type"] != "Qwen35VisionLanguageModelConfig"
        || !source_valid(&contract["source"], profile)
        || contract["processor_config"] != processor_profile()
        || contract["position_contract"] != json!({"mrope_section":[11,11,10],"interleaved":true})
        || !exact_keys(&contract["component_hashes"], &COMPONENTS)
    {
        return Err(invalid());
    }
    let (tower, vocab, tensors) = match profile {
        FamilyProfile::FixedV1 => {
            if config["decoder_config"] != decoder_profile() || !vision_semantics(&contract["vision_config"]) {
                return Err(invalid());
            }
            (
                super::official_tower_config(),
                248320,
                TensorContract {
                    tensors: expected_tensors(),
                },
            )
        },
        FamilyProfile::DynamicV2 => {
            let geometry = dynamic::geometry(&config["decoder_config"], &contract["vision_config"])?;
            (geometry.tower, geometry.vocab, geometry.tensors)
        },
    };
    for name in COMPONENTS {
        let digest = contract["component_hashes"][name].as_str().ok_or_else(invalid)?;
        if !valid_hash(digest) || hashes.get(name).map(String::as_str) != Some(digest) {
            return Err(InvalidInput("native vision family component identity mismatch"));
        }
    }
    // The first family varies trained weights while retaining the reviewed
    // tokenizer processing and bounded, reviewed template programs.
    if !hashes.get("tokenizer.json").is_some_and(|digest| profile.accepts_tokenizer(digest))
        || !hashes.get("chat_template.jinja").is_some_and(|digest| {
            TEMPLATE_HASHES.contains(&digest.as_str())
                || matches!(profile, FamilyProfile::DynamicV2) && digest == DYNAMIC_STOCK_TEMPLATE
        })
    {
        return Err(InvalidInput("unsupported native vision family text processing"));
    }
    let processor: Value = serde_json::from_slice(processor_bytes).map_err(|_| invalid())?;
    let template = std::str::from_utf8(template_bytes).map_err(|_| invalid())?;
    if processor != processor_profile()
        || config["token_codec_config"]["type"] != "ChatCodecConfig"
        || config["token_codec_config"]["prompt_template"].as_str() != Some(template)
    {
        return Err(invalid());
    }
    let token_map = &contract["special_token_ids"];
    if !exact_keys(token_map, &["vision_start", "vision_end", "image_pad", "video_pad"]) {
        return Err(invalid());
    }
    let mut ids = Vec::with_capacity(4);
    for key in ["vision_start", "vision_end", "image_pad", "video_pad"] {
        let id = token_map[key].as_u64().filter(|id| *id < u64::from(vocab)).ok_or_else(invalid)?;
        if ids.contains(&id) || tokenizer.token_to_id(&format!("<|{key}|>")) != Some(id as u32) {
            return Err(InvalidInput("native vision family token contract mismatch"));
        }
        let encoded = tokenizer.encode(format!("<|{key}|>"), false).map_err(|_| invalid())?;
        if encoded.get_ids() != [id as u32] {
            return Err(InvalidInput("native vision family special token is not atomic"));
        }
        ids.push(id);
    }
    let vocabulary = tokenizer.get_vocab(true);
    // The supported decoder has padded output rows. Tokenizer coverage is
    // smaller than that matrix, and must stay unique and entirely in range.
    if vocabulary.is_empty()
        || vocabulary.len() > vocab as usize
        || vocabulary.values().any(|id| *id >= vocab)
        || vocabulary.values().copied().collect::<std::collections::HashSet<_>>().len() != vocabulary.len()
    {
        return Err(InvalidInput("native vision family vocabulary differs from decoder"));
    }
    Ok(FamilyAdmission {
        tower,
        tokens: ImageTokenIds {
            vision_start: ids[0],
            vision_end: ids[1],
            image: ids[2],
            video: ids[3],
        },
        source: contract["source"]["repo"].as_str().ok_or_else(invalid)?.into(),
        revision: contract["source"]["revision"].as_str().map(str::to_owned),
        profile,
        tensors: Some(tensors),
    })
}

/// Header-only complete tensor and storage validation precedes GPU weights.
/// The selected profile supplies the complete validated tensor inventory.
pub(super) fn validate_weights(
    file: &mut File,
    admission: &FamilyAdmission,
) -> Result<(), InvalidInput> {
    file.rewind().map_err(|_| invalid())?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size).map_err(|_| invalid())?;
    let size = u64::from_le_bytes(size);
    if size == 0 || size > MAX_HEADER_BYTES {
        return Err(invalid());
    }
    let mut bytes = vec![0; size as usize];
    file.read_exact(&mut bytes).map_err(|_| invalid())?;
    let header: Value = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    let tensors = &admission.tensors.as_ref().ok_or_else(invalid)?.tensors;
    validate_tensor_header(&header, file.metadata().map_err(|_| invalid())?.len(), size + 8, tensors)?;
    file.rewind().map_err(|_| invalid())?;
    Ok(())
}

fn expected_tensors() -> BTreeMap<String, (&'static str, Vec<u32>)> {
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, dtype, shape: &[u32]| {
        tensors.insert(name, (dtype, shape.to_vec()));
    };
    add("decoder.embedding.embedding.weights".into(), "BF16", &[248320, 1024]);
    add("decoder.transformer.output_norm.scales".into(), "F32", &[1024]);
    for layer in 0..24 {
        let prefix = format!("decoder.transformer.layers.{layer}");
        for name in ["pre_mixer_norm.scales", "pre_mlp_norm.scales"] {
            add(format!("{prefix}.{name}"), "F32", &[1024]);
        }
        add(format!("{prefix}.mlp.up_projection.weights.weights"), "BF16", &[7168, 1024]);
        add(format!("{prefix}.mlp.down_projection.weights.weights"), "BF16", &[1024, 3584]);
        if layer % 4 == 3 {
            for name in ["query_norm", "key_norm"] {
                add(format!("{prefix}.mixer.{name}.scales"), "F32", &[256]);
            }
            add(format!("{prefix}.mixer.qkvg_projection.weights.weights"), "BF16", &[5120, 1024]);
            add(format!("{prefix}.mixer.out_projection.weights.weights"), "BF16", &[1024, 2048]);
        } else {
            for name in ["a_log", "dt_bias"] {
                add(format!("{prefix}.mixer.{name}"), "F32", &[16]);
            }
            add(format!("{prefix}.mixer.conv.weights"), "F32", &[6144, 4]);
            add(format!("{prefix}.mixer.norm.scales"), "F32", &[128]);
            add(format!("{prefix}.mixer.in_proj.weights.weights"), "BF16", &[8224, 1024]);
            add(format!("{prefix}.mixer.out_proj.weights.weights"), "BF16", &[1024, 2048]);
        }
    }
    add("vision.patch_embed.proj.weight".into(), "BF16", &[768, 3, 2, 16, 16]);
    add("vision.patch_embed.proj.bias".into(), "BF16", &[768]);
    add("vision.pos_embed.weight".into(), "BF16", &[2304, 768]);
    for layer in 0..12 {
        let prefix = format!("vision.blocks.{layer}");
        for norm in ["norm1", "norm2"] {
            for suffix in ["weight", "bias"] {
                add(format!("{prefix}.{norm}.{suffix}"), "BF16", &[768]);
            }
        }
        for (name, output, input) in [
            ("attn.qkv", 2304, 768),
            ("attn.proj", 768, 768),
            ("mlp.linear_fc1", 3072, 768),
            ("mlp.linear_fc2", 768, 3072),
        ] {
            add(format!("{prefix}.{name}.weight"), "BF16", &[output, input]);
            add(format!("{prefix}.{name}.bias"), "BF16", &[output]);
        }
    }
    for suffix in ["weight", "bias"] {
        add(format!("vision.merger.norm.{suffix}"), "BF16", &[768]);
    }
    for (name, output, input) in [("linear_fc1", 3072, 3072), ("linear_fc2", 1024, 3072)] {
        add(format!("vision.merger.{name}.weight"), "BF16", &[output, input]);
        add(format!("vision.merger.{name}.bias"), "BF16", &[output]);
    }
    tensors
}

#[cfg(test)]
fn validate_header(
    header: &Value,
    file_length: u64,
    data_start: u64,
) -> Result<(), InvalidInput> {
    let tensors = expected_tensors();
    validate_tensor_header(header, file_length, data_start, &tensors)
}

fn validate_tensor_header(
    header: &Value,
    file_length: u64,
    data_start: u64,
    tensors: &TensorMap,
) -> Result<(), InvalidInput> {
    let entries = header.as_object().ok_or_else(invalid)?;
    if entries.len() != tensors.len() + 1 {
        return Err(invalid());
    }
    let metadata = entries.get("__metadata__").and_then(Value::as_object).ok_or_else(invalid)?;
    let mut specs = BTreeMap::new();
    let mut spans = Vec::with_capacity(tensors.len());
    for (name, (dtype, shape)) in tensors {
        let entry = entries.get(name).ok_or_else(invalid)?;
        if !exact_keys(entry, &["dtype", "shape", "data_offsets"])
            || entry["dtype"] != *dtype
            || entry["shape"] != json!(shape)
        {
            return Err(invalid());
        }
        let offsets = entry["data_offsets"].as_array().filter(|items| items.len() == 2).ok_or_else(invalid)?;
        let begin = offsets[0].as_u64().ok_or_else(invalid)?;
        let end = offsets[1].as_u64().ok_or_else(invalid)?;
        let bytes = shape
            .iter()
            .try_fold(
                if *dtype == "F32" {
                    4u64
                } else {
                    2
                },
                |count, width| count.checked_mul(u64::from(*width)),
            )
            .ok_or_else(invalid)?;
        if end.checked_sub(begin) != Some(bytes) {
            return Err(invalid());
        }
        spans.push((begin, end));
        if let Some(prefix) = name.strip_suffix(".weights")
            && *dtype == "BF16"
            && prefix.starts_with("decoder.")
        {
            specs.insert(format!("{prefix}.spec"), json!({"type":"FullPrecisionSpec","layout":if matches!(prefix, "decoder.embedding.embedding" | "decoder.embedding.input_embedding") {"input_output"} else {"output_input"}}));
        }
    }
    if metadata.len() != specs.len() {
        return Err(invalid());
    }
    for (name, expected) in specs {
        let actual = metadata.get(&name).and_then(Value::as_str).ok_or_else(invalid)?;
        if serde_json::from_str::<Value>(actual).map_err(|_| invalid())? != expected {
            return Err(invalid());
        }
    }
    spans.sort_unstable();
    let mut end = 0;
    for (begin, next) in spans {
        if begin != end {
            return Err(invalid());
        }
        end = next;
    }
    if data_start.checked_add(end) != Some(file_length) {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn family_geometry_has_complete_bounded_decoder_and_tensor_contract() {
        let decoder = decoder_profile();
        assert_eq!(decoder["transformer_config"]["layer_configs"].as_array().unwrap().len(), 24);
        assert_eq!(expected_tensors().len(), 383);
        assert_eq!(processor_profile()["merge_size"], 2);
    }

    fn valid_header() -> (Value, u64) {
        valid_header_for(&expected_tensors())
    }

    fn valid_header_for(tensors: &TensorMap) -> (Value, u64) {
        let mut header = serde_json::Map::new();
        let mut metadata = serde_json::Map::new();
        let mut offset = 0;
        for (name, (dtype, shape)) in tensors {
            let size = shape.iter().fold(
                if *dtype == "F32" {
                    4u64
                } else {
                    2
                },
                |size, width| size * u64::from(*width),
            );
            header.insert(name.clone(), json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+size]}));
            if let Some(prefix) = name.strip_suffix(".weights") {
                if *dtype == "BF16" && prefix.starts_with("decoder.") {
                    metadata.insert(format!("{prefix}.spec"), Value::String(json!({"type":"FullPrecisionSpec","layout":if matches!(prefix, "decoder.embedding.embedding" | "decoder.embedding.input_embedding") {"input_output"} else {"output_input"}}).to_string()));
                }
            }
            offset += size;
        }
        header.insert("__metadata__".into(), Value::Object(metadata));
        (Value::Object(header), offset)
    }

    #[stark_test]
    fn family_header_validates_all_shapes_dtypes_specs_and_storage_before_allocating() {
        let (header, bytes) = valid_header();
        assert!(validate_header(&header, bytes + 8, 8).is_ok());
        for (pointer, value) in [
            ("/vision.pos_embed.weight/dtype", json!("F32")),
            ("/vision.pos_embed.weight/shape", json!([2305, 768])),
            ("/vision.pos_embed.weight/data_offsets", json!([1, 2])),
            ("/__metadata__/decoder.embedding.embedding.spec", json!("{\"type\":\"MLXSpec\"}")),
        ] {
            let mut changed = header.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(validate_header(&changed, bytes + 8, 8).is_err());
        }
        let mut changed = header.clone();
        changed.as_object_mut().unwrap().insert("unexpected".into(), json!({}));
        assert!(validate_header(&changed, bytes + 8, 8).is_err());
        assert!(validate_header(&header, bytes + 7, 8).is_err());
        assert!(validate_header(&header, bytes + 9, 8).is_err());
    }

    #[stark_test]
    fn dynamic_family_binds_untied_header_geometry_processing_and_dual_processor_provenance() {
        let tokenizer = tokenizer();
        let mut hashes: BTreeMap<&str, String> = COMPONENTS.into_iter().map(|name| (name, "a".repeat(64))).collect();
        hashes.insert("tokenizer.json", super::super::ARTIFACT_FILES[2].1.into());
        hashes.insert("chat_template.jinja", DYNAMIC_STOCK_TEMPLATE.into());
        let mut config = config(&hashes);
        config["artifact_contract"]["family"] = json!({"profile":DYNAMIC_PROFILE,"version":DYNAMIC_VERSION});
        config["artifact_contract"]["source"]["hashes"]["processor_config.json"] = json!("f".repeat(64));
        config["decoder_config"]["embedding_config"]["type"] = json!("UntiedEmbeddingConfig");
        config["decoder_config"]["transformer_config"]["model_dim"] = json!(4096);
        config["decoder_config"]["transformer_config"]["hidden_dim"] = json!(12288);
        config["artifact_contract"]["vision_config"]["out_hidden_size"] = json!(4096);
        config["artifact_contract"]["vision_config"]["hidden_size"] = json!(1152);
        config["artifact_contract"]["vision_config"]["intermediate_size"] = json!(4304);
        config["artifact_contract"]["vision_config"]["num_heads"] = json!(16);
        let processor = serde_json::to_vec(&processor_profile()).unwrap();
        let admitted = admit(&config, &hashes, &processor, b"synthetic-template", &tokenizer).unwrap();
        assert_eq!(admitted.profile.identity(), (DYNAMIC_PROFILE, DYNAMIC_VERSION));
        let tensors = &admitted.tensors.as_ref().unwrap().tensors;
        let (mut header, bytes) = valid_header_for(tensors);
        assert!(validate_tensor_header(&header, bytes + 8, 8, tensors).is_ok());
        let alternate_tokenizer = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
        assert!(!FamilyProfile::FixedV1.accepts_tokenizer(alternate_tokenizer));
        assert!(!FamilyProfile::DynamicV2.accepts_tokenizer(&"f".repeat(64)));
        let mut alternate_hashes = hashes.clone();
        alternate_hashes.insert("tokenizer.json", alternate_tokenizer.into());
        let mut alternate_config = config.clone();
        alternate_config["artifact_contract"]["component_hashes"] = json!(alternate_hashes);
        assert!(admit(&alternate_config, &alternate_hashes, &processor, b"synthetic-template", &tokenizer).is_ok());
        header["__metadata__"]["decoder.embedding.output_embedding.spec"] =
            json!(r#"{"type":"FullPrecisionSpec","layout":"input_output"}"#);
        assert!(validate_tensor_header(&header, bytes + 8, 8, tensors).is_err());
        config["artifact_contract"]["family"] = json!({"profile":PROFILE,"version":VERSION});
        assert!(admit(&config, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
    }

    #[stark_test]
    fn family_source_identity_is_logical_bounded_metadata() {
        for name in ["owner/fine-tune", "local:0123456789abcdef"] {
            assert!(logical_source(name));
        }
        for name in ["", "/private/model", "../model", "owner/../model", "owner\\model", "owner/model\n"] {
            assert!(!logical_source(name));
        }
        assert!(valid_hash(&"a".repeat(64)));
        assert!(!valid_hash(&"A".repeat(64)));

        let mut source = config(&BTreeMap::new())["artifact_contract"]["source"].clone();
        let weights = source["weight_files"].as_object_mut().unwrap();
        let weight = weights.remove("model.safetensors").unwrap();
        weights.insert("weights/shard-00001.safetensors".into(), weight.clone());
        assert!(source_valid(&source, FamilyProfile::FixedV1));
        for name in [
            "/weights/shard.safetensors",
            "weights//shard.safetensors",
            "weights/./shard.safetensors",
            "weights/../shard.safetensors",
            "weights\\shard.safetensors",
        ] {
            source["weight_files"] = Value::Object([(name.to_owned(), weight.clone())].into_iter().collect());
            assert!(!source_valid(&source, FamilyProfile::FixedV1));
        }
    }

    fn tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::{AddedToken, models::wordlevel::WordLevel};
        let names = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"];
        let ids = [248053, 248054, 248056, 248057];
        let model = WordLevel::builder()
            .vocab(
                (0..248320u32)
                    .map(|id| {
                        (
                            if let Some(index) = ids.iter().position(|token| *token == id) {
                                names[index].to_owned()
                            } else {
                                format!("token{id}")
                            },
                            id,
                        )
                    })
                    .collect(),
            )
            .unk_token("token0".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.add_special_tokens(&names.map(|name| AddedToken::from(name, true)));
        tokenizer
    }

    fn config(hashes: &BTreeMap<&str, String>) -> Value {
        json!({
            "type":"Qwen35VisionLanguageModelConfig","decoder_config":decoder_profile(),
            "token_codec_config":{"type":"ChatCodecConfig","prompt_template":"synthetic-template"},
            "artifact_contract":{
                "type":"Qwen35VisionLanguageArtifact","schema_version":3,"input_modalities":["text","image"],"dtype":"bfloat16",
                "source":{"repo":"local:synthetic-merged-model","revision":null,
                    "hashes":{"config.json":"a".repeat(64),"tokenizer.json":"a".repeat(64),"tokenizer_config.json":"a".repeat(64),"preprocessor_config.json":"a".repeat(64),"chat_template.jinja":"a".repeat(64)},"weight_files":{"model.safetensors":{"role":"weights","size":1,"sha256":"b".repeat(64)}}},
                "special_token_ids":{"vision_start":248053,"vision_end":248054,"image_pad":248056,"video_pad":248057},
                "vision_config":vision_profile(),"processor_config":processor_profile(),
                "position_contract":{"mrope_section":[11,11,10],"interleaved":true},
                "tensor_layout":"hf-qwen35-vision-v1","transport":"mediaref-v1","qualification":"unverified","runtime_supported":true,
                "family":{"profile":PROFILE,"version":VERSION},"component_hashes":hashes
            }
        })
    }

    #[stark_test]
    fn family_admits_changed_weight_identities_with_reviewed_processing() {
        let tokenizer = tokenizer();
        let mut hashes: BTreeMap<&str, String> = COMPONENTS.into_iter().map(|name| (name, "a".repeat(64))).collect();
        hashes.insert("tokenizer.json", super::super::ARTIFACT_FILES[2].1.into());
        hashes.insert("chat_template.jinja", super::super::ARTIFACT_FILES[4].1.into());
        let processor = serde_json::to_vec(&processor_profile()).unwrap();
        for digest in ["b", "c"] {
            hashes.insert("model.safetensors", digest.repeat(64));
            let admitted = admit(&config(&hashes), &hashes, &processor, b"synthetic-template", &tokenizer).unwrap();
            assert_eq!(admitted.tokens.image, 248056);
            assert_eq!(admitted.tower.out_hidden_size, 1024);
            assert_eq!(admitted.source, "local:synthetic-merged-model");
            assert_eq!(admitted.revision, None);
        }
        let good = config(&hashes);
        let mut optional_generation = good.clone();
        optional_generation["artifact_contract"]["source"]["hashes"]["generation_config.json"] = json!("f".repeat(64));
        assert!(admit(&optional_generation, &hashes, &processor, b"synthetic-template", &tokenizer).is_ok());
        for name in ["preprocessor_config.json", "processor_config.json"] {
            let mut alternate_processor = good.clone();
            let provenance = alternate_processor["artifact_contract"]["source"]["hashes"].as_object_mut().unwrap();
            let digest = provenance.remove("preprocessor_config.json").unwrap();
            provenance.insert(name.into(), digest);
            assert!(admit(&alternate_processor, &hashes, &processor, b"synthetic-template", &tokenizer).is_ok());
            alternate_processor["artifact_contract"]["source"]["hashes"]["generation_config.json"] =
                json!("f".repeat(64));
            assert!(admit(&alternate_processor, &hashes, &processor, b"synthetic-template", &tokenizer).is_ok());
        }
        for template_hash in TEMPLATE_HASHES {
            let mut reviewed_hashes = hashes.clone();
            reviewed_hashes.insert("chat_template.jinja", template_hash.into());
            assert!(
                admit(&config(&reviewed_hashes), &reviewed_hashes, &processor, b"synthetic-template", &tokenizer)
                    .is_ok()
            );
        }
        let mut ambiguous_processor = good.clone();
        ambiguous_processor["artifact_contract"]["source"]["hashes"]["processor_config.json"] = json!("f".repeat(64));
        assert!(admit(&ambiguous_processor, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
        let mut absent_processor = good.clone();
        absent_processor["artifact_contract"]["source"]["hashes"]
            .as_object_mut()
            .unwrap()
            .remove("preprocessor_config.json");
        absent_processor["artifact_contract"]["source"]["hashes"]["unrecognized.json"] = json!("f".repeat(64));
        assert!(admit(&absent_processor, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
        let mut missing_provenance = good.clone();
        missing_provenance["artifact_contract"]["source"]["hashes"].as_object_mut().unwrap().remove("config.json");
        assert!(admit(&missing_provenance, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
        let mut unknown_provenance = good.clone();
        unknown_provenance["artifact_contract"]["source"]["hashes"]["unrecognized.json"] = json!("f".repeat(64));
        assert!(admit(&unknown_provenance, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
        // Input hashes come from selected file descriptors in the production
        // loader. This synthetic tokenizer isolates structural token checks.
        for name in ["tokenizer.json", "chat_template.jinja"] {
            let mut changed_hashes = hashes.clone();
            changed_hashes.insert(name, "f".repeat(64));
            assert!(
                admit(&config(&changed_hashes), &changed_hashes, &processor, b"synthetic-template", &tokenizer)
                    .is_err()
            );
        }
        let mut training_only = good.clone();
        training_only["artifact_contract"]["vision_config"]["initializer_range"] = json!(0.01);
        assert!(admit(&training_only, &hashes, &processor, b"synthetic-template", &tokenizer).is_ok());
        training_only["artifact_contract"]["vision_config"].as_object_mut().unwrap().remove("initializer_range");
        assert!(admit(&training_only, &hashes, &processor, b"synthetic-template", &tokenizer).is_ok());
        for (pointer, value) in [
            ("/artifact_contract/schema_version", json!(4)),
            ("/artifact_contract/family/version", json!(2)),
            ("/artifact_contract/family/profile", json!("unknown")),
            ("/artifact_contract/special_token_ids/image_pad", json!(2)),
            ("/artifact_contract/special_token_ids/image_pad", json!(248320)),
            ("/artifact_contract/source/repo", json!("/private/secret-model")),
            ("/artifact_contract/component_hashes/model.safetensors", json!("f".repeat(64))),
            ("/artifact_contract/vision_config/depth", json!(24)),
            ("/artifact_contract/vision_config/initializer_range", json!(true)),
            ("/artifact_contract/vision_config/initializer_range", json!(-0.1)),
            ("/artifact_contract/processor_config/image_mean", json!([0.0, 0.0, 0.0])),
            ("/decoder_config/transformer_config/model_dim", json!(2048)),
            ("/decoder_config/transformer_config/layer_configs/0/mixer_config/reference_bf16", json!(false)),
            ("/token_codec_config/prompt_template", json!("different-template")),
        ] {
            let mut changed = good.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(admit(&changed, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
        }
        let mut unknown = good.clone();
        unknown["artifact_contract"]["unknown"] = json!(true);
        assert!(admit(&unknown, &hashes, &processor, b"synthetic-template", &tokenizer).is_err());
        assert!(admit(&good, &hashes, b"{}", b"synthetic-template", &tokenizer).is_err());
        assert!(admit(&good, &hashes, &processor, b"different-template", &tokenizer).is_err());
    }
}
