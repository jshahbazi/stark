//! Exact pinned GLM GGUF decoder and separate vision-sidecar admission.

use std::{collections::BTreeMap, io::Read, path::Path, sync::Arc};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::engine::{
    hybrid::{
        Error, Result,
        image::artifact::{hash, open, read, signature},
        weights::Weights,
    },
    vision::VisionArtifactIdentity,
};

pub(in crate::engine::hybrid) const PROFILE: &str = "glm53-flash-bf16-vision-v1";
const SOURCE: &str = "unsloth/GLM-5.3-Flash-GGUF";
const REVISION: &str = "a38483c8cd5df544f53d70fb281afe97369d5ab6";
const METADATA: &str = "zai-org/GLM-5.3-Flash";
const METADATA_REVISION: &str = "04c4e9e95c5da8862dced7e5056455116f83a7e0";
const PROCESSOR: &str = "8b68a75e444228e20018f2b69618698f446e872746a03903a4dfcf51bc639921";
const CONFIG: &str = "d1e229307a382fc1deae8e4a48b6ffc2f47fc47769ab8789c023bdbd806072fd";
const TOKENIZER: &str = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d";
const TEMPLATE: &str = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891";
pub(in crate::engine::hybrid) const SIDECAR: &str = "vision/mmproj-BF16.gguf";
const SHARDS: [(&str, u64, &str); 5] = [
    (
        "GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
        9_429_984,
        "2f32eba33ebc75a715e9c85a010faac4c67542e38e8970bd961d544bfe815f9e",
    ),
    (
        "GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf",
        49_294_975_936,
        "f4a9e1ab13d5d9620f5590c9a4aba4c169e6707a1554f3be3fe3112f80a66825",
    ),
    (
        "GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf",
        49_949_266_048,
        "330a8ad76c787b3ab6df062dd30abea7aafe6a0e3d5860e8720dfa4abb2434a5",
    ),
    (
        "GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf",
        9_466_399_584,
        "5c294f42edc5d69cf00a79ab444e61f0742e9933105fc5c85f97da54dab8946f",
    ),
    (SIDECAR, 1_164_010_176, "82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac"),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Weight {
    size: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    kind: String,
    schema_version: u32,
    family_profile: String,
    family_version: u32,
    precision_policy: String,
    model_source: String,
    model_revision: String,
    metadata_source: String,
    metadata_revision: String,
    config_sha256: String,
    artifact_sha256: String,
    tokenizer_sha256: String,
    processor_sha256: String,
    template_sha256: String,
    source_receipt_sha256: String,
    weight_files: BTreeMap<String, Weight>,
}

pub(in crate::engine::hybrid) struct Artifact {
    pub identity: VisionArtifactIdentity,
    pub config: Value,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    signatures: BTreeMap<String, Vec<u64>>,
}

/// Parse only the bytes actually hashed from this descriptor. A later pathname
/// signature cannot bind a separately reopened metadata parser to these bytes.
fn metadata(
    path: &Path,
    expected: &str,
    limit: u64,
) -> Result<(Vec<u8>, Vec<u64>)> {
    let mut file = open(path)?;
    let before = signature(&file)?;
    if file.metadata()?.len() > limit {
        return Err(Error::Invalid("GLM image metadata size"));
    }
    let mut bytes = Vec::new();
    (&mut file).take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit || signature(&file)? != before || format!("{:x}", Sha256::digest(&bytes)) != expected
    {
        return Err(Error::Invalid("GLM image metadata identity"));
    }
    Ok((bytes, before))
}

fn profile(manifest: &Manifest) -> Result<()> {
    if manifest.kind != "Glm53NativeVisionArtifact"
        || manifest.schema_version != 7
        || manifest.family_profile != PROFILE
        || manifest.family_version != 1
        || manifest.precision_policy != "f32-vision-activations-v1"
        || manifest.model_source != SOURCE
        || manifest.model_revision != REVISION
        || manifest.metadata_source != METADATA
        || manifest.metadata_revision != METADATA_REVISION
        || manifest.processor_sha256 != PROCESSOR
        || manifest.config_sha256 != CONFIG
        || manifest.tokenizer_sha256 != TOKENIZER
        || manifest.template_sha256 != TEMPLATE
        || manifest.weight_files.len() != SHARDS.len()
        || SHARDS.iter().any(|(name, size, digest)| {
            manifest.weight_files.get(*name).is_none_or(|weight| weight.size != *size || weight.sha256 != *digest)
        })
    {
        return Err(Error::Invalid("GLM image artifact profile"));
    }
    Ok(())
}

fn vision_profile() -> Value {
    json!({"attention_bias":true,"attention_dropout":0.0,"depth":24,
        "hidden_act":"silu","hidden_size":1024,"image_size":448,"in_channels":3,
        "intermediate_size":4096,"model_type":"glm5_next_vision","num_heads":16,
        "out_hidden_size":4096,"patch_size":14,"projection_intermediate_size":10240,
        "rms_norm_eps":1e-5,"spatial_merge_size":2,"swiglu_limit":10.0,"temporal_patch_size":2})
}

fn config_profile(config: &Value) -> Result<()> {
    let vision = vision_profile();
    if config["model_type"] != "glm5_next"
        || config["architectures"] != json!(["Glm5NextForConditionalGeneration"])
        || config.get("expert_streaming").is_some()
        || config.pointer("/text_config/model_type").and_then(Value::as_str) != Some("glm5_next_text")
        || config.pointer("/text_config/hidden_size").and_then(Value::as_u64) != Some(4096)
        || config.pointer("/text_config/vocab_size").and_then(Value::as_u64) != Some(154880)
        || config.pointer("/text_config/num_nextn_predict_layers").and_then(Value::as_u64) != Some(1)
        || vision.as_object().unwrap().iter().any(|(key, value)| config["vision_config"].get(key) != Some(value))
        || [
            ("image_token_id", 154854),
            ("video_token_id", 154855),
            ("image_start_token_id", 154830),
            ("image_end_token_id", 154831),
            ("video_start_token_id", 154832),
            ("video_end_token_id", 154833),
        ]
        .iter()
        .any(|(key, value)| config.get(key).and_then(Value::as_u64) != Some(*value))
    {
        return Err(Error::Invalid("GLM image source geometry/head declaration"));
    }
    Ok(())
}

impl Artifact {
    pub fn admit(path: &Path) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&read(&path.join("stark-image-artifact.json"), 128 * 1024)?)?;
        profile(&manifest)?;
        if path.join("vision").symlink_metadata()?.file_type().is_symlink() {
            return Err(Error::Invalid("GLM image sidecar directory"));
        }
        let mut signatures = BTreeMap::new();
        let mut metadata_bytes = BTreeMap::new();
        for (name, expected, limit) in [
            ("config.json", &manifest.config_sha256, 1024 * 1024),
            ("tokenizer.json", &manifest.tokenizer_sha256, 64 * 1024 * 1024),
            ("preprocessor_config.json", &manifest.processor_sha256, 64 * 1024),
            ("chat_template.jinja", &manifest.template_sha256, 256 * 1024),
            ("native-source.json", &manifest.source_receipt_sha256, 128 * 1024),
        ] {
            let (bytes, sig) = metadata(&path.join(name), expected, limit)?;
            signatures.insert(name.to_owned(), sig);
            metadata_bytes.insert(name, bytes);
        }
        let config: Value = serde_json::from_slice(&metadata_bytes["config.json"])?;
        config_profile(&config)?;
        let source: Value = serde_json::from_slice(&metadata_bytes["native-source.json"])?;
        if source["schema_version"].as_u64() != Some(1)
            || source["architecture"] != "glm5_next"
            || source["artifact_mode"] != "text"
            || source["quantization"] != "UD-Q2_K_XL"
            || source["weights_source"] != json!({"repo":SOURCE,"revision":REVISION})
            || source["metadata_source"] != json!({"repo":METADATA,"revision":METADATA_REVISION})
            || source["weight_bytes"].as_u64() != Some(108_720_071_552)
            || source["config_sha256"] != manifest.config_sha256
            || config.pointer("/token_codec_config/prompt_template").and_then(Value::as_str).map(str::as_bytes)
                != Some(metadata_bytes["chat_template.jinja"].as_slice())
        {
            return Err(Error::Invalid("GLM image source agreement"));
        }
        let mut digest = Sha256::new();
        for (name, size, expected) in SHARDS {
            let (actual, sig, bytes) = hash(&path.join(name), size)?;
            if bytes != size || actual != expected {
                return Err(Error::Invalid("GLM image payload identity"));
            }
            digest.update(name.as_bytes());
            digest.update([0]);
            digest.update(actual.as_bytes());
            signatures.insert(name.to_owned(), sig);
        }
        if format!("{:x}", digest.finalize()) != manifest.artifact_sha256 {
            return Err(Error::Invalid("GLM image aggregate identity"));
        }
        let result = Self {
            config,
            tokenizer: Arc::new(
                tokenizers::Tokenizer::from_bytes(&metadata_bytes["tokenizer.json"])
                    .map_err(|_| Error::Invalid("GLM image bound tokenizer"))?,
            ),
            identity: VisionArtifactIdentity {
                schema_version: 7,
                config_sha256: manifest.config_sha256,
                artifact_sha256: manifest.artifact_sha256,
                tokenizer_sha256: manifest.tokenizer_sha256,
                processor_sha256: manifest.processor_sha256,
                template_sha256: manifest.template_sha256,
            },
            signatures,
        };
        result.unchanged(path)?;
        Ok(result)
    }

    pub fn source(&self) -> (&str, Option<&str>) {
        (SOURCE, Some(REVISION))
    }

    pub fn bind_decoder(
        &self,
        weights: &Weights,
    ) -> Result<()> {
        let expected: Vec<_> = SHARDS[..4].iter().map(|(name, _, _)| self.signatures[*name].clone()).collect();
        if weights.mapped_file_signatures()? != expected {
            return Err(Error::Invalid("GLM image mapped decoder identity"));
        }
        Ok(())
    }

    pub fn unchanged(
        &self,
        path: &Path,
    ) -> Result<()> {
        for (name, expected) in &self.signatures {
            if signature(&open(&path.join(name))?)? != *expected {
                return Err(Error::Invalid("GLM image artifact changed during load"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_geometry_is_not_inferred_from_model_name() {
        assert!(config_profile(&json!({"model_type":"glm5_next"})).is_err());
        assert_eq!(SHARDS[..4].iter().map(|(_, n, _)| n).sum::<u64>(), 108_720_071_552);
        assert_eq!(SHARDS.iter().filter(|(name, _, _)| name.contains('/')).count(), 1);
    }

    #[test]
    fn parser_owns_exact_hashed_metadata_across_parent_replacement() {
        let root = std::env::temp_dir().join(format!(
            "stark-glm-bound-metadata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let selected = root.join("selected");
        let other = root.join("other");
        let saved = root.join("saved");
        std::fs::create_dir(&selected).unwrap();
        std::fs::create_dir(&other).unwrap();
        let a = br#"{"profile":"admitted"}"#;
        let b = br#"{"profile":"substituted"}"#;
        std::fs::write(selected.join("config.json"), a).unwrap();
        std::fs::write(other.join("config.json"), b).unwrap();
        let expected = format!("{:x}", Sha256::digest(a));
        let (bytes, sig) = metadata(&selected.join("config.json"), &expected, 128).unwrap();
        std::fs::rename(&selected, &saved).unwrap();
        std::fs::rename(&other, &selected).unwrap();
        assert!(metadata(&selected.join("config.json"), &expected, 128).is_err());
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["profile"], "admitted");
        std::fs::rename(&selected, &other).unwrap();
        std::fs::rename(&saved, &selected).unwrap();
        assert_eq!(sig, signature(&open(&selected.join("config.json")).unwrap()).unwrap());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn pinned_profile_rejects_rebound_geometry_head_and_streaming_routes() {
        let config = json!({"model_type":"glm5_next","architectures":["Glm5NextForConditionalGeneration"],
            "text_config":{"model_type":"glm5_next_text","hidden_size":4096,"vocab_size":154880,"num_nextn_predict_layers":1},
            "vision_config":vision_profile(),"image_token_id":154854,"video_token_id":154855,
            "image_start_token_id":154830,"image_end_token_id":154831,"video_start_token_id":154832,"video_end_token_id":154833});
        assert!(config_profile(&config).is_ok());
        for (pointer, replacement) in [
            ("/vision_config/depth", json!(23)),
            ("/vision_config/patch_size", json!(16)),
            ("/vision_config/attention_bias", json!(1)),
            ("/text_config/num_nextn_predict_layers", json!(0)),
            ("/text_config/num_nextn_predict_layers", json!(true)),
            ("/image_token_id", json!(154855)),
        ] {
            let mut changed = config.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            assert!(config_profile(&changed).is_err());
        }
        let mut streaming = config;
        streaming["expert_streaming"] = json!({});
        assert!(config_profile(&streaming).is_err());
    }

    #[test]
    fn manifest_requires_exact_schema_precision_payloads_and_source() {
        let mut records = serde_json::Map::new();
        for (name, size, digest) in SHARDS {
            records.insert(name.into(), json!({"size":size,"sha256":digest}));
        }
        let manifest = json!({"kind":"Glm53NativeVisionArtifact","schema_version":7,
            "family_profile":PROFILE,"family_version":1,"precision_policy":"f32-vision-activations-v1",
            "model_source":SOURCE,"model_revision":REVISION,"metadata_source":METADATA,"metadata_revision":METADATA_REVISION,
            "config_sha256":CONFIG,"artifact_sha256":"a","tokenizer_sha256":TOKENIZER,"processor_sha256":PROCESSOR,
            "template_sha256":TEMPLATE,"source_receipt_sha256":"s","weight_files":records});
        assert!(profile(&serde_json::from_value::<Manifest>(manifest.clone()).unwrap()).is_ok());
        for (key, replacement) in [
            ("schema_version", json!(6)),
            ("precision_policy", json!("bf16")),
            ("model_revision", json!("unbound")),
            ("family_version", json!(2)),
        ] {
            let mut changed = manifest.clone();
            changed[key] = replacement;
            assert!(profile(&serde_json::from_value::<Manifest>(changed).unwrap()).is_err());
        }
        for (key, replacement) in
            [("schema_version", json!(true)), ("schema_version", json!(7.0)), ("execute", json!(true))]
        {
            let mut changed = manifest.clone();
            changed[key] = replacement;
            assert!(serde_json::from_value::<Manifest>(changed).is_err());
        }
        let mut changed = manifest;
        changed["weight_files"][SIDECAR]["sha256"] = json!("another");
        assert!(profile(&serde_json::from_value::<Manifest>(changed).unwrap()).is_err());
    }
}
