//! Content-bound admission for explicit retained original-format image artifacts.

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Read,
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::super::{Error, Result};
use crate::engine::vision::VisionArtifactIdentity;

pub const PROFILE: &str = "qwen38-lily-retained-bf16-vision-v1";
const PROCESSOR: &str = "022f906548d5a89d3a18c61c0a87de760db8c81d618437501baa3dcdc14d4ecd";

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
    model_source: String,
    model_revision: String,
    config_sha256: String,
    artifact_sha256: String,
    tokenizer_sha256: String,
    processor_sha256: String,
    template_sha256: String,
    index_sha256: String,
    weight_files: BTreeMap<String, Weight>,
}

pub(in crate::engine::hybrid) struct ImageArtifact {
    pub identity: VisionArtifactIdentity,
    pub source: String,
    pub revision: String,
    signatures: BTreeMap<String, Vec<u64>>,
}

pub(in crate::engine::hybrid) fn open(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::Invalid("image artifact file type"));
    }
    Ok(file)
}

pub(in crate::engine::hybrid) fn signature(file: &File) -> Result<Vec<u64>> {
    let m = file.metadata()?;
    #[cfg(unix)]
    return Ok(vec![
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ]);
    #[cfg(not(unix))]
    Ok(vec![m.len()])
}

pub(in crate::engine::hybrid) fn read(
    path: &Path,
    maximum: u64,
) -> Result<Vec<u8>> {
    let mut f = open(path)?;
    let before = signature(&f)?;
    if f.metadata()?.len() > maximum {
        return Err(Error::Invalid("image artifact metadata size"));
    }
    let mut bytes = Vec::new();
    (&mut f).take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum || signature(&f)? != before {
        return Err(Error::Invalid("image artifact changed"));
    }
    Ok(bytes)
}

pub(in crate::engine::hybrid) fn hash(
    path: &Path,
    maximum: u64,
) -> Result<(String, Vec<u64>, u64)> {
    let mut f = open(path)?;
    let before = signature(&f)?;
    let size = f.metadata()?.len();
    if size == 0 || size > maximum {
        return Err(Error::Invalid("image artifact file size"));
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut remaining = size;
    while remaining != 0 {
        let maximum = remaining.min(buffer.len() as u64) as usize;
        let count = f.read(&mut buffer[..maximum])?;
        if count == 0 {
            return Err(Error::Invalid("image artifact changed"));
        }
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    if f.read(&mut buffer[..1])? != 0 || signature(&f)? != before {
        return Err(Error::Invalid("image artifact changed"));
    }
    Ok((format!("{:x}", digest.finalize()), before, size))
}

impl ImageArtifact {
    pub fn admit(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let manifest: Manifest = serde_json::from_slice(&read(&path.join("stark-image-artifact.json"), 128 * 1024)?)?;
        let source_ok = matches!(
            (manifest.model_source.as_str(), manifest.model_revision.as_str()),
            ("fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl", "0f94842e3682b49acace4daeb4cf2e65a6f94f83")
                | ("ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit", "788c07a8725642bccf13a0a3b8a21954ab84b4f2")
        );
        if manifest.kind != "Qwen38RetainedVisionArtifact"
            || manifest.schema_version != 6
            || manifest.family_profile != PROFILE
            || manifest.family_version != 1
            || !source_ok
            || manifest.processor_sha256 != PROCESSOR
            || !(1..=256).contains(&manifest.weight_files.len())
        {
            return Err(Error::Invalid("retained image artifact profile"));
        }
        let mut signatures = BTreeMap::new();
        for (name, expected, limit) in [
            ("config.json", &manifest.config_sha256, 8 * 1024 * 1024),
            ("tokenizer.json", &manifest.tokenizer_sha256, 64 * 1024 * 1024),
            ("preprocessor_config.json", &manifest.processor_sha256, 64 * 1024),
            ("chat_template.jinja", &manifest.template_sha256, 256 * 1024),
            ("model.safetensors.index.json", &manifest.index_sha256, 16 * 1024 * 1024),
        ] {
            let (actual, signature, _) = hash(&path.join(name), limit)?;
            if actual != *expected {
                return Err(Error::Invalid("image artifact component identity"));
            }
            signatures.insert(name.into(), signature);
        }
        let config: Value = serde_json::from_slice(&read(&path.join("config.json"), 8 * 1024 * 1024)?)?;
        let template = read(&path.join("chat_template.jinja"), 256 * 1024)?;
        if config.pointer("/token_codec_config/prompt_template").and_then(Value::as_str).map(str::as_bytes)
            != Some(template.as_slice())
            || config.get("vision_config").and_then(Value::as_object).is_none()
        {
            return Err(Error::Invalid("retained image metadata agreement"));
        }
        let index: Value =
            serde_json::from_slice(&read(&path.join("model.safetensors.index.json"), 16 * 1024 * 1024)?)?;
        let shard_names: std::collections::BTreeSet<&str> = index["weight_map"]
            .as_object()
            .ok_or(Error::Invalid("image shard index"))?
            .values()
            .map(|v| v.as_str().ok_or(Error::Invalid("image shard name")))
            .collect::<Result<_>>()?;
        let expected_shards: std::collections::BTreeSet<&str> =
            manifest.weight_files.keys().filter(|n| n.ends_with(".safetensors")).map(String::as_str).collect();
        if shard_names != expected_shards || shard_names.is_empty() {
            return Err(Error::Invalid("image shard inventory"));
        }
        let mut digest = Sha256::new();
        let mut total = 0u64;
        for (name, weight) in manifest.weight_files {
            if name.contains(['/', '\\'])
                || name.starts_with('.')
                || !(name.ends_with(".safetensors") || name == "ngram_table.bin")
            {
                return Err(Error::Invalid("image artifact shard name"));
            }
            total = total
                .checked_add(weight.size)
                .filter(|n| *n <= 160 * 1024 * 1024 * 1024)
                .ok_or(Error::Invalid("image artifact weight budget"))?;
            let (actual, signature, size) = hash(&path.join(&name), weight.size)?;
            if size != weight.size || actual != weight.sha256 {
                return Err(Error::Invalid("image artifact shard identity"));
            }
            digest.update(name.as_bytes());
            digest.update([0]);
            // The exact ASCII digest is unambiguous with fixed 64-byte length.
            digest.update(actual.as_bytes());
            signatures.insert(name, signature);
        }
        if format!("{:x}", digest.finalize()) != manifest.artifact_sha256 {
            return Err(Error::Invalid("image artifact aggregate identity"));
        }
        Ok(Self {
            identity: VisionArtifactIdentity {
                schema_version: 6,
                config_sha256: manifest.config_sha256,
                artifact_sha256: manifest.artifact_sha256,
                tokenizer_sha256: manifest.tokenizer_sha256,
                processor_sha256: manifest.processor_sha256,
                template_sha256: manifest.template_sha256,
            },
            source: manifest.model_source,
            revision: manifest.model_revision,
            signatures,
        })
    }

    pub fn unchanged(
        &self,
        path: &Path,
    ) -> Result<()> {
        for (name, expected) in &self.signatures {
            if signature(&open(&path.join(name))?)? != *expected {
                return Err(Error::Invalid("image artifact changed during load"));
            }
        }
        Ok(())
    }
}
