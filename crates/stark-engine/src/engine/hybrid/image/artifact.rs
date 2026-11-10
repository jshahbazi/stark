//! Content-bound admission for explicit retained original-format image artifacts.

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::super::{Error, Result};
use crate::engine::vision::VisionArtifactIdentity;

pub const PROFILE: &str = "qwen38-lily-retained-bf16-vision-v1";
const PROCESSOR: &str = "022f906548d5a89d3a18c61c0a87de760db8c81d618437501baa3dcdc14d4ecd";
// Keep validation bounded while allowing independent shards to use the disk's
// bandwidth. Each verifier owns an 8 MiB read buffer and no model state.
const HASH_WORKERS: usize = 4;
const HASH_CACHE: &str = ".stark-image-validation.json";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CachedWeight {
    sha256: String,
    signature: Vec<u64>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HashCache {
    schema_version: u32,
    manifest_sha256: String,
    files: BTreeMap<String, CachedWeight>,
}

fn cached_weights(
    path: &Path,
    manifest_sha256: &str,
) -> BTreeMap<String, CachedWeight> {
    fn load(
        path: &Path,
        manifest_sha256: &str,
    ) -> Option<HashCache> {
        if !cfg!(unix) {
            return None;
        }
        let mut file = open(&path.join(HASH_CACHE)).ok()?;
        let metadata = file.metadata().ok()?;
        #[cfg(unix)]
        if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o077 != 0 || metadata.nlink() != 1 {
            return None;
        }
        if metadata.len() > 128 * 1024 {
            return None;
        }
        let before = signature(&file).ok()?;
        let mut data = Vec::new();
        (&mut file).take(128 * 1024 + 1).read_to_end(&mut data).ok()?;
        if data.len() > 128 * 1024 || signature(&file).ok()? != before {
            return None;
        }
        let cache: HashCache = serde_json::from_slice(&data).ok()?;
        (cache.schema_version == 1
            && cache.manifest_sha256 == manifest_sha256
            && cache.files.len() <= 256
            && cache.files.values().all(|entry| {
                entry.signature.len() == 7
                    && entry.sha256.len() == 64
                    && entry.sha256.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }))
        .then_some(cache)
    }
    load(path, manifest_sha256).map(|cache| cache.files).unwrap_or_default()
}

fn save_weights(
    path: &Path,
    cache: &HashCache,
) {
    // This is a disposable, owner-private content-digest cache, never inference
    // state. Cache write failures leave full validation available next launch.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    let temporary = path.join(format!(".stark-image-validation-{}-{nonce}.tmp", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let Ok(mut file) = options.open(&temporary) else {
        return;
    };
    let result = (|| -> std::io::Result<()> {
        serde_json::to_writer(&mut file, cache)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, path.join(HASH_CACHE))
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
}

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

fn open(path: &Path) -> Result<File> {
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

fn signature(file: &File) -> Result<Vec<u64>> {
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

fn read(
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

fn hash(
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

fn hash_weights(
    path: &Path,
    weights: &BTreeMap<String, Weight>,
    cache: &BTreeMap<String, CachedWeight>,
) -> Result<Vec<(String, Vec<u64>)>> {
    let entries: Vec<_> = weights.iter().collect();
    let mut verified = Vec::with_capacity(entries.len());
    for batch in entries.chunks(HASH_WORKERS) {
        let results = std::thread::scope(|scope| -> Result<Vec<_>> {
            let mut workers = Vec::with_capacity(batch.len());
            for &(name, weight) in batch {
                workers.push(std::thread::Builder::new().spawn_scoped(scope, move || {
                    if let Some(cached) = cache.get(name) {
                        let file = open(&path.join(name))?;
                        let current = signature(&file)?;
                        if cached.sha256 == weight.sha256
                            && current == cached.signature
                            && file.metadata()?.len() == weight.size
                        {
                            return Ok((cached.sha256.clone(), current));
                        }
                    }
                    let (actual, signature, size) = hash(&path.join(name), weight.size)?;
                    if size != weight.size || actual != weight.sha256 {
                        return Err(Error::Invalid("image artifact shard identity"));
                    }
                    Ok((actual, signature))
                })?);
            }
            workers
                .into_iter()
                .map(|worker| worker.join().map_err(|_| Error::Invalid("image artifact hash worker"))?)
                .collect()
        })?;
        verified.extend(results);
    }
    Ok(verified)
}

impl ImageArtifact {
    pub fn admit(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let manifest_bytes = read(&path.join("stark-image-artifact.json"), 128 * 1024)?;
        let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
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
        let mut total = 0u64;
        for (name, weight) in &manifest.weight_files {
            if name.contains(['/', '\\'])
                || name.starts_with('.')
                || !(name.ends_with(".safetensors") || name == "ngram_table.bin")
            {
                return Err(Error::Invalid("image artifact shard name"));
            }
            if weight.size == 0 {
                return Err(Error::Invalid("image artifact file size"));
            }
            total = total
                .checked_add(weight.size)
                .filter(|n| *n <= 160 * 1024 * 1024 * 1024)
                .ok_or(Error::Invalid("image artifact weight budget"))?;
        }
        let cache = cached_weights(&path, &manifest_sha256);
        let verified = hash_weights(&path, &manifest.weight_files, &cache)?;
        let mut digest = Sha256::new();
        let mut files = BTreeMap::new();
        for ((name, _), (actual, signature)) in manifest.weight_files.into_iter().zip(verified) {
            digest.update(name.as_bytes());
            digest.update([0]);
            // The exact ASCII digest is unambiguous with fixed 64-byte length.
            digest.update(actual.as_bytes());
            files.insert(
                name.clone(),
                CachedWeight {
                    sha256: actual,
                    signature: signature.clone(),
                },
            );
            signatures.insert(name, signature);
        }
        if format!("{:x}", digest.finalize()) != manifest.artifact_sha256 {
            return Err(Error::Invalid("image artifact aggregate identity"));
        }
        save_weights(
            &path,
            &HashCache {
                schema_version: 1,
                manifest_sha256,
                files,
            },
        );
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

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use stark_engine_macros::stark_test;

    use super::*;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "stark-image-hash-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn weights(fixture: &Fixture) -> BTreeMap<String, Weight> {
        // Published SHA-256 vectors, repeated across the four-worker boundary.
        let vectors: &[(&[u8], &str)] = &[
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (b"hello", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
            (b"hello world", "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
        ];
        (0..HASH_WORKERS + 1)
            .map(|i| {
                let (bytes, digest) = vectors[i % vectors.len()];
                let name = format!("model-{i:02}.safetensors");
                fs::write(fixture.0.join(&name), bytes).unwrap();
                (
                    name,
                    Weight {
                        size: bytes.len() as u64,
                        sha256: digest.into(),
                    },
                )
            })
            .collect()
    }

    #[stark_test]
    fn parallel_hashes_preserve_manifest_order_and_sha256() {
        let fixture = Fixture::new();
        let weights = weights(&fixture);
        let actual = hash_weights(&fixture.0, &weights, &BTreeMap::new()).unwrap();
        assert_eq!(actual.len(), weights.len());
        for ((name, expected), (digest, metadata)) in weights.iter().zip(actual) {
            assert_eq!(digest, expected.sha256);
            assert_eq!(metadata, signature(&open(&fixture.0.join(name)).unwrap()).unwrap());
        }
    }

    #[stark_test]
    fn parallel_hashes_reject_changed_or_missing_shards() {
        let fixture = Fixture::new();
        let weights = weights(&fixture);
        let last = weights.last_key_value().unwrap().0;
        fs::write(fixture.0.join(last), b"wrong").unwrap();
        assert!(hash_weights(&fixture.0, &weights, &BTreeMap::new()).is_err());
        fs::remove_file(fixture.0.join(last)).unwrap();
        assert!(hash_weights(&fixture.0, &weights, &BTreeMap::new()).is_err());
    }

    fn cache_for(
        fixture: &Fixture,
        weights: &BTreeMap<String, Weight>,
    ) -> HashCache {
        let verified = hash_weights(&fixture.0, weights, &BTreeMap::new()).unwrap();
        HashCache {
            schema_version: 1,
            manifest_sha256: "manifest identity".into(),
            files: weights
                .keys()
                .cloned()
                .zip(verified.into_iter().map(|(sha256, signature)| CachedWeight {
                    sha256,
                    signature,
                }))
                .collect(),
        }
    }

    #[stark_test]
    fn digest_cache_is_private_and_bound_to_the_manifest() {
        let fixture = Fixture::new();
        let weights = weights(&fixture);
        let cache = cache_for(&fixture, &weights);
        save_weights(&fixture.0, &cache);
        let loaded = cached_weights(&fixture.0, &cache.manifest_sha256);
        assert_eq!(loaded.len(), weights.len());
        assert!(hash_weights(&fixture.0, &weights, &loaded).is_ok());
        assert!(cached_weights(&fixture.0, "changed manifest").is_empty());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(fixture.0.join(HASH_CACHE), fs::Permissions::from_mode(0o644)).unwrap();
            assert!(cached_weights(&fixture.0, &cache.manifest_sha256).is_empty());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(fixture.0.join(HASH_CACHE), fs::Permissions::from_mode(0o600)).unwrap();
        }
        fs::write(fixture.0.join(HASH_CACHE), b"invalid cache").unwrap();
        assert!(cached_weights(&fixture.0, &cache.manifest_sha256).is_empty());
    }

    #[stark_test]
    fn digest_cache_rehashes_same_size_edits_with_restored_mtime() {
        let fixture = Fixture::new();
        let weights = weights(&fixture);
        let cache = cache_for(&fixture, &weights);
        let last = weights.last_key_value().unwrap().0;
        let path = fixture.0.join(last);
        let before = fs::metadata(&path).unwrap();
        fs::write(&path, b"WRONG").unwrap();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_times(fs::FileTimes::new().set_modified(before.modified().unwrap())).unwrap();
        assert_eq!(file.metadata().unwrap().len(), before.len());
        assert_eq!(file.metadata().unwrap().modified().unwrap(), before.modified().unwrap());
        assert!(hash_weights(&fixture.0, &weights, &cache.files).is_err());
    }
}
