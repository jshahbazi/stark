// SPDX-License-Identifier: Apache-2.0
//! Checkpoint binding for the published, fixed-three-pass Inkling 2 Mini.

use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

pub(crate) const SOURCE: &str = "thinkingmachines/Inkling-2-Mini";
pub(crate) const REVISION: &str = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb";
pub(crate) const TARGET: &str = "model-00001-of-00001.safetensors";
pub(crate) const TARGET_SHA: &str = "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1";
pub(crate) const DRAFT_SHA: &str = "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2";
const CONFIG_SHA: &str = "8216dd43157f5474cfff88f486a1e433fef703c674f694e485a86a761c412b1e";
pub(crate) const TOKENIZER_SHA: &str = "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6";
pub(crate) const PROCESSOR_SHA: &str = "d7350adce18c8be4b607f261c8f0f7eabbf720bb7aaf648a43d4f8372bbacc2d";
pub(crate) const TEMPLATE_SHA: &str = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc";
pub(crate) const QUANT_SHA: &str = "7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143";
pub(crate) const TARGET_BYTES: usize = 19_526_019_168;
pub(crate) const DRAFT_BYTES: usize = 635_801_416;

const CONFIG_LIMIT: u64 = 1024 * 1024;

pub(crate) struct Artifact {
    pub target: VerifiedWeights,
    pub draft: VerifiedWeights,
    pub tokenizer: Tokenizer,
    pub config_sha256: String,
}

pub(crate) struct VerifiedWeights {
    file: File,
    stamp: (u64, i64, i64, i64, i64),
}

impl VerifiedWeights {
    fn stamp(file: &File) -> Result<(u64, i64, i64, i64, i64), &'static str> {
        let state = file.metadata().map_err(|_| "model_files_unavailable")?;
        Ok((state.len(), state.mtime(), state.mtime_nsec(), state.ctime(), state.ctime_nsec()))
    }

    pub(crate) fn descriptor(&self) -> Result<File, &'static str> {
        self.file.try_clone().map_err(|_| "model_files_unavailable")
    }

    pub(crate) fn verify_unchanged(&self) -> Result<(), &'static str> {
        if Self::stamp(&self.file)? != self.stamp {
            return Err("inkling_checkpoint_mismatch");
        }
        Ok(())
    }
}

fn file(path: &Path) -> Result<File, &'static str> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| "model_files_unavailable")?;
    if !file.metadata().map_err(|_| "model_files_unavailable")?.is_file() {
        return Err("invalid_model_config");
    }
    Ok(file)
}

fn read(
    path: &Path,
    limit: u64,
) -> Result<Vec<u8>, &'static str> {
    let file = file(path)?;
    let len = file.metadata().map_err(|_| "model_files_unavailable")?.len();
    if len == 0 || len > limit {
        return Err("invalid_model_config");
    }
    let mut value = Vec::with_capacity(len as usize);
    file.take(limit + 1).read_to_end(&mut value).map_err(|_| "invalid_model_config")?;
    if value.len() as u64 != len {
        return Err("invalid_model_config");
    }
    Ok(value)
}

fn checked_bytes(
    path: &Path,
    limit: u64,
    expected: &str,
) -> Result<Vec<u8>, &'static str> {
    let bytes = read(path, limit)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err("inkling_checkpoint_mismatch");
    }
    Ok(bytes)
}

fn check_weights(
    path: &Path,
    bytes: usize,
    expected: &str,
) -> Result<VerifiedWeights, &'static str> {
    let file = file(path)?;
    let stamp = VerifiedWeights::stamp(&file)?;
    if stamp.0 != bytes as u64 {
        return Err("inkling_checkpoint_mismatch");
    }
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hash = Sha256::new();
    let mut hashed = 0;
    loop {
        let block = reader.fill_buf().map_err(|_| "model_files_unavailable")?;
        if block.is_empty() {
            break;
        }
        hash.update(block);
        let n = block.len();
        hashed += n;
        if hashed > bytes {
            return Err("inkling_checkpoint_mismatch");
        }
        reader.consume(n);
    }
    if hashed != bytes || format!("{:x}", hash.finalize()) != expected {
        return Err("inkling_checkpoint_mismatch");
    }
    let weights = VerifiedWeights {
        file: reader.into_inner(),
        stamp,
    };
    weights.verify_unchanged()?;
    Ok(weights)
}

pub(crate) fn recognizes(model: &Path) -> bool {
    read(&model.join("config.json"), CONFIG_LIMIT)
        .ok()
        .and_then(|data| serde_json::from_slice::<Value>(&data).ok())
        .is_some_and(|config| config["model_type"] == "inkling2_loop")
}

impl Artifact {
    pub(crate) fn load(root: &Path) -> Result<Self, &'static str> {
        let config = read(&root.join("config.json"), CONFIG_LIMIT)?;
        let config_sha256 = format!("{:x}", Sha256::digest(&config));
        let template = checked_bytes(&root.join("chat_template.jinja"), 256 * 1024, TEMPLATE_SHA)?;
        if config_sha256 != CONFIG_SHA {
            let original = checked_bytes(&root.join("source-config.json"), CONFIG_LIMIT, CONFIG_SHA)?;
            let mut installed: Value = serde_json::from_slice(&config).map_err(|_| "invalid_model_config")?;
            let original: Value = serde_json::from_slice(&original).map_err(|_| "invalid_model_config")?;
            let extra = installed.as_object_mut().ok_or("invalid_model_config")?;
            let codec = extra.remove("token_codec_config").ok_or("invalid_model_config")?;
            if !codec.as_object().is_some_and(|codec| codec.len() == 2)
                || codec["type"] != "ChatCodecConfig"
                || codec["prompt_template"].as_str().map(str::as_bytes) != Some(template.as_slice())
                || extra.remove("stark_inkling") != Some(serde_json::json!({"source": SOURCE, "revision": REVISION}))
                || installed != original
            {
                return Err("inkling_checkpoint_mismatch");
            }
        }
        checked_bytes(&root.join("processor_config.json"), CONFIG_LIMIT, PROCESSOR_SHA)?;
        checked_bytes(&root.join("hf_quant_config.json"), CONFIG_LIMIT, QUANT_SHA)?;
        let token_bytes = checked_bytes(&root.join("tokenizer.json"), 32 * 1024 * 1024, TOKENIZER_SHA)?;
        let tokenizer = Tokenizer::from_bytes(&token_bytes).map_err(|_| "invalid_model_config")?;
        let target = check_weights(&root.join(TARGET), TARGET_BYTES, TARGET_SHA)?;
        let draft = check_weights(&root.join("mtp.safetensors"), DRAFT_BYTES, DRAFT_SHA)?;
        Ok(Self {
            target,
            draft,
            tokenizer,
            config_sha256,
        })
    }
}
