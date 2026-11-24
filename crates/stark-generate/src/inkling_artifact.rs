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

/// Stark's derived affine8 artifact: the original checkpoint with every attention
/// projection, the global layers' dynamic convolution projections, the token
/// embedding and the output head replaced by MLX affine 8-bit codes (group 64),
/// built by `stark_convert.inkling_affine8_artifact` from the pinned original.
/// Expert banks, router, gates, norms, convolutions, vision tower and the draft
/// head are unchanged. It is the artifact the default `inkling2-mini` choice
/// prepares; a prepared original still loads as a separate artifact.
pub(crate) const AFFINE8_LAYOUT: &str = "inkling2-mini-affine8-dense";
pub(crate) const AFFINE8_TARGET_SHA: &str = "c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd";
pub(crate) const AFFINE8_TARGET_BYTES: usize = 18_052_013_648;
/// Resident bytes above the original artifact: the affine kernel tensors are
/// kept beside the BF16 effective copies that prefill reads.
pub(crate) const AFFINE8_EXTRA_RESIDENT_BYTES: usize = 1_670_000_000;

const CONFIG_LIMIT: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    Original,
    Affine8,
}

impl Layout {
    pub(crate) fn target_sha256(self) -> &'static str {
        match self {
            Self::Original => TARGET_SHA,
            Self::Affine8 => AFFINE8_TARGET_SHA,
        }
    }

    fn target_bytes(self) -> usize {
        match self {
            Self::Original => TARGET_BYTES,
            Self::Affine8 => AFFINE8_TARGET_BYTES,
        }
    }

    /// Resident weight bytes above the original artifact.
    pub(crate) fn extra_resident_bytes(self) -> usize {
        match self {
            Self::Original => 0,
            Self::Affine8 => AFFINE8_EXTRA_RESIDENT_BYTES,
        }
    }

    /// The layout an installed config declares; unrecognized values are
    /// rejected at load.
    pub(crate) fn of(root: &Path) -> Self {
        let affine = read(&root.join("config.json"), CONFIG_LIMIT)
            .ok()
            .and_then(|data| serde_json::from_slice::<Value>(&data).ok())
            .is_some_and(|config| config.get("stark_inkling_affine8").is_some());
        if affine {
            Self::Affine8
        } else {
            Self::Original
        }
    }
}

fn affine8_binding() -> Value {
    serde_json::json!({"layout": AFFINE8_LAYOUT, "group_size": 64, "bits": 8})
}

pub(crate) struct Artifact {
    pub layout: Layout,
    pub target: VerifiedWeights,
    pub draft: VerifiedWeights,
    pub tokenizer: Tokenizer,
    pub config_sha256: String,
    pub context_limit: u32,
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
        let layout = Layout::of(root);
        let config = read(&root.join("config.json"), CONFIG_LIMIT)?;
        let config_sha256 = format!("{:x}", Sha256::digest(&config));
        let mut installed: Value = serde_json::from_slice(&config).map_err(|_| "invalid_model_config")?;
        let template = checked_bytes(&root.join("chat_template.jinja"), 256 * 1024, TEMPLATE_SHA)?;
        if config_sha256 != CONFIG_SHA || layout == Layout::Affine8 {
            let original = checked_bytes(&root.join("source-config.json"), CONFIG_LIMIT, CONFIG_SHA)?;
            let original: Value = serde_json::from_slice(&original).map_err(|_| "invalid_model_config")?;
            let extra = installed.as_object_mut().ok_or("invalid_model_config")?;
            let codec = extra.remove("token_codec_config").ok_or("invalid_model_config")?;
            if !codec.as_object().is_some_and(|codec| codec.len() == 2)
                || codec["type"] != "ChatCodecConfig"
                || codec["prompt_template"].as_str().map(str::as_bytes) != Some(template.as_slice())
                || extra.remove("stark_inkling") != Some(serde_json::json!({"source": SOURCE, "revision": REVISION}))
                || (layout == Layout::Affine8 && extra.remove("stark_inkling_affine8") != Some(affine8_binding()))
                || installed != original
            {
                return Err("inkling_checkpoint_mismatch");
            }
        }
        let context_limit = installed["text_config"]["model_max_length"]
            .as_u64()
            .and_then(|length| u32::try_from(length).ok())
            .filter(|length| *length >= 2)
            .ok_or("invalid_model_config")?;
        checked_bytes(&root.join("processor_config.json"), CONFIG_LIMIT, PROCESSOR_SHA)?;
        checked_bytes(&root.join("hf_quant_config.json"), CONFIG_LIMIT, QUANT_SHA)?;
        let token_bytes = checked_bytes(&root.join("tokenizer.json"), 32 * 1024 * 1024, TOKENIZER_SHA)?;
        let tokenizer = Tokenizer::from_bytes(&token_bytes).map_err(|_| "invalid_model_config")?;
        let target = check_weights(&root.join(TARGET), layout.target_bytes(), layout.target_sha256())?;
        let draft = check_weights(&root.join("mtp.safetensors"), DRAFT_BYTES, DRAFT_SHA)?;
        Ok(Self {
            layout,
            target,
            draft,
            tokenizer,
            config_sha256,
            context_limit,
        })
    }
}
