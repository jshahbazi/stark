// SPDX-License-Identifier: Apache-2.0
//! Trusted-local proof of a previous full hash of the original GGUF files.

use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use stark_engine::engine::hy4::{ENGRAM_SHARD, PROJECTOR, SHARD};

const RECEIPT: &str = ".stark-hy4-source.json";
const MAX_RECEIPT_BYTES: u64 = 8 * 1024;
pub(super) const REPOSITORY: &str = "AngelSlim/Hy4-flash-GGUF";
pub(super) const REVISION: &str = "c50d4dd30be42f2b91138e5731de0e2f32ad56b2";
// SHA-256 of the ordered name, NUL, lowercase digest triples below.
pub(super) const ARTIFACT_SHA256: &str = "f9a244c3a1f85e03847959b575528c3927ac05ab0a3f532b1c1831f9eb1ddd3c";

const GGUF: [(&str, u64, &str); 3] = [
    (SHARD, 92_010_389_504, "987b2902813ad94919e3e46e375bd8144d36113cced62d25f13013ccc88a3abb"),
    (ENGRAM_SHARD, 63_360_616_128, "ecd6a0ecd7f258b1ee9211bf566261ae33efc1743ad40f58cb5004cf7b051839"),
    (PROJECTOR, 1_044_397_440, "8c33a48ef57480abf4854126964058a2015b5da3b773b14c847c995cf8899bda"),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u32,
    repository: String,
    revision: String,
    files: [ReceiptFile; GGUF.len()],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptFile {
    name: String,
    sha256: String,
    fingerprint: Fingerprint,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Fingerprint {
    device: u64,
    inode: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
}

impl Fingerprint {
    fn from_metadata(metadata: &Metadata) -> Option<Self> {
        metadata.is_file().then(|| Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            mtime_sec: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime_sec: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
        })
    }
}

struct OpenSource {
    path: PathBuf,
    file: File,
    fingerprint: Fingerprint,
}

pub(super) struct VerifiedSource {
    files: Vec<OpenSource>,
}

impl VerifiedSource {
    pub(super) fn open(model: &Path) -> Result<Self, &'static str> {
        let mut receipt = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(model.join(RECEIPT))
            .map_err(|_| "hy4_source_verification_required")?;
        let metadata = receipt.metadata().map_err(|_| "invalid_hy4_source_receipt")?;
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.mode() & 0o077 != 0
            || metadata.nlink() != 1
            || metadata.len() > MAX_RECEIPT_BYTES
        {
            return Err("invalid_hy4_source_receipt");
        }
        let before = Fingerprint::from_metadata(&metadata).ok_or("invalid_hy4_source_receipt")?;
        let mut bytes = Vec::new();
        (&mut receipt).take(MAX_RECEIPT_BYTES + 1).read_to_end(&mut bytes).map_err(|_| "invalid_hy4_source_receipt")?;
        let after = receipt.metadata().map_err(|_| "invalid_hy4_source_receipt")?;
        if bytes.len() as u64 != before.size || Fingerprint::from_metadata(&after) != Some(before) {
            return Err("invalid_hy4_source_receipt");
        }
        let receipt: Receipt = serde_json::from_slice(&bytes).map_err(|_| "invalid_hy4_source_receipt")?;
        if receipt.schema_version != 1 || receipt.repository != REPOSITORY || receipt.revision != REVISION {
            return Err("invalid_hy4_source_receipt");
        }

        let mut files = Vec::with_capacity(GGUF.len());
        for (entry, (name, size, digest)) in receipt.files.into_iter().zip(GGUF) {
            if entry.name != name || entry.sha256 != digest || entry.fingerprint.size != size {
                return Err("invalid_hy4_source_receipt");
            }
            let path = model.join(name);
            // Hugging Face snapshots can symlink files to the verified cache blobs.
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
                .open(&path)
                .map_err(|_| "incomplete_hy4_artifact")?;
            files.push(OpenSource {
                path,
                file,
                fingerprint: entry.fingerprint,
            });
        }
        let source = Self {
            files,
        };
        source.verify_unchanged()?;
        Ok(source)
    }

    /// Compare the descriptor and the current path target around native initialization.
    pub(super) fn verify_unchanged(&self) -> Result<(), &'static str> {
        for source in &self.files {
            let descriptor = source.file.metadata().map_err(|_| "hy4_source_changed")?;
            let opened_path = fs::metadata(&source.path).map_err(|_| "hy4_source_changed")?;
            if Fingerprint::from_metadata(&descriptor).as_ref() != Some(&source.fingerprint)
                || Fingerprint::from_metadata(&opened_path).as_ref() != Some(&source.fingerprint)
            {
                return Err("hy4_source_changed");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{ARTIFACT_SHA256, GGUF};

    #[test]
    fn aggregate_identity_binds_every_pinned_gguf() {
        let mut digest = Sha256::new();
        for (name, _, sha) in GGUF {
            digest.update(name.as_bytes());
            digest.update([0]);
            digest.update(sha.as_bytes());
        }
        assert_eq!(format!("{:x}", digest.finalize()), ARTIFACT_SHA256);
    }
}
