//! Exact-artifact gate for the narrowly scoped Qwen3.8 paged-text request plan.
//!
//! The shared inventory is a candidate policy, not a general memory guarantee.
//! Any absent, changed, symlinked, malformed, or oversized identity fails closed.

use std::{fs, path::Path};

use sha2::{Digest, Sha256};

const INVENTORY: &str = include_str!("../../../../../tools/converter/stark_convert/guided_memory_profiles.json");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionPlan {
    pub request_headroom_divisor: usize,
    pub maximum_context_tokens: u32,
    pub host_reserve_bytes: usize,
    pub paged_working_set_bytes: usize,
    pub maximum_mtp_drafts: u32,
}

impl AdmissionPlan {
    pub fn applies_to(
        &self,
        context_tokens: u32,
        mtp_drafts: u32,
    ) -> bool {
        context_tokens <= self.maximum_context_tokens && mtp_drafts <= self.maximum_mtp_drafts
    }
}

fn safe_file(
    root: &Path,
    name: &str,
    expected_bytes: Option<u64>,
    expected_hash: Option<&str>,
) -> Option<Vec<u8>> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name == "." || name == ".." {
        return None;
    }
    let path = root.join(name);
    let metadata = fs::symlink_metadata(&path).ok()?;
    if !metadata.file_type().is_file()
        || expected_bytes.is_some_and(|size| metadata.len() != size)
        || ((expected_hash.is_some() || expected_bytes.is_none()) && metadata.len() > 2 * 1024 * 1024)
    {
        return None;
    }
    let bytes = if expected_hash.is_some() || expected_bytes.is_none() {
        let bytes = fs::read(path).ok()?;
        if expected_hash.is_some_and(|hash| format!("{:x}", Sha256::digest(&bytes)) != hash) {
            return None;
        }
        bytes
    } else {
        Vec::new()
    };
    Some(bytes)
}

/// Returns a request-plan candidate only when local identity exactly matches
/// one of the two pinned text artifacts. Images are rejected by the caller.
pub fn qualified_plan(root: &Path) -> Option<AdmissionPlan> {
    let inventory: serde_json::Value = serde_json::from_str(INVENTORY).ok()?;
    qualified_plan_with_inventory(root, &inventory)
}

fn qualified_plan_with_inventory(
    root: &Path,
    inventory: &serde_json::Value,
) -> Option<AdmissionPlan> {
    let root_metadata = fs::symlink_metadata(root).ok()?;
    if !root_metadata.file_type().is_dir() {
        return None;
    }
    if inventory["schema_version"].as_u64()? != 1
        || inventory["policy"].as_str()? != "qwen38-paged-text-v1"
        || inventory["request_headroom_divisor"].as_u64()? != 1
        || inventory["maximum_context_tokens"].as_u64()? != 32_768
        || inventory["maximum_mtp_drafts"].as_u64()? != 2
        || inventory["host_reserve_bytes"].as_u64()? != 8 * 1024_u64.pow(3)
        || inventory["paged_working_set_bytes"].as_u64()? != 1024_u64.pow(3)
    {
        return None;
    }
    let maximum_context_tokens = u32::try_from(inventory["maximum_context_tokens"].as_u64()?).ok()?;
    let host_reserve_bytes = usize::try_from(inventory["host_reserve_bytes"].as_u64()?).ok()?;
    let paged_working_set_bytes = usize::try_from(inventory["paged_working_set_bytes"].as_u64()?).ok()?;
    let config = safe_file(root, "config.json", None, None)?;
    let config_hash = format!("{:x}", Sha256::digest(&config));
    let profiles = inventory["profiles"].as_object()?;
    for profile in profiles.values() {
        let receipt_hash = profile["files_sha256"]["native-source.json"].as_str()?;
        let index_hash = profile["files_sha256"]["model.safetensors.index.json"].as_str()?;
        let Some(receipt_bytes) = safe_file(root, "native-source.json", None, Some(receipt_hash)) else {
            continue;
        };
        if profile["files_sha256"]["config.json"].as_str()? != config_hash
            || safe_file(root, "model.safetensors.index.json", None, Some(index_hash)).is_none()
        {
            continue;
        }
        let receipt_json: serde_json::Value = serde_json::from_slice(&receipt_bytes).ok()?;
        if receipt_json["artifact_mode"].as_str()? != "text"
            || receipt_json["weights_source"]["repo"] != profile["repo"]
            || receipt_json["weights_source"]["revision"] != profile["revision"]
            || receipt_json["weight_bytes"].as_u64()? != profile["weight_bytes"].as_u64()?
        {
            continue;
        }
        let payloads = profile["payloads"].as_object()?;
        if payloads.is_empty() || payloads.len() > 256 {
            continue;
        }
        let expected_names: std::collections::BTreeSet<_> = payloads.keys().map(String::as_str).collect();
        let mut actual_names = std::collections::BTreeSet::new();
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        let mut directory_ok = true;
        for entry in entries {
            let Ok(entry) = entry else {
                directory_ok = false;
                break;
            };
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                directory_ok = false;
                break;
            };
            if name.ends_with(".safetensors") || name.ends_with(".bin") {
                let Ok(metadata) = fs::symlink_metadata(entry.path()) else {
                    directory_ok = false;
                    break;
                };
                if !metadata.file_type().is_file() || !actual_names.insert(name.to_owned()) {
                    directory_ok = false;
                    break;
                }
            }
        }
        if !directory_ok
            || actual_names.iter().map(String::as_str).collect::<std::collections::BTreeSet<_>>() != expected_names
        {
            continue;
        }
        let mut valid = true;
        let mut payload_total = 0u128;
        for (name, size) in payloads {
            let Some(expected) = size.as_u64() else {
                valid = false;
                break;
            };
            payload_total = payload_total.saturating_add(u128::from(expected));
            if safe_file(root, name, Some(expected), None).is_none() {
                valid = false;
                break;
            }
        }
        if valid
            && payload_total == u128::from(profile["weight_bytes"].as_u64()?)
            && profile["paged_weight_bytes"].as_u64()? <= profile["weight_bytes"].as_u64()?
        {
            return Some(AdmissionPlan {
                request_headroom_divisor: 1,
                maximum_context_tokens,
                host_reserve_bytes,
                paged_working_set_bytes,
                maximum_mtp_drafts: 2,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use stark_engine_macros::stark_test;

    use super::*;

    struct Fixture {
        root: std::path::PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "stark-guided-admission-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            ));
            fs::create_dir(&root).unwrap();
            Self {
                root,
            }
        }

        fn inventory(&self) -> serde_json::Value {
            let config = b"{}";
            let receipt = json!({
                "artifact_mode":"text",
                "weights_source":{"repo":"example/model","revision":"0123456789012345678901234567890123456789"},
                "weight_bytes":4
            });
            let receipt = serde_json::to_vec(&receipt).unwrap();
            let index = b"{\"weight_map\":{}}";
            let payload = b"data";
            fs::write(self.root.join("config.json"), config).unwrap();
            fs::write(self.root.join("native-source.json"), &receipt).unwrap();
            fs::write(self.root.join("model.safetensors.index.json"), index).unwrap();
            fs::write(self.root.join("model-00001.safetensors"), payload).unwrap();
            json!({
                "schema_version":1,
                "policy":"qwen38-paged-text-v1",
                "maximum_context_tokens":32768,
                "request_headroom_divisor":1,
                "maximum_mtp_drafts":2,
                "host_reserve_bytes":8589934592u64,
                "paged_working_set_bytes":1073741824u64,
                "profiles":{
                    "fixture":{
                        "repo":"example/model",
                        "revision":"0123456789012345678901234567890123456789",
                        "weight_bytes":4,
                        "paged_weight_bytes":0,
                        "files_sha256":{
                            "config.json":format!("{:x}", Sha256::digest(config)),
                            "native-source.json":format!("{:x}", Sha256::digest(&receipt)),
                            "model.safetensors.index.json":format!("{:x}", Sha256::digest(index))
                        },
                        "payloads":{"model-00001.safetensors":4}
                    }
                }
            })
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[stark_test]
    fn shared_candidate_policy_is_bounded_and_exact() {
        let inventory: serde_json::Value = serde_json::from_str(INVENTORY).unwrap();
        assert_eq!(inventory["profiles"].as_object().unwrap().len(), 2);
        assert_eq!(inventory["maximum_context_tokens"], 32768);
        assert_eq!(inventory["request_headroom_divisor"], 1);
        assert_eq!(inventory["maximum_mtp_drafts"], 2);
        assert_eq!(inventory["host_reserve_bytes"], 8 * 1024_u64.pow(3));
        assert_eq!(inventory["paged_working_set_bytes"], 1024_u64.pow(3));
        assert!(qualified_plan(Path::new("/definitely/not/a/model")).is_none());
    }

    #[stark_test]
    fn exact_synthetic_identity_qualifies_and_changed_files_fail_closed() {
        let fixture = Fixture::new();
        let inventory = fixture.inventory();
        assert_eq!(
            qualified_plan_with_inventory(&fixture.root, &inventory),
            Some(AdmissionPlan {
                request_headroom_divisor: 1,
                maximum_context_tokens: 32768,
                host_reserve_bytes: 8 * 1024 * 1024 * 1024,
                paged_working_set_bytes: 1024 * 1024 * 1024,
                maximum_mtp_drafts: 2,
            })
        );
        fs::write(fixture.root.join("model-00001.safetensors"), b"bad!").unwrap();
        // Payload bytes are not rehashed: qualification binds memory layout,
        // not numerical identity. A size change must revoke the plan.
        assert!(qualified_plan_with_inventory(&fixture.root, &inventory).is_some());
        fs::write(fixture.root.join("model-00001.safetensors"), b"larger").unwrap();
        assert!(qualified_plan_with_inventory(&fixture.root, &inventory).is_none());
        fs::write(fixture.root.join("model-00001.safetensors"), b"data").unwrap();
        fs::write(fixture.root.join("config.json"), b"changed metadata").unwrap();
        assert!(qualified_plan_with_inventory(&fixture.root, &inventory).is_none());
    }

    #[stark_test]
    fn candidate_scope_is_limited_by_context_and_mtp() {
        let plan = AdmissionPlan {
            request_headroom_divisor: 1,
            maximum_context_tokens: 32768,
            host_reserve_bytes: 8 * 1024 * 1024 * 1024,
            paged_working_set_bytes: 1024 * 1024 * 1024,
            maximum_mtp_drafts: 2,
        };
        assert!(plan.applies_to(32768, 2));
        assert!(!plan.applies_to(32769, 0));
        assert!(!plan.applies_to(32768, 3));
    }

    #[stark_test]
    fn extra_payloads_symlinks_roots_and_policy_changes_fail_closed() {
        let fixture = Fixture::new();
        let inventory = fixture.inventory();
        fs::write(fixture.root.join("extra.safetensors"), b"data").unwrap();
        assert!(qualified_plan_with_inventory(&fixture.root, &inventory).is_none());
        fs::remove_file(fixture.root.join("extra.safetensors")).unwrap();

        #[cfg(unix)]
        {
            let payload = fixture.root.join("model-00001.safetensors");
            fs::remove_file(&payload).unwrap();
            std::os::unix::fs::symlink("config.json", &payload).unwrap();
            assert!(qualified_plan_with_inventory(&fixture.root, &inventory).is_none());
            fs::remove_file(&payload).unwrap();
            fs::write(&payload, b"data").unwrap();
        }

        let mut changed_policy = inventory.clone();
        changed_policy["host_reserve_bytes"] = json!(1);
        assert!(qualified_plan_with_inventory(&fixture.root, &changed_policy).is_none());

        let link = fixture.root.with_extension("link");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&fixture.root, &link).unwrap();
            assert!(qualified_plan_with_inventory(&link, &inventory).is_none());
            let _ = fs::remove_file(link);
        }
    }
}
