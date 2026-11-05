//! Link only an explicitly prepared, source-bound GLM bridge stage.
//! No network, native compilation or global installation occurs in Cargo.

use std::{
    env, fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn digest(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path).context("cannot read GLM bridge artifact")?;
    let mut hash = Sha256::new();
    let mut chunk = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        hash.update(&chunk[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn field<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a str> {
    value.get(key).and_then(Value::as_str).context("invalid GLM bridge stage identity")
}

fn relative_source(
    root: &Path,
    name: &str,
) -> Result<PathBuf> {
    let path = Path::new(name);
    ensure!(
        !path.as_os_str().is_empty() && path.components().all(|part| matches!(part, Component::Normal(_))),
        "invalid GLM bridge source path"
    );
    let path = root.join(path).canonicalize()?;
    ensure!(path.starts_with(root), "GLM bridge source escaped its directory");
    Ok(path)
}

pub fn link(
    target_arch: &str,
    target_os: &str,
) -> Result<()> {
    ensure!(target_arch == "aarch64" && target_os == "macos", "glm-exl3 requires macOS arm64 26.2+");
    println!("cargo::rerun-if-env-changed=STARK_GLM_EXL3_STAGE");
    println!("cargo::rerun-if-env-changed=STARK_GLM_EXL3_VALIDATE_PREFIX");
    println!("cargo::rerun-if-env-changed=STARK_GLM_EXL3_VALIDATE_NATIVE");
    println!("cargo::rustc-check-cfg=cfg(stark_glm_exl3_prefix_validation)");
    let vendor = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("../../vendor/glm-exl3-bridge").canonicalize()?;
    let lock_path = vendor.join("lock.json");
    println!("cargo::rerun-if-changed={}", lock_path.display());
    let lock: Value = serde_json::from_slice(&fs::read(&lock_path)?)?;
    ensure!(lock["lock_version"] == 1 && lock["abi_version"] == 1, "unsupported GLM bridge source lock");
    let sources = lock["owned_sources"].as_array().context("missing GLM bridge source inventory")?;
    let mut hash = Sha256::new();
    let mut prior = None;
    ensure!(!sources.is_empty(), "empty GLM bridge source inventory");
    for source in sources {
        let name = field(source, "path")?;
        ensure!(prior.is_none_or(|old| old < name), "unordered GLM bridge source inventory");
        prior = Some(name);
        let path = relative_source(&vendor, name)?;
        println!("cargo::rerun-if-changed={}", path.display());
        let bytes = fs::read(path)?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == field(source, "sha256")?,
            "GLM bridge source changed; rebuild its lock and stage"
        );
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(&bytes);
        hash.update([0]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == field(&lock, "bridge_source_sha256")?,
        "GLM bridge source bundle mismatch"
    );
    let stage = PathBuf::from(env::var_os("STARK_GLM_EXL3_STAGE").context(
        "set STARK_GLM_EXL3_STAGE to an explicitly built GLM bridge stage; see vendor/glm-exl3-bridge/UPSTREAM.md",
    )?)
    .canonicalize()?;
    let receipt_path = stage.join("stark-glm-stage.json");
    println!("cargo::rerun-if-changed={}", receipt_path.display());
    let receipt: Value = serde_json::from_slice(&fs::read(&receipt_path)?)?;
    ensure!(receipt["version"] == 1 && receipt["abi_version"] == 1, "unsupported GLM bridge stage ABI");
    for key in [
        "host_revision",
        "plugin_revision",
        "mlx_revision",
        "mlxc_revision",
        "mlxc_patch_sha256",
        "plugin_patch_sha256",
        "bridge_source_sha256",
        "mlxc_worktree_diff_sha256",
        "mlx_getter_patch_sha256",
        "mlx_worktree_diff_sha256",
        "dependency_headers_sha256",
        "sushi_revision",
        "zig_version",
        "zig_sha256",
    ] {
        ensure!(field(&receipt, key)? == field(&lock, key)?, "GLM bridge stage source mismatch: {key}");
    }
    let native_api_version = match receipt.get("native_decode_api_version") {
        None => 0,
        Some(value) => value.as_u64().context("invalid GLM native decode API version")?,
    };
    let expected_native_api = match lock.get("native_decode_api_version") {
        None => 0,
        Some(value) => value.as_u64().context("invalid GLM native source API version")?,
    };
    ensure!(
        native_api_version <= 1 && native_api_version == expected_native_api,
        "GLM native decode stage API mismatch"
    );
    if native_api_version == 1 {
        ensure!(
            field(&receipt, "native_decode_patch_sha256")? == field(&lock, "native_decode_patch_sha256")?,
            "GLM native decode patch mismatch"
        );
    }
    let private_native_caps = match receipt.get("private_native_caps") {
        None => false,
        Some(value) => value.as_bool().context("invalid GLM private native capability disposition")?,
    };
    let native_validation = match env::var("STARK_GLM_EXL3_VALIDATE_NATIVE").as_deref() {
        Err(env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => bail!("STARK_GLM_EXL3_VALIDATE_NATIVE must be 0 or 1"),
    };
    let experimental = receipt["experimental_prefix"].as_bool().context("missing GLM bridge prefix disposition")?;
    let profile_name = receipt
        .get("native_profile")
        .map(|value| value.as_str().context("invalid GLM native profile disposition"))
        .transpose()?;
    let profile = super::glm_exl3_profile::select(
        profile_name,
        private_native_caps,
        native_validation,
        native_api_version,
        experimental,
    )
    .map_err(anyhow::Error::msg)?;
    if profile == super::glm_exl3_profile::NativeProfile::Bf16NativeExperimental {
        ensure!(
            receipt.get("private_native_caps").and_then(serde_json::Value::as_bool) == Some(false),
            "experimental GLM profile requires an explicit non-private disposition"
        );
        ensure!(
            receipt["quality_qualified"] == false && receipt["throughput_qualified"] == false,
            "experimental GLM profile must retain its unqualified disposition"
        );
        println!("cargo::rustc-cfg=stark_glm_exl3_bf16_profile");
    }
    let validation = match env::var("STARK_GLM_EXL3_VALIDATE_PREFIX").as_deref() {
        Err(env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => bail!("STARK_GLM_EXL3_VALIDATE_PREFIX must be 0 or 1"),
    };
    ensure!(!validation || experimental, "GLM state validation requires a prefix-capable native stage");
    if experimental {
        println!("cargo::rustc-cfg=stark_glm_exl3_prefix_available");
    }
    if validation {
        println!("cargo::rustc-cfg=stark_glm_exl3_prefix_validation");
    }
    let lib_dir = PathBuf::from(field(&receipt, "mlx_lib_dir")?).canonicalize()?;
    let mut bridge_dir = None;
    for (key, filename) in [
        ("stark_glm", "libstark_glm.a"),
        ("mlx", "libmlx.dylib"),
        ("mlxc", "libmlxc.dylib"),
        ("jaccl", "libjaccl.dylib"),
        ("metallib", "mlx.metallib"),
    ] {
        let artifact = &receipt["artifacts"][key];
        let path = PathBuf::from(field(artifact, "path")?);
        let path = if path.is_absolute() {
            path
        } else {
            stage.join(path)
        }
        .canonicalize()?;
        ensure!(path.file_name().is_some_and(|name| name == filename), "invalid GLM bridge artifact name");
        if key == "stark_glm" {
            ensure!(path.starts_with(&stage), "GLM static library is outside its stage");
            bridge_dir = path.parent().map(Path::to_path_buf);
        } else {
            ensure!(path.parent() == Some(lib_dir.as_path()), "GLM MLX artifact directory mismatch");
        }
        println!("cargo::rerun-if-changed={}", path.display());
        let actual = digest(&path)?;
        ensure!(actual == field(artifact, "sha256")?, "GLM bridge artifact changed: {key}");
        if key != "stark_glm" {
            ensure!(
                actual == lock["dependency_artifacts"][key].as_str().context("missing GLM dependency artifact pin")?,
                "GLM dependency artifact differs from the qualified stage: {key}"
            );
        }
    }
    if native_api_version == 1 {
        println!("cargo::rustc-cfg=stark_glm_exl3_native_decode_stage");
    }
    if profile != super::glm_exl3_profile::NativeProfile::HeadlessReference {
        println!("cargo::rustc-cfg=stark_glm_exl3_native_decode_available");
    }
    println!("cargo::rustc-link-search=native={}", bridge_dir.context("missing GLM static library")?.display());
    println!("cargo::rustc-link-search=native={}", lib_dir.display());
    println!("cargo::metadata=glm_exl3_lib_dir={}", lib_dir.display());
    println!("cargo::rustc-link-lib=static=stark_glm");
    for name in ["mlxc", "mlx", "jaccl", "c++"] {
        println!("cargo::rustc-link-lib=dylib={name}");
    }
    for name in ["IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface"] {
        println!("cargo::rustc-link-lib=framework={name}");
    }
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    Ok(())
}
