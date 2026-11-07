//! Keep explicitly requested model-bound diagnostic artifacts private.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn ignored_destination(path: &Path) -> Result<PathBuf, &'static str> {
    let parent = path.parent().ok_or("output_parent_missing")?.canonicalize().map_err(|_| "output_parent_invalid")?;
    let destination = parent.join(path.file_name().ok_or("output_filename_missing")?);
    if destination.symlink_metadata().is_ok() {
        return Err("output_exists");
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ignored = Command::new("git")
        .current_dir(root)
        .args(["check-ignore", "--quiet", "--"])
        .arg(&destination)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| "output_ignore_check_failed")?;
    if !ignored.success() {
        return Err("output_not_ignored");
    }
    Ok(destination)
}

pub fn create_private_directory(path: &Path) -> Result<(), &'static str> {
    let destination = ignored_destination(path)?;
    fs::DirBuilder::new().mode(0o700).create(destination).map_err(|_| "output_create_failed")
}

pub fn private_write(
    path: &Path,
    contents: &[u8],
) -> Result<(), &'static str> {
    let destination = ignored_destination(path)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(destination)
        .map_err(|_| "output_create_failed")?;
    file.write_all(contents).map_err(|_| "output_write_failed")
}

#[cfg(test)]
mod tests {
    use std::{os::unix::fs::PermissionsExt, time::SystemTime};

    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn private_diagnostics_reject_public_paths_and_overwrite_and_set_permissions() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let suffix = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let name = format!("private-output-check-{}-{suffix}", std::process::id());
        let public = root.join("docs").join(&name);
        assert_eq!(create_private_directory(&public), Err("output_not_ignored"));
        assert!(!public.exists());
        let runtime = root.join("runtime");
        fs::create_dir_all(&runtime).unwrap();
        let output = runtime.join(&name);
        create_private_directory(&output).unwrap();
        assert_eq!(fs::metadata(&output).unwrap().permissions().mode() & 0o777, 0o700);
        assert_eq!(create_private_directory(&output), Err("output_exists"));
        let report = output.join("report.json");
        private_write(&report, b"{}").unwrap();
        assert_eq!(fs::metadata(&report).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(private_write(&report, b"replacement"), Err("output_exists"));
        assert_eq!(fs::read(&report).unwrap(), b"{}");
        fs::remove_file(report).unwrap();
        fs::remove_dir(output).unwrap();
    }
}
