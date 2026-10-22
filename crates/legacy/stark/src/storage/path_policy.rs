use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

pub(super) fn safe_component(value: &str) -> bool {
    safe_relative(value) && !value.contains('/')
}

pub(super) fn safe_relative(value: &str) -> bool {
    if value.is_empty() || value.contains(['\\', ':', '\0']) {
        return false;
    }
    value.split('/').all(|segment| {
        !segment.is_empty()
            && segment != "."
            && segment != ".."
            && ![".part", ".resume_data", ".checksum", ".lock"]
                .iter()
                .any(|suffix| segment.to_ascii_lowercase().ends_with(suffix))
    }) && Path::new(value).components().all(|component| matches!(component, Component::Normal(_)))
}

/// Reject links in every existing component, including the destination. The
/// private cache root belongs to this process owner; registry payloads can
/// create regular files only and cannot replace components with links.
pub(super) fn checked_join(
    root: &Path,
    relative: &str,
) -> io::Result<PathBuf> {
    if !safe_relative(relative) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid model cache path"));
    }
    let mut path = root.to_path_buf();
    for component in Path::new(relative).components() {
        path.push(component);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(io::Error::new(io::ErrorKind::PermissionDenied, "model cache links are forbidden"));
            },
            Ok(_) => {},
            Err(error) if error.kind() == io::ErrorKind::NotFound => {},
            Err(error) => return Err(error),
        }
    }
    Ok(path)
}

pub(super) fn prepare_root(path: &Path) -> io::Result<PathBuf> {
    fs::create_dir_all(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "invalid model cache root"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        unsafe extern "C" {
            fn geteuid() -> u32;
        }
        // SAFETY: geteuid has no arguments and cannot invalidate Rust state.
        if metadata.uid() != unsafe { geteuid() } {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "model cache has a foreign owner"));
        }
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    fs::canonicalize(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_paths_cannot_escape_or_alias_sidecars() {
        for invalid in [
            "",
            ".",
            "..",
            "../victim",
            "/victim",
            "a/../victim",
            "a//b",
            "a/./b",
            "C:\\victim",
            "a\\b",
            "a.lock",
            "a.PART",
            "a.checksum",
            "a.resume_data",
        ] {
            assert!(!safe_relative(invalid), "accepted {invalid}");
        }
        assert!(safe_relative("nested/model.safetensors"));
        assert!(safe_component("revision-123"));
        assert!(!safe_component("revision/123"));
    }

    #[cfg(unix)]
    #[test]
    fn existing_symlinks_are_rejected_before_any_file_effect() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("victim"), b"synthetic-victim").unwrap();
        symlink(outside.path(), root.path().join("nested")).unwrap();
        assert!(checked_join(root.path(), "nested/victim").is_err());
        symlink(outside.path().join("victim"), root.path().join("file")).unwrap();
        assert!(checked_join(root.path(), "file").is_err());
        assert_eq!(fs::read(outside.path().join("victim")).unwrap(), b"synthetic-victim");
        assert_eq!(checked_join(root.path(), "normal/file").unwrap(), root.path().join("normal/file"));
    }
}
