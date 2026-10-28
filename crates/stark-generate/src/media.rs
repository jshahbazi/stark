//! Bounded access to host-staged image objects through one private directory.

use std::{
    ffi::{CStr, CString},
    fs::{File, OpenOptions},
    io::Read,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::Path,
};

use serde::Deserialize;

pub(crate) const MAX_COMPRESSED_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageReference {
    pub id: String,
    pub byte_length: u64,
    pub media_type: String,
}

pub(crate) struct ImageLease {
    _file: File,
}

pub(crate) struct StagedImage {
    pub bytes: Vec<u8>,
    pub lease: ImageLease,
}

pub(crate) struct MediaStore {
    directory: File,
}

pub(crate) fn valid_id(value: &str) -> bool {
    value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn opaque_id() -> Result<String, &'static str> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|_| "image_request_setup_failed")?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

impl MediaStore {
    pub(crate) fn open(path: &Path) -> Result<Self, &'static str> {
        if !path.is_absolute() {
            return Err("invalid_vision_staging_root");
        }
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .map_err(|_| "invalid_vision_staging_root")?;
        let metadata = directory.metadata().map_err(|_| "invalid_vision_staging_root")?;
        if !metadata.is_dir()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err("invalid_vision_staging_root");
        }
        if !directory_names(directory.as_raw_fd()).map_err(|_| "invalid_vision_staging_root")?.is_empty() {
            return Err("invalid_vision_staging_root");
        }
        Ok(Self {
            directory,
        })
    }

    pub(crate) fn read(
        &self,
        reference: &ImageReference,
    ) -> Result<StagedImage, &'static str> {
        if !valid_id(&reference.id)
            || reference.byte_length == 0
            || reference.byte_length > MAX_COMPRESSED_BYTES
            || !matches!(reference.media_type.as_str(), "image/jpeg" | "image/png")
        {
            return Err("invalid_image_reference");
        }
        let name = CString::new(reference.id.as_bytes()).map_err(|_| "invalid_image_reference")?;
        // Nonblocking open prevents a forged FIFO from blocking before fstat.
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            )
        };
        if fd < 0 {
            return Err("invalid_image_reference");
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let before = file.metadata().map_err(|_| "invalid_image_reference")?;
        if !before.is_file()
            || before.uid() != unsafe { libc::geteuid() }
            || before.permissions().mode() & 0o077 != 0
            || before.nlink() != 1
            || before.len() != reference.byte_length
        {
            return Err("invalid_image_reference");
        }
        let mut bytes = Vec::with_capacity(reference.byte_length as usize);
        Read::by_ref(&mut file)
            .take(reference.byte_length + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "image_read_failed")?;
        let after = file.metadata().map_err(|_| "image_read_failed")?;
        if bytes.len() as u64 != reference.byte_length
            || after.len() != before.len()
            || after.mtime() != before.mtime()
            || after.mtime_nsec() != before.mtime_nsec()
            || after.ctime() != before.ctime()
            || after.ctime_nsec() != before.ctime_nsec()
        {
            return Err("invalid_image_reference");
        }
        let correct_format = match reference.media_type.as_str() {
            "image/png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "image/jpeg" => bytes.starts_with(&[0xff, 0xd8, 0xff]),
            _ => false,
        };
        if !correct_format {
            return Err("invalid_image_reference");
        }
        // Remove the claimed name now so an abrupt host exit during native
        // work cannot leave the encoded image behind. Keep the opened object
        // through terminal completion so its bytes remain request-owned.
        if unsafe { libc::unlinkat(self.directory.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err("image_read_failed");
        }
        Ok(StagedImage {
            bytes,
            lease: ImageLease {
                _file: file,
            },
        })
    }
}

impl Drop for MediaStore {
    fn drop(&mut self) {
        let Ok(names) = directory_names(self.directory.as_raw_fd()) else {
            return;
        };
        for name in names {
            let Ok(identifier) = std::str::from_utf8(&name) else {
                continue;
            };
            if !valid_id(identifier) {
                continue;
            }
            let Ok(name) = CString::new(name) else {
                continue;
            };
            let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
            if unsafe {
                libc::fstatat(
                    self.directory.as_raw_fd(),
                    name.as_ptr(),
                    metadata.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } != 0
            {
                continue;
            }
            let metadata = unsafe { metadata.assume_init() };
            let mode = metadata.st_mode & libc::S_IFMT;
            if mode != libc::S_IFREG
                || metadata.st_uid != unsafe { libc::geteuid() }
                || metadata.st_mode & 0o077 != 0
                || metadata.st_nlink != 1
            {
                continue;
            }
            unsafe {
                libc::unlinkat(self.directory.as_raw_fd(), name.as_ptr(), 0);
            }
        }
    }
}

fn directory_names(directory_fd: libc::c_int) -> std::io::Result<Vec<Vec<u8>>> {
    let dot = c".";
    let duplicate = unsafe {
        libc::openat(
            directory_fd,
            dot.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if duplicate < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stream = unsafe { libc::fdopendir(duplicate) };
    if stream.is_null() {
        let error = std::io::Error::last_os_error();
        unsafe { libc::close(duplicate) };
        return Err(error);
    }
    let mut names = Vec::new();
    loop {
        #[cfg(target_os = "macos")]
        unsafe {
            *libc::__error() = 0;
        }
        #[cfg(target_os = "linux")]
        unsafe {
            *libc::__errno_location() = 0;
        }
        let entry = unsafe { libc::readdir(stream) };
        if entry.is_null() {
            #[cfg(target_os = "macos")]
            let error = unsafe { *libc::__error() };
            #[cfg(target_os = "linux")]
            let error = unsafe { *libc::__errno_location() };
            unsafe { libc::closedir(stream) };
            return if error == 0 {
                Ok(names)
            } else {
                Err(std::io::Error::from_raw_os_error(error))
            };
        }
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
        if name != b"." && name != b".." {
            names.push(name.to_vec());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        fs,
        io::{Read, Seek, SeekFrom},
        os::unix::fs::symlink,
        path::PathBuf,
    };

    use super::*;

    pub(crate) struct PrivateRoot(pub PathBuf);

    impl PrivateRoot {
        pub(crate) fn new() -> Self {
            let path = std::env::temp_dir().join(format!("stark-worker-media-{}", opaque_id().unwrap()));
            fs::create_dir(&path).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }

        pub(crate) fn image(&self) -> ImageReference {
            let id = opaque_id().unwrap();
            let bytes = b"\x89PNG\r\n\x1a\nfixture";
            let path = self.0.join(&id);
            fs::write(&path, bytes).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            ImageReference {
                id,
                byte_length: bytes.len() as u64,
                media_type: "image/png".into(),
            }
        }
    }

    impl Drop for PrivateRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn staging_root_and_objects_are_owner_private_regular_files() {
        let root = PrivateRoot::new();
        let store = MediaStore::open(&root.0).unwrap();
        let reference = root.image();
        let staged = store.read(&reference).unwrap();
        assert_eq!(staged.bytes, b"\x89PNG\r\n\x1a\nfixture");
        assert!(!root.0.join(&reference.id).exists());
        let mut retained = staged.lease._file;
        retained.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new();
        retained.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, staged.bytes);

        let reference = root.image();
        fs::set_permissions(root.0.join(&reference.id), fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(store.read(&reference), Err("invalid_image_reference")));
        fs::set_permissions(&root.0, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(MediaStore::open(&root.0), Err("invalid_vision_staging_root")));
        assert!(matches!(MediaStore::open(Path::new("relative")), Err("invalid_vision_staging_root")));
    }

    #[test]
    fn startup_rejects_nonempty_root_without_removing_entries() {
        let root = PrivateRoot::new();
        let reference = root.image();
        let path = root.0.join(&reference.id);
        assert!(matches!(MediaStore::open(&root.0), Err("invalid_vision_staging_root")));
        assert!(path.exists());
    }

    #[test]
    fn drop_removes_only_unclaimed_private_staged_objects() {
        let root = PrivateRoot::new();
        let store = MediaStore::open(&root.0).unwrap();
        let partial = root.0.join(opaque_id().unwrap());
        fs::write(&partial, b"partial staged bytes").unwrap();
        fs::set_permissions(&partial, fs::Permissions::from_mode(0o600)).unwrap();
        let unknown = root.0.join("keep-this");
        fs::write(&unknown, b"unknown").unwrap();
        let symlink_target = root.0.join("keep-target");
        fs::write(&symlink_target, b"target").unwrap();
        symlink(&symlink_target, root.0.join(opaque_id().unwrap())).unwrap();
        drop(store);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 3);
        assert!(!partial.exists());
        assert!(unknown.exists());
        assert!(symlink_target.exists());
    }

    #[test]
    fn references_reject_paths_symlinks_hardlinks_lengths_and_mime_mismatches() {
        let root = PrivateRoot::new();
        let store = MediaStore::open(&root.0).unwrap();
        for bad in ["../other", "/absolute", "", "A123456789abcdef0123456789abcdef0"] {
            let mut reference = root.image();
            reference.id = bad.into();
            assert!(matches!(store.read(&reference), Err("invalid_image_reference")));
        }
        let mut reference = root.image();
        reference.byte_length += 1;
        assert!(store.read(&reference).is_err());
        reference.byte_length = MAX_COMPRESSED_BYTES + 1;
        assert!(store.read(&reference).is_err());
        let mut reference = root.image();
        reference.media_type = "image/jpeg".into();
        assert!(store.read(&reference).is_err());
        let reference = root.image();
        let alias = root.0.join(opaque_id().unwrap());
        fs::hard_link(root.0.join(&reference.id), &alias).unwrap();
        assert!(store.read(&reference).is_err());
        let reference = root.image();
        fs::remove_file(root.0.join(&reference.id)).unwrap();
        symlink(&alias, root.0.join(&reference.id)).unwrap();
        assert!(store.read(&reference).is_err());
        let reference = root.image();
        fs::remove_file(root.0.join(&reference.id)).unwrap();
        let name = CString::new(root.0.join(&reference.id).as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        assert!(store.read(&reference).is_err());
    }
}
