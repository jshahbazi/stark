use std::{
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use tracing_subscriber::fmt::writer::{MakeWriter, MutexGuardWriter};

#[derive(Clone)]
pub struct LogFile {
    file: Arc<Mutex<File>>,
    path: PathBuf,
}

impl LogFile {
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        // A fresh name prevents following a preexisting file or symlink.
        let mut options = OpenOptions::new();
        options.create_new(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            path: path.into(),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl<'a> MakeWriter<'a> for LogFile {
    type Writer = MutexGuardWriter<'a, File>;

    fn make_writer(&'a self) -> Self::Writer {
        self.file.make_writer()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::{MetadataExt, symlink};

    use super::*;

    #[test]
    fn logs_are_private_and_preexisting_paths_are_rejected() {
        let directory = std::env::temp_dir().join(format!("stark-log-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("session.log");
        let log = LogFile::new(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert!(LogFile::new(&path).is_err());
        let linked = directory.join("linked.log");
        symlink(&path, &linked).unwrap();
        assert!(LogFile::new(&linked).is_err());
        drop(log);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
