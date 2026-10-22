use std::{
    fs::{self},
    io::{self},
    path::{Path, PathBuf},
};

use chrono::{Local, NaiveDateTime, TimeDelta};
use tracing_subscriber::fmt::writer::MakeWriter;
use uuid::Uuid;

use crate::server::log::log_file::LogFile;

const LOG_FILE_DATE_FORMAT: &str = "%Y-%m-%d-%H-%M-%S";
const LOG_FILE_EXTENSION: &str = "log";
const LOG_FILE_RETENTION: TimeDelta = TimeDelta::weeks(1);

#[derive(Clone)]
pub struct FileLogger {
    file: LogFile,
}

impl FileLogger {
    pub fn new(logs_dir_path: PathBuf) -> io::Result<Self> {
        ensure_private_directory(&logs_dir_path)?;
        Self::remove_expired_log_files(&logs_dir_path)?;

        let date = Local::now().format(LOG_FILE_DATE_FORMAT);
        let session_id = Uuid::new_v4().simple().to_string();
        let file_name = format!("{}-{}.{}", date, session_id, LOG_FILE_EXTENSION);
        let file = LogFile::new(logs_dir_path.join(file_name))?;
        Ok(Self {
            file,
        })
    }

    pub fn file_path(&self) -> &PathBuf {
        self.file.path()
    }

    fn remove_expired_log_files(logs_dir_path: &Path) -> io::Result<()> {
        let cutoff = Local::now().naive_local() - LOG_FILE_RETENTION;

        for entry in fs::read_dir(logs_dir_path)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }

            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != LOG_FILE_EXTENSION) {
                continue;
            }

            let Some(file_stem) = path.file_stem().and_then(|file_stem| file_stem.to_str()) else {
                continue;
            };
            let Some((date, _session_id)) = file_stem.rsplit_once('-') else {
                continue;
            };
            let Ok(created_at) = NaiveDateTime::parse_from_str(date, LOG_FILE_DATE_FORMAT) else {
                continue;
            };

            if created_at < cutoff {
                fs::remove_file(path)?;
            }
        }

        Ok(())
    }
}

fn ensure_private_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};
        unsafe extern "C" {
            fn geteuid() -> u32;
        }
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder.create(path)?;
        let metadata = fs::symlink_metadata(path)?;
        // SAFETY: geteuid has no arguments and cannot invalidate Rust state.
        let owner = unsafe { geteuid() };
        if !metadata.is_dir()
            || metadata.file_type().is_symlink()
            || metadata.mode() & 0o077 != 0
            || metadata.uid() != owner
        {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "log directory must be private"));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path)
}

impl<'a> MakeWriter<'a> for FileLogger {
    type Writer = <LogFile as MakeWriter<'a>>::Writer;

    fn make_writer(&'a self) -> Self::Writer {
        self.file.make_writer()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};

    use super::*;

    #[test]
    fn logger_requires_an_owned_private_directory() {
        let root = std::env::temp_dir().join(format!("stark-directory-test-{}", Uuid::new_v4()));
        let directory = root.join("logs");
        let logger = FileLogger::new(directory.clone()).unwrap();
        assert_eq!(fs::metadata(&directory).unwrap().mode() & 0o777, 0o700);
        drop(logger);
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(FileLogger::new(directory.clone()).is_err());
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        let link = root.join("linked");
        symlink(&directory, &link).unwrap();
        assert!(FileLogger::new(link).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
