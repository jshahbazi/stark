//! Redirect native diagnostics before model initialization; only replies use stdout.
use std::{
    ffi::{OsStr, OsString},
    fs::{File, OpenOptions},
    io::{self, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::PathBuf,
};

fn arguments(mut args: impl Iterator<Item = OsString>) -> Result<(PathBuf, Option<PathBuf>), &'static str> {
    if args.next().as_deref() != Some(OsStr::new("--model-path")) {
        return Err("invalid_arguments");
    }
    let path = args.next().ok_or("invalid_arguments")?;
    if path.is_empty() {
        return Err("invalid_arguments");
    }
    let staging = match args.next() {
        None => None,
        Some(flag) if flag == OsStr::new("--vision-staging-root") => {
            let root = args.next().filter(|root| !root.is_empty()).ok_or("invalid_arguments")?;
            if args.next().is_some() {
                return Err("invalid_arguments");
            }
            Some(PathBuf::from(root))
        },
        Some(_) => return Err("invalid_arguments"),
    };
    Ok((PathBuf::from(path), staging))
}

fn protocol_output() -> Result<File, &'static str> {
    let null = OpenOptions::new().write(true).open("/dev/null").map_err(|_| "output_setup_failed")?;
    let fd = unsafe { libc::fcntl(libc::STDOUT_FILENO, libc::F_DUPFD_CLOEXEC, 3) };
    if fd < 0 {
        return Err("output_setup_failed");
    }
    let output = unsafe { File::from_raw_fd(fd) };
    if unsafe { libc::dup2(null.as_raw_fd(), libc::STDOUT_FILENO) } < 0
        || unsafe { libc::dup2(null.as_raw_fd(), libc::STDERR_FILENO) } < 0
    {
        return Err("output_setup_failed");
    }
    Ok(output)
}

pub fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let Ok(output) = protocol_output() else {
        std::process::exit(1)
    };
    // Remove capture settings before the engine can create native worker threads.
    unsafe {
        for (name, _) in std::env::vars_os() {
            let key = name.to_string_lossy();
            if key.starts_with("STARK_CAPTURE_") || key.starts_with("UZU_CAPTURE_") {
                std::env::remove_var(name);
            }
        }
    }
    let mut output = io::BufWriter::new(output);
    #[cfg(feature = "metal")]
    let mut _gpu_memory_sampler = None;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (path, staging) = arguments(std::env::args_os().skip(1))?;
        #[cfg(feature = "metal")]
        {
            let cancellation = crate::cancel::CancelSignal::from_environment()?;
            // SAFETY: startup is single-threaded here, and the inherited pipe
            // write FD is transferred exclusively to the sampler.
            _gpu_memory_sampler = unsafe { stark_engine::backends::metal::memory_usage::start_from_environment() };
            let media = staging.as_deref().map(crate::media::MediaStore::open).transpose()?;
            let hybrid = if media.is_none() {
                match stark_engine::engine::hybrid::recognizes(&path) {
                    Ok(hybrid) => hybrid,
                    // Preserve the ordinary loader's established unavailable-
                    // config error while keeping malformed metadata bounded.
                    Err(stark_engine::engine::hybrid::Error::Io(_)) => false,
                    Err(stark_engine::engine::hybrid::Error::ConversionRequired) => {
                        return Err("model_conversion_required");
                    },
                    Err(_) => return Err("invalid_model_config"),
                }
            } else {
                false
            };
            if hybrid {
                if media.is_some() {
                    return Err("images_not_supported");
                }
                let backend = crate::hybrid::HybridBackend::load(&path, cancellation)?;
                let mut worker = crate::generation::GenerationWorker::new_with_media(backend, None)?;
                return crate::generation::run_lines(&mut worker, &mut io::stdin().lock(), &mut output);
            }
            let backend = if media.is_some() {
                crate::native::NativeBackend::load_vision(&path, cancellation)?
            } else {
                crate::native::NativeBackend::load(&path, cancellation)?
            };
            let mut worker = crate::generation::GenerationWorker::new_with_media(backend, media)?;
            crate::generation::run_lines(&mut worker, &mut io::stdin().lock(), &mut output)
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (path, staging);
            Err::<(), _>("native_backend_unavailable")
        }
    }));
    let code = match result {
        Ok(Ok(())) => return,
        Ok(Err(code)) => code,
        Err(_) => "native_operation_failed",
    };
    let _ = serde_json::to_writer(&mut output, &serde_json::json!({"ready":false,"error":code}));
    let _ = output.write_all(b"\n");
    let _ = output.flush();
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(args: &[&str]) -> Result<(PathBuf, Option<PathBuf>), &'static str> {
        arguments(args.iter().map(OsString::from))
    }

    #[test]
    fn exactly_one_model_path_is_required() {
        assert_eq!(parse(&["--model-path", "/model"]), Ok((PathBuf::from("/model"), None)));
        assert_eq!(
            parse(&["--model-path", "/model", "--vision-staging-root", "/media"]),
            Ok((PathBuf::from("/model"), Some(PathBuf::from("/media"))))
        );
        for args in [
            vec![],
            vec!["--model-path"],
            vec!["--model-path", ""],
            vec!["--score-model-path", "/model"],
            vec!["--model-path", "/model", "--consecutive-cache"],
            vec!["--model-path", "/model", "--model-path", "/other"],
            vec!["--model-path", "/model", "--vision-staging-root"],
            vec!["--model-path", "/model", "--vision-staging-root", ""],
            vec!["--model-path", "/model", "--vision-staging-root", "/media", "--vision-staging-root", "/other"],
        ] {
            assert_eq!(parse(&args), Err("invalid_arguments"));
        }
    }
}
