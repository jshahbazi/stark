//! Redirect native diagnostics before model initialization; only replies use stdout.
use std::{
    ffi::{OsStr, OsString},
    fs::{File, OpenOptions},
    io::{self, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::PathBuf,
};

#[derive(Debug, PartialEq)]
struct Arguments {
    model: PathBuf,
    staging: Option<PathBuf>,
    prefix_cache: bool,
    validation_gemma_text_only: bool,
}

fn arguments(mut args: impl Iterator<Item = OsString>) -> Result<Arguments, &'static str> {
    if args.next().as_deref() != Some(OsStr::new("--model-path")) {
        return Err("invalid_arguments");
    }
    let path = args.next().ok_or("invalid_arguments")?;
    if path.is_empty() {
        return Err("invalid_arguments");
    }
    let mut parsed = Arguments {
        model: PathBuf::from(path),
        staging: None,
        prefix_cache: false,
        validation_gemma_text_only: false,
    };
    while let Some(flag) = args.next() {
        if flag == OsStr::new("--vision-staging-root") && parsed.staging.is_none() {
            let root = args.next().filter(|root| !root.is_empty()).ok_or("invalid_arguments")?;
            parsed.staging = Some(PathBuf::from(root));
        } else if flag == OsStr::new("--prefix-cache") && !parsed.prefix_cache {
            parsed.prefix_cache = true;
        } else if cfg!(feature = "gemma4-assistant-staging")
            && flag == OsStr::new("--validation-gemma-text-only")
            && !parsed.validation_gemma_text_only
        {
            parsed.validation_gemma_text_only = true;
        } else {
            return Err("invalid_arguments");
        }
    }
    Ok(parsed)
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
        let Arguments {
            model: path,
            staging,
            prefix_cache,
            validation_gemma_text_only,
        } = arguments(std::env::args_os().skip(1))?;
        // Validate operator resource policy before model loading allocates native state.
        let _request_budget = stark_resource_budget::Budget::from_environment()?;
        #[cfg(feature = "metal")]
        {
            let cancellation = crate::cancel::CancelSignal::from_environment()?;
            // SAFETY: startup is single-threaded here, and the inherited pipe
            // write FD is transferred exclusively to the sampler.
            _gpu_memory_sampler = unsafe { stark_engine::backends::metal::memory_usage::start_from_environment() };
            let media = staging.as_deref().map(crate::media::MediaStore::open).transpose()?;
            let hybrid = {
                match stark_engine::engine::hybrid::recognizes(&path) {
                    Ok(hybrid) => hybrid,
                    // Preserve the ordinary loader's established unavailable-
                    // config error while keeping malformed metadata bounded.
                    Err(stark_engine::engine::hybrid::Error::Io(_)) => false,
                    Err(_) => return Err("invalid_model_config"),
                }
            };
            if hybrid {
                if validation_gemma_text_only {
                    return Err("validation_family_not_supported");
                }
                let backend = if media.is_some() {
                    crate::hybrid::HybridBackend::load_vision(&path, cancellation)?
                } else {
                    crate::hybrid::HybridBackend::load(&path, cancellation)?
                };
                let mut worker = crate::generation::GenerationWorker::new_with_options(backend, media, prefix_cache)?;
                return crate::generation::run_lines(
                    &mut worker,
                    &mut crate::prefix_cache::StandardInput::reader(),
                    &mut output,
                );
            }
            let backend = if validation_gemma_text_only {
                crate::native::NativeBackend::load_gemma4_text_validation(&path, cancellation)?
            } else if media.is_some() {
                crate::native::NativeBackend::load_vision(&path, cancellation)?
            } else {
                crate::native::NativeBackend::load(&path, cancellation)?
            };
            let mut worker = crate::generation::GenerationWorker::new_with_options(backend, media, prefix_cache)?;
            crate::generation::run_lines(&mut worker, &mut crate::prefix_cache::StandardInput::reader(), &mut output)
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (path, staging, prefix_cache, validation_gemma_text_only);
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
    fn parse(args: &[&str]) -> Result<(PathBuf, Option<PathBuf>, bool), &'static str> {
        arguments(args.iter().map(OsString::from)).map(|parsed| (parsed.model, parsed.staging, parsed.prefix_cache))
    }

    #[test]
    fn exactly_one_model_path_is_required() {
        assert_eq!(parse(&["--model-path", "/model"]), Ok((PathBuf::from("/model"), None, false)));
        assert_eq!(
            parse(&["--model-path", "/model", "--vision-staging-root", "/media"]),
            Ok((PathBuf::from("/model"), Some(PathBuf::from("/media")), false))
        );
        assert_eq!(parse(&["--model-path", "/model", "--prefix-cache"]), Ok((PathBuf::from("/model"), None, true)));
        assert_eq!(
            parse(&["--model-path", "/model", "--prefix-cache", "--vision-staging-root", "/media"]),
            Ok((PathBuf::from("/model"), Some(PathBuf::from("/media")), true))
        );
        assert_eq!(
            parse(&["--model-path", "/model", "--vision-staging-root", "/media", "--prefix-cache"]),
            Ok((PathBuf::from("/model"), Some(PathBuf::from("/media")), true))
        );
        for args in [
            vec!["--model-path", "/model", "--prefix-cache", "--prefix-cache"],
            vec!["--prefix-cache", "--model-path", "/model"],
            vec!["--model-path", "/model", "--prefix-cache=1"],
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

    #[test]
    fn gemma_text_comparison_is_explicit_and_staging_only() {
        let ordinary = arguments(["--model-path", "/model"].into_iter().map(OsString::from)).unwrap();
        assert!(!ordinary.validation_gemma_text_only);
        let validation =
            arguments(["--model-path", "/model", "--validation-gemma-text-only"].into_iter().map(OsString::from));
        if cfg!(feature = "gemma4-assistant-staging") {
            assert!(validation.unwrap().validation_gemma_text_only);
        } else {
            assert_eq!(validation, Err("invalid_arguments"));
        }
    }
}
