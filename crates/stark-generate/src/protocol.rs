//! Redirect native diagnostics before model initialization; only replies use stdout.
use std::{
    ffi::{OsStr, OsString},
    fs::{File, OpenOptions},
    io::{self, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::PathBuf,
};

fn arguments(mut args: impl Iterator<Item = OsString>) -> Result<PathBuf, &'static str> {
    if args.next().as_deref() != Some(OsStr::new("--model-path")) {
        return Err("invalid_arguments");
    }
    let path = args.next().ok_or("invalid_arguments")?;
    if path.is_empty() || args.next().is_some() {
        return Err("invalid_arguments");
    }
    Ok(PathBuf::from(path))
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
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let path = arguments(std::env::args_os().skip(1))?;
        #[cfg(feature = "metal")]
        {
            let backend = crate::native::NativeBackend::load(&path)?;
            let mut worker = crate::generation::GenerationWorker::new(backend)?;
            crate::generation::run_lines(&mut worker, &mut io::stdin().lock(), &mut output)
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = path;
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
    fn parse(args: &[&str]) -> Result<PathBuf, &'static str> {
        arguments(args.iter().map(OsString::from))
    }

    #[test]
    fn exactly_one_model_path_is_required() {
        assert_eq!(parse(&["--model-path", "/model"]), Ok(PathBuf::from("/model")));
        for args in [
            vec![],
            vec!["--model-path"],
            vec!["--model-path", ""],
            vec!["--score-model-path", "/model"],
            vec!["--model-path", "/model", "--consecutive-cache"],
            vec!["--model-path", "/model", "--model-path", "/other"],
        ] {
            assert_eq!(parse(&args), Err("invalid_arguments"));
        }
    }
}
