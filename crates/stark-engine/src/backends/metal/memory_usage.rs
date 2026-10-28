//! Optional, best-effort live Metal allocation telemetry for worker hosts.
use std::{
    fs::File,
    io,
    os::fd::{FromRawFd, RawFd},
    thread,
    time::Duration,
};

use metal::MTLDevice;
use objc2::runtime::ProtocolObject;

const MEMORY_FD_ENV: &str = "STARK_GPU_MEMORY_FD";
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Start the opt-in sampler when the host supplied a valid nonblocking pipe FD.
/// The returned handle may be dropped to detach the sampler thread.
///
/// # Safety
/// Call only during single-threaded startup, before other threads can access
/// the process environment, and only when the descriptor named by
/// `STARK_GPU_MEMORY_FD` is exclusively owned by this process and may be closed.
pub unsafe fn start_from_environment() -> Option<thread::JoinHandle<()>> {
    let descriptor = std::env::var_os(MEMORY_FD_ENV);
    // SAFETY: called during single-threaded worker startup, before the sampler
    // or engine can create threads. Removing the opt-in prevents inherited
    // descriptor settings from leaking into child processes.
    unsafe { std::env::remove_var(MEMORY_FD_ENV) };
    let fd = descriptor?.to_string_lossy().parse::<RawFd>().ok()?;
    if fd < 3 || !is_nonblocking_pipe_writer(fd) {
        return None;
    }
    // SAFETY: descriptor validation above confirms this is an owned, open FD;
    // constructing File transfers ownership to the sampler thread.
    let output = unsafe { File::from_raw_fd(fd) };
    // If Metal is unavailable, dropping output closes the inherited descriptor.
    let device = <dyn MTLDevice>::system_default()?;
    thread::Builder::new().name("stark-gpu-memory".into()).spawn(move || sample(device, output)).ok()
}

fn is_nonblocking_pipe_writer(fd: RawFd) -> bool {
    // SAFETY: fcntl/fstat only inspect the supplied descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || flags & libc::O_NONBLOCK == 0 || flags & libc::O_ACCMODE != libc::O_WRONLY {
        return false;
    }
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: metadata points to writable storage for struct stat.
    if unsafe { libc::fstat(fd, metadata.as_mut_ptr()) } != 0 {
        return false;
    }
    // SAFETY: fstat initialized metadata on success.
    let metadata = unsafe { metadata.assume_init() };
    metadata.st_mode & libc::S_IFMT == libc::S_IFIFO
}

fn sample(
    device: objc2::rc::Retained<ProtocolObject<dyn MTLDevice>>,
    mut output: File,
) {
    loop {
        thread::sleep(SAMPLE_INTERVAL);
        // Apple defines currentAllocatedSize as bytes allocated by resources
        // created on this device; it is not total system GPU memory usage.
        let line = format!("{{\"gpu_memory_bytes\":{}}}\n", device.current_allocated_size());
        // Each record is far below PIPE_BUF. A nonblocking full pipe drops a
        // sample; every other write failure ends telemetry without affecting inference.
        match write_record(&mut output, line.as_bytes()) {
            Ok(()) => {},
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            Err(_) => return,
        }
    }
}

fn write_record(
    output: &mut File,
    bytes: &[u8],
) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: bytes remains valid for the duration of write; FD is owned by output.
    let written = unsafe { libc::write(output.as_raw_fd(), bytes.as_ptr().cast(), bytes.len()) };
    if written < 0 {
        return Err(io::Error::last_os_error());
    }
    if written as usize != bytes.len() {
        return Err(io::Error::new(io::ErrorKind::WriteZero, "partial telemetry record"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::fd::{AsRawFd, FromRawFd};

    use stark_engine_macros::stark_test;

    use super::*;

    fn nonblocking_pipe() -> (File, File) {
        let mut descriptors = [-1; 2];
        // SAFETY: pipe initializes both descriptor slots on success.
        assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
        // SAFETY: pipe returned two newly owned descriptors.
        let reader = unsafe { File::from_raw_fd(descriptors[0]) };
        // SAFETY: pipe returned two newly owned descriptors.
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        // SAFETY: fcntl only inspects and updates the owned write descriptor.
        let flags = unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_GETFL) };
        assert!(flags >= 0);
        // SAFETY: fcntl only updates the owned write descriptor.
        assert_eq!(unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) }, 0);
        (reader, writer)
    }

    #[stark_test]
    fn accepts_only_a_nonblocking_pipe_writer() {
        let (reader, writer) = nonblocking_pipe();
        assert!(is_nonblocking_pipe_writer(writer.as_raw_fd()));
        assert!(!is_nonblocking_pipe_writer(reader.as_raw_fd()));
    }

    #[stark_test]
    fn full_pipe_drops_sample_without_blocking() {
        let (_reader, mut writer) = nonblocking_pipe();
        let chunk = [0_u8; 4096];
        loop {
            // SAFETY: chunk remains readable for the duration of write.
            let written = unsafe { libc::write(writer.as_raw_fd(), chunk.as_ptr().cast(), chunk.len()) };
            if written < 0 {
                assert_eq!(io::Error::last_os_error().kind(), io::ErrorKind::WouldBlock);
                break;
            }
            assert_eq!(written as usize, chunk.len());
        }
        assert_eq!(
            write_record(&mut writer, b"{\"gpu_memory_bytes\":1}\n").unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}
