//! Request-scoped cancellation over an inherited private pipe.
use std::{
    fs::File,
    io::{ErrorKind, Read},
    os::fd::{AsRawFd, FromRawFd},
};

pub(crate) struct CancelSignal {
    file: File,
    pending: Vec<u8>,
}

impl CancelSignal {
    pub(crate) fn from_environment() -> Result<Option<Self>, &'static str> {
        let Some(value) = std::env::var_os("STARK_CANCEL_FD") else {
            return Ok(None);
        };
        // Called before model initialization creates threads.
        unsafe { std::env::remove_var("STARK_CANCEL_FD") };
        let fd: i32 =
            value.to_str().and_then(|s| s.parse().ok()).filter(|fd| *fd >= 3).ok_or("invalid_cancel_channel")?;
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut stat) } != 0 || stat.st_mode & libc::S_IFMT != libc::S_IFIFO {
            return Err("invalid_cancel_channel");
        }
        let file = unsafe { File::from_raw_fd(fd) };
        Self::new(file).map(Some)
    }

    fn new(file: File) -> Result<Self, &'static str> {
        let fd = file.as_raw_fd();
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0
            || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
            || unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } < 0
        {
            return Err("invalid_cancel_channel");
        }
        Ok(Self {
            file,
            pending: Vec::new(),
        })
    }

    pub(crate) fn check(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        let mut cancelled = false;
        loop {
            let mut bytes = [0; 256];
            match self.file.read(&mut bytes) {
                Ok(0) => return Ok(true), // Owning client vanished.
                Ok(count) => self.pending.extend_from_slice(&bytes[..count]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(cancelled),
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(_) => return Err("cancel_channel_failed"),
            }
            while self.pending.len() >= 8 {
                let id = u64::from_le_bytes(self.pending[..8].try_into().unwrap());
                self.pending.drain(..8);
                cancelled |= id == request_id;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn cancellation_is_scoped_and_handles_partial_frames_and_eof() {
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let mut signal = CancelSignal::new(unsafe { File::from_raw_fd(fds[0]) }).unwrap();
        let mut writer = unsafe { File::from_raw_fd(fds[1]) };
        assert!(!signal.check(2).unwrap());
        writer.write_all(&1_u64.to_le_bytes()).unwrap();
        assert!(!signal.check(2).unwrap());
        writer.write_all(&2_u64.to_le_bytes()[..3]).unwrap();
        assert!(!signal.check(2).unwrap());
        writer.write_all(&2_u64.to_le_bytes()[3..]).unwrap();
        assert!(signal.check(2).unwrap());
        assert!(!signal.check(3).unwrap());
        drop(writer);
        assert!(signal.check(3).unwrap());
    }
}
