// SPDX-License-Identifier: Apache-2.0
//! Private transport and committed target features for the pinned MLX draft.

use std::{
    collections::VecDeque,
    io::{Error, ErrorKind, Read, Write},
    os::fd::{AsRawFd, RawFd},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::{Duration, Instant},
};

use serde::Deserialize;
use stark_engine::engine::inkling_slice::{DRAFT_FEATURE_WORDS, Session, VOCABULARY};

use crate::inkling_artifact;

pub(super) const WEIGHTS_SHA: &str = "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee";
pub(super) const REVISION: &str = "4f9088ea66c21f2b99ed657d1941c306b03e84d2";
pub(super) const RESERVE_BYTES: usize = 4 * 1024 * 1024 * 1024;
pub(super) const WINDOW_ROWS: usize = 2047;
pub(super) const MAX_DRAFTS: usize = 7;
const CONFIG_SHA: &str = "d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa";
const WEIGHTS_BYTES: usize = 989_975_616;
const RESET: u32 = 1;
const PROPOSE: u32 = 2;
const PROPOSE_RUNNERS: u32 = 4;
const MODEL_LIMIT: u64 = 16 * 1024;
const LOAD_ERROR: &str = "inkling_dflash2_load_failed";
const REQUEST_ERROR: &str = "inkling_dflash2_request_failed";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    version: u32,
    python: PathBuf,
    companion: PathBuf,
    draft: PathBuf,
    dflash_repo: PathBuf,
    greedy_drafts: Option<u32>,
    greedy_self_correction: Option<bool>,
}

impl Profile {
    fn load(path: &Path) -> Result<Self, &'static str> {
        let bytes = inkling_artifact::read(path, MODEL_LIMIT).map_err(|_| LOAD_ERROR)?;
        let profile: Self = serde_json::from_slice(&bytes).map_err(|_| LOAD_ERROR)?;
        if profile.version != 1
            || profile.greedy_drafts.is_some_and(|drafts| !(1..=MAX_DRAFTS as u32).contains(&drafts))
            || ![&profile.python, &profile.companion, &profile.draft, &profile.dflash_repo]
                .iter()
                .all(|path| path.is_absolute())
        {
            return Err(LOAD_ERROR);
        }
        Ok(profile)
    }
}

/// The only draft state kept across requests is in the worker's prefix
/// snapshot. Its absolute position and chronological feature rows suffice to
/// rebuild the once-per-row draft K/V at the first proposal of a new request.
#[derive(Clone, Default)]
pub(super) struct FeatureWindow {
    pub(super) end: usize,
    words: VecDeque<u16>,
}

impl FeatureWindow {
    fn start(&self) -> usize {
        self.end - self.words.len() / DRAFT_FEATURE_WORDS
    }

    fn append(
        &mut self,
        end: usize,
        words: &[u16],
    ) {
        let limit = WINDOW_ROWS * DRAFT_FEATURE_WORDS;
        let overflow = (self.words.len() + words.len()).saturating_sub(limit);
        self.words.drain(..overflow);
        self.words.extend(words);
        self.end = end;
    }
}

#[derive(Default)]
pub(super) struct Features {
    pub(super) window: FeatureWindow,
    scratch: Vec<u16>,
}

impl Features {
    pub(super) fn capture(
        &mut self,
        target: &Session,
    ) -> Result<(), &'static str> {
        let end = target.position();
        let rows = end.checked_sub(self.window.end).ok_or(REQUEST_ERROR)?.min(WINDOW_ROWS);
        if rows == 0 {
            return Ok(());
        }
        self.scratch.clear();
        target.append_external_draft_features(rows, &mut self.scratch).map_err(|_| REQUEST_ERROR)?;
        if self.scratch.len() != rows * DRAFT_FEATURE_WORDS {
            return Err(REQUEST_ERROR);
        }
        self.window.append(end, &self.scratch);
        Ok(())
    }
}

pub(super) struct Drafter {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    sent: usize,
    failed: bool,
    pub(super) greedy_drafts: u32,
    pub(super) greedy_self_correction: bool,
}

pub(super) struct RunnerProposals {
    pub(super) primary: [u32; MAX_DRAFTS],
    pub(super) alternate: [u32; 2],
    pub(super) gap: [f32; 2],
}

impl Drafter {
    pub(super) fn load(
        profile: &Path,
        target: &Path,
    ) -> Result<Self, &'static str> {
        let profile = Profile::load(profile)?;
        inkling_artifact::checked_bytes(&profile.draft.join("config.json"), MODEL_LIMIT, CONFIG_SHA)
            .map_err(|_| LOAD_ERROR)?;
        let weights =
            inkling_artifact::check_weights(&profile.draft.join("draft.safetensors"), WEIGHTS_BYTES, WEIGHTS_SHA)
                .map_err(|_| LOAD_ERROR)?;
        let mut child = Command::new(&profile.python)
            .arg("-I")
            .arg(&profile.companion)
            .arg("--draft")
            .arg(&profile.draft)
            .arg("--target")
            .arg(target)
            .arg("--dflash-repo")
            .arg(&profile.dflash_repo)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| LOAD_ERROR)?;
        let mut drafter = Self {
            input: child.stdin.take().expect("configured companion stdin"),
            output: child.stdout.take().expect("configured companion stdout"),
            child,
            sent: 0,
            failed: false,
            greedy_drafts: profile.greedy_drafts.unwrap_or(super::GREEDY_DEFAULT_DRAFTS),
            greedy_self_correction: profile.greedy_self_correction.unwrap_or(false),
        };
        set_nonblocking(drafter.input.as_raw_fd()).map_err(|_| LOAD_ERROR)?;
        set_nonblocking(drafter.output.as_raw_fd()).map_err(|_| LOAD_ERROR)?;
        drafter.status(Instant::now() + Duration::from_secs(90)).map_err(|_| LOAD_ERROR)?;
        weights.verify_unchanged().map_err(|_| LOAD_ERROR)?;
        Ok(drafter)
    }

    pub(super) fn reset(&mut self) -> Result<(), &'static str> {
        let deadline = Instant::now() + Duration::from_secs(15);
        let result = self.send_header(RESET, 0, 0, 0, deadline).and_then(|()| self.status(deadline));
        self.sent = 0;
        self.finish(result)
    }

    pub(super) fn propose(
        &mut self,
        features: &FeatureWindow,
        anchor: u32,
    ) -> Result<[u32; MAX_DRAFTS], &'static str> {
        let result = self.propose_inner(features, anchor, false).map(|result| result.primary);
        self.finish(result)
    }

    pub(super) fn propose_with_runners(
        &mut self,
        features: &FeatureWindow,
        anchor: u32,
    ) -> Result<RunnerProposals, &'static str> {
        let result = self.propose_inner(features, anchor, true);
        self.finish(result)
    }

    fn propose_inner(
        &mut self,
        features: &FeatureWindow,
        anchor: u32,
        runners: bool,
    ) -> Result<RunnerProposals, &'static str> {
        let start = features.start();
        if self.sent > features.end || anchor as usize >= VOCABULARY {
            return Err(REQUEST_ERROR);
        }
        let offset = self.sent.max(start);
        let rows = features.end - offset;
        let deadline = Instant::now() + Duration::from_secs(30);
        self.send_header(
            if runners {
                PROPOSE_RUNNERS
            } else {
                PROPOSE
            },
            offset as u32,
            rows as u32,
            anchor,
            deadline,
        )?;
        let (first, second) = features.words.as_slices();
        let skip = (offset - start) * DRAFT_FEATURE_WORDS;
        self.write_features(&first[skip.min(first.len())..], deadline)?;
        self.write_features(&second[skip.saturating_sub(first.len())..], deadline)?;
        self.status(deadline)?;
        let mut primary_bytes = [0; MAX_DRAFTS * size_of::<u32>()];
        self.read_exact(&mut primary_bytes, deadline)?;
        let primary = std::array::from_fn(|i| u32::from_le_bytes(primary_bytes[i * 4..i * 4 + 4].try_into().unwrap()));
        let (alternate, gap) = if runners {
            let mut extra = [0; 2 * size_of::<u32>() + 2 * size_of::<f32>()];
            self.read_exact(&mut extra, deadline)?;
            let alternate = std::array::from_fn(|i| u32::from_le_bytes(extra[i * 4..i * 4 + 4].try_into().unwrap()));
            let gap = std::array::from_fn(|i| f32::from_le_bytes(extra[8 + i * 4..12 + i * 4].try_into().unwrap()));
            (alternate, gap)
        } else {
            ([0; 2], [f32::INFINITY; 2])
        };
        if primary.iter().chain(alternate.iter()).any(|&id| id as usize >= VOCABULARY)
            || runners && gap.iter().any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(REQUEST_ERROR);
        }
        self.sent = features.end;
        Ok(RunnerProposals {
            primary,
            alternate,
            gap,
        })
    }

    fn write_features(
        &mut self,
        words: &[u16],
        deadline: Instant,
    ) -> Result<(), &'static str> {
        const { assert!(cfg!(target_endian = "little")) };
        // The wire representation is the BF16 words of this little-endian
        // host. Reading them as bytes preserves each word without a copy.
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) };
        self.write_all(bytes, deadline)
    }

    fn send_header(
        &mut self,
        operation: u32,
        start: u32,
        rows: u32,
        anchor: u32,
        deadline: Instant,
    ) -> Result<(), &'static str> {
        if self.failed {
            return Err(REQUEST_ERROR);
        }
        let mut header = [0; 16];
        for (slot, field) in header.chunks_exact_mut(4).zip([operation, start, rows, anchor]) {
            slot.copy_from_slice(&field.to_le_bytes());
        }
        self.write_all(&header, deadline)
    }

    fn status(
        &mut self,
        deadline: Instant,
    ) -> Result<(), &'static str> {
        let mut status = [0; size_of::<u32>()];
        self.read_exact(&mut status, deadline)?;
        if u32::from_le_bytes(status) != 0 {
            return Err(REQUEST_ERROR);
        }
        Ok(())
    }

    fn read_exact(
        &mut self,
        mut bytes: &mut [u8],
        deadline: Instant,
    ) -> Result<(), &'static str> {
        while !bytes.is_empty() {
            if Instant::now() >= deadline {
                return Err(REQUEST_ERROR);
            }
            match self.output.read(bytes) {
                Ok(0) => return Err(REQUEST_ERROR),
                Ok(read) => bytes = &mut bytes[read..],
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    wait_on_pipe(self.output.as_raw_fd(), libc::POLLIN, deadline)?;
                },
                Err(_) => return Err(REQUEST_ERROR),
            }
        }
        Ok(())
    }

    fn write_all(
        &mut self,
        mut bytes: &[u8],
        deadline: Instant,
    ) -> Result<(), &'static str> {
        while !bytes.is_empty() {
            if Instant::now() >= deadline {
                return Err(REQUEST_ERROR);
            }
            match self.input.write(bytes) {
                Ok(0) => return Err(REQUEST_ERROR),
                Ok(written) => bytes = &bytes[written..],
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    wait_on_pipe(self.input.as_raw_fd(), libc::POLLOUT, deadline)?;
                },
                Err(_) => return Err(REQUEST_ERROR),
            }
        }
        Ok(())
    }

    fn finish<T>(
        &mut self,
        result: Result<T, &'static str>,
    ) -> Result<T, &'static str> {
        if result.is_err() && !self.failed {
            self.failed = true;
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        result
    }
}

impl Drop for Drafter {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn set_nonblocking(fd: RawFd) -> Result<(), &'static str> {
    // Both pipes are exclusively owned by this worker and never duplicated.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(REQUEST_ERROR);
    }
    Ok(())
}

fn wait_on_pipe(
    fd: RawFd,
    events: i16,
    deadline: Instant,
) -> Result<(), &'static str> {
    let mut descriptor = libc::pollfd {
        fd,
        events,
        revents: 0,
    };
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(REQUEST_ERROR);
        }
        // The wire deadlines are far below the range of a signed poll timeout.
        let waited = unsafe { libc::poll(&mut descriptor, 1, remaining.as_millis().max(1) as i32) };
        match waited {
            1 if descriptor.revents & events != 0 => return Ok(()),
            -1 if Error::last_os_error().kind() == ErrorKind::Interrupted => continue,
            _ => return Err(REQUEST_ERROR),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_feature_window_restores_absolute_positions_after_wrap() {
        let mut window = FeatureWindow::default();
        let mut batch = vec![1; WINDOW_ROWS * DRAFT_FEATURE_WORDS];
        batch[2 * DRAFT_FEATURE_WORDS] = 3;
        window.append(WINDOW_ROWS + 9, &batch);
        let snapshot = window.clone();
        window.append(WINDOW_ROWS + 11, &vec![4; 2 * DRAFT_FEATURE_WORDS]);
        assert_eq!(window.start(), 11);
        assert_eq!(window.words.len(), WINDOW_ROWS * DRAFT_FEATURE_WORDS);
        assert_eq!(window.words.front(), Some(&3));
        assert_eq!(window.words.back(), Some(&4));
        assert_eq!(snapshot.start(), 9);
        assert_eq!(snapshot.words.front(), Some(&1));
        assert_eq!(snapshot.words.back(), Some(&1));
    }
}
