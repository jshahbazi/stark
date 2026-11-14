//! Consecutive generation prefix reuse.
//!
//! One in-memory entry holds the native state of the previous successful text
//! generation, keyed by the loaded model instance, `context_tokens` and an
//! opaque host scope. Only rows produced by full, non-final prefill chunks are
//! reused, and only up to a chunk boundary, so the rest of a prompt is
//! prefilled exactly as on a fresh state. The entry is never persisted, logged
//! or included in replies.
use std::{
    io::{self, BufRead, BufReader, Cursor, Read},
    time::Duration,
};

/// Must equal the engine's prefill chunk length (checked by the native backend).
pub(crate) const PREFILL_CHUNK_TOKENS: usize = 1024;
/// A retained entry is dropped when no request arrives for this long.
pub(crate) const IDLE_SECONDS: u64 = 600;
pub(crate) const POLICY: &str = "consecutive-prefix-v1";
pub(crate) const KEY: &str = "model-instance+context-tokens+cache-scope";

pub(crate) type CacheScope = [u8; 16];

/// Accept exactly 32 lowercase hexadecimal characters.
pub(crate) fn parse_scope(value: &str) -> Option<CacheScope> {
    let bytes = value.as_bytes();
    if bytes.len() != 32 {
        return None;
    }
    let nibble = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    let mut scope = [0; 16];
    for (index, [high, low]) in bytes.as_chunks::<2>().0.iter().enumerate() {
        scope[index] = nibble(*high)? << 4 | nibble(*low)?;
    }
    Some(scope)
}

/// Prompt rows produced by full, non-final prefill chunks. The final chunk can
/// be partial or split for logits, and generated rows come from decode passes,
/// so only these rows are computed identically for any longer prompt that
/// shares them.
pub(crate) fn reusable_prefill_rows(prompt_tokens: usize) -> usize {
    chunk_boundary(prompt_tokens, PREFILL_CHUNK_TOKENS)
}

/// The start of a prompt's final prefill chunk on a grid of `chunk` tokens.
pub(crate) fn chunk_boundary(
    prompt_tokens: usize,
    chunk: usize,
) -> usize {
    prompt_tokens.saturating_sub(1) / chunk * chunk
}

pub(crate) struct PrefixEntry<S> {
    pub state: S,
    scope: CacheScope,
    context_tokens: u32,
    prefix: Vec<u32>,
}

impl<S> PrefixEntry<S> {
    pub(crate) fn context_tokens(&self) -> u32 {
        self.context_tokens
    }

    pub(crate) fn new(
        state: S,
        scope: CacheScope,
        context_tokens: u32,
        prompt: &[u32],
    ) -> Self {
        Self {
            state,
            scope,
            context_tokens,
            prefix: prompt[..reusable_prefill_rows(prompt.len())].to_vec(),
        }
    }

    /// Rows a request may reuse: the chunk-aligned common token prefix, kept
    /// below the request's own final prefill chunk. Zero on any key mismatch.
    pub(crate) fn reusable(
        &self,
        input: &[u32],
        context_tokens: u32,
        scope: Option<&CacheScope>,
    ) -> usize {
        if scope != Some(&self.scope) || context_tokens != self.context_tokens {
            return 0;
        }
        let common = self.prefix.iter().zip(input).take_while(|(cached, token)| cached == token).count();
        common.min(reusable_prefill_rows(input.len())) / PREFILL_CHUNK_TOKENS * PREFILL_CHUNK_TOKENS
    }
}

/// A state valid only at the end of its prefix, such as a recurrent-state
/// snapshot taken at a prefill chunk boundary. A request reuses the whole
/// prefix or nothing, and must leave at least two prompt tokens to prefill so
/// its final token takes the same route as on a fresh state.
pub(crate) struct BoundaryEntry<S> {
    pub state: S,
    scope: CacheScope,
    context_tokens: u32,
    prefix: Vec<u32>,
}

impl<S> BoundaryEntry<S> {
    pub(crate) fn context_tokens(&self) -> u32 {
        self.context_tokens
    }

    pub(crate) fn new(
        state: S,
        scope: CacheScope,
        context_tokens: u32,
        prefix: &[u32],
    ) -> Self {
        Self {
            state,
            scope,
            context_tokens,
            prefix: prefix.to_vec(),
        }
    }

    pub(crate) fn reusable(
        &self,
        input: &[u32],
        context_tokens: u32,
        scope: Option<&CacheScope>,
    ) -> usize {
        let fits = input.len().checked_sub(2).is_some_and(|limit| self.prefix.len() <= limit);
        if scope != Some(&self.scope)
            || context_tokens != self.context_tokens
            || !fits
            || !input.starts_with(&self.prefix)
        {
            return 0;
        }
        self.prefix.len()
    }
}

/// Request input that can wait for the next request with a timeout, so an
/// idle worker can release its retained state.
pub(crate) trait RequestInput: BufRead {
    /// Return true when input (or end of input) is available within `timeout`.
    fn wait_readable(
        &mut self,
        timeout: Duration,
    ) -> io::Result<bool>;
}

impl<T: AsRef<[u8]>> RequestInput for Cursor<T> {
    fn wait_readable(
        &mut self,
        _timeout: Duration,
    ) -> io::Result<bool> {
        Ok(true)
    }
}

/// The worker's standard input, read without the standard library's
/// process-wide buffer so pending bytes are visible to `wait_readable`.
pub(crate) struct StandardInput(std::mem::ManuallyDrop<std::fs::File>);

impl StandardInput {
    pub(crate) fn reader() -> BufReader<Self> {
        use std::os::fd::FromRawFd;
        // SAFETY: descriptor 0 stays open for the process lifetime and is never
        // closed through this handle.
        BufReader::with_capacity(
            64 * 1024,
            Self(std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(libc::STDIN_FILENO) })),
        )
    }
}

impl Read for StandardInput {
    fn read(
        &mut self,
        buffer: &mut [u8],
    ) -> io::Result<usize> {
        self.0.read(buffer)
    }
}

impl RequestInput for BufReader<StandardInput> {
    fn wait_readable(
        &mut self,
        timeout: Duration,
    ) -> io::Result<bool> {
        if !self.buffer().is_empty() {
            return Ok(true);
        }
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        let milliseconds = timeout.as_millis().min(libc::c_int::MAX as u128) as libc::c_int;
        loop {
            match unsafe { libc::poll(&mut descriptor, 1, milliseconds) } {
                0 => return Ok(false),
                count if count > 0 => return Ok(true),
                _ => {
                    let error = io::Error::last_os_error();
                    if error.kind() != io::ErrorKind::Interrupted {
                        return Err(error);
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = PREFILL_CHUNK_TOKENS;
    const SCOPE: CacheScope = [7; 16];

    fn prompt(length: usize) -> Vec<u32> {
        (0..length as u32).collect()
    }

    #[test]
    fn scopes_are_exact_lowercase_hex() {
        assert_eq!(parse_scope("000102030405060708090a0b0c0d0e0f"), Some(core::array::from_fn(|i| i as u8)));
        for invalid in [
            "",
            "00",
            "000102030405060708090A0B0C0D0E0F",
            "000102030405060708090a0b0c0d0e0g",
            "000102030405060708090a0b0c0d0e0f00",
            " 00102030405060708090a0b0c0d0e0f",
        ] {
            assert_eq!(parse_scope(invalid), None, "{invalid}");
        }
    }

    #[test]
    fn only_full_non_final_chunks_are_reusable() {
        assert_eq!(reusable_prefill_rows(0), 0);
        assert_eq!(reusable_prefill_rows(1), 0);
        assert_eq!(reusable_prefill_rows(CHUNK), 0);
        assert_eq!(reusable_prefill_rows(CHUNK + 1), CHUNK);
        assert_eq!(reusable_prefill_rows(2 * CHUNK), CHUNK);
        assert_eq!(reusable_prefill_rows(2 * CHUNK + 1), 2 * CHUNK);
        assert_eq!(reusable_prefill_rows(18_544), 18 * CHUNK);
    }

    #[test]
    fn reuse_is_the_aligned_common_prefix_below_the_final_chunk() {
        let entry = PrefixEntry::new((), SCOPE, 40_960, &prompt(18_544));
        // The previous final partial chunk is never reused.
        assert_eq!(entry.prefix.len(), 18 * CHUNK);
        // A longer continuation reuses every retained full chunk.
        assert_eq!(entry.reusable(&prompt(20_000), 40_960, Some(&SCOPE)), 18 * CHUNK);
        // The new prompt keeps at least its own final chunk to prefill.
        assert_eq!(entry.reusable(&prompt(18 * CHUNK), 40_960, Some(&SCOPE)), 17 * CHUNK);
        assert_eq!(entry.reusable(&prompt(18 * CHUNK + 1), 40_960, Some(&SCOPE)), 18 * CHUNK);
        assert_eq!(entry.reusable(&prompt(CHUNK), 40_960, Some(&SCOPE)), 0);
        // A divergence rounds down to the chunk containing it.
        let mut diverged = prompt(20_000);
        diverged[5 * CHUNK + 17] = u32::MAX;
        assert_eq!(entry.reusable(&diverged, 40_960, Some(&SCOPE)), 5 * CHUNK);
        diverged[3] = u32::MAX;
        assert_eq!(entry.reusable(&diverged, 40_960, Some(&SCOPE)), 0);
    }

    #[test]
    fn chunk_boundaries_start_the_final_chunk() {
        assert_eq!(chunk_boundary(0, 4096), 0);
        assert_eq!(chunk_boundary(4096, 4096), 0);
        assert_eq!(chunk_boundary(4097, 4096), 4096);
        assert_eq!(chunk_boundary(18_544, 4096), 16_384);
        assert_eq!(chunk_boundary(18_544, CHUNK), reusable_prefill_rows(18_544));
    }

    #[test]
    fn boundary_entries_reuse_their_whole_prefix_or_nothing() {
        let boundary = 4 * 4096;
        let entry = BoundaryEntry::new((), SCOPE, 40_960, &prompt(boundary));
        assert_eq!(entry.reusable(&prompt(20_000), 40_960, Some(&SCOPE)), boundary);
        // At least two prompt tokens remain to prefill.
        assert_eq!(entry.reusable(&prompt(boundary + 2), 40_960, Some(&SCOPE)), boundary);
        assert_eq!(entry.reusable(&prompt(boundary + 1), 40_960, Some(&SCOPE)), 0);
        assert_eq!(entry.reusable(&prompt(boundary), 40_960, Some(&SCOPE)), 0);
        assert_eq!(entry.reusable(&prompt(3), 40_960, Some(&SCOPE)), 0);
        // A snapshot cannot rewind: any divergence before the boundary misses.
        let mut diverged = prompt(20_000);
        diverged[boundary - 1] = u32::MAX;
        assert_eq!(entry.reusable(&diverged, 40_960, Some(&SCOPE)), 0);
        diverged[boundary - 1] = (boundary - 1) as u32;
        diverged[boundary] = u32::MAX;
        assert_eq!(entry.reusable(&diverged, 40_960, Some(&SCOPE)), boundary);
        assert_eq!(entry.reusable(&prompt(20_000), 40_960, None), 0);
        assert_eq!(entry.reusable(&prompt(20_000), 40_960, Some(&[8; 16])), 0);
        assert_eq!(entry.reusable(&prompt(20_000), 32_768, Some(&SCOPE)), 0);
    }

    #[test]
    fn reuse_requires_the_same_scope_and_context() {
        let entry = PrefixEntry::new((), SCOPE, 40_960, &prompt(4 * CHUNK));
        assert_eq!(entry.reusable(&prompt(5 * CHUNK), 40_960, Some(&SCOPE)), 3 * CHUNK);
        assert_eq!(entry.reusable(&prompt(5 * CHUNK), 40_960, None), 0);
        assert_eq!(entry.reusable(&prompt(5 * CHUNK), 40_960, Some(&[8; 16])), 0);
        assert_eq!(entry.reusable(&prompt(5 * CHUNK), 32_768, Some(&SCOPE)), 0);
    }
}
