//! Bounded, request-owned still images for retained Qwen4Exp towers.

use std::sync::Arc;

use super::{Error, Result};

pub(super) mod artifact;
use crate::engine::vision::preprocess::{PreparedImage, PreprocessLimits, decode_image, prepare_image};

/// Model-bound CPU patches. Pixels, patches and embeddings cannot be inspected
/// or serialized through this interface. Preparing an image does no GPU work.
pub struct PreparedHybridImage {
    pub(super) owner: Arc<()>,
    pub(super) prepared: PreparedImage,
}

impl PreparedHybridImage {
    pub fn grid(&self) -> [u32; 3] {
        self.prepared.grid
    }

    pub fn image_token_count(&self) -> u32 {
        self.prepared.grid[1] * self.prepared.grid[2] / 4
    }

    pub fn processed_dimensions(&self) -> [u32; 2] {
        [self.prepared.resized_width, self.prepared.resized_height]
    }
}

pub(super) fn prepare(
    owner: Arc<()>,
    encoded: &[u8],
) -> Result<PreparedHybridImage> {
    // Retained streaming attention supports the source processor maximum.
    let limits = PreprocessLimits {
        max_processed_pixels: 16_777_216,
        max_encoder_patches: 65_536,
        ..PreprocessLimits::default()
    };
    let image = decode_image(encoded, &limits).map_err(|_| Error::Invalid("image decoding failed"))?;
    let prepared = prepare_image(&image, &limits).map_err(|_| Error::Invalid("image preprocessing failed"))?;
    let tokens = prepared.grid[1]
        .checked_mul(prepared.grid[2])
        .and_then(|patches| patches.checked_div(4))
        .filter(|tokens| (1..=16_384).contains(tokens))
        .ok_or(Error::Invalid("image patch bounds"))?;
    if prepared.grid[0] != 1 || tokens == 0 {
        return Err(Error::Invalid("static image grid"));
    }
    Ok(PreparedHybridImage {
        owner,
        prepared,
    })
}

#[derive(Clone, Copy)]
pub(super) struct Tokens {
    pub start: u32,
    pub end: u32,
    pub pad: u32,
    pub video: u32,
}

impl Tokens {
    /// Exactly one complete image run, with no video or extra image controls.
    pub fn span(
        self,
        tokens: &[u32],
        image_count: usize,
    ) -> Result<usize> {
        if !(1..=16_384).contains(&image_count) {
            return Err(Error::Invalid("image token count"));
        }
        let start = tokens
            .iter()
            .position(|token| *token == self.start)
            .and_then(|start| start.checked_add(1))
            .ok_or(Error::Invalid("image token span"))?;
        let end = start.checked_add(image_count).ok_or(Error::Invalid("image token span"))?;
        if tokens.get(end) != Some(&self.end)
            || tokens.get(start..end).is_none_or(|run| run.iter().any(|token| *token != self.pad))
            || tokens.iter().filter(|token| **token == self.start).count() != 1
            || tokens.iter().filter(|token| **token == self.end).count() != 1
            || tokens.iter().filter(|token| **token == self.pad).count() != image_count
            || tokens.contains(&self.video)
        {
            return Err(Error::Invalid("image token span"));
        }
        Ok(start)
    }

    pub fn expand(
        self,
        tokens: &[u32],
        image_count: usize,
        capacity: usize,
    ) -> Result<Box<[u32]>> {
        let start = self.span(tokens, 1)?;
        let length = tokens
            .len()
            .checked_sub(1)
            .and_then(|n| n.checked_add(image_count))
            .filter(|n| *n <= capacity)
            .ok_or(Error::Invalid("image prompt context capacity"))?;
        if !(1..=16_384).contains(&image_count) {
            return Err(Error::Invalid("image token count"));
        }
        let mut expanded = Vec::new();
        expanded.try_reserve_exact(length).map_err(|_| Error::Invalid("image prompt allocation failed"))?;
        expanded.extend_from_slice(&tokens[..start]);
        expanded.resize(start + image_count, self.pad);
        expanded.extend_from_slice(&tokens[start + 1..]);
        Ok(expanded.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    const TOKENS: Tokens = Tokens {
        start: 10,
        end: 11,
        pad: 12,
        video: 13,
    };

    #[stark_test]
    fn exact_image_expansion_and_span_reject_extra_controls() {
        let maximum = TOKENS.expand(&[1, 10, 12, 11, 2], 16_384, 16_388).unwrap();
        assert_eq!(TOKENS.span(&maximum, 16_384).unwrap(), 2);
        let expanded = TOKENS.expand(&[1, 10, 12, 11, 2], 4, 8).unwrap();
        assert_eq!(&*expanded, &[1, 10, 12, 12, 12, 12, 11, 2]);
        assert_eq!(TOKENS.span(&expanded, 4).unwrap(), 2);
        assert!(TOKENS.expand(&[1, 10, 12, 11, 2], 4, 7).is_err());
        for tokens in [
            vec![10, 12],
            vec![12, 11],
            vec![10, 12, 11, 12],
            vec![10, 12, 11, 13],
            vec![10, 12, 11, 10, 12, 11],
            vec![10, 12, 1, 11],
            vec![10, 12, 11, 11],
        ] {
            assert!(TOKENS.span(&tokens, 1).is_err());
        }
        assert!(TOKENS.expand(&[10, 12, 11], usize::MAX, usize::MAX).is_err());
        assert!(TOKENS.expand(&[10, 12, 11], 16_385, 32_768).is_err());
    }
}
