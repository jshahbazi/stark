// SPDX-License-Identifier: Apache-2.0
use stark_engine::engine::{
    hy4::{ImageInput, VisionGeometry},
    vision::{Hy4ImagePixels, prepare_hy4_image_rgb},
};

use crate::generation::PreparedImageMetadata;

pub(super) const IMAGE_PAD: u32 = 120_120;
const IMAGE_START: u32 = 120_118;
const IMAGE_END: u32 = 120_119;

pub(crate) struct Hy4PreparedImage {
    pub(super) tokens: Vec<u32>,
    pixels: Hy4ImagePixels,
}

impl Hy4PreparedImage {
    pub(super) fn prepare(
        mut tokens: Vec<u32>,
        encoded: &[u8],
        context_tokens: u32,
        geometry: VisionGeometry,
    ) -> Result<(Self, PreparedImageMetadata), &'static str> {
        if geometry.image_token != IMAGE_PAD
            || (geometry.patch, geometry.merge, geometry.min_pixels, geometry.max_pixels) != (16, 2, 4096, 4_194_304)
        {
            return Err("invalid_model_config");
        }
        let slot = image_slot(&tokens)?;
        let pixels = prepare_hy4_image_rgb(encoded).map_err(|_| "invalid_image_request")?;
        let grid = [1, pixels.height / geometry.patch, pixels.width / geometry.patch];
        let image_tokens = (grid[1] / geometry.merge) * (grid[2] / geometry.merge + 1);
        let extra = image_tokens as usize - 1;
        let input_tokens = tokens.len().checked_add(extra).ok_or("context_overflow")?;
        if input_tokens >= context_tokens as usize {
            return Err("context_overflow");
        }
        tokens.try_reserve_exact(extra).map_err(|_| "invalid_image_request")?;
        tokens.splice(slot..=slot, std::iter::repeat_n(IMAGE_PAD, image_tokens as usize));
        let metadata = PreparedImageMetadata {
            input_tokens,
            image_tokens,
            grid,
            processed_dimensions: [pixels.width, pixels.height],
        };
        Ok((
            Self {
                tokens,
                pixels,
            },
            metadata,
        ))
    }

    pub(super) fn image_input(&self) -> ImageInput<'_> {
        ImageInput::new(&self.pixels.rgb, self.pixels.width, self.pixels.height)
    }
}

fn image_slot(tokens: &[u32]) -> Result<usize, &'static str> {
    let mut positions = tokens.iter().enumerate().filter_map(|(index, &token)| (token == IMAGE_PAD).then_some(index));
    let slot = positions.next().ok_or("invalid_image_request")?;
    if positions.next().is_some()
        || slot == 0
        || tokens[slot - 1] != IMAGE_START
        || tokens.get(slot + 1) != Some(&IMAGE_END)
    {
        return Err("invalid_image_request");
    }
    Ok(slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_wrapped_image_slot_is_accepted() {
        assert_eq!(image_slot(&[2, IMAGE_START, IMAGE_PAD, IMAGE_END, 3]), Ok(2));
        for prompt in [
            vec![],
            vec![IMAGE_START, IMAGE_END],
            vec![IMAGE_PAD, IMAGE_END],
            vec![IMAGE_START, IMAGE_PAD],
            vec![IMAGE_START, IMAGE_PAD, IMAGE_PAD, IMAGE_END],
            vec![IMAGE_PAD, IMAGE_START, IMAGE_PAD, IMAGE_END],
        ] {
            assert_eq!(image_slot(&prompt), Err("invalid_image_request"));
        }
    }
}
