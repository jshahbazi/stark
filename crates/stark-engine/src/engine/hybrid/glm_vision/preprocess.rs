// SPDX-License-Identifier: Apache-2.0
//! Bounded GLM static-image canvas, padding and Conv3d patch layout.
//!
//! Independent implementation of Transformers GLM5-Next processor at
//! b6401d6605d16620e0f95b54fb2906f92de7e33f (Hugging Face, Apache-2.0).
//! Static temporal duplication cancels its temporal budget factor, leaving
//! the same 12544/6272000 spatial limits recorded in the pinned GGUF sidecar.

use crate::{
    backends::common::InvalidInput,
    engine::vision::preprocess::{PreprocessLimits, RgbImage, decode_image_discard_alpha, resize_rgb},
};

pub(super) const PATCH: usize = 14;
pub(super) const CHANNEL_PATCH: usize = PATCH * PATCH;
pub(super) const PATCH_WIDTH: usize = 3 * 2 * CHANNEL_PATCH;
pub(super) const MAX_PATCHES: usize = 1024;
const ALIGN: u32 = 28;
const MIN_PIXELS: u64 = 12544;
const MAX_PIXELS: u64 = 6272000;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];

pub(in crate::engine::hybrid) struct Patches {
    pub data: Box<[f32]>,
    pub grid: [u32; 3],
    pub dimensions: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Canvas {
    width: u32,
    height: u32,
    content_width: u32,
    content_height: u32,
}

fn align(value: u32) -> Result<u32, InvalidInput> {
    value.checked_add(ALIGN - 1).map(|v| v / ALIGN * ALIGN).ok_or(InvalidInput("GLM image alignment overflow"))
}

fn canvas(
    width: u32,
    height: u32,
) -> Result<Canvas, InvalidInput> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || width > 16384 || height > 16384 || pixels > 32_000_000 {
        return Err(InvalidInput("GLM image source dimensions exceed bounds"));
    }
    let (mut target_w, mut target_h) = (align(width)?, align(height)?);
    let mut aligned_pixels = u64::from(target_w) * u64::from(target_h);
    if aligned_pixels < MIN_PIXELS {
        let scale = (MIN_PIXELS as f64 / pixels as f64).sqrt();
        target_w = align((f64::from(width) * scale).ceil().max(1.0) as u32)?;
        target_h = align((f64::from(height) * scale).ceil().max(1.0) as u32)?;
        aligned_pixels = u64::from(target_w) * u64::from(target_h);
    }
    if aligned_pixels > MAX_PIXELS {
        let (mut low, mut high) = (1, height);
        (target_w, target_h) = (ALIGN, ALIGN);
        while low <= high {
            let content_h = low + (high - low) / 2;
            let content_w = (u64::from(width) * u64::from(content_h) / u64::from(height)).max(1) as u32;
            let (candidate_w, candidate_h) = (align(content_w)?, align(content_h)?);
            if u64::from(candidate_w) * u64::from(candidate_h) <= MAX_PIXELS {
                (target_w, target_h) = (candidate_w, candidate_h);
                low = content_h + 1;
            } else {
                high = content_h - 1;
            }
        }
    }
    let patches = u64::from(target_w / PATCH as u32) * u64::from(target_h / PATCH as u32);
    // Reject the source recipe's oversized result before decoding/allocation.
    // Native admission never shrinks a canvas merely to fit a lower budget.
    if patches > MAX_PATCHES as u64 {
        return Err(InvalidInput("GLM processed image exceeds native patch bounds"));
    }
    let mut scale = (f64::from(target_w) / f64::from(width)).min(f64::from(target_h) / f64::from(height));
    if pixels >= MIN_PIXELS {
        scale = scale.min(1.0);
    }
    Ok(Canvas {
        width: target_w,
        height: target_h,
        content_width: (f64::from(width) * scale).floor().max(1.0).min(f64::from(target_w)) as u32,
        content_height: (f64::from(height) * scale).floor().max(1.0).min(f64::from(target_h)) as u32,
    })
}

pub(super) fn prepare(encoded: &[u8]) -> Result<Patches, InvalidInput> {
    let image = decode_image_discard_alpha(encoded, &PreprocessLimits::default(), |w, h| canvas(w, h).map(|_| ()))?;
    prepare_rgb(&image)
}

fn prepare_rgb(image: &RgbImage) -> Result<Patches, InvalidInput> {
    let canvas = canvas(image.width(), image.height())?;
    let content = resize_rgb(image, canvas.content_width, canvas.content_height)?;
    let (grid_w, grid_h) = (canvas.width as usize / PATCH, canvas.height as usize / PATCH);
    let length = grid_w * grid_h * PATCH_WIDTH;
    let mut patches = Vec::new();
    patches.try_reserve_exact(length).map_err(|_| InvalidInput("GLM image patch allocation failed"))?;
    // Merge block/y/x, intra-block/y/x, channel, temporal copy, patch/y/x.
    for block_y in 0..grid_h / 2 {
        for block_x in 0..grid_w / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    for channel in 0..3 {
                        for _temporal in 0..2 {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    let y = (block_y * 2 + dy) * PATCH + py;
                                    let x = (block_x * 2 + dx) * PATCH + px;
                                    let value =
                                        if x < canvas.content_width as usize && y < canvas.content_height as usize {
                                            content[(y * canvas.content_width as usize + x) * 3 + channel]
                                        } else {
                                            0
                                        };
                                    let scaled = (f64::from(value) / 255.0) as f32;
                                    patches.push((scaled - MEAN[channel]) / STD[channel]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Patches {
        data: patches.into_boxed_slice(),
        grid: [1, grid_h as u32, grid_w as u32],
        dimensions: [canvas.width, canvas.height],
    })
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn glm_canvas_preserves_content_and_adds_only_bottom_right_padding() {
        assert_eq!(
            canvas(320, 200).unwrap(),
            Canvas {
                width: 336,
                height: 224,
                content_width: 320,
                content_height: 200
            }
        );
        assert_eq!(
            canvas(32, 32).unwrap(),
            Canvas {
                width: 112,
                height: 112,
                content_width: 112,
                content_height: 112
            }
        );
        assert!(canvas(0, 1).is_err());
        assert!(canvas(16385, 1).is_err());
        assert!(canvas(1024, 1024).is_err());
    }

    #[stark_test]
    fn glm_patches_duplicate_temporal_planes_and_normalize_black_padding() {
        let image = RgbImage::try_new(120, 120, vec![255; 120 * 120 * 3], &PreprocessLimits::default()).unwrap();
        let patches = prepare_rgb(&image).unwrap();
        assert_eq!(patches.grid, [1, 10, 10]);
        assert_eq!(patches.dimensions, [140, 140]);
        for row in patches.data.chunks_exact(PATCH_WIDTH) {
            for channel in 0..3 {
                let start = channel * 2 * CHANNEL_PATCH;
                assert_eq!(&row[start..start + CHANNEL_PATCH], &row[start + CHANNEL_PATCH..start + 2 * CHANNEL_PATCH]);
            }
        }
        let last = &patches.data[patches.data.len() - PATCH_WIDTH..];
        for channel in 0..3 {
            assert!(
                last[channel * 2 * CHANNEL_PATCH..(channel + 1) * 2 * CHANNEL_PATCH]
                    .iter()
                    .all(|v| *v == -MEAN[channel] / STD[channel])
            );
        }
    }
}
