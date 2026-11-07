//! Bounded Gemma 3 fixed-resolution image preprocessing.
//!
//! Follows the pinned Transformers 5.15.1 CPU Gemma3ImageProcessor backend:
//! upright RGB, alpha discarded, Torchvision antialiased bilinear uint8 resize to
//! 896 square, rescale by 1/255, normalize by mean/std 0.5. Pan-and-scan is
//! disabled in this versioned profile. Conv2d patches use channel/y/x order.

use half::bf16;

use super::super::preprocess::{PreprocessLimits, RgbImage, decode_image_discard_alpha};
use crate::backends::common::InvalidInput;

pub(crate) const SIDE: u32 = 896;
pub(crate) const PATCH: u32 = 14;
pub(crate) const PATCH_ROWS: u32 = 4096;
pub(crate) const PATCH_WIDTH: u32 = 588;

pub(crate) struct Gemma3Patches {
    pub(crate) patches: Box<[bf16]>,
}

fn dimensions(
    width: u32,
    height: u32,
) -> Result<(), InvalidInput> {
    if width == 0 || height == 0 || width > 16384 || height > 16384 || u64::from(width) * u64::from(height) > 32_000_000
    {
        return Err(InvalidInput("Gemma 3 image dimensions exceed preprocessing bounds"));
    }
    Ok(())
}

// The pinned CPU Torchvision backend uses a half-pixel center and widens
// antialias support for downsampling. Its uint8 kernel quantizes coefficients
// into signed int16, selecting one precision per axis from the maximum weight.
// Fixed Pillow 22-bit coefficients differ on nonuniform images.
struct Coefficients {
    start: usize,
    weights: Vec<i32>,
    precision: u32,
}

fn coefficients(
    input: u32,
    output: u32,
) -> Result<Vec<Coefficients>, InvalidInput> {
    if input == 0 || output == 0 || input > 16384 || output > SIDE {
        return Err(InvalidInput("Gemma 3 resize dimensions exceed bounds"));
    }
    let scale = f64::from(input) / f64::from(output);
    let filter_scale = scale.max(1.0);
    let mut rows = Vec::with_capacity(output as usize);
    let mut maximum = 0.0f64;
    for index in 0..output {
        let center = (f64::from(index) + 0.5) * scale;
        let start = ((center - filter_scale + 0.5) as i64).max(0) as usize;
        let end = ((center + filter_scale + 0.5) as i64).min(i64::from(input)).max(start as i64) as usize;
        let invscale = if scale >= 1.0 {
            1.0 / scale
        } else {
            1.0
        };
        let mut weights: Vec<f64> =
            (start..end).map(|position| (1.0 - ((position as f64 - center + 0.5) * invscale).abs()).max(0.0)).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(InvalidInput("Gemma 3 resize has empty coefficients"));
        }
        for weight in &mut weights {
            *weight /= total;
            maximum = maximum.max(*weight);
        }
        rows.push((start, weights));
    }
    let precision = (0..22).find(|bits| (0.5 + maximum * f64::from(1 << (bits + 1))) as i32 >= (1 << 15)).unwrap_or(22);
    Ok(rows
        .into_iter()
        .map(|(start, weights)| Coefficients {
            start,
            weights: weights.into_iter().map(|weight| (0.5 + weight * f64::from(1 << precision)) as i32).collect(),
            precision,
        })
        .collect())
}

pub(crate) fn resize_rgb(image: &RgbImage) -> Result<Vec<u8>, InvalidInput> {
    dimensions(image.width(), image.height())?;
    let horizontal = if image.width() == SIDE {
        image.pixels().to_vec()
    } else {
        let coefficients = coefficients(image.width(), SIDE)?;
        let mut pixels = vec![0u8; (SIDE * image.height() * 3) as usize];
        for y in 0..image.height() as usize {
            for (x, row) in coefficients.iter().enumerate() {
                for channel in 0..3 {
                    let mut sum = 1i64 << (row.precision - 1);
                    for (index, weight) in row.weights.iter().enumerate() {
                        sum +=
                            i64::from(image.pixels()[(y * image.width() as usize + row.start + index) * 3 + channel])
                                * i64::from(*weight);
                    }
                    pixels[(y * SIDE as usize + x) * 3 + channel] = (sum >> row.precision).clamp(0, 255) as u8;
                }
            }
        }
        pixels
    };
    if image.height() == SIDE {
        return Ok(horizontal);
    }
    let coefficients = coefficients(image.height(), SIDE)?;
    let mut pixels = vec![0u8; (SIDE * SIDE * 3) as usize];
    for (y, row) in coefficients.iter().enumerate() {
        for x in 0..SIDE as usize {
            for channel in 0..3 {
                let mut sum = 1i64 << (row.precision - 1);
                for (index, weight) in row.weights.iter().enumerate() {
                    sum += i64::from(horizontal[((row.start + index) * SIDE as usize + x) * 3 + channel])
                        * i64::from(*weight);
                }
                pixels[(y * SIDE as usize + x) * 3 + channel] = (sum >> row.precision).clamp(0, 255) as u8;
            }
        }
    }
    Ok(pixels)
}

pub(crate) fn prepare(encoded: &[u8]) -> Result<Gemma3Patches, InvalidInput> {
    let image = decode_image_discard_alpha(encoded, &PreprocessLimits::default(), dimensions)?;
    prepare_rgb(&image)
}

pub(crate) fn prepare_rgb(image: &RgbImage) -> Result<Gemma3Patches, InvalidInput> {
    let pixels = resize_rgb(image)?;
    let mut patches = Vec::with_capacity((PATCH_ROWS * PATCH_WIDTH) as usize);
    for py in 0..SIDE / PATCH {
        for px in 0..SIDE / PATCH {
            for channel in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let position = (((py * PATCH + y) * SIDE + px * PATCH + x) * 3 + channel) as usize;
                        let rescaled = (f64::from(pixels[position]) / 255.0) as f32;
                        patches.push(bf16::from_f32((rescaled - 0.5) / 0.5));
                    }
                }
            }
        }
    }
    Ok(Gemma3Patches {
        patches: patches.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use image::ImageEncoder;
    use sha2::{Digest, Sha256};
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn gemma3_patches_preserve_conv_channel_order_and_normalization() {
        let image = RgbImage::try_new(1, 1, vec![0, 127, 255], &PreprocessLimits::default()).unwrap();
        let prepared = prepare_rgb(&image).unwrap();
        assert_eq!(prepared.patches.len(), (PATCH_ROWS * PATCH_WIDTH) as usize);
        for patch in prepared.patches.chunks_exact(PATCH_WIDTH as usize) {
            assert!(patch[..196].iter().all(|x| x.to_f32() == -1.0));
            assert!(patch[196..392].iter().all(|x| *x == bf16::from_f32((127.0f32 / 255.0 - 0.5) / 0.5)));
            assert!(patch[392..].iter().all(|x| x.to_f32() == 1.0));
        }
    }

    #[stark_test]
    fn gemma3_discards_alpha_and_rejects_invalid_geometry() {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(&[17, 83, 149, 0], 1, 1, image::ExtendedColorType::Rgba8)
            .unwrap();
        let native = prepare(&encoded).unwrap();
        let expected =
            prepare_rgb(&RgbImage::try_new(1, 1, vec![17, 83, 149], &PreprocessLimits::default()).unwrap()).unwrap();
        assert_eq!(native.patches, expected.patches);
        assert!(dimensions(0, 1).is_err());
        assert!(dimensions(16385, 1).is_err());
        assert!(dimensions(8000, 8000).is_err());
        assert!(prepare(b"invalid image").is_err());
        assert!(coefficients(0, 896).is_err());
    }

    #[stark_test]
    fn gemma3_nonuniform_resize_matches_pinned_torchvision_cpu() {
        // Original CPU processor, Transformers 5.15.1 / Torch 2.14.1. Hashes
        // cover every BF16 patch converted to little-endian FP32, including
        // upsampling, antialiased downsampling and each unchanged-size axis.
        for (width, height, expected) in [
            (111, 79, "b4000b96dc6cd6eed18564b526bcdb6b476ae6f941b3d07013833b30f4dd3687"),
            (1793, 1297, "b3f03bdd12c6c1fc15349e1603d4a7bec4fb0f2e1e54dfd71a57b1f4b505f0c2"),
            (896, 79, "ebee190f73267d4c561fbfd8162bd03c4b32722626901e624959446845b16d5e"),
            (111, 896, "a4197804a4280661063a14b978d8124e7e8de34d8a7e99cd6da06af314bcf0fc"),
        ] {
            let mut pixels = Vec::with_capacity((width * height * 3) as usize);
            for y in 0..height {
                for x in 0..width {
                    pixels.extend_from_slice(&[
                        ((x * 17 + y * 3) % 256) as u8,
                        ((x * 5 + y * 19) % 256) as u8,
                        ((x * 11 + y * 7) % 256) as u8,
                    ]);
                }
            }
            let image = RgbImage::try_new(width, height, pixels, &PreprocessLimits::default()).unwrap();
            let patches = prepare_rgb(&image).unwrap();
            let mut hash = Sha256::new();
            for value in &patches.patches {
                hash.update(value.to_f32().to_le_bytes());
            }
            assert_eq!(format!("{:x}", hash.finalize()), expected, "pinned nonuniform processor differs");
        }
    }
}
