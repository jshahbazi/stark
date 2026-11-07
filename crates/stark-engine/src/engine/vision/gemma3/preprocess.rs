//! Bounded Gemma 3 fixed-resolution image preprocessing.
//!
//! Follows Google/Hugging Face Transformers v4.53.3 Gemma3ImageProcessor:
//! upright RGB, alpha discarded, PIL antialiased bilinear uint8 resize to
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

// Pillow 11.x Resample.c uses 22-bit normalized coefficients, a half-pixel
// center and antialias filter support widened by the downsampling factor.
struct Coefficients {
    start: usize,
    weights: Vec<i32>,
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
    let mut result = Vec::with_capacity(output as usize);
    for index in 0..output {
        let center = (f64::from(index) + 0.5) * scale;
        let start = ((center - filter_scale + 0.5) as i64).max(0) as usize;
        let end = ((center + filter_scale + 0.5) as i64).min(i64::from(input)).max(start as i64) as usize;
        let weights: Vec<f64> = (start..end)
            .map(|position| (1.0 - ((position as f64 - center + 0.5) / filter_scale).abs()).max(0.0))
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(InvalidInput("Gemma 3 resize has empty coefficients"));
        }
        result.push(Coefficients {
            start,
            weights: weights.into_iter().map(|weight| (0.5 + weight / total * f64::from(1 << 22)) as i32).collect(),
        });
    }
    Ok(result)
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
                    let mut sum = 1i64 << 21;
                    for (index, weight) in row.weights.iter().enumerate() {
                        sum +=
                            i64::from(image.pixels()[(y * image.width() as usize + row.start + index) * 3 + channel])
                                * i64::from(*weight);
                    }
                    pixels[(y * SIDE as usize + x) * 3 + channel] = (sum >> 22).clamp(0, 255) as u8;
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
                let mut sum = 1i64 << 21;
                for (index, weight) in row.weights.iter().enumerate() {
                    sum += i64::from(horizontal[((row.start + index) * SIDE as usize + x) * 3 + channel])
                        * i64::from(*weight);
                }
                pixels[(y * SIDE as usize + x) * 3 + channel] = (sum >> 22).clamp(0, 255) as u8;
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
}
