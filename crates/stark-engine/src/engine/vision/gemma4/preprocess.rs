//! Native Gemma 4 E2B image preprocessing. The aspect-ratio patch budget follows
//! MLX-VLM 53ebd1647cb83549b482a031ea70ba9d9ea60a04 / Google's Gemma4ImageProcessor.
//! Reuses Stark's bounded upright RGB decoder and independently implemented
//! antialiased uint8 bicubic resizer; no Python image runtime is involved.

use half::bf16;

use super::super::preprocess::{PreprocessLimits, RgbImage, decode_image_discard_alpha, resize_rgb};
use crate::backends::common::InvalidInput;

pub(crate) struct GemmaPatches {
    pub(crate) patches: Box<[bf16]>,
    pub(crate) grid: [u32; 3],
    pub(crate) width: u32,
    pub(crate) height: u32,
}

pub(crate) fn dimensions(
    width: u32,
    height: u32,
) -> Result<(u32, u32), InvalidInput> {
    if width == 0 || height == 0 || width > 16384 || height > 16384 || u64::from(width) * u64::from(height) > 32_000_000
    {
        return Err(InvalidInput("Gemma source image dimensions exceed bounds"));
    }
    let factor = (2520.0 * 256.0 / (f64::from(width) * f64::from(height))).sqrt();
    let mut target_width = (factor * f64::from(width) / 48.0).floor() as u32 * 48;
    let mut target_height = (factor * f64::from(height) / 48.0).floor() as u32 * 48;
    if target_width == 0 && target_height == 0 {
        return Err(InvalidInput("Gemma processed image is empty"));
    }
    if target_width == 0 {
        target_width = 48;
        target_height = ((f64::from(height) / f64::from(width)).floor() as u32).min(280) * 48;
    }
    if target_height == 0 {
        target_height = 48;
        target_width = ((f64::from(width) / f64::from(height)).floor() as u32).min(280) * 48;
    }
    if target_width == 0 || target_height == 0 || u64::from(target_width) * u64::from(target_height) > 645120 {
        return Err(InvalidInput("Gemma processed image exceeds patch budget"));
    }
    Ok((target_width, target_height))
}

pub(crate) fn prepare(encoded: &[u8]) -> Result<GemmaPatches, InvalidInput> {
    // Transformers 5.15.1's Gemma4ImageProcessor uses PIL convert("RGB"),
    // retaining color samples while discarding alpha before patch resizing.
    let image = decode_image_discard_alpha(encoded, &PreprocessLimits::default(), |width, height| {
        dimensions(width, height).map(|_| ())
    })?;
    prepare_rgb(&image)
}

pub(crate) fn prepare_rgb(image: &RgbImage) -> Result<GemmaPatches, InvalidInput> {
    let (width, height) = dimensions(image.width(), image.height())?;
    let pixels = resize_rgb(image, width, height)?;
    let gw = width / 16;
    let gh = height / 16;
    let mut patches = Vec::with_capacity((gw * gh * 768) as usize);
    for patch_y in 0..gh {
        for patch_x in 0..gw {
            for y in 0..16 {
                for x in 0..16 {
                    for c in 0..3 {
                        let index = (((patch_y * 16 + y) * width + patch_x * 16 + x) * 3 + c) as usize;
                        let rescaled = f32::from(pixels[index]) * (1.0 / 255.0);
                        patches.push(bf16::from_f32(2.0 * (rescaled - 0.5)));
                    }
                }
            }
        }
    }
    Ok(GemmaPatches {
        patches: patches.into_boxed_slice(),
        grid: [1, gh, gw],
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use image::ImageEncoder;
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn gemma_image_resize_budget_is_aspect_preserving_and_pool_divisible() {
        assert_eq!(dimensions(256, 256).unwrap(), (768, 768));
        assert_eq!(dimensions(640, 480).unwrap(), (912, 672));
        assert_eq!(dimensions(1, 280).unwrap(), (48, 13440));
        for (w, h) in [(1, 1), (16, 9), (9, 16), (8192, 1), (1, 8192), (640, 480)] {
            let (w, h) = dimensions(w, h).unwrap();
            assert_eq!(w % 48, 0);
            assert_eq!(h % 48, 0);
            assert!(w * h / 256 <= 2520);
        }
        assert!(dimensions(0, 1).is_err());
        assert!(dimensions(u32::MAX, 1).is_err());
    }

    #[stark_test]
    fn gemma_image_rgb_conversion_discards_alpha_before_resize() {
        // Canonical Transformers 5.15.1 convert_to_rgb delegates to PIL's RGB
        // conversion, which discards alpha without changing color samples.
        for (color, samples, rgb) in [
            (image::ExtendedColorType::Rgba8, vec![17, 83, 149, 0, 71, 113, 229, 128], vec![17, 83, 149, 71, 113, 229]),
            (image::ExtendedColorType::La8, vec![37, 0, 101, 128], vec![37, 37, 37, 101, 101, 101]),
        ] {
            let mut encoded = Vec::new();
            image::codecs::png::PngEncoder::new(&mut encoded).write_image(&samples, 2, 1, color).unwrap();
            let limits = PreprocessLimits::default();
            let native = prepare(&encoded).unwrap();
            let expected = prepare_rgb(&RgbImage::try_new(2, 1, rgb.clone(), &limits).unwrap()).unwrap();
            assert_eq!(native.grid, expected.grid);
            assert_eq!(native.patches, expected.patches);
            let qwen = super::super::super::preprocess::decode_image(&encoded, &limits).unwrap();
            assert_ne!(qwen.pixels_for_reference(), rgb);
        }
    }

    #[stark_test]
    fn gemma_image_codec_admits_its_own_thin_image_geometry() {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(&vec![127; 280 * 3], 1, 280, image::ExtendedColorType::Rgb8)
            .unwrap();
        let native = prepare(&encoded).unwrap();
        assert_eq!((native.width, native.height), (48, 13440));
        assert_eq!(native.grid, [1, 840, 3]);
        assert!(super::super::super::preprocess::decode_image(&encoded, &PreprocessLimits::default()).is_err());
    }
}
