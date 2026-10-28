//! Owned, bounded static-image decoding and Qwen3.5 patch preparation.
//!
//! The layout, resize dimensions and fused normalization independently implement
//! Transformers 2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397 Qwen2VLImageProcessor
//! and TorchvisionBackend (Qwen/Hugging Face, Apache-2.0). The bicubic convolution
//! uses the standard a=-0.5 antialias kernel and uint8 coefficient quantization
//! described by PyTorch v2.11.0 UpSampleKernel.cpp. No Python runtime is used.
//!
//! Decoder policy: one 8-bit JPEG/PNG, upright EXIF orientation, RGB channels,
//! grayscale expansion and transparency composited over white. ICC profiles are
//! not color transformed. Compressed PNG metadata, CMYK, 16-bit, animated and
//! multi-image inputs fail before decoding.
//! Errors are content-free; request pixels and patch vectors have no Debug impl.

use std::io::Cursor;

use half::bf16;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};

use crate::backends::common::InvalidInput;

const PATCH: u32 = 16;
const MERGE: u32 = 2;
const TEMPORAL: u32 = 2;
const MIN_PIXELS: u64 = 65_536;
const MAX_PIXELS: u64 = 16_777_216;
const MAX_JPEG_SCANS: usize = 64;

/// Internal admission bounds, not qualified product/hardware capabilities.
#[derive(Clone, Copy)]
pub(crate) struct PreprocessLimits {
    pub(crate) max_encoded_bytes: usize,
    pub(crate) max_source_dimension: u32,
    pub(crate) max_source_pixels: u64,
    pub(crate) max_processed_pixels: u64,
    pub(crate) max_encoder_patches: u32,
}

impl Default for PreprocessLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 20 * 1024 * 1024,
            max_source_dimension: 16_384,
            max_source_pixels: 32_000_000,
            max_processed_pixels: 1_048_576,
            max_encoder_patches: 4096,
        }
    }
}

pub(crate) struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

pub(crate) struct PreparedImage {
    pub(crate) patches: Box<[bf16]>,
    pub(crate) grid: [u32; 3],
    pub(crate) resized_width: u32,
    pub(crate) resized_height: u32,
}

impl RgbImage {
    pub(crate) fn try_new(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        limits: &PreprocessLimits,
    ) -> Result<Self, InvalidInput> {
        check_source_dimensions(width, height, limits)?;
        if pixels.len() != rgb_bytes(width, height)? {
            return Err(InvalidInput("RGB image size differs from dimensions"));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    #[cfg(test)]
    pub(super) fn pixels_for_reference(&self) -> &[u8] {
        &self.pixels
    }
}

fn rgb_bytes(
    width: u32,
    height: u32,
) -> Result<usize, InvalidInput> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(InvalidInput("RGB image size overflow"))
}

fn zeroed_bytes(count: usize) -> Result<Vec<u8>, InvalidInput> {
    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|_| InvalidInput("image allocation failed"))?;
    data.resize(count, 0);
    Ok(data)
}

fn check_source_dimensions(
    width: u32,
    height: u32,
    limits: &PreprocessLimits,
) -> Result<(), InvalidInput> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0
        || height == 0
        || width > limits.max_source_dimension
        || height > limits.max_source_dimension
        || pixels > limits.max_source_pixels
    {
        return Err(InvalidInput("image dimensions exceed preprocessing bounds"));
    }
    rgb_bytes(width, height)?;
    Ok(())
}

/// Match pinned smart_resize, including Python's ties-to-even rounding.
/// Product admission rejects an oversized pinned result rather than changing it.
pub(crate) fn resized_dimensions(
    width: u32,
    height: u32,
    limits: &PreprocessLimits,
) -> Result<(u32, u32), InvalidInput> {
    check_source_dimensions(width, height, limits)?;
    if f64::from(width.max(height)) / f64::from(width.min(height)) > 200.0 {
        return Err(InvalidInput("image aspect ratio exceeds processor bounds"));
    }
    let factor = f64::from(PATCH * MERGE);
    let (h, w) = (f64::from(height), f64::from(width));
    let mut target_h = (h / factor).round_ties_even() * factor;
    let mut target_w = (w / factor).round_ties_even() * factor;
    if target_h * target_w > MAX_PIXELS as f64 {
        let beta = ((h * w) / MAX_PIXELS as f64).sqrt();
        target_h = factor.max((h / beta / factor).floor() * factor);
        target_w = factor.max((w / beta / factor).floor() * factor);
    } else if target_h * target_w < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        target_h = (h * beta / factor).ceil() * factor;
        target_w = (w * beta / factor).ceil() * factor;
    }
    if target_h < factor || target_w < factor || target_h > u32::MAX as f64 || target_w > u32::MAX as f64 {
        return Err(InvalidInput("processed image dimensions overflow"));
    }
    let (target_w, target_h) = (target_w as u32, target_h as u32);
    let pixels = u64::from(target_w) * u64::from(target_h);
    let patches = pixels / u64::from(PATCH * PATCH);
    if pixels > limits.max_processed_pixels || patches > u64::from(limits.max_encoder_patches) {
        return Err(InvalidInput("processed image exceeds native admission bounds"));
    }
    rgb_bytes(target_w, target_h)?;
    Ok((target_w, target_h))
}

fn be_u16(bytes: &[u8]) -> Result<u16, InvalidInput> {
    bytes
        .get(..2)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u16::from_be_bytes)
        .ok_or(InvalidInput("truncated image header"))
}

fn be_u32(bytes: &[u8]) -> Result<u32, InvalidInput> {
    bytes
        .get(..4)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_be_bytes)
        .ok_or(InvalidInput("truncated image header"))
}

fn exif_orientation(tiff: &[u8]) -> Result<u8, InvalidInput> {
    let little = match tiff.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Err(InvalidInput("invalid EXIF metadata")),
    };
    let u16_at = |offset: usize| -> Result<u16, InvalidInput> {
        let bytes: [u8; 2] = tiff
            .get(offset..offset.checked_add(2).ok_or(InvalidInput("EXIF offset overflow"))?)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(InvalidInput("invalid EXIF metadata"))?;
        Ok(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let u32_at = |offset: usize| -> Result<u32, InvalidInput> {
        let bytes: [u8; 4] = tiff
            .get(offset..offset.checked_add(4).ok_or(InvalidInput("EXIF offset overflow"))?)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(InvalidInput("invalid EXIF metadata"))?;
        Ok(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };
    if u16_at(2)? != 42 {
        return Err(InvalidInput("invalid EXIF metadata"));
    }
    let offset = usize::try_from(u32_at(4)?).map_err(|_| InvalidInput("EXIF offset overflow"))?;
    let count = usize::from(u16_at(offset)?);
    let start = offset.checked_add(2).ok_or(InvalidInput("EXIF offset overflow"))?;
    let end = start
        .checked_add(count.checked_mul(12).ok_or(InvalidInput("EXIF size overflow"))?)
        .ok_or(InvalidInput("EXIF offset overflow"))?;
    if offset < 8 || end.checked_add(4).is_none_or(|end| end > tiff.len()) {
        return Err(InvalidInput("invalid EXIF metadata"));
    }
    let mut orientation = None;
    for index in 0..count {
        let entry = start + index * 12;
        if u16_at(entry)? == 0x112 {
            if orientation.is_some() || u16_at(entry + 2)? != 3 || u32_at(entry + 4)? != 1 {
                return Err(InvalidInput("invalid EXIF orientation"));
            }
            let value = u16_at(entry + 8)?;
            if !(1..=8).contains(&value) {
                return Err(InvalidInput("invalid EXIF orientation"));
            }
            orientation = Some(value as u8);
        }
    }
    Ok(orientation.unwrap_or(1))
}

fn png_header(data: &[u8]) -> Result<(u32, u32, u8), InvalidInput> {
    let mut cursor = 8usize;
    let mut shape = None;
    let mut orientation = None;
    let mut ended = false;
    while cursor < data.len() {
        let length = be_u32(data.get(cursor..).ok_or(InvalidInput("invalid PNG framing"))?)? as usize;
        let kind_start = cursor.checked_add(4).ok_or(InvalidInput("PNG chunk overflow"))?;
        let payload_start = kind_start.checked_add(4).ok_or(InvalidInput("PNG chunk overflow"))?;
        let payload_end = payload_start.checked_add(length).ok_or(InvalidInput("PNG chunk overflow"))?;
        let chunk_end = payload_end.checked_add(4).ok_or(InvalidInput("PNG chunk overflow"))?;
        let kind = data.get(kind_start..payload_start).ok_or(InvalidInput("truncated PNG chunk"))?;
        let payload = data.get(payload_start..payload_end).ok_or(InvalidInput("truncated PNG chunk"))?;
        if chunk_end > data.len() || ended {
            return Err(InvalidInput("invalid PNG framing"));
        }
        match kind {
            b"IHDR" => {
                if shape.is_some() || cursor != 8 || length != 13 || payload[8] != 8 {
                    return Err(InvalidInput("only static 8-bit PNG is supported"));
                }
                shape = Some((be_u32(payload)?, be_u32(&payload[4..])?));
            },
            b"acTL" | b"fcTL" | b"fdAT" => return Err(InvalidInput("animated PNG is unsupported")),
            // The image wrapper eagerly parses ancillary metadata. Reject its
            // compressed forms before codec construction, rather than relying
            // on a best-effort allocation limit to bound unrelated inflation.
            b"iCCP" | b"zTXt" => return Err(InvalidInput("compressed PNG metadata is unsupported")),
            b"iTXt" => {
                let flag = payload.iter().position(|byte| *byte == 0).and_then(|end| payload.get(end + 1));
                match flag {
                    Some(0) => {},
                    Some(1) => return Err(InvalidInput("compressed PNG metadata is unsupported")),
                    _ => return Err(InvalidInput("invalid PNG text metadata")),
                }
            },
            b"eXIf" => {
                if orientation.is_some() {
                    return Err(InvalidInput("duplicate EXIF metadata"));
                }
                orientation = Some(exif_orientation(payload)?);
            },
            b"IEND" => {
                if length != 0 || chunk_end != data.len() {
                    return Err(InvalidInput("invalid PNG termination"));
                }
                ended = true;
            },
            _ => {},
        }
        cursor = chunk_end;
    }
    let (width, height) = shape.filter(|_| ended).ok_or(InvalidInput("incomplete PNG image"))?;
    Ok((width, height, orientation.unwrap_or(1)))
}

fn jpeg_header(data: &[u8]) -> Result<(u32, u32, u8), InvalidInput> {
    let mut cursor = 2usize;
    let mut entropy = false;
    let mut shape = None;
    let mut orientation = None;
    let mut scans = 0usize;
    loop {
        let marker = loop {
            let byte = *data.get(cursor).ok_or(InvalidInput("incomplete JPEG image"))?;
            cursor += 1;
            if byte != 0xff {
                if entropy {
                    continue;
                }
                return Err(InvalidInput("invalid JPEG framing"));
            }
            while data.get(cursor) == Some(&0xff) {
                cursor += 1;
            }
            let marker = *data.get(cursor).ok_or(InvalidInput("incomplete JPEG marker"))?;
            cursor += 1;
            if entropy && (marker == 0 || (0xd0..=0xd7).contains(&marker)) {
                continue;
            }
            break marker;
        };
        entropy = false;
        if marker == 0xd9 {
            if cursor != data.len() {
                return Err(InvalidInput("multiple or trailing JPEG images are unsupported"));
            }
            let (width, height) = shape.ok_or(InvalidInput("missing JPEG frame"))?;
            return Ok((width, height, orientation.unwrap_or(1)));
        }
        if marker == 0xd8 || marker == 0 || (0xd0..=0xd7).contains(&marker) {
            return Err(InvalidInput("invalid or multiple JPEG frames"));
        }
        if marker == 1 {
            continue;
        }
        let length = usize::from(be_u16(data.get(cursor..).ok_or(InvalidInput("truncated JPEG segment"))?)?);
        if length < 2 {
            return Err(InvalidInput("invalid JPEG segment length"));
        }
        let start = cursor.checked_add(2).ok_or(InvalidInput("JPEG segment overflow"))?;
        let end = cursor.checked_add(length).ok_or(InvalidInput("JPEG segment overflow"))?;
        let payload = data.get(start..end).ok_or(InvalidInput("truncated JPEG segment"))?;
        match marker {
            0xc0..=0xc2 => {
                if shape.is_some() || payload.len() < 6 || payload[0] != 8 || ![1, 3].contains(&payload[5]) {
                    return Err(InvalidInput("unsupported JPEG frame or color format"));
                }
                if payload.len() != 6 + usize::from(payload[5]) * 3
                    || payload[6..].as_chunks::<3>().0.iter().any(|component| {
                        !(1..=4).contains(&(component[1] >> 4)) || !(1..=4).contains(&(component[1] & 15))
                    })
                {
                    return Err(InvalidInput("invalid JPEG component layout"));
                }
                shape = Some((u32::from(be_u16(&payload[3..])?), u32::from(be_u16(&payload[1..])?)));
            },
            0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                return Err(InvalidInput("unsupported JPEG frame encoding"));
            },
            0xe1 if payload.starts_with(b"Exif\0\0") => {
                if orientation.is_some() {
                    return Err(InvalidInput("duplicate EXIF metadata"));
                }
                orientation = Some(exif_orientation(&payload[6..])?);
            },
            0xe2 if payload.starts_with(b"MPF\0") => return Err(InvalidInput("multi-picture JPEG is unsupported")),
            0xda => {
                if shape.is_none() {
                    return Err(InvalidInput("missing JPEG frame"));
                }
                scans += 1;
                if scans > MAX_JPEG_SCANS {
                    return Err(InvalidInput("JPEG scan count exceeds preprocessing bounds"));
                }
                entropy = true;
            },
            _ => {},
        }
        cursor = end;
    }
}

fn decode_jpeg_rgb(
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, InvalidInput> {
    // Header geometry/component checks precede coefficient storage allocation.
    // The decoder limit additionally bounds its final pixel output buffer.
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(data));
    decoder.set_max_decoding_buffer_size(rgb_bytes(width, height)?);
    decoder.read_info().map_err(|_| InvalidInput("JPEG header decoding failed"))?;
    let info = decoder.info().ok_or(InvalidInput("missing JPEG image information"))?;
    if (u32::from(info.width), u32::from(info.height)) != (width, height)
        || !matches!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24 | jpeg_decoder::PixelFormat::L8)
    {
        return Err(InvalidInput("unsupported JPEG decoded layout"));
    }
    let pixels = decoder.decode().map_err(|_| InvalidInput("JPEG decoding failed"))?;
    match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 if pixels.len() == rgb_bytes(width, height)? => Ok(pixels),
        jpeg_decoder::PixelFormat::L8 if pixels.len() == rgb_bytes(width, height)? / 3 => {
            let mut rgb = zeroed_bytes(rgb_bytes(width, height)?)?;
            for (pixel, gray) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(pixels) {
                pixel.fill(gray);
            }
            Ok(rgb)
        },
        _ => Err(InvalidInput("JPEG pixel count differs from dimensions")),
    }
}

/// Decode only bounded static JPEG/PNG bytes; no path or network input exists.
pub(crate) fn decode_image(
    data: &[u8],
    limits: &PreprocessLimits,
) -> Result<RgbImage, InvalidInput> {
    decode_image_alpha(data, limits, true, |width, height| resized_dimensions(width, height, limits).map(|_| ()))
}

/// Match PIL's RGB conversion for processors that discard alpha channels.
/// Codec and allocation bounds are shared with the white-compositing profile.
pub(super) fn decode_image_discard_alpha<F: FnOnce(u32, u32) -> Result<(), InvalidInput>>(
    data: &[u8],
    limits: &PreprocessLimits,
    validate_processed: F,
) -> Result<RgbImage, InvalidInput> {
    decode_image_alpha(data, limits, false, validate_processed)
}

fn decode_image_alpha<F: FnOnce(u32, u32) -> Result<(), InvalidInput>>(
    data: &[u8],
    limits: &PreprocessLimits,
    white_background: bool,
    validate_processed: F,
) -> Result<RgbImage, InvalidInput> {
    if data.is_empty() || data.len() > limits.max_encoded_bytes {
        return Err(InvalidInput("encoded image exceeds preprocessing bounds"));
    }
    let (format, (width, height, orientation)) = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        (ImageFormat::Png, png_header(data)?)
    } else if data.starts_with(&[0xff, 0xd8]) {
        (ImageFormat::Jpeg, jpeg_header(data)?)
    } else {
        return Err(InvalidInput("only JPEG and PNG are supported"));
    };
    check_source_dimensions(width, height, limits)?;
    // Reject a processor/admission overflow before allocating the decoded image.
    validate_processed(width, height)?;
    if format == ImageFormat::Jpeg {
        let pixels = decode_jpeg_rgb(data, width, height)?;
        let (pixels, width, height) = orient_rgb(pixels, width, height, orientation)?;
        return RgbImage::try_new(width, height, pixels, limits);
    }
    let mut codec_limits = Limits::default();
    codec_limits.max_image_width = Some(limits.max_source_dimension);
    codec_limits.max_image_height = Some(limits.max_source_dimension);
    // Decoder max_alloc is best effort; strict dimension checks and the checked
    // pixel/output sizes above provide the independent geometry boundary.
    codec_limits.max_alloc = Some(limits.max_source_pixels.saturating_mul(8));
    let mut reader = ImageReader::with_format(Cursor::new(data), format);
    reader.limits(codec_limits);
    let decoder = reader.into_decoder().map_err(|_| InvalidInput("image header decoding failed"))?;
    if decoder.dimensions() != (width, height) || decoder.total_bytes() > limits.max_source_pixels.saturating_mul(4) {
        return Err(InvalidInput("decoded image layout exceeds preprocessing bounds"));
    }
    let decoded = DynamicImage::from_decoder(decoder).map_err(|_| InvalidInput("image decoding failed"))?;
    let pixels = match decoded {
        DynamicImage::ImageRgb8(image) => image.into_raw(),
        DynamicImage::ImageLuma8(image) => {
            let mut rgb = zeroed_bytes(rgb_bytes(width, height)?)?;
            for (pixel, gray) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(image.into_raw()) {
                pixel.fill(gray);
            }
            rgb
        },
        DynamicImage::ImageLumaA8(image) => {
            let mut rgb = zeroed_bytes(rgb_bytes(width, height)?)?;
            for (pixel, gray) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(image.into_raw().as_chunks::<2>().0.iter()) {
                pixel.fill(if white_background {
                    composite_white(gray[0], gray[1])
                } else {
                    gray[0]
                });
            }
            rgb
        },
        DynamicImage::ImageRgba8(image) => {
            let mut rgb = zeroed_bytes(rgb_bytes(width, height)?)?;
            for (pixel, rgba) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(image.into_raw().as_chunks::<4>().0.iter()) {
                for channel in 0..3 {
                    pixel[channel] = if white_background {
                        composite_white(rgba[channel], rgba[3])
                    } else {
                        rgba[channel]
                    };
                }
            }
            rgb
        },
        _ => return Err(InvalidInput("only 8-bit image channels are supported")),
    };
    let (pixels, width, height) = orient_rgb(pixels, width, height, orientation)?;
    RgbImage::try_new(width, height, pixels, limits)
}

fn composite_white(
    value: u8,
    alpha: u8,
) -> u8 {
    ((u32::from(value) * u32::from(alpha) + 255 * (255 - u32::from(alpha)) + 127) / 255) as u8
}

fn orient_rgb(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    orientation: u8,
) -> Result<(Vec<u8>, u32, u32), InvalidInput> {
    if orientation == 1 {
        return Ok((pixels, width, height));
    }
    let (out_width, out_height) = if orientation >= 5 {
        (height, width)
    } else {
        (width, height)
    };
    let mut output = zeroed_bytes(rgb_bytes(out_width, out_height)?)?;
    for y in 0..out_height {
        for x in 0..out_width {
            let (source_x, source_y) = match orientation {
                2 => (width - 1 - x, y),
                3 => (width - 1 - x, height - 1 - y),
                4 => (x, height - 1 - y),
                5 => (y, x),
                6 => (y, height - 1 - x),
                7 => (width - 1 - y, height - 1 - x),
                8 => (width - 1 - y, x),
                _ => return Err(InvalidInput("invalid EXIF orientation")),
            };
            let source = (source_y as usize * width as usize + source_x as usize) * 3;
            let destination = (y as usize * out_width as usize + x as usize) * 3;
            output[destination..destination + 3].copy_from_slice(&pixels[source..source + 3]);
        }
    }
    Ok((output, out_width, out_height))
}

fn cubic(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

struct ResizeWeights {
    start: usize,
    weights: Vec<i32>,
}

fn resize_weights(
    input: u32,
    output: u32,
) -> Result<(Vec<ResizeWeights>, u32), InvalidInput> {
    let scale = f64::from(input) / f64::from(output);
    let support = 2.0 * scale.max(1.0);
    let invscale = 1.0 / scale.max(1.0);
    let mut rows = Vec::new();
    rows.try_reserve_exact(output as usize).map_err(|_| InvalidInput("resize coefficient allocation failed"))?;
    let mut maximum = 0.0f64;
    for index in 0..output {
        let center = scale * (f64::from(index) + 0.5);
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(i64::from(input)).max(start as i64) as usize;
        let mut weights = Vec::new();
        weights.try_reserve_exact(end - start).map_err(|_| InvalidInput("resize coefficient allocation failed"))?;
        let mut total = 0.0;
        for source in start..end {
            let weight = cubic((source as f64 - center + 0.5) * invscale);
            weights.push(weight);
            total += weight;
        }
        if total == 0.0 {
            return Err(InvalidInput("invalid resize coefficients"));
        }
        for weight in &mut weights {
            *weight /= total;
            maximum = maximum.max(*weight);
        }
        rows.push((start, weights));
    }
    let mut precision = 0u32;
    while precision < 22 {
        if (0.5 + maximum * f64::from(1u32 << (precision + 1))) as i32 >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let multiplier = f64::from(1u32 << precision);
    Ok((
        rows.into_iter()
            .map(|(start, weights)| ResizeWeights {
                start,
                weights: weights.into_iter().map(|weight| (weight * multiplier).round() as i32).collect(),
            })
            .collect(),
        precision,
    ))
}

pub(super) fn resize_rgb(
    image: &RgbImage,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, InvalidInput> {
    let horizontal = if image.width == width {
        let mut copy = zeroed_bytes(image.pixels.len())?;
        copy.copy_from_slice(&image.pixels);
        copy
    } else {
        let (rows, precision) = resize_weights(image.width, width)?;
        let mut output = zeroed_bytes(rgb_bytes(width, image.height)?)?;
        for y in 0..image.height as usize {
            for (x, row) in rows.iter().enumerate() {
                for channel in 0..3 {
                    let mut value = 1i64 << (precision - 1);
                    for (index, &weight) in row.weights.iter().enumerate() {
                        value += i64::from(image.pixels[(y * image.width as usize + row.start + index) * 3 + channel])
                            * i64::from(weight);
                    }
                    output[(y * width as usize + x) * 3 + channel] = (value >> precision).clamp(0, 255) as u8;
                }
            }
        }
        output
    };
    if image.height == height {
        return Ok(horizontal);
    }
    let (rows, precision) = resize_weights(image.height, height)?;
    let mut output = zeroed_bytes(rgb_bytes(width, height)?)?;
    for (y, row) in rows.iter().enumerate() {
        for x in 0..width as usize {
            for channel in 0..3 {
                let mut value = 1i64 << (precision - 1);
                for (index, &weight) in row.weights.iter().enumerate() {
                    value += i64::from(horizontal[((row.start + index) * width as usize + x) * 3 + channel])
                        * i64::from(weight);
                }
                output[(y * width as usize + x) * 3 + channel] = (value >> precision).clamp(0, 255) as u8;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
pub(super) fn resize_rgb_for_reference(
    image: &RgbImage,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, InvalidInput> {
    resize_rgb(image, width, height)
}

pub(crate) fn prepare_image(
    image: &RgbImage,
    limits: &PreprocessLimits,
) -> Result<PreparedImage, InvalidInput> {
    let (width, height) = resized_dimensions(image.width, image.height, limits)?;
    let resized = resize_rgb(image, width, height)?;
    let (grid_h, grid_w) = (height / PATCH, width / PATCH);
    let count = u64::from(grid_h)
        .checked_mul(u64::from(grid_w))
        .and_then(|rows| rows.checked_mul(u64::from(3 * TEMPORAL * PATCH * PATCH)))
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| *count <= isize::MAX as usize / size_of::<bf16>())
        .ok_or(InvalidInput("image patch allocation overflow"))?;
    let mut patches = Vec::new();
    patches.try_reserve_exact(count).map_err(|_| InvalidInput("image patch allocation failed"))?;
    for block_y in 0..grid_h / MERGE {
        for block_x in 0..grid_w / MERGE {
            for inner_y in 0..MERGE {
                for inner_x in 0..MERGE {
                    for channel in 0..3 {
                        for _temporal in 0..TEMPORAL {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    let y = (block_y * MERGE + inner_y) * PATCH + py;
                                    let x = (block_x * MERGE + inner_x) * PATCH + px;
                                    let pixel = resized[(y as usize * width as usize + x as usize) * 3 + channel];
                                    // Match fused Torchvision normalization order, then model BF16 input.
                                    patches.push(bf16::from_f32((f32::from(pixel) - 127.5) / 127.5));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(PreparedImage {
        patches: patches.into_boxed_slice(),
        grid: [1, grid_h, grid_w],
        resized_width: width,
        resized_height: height,
    })
}

#[cfg(test)]
#[path = "preprocess_tests.rs"]
mod tests;
