use image::{
    ExtendedColorType, ImageEncoder,
    codecs::{jpeg::JpegEncoder, png::PngEncoder},
};
use stark_engine_macros::stark_test;

use super::*;

fn limits() -> PreprocessLimits {
    PreprocessLimits::default()
}

fn png(
    width: u32,
    height: u32,
    color: ExtendedColorType,
    pixels: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(pixels, width, height, color).unwrap();
    bytes
}

fn insert_png_chunk(
    bytes: &mut Vec<u8>,
    kind: &[u8; 4],
    payload: &[u8],
) {
    // Tests requiring a rejected header need no CRC: rejection precedes codec dispatch.
    let mut chunk = Vec::new();
    chunk.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(payload);
    chunk.extend_from_slice(&[0; 4]);
    bytes.splice(33..33, chunk);
}

fn exif(orientation: u16) -> Vec<u8> {
    let mut data = Vec::from(&b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0"[..]);
    data.extend_from_slice(&orientation.to_le_bytes());
    data.extend_from_slice(&[0; 6]);
    data
}

#[stark_test]
fn image_dimensions_match_pinned_smart_resize_and_admission() {
    let bounds = limits();
    // Includes ties-to-even: 240/32=7.5 ->8 and 272/32=8.5 ->8.
    for (input, expected) in [
        ((256, 256), (256, 256)),
        ((240, 320), (256, 320)),
        ((272, 272), (256, 256)),
        ((3, 5), (224, 352)),
        ((81, 47), (352, 224)),
    ] {
        assert_eq!(resized_dimensions(input.0, input.1, &bounds).unwrap(), expected);
    }
    assert!(resized_dimensions(1536, 1024, &bounds).is_err());
    assert!(resized_dimensions(0, 32, &bounds).is_err());
    assert!(resized_dimensions(1, 201, &bounds).is_err());
    assert!(resized_dimensions(16_385, 32, &bounds).is_err());
    let constrained = PreprocessLimits {
        max_encoder_patches: 255,
        ..bounds
    };
    assert!(resized_dimensions(256, 256, &constrained).is_err());
}

#[stark_test]
fn image_patch_order_temporal_repeat_and_normalization_match_index_oracle() {
    let bounds = limits();
    let pixels = (0..256 * 256 * 3).map(|index| ((index * 37 + index / 11) % 256) as u8).collect::<Vec<_>>();
    let rgb = RgbImage::try_new(256, 256, pixels.clone(), &bounds).unwrap();
    let prepared = prepare_image(&rgb, &bounds).unwrap();
    assert_eq!(prepared.grid, [1, 16, 16]);
    assert_eq!(prepared.patches.len(), 256 * 1536);
    // Invert the flattened reference axes: [outer_h,outer_w,inner_h,
    // inner_w,channel,temporal,patch_h,patch_w], not the production loops.
    for (index, value) in prepared.patches.iter().enumerate() {
        let mut remainder = index;
        let px = remainder % 16;
        remainder /= 16;
        let py = remainder % 16;
        remainder /= 16;
        let _time = remainder % 2;
        remainder /= 2;
        let channel = remainder % 3;
        remainder /= 3;
        let inner_x = remainder % 2;
        remainder /= 2;
        let inner_y = remainder % 2;
        remainder /= 2;
        let outer_x = remainder % 8;
        let outer_y = remainder / 8;
        let x = (outer_x * 2 + inner_x) * 16 + px;
        let y = (outer_y * 2 + inner_y) * 16 + py;
        let expected = bf16::from_f32((f32::from(pixels[(y * 256 + x) * 3 + channel]) - 127.5) / 127.5);
        assert_eq!(value.to_bits(), expected.to_bits(), "patch coordinate differs from independent flattened index");
    }
}

#[stark_test]
fn image_decode_png_rgb_grayscale_and_alpha_policy() {
    let bounds = limits();
    let rgb = decode_image(&png(2, 1, ExtendedColorType::Rgb8, &[1, 2, 3, 4, 5, 6]), &bounds).unwrap();
    assert_eq!((rgb.width, rgb.height), (2, 1));
    assert_eq!(rgb.pixels, [1, 2, 3, 4, 5, 6]);
    let gray = decode_image(&png(2, 1, ExtendedColorType::L8, &[17, 231]), &bounds).unwrap();
    assert_eq!(gray.pixels, [17, 17, 17, 231, 231, 231]);
    let rgba =
        decode_image(&png(3, 1, ExtendedColorType::Rgba8, &[0, 24, 255, 0, 0, 24, 255, 255, 0, 24, 255, 128]), &bounds)
            .unwrap();
    assert_eq!(rgba.pixels, [255, 255, 255, 0, 24, 255, 127, 139, 255]);
    let gray_alpha = decode_image(&png(2, 1, ExtendedColorType::La8, &[0, 0, 128, 128]), &bounds).unwrap();
    assert_eq!(gray_alpha.pixels, [255, 255, 255, 191, 191, 191]);
}

#[stark_test]
fn image_decode_rejects_animation_depth_truncation_bounds_and_trailing_frames() {
    let bounds = limits();
    let valid = png(2, 2, ExtendedColorType::Rgb8, &[31; 12]);
    let mut animated = valid.clone();
    insert_png_chunk(&mut animated, b"acTL", &[0, 0, 0, 1, 0, 0, 0, 0]);
    assert!(decode_image(&animated, &bounds).is_err());
    for (kind, payload) in [
        (b"iCCP", &b"profile\0\0compressed"[..]),
        (b"zTXt", &b"key\0\0compressed"[..]),
        (b"iTXt", &b"key\0\x01\0\0\0compressed"[..]),
    ] {
        let mut metadata = valid.clone();
        insert_png_chunk(&mut metadata, kind, payload);
        assert!(decode_image(&metadata, &bounds).is_err());
    }
    let depth16 = png(1, 1, ExtendedColorType::L16, &[0, 0]);
    assert!(decode_image(&depth16, &bounds).is_err());
    for end in [0, 1, 8, 16, 32, valid.len() - 1] {
        assert!(decode_image(&valid[..end], &bounds).is_err());
    }
    let mut trailing = valid.clone();
    trailing.extend_from_slice(&valid);
    assert!(decode_image(&trailing, &bounds).is_err());
    assert!(
        decode_image(
            &valid,
            &PreprocessLimits {
                max_encoded_bytes: valid.len() - 1,
                ..bounds
            }
        )
        .is_err()
    );
    assert!(
        decode_image(
            &valid,
            &PreprocessLimits {
                max_source_pixels: 3,
                ..bounds
            }
        )
        .is_err()
    );
    assert!(
        decode_image(
            &valid,
            &PreprocessLimits {
                max_processed_pixels: 65_535,
                ..bounds
            }
        )
        .is_err()
    );
    assert!(decode_image(b"GIF89a", &bounds).is_err());
    assert!(RgbImage::try_new(2, 2, vec![0; 11], &bounds).is_err());
}

#[stark_test]
fn image_exif_orientation_maps_all_eight_coordinate_transforms() {
    let pixels = (1..=6).flat_map(|value| [value, value, value]).collect::<Vec<_>>();
    let expected = [
        (3, 2, vec![1, 2, 3, 4, 5, 6]),
        (3, 2, vec![3, 2, 1, 6, 5, 4]),
        (3, 2, vec![6, 5, 4, 3, 2, 1]),
        (3, 2, vec![4, 5, 6, 1, 2, 3]),
        (2, 3, vec![1, 4, 2, 5, 3, 6]),
        (2, 3, vec![4, 1, 5, 2, 6, 3]),
        (2, 3, vec![6, 3, 5, 2, 4, 1]),
        (2, 3, vec![3, 6, 2, 5, 1, 4]),
    ];
    for (index, (width, height, values)) in expected.into_iter().enumerate() {
        assert_eq!(exif_orientation(&exif(index as u16 + 1)).unwrap(), index as u8 + 1);
        let (actual, actual_width, actual_height) = orient_rgb(pixels.clone(), 3, 2, index as u8 + 1).unwrap();
        assert_eq!((actual_width, actual_height), (width, height));
        assert_eq!(actual.chunks_exact(3).map(|pixel| pixel[0]).collect::<Vec<_>>(), values);
    }
    assert!(exif_orientation(&exif(9)).is_err());
    for length in [0, 1, 4, 8, 10, 20] {
        assert!(exif_orientation(&exif(1)[..length]).is_err());
    }
}

#[stark_test]
fn image_decode_jpeg_and_exif_uses_owned_bounded_rgb() {
    let bounds = limits();
    let mut encoded = Vec::new();
    JpegEncoder::new_with_quality(&mut encoded, 100)
        .encode(&[100; 3 * 8 * 16], 8, 16, ExtendedColorType::Rgb8)
        .unwrap();
    let base = decode_image(&encoded, &bounds).unwrap();
    assert_eq!((base.width, base.height), (8, 16));
    assert!(base.pixels.iter().all(|value| u8::abs_diff(*value, 100) <= 1));
    let mut oriented = encoded.clone();
    let mut app = Vec::from(&b"Exif\0\0"[..]);
    app.extend_from_slice(&exif(6));
    let mut segment = vec![0xff, 0xe1];
    segment.extend_from_slice(&((app.len() + 2) as u16).to_be_bytes());
    segment.extend_from_slice(&app);
    oriented.splice(2..2, segment);
    let upright = decode_image(&oriented, &bounds).unwrap();
    assert_eq!((upright.width, upright.height), (16, 8));
    let mut trailing = encoded.clone();
    trailing.extend_from_slice(&encoded);
    assert!(decode_image(&trailing, &bounds).is_err());
    let mut mpo = encoded.clone();
    mpo.splice(2..2, [0xff, 0xe2, 0, 6, b'M', b'P', b'F', 0]);
    assert!(decode_image(&mpo, &bounds).is_err());
    assert!(decode_image(&encoded[..encoded.len() - 2], &bounds).is_err());
    assert!(jpeg_header(&[0xff, 0xd8, 0xff, 0xc3, 0, 8, 16, 0, 1, 0, 1, 1, 0xff, 0xd9]).is_err());
    let mut too_many_scans = vec![0xff, 0xd8, 0xff, 0xc0, 0, 11, 8, 0, 1, 0, 1, 1, 1, 0x11, 0];
    for _ in 0..=MAX_JPEG_SCANS {
        too_many_scans.extend_from_slice(&[0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0]);
    }
    too_many_scans.extend_from_slice(&[0xff, 0xd9]);
    assert!(jpeg_header(&too_many_scans).is_err());
}

#[stark_test]
fn image_bicubic_resize_preserves_constant_rgb_and_saturates_edges() {
    let bounds = limits();
    let image = RgbImage::try_new(3, 5, [17, 127, 231].repeat(15), &bounds).unwrap();
    let resized = resize_rgb(&image, 224, 352).unwrap();
    assert!(resized.chunks_exact(3).all(|pixel| pixel == [17, 127, 231]));
    let image = RgbImage::try_new(3, 1, vec![0, 0, 0, 255, 255, 255, 0, 0, 0], &bounds).unwrap();
    let resized = resize_rgb(&image, 32, 32).unwrap();
    assert_eq!(resized.len(), 32 * 32 * 3);
    assert_eq!(resized[0], 0);
    assert!(resized.iter().copied().max().unwrap() > 240);
}
