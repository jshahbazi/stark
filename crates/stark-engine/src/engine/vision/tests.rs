//! Native loader and complete image-request boundary checks.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use stark_engine_macros::stark_test;

use super::*;

struct PrivateDirectory(PathBuf);

impl PrivateDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "stark-vision-contract-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path).unwrap();
        Self(path)
    }

    fn write(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> File {
        let mut options = OpenOptions::new();
        options.create_new(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(self.0.join(name)).unwrap();
        file.write_all(bytes).unwrap();
        file.rewind().unwrap();
        file
    }
}

impl Drop for PrivateDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[stark_test]
fn native_vision_artifact_hashes_bind_exact_bytes_and_open_descriptor() {
    let directory = PrivateDirectory::new();
    let payload = br#"{"format":"synthetic-contract","schema":1}"#;
    let expected = format!("{:x}", Sha256::digest(payload));
    let mut file = directory.write("contract", payload);
    assert!(file_matches(&mut file, &expected).unwrap());

    // Replacing a path cannot replace the descriptor selected by admission.
    fs::rename(directory.0.join("contract"), directory.0.join("original")).unwrap();
    let mut replacement = directory.write("contract", b"different bytes");
    file.rewind().unwrap();
    assert!(file_matches(&mut file, &expected).unwrap());
    assert!(!file_matches(&mut replacement, &expected).unwrap());

    // In-place writes to the selected file must fail a subsequent identity check.
    file.set_len(0).unwrap();
    file.rewind().unwrap();
    file.write_all(b"changed selected object").unwrap();
    file.rewind().unwrap();
    assert!(!file_matches(&mut file, &expected).unwrap());
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_vision_loader_rejects_unbound_and_speculative_artifacts() {
    use crate::backends::cpu::Cpu;

    let engine = Engine::<Cpu>::new().unwrap();
    let directory = PrivateDirectory::new();
    assert!(matches!(engine.load_vision_language_model(&directory.0), Err(NativeVisionError::IO(_))));
    directory.write("config.json", br#"{"type":"Qwen35VisionLanguageModelConfig"}"#);
    assert!(matches!(engine.load_vision_language_model(&directory.0), Err(NativeVisionError::InvalidInput(_))));
    fs::remove_file(directory.0.join("config.json")).unwrap();
    fs::create_dir(directory.0.join("speculator")).unwrap();
    assert!(matches!(engine.load_vision_language_model(&directory.0), Err(NativeVisionError::InvalidInput(_))));
}

#[cfg(backend = "metal")]
fn official_image_tokens() -> Box<[u64]> {
    [100, 248053].into_iter().chain(std::iter::repeat_n(248056, 64)).chain([248054, 101]).collect()
}

#[cfg(backend = "metal")]
fn synthetic_png() -> Vec<u8> {
    use image::ImageEncoder;

    let pixels = (0..256u32)
        .flat_map(|y| {
            (0..256u32).flat_map(move |x| [(x * 17 + y * 3) as u8, (x * 5 + y * 11) as u8, (x * 7 + y * 13) as u8])
        })
        .collect::<Vec<_>>();
    let mut encoded = Vec::new();
    image::codecs::png::PngEncoder::new(&mut encoded)
        .write_image(&pixels, 256, 256, image::ExtendedColorType::Rgb8)
        .unwrap();
    encoded
}

#[cfg(backend = "metal")]
fn greedy_options<B: Backend>(model: &VisionLanguageModel<B>) -> LanguageModelStreamOptions {
    let mut options = model.default_stream_options();
    options.sampling_method = crate::encodable_block::sampling::SamplingMethod::Greedy;
    options
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore] // Requires the exact private official artifact and physical Metal.
fn native_vision_official_artifact_image_request_metal() {
    use crate::backends::metal::Metal;

    let path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("private model path is required"));
    let engine = Engine::<Metal>::new().unwrap();
    let mut model = engine.load_vision_language_model(&path).unwrap();
    let encoded = synthetic_png();
    assert!(matches!(
        model.prepare_prefill(official_image_tokens(), b"invalid image", 2, 128),
        Err(NativeVisionError::InvalidInput(_))
    ));
    assert!(!model.failed);
    let mut forged_span = official_image_tokens();
    forged_span[3] = 102;
    assert!(matches!(model.prepare_prefill(forged_span, &encoded, 2, 128), Err(NativeVisionError::InvalidInput(_))));
    assert!(!model.failed);

    let prepared = model.prepare_prefill(official_image_tokens(), &encoded, 2, 128).unwrap();
    assert_eq!(prepared.tokens.len(), 68);
    assert_eq!(prepared.image.features.len(), 64 * 1024);
    assert!(prepared.image.features.iter().all(|value| value.is_finite()));
    assert_eq!(prepared.position_delta, -56);
    assert_eq!(prepared.positions.len(), 68);
    drop(prepared);

    // A state from another loaded instance must fail before even decoding the
    // deliberately malformed image, and must not poison either loaded model.
    let mut foreign = engine.load_vision_language_model(&path).unwrap();
    let mut foreign_state = foreign.create_empty_state(Some(128), 0).unwrap();
    assert!(matches!(
        model.stream_image(
            official_image_tokens(),
            b"invalid image",
            2,
            128,
            &mut foreign_state,
            greedy_options(&model)
        ),
        Err(NativeVisionError::Stream(LanguageModelStreamError::ForeignState))
    ));
    assert!(foreign_state.tokens().is_empty());
    assert!(!model.failed);
    drop(foreign_state);
    drop(foreign);

    let mut undersized = model.create_empty_state(Some(64), 0).unwrap();
    assert!(matches!(
        model.stream_image(official_image_tokens(), b"invalid image", 2, 128, &mut undersized, greedy_options(&model)),
        Err(NativeVisionError::Stream(LanguageModelStreamError::ContextOverflow))
    ));
    assert!(undersized.tokens().is_empty());
    assert!(!model.failed);
    drop(undersized);

    // The requested image budget must match its allocated state. Otherwise a
    // larger state would silently allow decoding beyond the caller's budget.
    let mut generous = model.create_empty_state(Some(128), 0).unwrap();
    assert!(matches!(
        model.stream_image(official_image_tokens(), b"invalid image", 2, 70, &mut generous, greedy_options(&model)),
        Err(NativeVisionError::InvalidInput(InvalidInput(
            "native image context must match the allocated request state"
        )))
    ));
    assert!(generous.tokens().is_empty());
    assert_eq!(generous.vision_validation_state(), (0, None));
    assert!(!model.failed);
    drop(generous);

    // Even matching capacities must leave an output slot. The distinct context
    // error proves this check runs before the malformed image is decoded.
    let mut no_headroom = model.create_empty_state(Some(68), 0).unwrap();
    assert!(matches!(
        model.stream_image(official_image_tokens(), b"invalid image", 2, 68, &mut no_headroom, greedy_options(&model)),
        Err(NativeVisionError::Stream(LanguageModelStreamError::ContextOverflow))
    ));
    assert!(no_headroom.tokens().is_empty());
    assert_eq!(no_headroom.vision_validation_state(), (0, None));
    assert!(!model.failed);
    drop(no_headroom);

    let mut state = model.create_empty_state(Some(128), 0).unwrap();
    let mut out_of_vocabulary = official_image_tokens();
    out_of_vocabulary[0] = 248320;
    assert!(matches!(
        model.stream_image(out_of_vocabulary, b"invalid image", 2, 128, &mut state, greedy_options(&model)),
        Err(NativeVisionError::Stream(LanguageModelStreamError::InvalidToken))
    ));
    assert!(state.tokens().is_empty());
    assert!(!model.failed);
    let mut returned = Vec::new();
    {
        let mut stream =
            model.stream_image(official_image_tokens(), &encoded, 2, 128, &mut state, greedy_options(&model)).unwrap();
        assert_eq!(stream.metrics().num_tokens_prefilled, 68);
        for _ in 0..2 {
            let token = stream.next().unwrap().unwrap();
            assert!(token < 248320);
            returned.push(token);
        }
    }
    assert_eq!(state.tokens().len(), 70);
    assert_eq!(state.vision_validation_state(), (70, Some(-56)));
    assert!(!model.failed);
    assert!(matches!(
        model.stream_image(official_image_tokens(), b"invalid image", 2, 128, &mut state, greedy_options(&model)),
        Err(NativeVisionError::Stream(LanguageModelStreamError::InvalidMultimodal))
    ));

    // Recoverable validation errors and completed requests leave the immutable
    // model reusable with fresh state. Compare privately; never print tokens.
    let mut fresh = model.create_empty_state(Some(128), 0).unwrap();
    {
        let mut stream =
            model.stream_image(official_image_tokens(), &encoded, 2, 128, &mut fresh, greedy_options(&model)).unwrap();
        for expected in returned {
            assert!(stream.next().unwrap().unwrap() == expected, "fresh image request differs");
        }
    }
    assert_eq!(fresh.tokens().len(), 70);
    assert_eq!(fresh.vision_validation_state(), (70, Some(-56)));
    assert!(!model.failed);
}
