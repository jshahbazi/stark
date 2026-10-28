//! Opt-in private reference replay. The generated vectors are never committed
//! or printed; diagnostics contain aggregate error metrics only.

use std::{fs::File, path::PathBuf};

use half::bf16;
use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::{
    preprocess::{PreprocessLimits, RgbImage, decode_image, prepare_image, resize_rgb_for_reference},
    tower::{VisionBlockStage, VisionStage, VisionTower, VisionTowerConfig},
};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    },
    parameters::ParameterLoader,
};

fn reference_root() -> PathBuf {
    std::env::var_os("STARK_VISION_REFERENCE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set the private vision reference directory"))
}

fn read(path: impl AsRef<std::path::Path>) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|_| panic!("unable to read private vision reference artifact"))
}

fn metadata(path: impl AsRef<std::path::Path>) -> Value {
    serde_json::from_slice(&read(path)).unwrap_or_else(|_| panic!("invalid private vision reference metadata"))
}

fn bf16_values(bytes: &[u8]) -> Vec<bf16> {
    assert!(bytes.len().is_multiple_of(2), "invalid private BF16 artifact length");
    bytes.chunks_exact(2).map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]]))).collect()
}

fn float_values(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4), "invalid private FP32 artifact length");
    bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect()
}

fn grid(value: &Value) -> [u32; 3] {
    serde_json::from_value(value.clone()).unwrap_or_else(|_| panic!("invalid private vision grid"))
}

fn tower_config(model: &std::path::Path) -> VisionTowerConfig {
    let config = metadata(model.join("config.json"));
    let vision = &config["artifact_contract"]["vision_config"];
    let integer = |key: &str| {
        vision[key]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or_else(|| panic!("invalid private tower configuration"))
    };
    VisionTowerConfig {
        depth: integer("depth"),
        hidden_size: integer("hidden_size"),
        intermediate_size: integer("intermediate_size"),
        num_heads: integer("num_heads"),
        num_position_embeddings: integer("num_position_embeddings"),
        out_hidden_size: integer("out_hidden_size"),
        patch_size: integer("patch_size"),
        spatial_merge_size: integer("spatial_merge_size"),
        temporal_patch_size: integer("temporal_patch_size"),
        in_channels: integer("in_channels"),
        layer_norm_eps: 1e-6,
    }
}

fn stage_name(stage: VisionStage) -> String {
    match stage {
        VisionStage::PatchProjection => "patch_projection".into(),
        VisionStage::PositionAdded => "position_added".into(),
        VisionStage::Block(index) => format!("block_{index}"),
        VisionStage::Merger => "merger".into(),
    }
}

fn tower_reference<B: Backend>(backend: &str) {
    let reference = reference_root();
    let manifest = metadata(reference.join("reference-manifest.json"));
    let model = match manifest["model_directory"].as_str() {
        Some(directory) => reference.join(directory),
        None => std::env::var_os("STARK_TEST_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("set the private converted model directory")),
    };
    let config = tower_config(&model);
    let context = B::Context::new().unwrap_or_else(|_| panic!("native reference context failed"));
    let file = File::open(model.join("model.safetensors"))
        .unwrap_or_else(|_| panic!("unable to open private converted weights"));
    let loader = ParameterLoader::<B>::new(&file, &context)
        .unwrap_or_else(|_| panic!("private reference parameter indexing failed"));
    let tree = loader.tree().subtree("vision");
    let tower = VisionTower::<B>::new(&context, &config, &tree)
        .unwrap_or_else(|_| panic!("private reference tower loading failed"));
    tree.assert_all_tensors_validated().unwrap_or_else(|_| panic!("unvalidated private vision parameters"));
    let mut metrics = Vec::new();
    let mut within_budget = true;
    for definition in manifest["tower"].as_array().unwrap() {
        let name = definition["name"].as_str().unwrap();
        let folder = reference.join(name);
        let case = metadata(folder.join("metadata.json"));
        let expected_patches = bf16_values(&read(folder.join(case["patches"]["file"].as_str().unwrap())));
        let patches = if let Some(rgb) = definition.get("source_rgb") {
            let limits = PreprocessLimits::default();
            let image = RgbImage::try_new(
                rgb["width"].as_u64().unwrap() as u32,
                rgb["height"].as_u64().unwrap() as u32,
                read(folder.join(rgb["file"].as_str().unwrap())),
                &limits,
            )
            .unwrap_or_else(|_| panic!("private tower RGB source admission failed"));
            let prepared =
                prepare_image(&image, &limits).unwrap_or_else(|_| panic!("private tower RGB preprocessing failed"));
            assert_eq!(prepared.grid, grid(&case["grid"]), "private tower RGB grid differs from reference");
            assert_eq!(prepared.patches.len(), expected_patches.len(), "private tower RGB patch shape mismatch");
            assert!(
                prepared
                    .patches
                    .iter()
                    .zip(&expected_patches)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                "private tower RGB prepared patches differ from reference"
            );
            prepared.patches.into_vec()
        } else {
            expected_patches
        };
        let mut commands = context
            .create_command_buffer(None, None)
            .unwrap_or_else(|_| panic!("native reference command setup failed"));
        let mut readbacks = Vec::new();
        let output = tower
            .encode_with_observer(&patches, grid(&case["grid"]), &mut commands, |stage, values, commands| {
                let mut readback = context.create_buffer(values.size())?;
                commands.encode_copy(values, &mut readback);
                readbacks.push((stage_name(stage), readback));
                Ok(())
            })
            .unwrap_or_else(|_| panic!("native reference tower encoding failed"));
        drop(output);
        commands
            .end_encoding()
            .submit()
            .wait_until_completed()
            .unwrap_or_else(|_| panic!("native reference tower completion failed"));
        assert_eq!(readbacks.len(), config.depth as usize + 3, "missing private reference stage");
        for (stage, readback) in readbacks {
            let expected = float_values(&read(folder.join(case["stages"][&stage]["file"].as_str().unwrap())));
            let actual = (&readback).as_slice::<bf16>();
            assert_eq!(actual.len(), expected.len(), "private reference stage shape mismatch");
            let mut squared_error = 0.0f64;
            let mut squared_reference = 0.0f64;
            let mut squared_actual = 0.0f64;
            let mut dot = 0.0f64;
            let mut maximum = 0.0f64;
            for (actual, expected) in actual.iter().zip(&expected) {
                let actual = f64::from(actual.to_f32());
                let expected = f64::from(*expected);
                assert!(actual.is_finite() && expected.is_finite(), "nonfinite private reference stage");
                let error = actual - expected;
                maximum = maximum.max(error.abs());
                squared_error += error * error;
                squared_reference += expected * expected;
                squared_actual += actual * actual;
                dot += actual * expected;
            }
            let relative_rms = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
            let cosine = dot / (squared_actual * squared_reference).sqrt().max(f64::MIN_POSITIVE);
            // Fixed BF16 replay targets. These are validation budgets for the
            // synthetic fixtures, not claims about every image or context.
            let relative_rms_budget = if stage == "patch_projection" {
                0.005
            } else {
                0.02
            };
            let passed = relative_rms <= relative_rms_budget && cosine >= 0.999;
            within_budget &= passed;
            let metric = json!({
                "backend":backend, "fixture":name, "stage":stage, "elements":actual.len(),
                "attention_reference":case["attention_reference"].as_str().unwrap_or("pinned-eager"),
                "maximum_absolute_error":maximum, "relative_rms_error":relative_rms,
                "cosine_similarity":cosine, "relative_rms_budget":relative_rms_budget, "passed":passed,
            });
            println!("{metric}");
            metrics.push(metric);
        }
    }
    std::fs::write(
        reference.join(format!("native-tower-{backend}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("unable to save private aggregate vision metrics"));
    assert!(within_budget, "private native vision reference replay exceeded numeric budget");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore] // Requires private reference artifacts and the official BF16 model.
fn qwen35_vision_private_reference_cpu() {
    tower_reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore] // Requires private reference artifacts and the official BF16 model.
fn qwen35_vision_private_reference_metal() {
    tower_reference::<crate::backends::metal::Metal>("metal");
}

#[stark_test]
#[ignore] // Requires private preprocessing reference artifacts.
fn qwen35_preprocessing_private_reference() {
    let reference = reference_root();
    let manifest = metadata(reference.join("reference-manifest.json"));
    let limits = PreprocessLimits::default();
    let mut checked = 0usize;
    let mut exact = true;
    for case in manifest["preprocessing"].as_array().unwrap() {
        let folder = reference.join(case["name"].as_str().unwrap());
        let width = case["width"].as_u64().unwrap() as u32;
        let height = case["height"].as_u64().unwrap() as u32;
        let source = RgbImage::try_new(width, height, read(folder.join("rgb.u8")), &limits)
            .unwrap_or_else(|_| panic!("private reference source admission failed"));
        let prepared = prepare_image(&source, &limits);
        if !case["default_admitted"].as_bool().unwrap() {
            assert!(prepared.is_err(), "private reference exceeded processed image budget");
            continue;
        }
        let prepared = prepared.unwrap_or_else(|_| panic!("private reference preprocessing failed"));
        assert_eq!(prepared.grid, grid(&case["grid"]), "private reference processed grid mismatch");
        assert_eq!(prepared.resized_width, case["resized_width"].as_u64().unwrap() as u32);
        assert_eq!(prepared.resized_height, case["resized_height"].as_u64().unwrap() as u32);
        let expected = bf16_values(&read(folder.join(case["patches"]["file"].as_str().unwrap())));
        assert_eq!(prepared.patches.len(), expected.len(), "private reference patch shape mismatch");
        let mismatches = prepared.patches.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("{}", json!({"fixture":case["name"],"elements":expected.len(),"bf16_mismatches":mismatches}));
        exact &= mismatches == 0;
        checked += 1;
    }
    assert_eq!(checked, 5, "missing private reference preprocessing cases");
    for case in manifest["resize"].as_array().unwrap() {
        let folder = reference.join(case["name"].as_str().unwrap());
        let source = RgbImage::try_new(
            case["width"].as_u64().unwrap() as u32,
            case["height"].as_u64().unwrap() as u32,
            read(folder.join("rgb.u8")),
            &limits,
        )
        .unwrap_or_else(|_| panic!("private resize source admission failed"));
        let resized = resize_rgb_for_reference(
            &source,
            case["resized_width"].as_u64().unwrap() as u32,
            case["resized_height"].as_u64().unwrap() as u32,
        )
        .unwrap_or_else(|_| panic!("private reference resize failed"));
        let expected = read(folder.join("resized.u8"));
        assert_eq!(resized.len(), expected.len(), "private reference resize shape mismatch");
        let mismatches = resized.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let maximum = resized.iter().zip(&expected).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        println!(
            "{}",
            json!({"fixture":case["name"],"elements":expected.len(),"uint8_mismatches":mismatches,"maximum_absolute_error":maximum})
        );
        exact &= mismatches == 0;
    }
    assert!(exact, "private reference preprocessing differs from pinned equations");
}

#[stark_test]
#[ignore] // Requires private Pillow decoder reference artifacts.
fn qwen35_decoder_private_reference() {
    let reference = reference_root();
    let manifest = metadata(reference.join("decode-reference-manifest.json"));
    let limits = PreprocessLimits::default();
    let mut within_budget = true;
    let mut metrics = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let encoded = read(reference.join(case["encoded"].as_str().unwrap()));
        let decoded = decode_image(&encoded, &limits);
        if !case["admitted"].as_bool().unwrap() {
            assert!(decoded.is_err(), "private decoder reference admitted an unsupported frame layout");
            continue;
        }
        let decoded = decoded.unwrap_or_else(|_| panic!("private decoder reference failed"));
        assert_eq!(decoded.width(), case["width"].as_u64().unwrap() as u32);
        assert_eq!(decoded.height(), case["height"].as_u64().unwrap() as u32);
        let actual = decoded.pixels_for_reference();
        let expected = read(reference.join(case["rgb"].as_str().unwrap()));
        assert_eq!(actual.len(), expected.len(), "private decoder reference shape mismatch");
        let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let maximum = actual.iter().zip(&expected).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        let squared_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| {
                let error = f64::from(*a) - f64::from(*b);
                error * error
            })
            .sum::<f64>();
        let rms = (squared_error / actual.len() as f64).sqrt();
        let lossless = case["format"] == "PNG";
        let passed = if lossless {
            mismatches == 0
        } else {
            // Cross-codec JPEG budget is intentionally distinct from exact
            // lossless PNG/color/orientation agreement.
            maximum <= 3 && rms <= 0.6
        };
        within_budget &= passed;
        let metric = json!({
            "fixture":case["name"],"format":case["format"],"elements":actual.len(),
            "uint8_mismatches":mismatches,"maximum_absolute_error":maximum,"rms_error":rms,"passed":passed,
        });
        println!("{metric}");
        metrics.push(metric);
    }
    std::fs::write(reference.join("native-decoder-metrics.json"), serde_json::to_vec_pretty(&metrics).unwrap())
        .unwrap_or_else(|_| panic!("unable to save private decoder aggregate metrics"));
    assert!(within_budget, "private native decoder reference replay exceeded numeric budget");
}

fn block_stage_name(stage: VisionBlockStage) -> &'static str {
    match stage {
        VisionBlockStage::Norm1 => "norm1",
        VisionBlockStage::Qkv => "qkv",
        VisionBlockStage::Queries => "queries",
        VisionBlockStage::Keys => "keys",
        VisionBlockStage::Values => "values",
        VisionBlockStage::Attention => "attention",
        VisionBlockStage::Projection => "projection",
        VisionBlockStage::AttentionResidual => "attention_residual",
        VisionBlockStage::Norm2 => "norm2",
        VisionBlockStage::Fc1 => "fc1",
        VisionBlockStage::Gelu => "gelu",
        VisionBlockStage::Fc2 => "fc2",
        VisionBlockStage::Output => "output",
    }
}

/// A diagnostic replay, not a replacement qualification budget. It records all
/// first-block boundaries so a failure can be traced to its first operation.
fn first_block_reference<B: Backend>(backend: &str) {
    let reference = reference_root().join("first_block");
    let model = std::env::var_os("STARK_TEST_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set the private converted model directory"));
    let config = tower_config(&model);
    let context = B::Context::new().unwrap_or_else(|_| panic!("native block diagnostic context failed"));
    let file = File::open(model.join("model.safetensors"))
        .unwrap_or_else(|_| panic!("unable to open private converted weights"));
    let loader = ParameterLoader::<B>::new(&file, &context)
        .unwrap_or_else(|_| panic!("private block parameter indexing failed"));
    let tree = loader.tree().subtree("vision");
    let tower = VisionTower::<B>::new(&context, &config, &tree)
        .unwrap_or_else(|_| panic!("private block tower loading failed"));
    let mut metrics = Vec::new();
    for name in ["tower_32x32", "tower_256x256"] {
        let folder = reference.join(name);
        let case = metadata(folder.join("metadata.json"));
        let patches = bf16_values(&read(folder.join(case["patches"]["file"].as_str().unwrap())));
        let mut commands =
            context.create_command_buffer(None, None).unwrap_or_else(|_| panic!("native block command setup failed"));
        let mut readbacks = Vec::new();
        let output = tower
            .encode_first_block_with_observer(
                &patches,
                grid(&case["grid"]),
                &mut commands,
                |stage, values, commands| {
                    let mut readback = context.create_buffer(values.size())?;
                    commands.encode_copy(values, &mut readback);
                    readbacks.push((block_stage_name(stage), readback));
                    Ok(())
                },
            )
            .unwrap_or_else(|_| panic!("native first block encoding failed"));
        drop(output);
        commands
            .end_encoding()
            .submit()
            .wait_until_completed()
            .unwrap_or_else(|_| panic!("native first block completion failed"));
        assert_eq!(readbacks.len(), 13, "missing private first block boundary");
        let native_folder = reference.join(format!("native-{backend}")).join(name);
        std::fs::create_dir_all(&native_folder)
            .unwrap_or_else(|_| panic!("unable to create private first block output directory"));
        for (stage, readback) in readbacks {
            let expected = float_values(&read(folder.join(case["stages"][stage]["file"].as_str().unwrap())));
            let actual = (&readback).as_slice::<bf16>();
            assert_eq!(actual.len(), expected.len(), "private first block boundary shape mismatch");
            let mut squared_error = 0.0f64;
            let mut squared_reference = 0.0f64;
            let mut squared_actual = 0.0f64;
            let mut dot = 0.0f64;
            let mut maximum = 0.0f64;
            let mut mismatches = 0usize;
            let mut raw = Vec::with_capacity(actual.len() * 4);
            for (actual, expected) in actual.iter().zip(&expected) {
                let actual = actual.to_f32();
                assert!(actual.is_finite() && expected.is_finite(), "nonfinite first block boundary");
                raw.extend_from_slice(&actual.to_le_bytes());
                mismatches += usize::from(actual.to_bits() != expected.to_bits());
                let (actual, expected) = (f64::from(actual), f64::from(*expected));
                let error = actual - expected;
                maximum = maximum.max(error.abs());
                squared_error += error * error;
                squared_reference += expected * expected;
                squared_actual += actual * actual;
                dot += actual * expected;
            }
            std::fs::write(native_folder.join(format!("{stage}.f32")), raw)
                .unwrap_or_else(|_| panic!("unable to save private first block boundary"));
            let relative_rms = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
            let cosine = dot / (squared_actual * squared_reference).sqrt().max(f64::MIN_POSITIVE);
            let metric = json!({
                "backend":backend, "fixture":name, "stage":stage, "elements":actual.len(),
                "bf16_mismatches":mismatches,"maximum_absolute_error":maximum,
                "relative_rms_error":relative_rms,"cosine_similarity":cosine,
                "attention_reference":"pinned-eager", "diagnostic_only":true,
            });
            println!("{metric}");
            metrics.push(metric);
        }
    }
    std::fs::write(
        reference.join(format!("native-first-block-{backend}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("unable to save private first block metrics"));
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore] // Requires private first-block reference artifacts and official weights.
fn qwen35_vision_first_block_reference_cpu() {
    first_block_reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore] // Requires private first-block reference artifacts and official weights.
fn qwen35_vision_first_block_reference_metal() {
    first_block_reference::<crate::backends::metal::Metal>("metal");
}
