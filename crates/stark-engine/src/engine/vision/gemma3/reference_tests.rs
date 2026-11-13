//! Explicit opt-in full-checkpoint reference replay. Fixture contents and
//! native vectors stay private; only aggregate metrics and hashes are emitted.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;

fn read(
    root: &Path,
    name: &str,
) -> Vec<u8> {
    assert!(
        name.len() < 128 && !name.contains('/') && !name.contains('\\'),
        "invalid private image reference filename"
    );
    std::fs::read(root.join(name)).unwrap_or_else(|_| panic!("private image reference unavailable"))
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4), "private reference size differs from FP32 layout");
    bytes.chunks_exact(4).map(|row| f32::from_le_bytes(row.try_into().unwrap())).collect()
}

fn compare(
    actual: &[bf16],
    expected: &[f32],
) -> Value {
    assert!(actual.len() == expected.len() && !actual.is_empty(), "private reference shape mismatch");
    let (mut sq_error, mut sq_ref, mut sq_native, mut dot, mut max, mut mean_error) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut actual_hash = Sha256::new();
    let mut reference_hash = Sha256::new();
    let mut greedy_native = 0;
    let mut greedy_ref = 0;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let a = a.to_f32();
        assert!(a.is_finite() && b.is_finite(), "nonfinite private reference output");
        let (x, y) = (f64::from(a), f64::from(b));
        let error = x - y;
        sq_error += error * error;
        sq_ref += y * y;
        sq_native += x * x;
        dot += x * y;
        max = max.max(error.abs());
        mean_error += error;
        actual_hash.update(a.to_le_bytes());
        reference_hash.update(b.to_le_bytes());
        if a > actual[greedy_native].to_f32() {
            greedy_native = index;
        }
        if b > expected[greedy_ref] {
            greedy_ref = index;
        }
    }
    let n = actual.len() as f64;
    mean_error /= n;
    let rmse = (sq_error / n).sqrt();
    let centered_rmse = (sq_error / n - mean_error * mean_error).max(0.0).sqrt();
    let ref_mean = expected.iter().map(|&value| f64::from(value)).sum::<f64>() / n;
    let ref_variance = (sq_ref / n - ref_mean * ref_mean).max(f64::MIN_POSITIVE);
    let centered_max = actual
        .iter()
        .zip(expected)
        .map(|(a, &b)| (f64::from(a.to_f32()) - f64::from(b) - mean_error).abs())
        .fold(0.0f64, f64::max);
    let partition = |values: &[f32]| {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        max + values.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>().ln()
    };
    let native = actual.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    let rp = partition(expected);
    let ap = partition(&native);
    let kl = expected
        .iter()
        .zip(&native)
        .map(|(&r, &a)| {
            let p = f64::from(r) - rp;
            p.exp() * (p - (f64::from(a) - ap))
        })
        .sum::<f64>()
        .max(0.0);
    json!({"elements":actual.len(),"maximum_absolute_error":max,"rmse":rmse,
        "relative_rms_error":(sq_error/sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
        "centered_relative_rms_error":centered_rmse/ref_variance.sqrt(),
        "centered_rmse":centered_rmse,"reference_centered_rms":ref_variance.sqrt(),
        "centered_maximum_absolute_error":centered_max,
        "cosine_similarity":dot/(sq_ref*sq_native).sqrt().max(f64::MIN_POSITIVE),
        "mean_error":mean_error,"reference_to_native_kl":kl,"greedy_agreement":greedy_native==greedy_ref,
        "native_sha256":format!("{:x}",actual_hash.finalize()),"reference_sha256":format!("{:x}",reference_hash.finalize())})
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma3_image_full_reference_metal() {
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA3_FIXTURE_PATH").expect("set private Gemma 3 fixture path"));
    let model_path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private Gemma 3 model path"));
    let manifest: Value =
        serde_json::from_slice(&read(&root, "manifest.json")).expect("invalid private Gemma 3 manifest");
    assert!(
        manifest["source_repo"] == "google/gemma-3-4b-it"
            && manifest["source_revision"] == "093f9f388b31de276ce2de164bdc2081324b9767"
            && manifest["reference_profile"] == "transformers-canonical-eager-bf16"
            && manifest["torch"] == "2.14.1"
            && manifest["transformers"] == "5.15.1",
        "private Gemma 3 reference identity changed"
    );
    assert!(
        manifest["reference_hashes"]
            == json!({
                "modeling_gemma3.py":"0d8d5dff8e72b1207e54edce0bd2b22a34b3194edfa114996b04cef1acf4ef6a",
                "modeling_siglip.py":"0a0d2b55f4fcb977b74549540211e42a3c3f41d0d446dc474dcc244a925791aa"
            }),
        "private Gemma 3 reference implementation changed"
    );
    let engine = Engine::<crate::backends::metal::Metal>::new()
        .unwrap_or_else(|_| panic!("private Gemma 3 engine setup failed"));
    let mut model = engine
        .load_gemma3_vision_language_model(&model_path)
        .unwrap_or_else(|_| panic!("private Gemma 3 model loading failed"));
    let mut reports = Vec::new();
    let mut passed = true;
    let features_only = std::env::var_os("STARK_GEMMA3_FEATURE_ONLY").is_some();
    for (index, case) in manifest["cases"].as_array().expect("missing private Gemma 3 cases").iter().enumerate() {
        for (name, hash) in case["hashes"].as_object().expect("missing private fixture hashes") {
            assert!(
                format!("{:x}", Sha256::digest(read(&root, name))) == hash.as_str().unwrap(),
                "private Gemma 3 fixture changed"
            );
        }
        let encoded = read(&root, case["image"].as_str().unwrap());
        let prepared =
            model.prepare_image(&encoded).unwrap_or_else(|_| panic!("private Gemma 3 image preparation failed"));
        let patch_metrics =
            compare(&prepared.patches.patches, &floats(&read(&root, case["patches"].as_str().unwrap())));
        let patch_exact = patch_metrics["maximum_absolute_error"] == 0.0;
        let native_features = model
            .project(&prepared, &mut || false)
            .unwrap_or_else(|_| panic!("private Gemma 3 vision projection failed"));
        let feature_metrics = compare(&native_features, &floats(&read(&root, case["features"].as_str().unwrap())));
        let feature_pass = feature_metrics["relative_rms_error"].as_f64().unwrap() <= 0.02
            && feature_metrics["cosine_similarity"].as_f64().unwrap() >= 0.999;
        if features_only {
            passed &= patch_exact && feature_pass;
            reports.push(json!({"case_index":index,"patch_exact":patch_exact,"features_passed":feature_pass,
                "patches":patch_metrics,"features":feature_metrics}));
            continue;
        }
        let tokens: Value = serde_json::from_slice(&read(&root, case["tokens"].as_str().unwrap())).unwrap();
        let raw: Vec<u64> = serde_json::from_value(tokens["raw"].clone()).unwrap();
        let expected: Vec<u64> = serde_json::from_value(tokens["expanded"].clone()).unwrap();
        let generated: Vec<u64> = serde_json::from_value(tokens["generated"].clone()).unwrap();
        let capacity = 1024;
        let (expanded, start) = model
            .prepare_prompt(raw.into_boxed_slice(), &prepared, capacity)
            .unwrap_or_else(|_| panic!("private Gemma 3 prompt preparation failed"));
        assert!(expanded.as_ref() == expected.as_slice(), "private Gemma 3 expanded prompt differs from HF processor");
        let mut input = model.preflight(expanded, &prepared, start, capacity).unwrap();
        input.image.features = native_features;
        let mut state = model.create_empty_state(Some(capacity), 0).unwrap();
        state.vision_prefill_logits_readback =
            Some(engine.context.create_buffer(VOCAB as usize * size_of::<bf16>()).unwrap());
        let mut options = model.default_stream_options();
        options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
        let native_generated: Vec<u64> = {
            let mut stream = model
                .language_model
                .stream_multimodal(&input, &mut state, options)
                .unwrap_or_else(|_| panic!("private Gemma 3 image prefill failed"));
            (0..generated.len())
                .map(|_| stream.next().unwrap().unwrap_or_else(|_| panic!("private Gemma 3 decode failed")))
                .collect()
        };
        let reference_logits = floats(&read(&root, case["logits"].as_str().unwrap()));
        assert!(reference_logits.len() == generated.len() * VOCAB as usize, "private Gemma 3 logits shape changed");
        let logits = state.vision_prefill_logits_readback.take().unwrap();
        let logit_metrics = compare((&logits).as_slice::<bf16>(), &reference_logits[..VOCAB as usize]);
        let trajectory = native_generated == generated;
        let logit_pass = logit_metrics["centered_relative_rms_error"].as_f64().unwrap() <= 0.1
            && logit_metrics["reference_to_native_kl"].as_f64().unwrap() <= 0.03
            && logit_metrics["greedy_agreement"] == true;
        passed &= patch_exact && feature_pass && logit_pass && trajectory;
        reports.push(json!({"case_index":index,"patch_exact":patch_exact,"features_passed":feature_pass,
            "logits_passed":logit_pass,"greedy_trajectory_agreement":trajectory,"greedy_steps":generated.len(),
            "patches":patch_metrics,"features":feature_metrics,"logits":logit_metrics}));
    }
    let report = json!({"passed":passed,"cases":reports,"backend":"metal","gate_profile":"gemma3-hf-bf16-image-v1",
        "scope":if features_only {"image-features-only"} else {"image-features-logits-greedy-trajectory"},
        "reference_hashes":manifest["reference_hashes"],"artifact_sha256":model.artifact_identity().artifact_sha256});
    std::fs::write(
        root.join(if features_only {
            "native-feature-metrics.json"
        } else {
            "native-metrics.json"
        }),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
    assert!(passed, "private Gemma 3 full reference gates failed");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma3_completed_tower_cancel_keeps_model_usable_metal() {
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA3_FIXTURE_PATH").expect("set private Gemma 3 fixture path"));
    let model_path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private Gemma 3 model path"));
    let engine =
        Engine::<crate::backends::metal::Metal>::new().unwrap_or_else(|_| panic!("private cancellation engine failed"));
    let mut model = engine
        .load_gemma3_vision_language_model(&model_path)
        .unwrap_or_else(|_| panic!("private cancellation model failed"));
    let image =
        model.prepare_image(&read(&root, "red.png")).unwrap_or_else(|_| panic!("private cancellation image failed"));
    let mut checks = 0;
    let result = model.project(&image, &mut || {
        checks += 1;
        checks == 2
    });
    assert!(
        result.is_err_and(|error| error.kind() == super::super::NativeVisionErrorKind::Cancelled),
        "completed-boundary cancellation did not preserve its category"
    );
    assert!(!model.is_fenced(), "completed-boundary cancellation fenced immutable weights");
    let mut state = model.create_empty_state(Some(1024), 0).unwrap_or_else(|_| panic!("private recovery state failed"));
    let text = model
        .tokenizer()
        .encode("<bos><start_of_turn>user\nSay hello.<end_of_turn>\n<start_of_turn>model\n", false)
        .unwrap_or_else(|_| panic!("private recovery tokenization failed"));
    let text: Vec<u64> = text.get_ids().iter().copied().map(u64::from).collect();
    let mut options = model.default_stream_options();
    options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
    {
        let mut stream =
            model.stream(&text, &mut state, options).unwrap_or_else(|_| panic!("private text recovery failed"));
        assert!(stream.next().unwrap().is_ok(), "private recovery generation failed");
    }
    assert!(!model.is_fenced(), "completed cancellation recovery fenced model");
    let report = json!({"passed":true,"cancelled_after_completed_block":1,"same_instance_text_recovery":true,
        "artifact_sha256":model.artifact_identity().artifact_sha256});
    std::fs::write(root.join("native-cancellation-metrics.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma3_first_block_stage_reference_metal() {
    type B = crate::backends::metal::Metal;
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA3_STAGE_FIXTURE_PATH").expect("set private stage fixture"));
    let model_path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private Gemma 3 artifact"));
    let manifest: Value = serde_json::from_slice(&read(&root, "manifest.json")).unwrap();
    assert_eq!(manifest["source_revision"], "093f9f388b31de276ce2de164bdc2081324b9767");
    assert_eq!(manifest["retained_rows"], 64);
    let engine = Engine::<B>::new().unwrap();
    let mut model = engine.load_gemma3_vision_language_model(&model_path).unwrap();
    let image_root = PathBuf::from(std::env::var_os("STARK_GEMMA3_FIXTURE_PATH").expect("set private image fixture"));
    let image_bytes = read(&image_root, "pattern.png");
    assert_eq!(format!("{:x}", Sha256::digest(&image_bytes)), manifest["image_sha256"].as_str().unwrap());
    let image = model.prepare_image(&image_bytes).unwrap();
    let mut captures: Vec<(String, <B as Backend>::GlobalBuffer)> = Vec::new();
    let mut observe =
        |name: &str,
         buffer: &<B as Backend>::ScratchBuffer,
         commands: &mut <<B as Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding| {
            let stage = manifest["stages"].as_array().unwrap().iter().find(|entry| entry["name"] == name).unwrap();
            let count = stage["elements"].as_u64().unwrap() as usize;
            assert!(count > 0 && count <= 1_048_576, "private stage exceeds bounded prefix");
            let mut readback = engine.context.create_buffer(count * size_of::<bf16>())?;
            commands.encode_copy(buffer.subrange(0..count * size_of::<bf16>()), &mut readback);
            captures.push((name.to_owned(), readback));
            Ok(())
        };
    let mut hidden = engine.context.create_buffer(4096 * 1152 * size_of::<bf16>()).unwrap();
    let mut commands = engine.context.create_command_buffer(Some("Gemma 3 private initial stages"), None).unwrap();
    let initial =
        model.tower.encode_initial_with_observer(&image.patches.patches, &mut commands, &mut observe).unwrap();
    commands.encode_copy(&initial, &mut hidden);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let mut commands = engine.context.create_command_buffer(Some("Gemma 3 private first block"), None).unwrap();
    model.tower.encode_block_with_observer(0, &hidden, &mut commands, &mut observe).unwrap();
    commands.end_encoding().submit().wait_until_completed().unwrap();
    drop(observe);
    let reports = captures
        .into_iter()
        .map(|(name, buffer)| {
            let stage = manifest["stages"].as_array().unwrap().iter().find(|entry| entry["name"] == name).unwrap();
            let bytes = read(&root, stage["filename"].as_str().unwrap());
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), stage["sha256"].as_str().unwrap());
            json!({"stage":name,"metrics":compare((&buffer).as_slice::<bf16>(), &floats(&bytes))})
        })
        .collect::<Vec<_>>();
    assert_eq!(reports.len(), 11, "private stage observer coverage changed");
    let report = json!({"scope":"diagnostic-first-block-prefix-only","stages":reports,
        "artifact_sha256":model.artifact_identity().artifact_sha256,
        "reference_manifest_sha256":format!("{:x}",Sha256::digest(read(&root,"manifest.json")))});
    std::fs::write(root.join("native-stage-metrics.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}
