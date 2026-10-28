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
fn gemma4_image_full_reference_metal() {
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_FIXTURE_PATH").expect("set private Gemma fixture path"));
    let model_path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private Gemma model path"));
    let manifest: Value =
        serde_json::from_slice(&read(&root, "manifest.json")).expect("invalid private Gemma manifest");
    let canonical = manifest["reference_profile"] == "transformers-canonical";
    if canonical {
        assert!(
            manifest["reference_version"] == "5.15.1"
                && manifest["source_sha256"] == "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3",
            "private canonical source identity changed"
        );
    } else {
        assert!(
            manifest["reference_revision"] == "53ebd1647cb83549b482a031ea70ba9d9ea60a04",
            "private source revision changed"
        );
    }
    let features_only = std::env::var_os("STARK_GEMMA4_FEATURE_ONLY").is_some();
    let engine =
        Engine::<crate::backends::metal::Metal>::new().unwrap_or_else(|_| panic!("private image engine setup failed"));
    let mut model = engine
        .load_gemma4_vision_language_model(&model_path)
        .unwrap_or_else(|_| panic!("private Gemma loading failed"));
    let mut reports = Vec::new();
    let mut passed = true;
    for (index, case) in manifest["cases"].as_array().expect("private image cases missing").iter().enumerate() {
        let encoded = read(&root, case["image"].as_str().unwrap());
        let prepared = model.prepare_image(&encoded).unwrap_or_else(|_| panic!("private image preparation failed"));
        let expected_grid: [u32; 3] = serde_json::from_value(case["grid"].clone()).unwrap();
        assert!(prepared.grid() == expected_grid, "private image grid differs");
        let expected_patches = floats(&read(&root, case["patches"].as_str().unwrap()));
        let patch_metrics = compare(&prepared.patches.patches, &expected_patches);
        let patch_exact = patch_metrics["maximum_absolute_error"] == 0.0;
        let reference_features = floats(&read(&root, case["features"].as_str().unwrap()));
        let native_features = model.project(&prepared).unwrap_or_else(|_| panic!("private image projection failed"));
        let feature_metrics = compare(&native_features, &reference_features);
        let feature_pass = feature_metrics["relative_rms_error"].as_f64().unwrap() <= 0.02
            && feature_metrics["cosine_similarity"].as_f64().unwrap() >= 0.999;
        if features_only {
            passed &= patch_exact && feature_pass;
            reports
                .push(json!({"case_index":index,"image_tokens":prepared.image_token_count(),"patch_exact":patch_exact,
                "patches":patch_metrics,"features":feature_metrics,"features_passed":feature_pass}));
            continue;
        }
        let unexpanded: Vec<u64> = serde_json::from_value(case["unexpanded_tokens"].clone()).unwrap();
        let expected_tokens: Vec<u64> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let capacity = case["context_tokens"].as_u64().unwrap() as u32;
        let (tokens, start) = model
            .prepare_prompt(unexpanded.into_boxed_slice(), &prepared, capacity)
            .unwrap_or_else(|_| panic!("private Gemma prompt preparation failed"));
        assert!(
            tokens.as_ref() == expected_tokens && start == case["token_start"].as_u64().unwrap() as usize,
            "private expanded Gemma prompt differs"
        );
        let mut input = model.preflight(tokens, &prepared, start, capacity).unwrap();
        input.image.features = native_features;
        let mut state = model.create_empty_state(Some(capacity), 0).unwrap();
        state.vision_prefill_logits_readback =
            Some(engine.context.create_buffer(VOCAB as usize * size_of::<bf16>()).unwrap());
        let options = model.default_stream_options();
        {
            let stream = model
                .language_model
                .stream_multimodal(&input, &mut state, options)
                .unwrap_or_else(|_| panic!("private Gemma image prefill failed"));
            drop(stream);
        }
        let reference_logits = floats(&read(&root, case["logits"].as_str().unwrap()));
        let logits = state.vision_prefill_logits_readback.take().expect("private image logits missing");
        let native = (&logits).as_slice::<bf16>();
        let logit_metrics = compare(native, &reference_logits);
        // Match the independently calibrated Gemma canonical BF16 policy:
        // an absolute BF16 floor plus relative centered RMS, bounded common
        // offset, centered extrema, KL, and greedy agreement. Fixed feature
        // tolerances above are intentionally independent of this logit policy.
        let centered_rms = logit_metrics["centered_rmse"].as_f64().unwrap();
        let reference_rms = logit_metrics["reference_centered_rms"].as_f64().unwrap();
        let logit_pass = centered_rms <= 1.0 / 128.0 + 0.08 * reference_rms
            && centered_rms <= 0.15
            && logit_metrics["centered_maximum_absolute_error"].as_f64().unwrap() <= 0.75
            && logit_metrics["mean_error"].as_f64().unwrap().abs() <= 0.75
            && logit_metrics["reference_to_native_kl"].as_f64().unwrap() <= 0.01
            && logit_metrics["greedy_agreement"] == true;
        passed &= patch_exact && feature_pass && logit_pass;
        reports.push(json!({"case_index":index,"image_tokens":prepared.image_token_count(),"patch_exact":patch_exact,
            "patches":patch_metrics,"features":feature_metrics,"features_passed":feature_pass,
            "logits":logit_metrics,"logits_passed":logit_pass}));
    }
    let report = json!({"passed":passed,"cases":reports,"backend":"metal",
        "scope":if features_only {"image-features-only"} else {"image-features-and-last-logits"},
        "gate_profile":"gemma4-canonical-bf16-image-v2",
        "reference_profile":if canonical {"transformers-canonical"} else {"mlx-vlm"},
        "reference_revision":manifest["reference_revision"]});
    let output = if features_only {
        "native-feature-metrics.json"
    } else {
        "native-metrics-v2.json"
    };
    std::fs::write(root.join(output), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
    assert!(passed, "private Gemma image reference gates failed");
}
