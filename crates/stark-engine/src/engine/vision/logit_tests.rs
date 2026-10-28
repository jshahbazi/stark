//! Private full-model last-prompt-logit qualification. No production capture API.

use std::path::{Path, PathBuf};

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;

fn checked_bytes(
    root: &Path,
    definition: &Value,
) -> Vec<u8> {
    let path = definition["file"].as_str().expect("invalid private logit reference name");
    let data = std::fs::read(root.join(path)).unwrap_or_else(|_| panic!("private logit reference unavailable"));
    let hash = format!("{:x}", Sha256::digest(&data));
    assert!(definition["sha256"].as_str() == Some(hash.as_str()), "private logit reference hash mismatch");
    data
}

fn u32_values(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "invalid private token/position byte count");
    bytes.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

fn reference<B: Backend>(backend: &str) {
    let path = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private exact artifact path"));
    let root = PathBuf::from(
        std::env::var_os("STARK_VISION_LOGIT_REFERENCE_PATH").expect("set private full-model logit reference path"),
    );
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(root.join("manifest.json")).unwrap_or_else(|_| panic!("private logit manifest unavailable")),
    )
    .unwrap_or_else(|_| panic!("invalid private logit manifest"));
    assert!(manifest["threads"].as_u64() == Some(8), "unexpected private reference thread count");
    assert!(manifest["vocab_size"].as_u64() == Some(248320), "unexpected private logit vocabulary");
    let capacity = manifest["context_capacity"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .expect("invalid private logit context capacity");
    let engine = Engine::<B>::new().unwrap_or_else(|_| panic!("private logit engine setup failed"));
    let mut model = engine
        .load_vision_language_model(&path)
        .unwrap_or_else(|_| panic!("private logit exact artifact loading failed"));
    let output = root.join(format!("native-{backend}"));
    std::fs::create_dir_all(&output).unwrap_or_else(|_| panic!("private logit output setup failed"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut metrics = Vec::new();
    for (index, case) in manifest["cases"].as_array().expect("invalid private logit cases").iter().enumerate() {
        let folder = root.join(case["name"].as_str().expect("invalid private logit case"));
        let image_bytes = checked_bytes(&folder, &case["image"]);
        let unexpanded = u32_values(&checked_bytes(&folder, &case["files"]["unexpanded_tokens"]));
        let expected_tokens = u32_values(&checked_bytes(&folder, &case["files"]["tokens"]));
        let expected_positions = u32_values(&checked_bytes(&folder, &case["files"]["positions"]));
        let expected_logits = checked_bytes(&folder, &case["files"]["logits"]);
        assert!(expected_logits.len() == 248320 * 4, "incomplete private full-vocabulary reference");
        let image = model.prepare_image(&image_bytes).unwrap_or_else(|_| panic!("private image preparation failed"));
        assert!(
            u64::from(image.image_token_count()) == case["image_tokens"].as_u64().unwrap(),
            "private image token count differs"
        );
        for (axis, size) in image.grid().iter().enumerate() {
            assert!(u64::from(*size) == case["grid"][axis].as_u64().unwrap(), "private processed grid differs");
        }
        let (tokens, start) = model
            .prepare_prompt(unexpanded.into_iter().map(u64::from).collect(), &image, capacity)
            .unwrap_or_else(|_| panic!("private image prompt preparation failed"));
        assert!(
            tokens.len() == expected_tokens.len()
                && tokens.iter().zip(&expected_tokens).all(|(a, e)| *a == u64::from(*e)),
            "private expanded image prompt differs"
        );
        assert!(start as u64 == case["token_start"].as_u64().unwrap(), "private image span differs");
        let input = model
            .preflight_prepared_prefill(tokens.clone(), &image, start, capacity)
            .unwrap_or_else(|_| panic!("private image preflight failed"));
        assert!(
            input.positions.len() * 3 == expected_positions.len()
                && input.positions.iter().flatten().copied().eq(expected_positions),
            "private full-model M-RoPE positions differ"
        );
        assert!(
            input.position_delta == case["position_delta"].as_i64().unwrap(),
            "private full-model M-RoPE delta differs"
        );
        drop(input);
        let mut state =
            model.create_empty_state(Some(capacity), 0).unwrap_or_else(|_| panic!("private logit state setup failed"));
        state.vision_prefill_logits_readback = Some(
            engine
                .context
                .create_buffer(248320 * size_of::<bf16>())
                .unwrap_or_else(|_| panic!("private logit readback allocation failed")),
        );
        let input_len = tokens.len();
        if std::env::var_os("STARK_VISION_DECODER_LAYER_TRACE").is_some() {
            // Exact pinned model width/layer count. These buffers never leave
            // this ignored test and are read only after stream Drop drains.
            let row_bytes = 1024 * size_of::<bf16>();
            let buffers =
                (0..24).map(|_| engine.context.create_buffer(input_len * row_bytes).unwrap()).collect::<Box<[_]>>();
            state.vision_prefill_layer_readbacks = Some((row_bytes, buffers));
        }
        let mut options = model.default_stream_options();
        options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
        {
            let stream = model
                .stream_prepared_image(tokens, image, start, capacity, &mut state, options)
                .unwrap_or_else(|_| panic!("private image prefill failed"));
            assert!(stream.metrics().num_tokens_prefilled == input_len, "private prefill count differs");
            assert!(stream.metrics().num_decode_forward_passes == 0, "private logit capture entered decode");
            drop(stream);
        }
        assert!(state.tokens().len() == input_len, "private logit capture consumed an extra token");
        assert!(!model.is_fenced(), "private image prefill fenced the instance");
        if let Some((_, readbacks)) = state.vision_prefill_layer_readbacks.take() {
            for (layer, readback) in readbacks.iter().enumerate() {
                let values = readback.as_slice::<bf16>();
                let raw = values.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect::<Vec<_>>();
                std::fs::write(output.join(format!("case_{index}_layer_{layer}.bf16")), raw)
                    .unwrap_or_else(|_| panic!("private layer trace write failed"));
            }
        }
        let readback = state.vision_prefill_logits_readback.take().expect("private logit readback missing");
        let actual = (&readback).as_slice::<bf16>();
        let (mut squared_error, mut squared_actual, mut squared_expected, mut product) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut actual_top, mut expected_top, mut actual_max, mut expected_max) =
            (0usize, 0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (mut maximum_error, mut mismatches) = (0.0f64, 0usize);
        let mut raw = Vec::with_capacity(actual.len() * 4);
        for (token, (actual, expected)) in actual.iter().zip(expected_logits.chunks_exact(4)).enumerate() {
            let a = actual.to_f32();
            let e = f32::from_le_bytes(expected.try_into().unwrap());
            assert!(a.is_finite() && e.is_finite(), "nonfinite private last-prompt logits");
            if a > actual_max {
                actual_max = a;
                actual_top = token;
            }
            if e > expected_max {
                expected_max = e;
                expected_top = token;
            }
            let difference = f64::from(a) - f64::from(e);
            squared_error += difference * difference;
            squared_actual += f64::from(a) * f64::from(a);
            squared_expected += f64::from(e) * f64::from(e);
            product += f64::from(a) * f64::from(e);
            maximum_error = maximum_error.max(difference.abs());
            mismatches += usize::from(a.to_bits() != e.to_bits());
            raw.extend_from_slice(&a.to_le_bytes());
        }
        std::fs::write(output.join(format!("case_{index}_logits.f32")), raw)
            .unwrap_or_else(|_| panic!("private native logit write failed"));
        metrics.push(json!({"case_index":index,"backend":backend,"input_tokens":input_len,"vocabulary":actual.len(),"relative_rms_error":(squared_error/squared_expected).sqrt(),"cosine_similarity":product/(squared_actual*squared_expected).sqrt(),"top1_matches":actual_top==expected_top,"bf16_mismatches":mismatches,"maximum_absolute_error":maximum_error}));
    }
    println!("{}", json!({"last_prompt_logit_metrics":metrics}));
    std::fs::write(
        root.join(format!("native-logit-{backend}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("private logit aggregate write failed"));
    assert!(
        metrics.iter().all(|case| case["relative_rms_error"].as_f64().unwrap() <= 0.02
            && case["cosine_similarity"].as_f64().unwrap() >= 0.999
            && case["top1_matches"].as_bool() == Some(true)),
        "private last-prompt logit gate failed"
    );
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_last_prompt_logit_reference_cpu() {
    reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_last_prompt_logit_reference_metal() {
    reference::<crate::backends::metal::Metal>("metal");
}
