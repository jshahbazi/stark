//! Opt-in, serialized same-artifact tower residency control. This does not
//! qualify a different prior text export or every model in the image family.

use std::{path::PathBuf, time::Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::VisionLanguageModel;
use crate::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            stream::{LanguageModelStream, SamplingMethod},
        },
    },
};

enum Loaded {
    Text(LanguageModel<Metal>),
    Vision(VisionLanguageModel<Metal>),
}

fn measure_stream(mut stream: LanguageModelStream<'_, Metal>) -> (String, usize, f64, f64) {
    let started = Instant::now();
    let first =
        stream.next().expect("private parity stream ended before output").expect("private parity stream failed");
    let first_seconds = started.elapsed().as_secs_f64();
    let mut tokens = vec![first];
    for token in stream.take(31) {
        tokens.push(token.expect("private parity stream failed"));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let mut digest = Sha256::new();
    for token in &tokens {
        digest.update(token.to_le_bytes());
    }
    (format!("{:x}", digest.finalize()), tokens.len(), first_seconds, elapsed)
}

fn run(
    model: &mut Loaded,
    input: &[u64],
) -> (String, usize, f64, f64) {
    match model {
        Loaded::Text(model) => {
            let mut state = model.create_empty_state(Some(4096), 42).expect("private parity state failed");
            let mut options = model.default_stream_options();
            options.sampling_method = SamplingMethod::Greedy;
            measure_stream(model.stream(input, &mut state, options).expect("private parity stream setup failed"))
        },
        Loaded::Vision(model) => {
            let mut state = model.create_empty_state(Some(4096), 42).expect("private parity state failed");
            let mut options = model.default_stream_options();
            options.sampling_method = SamplingMethod::Greedy;
            measure_stream(model.stream(input, &mut state, options).expect("private parity stream setup failed"))
        },
    }
}

#[stark_test]
#[ignore]
fn qwen_image_text_tower_residency_control() {
    let path = PathBuf::from(std::env::var_os("STARK_QWEN_IMAGE_PARITY_MODEL_PATH").expect("set private model path"));
    let output = PathBuf::from(std::env::var_os("STARK_QWEN_IMAGE_PARITY_OUTPUT").expect("set private result path"));
    let config = std::fs::read(path.join("config.json")).expect("private parity config unavailable");
    let mut records = Vec::new();
    let mut expected = None;
    let mut identity = None;
    let mut input = Vec::new();
    // ABBA loading order, one excluded warmup plus three measured fresh states
    // per load. No other GPU jobs may overlap this explicitly ignored test.
    for (load, vision) in [false, true, true, false].into_iter().enumerate() {
        let engine = Engine::<Metal>::new().expect("private parity engine failed");
        let started = Instant::now();
        let mut model = if vision {
            let model = engine.load_vision_language_model(&path).expect("private image artifact admission failed");
            assert!(!model.has_dense_mtp(), "this stream control excludes dense MTP worker dispatch");
            let admitted = model.artifact_identity();
            identity = Some(json!({"config_sha256":admitted.config_sha256,
                "artifact_sha256":admitted.artifact_sha256,"tokenizer_sha256":admitted.tokenizer_sha256,
                "processor_sha256":admitted.processor_sha256,"template_sha256":admitted.template_sha256}));
            Loaded::Vision(model)
        } else {
            let model = engine.load_vision_text_validation_model(&path).expect("private text control admission failed");
            assert!(!model.has_dense_mtp(), "this stream control excludes dense MTP worker dispatch");
            Loaded::Text(model)
        };
        let load_seconds = started.elapsed().as_secs_f64();
        if input.is_empty() {
            let tokenizer = match &model {
                Loaded::Text(model) => model.tokenizer(),
                Loaded::Vision(model) => model.tokenizer(),
            };
            let seed = tokenizer
                .encode("Describe a reliable way to organize a software project. ", false)
                .expect("private parity tokenization failed");
            assert!(!seed.get_ids().is_empty());
            input = seed.get_ids().iter().copied().cycle().take(192).map(u64::from).collect();
        }
        for replay in 0..4 {
            let (hash, count, first, elapsed) = run(&mut model, &input);
            if let Some((expected_hash, expected_count)) = &expected {
                assert!(hash == *expected_hash && count == *expected_count, "same-artifact parity mismatch");
            } else {
                expected = Some((hash.clone(), count));
            }
            records.push(json!({"load":load,"vision_resident":vision,"warmup":replay==0,
                "load_seconds":load_seconds,"output_sha256":hash,"sampled_tokens":count,
                "first_token_seconds":first,"end_to_end_seconds":elapsed,
                "decode_tokens_per_second":(count.saturating_sub(1)) as f64/(elapsed-first)}));
        }
        drop(model);
        drop(engine);
    }
    let receipt = json!({"schema_version":1,"attempt":"20261109-QDIP-02",
        "scope":"same-admitted-target tower residency; prior text export is a separate required control",
        "artifact":identity,"selected_config_sha256":format!("{:x}",Sha256::digest(&config)),
        "input_tokens":input.len(),"context_tokens":4096,"max_new_tokens":32,"sampling":"greedy",
        "mtp_drafts":0,"prefix_cache":false,"same_output_hashes":true,"records":records,
        "prior_text_comparison":"not run by this control","host_screening":"exclusive GPU supplied by coordinator; no thermal or peak RSS qualification"});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("private parity output setup failed");
    }
    std::fs::write(output, serde_json::to_vec_pretty(&receipt).unwrap()).expect("private parity receipt failed");
    let _: Value = receipt;
}

#[stark_test]
#[ignore]
fn qwen_image_dense_mtp_cancelled_prefill_keeps_loaded_instance_usable() {
    use crate::engine::language_model::dense_mtp::DenseMtpError;

    let path = PathBuf::from(std::env::var_os("STARK_QWEN_IMAGE_PARITY_MODEL_PATH").expect("set private model path"));
    let output = PathBuf::from(std::env::var_os("STARK_QWEN_IMAGE_PARITY_OUTPUT").expect("set private result path"));
    let engine = Engine::<Metal>::new().expect("private parity engine failed");
    let mut model = engine.load_vision_language_model(&path).expect("private image artifact admission failed");
    assert!(model.has_dense_mtp(), "private artifact requires a compatible head");
    let input = model
        .tokenizer()
        .encode("Describe an algorithm in one sentence.", false)
        .expect("private parity tokenization failed")
        .get_ids()
        .to_vec();
    assert!(!input.is_empty() && input.len() < 1024);
    assert!(matches!(model.prefill_dense_mtp_controlled(&[], 2048, &mut || false), Err(DenseMtpError::Invalid)));
    assert!(!model.is_fenced());
    let mut checks = 0;
    let cancelled = model.prefill_dense_mtp_controlled(&input, 2048, &mut || {
        checks += 1;
        // One full target/predictor prefill chunk has completed and drained
        // before the second callback, so this is not pre-inference cancellation.
        checks >= 2
    });
    assert!(matches!(cancelled, Err(DenseMtpError::Cancelled)));
    assert!(checks >= 2 && !model.is_fenced() && model.has_dense_mtp());
    let session = model
        .prefill_dense_mtp_controlled(&input, 2048, &mut || false)
        .expect("fresh text after drained cancellation failed");
    assert_eq!(session.position(), input.len());
    assert!(!session.is_fenced() && !model.is_fenced());
    let receipt = json!({"attempt":"20261109-QDIP-04","passed":true,
        "artifact_sha256":model.artifact_identity().artifact_sha256,
        "config_sha256":model.artifact_identity().config_sha256,
        "input_tokens":input.len(),"context_tokens":2048,
        "cancel_checks":checks,"cancelled_after_completed_prefill_chunk":true,
        "invalid_preflight_recoverable":true,"fresh_text_after_cancel":true});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("private parity output setup failed");
    }
    std::fs::write(output, serde_json::to_vec_pretty(&receipt).unwrap()).expect("private parity receipt failed");
}
