//! Explicit model-bound component check; not worker MTP qualification.
use std::{
    io::Write,
    process::{Command, Stdio},
};

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::metal::Metal,
    encodable_block::gemma4_assistant::{Gemma4Assistant, Gemma4AssistantProfile},
};

#[stark_test]
#[ignore]
fn private_e2b_assistant_real_drafts_with_image_tower_resident() {
    let target = std::env::var("STARK_GEMMA_ASSISTANT_TARGET").expect("private target required");
    let companion = std::env::var("STARK_GEMMA_ASSISTANT_COMPANION").expect("private companion required");
    let output = std::env::var("STARK_GEMMA_ASSISTANT_RECEIPT").expect("private aggregate receipt required");
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_gemma4_vision_language_model(Path::new(&target)).unwrap();
    let identity = model.artifact_identity();
    let config: serde_json::Value =
        serde_json::from_reader(File::open(Path::new(&companion).join("config.json")).unwrap()).unwrap();
    assert!(config["target"]["config_sha256"] == identity.config_sha256);
    assert!(config["target"]["weights_sha256"] == identity.artifact_sha256);
    assert!(config["source"]["repo"] == "google/gemma-4-E2B-it-assistant");
    assert!(config["source"]["revision"] == "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df");
    assert!(config["schema_version"] == 2 && config["backbone_dim"] == 1536);
    let mut weights = super::super::open_artifact_component(&Path::new(&companion).join("model.safetensors")).unwrap();
    let before = super::super::file_digest(&mut weights).unwrap();
    assert!(config["artifact_sha256"] == before);
    weights.rewind().unwrap();
    let loader = ParameterLoader::new(&weights, &*engine.context).unwrap();
    let head = Gemma4Assistant::new(Gemma4AssistantProfile::E2B, &loader.tree().subtree("assistant"), &*engine.context)
        .unwrap();
    loader.tree().assert_all_tensors_validated().unwrap();
    drop(loader);
    weights.rewind().unwrap();
    assert!(super::super::file_digest(&mut weights).unwrap() == before);
    let input = model.tokenizer().encode("Describe an algorithm in one sentence.", false).unwrap().get_ids().to_vec();
    let frame = model
        .language_model
        .prefill_gemma4_assistant_target(&input, 128, Gemma4AssistantProfile::E2B, &mut || false)
        .unwrap();
    assert_eq!(frame.position(), input.len() as u32);
    let first =
        frame.logits().iter().enumerate().max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a))).unwrap().0
            as u32;
    let mut ordinary = model.language_model.create_empty_state(Some(128), 42).unwrap();
    let mut options = model.language_model.default_stream_options();
    options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
    let ordinary_input = input.iter().map(|&t| t as u64).collect::<Vec<_>>();
    let mut stream = model.language_model.stream(&ordinary_input, &mut ordinary, options).unwrap();
    assert!(stream.next().unwrap().unwrap() == first as u64, "assistant frame root differs from ordinary target draw");
    drop(stream);
    let drafts = model.language_model.draft_gemma4_assistant_greedy(&frame, &head, first, 3).unwrap();
    assert_eq!(drafts.len(), 3);
    assert!(drafts.iter().all(|&t| t < 262144));
    let mut cancellation_checks = 0;
    assert!(matches!(
        model.language_model.draft_gemma4_assistant_greedy_controlled(&frame, &head, first, 3, &mut || {
            cancellation_checks += 1;
            cancellation_checks == 2
        }),
        Err(crate::engine::language_model::gemma4_assistant::Gemma4AssistantSessionError::Cancelled)
    ));
    let retry = model.language_model.draft_gemma4_assistant_greedy(&frame, &head, first, 3).unwrap();
    assert!(retry == drafts, "drained draft cancellation changed retry proposals");
    let oracle_python =
        std::env::var("STARK_GEMMA_ASSISTANT_ORACLE_PYTHON").expect("private oracle interpreter required");
    let oracle_script = std::env::var("STARK_GEMMA_ASSISTANT_ORACLE_SCRIPT").expect("private oracle script required");
    let payload = model.language_model.assistant_validation_payload(&frame, &head, first).unwrap();
    let mut child = Command::new(oracle_python)
        .arg(oracle_script)
        .arg(std::env::var("STARK_GEMMA_ASSISTANT_SOURCE").unwrap())
        .arg(Path::new(&target).join("model.safetensors"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&serde_json::to_vec(&payload).unwrap()).unwrap();
    drop(payload);
    let result = child.wait_with_output().unwrap();
    // Never print reference stderr, which may include tensors on failure.
    assert!(result.status.success(), "private reference process failed");
    let oracle: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let mut hash = Sha256::new();
    for token in &drafts {
        hash.update(token.to_le_bytes());
    }
    let aggregate = serde_json::json!({"attempt":"20261109-G4AP-01","scope":"functional native assistant component; no target verification or speed qualification",
        "target_config_sha256":identity.config_sha256,"target_weights_sha256":identity.artifact_sha256,
        "companion_sha256":before,"input_tokens":input.len(),"drafts":drafts.len(),"draft_sha256":format!("{:x}",hash.finalize()),
        "ordinary_target_root_equal":true,"target_position_unchanged":frame.position()==input.len() as u32,"image_tower_resident":true,
        "worker_mtp_enabled":false,"drained_draft_cancellation_retry":true,"reference_numerical_parity":oracle});
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    file.write_all(serde_json::to_string_pretty(&aggregate).unwrap().as_bytes()).unwrap();
    assert!(
        aggregate["reference_numerical_parity"]["passed"] == true,
        "assistant numerical reference gate failed; see aggregate receipt"
    );
}
