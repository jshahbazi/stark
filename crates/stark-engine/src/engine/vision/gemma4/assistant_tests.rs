//! Explicit model-bound component check; not worker MTP qualification.
use std::{
    io::Write,
    process::{Command, Stdio},
};

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{backends::metal::Metal, encodable_block::gemma4_assistant::Gemma4AssistantProfile};

#[stark_test]
#[ignore]
fn private_e2b_assistant_real_drafts_with_image_tower_resident() {
    let target = std::env::var("STARK_GEMMA_ASSISTANT_TARGET").expect("private target required");
    let companion = std::env::var("STARK_GEMMA_ASSISTANT_COMPANION").expect("private companion required");
    let output = std::env::var("STARK_GEMMA_ASSISTANT_RECEIPT").expect("private aggregate receipt required");
    let engine = Engine::<Metal>::new().unwrap();
    let mut model = engine.load_gemma4_vision_language_model(Path::new(&target)).unwrap();
    let identity = model.artifact_identity().clone();
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
    let invalid = super::super::tests::PrivateDirectory::new();
    let invalid_file = File::create(invalid.path().join("model.safetensors")).unwrap();
    assert!(model.load_assistant_component(invalid.path()).is_err());
    invalid_file.set_len(157_581_400).unwrap();
    assert!(model.load_assistant_component(invalid.path()).is_err());
    #[cfg(unix)]
    {
        let linked = super::super::tests::PrivateDirectory::new();
        std::os::unix::fs::symlink(
            Path::new(&companion).join("model.safetensors"),
            linked.path().join("model.safetensors"),
        )
        .unwrap();
        assert!(model.load_assistant_component(linked.path()).is_err());
    }
    let config_hash = std::mem::replace(&mut model.identity.config_sha256, "0".repeat(64));
    assert!(model.load_assistant_component(Path::new(&companion)).is_err());
    model.identity.config_sha256 = config_hash;
    let head = model.load_assistant_component(Path::new(&companion)).unwrap();
    assert!(head.matches_target(&model.language_model.model_identity));
    assert!(!head.matches_target(&std::sync::Arc::new(())));
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
    let ordinary_tokens = (0..5).map(|_| stream.next().unwrap().unwrap() as u32).collect::<Vec<_>>();
    assert!(ordinary_tokens[0] == first, "assistant frame root differs from ordinary target draw");
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
    let initial_position = frame.position();
    let verified = model.language_model.verify_gemma4_assistant_greedy_serial(frame, &drafts, &mut || false).unwrap();
    let committed = verified.committed_tokens.len();
    assert!(
        verified.committed_tokens == ordinary_tokens[..committed] && verified.next_token == ordinary_tokens[committed],
        "serial verified output differs from ordinary target"
    );
    assert!(
        verified.frame.position() == initial_position + committed as u32 && committed == verified.accepted_drafts + 1
    );
    let accepted = verified.accepted_drafts;
    drop(verified);
    let fresh = || {
        model
            .language_model
            .prefill_gemma4_assistant_target(&input, 128, Gemma4AssistantProfile::E2B, &mut || false)
            .unwrap()
    };
    let all = model
        .language_model
        .verify_gemma4_assistant_greedy_serial(fresh(), &ordinary_tokens[1..4], &mut || false)
        .unwrap();
    assert!(
        all.committed_tokens == ordinary_tokens[..4]
            && all.next_token == ordinary_tokens[4]
            && all.accepted_drafts == 3
            && all.frame.position() == initial_position + 4
    );
    drop(all);
    let wrong = (ordinary_tokens[1] + 1) % 262144;
    let rejected =
        model.language_model.verify_gemma4_assistant_greedy_serial(fresh(), &[wrong], &mut || false).unwrap();
    assert!(
        rejected.committed_tokens == ordinary_tokens[..1]
            && rejected.next_token == ordinary_tokens[1]
            && rejected.accepted_drafts == 0
            && rejected.frame.position() == initial_position + 1
    );
    drop(rejected);
    let partial = model
        .language_model
        .verify_gemma4_assistant_greedy_serial(
            fresh(),
            &[ordinary_tokens[1], (ordinary_tokens[2] + 1) % 262144],
            &mut || false,
        )
        .unwrap();
    assert!(
        partial.committed_tokens == ordinary_tokens[..2]
            && partial.next_token == ordinary_tokens[2]
            && partial.accepted_drafts == 1
            && partial.frame.position() == initial_position + 2
    );
    drop(partial);
    let zero = model.language_model.verify_gemma4_assistant_greedy_serial(fresh(), &[], &mut || false).unwrap();
    assert!(
        zero.committed_tokens == ordinary_tokens[..1]
            && zero.next_token == ordinary_tokens[1]
            && zero.accepted_drafts == 0
            && zero.frame.position() == initial_position + 1
    );
    drop(zero);
    let mut checks = 0;
    assert!(matches!(
        model.language_model.verify_gemma4_assistant_greedy_serial(fresh(), &drafts, &mut || {
            checks += 1;
            checks == 2
        }),
        Err(crate::engine::language_model::gemma4_assistant::Gemma4AssistantSessionError::Cancelled)
    ));
    let recovered = model
        .language_model
        .verify_gemma4_assistant_greedy_serial(fresh(), &ordinary_tokens[1..4], &mut || false)
        .unwrap();
    assert!(
        recovered.committed_tokens == ordinary_tokens[..4]
            && recovered.next_token == ordinary_tokens[4]
            && recovered.accepted_drafts == 3
            && recovered.frame.position() == initial_position + 4
    );
    drop(recovered);
    // Public drafting now admits the separately qualified 513-row handoff;
    // the staged context/position upper bounds remain explicit.
    let boundary_input = (0..513).map(|i| input[i % input.len()]).collect::<Vec<_>>();
    let boundary = model
        .language_model
        .prefill_gemma4_assistant_target(&boundary_input[..512], 520, Gemma4AssistantProfile::E2B, &mut || false)
        .unwrap();
    let boundary_root =
        boundary.logits().iter().enumerate().max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a))).unwrap().0
            as u32;
    assert!(model.language_model.draft_gemma4_assistant_greedy(&boundary, &head, boundary_root, 1).is_ok());
    drop(boundary);
    let beyond = model
        .language_model
        .prefill_gemma4_assistant_target(&boundary_input, 520, Gemma4AssistantProfile::E2B, &mut || false)
        .unwrap();
    assert!(beyond.assistant_drafting_supported());
    assert!(model.language_model.draft_gemma4_assistant_greedy(&beyond, &head, first, 1).is_ok());
    let continued = model.language_model.verify_gemma4_assistant_greedy_serial(beyond, &[], &mut || false).unwrap();
    assert!(continued.frame.position() == 514 && continued.committed_tokens.len() == 1);
    drop(continued);
    let mut hash = Sha256::new();
    for token in &drafts {
        hash.update(token.to_le_bytes());
    }
    let aggregate = serde_json::json!({"attempt":"20261109-G4AP-06","scope":"assistant source MLP, committed-query and serial verification; no serving or speed qualification",
        "assistant_numerical_policy":head.numerical_policy(),
        "target_config_sha256":identity.config_sha256,"target_weights_sha256":identity.artifact_sha256,
        "companion_sha256":before,"input_tokens":input.len(),"drafts":drafts.len(),"draft_sha256":format!("{:x}",hash.finalize()),
        "ordinary_target_root_equal":true,"target_position_unchanged":initial_position==input.len() as u32,"image_tower_resident":true,
        "worker_mtp_enabled":false,"drained_draft_cancellation_retry":true,
        "companion_admission":{"wrong_size_rejected":true,"wrong_digest_rejected":true,"symlink_rejected":cfg!(unix),"wrong_target_rejected":true,"foreign_model_identity_rejected":true},
        "serial_verification":{"actual_proposals_accepted":accepted,"fresh_output_equal":true,"all_accept_three":true,
            "deliberate_reject":true,"partial_accept_then_reject":true,"zero_drafts":true,
            "drained_cancel_then_fresh":true,"performance_qualified":false},
        "context_guard":{"draft_position_512_admitted":true,"draft_position_513_admitted":true,
            "ordinary_target_continues":true,"qualified_context_max":2048,"qualified_draft_position_max":1792},
        "reference_numerical_parity":oracle});
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    file.write_all(serde_json::to_string_pretty(&aggregate).unwrap().as_bytes()).unwrap();
    assert!(
        aggregate["reference_numerical_parity"]["passed"] == true,
        "assistant numerical reference gate failed; see aggregate receipt"
    );
}

#[stark_test]
#[ignore]
fn private_e2b_assistant_wrapped_handoff_and_transactional_batch() {
    let target = std::env::var("STARK_GEMMA_ASSISTANT_TARGET").expect("private target required");
    let companion = std::env::var("STARK_GEMMA_ASSISTANT_COMPANION").expect("private companion required");
    let output = std::env::var("STARK_GEMMA_ASSISTANT_RECEIPT").expect("private aggregate receipt required");
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_gemma4_vision_language_model(Path::new(&target)).unwrap();
    let head = model.load_assistant_component(Path::new(&companion)).unwrap();
    let mut receipt = std::fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    let public_text = (0..600)
        .map(|index| format!("Step {index}: verify the distinct state and then continue. "))
        .collect::<String>();
    let source_input = model.tokenizer().encode(public_text, false).unwrap().get_ids().to_vec();
    assert!(source_input.len() >= 1792);
    for length in [512usize, 513, 1024, 1026, 1792] {
        let input = &source_input[..length];
        let fresh = || {
            model
                .language_model
                .prefill_gemma4_assistant_target(input, 2048, Gemma4AssistantProfile::E2B, &mut || false)
                .unwrap()
        };
        let frame = fresh();
        let mut ordinary = model.language_model.create_empty_state(Some(2048), 42).unwrap();
        let mut options = model.language_model.default_stream_options();
        options.sampling_method = crate::engine::language_model::stream::SamplingMethod::Greedy;
        let ordinary_input = input.iter().map(|&token| u64::from(token)).collect::<Vec<_>>();
        let mut stream = model.language_model.stream(&ordinary_input, &mut ordinary, options).unwrap();
        let ordinary_tokens = (0..5).map(|_| stream.next().unwrap().unwrap() as u32).collect::<Vec<_>>();
        drop(stream);
        let first =
            frame.logits().iter().enumerate().max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a))).unwrap().0
                as u32;
        assert!(first == ordinary_tokens[0]);
        let payload = model.language_model.assistant_validation_round_payload(&frame, &head, first).unwrap();
        let drafts = payload["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| {
                step["logits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .max_by(|(a, x), (b, y)| x.as_f64().unwrap().total_cmp(&y.as_f64().unwrap()).then_with(|| b.cmp(a)))
                    .unwrap()
                    .0 as u32
            })
            .collect::<Vec<_>>();
        let mut child = Command::new(std::env::var("STARK_GEMMA_ASSISTANT_ORACLE_PYTHON").unwrap())
            .arg(std::env::var("STARK_GEMMA_ASSISTANT_ORACLE_SCRIPT").unwrap())
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
        assert!(result.status.success(), "private chained reference process failed");
        let oracle: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        let numeric_record = serde_json::json!({"attempt":"20261109-G4AP-06","gate":"three-step-source-oracle",
            "assistant_numerical_policy":head.numerical_policy(),
            "input_tokens":length,"target_config_sha256":model.artifact_identity().config_sha256,
            "target_weights_sha256":model.artifact_identity().artifact_sha256,"image_tower_resident":true,
            "reference":oracle});
        writeln!(receipt, "{}", serde_json::to_string(&numeric_record).unwrap()).unwrap();
        receipt.flush().unwrap();
        assert!(
            numeric_record["reference"]["passed"] == true,
            "chained source numerical gate failed; aggregate receipt retained"
        );
        let public_drafts = model
            .language_model
            .draft_gemma4_assistant_greedy_controlled(&frame, &head, first, 3, &mut || false)
            .unwrap();
        assert!(public_drafts == drafts, "public drafting differs from the private qualified head payload");
        if length >= 1026 {
            let captured = model
                .language_model
                .prefill_gemma4_assistant_target_with_prefix(
                    input,
                    2048,
                    Gemma4AssistantProfile::E2B,
                    1024,
                    &mut || false,
                )
                .unwrap();
            assert!(captured.prefix_position() == Some(1024) && captured.prefix_payload_bytes() > 0);
            let advanced = model
                .language_model
                .verify_gemma4_assistant_greedy_serial(captured, &ordinary_tokens[1..4], &mut || false)
                .unwrap();
            let restored = model
                .language_model
                .restore_and_prefill_gemma4_assistant_target(advanced.frame, &input[1024..], 1024, &mut || false)
                .unwrap();
            assert!(restored.position() == length as u32 && restored.logits() == frame.logits());
            let fresh_payload = model.language_model.assistant_validation_payload(&frame, &head, first).unwrap();
            let restored_payload = model.language_model.assistant_validation_payload(&restored, &head, first).unwrap();
            assert!(fresh_payload == restored_payload, "prefix restoration changed private target handoff");
            drop(fresh_payload);
            drop(restored_payload);
            let restored_round = model
                .language_model
                .verify_gemma4_assistant_greedy_serial(restored, &ordinary_tokens[1..4], &mut || false)
                .unwrap();
            assert!(
                restored_round.committed_tokens == ordinary_tokens[..4]
                    && restored_round.next_token == ordinary_tokens[4]
            );
            let mut changed = input.to_vec();
            changed[length - 1] = (changed[length - 1] + 1) % 262144;
            let changed_fresh = model
                .language_model
                .prefill_gemma4_assistant_target(&changed, 2048, Gemma4AssistantProfile::E2B, &mut || false)
                .unwrap();
            let changed_restored = model
                .language_model
                .restore_and_prefill_gemma4_assistant_target(restored_round.frame, &changed[1024..], 1024, &mut || {
                    false
                })
                .unwrap();
            assert!(
                changed_fresh.logits() == changed_restored.logits() && changed_restored.position() == length as u32
            );
            drop(changed_fresh);
            drop(changed_restored);
        }
        let actual = model.language_model.verify_gemma4_assistant_greedy_batch(frame, &drafts, &mut || false).unwrap();
        let committed = actual.committed_tokens.len();
        assert!(
            actual.committed_tokens == ordinary_tokens[..committed] && actual.next_token == ordinary_tokens[committed]
        );
        assert!(actual.frame.position() == length as u32 + committed as u32 && committed == actual.accepted_drafts + 1);
        let actual_accepted = actual.accepted_drafts;
        drop(actual);
        let all = model
            .language_model
            .verify_gemma4_assistant_greedy_batch(fresh(), &ordinary_tokens[1..4], &mut || false)
            .unwrap();
        assert!(
            all.committed_tokens == ordinary_tokens[..4]
                && all.next_token == ordinary_tokens[4]
                && all.accepted_drafts == 3
                && all.frame.position() == length as u32 + 4
                && all.repaired_rows == 0
        );
        drop(all);
        let wrong = (ordinary_tokens[1] + 1) % 262144;
        let reject =
            model.language_model.verify_gemma4_assistant_greedy_batch(fresh(), &[wrong], &mut || false).unwrap();
        assert!(
            reject.committed_tokens == ordinary_tokens[..1]
                && reject.next_token == ordinary_tokens[1]
                && reject.accepted_drafts == 0
                && reject.repaired_rows == 1
                && reject.frame.position() == length as u32 + 1
        );
        drop(reject);
        let partial = model
            .language_model
            .verify_gemma4_assistant_greedy_batch(
                fresh(),
                &[ordinary_tokens[1], (ordinary_tokens[2] + 1) % 262144],
                &mut || false,
            )
            .unwrap();
        assert!(
            partial.committed_tokens == ordinary_tokens[..2]
                && partial.next_token == ordinary_tokens[2]
                && partial.accepted_drafts == 1
                && partial.repaired_rows == 2
                && partial.frame.position() == length as u32 + 2
        );
        drop(partial);
        let zero = model.language_model.verify_gemma4_assistant_greedy_batch(fresh(), &[], &mut || false).unwrap();
        assert!(
            zero.committed_tokens == ordinary_tokens[..1]
                && zero.next_token == ordinary_tokens[1]
                && zero.accepted_drafts == 0
                && zero.frame.position() == length as u32 + 1
        );
        drop(zero);
        for stop_at in [2usize, 3] {
            let mut checks = 0;
            assert!(matches!(
                model.language_model.verify_gemma4_assistant_greedy_batch(fresh(), &[wrong], &mut || {
                    checks += 1;
                    checks == stop_at
                }),
                Err(crate::engine::language_model::gemma4_assistant::Gemma4AssistantSessionError::Cancelled)
            ));
        }
        let recovered = model
            .language_model
            .verify_gemma4_assistant_greedy_batch(fresh(), &ordinary_tokens[1..4], &mut || false)
            .unwrap();
        assert!(recovered.committed_tokens == ordinary_tokens[..4] && recovered.next_token == ordinary_tokens[4]);
        drop(recovered);
        let control_record = serde_json::json!({"attempt":"20261109-G4AP-06","gate":"transactional-target-controls",
            "assistant_numerical_policy":head.numerical_policy(),
            "input_tokens":length,"actual_drafts":drafts.len(),"actual_accepted":actual_accepted,
            "ordinary_output_equal":true,"supplied_all_accept":true,"supplied_partial_accept":true,
            "deliberate_rejection":true,"zero_drafts":true,"post_batch_and_post_restore_cancel":true,
            "fresh_recovery":true,"prefix_reuse":{"tested":length>=1026,"boundary":if length>=1026 {1024}else{0},
                "suffix_rows":if length>=1026 {length-1024}else{0},"fresh_handoff_equal":length>=1026,
                "changed_tail_equal":length>=1026},"serving_mtp_enabled":false,"performance_qualified":false});
        writeln!(receipt, "{}", serde_json::to_string(&control_record).unwrap()).unwrap();
        receipt.flush().unwrap();
    }
}

#[stark_test]
#[ignore]
fn private_e2b_assistant_first_divergence_trace() {
    let target = std::env::var("STARK_GEMMA_ASSISTANT_TARGET").expect("private target required");
    let companion = std::env::var("STARK_GEMMA_ASSISTANT_COMPANION").expect("private companion required");
    let output = std::env::var("STARK_GEMMA_ASSISTANT_RECEIPT").expect("private aggregate receipt required");
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_gemma4_vision_language_model(Path::new(&target)).unwrap();
    let head = model.load_assistant_component(Path::new(&companion)).unwrap();
    let public_text = (0..600)
        .map(|index| format!("Step {index}: verify the distinct state and then continue. "))
        .collect::<String>();
    let source_input = model.tokenizer().encode(public_text, false).unwrap().get_ids().to_vec();
    let input = &source_input[..512];
    let frame = model
        .language_model
        .prefill_gemma4_assistant_target(input, 2048, Gemma4AssistantProfile::E2B, &mut || false)
        .unwrap();
    let first =
        frame.logits().iter().enumerate().max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a))).unwrap().0
            as u32;
    let payload = model.language_model.assistant_validation_trace_payload(&frame, &head, first).unwrap();
    let mut child = Command::new(std::env::var("STARK_GEMMA_ASSISTANT_ORACLE_PYTHON").unwrap())
        .arg(std::env::var("STARK_GEMMA_ASSISTANT_ORACLE_SCRIPT").unwrap())
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
    let reference = if result.status.success() {
        serde_json::from_slice::<serde_json::Value>(&result.stdout).unwrap()
    } else {
        serde_json::json!({"diagnostic_complete":false,"reference_process_succeeded":false,"returncode":result.status.code()})
    };
    let record = serde_json::json!({"attempt":"20261109-G4AP-06-diagnostic","diagnostic_only":true,
        "target_config_sha256":model.artifact_identity().config_sha256,
        "target_weights_sha256":model.artifact_identity().artifact_sha256,"input_tokens":input.len(),
        "ordinary_target_arithmetic_changed":false,"assistant_numerical_policy":head.numerical_policy(),
        "serving_mtp_enabled":false,"reference":reference});
    let mut receipt = std::fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    receipt.write_all(serde_json::to_string_pretty(&record).unwrap().as_bytes()).unwrap();
    assert!(record["reference"]["diagnostic_complete"] == true, "diagnostic failed; aggregate receipt retained");
}
