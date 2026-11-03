// SPDX-License-Identifier: Apache-2.0
//! Private source-bound tests for the opt-in GLM prompt ingest overlap.

use stark_engine_macros::stark_test;

use super::*;

fn fixture() -> (std::path::PathBuf, Value, Vec<u32>) {
    let path = std::env::var_os("STARK_GLM_REFERENCE_FIXTURE_PATH")
        .map(std::path::PathBuf::from)
        .expect("GLM private fixture path required");
    let config: Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).expect("private GLM config read failed"))
            .expect("private GLM config is invalid");
    let tokens: Vec<u32> = serde_json::from_slice(
        &std::fs::read(path.join("tokens.json")).expect("private GLM token fixture read failed"),
    )
    .expect("private GLM token fixture is invalid");
    assert!(!tokens.is_empty(), "private GLM fixture is empty");
    // Synthetic repeated prefix tests capacity and chronology; the separate
    // source-oracle replay continues to use its unchanged original tokens.
    let prefix = tokens.into_iter().cycle().take(66).collect();
    (path, config, prefix)
}

fn new_model(
    path: &std::path::Path,
    config: &Value,
) -> GlmModel {
    let mut model = GlmModel::new(Weights::open(path).expect("private GLM fixture weights are invalid"), config)
        .expect("private GLM fixture admission failed");
    assert!(model.decode.is_none());
    // Keep the source metadata admission intact; enlarge only synthetic test
    // capacity before lazy Metal state creation so the 65-token case grows KV.
    model.config.max_context = 128;
    model
}

fn require_overlap_environment(queued: bool) {
    assert!(
        ["STARK_GLM53_PREFILL_OVERLAP", "STARK_GLM53_PREFILL_QUEUED"]
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| value == "1")),
        "a prefill experiment flag is required"
    );
    for name in [
        "STARK_GLM53_ENCODER_BANKS",
        "STARK_GLM53_RESIDENT_BANKS",
        "STARK_GLM53_WIDE_ENCODER",
        "STARK_GLM53_GREEDY_OVERLAP",
    ] {
        assert!(std::env::var_os(name).is_some_and(|value| value == "1"), "{name}=1 is required");
    }
    if queued {
        assert!(
            std::env::var_os("STARK_GLM53_UNRETAINED_GRAPH").is_some_and(|value| value == "1"),
            "STARK_GLM53_UNRETAINED_GRAPH=1 is required"
        );
    }
    for name in [
        "STARK_QWEN38_PHASE_TIMING",
        "STARK_QWEN38_ENCODER_TIMING",
        "STARK_GLM53_CONCURRENT_ENCODER",
        "STARK_GLM53_PARALLEL_PROJECTIONS",
    ] {
        assert!(!std::env::var_os(name).is_some_and(|value| value == "1"), "{name} must be disabled");
    }
}

fn pool_lengths(model: &GlmModel) -> Vec<usize> {
    model
        .decode
        .as_ref()
        .expect("GLM decode state was not initialized")
        .attention
        .iter()
        .flatten()
        .map(|attention| attention.test_pool_keys().len)
        .collect()
}

fn same_logits(
    actual: &[f32],
    expected: &[f32],
) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[stark_test]
#[ignore]
fn glm_prefill_overlap_matches_scalar_prefix_and_resets() {
    let (path, config, tokens) = fixture();
    for queued in [false, true] {
        require_overlap_environment(queued);
        for length in [0usize, 1, 2, 13, 65] {
            let mut scalar = new_model(&path, &config);
            let mut overlapped = new_model(&path, &config);
            let prefix = &tokens[..length];
            for &token in prefix {
                scalar.ingest(token).expect("scalar GLM prefix ingest failed");
            }
            let mut callback_count = 0usize;
            overlapped
                .ingest_prefix_policy(
                    prefix,
                    &mut || {
                        callback_count += 1;
                        Ok(())
                    },
                    true,
                    queued,
                )
                .expect("overlapped GLM prefix ingest failed");
            assert_eq!(scalar.position, length);
            assert_eq!(overlapped.position, length);
            assert!(callback_count >= length);
            let scalar_logits = scalar.step(tokens[length]).expect("scalar GLM readout failed");
            let overlapped_logits = overlapped.step(tokens[length]).expect("overlapped GLM readout failed");
            assert!(
                scalar_logits.iter().zip(&overlapped_logits).all(|(a, b)| a.to_bits() == b.to_bits()),
                "queued={queued}, prefix length {length} changed logits"
            );
            assert_eq!(scalar.position, length + 1);
            assert_eq!(overlapped.position, length + 1);
            if length == 65 {
                let initial = 64usize.div_ceil(overlapped.config.index_pool) * overlapped.config.index_dim;
                let lengths = pool_lengths(&overlapped);
                assert!(!lengths.is_empty());
                assert!(lengths.iter().all(|&length| length > initial), "prefill did not grow the MLA allocation");
            }

            let replay = scalar_logits;
            overlapped.reset();
            assert_eq!(overlapped.position, 0);
            overlapped.ingest_prefix_policy(prefix, &mut || Ok(()), true, queued).expect("GLM prefix replay failed");
            assert!(
                same_logits(&overlapped.step(tokens[length]).expect("GLM replay readout failed"), &replay),
                "GLM prefix replay changed logits"
            );
        }
    }
}

#[stark_test]
#[ignore]
fn glm_prefill_overlap_rejects_invalid_tokens_context_and_cancellation() {
    let (path, config, tokens) = fixture();
    for queued in [false, true] {
        require_overlap_environment(queued);
        let mut invalid = new_model(&path, &config);
        assert!(invalid.ingest_prefix_policy(&[u32::MAX], &mut || Ok(()), true, queued).is_err());
        assert_eq!(invalid.position, 0);
        let expected = invalid.step(tokens[0]).expect("GLM recovery after invalid prefix failed");

        let mut boundary = new_model(&path, &config);
        boundary.config.max_context = 2;
        let mut calls = 0usize;
        assert!(
            boundary
                .ingest_prefix_policy(
                    &tokens[..3],
                    &mut || {
                        calls += 1;
                        Ok(())
                    },
                    true,
                    queued
                )
                .is_err()
        );
        assert_eq!(boundary.position, 0);
        assert_eq!(calls, 0, "invalid context reached token reservation");
        boundary.config.max_context = 128;
        assert!(
            same_logits(&boundary.step(tokens[0]).expect("GLM recovery after context failure failed"), &expected),
            "GLM context recovery changed logits"
        );

        for cancel_at in [1, 3, 4, 6] {
            let mut cancelled = new_model(&path, &config);
            let mut calls = 0usize;
            assert!(
                cancelled
                    .ingest_prefix_policy(
                        &tokens[..8],
                        &mut || {
                            calls += 1;
                            if calls == cancel_at {
                                Err(Error::Invalid("GLM test cancellation"))
                            } else {
                                Ok(())
                            }
                        },
                        true,
                        queued
                    )
                    .is_err()
            );
            assert_eq!(cancelled.position, 0);
            assert_eq!(calls, cancel_at);
            assert!(
                same_logits(&cancelled.step(tokens[0]).expect("GLM recovery after cancellation failed"), &expected),
                "GLM cancellation recovery changed logits"
            );
        }
    }
}

#[stark_test]
#[ignore]
fn glm_prefill_default_path_keeps_scalar_calls_and_logits() {
    let (path, config, tokens) = fixture();
    let mut scalar = new_model(&path, &config);
    let mut bulk = new_model(&path, &config);
    let prefix = &tokens[..13];
    for &token in prefix {
        scalar.ingest(token).unwrap();
    }
    let mut callbacks = 0;
    bulk.ingest_prefix_policy(
        prefix,
        &mut || {
            callbacks += 1;
            Ok(())
        },
        false,
        false,
    )
    .unwrap();
    assert_eq!(callbacks, prefix.len());
    assert_eq!(bulk.position, scalar.position);
    let expected = scalar.step(tokens[13]).unwrap();
    let actual = bulk.step(tokens[13]).unwrap();
    assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[stark_test]
#[ignore]
fn glm_prefill_failed_pending_cleanup_keeps_persistent_fence() {
    require_overlap_environment(false);
    let (path, config, tokens) = fixture();
    for failed in [false, true] {
        let mut model = new_model(&path, &config);
        model.ingest(tokens[0]).unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        let input = graph
            .upload_into(&[if failed {
                f32::MAX
            } else {
                1.0
            }])
            .unwrap();
        graph.add(&input, &input).unwrap();
        let pending = graph.submit(&[]).unwrap();
        // Simulate a callback/staging error after a future pass reserved host
        // metadata. The unsubmitted owner is discarded before pending drains.
        let future = model.encode_pass(None).unwrap();
        drop(future);
        model.drain_prefill_error(Some(pending));
        assert_eq!(model.position, 0);
        assert_eq!(model.is_fenced(), failed);
        model.reset();
        assert_eq!(model.is_fenced(), failed, "reset hid a failed prefill cleanup fence");
        if failed {
            assert!(model.step(tokens[0]).is_err(), "failed prefill cleanup reused its model");
        } else {
            assert!(model.step(tokens[0]).is_ok(), "successful prefill cleanup prevented recovery");
        }
    }
}

#[stark_test]
#[ignore]
fn glm_prefill_queued_cleanup_drains_second_writer_after_first_error() {
    require_overlap_environment(true);
    let (path, config, _) = fixture();
    let mut model = new_model(&path, &config);

    let mut first = DecodeGraph::new_unretained().unwrap();
    let first_input = first.upload_into(&[f32::MAX]).unwrap();
    first.add(&first_input, &first_input).unwrap();
    first.validate_queued_submission().unwrap();
    let first_pending = first.submit(&[]).unwrap();

    let mut second = DecodeGraph::new_unretained().unwrap();
    let second_input = second.upload_into(&[3.0]).unwrap();
    let second_output = second.add(&second_input, &second_input).unwrap();
    second.validate_queued_submission().unwrap();
    let second_pending = second.submit(std::slice::from_ref(&second_output)).unwrap();

    let mut queued = std::collections::VecDeque::with_capacity(2);
    queued.push_back(first_pending);
    queued.push_back(second_pending);
    model.drain_prefill_queued_error(queued);
    // SAFETY: cleanup drained every submitted state writer before returning;
    // this separately owned Shared output has no remaining GPU writer.
    let value =
        unsafe { second_output.buffer.contents().as_ptr().cast::<u8>().add(second_output.offset).cast::<f32>().read() };
    assert_eq!(value, 6.0, "queued cleanup returned before the second writer completed");
    assert!(model.is_fenced(), "failed first completion did not fence the model");
    model.reset();
    assert!(model.is_fenced(), "reset cleared the queued cleanup failure");
    assert!(model.step(0).is_err(), "failed queued cleanup reused its model");

    // The second command is a submitted writer behind the failed first one.
    // Its escaped output must remain usable after the helper drains both owners.
    let mut consumer = DecodeGraph::new_unretained().unwrap();
    let observed = consumer.add(&second_output, &second_output).unwrap();
    let values = consumer.finish(&[observed]).unwrap();
    assert_eq!(values, vec![vec![12.0]], "second queued writer did not complete before cleanup returned");
}
