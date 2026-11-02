// SPDX-License-Identifier: Apache-2.0
//! Private fixture chronology/cleanup gates for synchronous layer-major chunks.

use stark_engine_macros::stark_test;

use super::*;

fn fixture() -> (std::path::PathBuf, Value, Vec<u32>) {
    let path = std::env::var_os("STARK_GLM_REFERENCE_FIXTURE_PATH")
        .map(std::path::PathBuf::from)
        .expect("GLM private fixture path required");
    let config =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).expect("private fixture config read failed"))
            .expect("private fixture config invalid");
    let tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read(path.join("tokens.json")).expect("private fixture tokens read failed"))
            .expect("private fixture tokens invalid");
    assert!(!tokens.is_empty(), "private fixture is empty");
    (path, config, tokens.into_iter().cycle().take(72).collect())
}

fn model(
    path: &std::path::Path,
    config: &Value,
) -> GlmModel {
    let mut model = GlmModel::new(Weights::open(path).expect("private fixture weights invalid"), config)
        .expect("private fixture admission failed");
    // Source admission remains unchanged. Synthetic capacity only, before lazy
    // attention construction, exercises growth beyond the initial 64 positions.
    model.config.max_context = 128;
    model
}

fn same(
    actual: &[f32],
    expected: &[f32],
) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[stark_test]
#[ignore]
fn glm_prefill_chunked_matches_scalar_state_tails_and_growth() {
    let (path, config, tokens) = fixture();
    for use_mma in [false, true] {
        for length in [1usize, 2, 3, 4, 5, 13, 65] {
            for start in [0usize, 3] {
                let mut scalar = model(&path, &config);
                let mut batched = model(&path, &config);
                for &token in &tokens[..start] {
                    scalar.ingest(token).expect("scalar initial state failed");
                    batched.ingest(token).expect("chunk initial state failed");
                }
                for &token in &tokens[start..start + length] {
                    scalar.ingest(token).expect("scalar prefix failed");
                }
                batched
                    .ingest_prefix_chunked(&tokens[start..start + length], &mut || Ok(()), use_mma)
                    .expect("chunked prefix failed");
                assert_eq!(batched.position, start + length);
                let expected = scalar.step(tokens[start + length]).expect("scalar final readout failed");
                let actual = batched.step(tokens[start + length]).expect("chunked final readout failed");
                // Tiny fixtures use legacy shapes. Exact bits gate scheduling,
                // mHC snapshots, recurrent stashes and causal MLA chronology.
                // Actual Q6 MMA is qualified separately against source/F64.
                assert!(same(&actual, &expected), "layer-major prefix changed fixture logits");
                if length == 65 {
                    let initial = 64usize.div_ceil(batched.config.index_pool) * batched.config.index_dim;
                    let attention = batched.decode.as_ref().unwrap().attention.iter().flatten().collect::<Vec<_>>();
                    assert!(!attention.is_empty(), "fixture lacks MLA growth coverage");
                    assert!(
                        attention.iter().all(|state| state.test_pool_keys().len > initial),
                        "chunked MLA did not grow"
                    );
                }
                batched.reset();
                batched
                    .ingest_prefix_chunked(&tokens[..start + length], &mut || Ok(()), use_mma)
                    .expect("chunked reset replay failed");
                let replay = batched.step(tokens[start + length]).expect("chunked replay readout failed");
                assert!(same(&replay, &expected), "chunked reset replay changed logits");
            }
        }
    }
}

#[stark_test]
#[ignore]
fn glm_prefill_chunked_validates_and_drains_cancelled_reservations() {
    let (path, config, tokens) = fixture();
    let mut reference = model(&path, &config);
    let expected = reference.step(tokens[0]).expect("reference readout failed");
    for use_mma in [false, true] {
        let mut invalid = model(&path, &config);
        let mut calls = 0;
        assert!(
            invalid
                .ingest_prefix_chunked(
                    &[u32::MAX],
                    &mut || {
                        calls += 1;
                        Ok(())
                    },
                    use_mma
                )
                .is_err()
        );
        assert_eq!(calls, 0);
        assert_eq!(invalid.position, 0);
        invalid.config.max_context = 2;
        assert!(
            invalid
                .ingest_prefix_chunked(
                    &tokens[..3],
                    &mut || {
                        calls += 1;
                        Ok(())
                    },
                    use_mma
                )
                .is_err()
        );
        assert_eq!(calls, 0);
        invalid.config.max_context = 128;
        assert!(same(&invalid.step(tokens[0]).unwrap(), &expected), "invalid chunk did not recover");
        for cancel in [1usize, 4, 5, 6, 9, 10, 11] {
            let mut cancelled = model(&path, &config);
            let mut calls = 0;
            assert!(
                cancelled
                    .ingest_prefix_chunked(
                        &tokens[..13],
                        &mut || {
                            calls += 1;
                            if calls == cancel {
                                Err(Error::Invalid("synthetic chunk cancellation"))
                            } else {
                                Ok(())
                            }
                        },
                        use_mma
                    )
                    .is_err()
            );
            assert_eq!(cancelled.position, 0);
            assert_eq!(calls, cancel);
            assert!(!cancelled.is_fenced(), "successful chunk drain fenced its model");
            assert!(same(&cancelled.step(tokens[0]).unwrap(), &expected), "chunk cancellation did not recover");
        }
    }
}

#[stark_test]
#[ignore]
fn glm_prefill_chunked_discards_failed_encoding_and_keeps_bank_admission() {
    let (path, config, tokens) = fixture();
    let mut reference = model(&path, &config);
    let expected = reference.step(tokens[0]).expect("reference readout failed");
    let mut failed = model(&path, &config);
    let limit = failed.config.swiglu_limit;
    // Test-only invalid arithmetic parameter fails after the first attention
    // reservation, before commit. Reset must discard all reserved MLA metadata
    // and allow an identical replay through the original valid configuration.
    failed.config.swiglu_limit = f32::NAN;
    assert!(failed.ingest_prefix_chunked(&tokens[..4], &mut || Ok(()), true).is_err());
    assert_eq!(failed.position, 0);
    assert!(!failed.is_fenced(), "discarded encoding incorrectly fenced its model");
    failed.config.swiglu_limit = limit;
    assert!(same(&failed.step(tokens[0]).unwrap(), &expected), "failed chunk encoding did not recover");

    let decode = failed.decode.as_ref().unwrap();
    let plan = decode.bounded_plan.as_ref().unwrap();
    let group = &plan.groups[0];
    let banks = failed.weights.encoder_indexed_group(&group.names, plan.budget).unwrap();
    let mut graph = new_encoder_graph(decode).unwrap();
    let input = graph.upload_into(&vec![0.0; 4 * failed.config.hidden]).unwrap();
    let expert = format!("blk.{}.ffn_gate_exps.weight", failed.config.dense_layers);
    assert!(
        banks.encode_same_weight_batch(&mut graph, &expert, &input, 4, true).is_err(),
        "dense batch API admitted a rank-three expert bank"
    );
    let ids = graph.upload_into(&[0.0]).unwrap();
    assert!(
        banks.encode_indexed_matvec(&mut graph, "output.weight", &ids, 1, &input, false).is_err(),
        "expert API admitted an unlisted static tensor"
    );
}
