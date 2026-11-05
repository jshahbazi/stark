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
    (path, config, tokens.into_iter().cycle().take(2060).collect())
}

fn model(
    path: &std::path::Path,
    config: &Value,
) -> GlmModel {
    let mut model = GlmModel::new(Weights::open(path).expect("private fixture weights invalid"), config)
        .expect("private fixture admission failed");
    // Source admission remains unchanged. Synthetic capacity only, before lazy
    // attention construction, exercises growth beyond the initial 64 positions.
    model.config.max_context = 3072;
    model
}

fn same(
    actual: &[f32],
    expected: &[f32],
) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[stark_test]
fn glm_prefill_chunk_limits_keep_default_four_and_greedy_eight_tails() {
    use std::ffi::OsStr;
    assert_eq!(chunk_limit(None).unwrap(), 4);
    assert_eq!(chunk_limit(Some(OsStr::new("4"))).unwrap(), 4);
    assert_eq!(chunk_limit(Some(OsStr::new("8"))).unwrap(), 8);
    for invalid in ["", "0", "08", "16", "-1"] {
        assert!(chunk_limit(Some(OsStr::new(invalid))).is_err());
    }
    let sizes = |tokens, maximum| {
        let first = eight_token_prefix(tokens, maximum).unwrap();
        let dummy = vec![0u8; tokens];
        dummy[..first].chunks(8).chain(dummy[first..].chunks(4)).map(<[u8]>::len).collect::<Vec<_>>()
    };
    assert_eq!(sizes(13, 4), vec![4, 4, 4, 1]);
    assert_eq!(sizes(7, 8), vec![4, 3]);
    assert_eq!(sizes(8, 8), vec![8]);
    assert_eq!(sizes(9, 8), vec![8, 1]);
    assert_eq!(sizes(13, 8), vec![8, 4, 1]);
    assert_eq!(sizes(17, 8), vec![8, 8, 1]);
    assert_eq!(sizes(65, 8), [vec![8; 8], vec![1]].concat());
    assert!(eight_token_prefix(13, 16).is_err());
}

fn workspace_config() -> Config {
    Config {
        hidden: 4096,
        heads: 32,
        kda_heads: 48,
        kda_dim: 128,
        conv_kernel: 4,
        layer_types: (0..45).map(|layer| layer % 4 != 3).collect(),
        dense_layers: 3,
        intermediate: 11_008,
        expert_intermediate: 2048,
        experts: 288,
        active_experts: 8,
        shared_experts: 1,
        hc: 4,
        hc_iters: 20,
        rms_eps: 1e-5,
        hc_eps: 1e-6,
        decay_lower_bound: -5.0,
        q_rank: 512,
        kv_rank: 512,
        qk_dim: 256,
        value_dim: 256,
        index_heads: 4,
        index_dim: 128,
        index_pool: 4,
        index_topk: 2048,
        swiglu_limit: 10.0,
        routed_scale: 2.5,
        vocab: 154_880,
        max_context: 4096,
    }
}

#[stark_test]
fn glm_wide_tiles_are_opt_in_and_keep_the_absolute_prefix_grid() {
    use std::ffi::OsStr;
    assert_eq!(wide_limit(None).unwrap(), None);
    for maximum in [16, 32, 64, 128] {
        assert_eq!(wide_limit(Some(OsStr::new(&maximum.to_string()))).unwrap(), Some(maximum));
        for start in [0, 3, 1000, 1024, 1041] {
            let mut position = start;
            let end = start + 2217;
            while position < end {
                let count = next_tile(position, end - position, maximum);
                assert!(count > 0 && count <= maximum);
                assert!(count <= PREFIX_GRID - position % PREFIX_GRID);
                assert!(count < 4 || count.is_power_of_two());
                position += count;
            }
            assert_eq!(position, end);
        }
        let complete: Vec<_> = (0..1024 / maximum).map(|_| maximum).collect();
        assert_eq!(complete.iter().sum::<usize>(), PREFIX_GRID);
    }
    for invalid in ["", "0", "4", "8", "016", "256", "-1"] {
        assert!(wide_limit(Some(OsStr::new(invalid))).is_err());
    }
}

#[stark_test]
fn glm_wide_workspace_admits_groups_and_charges_external_mla_scores() {
    let mut config = workspace_config();
    for tokens in [1, 4, 8, 16, 32, 64, 128] {
        let groups = wide_groups(&config, tokens, 1024, usize::MAX).unwrap();
        assert_eq!(groups.first().unwrap().start, 0);
        assert_eq!(groups.last().unwrap().end, config.layer_types.len());
        assert!(groups.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(groups.iter().all(|group| group.scratch + group.external <= WIDE_OWNER_BYTES));
        assert!(groups.iter().any(|group| group.external > 0));
        assert!(2 * (WIDE_OWNER_BYTES + WIDE_CARRY_BYTES) <= 2 * 512 * MIB);
    }
    let (_, early) = layer_workspace(&config, 3, 64, 1024).unwrap();
    config.max_context = 1_048_576;
    let (_, late) = layer_workspace(&config, 3, 64, config.max_context).unwrap();
    assert!(late > early);
    assert!(wide_groups(&config, 128, config.max_context, usize::MAX).is_err());
    assert!(wide_groups(&config, 0, 1024, usize::MAX).is_err());
    assert!(wide_groups(&config, 129, 1024, usize::MAX).is_err());
    assert!(wide_groups(&config, 16, 15, usize::MAX).is_err());
    assert!(wide_groups(&config, 16, 1024, 0).is_err());
    let forced = wide_groups(&config, 16, 1024, 1).unwrap();
    assert_eq!(forced.len(), config.layer_types.len());
    assert!(forced.iter().all(|group| group.end == group.start + 1));
    config.hidden = 32768;
    config.hc = 8;
    assert!(wide_groups(&config, 128, 1024, usize::MAX).is_err());
}

#[stark_test]
#[ignore]
fn glm_prefill_chunked_matches_scalar_state_tails_and_growth() {
    compare_prefixes(&[1, 2, 3, 4, 5, 13, 65], 4);
}

#[stark_test]
#[ignore]
fn glm_prefill_chunked_eight_matches_scalar_state_tails_and_growth() {
    compare_prefixes(&[7, 8, 9, 13, 17, 65], 8);
}

// Private fixture tests run serially. Restore the selector even when an
// assertion unwinds, so a following default-path test cannot inherit it.
fn with_wide(
    maximum: usize,
    test: impl FnOnce(),
) {
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var("STARK_GLM53_PREFILL_WIDE", value),
                    None => std::env::remove_var("STARK_GLM53_PREFILL_WIDE"),
                }
            }
        }
    }
    let _restore = Restore(std::env::var_os("STARK_GLM53_PREFILL_WIDE"));
    unsafe {
        std::env::set_var("STARK_GLM53_PREFILL_WIDE", maximum.to_string());
    }
    test();
}

#[stark_test]
#[ignore]
fn glm_prefill_wide_matches_scalar_state_tails_growth_and_prefix_grid() {
    for maximum in [16, 32, 64, 128] {
        with_wide(maximum, || compare_prefixes(&[maximum - 1, maximum, maximum + 1], 8));
    }
    with_wide(128, || compare_prefixes(&[1025], 8));
}

#[stark_test]
#[ignore]
fn glm_prefill_wide_cancellation_and_encoding_failure_reset_completed_tiles() {
    let (path, config, tokens) = fixture();
    let mut reference = model(&path, &config);
    let expected = reference.step(tokens[0]).unwrap();
    for maximum in [16, 32, 64, 128] {
        with_wide(maximum, || {
            for cancel in [1, maximum, maximum + 1, maximum + 2, maximum + 3] {
                let mut cancelled = model(&path, &config);
                let mut calls = 0;
                assert!(
                    cancelled
                        .ingest_prefix_chunked(
                            &tokens[..2 * maximum + 1],
                            &mut || {
                                calls += 1;
                                if calls == cancel {
                                    Err(Error::Invalid("synthetic wide cancellation"))
                                } else {
                                    Ok(())
                                }
                            },
                            true,
                            8
                        )
                        .is_err()
                );
                assert_eq!(calls, cancel);
                assert_eq!(cancelled.position, 0);
                assert!(!cancelled.is_fenced(), "wide cancellation failed to drain");
                assert!(same(&cancelled.step(tokens[0]).unwrap(), &expected));
            }
            let mut failed = model(&path, &config);
            let limit = failed.config.swiglu_limit;
            failed.config.swiglu_limit = f32::NAN;
            assert!(failed.ingest_prefix_chunked(&tokens[..maximum], &mut || Ok(()), true, 8).is_err());
            assert_eq!(failed.position, 0);
            assert!(!failed.is_fenced());
            failed.config.swiglu_limit = limit;
            assert!(same(&failed.step(tokens[0]).unwrap(), &expected));
        });
    }
}

#[stark_test]
#[ignore]
fn glm_prefill_wide_group_failures_drain_and_recheck_after_wait() {
    use super::super::super::profile;

    struct RestoreProfile(bool);
    impl Drop for RestoreProfile {
        fn drop(&mut self) {
            profile::enable(self.0);
        }
    }
    let _restore = RestoreProfile(profile::enabled());
    let (path, config, tokens) = fixture();
    let mut reference = model(&path, &config);
    let expected = reference.step(tokens[0]).unwrap();
    let calls = |name| profile::snapshot().get(name).map_or(0, |counter| counter.calls);
    with_wide(16, || {
        // Force actual ownership boundaries even for the tiny fixture. Cancel
        // once the first wait has completed: no second command may commit.
        let mut cancelled = model(&path, &config);
        assert!(cancelled.states.len() >= 2, "group fixture requires two layers");
        profile::reset();
        profile::enable(true);
        assert!(
            cancelled
                .ingest_prefix_chunked_grouped(
                    &tokens[..16],
                    &mut || {
                        if calls("native.metal.wait") > 0 {
                            Err(Error::Invalid("synthetic cancellation observed after wait"))
                        } else {
                            Ok(())
                        }
                    },
                    true,
                    8,
                    1
                )
                .is_err()
        );
        assert_eq!(calls("native.metal.submit"), 1, "next group committed after cancellation");
        assert_eq!(calls("native.metal.wait"), 1);
        assert_eq!(cancelled.position, 0);
        assert!(!cancelled.is_fenced());
        assert!(same(&cancelled.step(tokens[0]).unwrap(), &expected));

        // Cancel after group two is fully encoded while group one remains
        // submitted. The unsubmitted owner drops before cleanup waits.
        let mut pending = model(&path, &config);
        let mut callbacks = 0;
        profile::reset();
        assert!(
            pending
                .ingest_prefix_chunked_grouped(
                    &tokens[..16],
                    &mut || {
                        callbacks += 1;
                        if callbacks == 21 {
                            Err(Error::Invalid("synthetic pending group cancellation"))
                        } else {
                            Ok(())
                        }
                    },
                    true,
                    8,
                    1
                )
                .is_err()
        );
        assert_eq!(calls("native.metal.submit"), 1);
        assert_eq!(calls("native.metal.wait"), 1);
        assert_eq!(pending.position, 0);
        assert!(!pending.is_fenced());
        assert!(same(&pending.step(tokens[0]).unwrap(), &expected));

        // A deliberately mismatched second-layer state fails after encoding
        // its control prepass. Its predecessor must still drain before reset.
        let mut failed = model(&path, &config);
        let replacement = if failed.config.layer_types[1] {
            LayerState::Attention(AttentionState::default())
        } else {
            LayerState::Kda(KdaState::new(&failed.config))
        };
        let original = std::mem::replace(&mut failed.states[1], replacement);
        profile::reset();
        assert!(failed.ingest_prefix_chunked_grouped(&tokens[..16], &mut || Ok(()), true, 8, 1).is_err());
        assert_eq!(calls("native.metal.submit"), 1);
        assert_eq!(calls("native.metal.wait"), 1);
        assert_eq!(failed.position, 0);
        assert!(!failed.is_fenced());
        failed.states[1] = original;
        assert!(same(&failed.step(tokens[0]).unwrap(), &expected));
    });
}

fn compare_prefixes(
    lengths: &[usize],
    maximum_chunk: usize,
) {
    let (path, config, tokens) = fixture();
    for use_mma in [false, true] {
        for &length in lengths {
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
                    .ingest_prefix_chunked(&tokens[start..start + length], &mut || Ok(()), use_mma, maximum_chunk)
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
                    .ingest_prefix_chunked(&tokens[..start + length], &mut || Ok(()), use_mma, maximum_chunk)
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
        for maximum_chunk in [4, 8] {
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
                        use_mma,
                        maximum_chunk
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
                        use_mma,
                        maximum_chunk
                    )
                    .is_err()
            );
            assert_eq!(calls, 0);
            invalid.config.max_context = 128;
            assert!(same(&invalid.step(tokens[0]).unwrap(), &expected), "invalid chunk did not recover");
            for cancel in [1usize, 4, 5, 6, 9, 10, 11, 14, 15] {
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
                            use_mma,
                            maximum_chunk
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
}

#[stark_test]
#[ignore]
fn glm_prefill_chunked_discards_failed_encoding_and_keeps_bank_admission() {
    let (path, config, tokens) = fixture();
    let mut reference = model(&path, &config);
    let expected = reference.step(tokens[0]).expect("reference readout failed");
    for maximum_chunk in [4, 8] {
        let mut failed = model(&path, &config);
        let limit = failed.config.swiglu_limit;
        // Test-only invalid arithmetic parameter fails after the first attention
        // reservation, before commit. Reset must discard all reserved MLA metadata
        // and allow an identical replay through the original valid configuration.
        failed.config.swiglu_limit = f32::NAN;
        assert!(failed.ingest_prefix_chunked(&tokens[..maximum_chunk], &mut || Ok(()), true, maximum_chunk).is_err());
        assert_eq!(failed.position, 0);
        assert!(!failed.is_fenced(), "discarded encoding incorrectly fenced its model");
        failed.config.swiglu_limit = limit;
        assert!(same(&failed.step(tokens[0]).unwrap(), &expected), "failed chunk encoding did not recover");

        let decode = failed.decode.as_ref().unwrap();
        let plan = decode.bounded_plan.as_ref().unwrap();
        let group = &plan.groups[0];
        let banks = failed.weights.encoder_indexed_group(&group.names, plan.budget).unwrap();
        let mut graph = new_encoder_graph(decode).unwrap();
        let input = graph.upload_into(&vec![0.0; maximum_chunk * failed.config.hidden]).unwrap();
        let expert = format!("blk.{}.ffn_gate_exps.weight", failed.config.dense_layers);
        assert!(
            banks.encode_same_weight_batch(&mut graph, &expert, &input, maximum_chunk, true).is_err(),
            "dense batch API admitted a rank-three expert bank"
        );
        let ids = graph.upload_into(&[0.0]).unwrap();
        assert!(
            banks.encode_indexed_matvec(&mut graph, "output.weight", &ids, 1, &input, false).is_err(),
            "expert API admitted an unlisted static tensor"
        );
    }
}
