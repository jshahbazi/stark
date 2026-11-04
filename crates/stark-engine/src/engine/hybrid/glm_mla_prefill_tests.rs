// SPDX-License-Identifier: Apache-2.0
//! Private fixture checks for independent projection batching around causality.

use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
#[ignore]
fn mla_prefill_batch_matches_scalar_tails_growth_and_reset() {
    assert!(std::env::var_os("STARK_GLM53_MLA_PREFILL_BATCH").is_some_and(|v| v == "1"), "MLA batch selector required");
    let path = std::env::var_os("STARK_GLM_REFERENCE_FIXTURE_PATH")
        .map(std::path::PathBuf::from)
        .expect("private fixture path required");
    let config: Value = serde_json::from_slice(
        &std::fs::read(path.join("config.json")).expect("private fixture configuration unavailable"),
    )
    .expect("private fixture configuration invalid");
    let tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read(path.join("tokens.json")).expect("private fixture tokens unavailable"))
            .expect("private fixture tokens invalid");
    assert!(!tokens.is_empty());
    let tokens: Vec<_> = tokens.into_iter().cycle().take(140).collect();
    let create = || {
        let mut model = GlmModel::new(Weights::open(&path).expect("private fixture weights invalid"), &config)
            .expect("private fixture model rejected");
        model.config.max_context = 256;
        model
    };
    let same = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
    for length in [1, 3, 7, 9, 17, 65, 129] {
        let mut scalar = create();
        let mut batch = create();
        for &token in &tokens[..length] {
            scalar.ingest(token).expect("scalar ingestion failed");
        }
        batch.ingest_prefix_chunked(&tokens[..length], &mut || Ok(()), true, 8).expect("batched ingestion failed");
        let expected = scalar.step(tokens[length]).expect("scalar readout failed");
        let actual = batch.step(tokens[length]).expect("batched readout failed");
        assert!(same(&actual, &expected), "MLA batch changed causal fixture readout");
        batch.reset();
        batch.ingest_prefix_chunked(&tokens[..length], &mut || Ok(()), true, 8).expect("reset ingestion failed");
        assert!(
            same(&batch.step(tokens[length]).expect("reset readout failed"), &expected),
            "MLA batch reset changed readout"
        );
    }
}
