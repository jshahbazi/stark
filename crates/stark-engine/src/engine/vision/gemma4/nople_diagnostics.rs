//! Opt-in private layerwise reference capture; vectors never enter logs.

use stark_engine_macros::stark_test;

use super::*;
use crate::{
    config::decoder::DecoderConfig,
    encodable_block::{batch_topology::BatchTopology, decoder::Decoder},
};

#[stark_test]
#[ignore]
fn gemma_nople_stagewise_cpu_fixture() {
    let root = std::path::PathBuf::from(std::env::var_os("STARK_GEMMA4_DENSE_FIXTURE_PATH").unwrap());
    let path = std::path::PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let config: DecoderConfig = serde_json::from_value(raw["decoder_config"].clone()).unwrap();
    assert_eq!(manifest["reference_profile"], "transformers-canonical");
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    let engine = Engine::<crate::backends::cpu::Cpu>::new().unwrap();
    let file = File::open(path.join("model.safetensors")).unwrap();
    let loader = ParameterLoader::new(&file, &*engine.context).unwrap();
    let decoder = Decoder::new(&*engine.context, &config, &loader.tree().subtree("decoder"), DataType::BF16).unwrap();
    loader.tree().assert_all_tensors_validated().unwrap();
    let indices = (0..config.transformer_config.layer_configs.len() as u32).collect::<Vec<_>>();
    let mut reports = Vec::new();
    for (case_index, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let rows = tokens.len() as u32;
        let nodes = (0..rows)
            .map(|index| crate::backends::common::gpu_types::trie::TrieNode {
                trie_start: index,
                trie_end: rows - 1,
                height: index,
            })
            .collect::<Vec<_>>();
        let batch = BatchTopology::new(&nodes, true);
        let mut state = decoder.create_empty_state(Some(128), &*engine.context).unwrap();
        state.prepare(0, rows).unwrap();
        let mut commands = engine.context.create_command_buffer(Some("private no-PLE stages"), None).unwrap();
        let token_ids = commands.allocate_constant_from_slice(&tokens).unwrap();
        let output = decoder.encode(&token_ids, &batch, None, Some(&indices), &mut state, &mut commands).unwrap();
        drop(token_ids);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let mut layers = Vec::new();
        for (layer, feature) in output.hidden_features.unwrap().iter().enumerate() {
            let relative = case["hidden_features"][layer].as_str().unwrap();
            assert!(!relative.contains('/') && !relative.contains('\\'));
            let bytes = std::fs::read(root.join(relative)).unwrap();
            let reference =
                bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
            let native = feature.as_slice::<bf16>();
            assert_eq!(native.len(), reference.len());
            let n = native.len() as f64;
            let error = (native
                .iter()
                .zip(&reference)
                .map(|(a, b)| (f64::from(a.to_f32()) - f64::from(*b)).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            let scale = (reference.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>() / n).sqrt();
            let different = native.iter().zip(&reference).filter(|(a, b)| a.to_f32() != **b).count();
            let prefix = (rows.min(16) * config.transformer_config.model_dim) as usize;
            let prefix_different =
                native[..prefix].iter().zip(&reference[..prefix]).filter(|(a, b)| a.to_f32() != **b).count();
            let first_different_row = native
                .iter()
                .zip(&reference)
                .position(|(a, b)| a.to_f32() != *b)
                .map(|index| index / config.transformer_config.model_dim as usize);
            layers.push(serde_json::json!({"layer":layer,"relative_rms":error/scale.max(f64::MIN_POSITIVE),
                "different_elements":different,"elements":native.len(),"first_different_row":first_different_row,
                "prefix16_different_elements":prefix_different,"suffix_different_elements":different-prefix_different}));
        }
        reports.push(serde_json::json!({"case_index":case_index,"layers":layers}));
    }
    let report = serde_json::json!({"scope":"diagnostic-layerwise-residual","backend":"cpu","cases":reports});
    std::fs::write(root.join("native-layerwise-metrics.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}
