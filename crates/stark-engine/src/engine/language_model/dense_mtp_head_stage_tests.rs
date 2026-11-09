//! Ignored private synthetic head replay. Persist only inside the task's
//! ignored validation directory; test output contains no tensor payloads.
use std::{collections::BTreeMap, path::Path};

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::{common::BufferMut, metal::Metal},
    engine::Engine,
};

fn write_stage(
    root: &Path,
    name: &str,
    bytes: &[u8],
    shape: &[usize],
    dtype: &str,
) -> Value {
    std::fs::write(root.join(name), bytes).unwrap();
    json!({"file":name,"shape":shape,"dtype":dtype,"sha256":format!("{:x}",Sha256::digest(bytes))})
}

/// Exercise the complete score/shortlist/rescore seam, including buffer types.
#[ignore]
#[stark_test]
fn dense_mtp_coarse_head_shortlist_and_exact_rescore() {
    let artifact = std::path::PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private artifact"));
    let stages = std::path::PathBuf::from(std::env::var_os("STARK_DENSE_HEAD_STAGE_PATH").expect("set private stages"));
    let output =
        std::path::PathBuf::from(std::env::var_os("STARK_DENSE_COARSE_AUDIT_OUTPUT").expect("set private output"));
    let manifest: Value = serde_json::from_slice(&std::fs::read(stages.join("manifest.json")).unwrap()).unwrap();
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_language_model(&artifact).unwrap();
    let embedding = model.decoder.embedding();
    let context = &model.engine.context;
    let dim = embedding.model_dim() as usize;
    let vocab = embedding.vocab_size() as usize;
    let mut reports = Vec::new();
    for bits in [3, 4] {
        let coarse = embedding.coarse_draft_readout(context, bits).unwrap().unwrap();
        let mut rows = 0;
        let mut covered = 0;
        let mut top1 = 0;
        for stage in manifest["stages"].as_array().unwrap() {
            let record = &stage["normalized_hidden"];
            let bytes = std::fs::read(stages.join(record["file"].as_str().unwrap())).unwrap();
            assert!(format!("{:x}", Sha256::digest(&bytes)) == record["sha256"].as_str().unwrap());
            assert!(bytes.len().is_multiple_of(dim * 2));
            for row in bytes.chunks_exact(dim * 2) {
                let mut hidden = context.create_buffer(row.len()).unwrap();
                hidden.copyin(row);
                let mut command = context.create_command_buffer(None, None).unwrap();
                let scores = coarse.encode_scores(&hidden, &mut command).unwrap();
                let (ids, rescored) = coarse.encode(embedding, &hidden, &mut command).unwrap();
                let full = embedding
                    .encode_readout(
                        1,
                        &hidden,
                        vocab as u32,
                        None::<&<Metal as Backend>::ScratchBuffer>,
                        true,
                        &mut command,
                    )
                    .unwrap();
                command.end_encoding().submit().wait_until_completed().unwrap();
                let scores = scores.copyout::<f32>();
                let selected = ids.copyout::<u32>();
                let rescored = rescored.copyout::<bf16>();
                let full = full.copyout::<bf16>();
                let mut expected = (0..vocab).collect::<Vec<_>>();
                expected.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
                assert!(
                    selected.iter().copied().map(|x| x as usize).eq(expected[..64].iter().copied()),
                    "GPU shortlist differs from F32 reference ranking"
                );
                assert!(
                    selected.iter().zip(&rescored).all(|(&id, &score)| full[id as usize].to_bits() == score.to_bits()),
                    "gathered rescore differs from full target head"
                );
                expected
                    .sort_unstable_by(|&a, &b| full[b].to_f32().total_cmp(&full[a].to_f32()).then_with(|| a.cmp(&b)));
                covered += expected[..20].iter().filter(|&&id| selected.contains(&(id as u32))).count();
                top1 += usize::from(selected.contains(&(expected[0] as u32)));
                rows += 1;
            }
        }
        reports.push(json!({"bits":bits,"rows":rows,"top20_candidates_covered":covered,
            "top20_candidates_total":rows*20,"top1_covered_rows":top1,
            "shortlist_f32_ranking_exact":true,"rescore_bf16_exact":true}));
    }
    std::fs::write(output, serde_json::to_vec_pretty(&json!({"cases":reports})).unwrap()).unwrap();
}

// Requires the exact private native artifact and a private ignored output dir.
#[ignore]
#[stark_test]
fn dense_mtp_export_normalized_head_stages() {
    let artifact = std::path::PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private artifact"));
    let root = std::path::PathBuf::from(std::env::var_os("STARK_DENSE_HEAD_STAGE_PATH").expect("set private output"));
    std::fs::create_dir_all(&root).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let engine = Engine::<Metal>::new().unwrap();
    let mut model = engine.load_language_model(&artifact).unwrap();
    let reference_fusion = std::env::var_os("STARK_TEST_HEAD_REFERENCE_FUSION").is_some();
    let reference_linears = std::env::var_os("STARK_TEST_HEAD_REFERENCE_LINEARS").is_some();
    if reference_fusion {
        model.dense_mtp.as_mut().unwrap().validation_use_reference_fusion().unwrap();
    }
    let dim = model.decoder.embedding().model_dim() as usize;
    let vocab = model.decoder.embedding().vocab_size() as usize;
    let context = &model.engine.context;
    let raw = (0..4 * dim).map(|i| bf16::from_f32(((i * 17 % 103) as f32 - 51.0) * 0.013)).collect::<Vec<_>>();
    let mut source = context.create_buffer(raw.len() * 2).unwrap();
    source.copyin(&raw);
    let mut command = context.create_command_buffer(None, None).unwrap();
    let normalized = model.decoder.normalize_captured_output(&source, 4, &mut command).unwrap();
    let mut first = context.create_buffer(normalized.size()).unwrap();
    command.encode_copy(&normalized, &mut first);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let mut manifest = BTreeMap::new();
    manifest.insert("raw_target", write_stage(&root, "raw-target.bf16", &source.copyout::<u8>(), &[1, 4, dim], "bf16"));
    manifest.insert(
        "normalized_target",
        write_stage(&root, "normalized-target.bf16", &first.copyout::<u8>(), &[1, 4, dim], "bf16"),
    );
    let mut state = model.dense_mtp.as_ref().unwrap().create_state(8192, context).unwrap();
    let mut input = first;
    let mut stages = Vec::new();
    let mut input_stages = BTreeMap::new();
    for (stage, ids) in [&[17u32, 23, 31, 43][..], &[53u32][..], &[61u32][..]].into_iter().enumerate() {
        let rows = ids.len();
        let input_record =
            write_stage(&root, &format!("head-{stage}-input.bf16"), &input.copyout::<u8>(), &[1, rows, dim], "bf16");
        if stage == 0 {
            let mut command = context.create_command_buffer(None, None).unwrap();
            let tokens = command.allocate_constant_from_slice(ids).unwrap();
            let embedding = model.decoder.embedding().encode_lookup(&tokens, rows as u32, &mut command).unwrap();
            let mut saved_embedding = context.create_buffer(embedding.size()).unwrap();
            command.encode_copy(&embedding, &mut saved_embedding);
            let captured = model
                .dense_mtp
                .as_ref()
                .unwrap()
                .validation_fusion_stages(&embedding, &input, rows as u32, context.as_ref(), &mut command)
                .unwrap();
            drop(tokens);
            command.end_encoding().submit().wait_until_completed().unwrap();
            input_stages.insert(
                "embedding",
                write_stage(&root, "head-embedding.bf16", &saved_embedding.copyout::<u8>(), &[1, rows, dim], "bf16"),
            );
            for (name, buffer) in captured {
                input_stages.insert(
                    name,
                    write_stage(&root, &format!("head-{name}.bf16"), &buffer.copyout::<u8>(), &[1, rows, dim], "bf16"),
                );
            }
        }
        let pass = model.dense_predictor_pass(ids, &input, &mut state, true).unwrap();
        let hidden = pass.hidden.unwrap();
        let hidden_record = write_stage(
            &root,
            &format!("head-{stage}-normalized.bf16"),
            &hidden.copyout::<u8>(),
            &[1, rows, dim],
            "bf16",
        );
        let logits_record = write_stage(
            &root,
            &format!("head-{stage}-logits.f32"),
            bytemuck::cast_slice(&pass.logits),
            &[1, rows, vocab],
            "f32",
        );
        input = model.dense_last_hidden(&hidden, rows - 1).unwrap();
        stages
            .push(json!({"stage":stage,"input":input_record,"normalized_hidden":hidden_record,"logits":logits_record}));
    }
    let report = json!({"schema":"private-dense-qwen-normalized-head-stages-v1","model_dim":dim,"vocab":vocab,"target_head_policy":"trunk_output_norm","chain_head_policy":"predictor_output_norm","arrays":manifest,"stages":stages,"input_stages":input_stages,"test_only_reference_fusion":reference_fusion,"test_only_reference_linears":reference_linears});
    std::fs::write(root.join("manifest.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
