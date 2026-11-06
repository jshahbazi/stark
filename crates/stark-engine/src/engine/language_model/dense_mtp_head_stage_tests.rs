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
    let model = engine.load_language_model(&artifact).unwrap();
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
    for (stage, ids) in [&[17u32, 23, 31, 43][..], &[53u32][..], &[61u32][..]].into_iter().enumerate() {
        let rows = ids.len();
        let input_record =
            write_stage(&root, &format!("head-{stage}-input.bf16"), &input.copyout::<u8>(), &[1, rows, dim], "bf16");
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
    let report = json!({"schema":"private-dense-qwen-normalized-head-stages-v1","model_dim":dim,"vocab":vocab,"target_head_policy":"trunk_output_norm","chain_head_policy":"predictor_output_norm","arrays":manifest,"stages":stages});
    std::fs::write(root.join("manifest.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
