//! Private same-input larger MLP diagnostics; no feature-gate changes.

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::tests::helpers::{buffer_to_vec, create_buffer_with_data};

#[stark_test]
#[ignore]
fn gemma4_larger_mlp_down_split_metal() {
    type M = crate::backends::metal::Metal;
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_FIXTURE_PATH").unwrap());
    let manifest: Value = serde_json::from_slice(&read(&root, "manifest.json")).unwrap();
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    assert_eq!(manifest["model_revision"], PROFILES[2].revision);
    assert_eq!(manifest["stage_focus"], "first-layer-mlp-down");
    assert_eq!(manifest["stage_case_index"], 1);
    let case = &manifest["cases"][1];
    let stages = case["stages"].as_object().unwrap();
    assert_eq!(stages.len(), 7);
    let source = |key: &str| {
        let record = &stages[key];
        let bytes = read(&root, record["file"].as_str().unwrap());
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
        floats(&bytes)
    };
    let measure = |actual: &[bf16], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        let different = actual.iter().zip(expected).filter(|(a, b)| a.to_f32().to_bits() != b.to_bits()).count();
        json!({"different_bf16_elements":different,"metrics":compare(actual,expected)})
    };
    let image = read(&root, case["image"].as_str().unwrap());
    assert_eq!(format!("{:x}", Sha256::digest(&image)), case["image_sha256"].as_str().unwrap());
    let patches = preprocess::prepare(&image).unwrap();
    let rows = patches.grid[1] * patches.grid[2];
    let engine = Engine::<M>::new().unwrap();
    let file = File::open(PathBuf::from(std::env::var_os("STARK_GEMMA4_TOWER_PATH").unwrap())).unwrap();
    let loader = ParameterLoader::new(&file, &*engine.context).unwrap();
    let tree = loader.tree().subtree("vision");
    let tower = Gemma4VisionTower::<M>::new(&*engine.context, &tree, 5376)
        .unwrap_or_else(|_| panic!("private MLP tower construction failed"));
    loader.tree().assert_all_tensors_validated().unwrap();
    let mut captures = Vec::new();
    let mut commands = engine.context.create_command_buffer(None, None).unwrap();
    let mut observer =
        |tag: &'static str,
         layer: Option<u32>,
         value: &<M as Backend>::ScratchBuffer,
         command: &mut <<M as Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding|
         -> Result<(), <M as Backend>::Error> {
            if layer != Some(0) {
                return Ok(());
            }
            let key = format!("{tag}_0");
            let Some(record) = stages.get(&key) else {
                return Ok(());
            };
            let shape = record["shape"].as_array().unwrap();
            assert_eq!(shape.len(), 2);
            let stage_rows = shape[0].as_u64().unwrap() as usize;
            let width = shape[1].as_u64().unwrap() as usize;
            assert!(matches!(width, 1152 | 4304) && (stage_rows == 8 || stage_rows == rows as usize));
            let bytes = stage_rows * width * 2;
            assert!(bytes <= value.size());
            let mut buffer = engine.context.create_buffer(bytes)?;
            command.encode_copy(value.subrange(..bytes), &mut buffer);
            captures.push((key, buffer));
            Ok(())
        };
    let _output = tower
        .encode_with_observer(&patches.patches, patches.grid, &mut commands, &mut observer)
        .unwrap_or_else(|_| panic!("private MLP tower execution failed"));
    drop(observer);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let mut measured = Vec::new();
    for (key, buffer) in captures {
        measured.push(json!({"stage":key,"comparison":measure(&buffer_to_vec::<M,bf16>(&buffer),&source(&key))}));
    }
    let canonical = source("mlp_down_input_0").into_iter().map(bf16::from_f32).collect::<Vec<_>>();
    assert_eq!(canonical.len(), rows as usize * 4304);
    let canonical = create_buffer_with_data::<M, bf16>(&engine.context, &canonical);
    let linear = super::super::tower::Linear::<M>::new(
        &*engine.context,
        &tree.subtree("encoder.layers.0.mlp.down_proj"),
        4304,
        1152,
        false,
    )
    .unwrap_or_else(|_| panic!("private MLP projection construction failed"));
    let mut commands = engine.context.create_command_buffer(None, None).unwrap();
    let mut input = commands.allocate_scratch_for_shape(&[rows, 4304], DataType::BF16).unwrap();
    commands.encode_copy(&canonical, &mut input);
    let output = linear.encode(&input, rows, &mut commands).unwrap();
    let mut readback = engine.context.create_buffer(rows as usize * 1152 * 2).unwrap();
    commands.encode_copy(&output, &mut readback);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let report = json!({"scope":"same-canonical-input-mlp-down-split","backend":"metal","case_index":1,
        "tower_stages":measured,"same_canonical_input_down":measure(&buffer_to_vec::<M,bf16>(&readback),&source("mlp_down_output_0"))});
    std::fs::write(root.join("native-mlp-down-split.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}
