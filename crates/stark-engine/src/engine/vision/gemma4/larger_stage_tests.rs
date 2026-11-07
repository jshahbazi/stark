//! Private larger-tower diagnostics. Measurements do not relax full gates.

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
#[ignore]
fn gemma4_larger_tower_stage_reference_metal() {
    type M = crate::backends::metal::Metal;
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_FIXTURE_PATH").unwrap());
    let weights = PathBuf::from(std::env::var_os("STARK_GEMMA4_TOWER_PATH").unwrap());
    let manifest: Value = serde_json::from_slice(&read(&root, "manifest.json")).unwrap();
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    assert_eq!(manifest["model_revision"], PROFILES[2].revision);
    let engine = Engine::<M>::new().unwrap();
    let file = File::open(weights).unwrap();
    let loader = ParameterLoader::new(&file, &*engine.context).unwrap();
    let tree = loader.tree().subtree("vision");
    let tower = Gemma4VisionTower::<M>::new(&*engine.context, &tree, 5376)
        .unwrap_or_else(|_| panic!("private stage tower construction failed"));
    loader.tree().assert_all_tensors_validated().unwrap();
    let mut reports = Vec::new();
    for (case_index, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let patches = preprocess::prepare(&read(&root, case["image"].as_str().unwrap())).unwrap();
        let rows = (patches.grid[1] * patches.grid[2]) as usize;
        let stages = case["stages"].as_object().unwrap();
        let reference = |key: &str| {
            let record = &stages[key];
            let bytes = read(&root, record["file"].as_str().unwrap());
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
            floats(&bytes)
        };
        let mut captures = Vec::new();
        let mut commands = engine.context.create_command_buffer(Some("private Gemma larger stages"), None).unwrap();
        let mut observer =
            |tag: &'static str,
             layer: Option<u32>,
             value: &<M as Backend>::ScratchBuffer,
             command: &mut <<M as Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding|
             -> Result<(), <M as Backend>::Error> {
                let key = layer.map_or_else(|| tag.to_owned(), |layer| format!("{tag}_{layer}"));
                if !stages.contains_key(&key) {
                    return Ok(());
                }
                let all =
                    tag.starts_with("prepared_") || matches!(tag, "encoder_final" | "standardized" | "projector_input");
                let bytes = if all {
                    value.size()
                } else {
                    rows.min(8) * 1152 * 2
                };
                assert!(bytes > 0 && bytes <= value.size());
                let mut buffer = engine.context.create_buffer(bytes)?;
                command.encode_copy(value.subrange(..bytes), &mut buffer);
                captures.push((key, buffer));
                Ok(())
            };
        let _output = tower
            .encode_with_observer(&patches.patches, patches.grid, &mut commands, &mut observer)
            .unwrap_or_else(|_| panic!("private stage tower execution failed"));
        drop(observer);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let mut measured = Vec::new();
        let mut all_tails_zero = true;
        for (key, buffer) in captures {
            let full = (&buffer).as_slice::<bf16>();
            let actual = if key.starts_with("prepared_") {
                assert_eq!(full.len(), rows * 16 * 128);
                let query = key == "prepared_queries_0";
                for row in 0..rows {
                    for head in 0..16 {
                        let base = if query {
                            (head * rows + row) * 128
                        } else {
                            (row * 16 + head) * 128
                        };
                        all_tails_zero &= full[base + 72..base + 128].iter().all(|x| *x == bf16::ZERO);
                    }
                }
                (0..rows)
                    .flat_map(|row| {
                        let base = if query {
                            row * 128
                        } else {
                            row * 16 * 128
                        };
                        full[base..base + 72].iter().copied()
                    })
                    .collect::<Vec<_>>()
            } else {
                full.to_vec()
            };
            measured.push(json!({"stage":key,"metrics":compare(&actual,&reference(&key))}));
        }
        assert!(all_tails_zero, "logical72 padding contains nonzero tail values");
        let mut patch_oracles = Vec::new();
        let mut pool_oracle = Value::Null;
        #[cfg(backend = "cpu")]
        {
            let patch_weights = tree
                .leaf("patch_embedder.input_proj.weight")
                .unwrap()
                .validate(&[1152, 768], DataType::BF16)
                .unwrap()
                .read_slice::<bf16>()
                .unwrap();
            for streams in [4, 32] {
                let mut result = vec![0.0f32; rows.min(8) * 1152];
                crate::backends::cpu::kernel::vision_matmul_bf16(
                    patches.patches.as_ptr(),
                    patch_weights.as_ptr(),
                    result.as_mut_ptr(),
                    rows.min(8) as u32,
                    1152,
                    768,
                    768,
                    1,
                    streams,
                );
                let actual = result.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
                patch_oracles
                    .push(json!({"streams":streams,"metrics":compare(&actual,&reference("patch_projection"))}));
            }
            let source = reference("encoder_final").into_iter().map(bf16::from_f32).collect::<Vec<_>>();
            let bias =
                tree.leaf("std_bias").unwrap().validate(&[1152], DataType::BF16).unwrap().read_slice::<bf16>().unwrap();
            let scale = tree
                .leaf("std_scale")
                .unwrap()
                .validate(&[1152], DataType::BF16)
                .unwrap()
                .read_slice::<bf16>()
                .unwrap();
            let mut output = vec![bf16::ZERO; rows / 9 * 1152];
            crate::backends::cpu::kernel::gemma4_vision_pool_standardized_bf16(
                source.as_ptr(),
                bias.as_ptr(),
                scale.as_ptr(),
                output.as_mut_ptr(),
                patches.grid[2],
                patches.grid[1],
                1152,
            );
            pool_oracle = compare(&output, &reference("standardized"));
        }
        reports.push(json!({"case_index":case_index,"logical_head_dim":72,"padded_head_dim":128,
            "all_padding_tails_zero":all_tails_zero,"stages":measured,"patch_accumulation_oracles":patch_oracles,
            "cpu_pool_from_reference_encoder":pool_oracle}));
    }
    let report = json!({"scope":"larger-vision-stage-diagnostics","backend":"metal","cases":reports});
    std::fs::write(root.join("native-larger-stage-metrics.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}
