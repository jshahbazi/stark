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

#[stark_test]
#[ignore]
fn gemma4_larger_full_attention_split_metal() {
    type M = crate::backends::metal::Metal;
    use crate::tests::helpers::{buffer_to_vec, create_buffer_with_data};

    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_FIXTURE_PATH").unwrap());
    let manifest: Value = serde_json::from_slice(&read(&root, "manifest.json")).unwrap();
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    assert_eq!(manifest["model_revision"], PROFILES[2].revision);
    assert_eq!(manifest["stage_focus"], "first-layer-full-attention");
    assert_eq!(manifest["stage_case_index"], 1);
    let case = &manifest["cases"][1];
    let stages = case["stages"].as_object().unwrap();
    assert_eq!(stages.len(), 4);
    let expected = |key: &str| {
        let record = &stages[key];
        let bytes = read(&root, record["file"].as_str().unwrap());
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
        floats(&bytes)
    };
    let image = read(&root, case["image"].as_str().unwrap());
    assert_eq!(format!("{:x}", Sha256::digest(&image)), case["image_sha256"].as_str().unwrap());
    let patches = preprocess::prepare(&image).unwrap();
    let rows = (patches.grid[1] * patches.grid[2]) as usize;
    let source_attention = expected("attention_0");
    assert_eq!(source_attention.len(), rows * 1152);
    let measure = |actual: &[bf16], reference: &[f32]| {
        assert_eq!(actual.len(), rows * 1152);
        assert_eq!(actual.len(), reference.len());
        let different = actual.iter().zip(reference).filter(|(a, b)| a.to_f32().to_bits() != b.to_bits()).count();
        let first_row = actual.chunks_exact(1152).zip(reference.chunks_exact(1152)).position(|(actual, reference)| {
            actual.iter().zip(reference).any(|(a, b)| a.to_f32().to_bits() != b.to_bits())
        });
        json!({"different_bf16_elements":different,"first_differing_patch_row":first_row,"metrics":compare(actual,reference)})
    };
    let engine = Engine::<M>::new().unwrap();
    let file = File::open(PathBuf::from(std::env::var_os("STARK_GEMMA4_TOWER_PATH").unwrap())).unwrap();
    let loader = ParameterLoader::new(&file, &*engine.context).unwrap();
    let tower = Gemma4VisionTower::<M>::new(&*engine.context, &loader.tree().subtree("vision"), 5376)
        .unwrap_or_else(|_| panic!("private full attention tower construction failed"));
    loader.tree().assert_all_tensors_validated().unwrap();
    let mut captures = Vec::new();
    let mut command = engine.context.create_command_buffer(None, None).unwrap();
    let mut observer =
        |tag: &'static str,
         layer: Option<u32>,
         value: &<M as Backend>::ScratchBuffer,
         commands: &mut <<M as Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding|
         -> Result<(), <M as Backend>::Error> {
            if layer != Some(0)
                || !matches!(tag, "prepared_queries" | "prepared_keys" | "prepared_values" | "attention")
            {
                return Ok(());
            }
            let mut output = engine.context.create_buffer(value.size())?;
            commands.encode_copy(value, &mut output);
            captures.push((tag, output));
            Ok(())
        };
    let _output = tower
        .encode_with_observer(&patches.patches, patches.grid, &mut command, &mut observer)
        .unwrap_or_else(|_| panic!("private full attention tower execution failed"));
    drop(observer);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let mut measured = Vec::new();
    for (tag, buffer) in captures {
        let values = buffer_to_vec::<M, bf16>(&buffer);
        let actual = if tag == "attention" {
            values
        } else {
            let mut actual = Vec::with_capacity(rows * 1152);
            for row in 0..rows {
                for head in 0..16 {
                    let base = if tag == "prepared_queries" {
                        (head * rows + row) * 128
                    } else {
                        (row * 16 + head) * 128
                    };
                    assert!(values[base + 72..base + 128].iter().all(|value| *value == bf16::ZERO));
                    actual.extend_from_slice(&values[base..base + 72]);
                }
            }
            actual
        };
        measured.push(json!({"stage":tag,"comparison":measure(&actual,&expected(&format!("{tag}_0")))}));
    }
    let mut canonical = Vec::new();
    for key in ["prepared_queries_0", "prepared_keys_0", "prepared_values_0"] {
        let reference = expected(key);
        assert_eq!(reference.len(), rows * 1152);
        let mut padded = vec![bf16::ZERO; rows * 16 * 128];
        for row in 0..rows {
            for head in 0..16 {
                let destination = if key == "prepared_queries_0" {
                    (head * rows + row) * 128
                } else {
                    (row * 16 + head) * 128
                };
                for channel in 0..72 {
                    padded[destination + channel] = bf16::from_f32(reference[(row * 16 + head) * 72 + channel]);
                }
            }
        }
        canonical.push(create_buffer_with_data::<M, bf16>(&engine.context, &padded));
    }
    let attention = super::super::flash::FlashAttention::<M>::new(&engine.context).unwrap();
    let mut command = engine.context.create_command_buffer(None, None).unwrap();
    let mut q = command.allocate_scratch_for_shape(&[16, rows as u32, 128], DataType::BF16).unwrap();
    let mut k = command.allocate_scratch_for_shape(&[rows as u32, 16, 128], DataType::BF16).unwrap();
    let mut v = command.allocate_scratch_for_shape(&[rows as u32, 16, 128], DataType::BF16).unwrap();
    command.encode_copy(&canonical[0], &mut q);
    command.encode_copy(&canonical[1], &mut k);
    command.encode_copy(&canonical[2], &mut v);
    let output = attention.encode(&q, &k, &v, rows as u32, rows as u32, 16, &mut command).unwrap();
    let mut readback = engine.context.create_buffer(rows * 1152 * 2).unwrap();
    command.encode_copy(&output, &mut readback);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<M, bf16>(&readback);
    let report = json!({"scope":"full-first-layer-attention-split","case_index":1,"backend":"metal",
        "tower_stages":measured,"same_canonical_input_attention":measure(&actual,&source_attention)});
    std::fs::write(root.join("native-full-attention-split.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}

fn checkpoint_flash_prefix<B: Backend>() {
    use crate::tests::helpers::{buffer_to_vec, create_buffer_with_data};
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_FIXTURE_PATH").unwrap());
    let manifest: Value = serde_json::from_slice(&read(&root, "manifest.json")).unwrap();
    assert_eq!(manifest["stage_focus"], "first-layer-full-attention");
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    assert_eq!(manifest["model_revision"], PROFILES[2].revision);
    let stages = &manifest["cases"][1]["stages"];
    let key_rows =
        manifest["cases"][1]["grid"][1].as_u64().unwrap() * manifest["cases"][1]["grid"][2].as_u64().unwrap();
    assert_eq!(key_rows, 2394);
    let context = B::Context::new().unwrap();
    let mut inputs = Vec::new();
    for key in ["prepared_queries_0", "prepared_keys_0", "prepared_values_0"] {
        let record = &stages[key];
        let bytes = read(&root, record["file"].as_str().unwrap());
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
        let source = floats(&bytes);
        assert_eq!(source.len(), key_rows as usize * 1152);
        let rows = if key == "prepared_queries_0" {
            8
        } else {
            key_rows as usize
        };
        let mut input = vec![bf16::ZERO; rows * 16 * 128];
        for row in 0..rows {
            for head in 0..16 {
                let destination = if key == "prepared_queries_0" {
                    (head * rows + row) * 128
                } else {
                    (row * 16 + head) * 128
                };
                for channel in 0..72 {
                    input[destination + channel] = bf16::from_f32(source[(row * 16 + head) * 72 + channel]);
                }
            }
        }
        inputs.push(create_buffer_with_data::<B, bf16>(&context, &input));
    }
    let record = &stages["attention_0"];
    let bytes = read(&root, record["file"].as_str().unwrap());
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
    let expected = floats(&bytes);
    let attention = super::super::flash::FlashAttention::<B>::new(&context).unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut q = command.allocate_scratch_for_shape(&[16, 8, 128], DataType::BF16).unwrap();
    let mut k = command.allocate_scratch_for_shape(&[key_rows as u32, 16, 128], DataType::BF16).unwrap();
    let mut v = command.allocate_scratch_for_shape(&[key_rows as u32, 16, 128], DataType::BF16).unwrap();
    command.encode_copy(&inputs[0], &mut q);
    command.encode_copy(&inputs[1], &mut k);
    command.encode_copy(&inputs[2], &mut v);
    let output = attention.encode(&q, &k, &v, 8, key_rows as u32, 16, &mut command).unwrap();
    let mut readback = context.create_buffer(8 * 1152 * 2).unwrap();
    command.encode_copy(&output, &mut readback);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, bf16>(&readback);
    let different =
        actual.iter().zip(&expected[..8 * 1152]).filter(|(a, b)| a.to_f32().to_bits() != b.to_bits()).count();
    let report = json!({"scope":"checkpoint-flash-query-prefix","backend":B::NAME,"query_rows":8,"heads":16,
        "different_bf16_elements":different,"metrics":compare(&actual,&expected[..8 * 1152])});
    std::fs::write(
        root.join(format!("native-prefix-flash-{}.json", B::NAME)),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
    assert_eq!(different, 0, "canonical all-head query prefix differs");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn gemma4_larger_checkpoint_flash_prefix_cpu() {
    checkpoint_flash_prefix::<crate::backends::cpu::Cpu>();
}

#[stark_test]
#[ignore]
fn gemma4_larger_checkpoint_flash_prefix_metal() {
    checkpoint_flash_prefix::<crate::backends::metal::Metal>();
}
