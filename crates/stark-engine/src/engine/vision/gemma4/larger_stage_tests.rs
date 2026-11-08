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
        let stages = case["stages"].as_object().unwrap();
        if stages.is_empty() {
            continue;
        }
        let patches = preprocess::prepare(&read(&root, case["image"].as_str().unwrap())).unwrap();
        let rows = (patches.grid[1] * patches.grid[2]) as usize;
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
                let all = tag.starts_with("prepared_")
                    || matches!(tag, "encoder_final" | "standardized" | "projector_input")
                    || (manifest["stage_focus"] == "first-six-full-blocks"
                        && tag == "block"
                        && layer.is_some_and(|layer| layer < 6));
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
            let expected = reference(&key);
            let different = actual.iter().zip(&expected).filter(|(a, b)| a.to_f32().to_bits() != b.to_bits()).count();
            measured
                .push(json!({"stage":key,"different_bf16_elements":different,"metrics":compare(&actual,&expected)}));
        }
        assert!(all_tails_zero, "logical72 padding contains nonzero tail values");
        let mut patch_oracles = Vec::new();
        let mut pool_oracle = Value::Null;
        #[cfg(backend = "cpu")]
        if stages.contains_key("patch_projection")
            && stages.contains_key("encoder_final")
            && stages.contains_key("standardized")
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
                crate::backends::cpu::kernel::matmul::vision::vision_matmul_bf16(
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
            crate::backends::cpu::kernel::attention::gemma4_vision::gemma4_vision_pool_standardized_bf16(
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
    let (case_index, focus_layer) = full_attention_focus(&manifest);
    let case = &manifest["cases"][case_index];
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
    let source_attention = expected(&format!("attention_{focus_layer}"));
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
            if layer != Some(focus_layer)
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
        measured.push(json!({"stage":tag,"comparison":measure(&actual,&expected(&format!("{tag}_{focus_layer}")))}));
    }
    let mut canonical = Vec::new();
    for key in ["prepared_queries", "prepared_keys", "prepared_values"] {
        let reference = expected(&format!("{key}_{focus_layer}"));
        assert_eq!(reference.len(), rows * 1152);
        let mut padded = vec![bf16::ZERO; rows * 16 * 128];
        for row in 0..rows {
            for head in 0..16 {
                let destination = if key == "prepared_queries" {
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
    let coordinates = actual
        .iter()
        .zip(&source_attention)
        .enumerate()
        .filter(|(_, (a, b))| a.to_f32().to_bits() != b.to_bits())
        .map(|(index, _)| [index / 1152, (index % 1152) / 72, index % 72])
        .collect::<Vec<_>>();
    let coordinate_bytes = serde_json::to_vec(&coordinates).unwrap();
    std::fs::write(root.join("native-full-attention-coordinates.json"), &coordinate_bytes).unwrap();
    let selected_rows = coordinates
        .iter()
        .map(|value| value[0])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let affected_replay_within_budget = selected_rows.len() <= 6
        && coordinates.iter().map(|value| (value[0], value[1])).collect::<std::collections::BTreeSet<_>>().len() <= 6;
    #[cfg(backend = "cpu")]
    let affected_cpu = if selected_rows.is_empty() || !affected_replay_within_budget {
        Value::Null
    } else {
        checkpoint_flash_rows::<crate::backends::cpu::Cpu>(&selected_rows, "affected")
    };
    #[cfg(not(backend = "cpu"))]
    let affected_cpu = Value::Null;
    let affected_metal = if selected_rows.is_empty() || !affected_replay_within_budget {
        Value::Null
    } else {
        checkpoint_flash_rows::<M>(&selected_rows, "affected")
    };
    let report = json!({"scope":"full-selected-layer-attention-split","case_index":case_index,"layer_index":focus_layer,"backend":"metal",
        "tower_stages":measured,"same_canonical_input_attention":measure(&actual,&source_attention),
        "coordinate_file_sha256":format!("{:x}",Sha256::digest(&coordinate_bytes)),"affected_query_rows":selected_rows.len(),
        "affected_replay_within_budget":affected_replay_within_budget,
        "affected_rows_cpu":affected_cpu,"affected_rows_metal":affected_metal});
    std::fs::write(root.join("native-full-attention-split.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}

fn full_attention_focus(manifest: &Value) -> (usize, u32) {
    let layer = manifest["stage_layer_index"].as_u64().unwrap_or(0);
    let case = manifest["stage_case_index"].as_u64().unwrap();
    assert!(layer < 27 && case < 3);
    assert_eq!(
        manifest["stage_focus"],
        if layer == 0 {
            "first-layer-full-attention"
        } else {
            "selected-layer-full-attention"
        }
    );
    (case as usize, layer as u32)
}

fn checkpoint_flash_rows<B: Backend>(
    selected_rows: &[usize],
    label: &str,
) -> Value {
    use crate::{
        backends::common::{Kernels, kernel::Gemma4VisionFlashReciprocalF32Kernel},
        tests::helpers::{buffer_to_vec, create_buffer_with_data},
    };
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_FIXTURE_PATH").unwrap());
    let manifest: Value = serde_json::from_slice(&read(&root, "manifest.json")).unwrap();
    let (case_index, focus_layer) = full_attention_focus(&manifest);
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    assert_eq!(manifest["model_revision"], PROFILES[2].revision);
    let case = &manifest["cases"][case_index];
    let stages = &case["stages"];
    let key_rows = case["grid"][1].as_u64().unwrap() * case["grid"][2].as_u64().unwrap();
    assert!((1..=2520).contains(&key_rows));
    assert!(
        !selected_rows.is_empty()
            && selected_rows.len() <= 8
            && selected_rows.iter().all(|row| *row < key_rows as usize)
    );
    assert!(matches!(label, "prefix" | "affected"));
    let query_rows = selected_rows.len();
    let context = B::Context::new().unwrap();
    let mut inputs = Vec::new();
    for key in ["prepared_queries", "prepared_keys", "prepared_values"] {
        let record = &stages[format!("{key}_{focus_layer}")];
        let bytes = read(&root, record["file"].as_str().unwrap());
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
        let source = floats(&bytes);
        assert_eq!(source.len(), key_rows as usize * 1152);
        let rows = if key == "prepared_queries" {
            query_rows
        } else {
            key_rows as usize
        };
        let mut input = vec![bf16::ZERO; rows * 16 * 128];
        for row in 0..rows {
            for head in 0..16 {
                let destination = if key == "prepared_queries" {
                    (head * rows + row) * 128
                } else {
                    (row * 16 + head) * 128
                };
                for channel in 0..72 {
                    let source_row = if key == "prepared_queries" {
                        selected_rows[row]
                    } else {
                        row
                    };
                    input[destination + channel] = bf16::from_f32(source[(source_row * 16 + head) * 72 + channel]);
                }
            }
        }
        inputs.push(create_buffer_with_data::<B, bf16>(&context, &input));
    }
    let record = &stages[format!("attention_{focus_layer}")];
    let bytes = read(&root, record["file"].as_str().unwrap());
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
    let source_attention = floats(&bytes);
    let expected = selected_rows
        .iter()
        .flat_map(|row| source_attention[*row * 1152..(*row + 1) * 1152].iter().copied())
        .collect::<Vec<_>>();
    let attention = super::super::flash::FlashAttention::<B>::new(&context).unwrap();
    let pairs = if label == "affected" {
        let coordinates: Vec<[usize; 3]> =
            serde_json::from_slice(&std::fs::read(root.join("native-full-attention-coordinates.json")).unwrap())
                .unwrap();
        coordinates
            .into_iter()
            .map(|[row, head, _]| {
                assert!(head < 16);
                (selected_rows.iter().position(|value| *value == row).unwrap(), head)
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    assert!(pairs.len() <= 6);
    let reciprocal = if pairs.is_empty() {
        None
    } else {
        Some(<B::Kernels as Kernels>::Gemma4VisionFlashReciprocalF32Kernel::new(&context).unwrap())
    };
    let mut traces = Vec::new();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut q = command.allocate_scratch_for_shape(&[16, query_rows as u32, 128], DataType::BF16).unwrap();
    let mut k = command.allocate_scratch_for_shape(&[key_rows as u32, 16, 128], DataType::BF16).unwrap();
    let mut v = command.allocate_scratch_for_shape(&[key_rows as u32, 16, 128], DataType::BF16).unwrap();
    command.encode_copy(&inputs[0], &mut q);
    command.encode_copy(&inputs[1], &mut k);
    command.encode_copy(&inputs[2], &mut v);
    let mut observer = |tag: &'static str,
                        key_start: u32,
                        value: &B::ScratchBuffer,
                        command: &mut <B::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding|
     -> Result<(), B::Error> {
        if pairs.is_empty() {
            return Ok(());
        }
        let (stride, bytes, bf16) = match tag {
            "qk" => (512 * 4, 512 * 4, false),
            "exp" => (512 * 2, 512 * 2, true),
            "state" => (3 * 4, 3 * 4, false),
            "pv" => (72 * 4, 72 * 4, false),
            _ => return Ok(()),
        };
        for (index, (row, head)) in pairs.iter().enumerate() {
            let offset = (head * query_rows + row) * stride;
            assert!(offset + bytes <= value.size());
            let mut output = context.create_buffer(bytes)?;
            command.encode_copy(value.subrange(offset..offset + bytes), &mut output);
            traces.push((tag, key_start, index, bf16, output));
        }
        if tag == "state" {
            let mut inverse = command.allocate_scratch_for_shape(&[16, query_rows as u32], DataType::F32)?;
            reciprocal.as_ref().unwrap().encode(value, &mut inverse, query_rows as u32, 16, command);
            for (index, (row, head)) in pairs.iter().enumerate() {
                let offset = (head * query_rows + row) * 4;
                let mut output = context.create_buffer(4)?;
                command.encode_copy(inverse.subrange(offset..offset + 4), &mut output);
                traces.push(("reciprocal", key_start, index, false, output));
            }
        }
        Ok(())
    };
    let output = attention
        .encode_with_observer(&q, &k, &v, query_rows as u32, key_rows as u32, 16, &mut command, &mut observer)
        .unwrap();
    drop(observer);
    let mut readback = context.create_buffer(query_rows * 1152 * 2).unwrap();
    command.encode_copy(&output, &mut readback);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, bf16>(&readback);
    let different = actual.iter().zip(&expected).filter(|(a, b)| a.to_f32().to_bits() != b.to_bits()).count();
    let scope = if label == "prefix" {
        "checkpoint-flash-query-prefix"
    } else {
        "checkpoint-flash-affected-query-rows"
    };
    let mut report = json!({"scope":scope,"backend":B::NAME,"query_rows":query_rows,"heads":16,
        "different_bf16_elements":different,"metrics":compare(&actual,&expected)});
    if !pairs.is_empty() {
        let mut records = Vec::new();
        let mut total = 0;
        for (tag, key_start, pair, bf16, buffer) in traces {
            let bytes = if bf16 {
                buffer_to_vec::<B, half::bf16>(&buffer)
                    .into_iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                buffer_to_vec::<B, f32>(&buffer).into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()
            };
            total += bytes.len();
            assert!(total <= 256 * 1024, "private trace byte cap exceeded");
            let name = format!(
                "flash-trace-{}-{key_start}-{pair}-{tag}.{}",
                B::NAME,
                if bf16 {
                    "bf16"
                } else {
                    "f32"
                }
            );
            std::fs::write(root.join(&name), &bytes).unwrap();
            records.push(json!({"file":name,"pair":pair,"key_start":key_start,"stage":tag,"dtype":if bf16{"bf16"}else{"f32"},"sha256":format!("{:x}",Sha256::digest(&bytes))}));
        }
        let metadata = json!({"scope":"private-affected-flash-f32-trace","backend":B::NAME,"pairs":pairs.iter().map(|(row,head)|json!({"image_patch_row":selected_rows[*row],"head":head})).collect::<Vec<_>>(),"bytes":total,"records":records});
        let bytes = serde_json::to_vec_pretty(&metadata).unwrap();
        std::fs::write(root.join(format!("flash-trace-{}.json", B::NAME)), &bytes).unwrap();
        report["trace_manifest_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        report["trace_bytes"] = json!(total);
    }
    std::fs::write(
        root.join(format!("native-{label}-flash-{}.json", B::NAME)),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
    report
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn gemma4_larger_checkpoint_flash_prefix_cpu() {
    let report = checkpoint_flash_rows::<crate::backends::cpu::Cpu>(&[0, 1, 2, 3, 4, 5, 6, 7], "prefix");
    assert_eq!(report["different_bf16_elements"], 0, "canonical CPU prefix differs");
}

#[stark_test]
#[ignore]
fn gemma4_larger_checkpoint_flash_prefix_metal() {
    let report = checkpoint_flash_rows::<crate::backends::metal::Metal>(&[0, 1, 2, 3, 4, 5, 6, 7], "prefix");
    assert_eq!(report["different_bf16_elements"], 0, "canonical Metal prefix differs");
}
