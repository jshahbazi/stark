//! Private same-input block diagnostics. These do not replace whole-tower gates.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    fs::File,
    path::{Path, PathBuf},
};

use half::bf16;
use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::tower::{VisionAttentionStage, VisionBlockStage, VisionMergerStage, VisionTower, VisionTowerConfig};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
    },
    parameters::ParameterLoader,
};

fn bytes(path: impl AsRef<Path>) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|_| panic!("private local-step artifact unavailable"))
}

fn metadata(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&bytes(path)).unwrap_or_else(|_| panic!("invalid private local-step metadata"))
}

fn private_directory(path: &Path) {
    std::fs::create_dir_all(path).unwrap_or_else(|_| panic!("private diagnostic output setup failed"));
    #[cfg(unix)]
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .unwrap_or_else(|_| panic!("private diagnostic permissions failed"));
}

fn bf16_values(data: &[u8]) -> Vec<bf16> {
    assert!(data.len().is_multiple_of(2), "invalid private BF16 boundary");
    data.chunks_exact(2).map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect()
}

fn f32_values(data: &[u8]) -> Vec<f32> {
    assert!(data.len().is_multiple_of(4), "invalid private FP32 boundary");
    data.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect()
}

fn config(model: &Path) -> VisionTowerConfig {
    let value = metadata(model.join("config.json"));
    let c = &value["artifact_contract"]["vision_config"];
    let u = |name: &str| c[name].as_u64().and_then(|x| u32::try_from(x).ok()).expect("invalid private tower dimension");
    VisionTowerConfig {
        depth: u("depth"),
        hidden_size: u("hidden_size"),
        intermediate_size: u("intermediate_size"),
        num_heads: u("num_heads"),
        num_position_embeddings: u("num_position_embeddings"),
        out_hidden_size: u("out_hidden_size"),
        patch_size: u("patch_size"),
        spatial_merge_size: u("spatial_merge_size"),
        temporal_patch_size: u("temporal_patch_size"),
        in_channels: u("in_channels"),
        layer_norm_eps: 1e-6,
    }
}

fn block_name(stage: VisionBlockStage) -> &'static str {
    match stage {
        VisionBlockStage::Norm1 => "norm1",
        VisionBlockStage::Qkv => "qkv",
        VisionBlockStage::Queries => "queries",
        VisionBlockStage::Keys => "keys",
        VisionBlockStage::Values => "values",
        VisionBlockStage::Attention => "attention",
        VisionBlockStage::Projection => "projection",
        VisionBlockStage::AttentionResidual => "attention_residual",
        VisionBlockStage::Norm2 => "norm2",
        VisionBlockStage::Fc1 => "fc1",
        VisionBlockStage::Gelu => "gelu",
        VisionBlockStage::Fc2 => "fc2",
        VisionBlockStage::Output => "output",
    }
}

fn merger_name(stage: VisionMergerStage) -> &'static str {
    match stage {
        VisionMergerStage::Norm => "norm",
        VisionMergerStage::Fc1 => "fc1",
        VisionMergerStage::Gelu => "gelu",
        VisionMergerStage::Output => "output",
    }
}

fn measure<B: Backend>(
    backend: &str,
    case: &str,
    index: Value,
    folder: &Path,
    stages: &Value,
    native: &Path,
    readbacks: Vec<(&'static str, B::GlobalBuffer)>,
) -> Vec<Value> {
    let mut metrics = Vec::new();
    for (stage, buffer) in readbacks {
        let expected =
            f32_values(&bytes(folder.join(stages[stage]["file"].as_str().expect("missing private stage file"))));
        let actual = (&buffer).as_slice::<bf16>();
        assert_eq!(actual.len(), expected.len(), "private local-step shape mismatch");
        let (mut error, mut reference, mut actual_squared, mut dot, mut maximum, mut mismatches) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        let mut raw = Vec::with_capacity(actual.len() * 4);
        for (a, b) in actual.iter().zip(expected.iter()) {
            let a = a.to_f32();
            assert!(a.is_finite() && b.is_finite(), "nonfinite private local-step output");
            raw.extend_from_slice(&a.to_le_bytes());
            mismatches += usize::from(a.to_bits() != b.to_bits());
            let (a, b) = (f64::from(a), f64::from(*b));
            let d = a - b;
            error += d * d;
            reference += b * b;
            actual_squared += a * a;
            dot += a * b;
            maximum = maximum.max(d.abs());
        }
        let prefix = match index.as_u64() {
            Some(n) => format!("block_{n}"),
            None => "merger".into(),
        };
        std::fs::write(native.join(format!("{prefix}_{stage}.f32")), raw)
            .unwrap_or_else(|_| panic!("private local-step output write failed"));
        metrics.push(json!({"backend":backend,"fixture":case,"block":index,"stage":stage,"elements":actual.len(),"bf16_mismatches":mismatches,"relative_rms_error":(error/reference.max(f64::MIN_POSITIVE)).sqrt(),"cosine_similarity":dot/(reference*actual_squared).sqrt().max(f64::MIN_POSITIVE),"maximum_absolute_error":maximum,"same_reference_input":true,"diagnostic_only":true}));
    }
    metrics
}

fn local_reference<B: Backend>(backend: &str) {
    let root =
        PathBuf::from(std::env::var_os("STARK_VISION_LOCAL_PATH").expect("set private local-step reference directory"));
    let manifest = metadata(root.join("manifest.json"));
    let model = PathBuf::from(manifest["model_directory"].as_str().expect("missing private model directory"));
    let tag = std::env::var("STARK_VISION_RUN_TAG").unwrap_or_else(|_| "initial".into());
    assert!(
        !tag.is_empty() && tag.len() <= 64 && tag.bytes().all(|x| x.is_ascii_alphanumeric() || x == b'-' || x == b'_'),
        "invalid private run tag"
    );
    let context = B::Context::new().unwrap_or_else(|_| panic!("private local-step context failed"));
    let file = File::open(model.join("model.safetensors"))
        .unwrap_or_else(|_| panic!("private local-step weights unavailable"));
    let loader =
        ParameterLoader::<B>::new(&file, &context).unwrap_or_else(|_| panic!("private local-step indexing failed"));
    let tower = VisionTower::<B>::new(&context, &config(&model), &loader.tree().subtree("vision"))
        .unwrap_or_else(|_| panic!("private local-step model loading failed"));
    let mut metrics = Vec::new();
    for definition in manifest["cases"].as_array().expect("missing private local cases") {
        let name = definition["name"].as_str().expect("missing private case label");
        let folder = root.join(name);
        let case = metadata(root.join(definition["metadata_file"].as_str().unwrap()));
        let grid: [u32; 3] = serde_json::from_value(case["grid"].clone()).expect("invalid private local grid");
        let native = root.join(format!("native-{backend}-{tag}")).join(name);
        private_directory(&native);
        for block in case["blocks"].as_array().unwrap() {
            let index = block["index"].as_u64().unwrap() as usize;
            let hidden = bf16_values(&bytes(folder.join(block["input"]["file"].as_str().unwrap())));
            let mut commands = context
                .create_command_buffer(None, None)
                .unwrap_or_else(|_| panic!("private local command setup failed"));
            let mut readbacks = Vec::new();
            let output = tower
                .encode_isolated_block_with_observer(index, &hidden, grid, &mut commands, |stage, values, commands| {
                    let mut readback = context.create_buffer(values.size())?;
                    commands.encode_copy(values, &mut readback);
                    readbacks.push((block_name(stage), readback));
                    Ok(())
                })
                .unwrap_or_else(|_| panic!("private local block encoding failed"));
            drop(output);
            commands
                .end_encoding()
                .submit()
                .wait_until_completed()
                .unwrap_or_else(|_| panic!("private local block completion failed"));
            assert_eq!(readbacks.len(), 13, "missing local block stages");
            metrics.extend(measure::<B>(backend, name, json!(index), &folder, &block["stages"], &native, readbacks));
        }
        let m = &case["merger"];
        let hidden = bf16_values(&bytes(folder.join(m["input"]["file"].as_str().unwrap())));
        let mut commands =
            context.create_command_buffer(None, None).unwrap_or_else(|_| panic!("private merger command setup failed"));
        let mut readbacks = Vec::new();
        let output = tower
            .encode_isolated_merger_with_observer(&hidden, grid, &mut commands, |stage, values, commands| {
                let mut readback = context.create_buffer(values.size())?;
                commands.encode_copy(values, &mut readback);
                readbacks.push((merger_name(stage), readback));
                Ok(())
            })
            .unwrap_or_else(|_| panic!("private isolated merger failed"));
        drop(output);
        commands
            .end_encoding()
            .submit()
            .wait_until_completed()
            .unwrap_or_else(|_| panic!("private merger completion failed"));
        assert_eq!(readbacks.len(), 4, "missing merger stages");
        metrics.extend(measure::<B>(backend, name, Value::Null, &folder, &m["stages"], &native, readbacks));
    }
    let mut worst = metrics.iter().collect::<Vec<_>>();
    worst.sort_by(|a, b| {
        b["relative_rms_error"].as_f64().unwrap().total_cmp(&a["relative_rms_error"].as_f64().unwrap())
    });
    println!(
        "{}",
        json!({"backend":backend,"local_stages":metrics.len(),"worst_boundaries":worst.into_iter().take(12).collect::<Vec<_>>()})
    );
    std::fs::write(
        root.join(format!("native-local-{backend}-{tag}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("private local aggregate write failed"));
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_local_reference_cpu() {
    local_reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_local_reference_metal() {
    local_reference::<crate::backends::metal::Metal>("metal");
}

fn attention_name(stage: VisionAttentionStage) -> (&'static str, bool) {
    match stage {
        VisionAttentionStage::FullPrecisionScores => ("full_precision_scores", true),
        VisionAttentionStage::Scores => ("scores", false),
        VisionAttentionStage::ScaledScores => ("scaled_scores", false),
        VisionAttentionStage::Probabilities => ("probabilities", false),
        VisionAttentionStage::FullPrecisionHead => ("full_precision_head", true),
        VisionAttentionStage::Head => ("head", false),
    }
}

fn attention_reference<B: Backend>(backend: &str) {
    let root = PathBuf::from(
        std::env::var_os("STARK_VISION_ATTENTION_PATH").expect("set private attention reference directory"),
    );
    let manifest = metadata(root.join("manifest.json"));
    let model = PathBuf::from(manifest["model_directory"].as_str().expect("missing private model directory"));
    let context = B::Context::new().unwrap_or_else(|_| panic!("private attention context failed"));
    let file =
        File::open(model.join("model.safetensors")).unwrap_or_else(|_| panic!("private attention weights unavailable"));
    let loader =
        ParameterLoader::<B>::new(&file, &context).unwrap_or_else(|_| panic!("private attention indexing failed"));
    let tower = VisionTower::<B>::new(&context, &config(&model), &loader.tree().subtree("vision"))
        .unwrap_or_else(|_| panic!("private attention model loading failed"));
    let mut metrics = Vec::new();
    for case in manifest["cases"].as_array().expect("missing private attention cases") {
        if case["backend"].as_str() != Some(backend) {
            continue;
        }
        let fixture = case["fixture"].as_str().expect("missing private attention fixture");
        let index = case["block"].as_u64().expect("missing private attention block") as usize;
        let grid: [u32; 3] = serde_json::from_value(case["grid"].clone()).expect("invalid private attention grid");
        let inputs = &case["inputs"];
        let input = |name: &str| bf16_values(&bytes(root.join(inputs[name]["file"].as_str().unwrap())));
        let queries = input("queries");
        let keys = input("keys");
        let values = input("values");
        let mut commands = context
            .create_command_buffer(None, None)
            .unwrap_or_else(|_| panic!("private attention command setup failed"));
        let mut readbacks = Vec::new();
        let output = tower
            .encode_isolated_attention_with_observer(
                index,
                &queries,
                &keys,
                &values,
                grid,
                &mut commands,
                |head, stage, values, commands| {
                    let mut readback = context.create_buffer(values.size())?;
                    commands.encode_copy(values, &mut readback);
                    readbacks.push((head, stage, readback));
                    Ok(())
                },
            )
            .unwrap_or_else(|_| panic!("private attention encoding failed"));
        drop(output);
        commands
            .end_encoding()
            .submit()
            .wait_until_completed()
            .unwrap_or_else(|_| panic!("private attention completion failed"));
        assert_eq!(readbacks.len(), case["heads"].as_array().unwrap().len() * 6);
        let native = root.join(format!("native-{backend}")).join(fixture).join(format!("block_{index}"));
        private_directory(&native);
        for (head, stage, buffer) in readbacks {
            let (name, full_precision) = attention_name(stage);
            let definition = &case["heads"][head as usize]["stages"][name];
            let expected = f32_values(&bytes(root.join(definition["file"].as_str().unwrap())));
            let actual = if full_precision {
                (&buffer).as_slice::<f32>().to_vec()
            } else {
                (&buffer).as_slice::<bf16>().iter().map(|x| x.to_f32()).collect::<Vec<_>>()
            };
            assert_eq!(actual.len(), expected.len(), "private attention boundary shape mismatch");
            let (mut error, mut denominator, mut maximum, mut mismatches) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            let mut raw = Vec::with_capacity(actual.len() * 4);
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!(a.is_finite() && e.is_finite(), "nonfinite private attention boundary");
                raw.extend_from_slice(&a.to_le_bytes());
                mismatches += usize::from(a.to_bits() != e.to_bits());
                let difference = f64::from(*a) - f64::from(*e);
                error += difference * difference;
                denominator += f64::from(*e).powi(2);
                maximum = maximum.max(difference.abs());
            }
            std::fs::write(native.join(format!("head_{head}_{name}.f32")), raw)
                .unwrap_or_else(|_| panic!("private attention boundary write failed"));
            metrics.push(json!({"backend":backend,"fixture":fixture,"block":index,"head":head,"stage":name,"full_precision":full_precision,"elements":actual.len(),"mismatches":mismatches,"relative_rms_error":(error/denominator.max(f64::MIN_POSITIVE)).sqrt(),"maximum_absolute_error":maximum,"diagnostic_only":true}));
        }
    }
    assert!(!metrics.is_empty(), "no matching private attention cases");
    let mut worst = metrics.iter().collect::<Vec<_>>();
    worst.sort_by(|a, b| {
        b["relative_rms_error"].as_f64().unwrap().total_cmp(&a["relative_rms_error"].as_f64().unwrap())
    });
    println!(
        "{}",
        json!({"backend":backend,"attention_boundaries":metrics.len(),"worst_boundaries":worst.into_iter().take(12).collect::<Vec<_>>()})
    );
    std::fs::write(
        root.join(format!("native-attention-{backend}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("private attention aggregate write failed"));
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_attention_reference_cpu() {
    attention_reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_attention_reference_metal() {
    attention_reference::<crate::backends::metal::Metal>("metal");
}

fn activation_reference<B: Backend>(backend: &str) {
    use crate::backends::common::kernel::VisionGeluBF16Kernel;

    let root = PathBuf::from(
        std::env::var_os("STARK_VISION_ACTIVATION_PATH").expect("set private activation reference directory"),
    );
    let manifest = metadata(root.join("manifest.json"));
    let definitions = &manifest["files"];
    let input = bf16_values(&bytes(root.join(definitions["input"]["file"].as_str().unwrap())));
    assert_eq!(input.len(), 65_280, "incomplete private finite BF16 activation domain");
    assert!(input.iter().all(|x| x.is_finite()), "nonfinite private activation input");
    let context = B::Context::new().unwrap_or_else(|_| panic!("private activation context failed"));
    let kernel = <B::Kernels as Kernels>::VisionGeluBF16Kernel::new(&context)
        .unwrap_or_else(|_| panic!("private activation kernel creation failed"));
    let native = root.join(format!("native-{backend}"));
    private_directory(&native);
    let mut metrics = Vec::new();
    for (name, approximate) in [("exact", 0u32), ("approximate", 1u32)] {
        let mut readback = context
            .create_buffer(input.len() * size_of::<bf16>())
            .unwrap_or_else(|_| panic!("private activation readback allocation failed"));
        let mut commands = context
            .create_command_buffer(None, None)
            .unwrap_or_else(|_| panic!("private activation command setup failed"));
        let source = commands
            .allocate_constant_from_slice(&input)
            .unwrap_or_else(|_| panic!("private activation input upload failed"));
        let mut values = commands
            .allocate_scratch_for_shape(&[input.len() as u32], crate::data_type::DataType::BF16)
            .unwrap_or_else(|_| panic!("private activation scratch allocation failed"));
        commands.encode_copy(&source, &mut values);
        kernel.encode(&mut values, input.len() as u32, approximate, &mut commands);
        commands.encode_copy(&values, &mut readback);
        drop(values);
        drop(source);
        commands
            .end_encoding()
            .submit()
            .wait_until_completed()
            .unwrap_or_else(|_| panic!("private activation completion failed"));
        let expected = f32_values(&bytes(root.join(definitions[name]["file"].as_str().unwrap())));
        let actual = (&readback).as_slice::<bf16>();
        assert_eq!(actual.len(), expected.len());
        let (mut mismatches, mut maximum) = (0usize, 0.0f32);
        let mut raw = Vec::with_capacity(actual.len() * 4);
        for (a, e) in actual.iter().zip(expected.iter()) {
            let a = a.to_f32();
            mismatches += usize::from(a.to_bits() != e.to_bits());
            maximum = maximum.max((a - e).abs());
            raw.extend_from_slice(&a.to_le_bytes());
        }
        std::fs::write(native.join(format!("{name}.f32")), raw)
            .unwrap_or_else(|_| panic!("private activation output write failed"));
        metrics.push(json!({"backend":backend,"activation":name,"finite_bf16_inputs":actual.len(),"bf16_mismatches":mismatches,"maximum_absolute_error":maximum}));
    }
    println!("{}", json!({"activation_metrics":metrics}));
    std::fs::write(
        root.join(format!("native-activation-{backend}-metrics.json")),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap_or_else(|_| panic!("private activation aggregate write failed"));
    assert!(metrics.iter().all(|x| x["bf16_mismatches"].as_u64() == Some(0)), "finite BF16 activation parity failed");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_activation_reference_cpu() {
    activation_reference::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_activation_reference_metal() {
    activation_reference::<crate::backends::metal::Metal>("metal");
}
