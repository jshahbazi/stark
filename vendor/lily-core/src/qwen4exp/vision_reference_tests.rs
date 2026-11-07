//! Stark's ignored source-bound tower replay; no production API is added.

use std::{fs, path::Path};

use half::bf16;
use serde_json::{Value, json};

use super::*;
use crate::{
    qwen4exp::config::Qwen4ExpConfig,
    safetensors::Checkpoint,
    sha256::{hex, sha256},
    weights::Loader,
};

fn checked_file(root: &Path, record: &Value, cap: usize) -> Vec<u8> {
    let filename =
        record["file"].as_str().expect("private reference filename unavailable");
    assert!(!filename.contains(['/', '\\']) && filename != "." && filename != "..");
    let path = root.join(filename);
    assert!(
        fs::metadata(&path).expect("private reference metadata unavailable").len()
            <= cap as u64
    );
    let data = fs::read(path).expect("private reference unavailable");
    assert_eq!(
        hex(&sha256(&data)),
        record["sha256"].as_str().expect("private reference digest unavailable")
    );
    data
}

fn checked_tensor(root: &Path, record: &Value) -> Vec<bf16> {
    let bytes = checked_file(root, record, 2 * 256 * 4608);
    let shape = record["shape"].as_array().expect("private tensor shape unavailable");
    let elements = shape
        .iter()
        .try_fold(1_usize, |count, value| {
            count.checked_mul(usize::try_from(value.as_u64()?).ok()?)
        })
        .expect("private tensor shape invalid");
    assert_eq!(elements.checked_mul(2), Some(bytes.len()));
    bytes
        .chunks_exact(2)
        .map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]])))
        .collect()
}

fn check_weight(record: &Value, name: &str, tensor: &Tensor) {
    assert_eq!(tensor.dtype(), DType::BF16);
    assert_eq!(
        hex(&sha256(tensor.raw_bytes())),
        record[name].as_str().expect("private tensor identity missing")
    );
}

fn numerical_metric(actual: &[f32], expected: &[bf16]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut reference_norm, mut actual_norm, mut dot) =
        (0_f64, 0_f64, 0_f64, 0_f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(f32::from(b)));
        assert!(a.is_finite() && b.is_finite());
        error += (a - b).powi(2);
        reference_norm += b * b;
        actual_norm += a * a;
        dot += a * b;
    }
    ((error / reference_norm).sqrt(), dot / (actual_norm * reference_norm).sqrt())
}

/// Each operator gets the source's input, so accumulated error from earlier
/// operators cannot hide the first arithmetic difference. These are private
/// fixture controls over existing kernels; the production graph is unchanged.
fn operator_metrics(
    context: &MetalContext,
    vision: &super::super::config::VisionConfig,
    weights: &super::super::vision_weights::VisionWeights,
    folder: &Path,
    case: &Value,
    gh: usize,
    gw: usize,
) -> Vec<Value> {
    let Some(operators) = case["operators"].as_array() else { return Vec::new() };
    let block = &weights.blocks[0];
    operators.iter().enumerate().map(|(index, record)| {
        let name = record["operator"].as_str().expect("private operator name missing");
        let source = checked_tensor(folder, &record["input"]);
        let expected = checked_tensor(folder, &record["output"]);
        let shape = |key: &str| record[key]["shape"].as_array().unwrap().iter()
            .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap()).collect::<Vec<_>>();
        let input = Tensor::from_bytes(context, bytemuck::cast_slice(&source), &shape("input"), DType::BF16).unwrap();
        let output = Tensor::zeros(context, &shape("output"), DType::BF16).unwrap();
        let pass = context.begin().unwrap();
        let linear = match name {
            "patch" => Some((&weights.patch_proj_w, &weights.patch_proj_b)),
            "qkv" => Some((&block.qkv_w, &block.qkv_b)),
            "projection" => Some((&block.proj_w, &block.proj_b)),
            "fc1" => Some((&block.fc1_w, &block.fc1_b)),
            "fc2" => Some((&block.fc2_w, &block.fc2_b)),
            "merger_fc1" => Some((&weights.merger_fc1_w, &weights.merger_fc1_b)),
            "merger_fc2" => Some((&weights.merger_fc2_w, &weights.merger_fc2_b)),
            _ => None,
        };
        let result = if let Some((weight, bias)) = linear {
            gemm_bf16_nt_bias(context, &pass, &input, weight, bias, &output)
        } else {
            match name {
                "norm1" => layernorm_bf16(context, &pass, &input, &block.norm1_w, &block.norm1_b, &output, LAYERNORM_EPS),
                "norm2" => layernorm_bf16(context, &pass, &input, &block.norm2_w, &block.norm2_b, &output, LAYERNORM_EPS),
                "merger_norm" => layernorm_bf16(context, &pass, &input, &weights.merger_norm_w, &weights.merger_norm_b, &output, LAYERNORM_EPS),
                "attention" => attention_full_bf16(context, &pass, &input, &output, vision.num_heads, (vision.head_dim() as f32).powf(-0.5)),
                "position" | "rotary" | "gelu_tanh" | "gelu_erf" => {
                    // These kernels operate in place; preserve the exact fixture input.
                    output.write_bytes(bytemuck::cast_slice(&source)).unwrap();
                    match name {
                        "position" => patch_pos_embed_bf16(context, &pass, &output, &weights.pos_embed, gh, gw),
                        "rotary" => qkv_rope_2d_bf16(context, &pass, &output, vision.num_heads, gh, gw, ROPE_THETA),
                        "gelu_tanh" => gelu_bf16(context, &pass, &output, Gelu::Tanh),
                        _ => gelu_bf16(context, &pass, &output, Gelu::Erf),
                    }
                },
                _ => panic!("private operator unsupported"),
            }
        };
        result.unwrap_or_else(|_| panic!("private operator execution failed"));
        pass.commit_wait().unwrap_or_else(|_| panic!("private operator synchronization failed"));
        let actual = output.to_f32().unwrap_or_else(|_| panic!("private operator readback failed"));
        let (rms, cosine) = numerical_metric(&actual, &expected);
        if name == "attention" && record["stages"].is_object() {
            let n = input.shape()[0];
            let plane = vision.num_heads * n * n;
            let stages = Tensor::zeros(context, &[3, vision.num_heads, n, n], DType::F32).unwrap();
            let traced = Tensor::zeros(context, output.shape(), DType::BF16).unwrap();
            let pass = context.begin().unwrap();
            crate::kernels::vision::attention_stages_bf16(context, &pass, &input, &traced, &stages, vision.num_heads, (vision.head_dim() as f32).powf(-0.5))
                .unwrap_or_else(|_| panic!("private attention trace execution failed"));
            pass.commit_wait().unwrap_or_else(|_| panic!("private attention trace synchronization failed"));
            let values = stages.to_f32().unwrap();
            let stage_metrics = ["qk", "scaled", "probabilities"].iter().enumerate().map(|(stage_index, key)| {
                let expected = checked_tensor(folder, &record["stages"][key]);
                let (rms, cosine) = numerical_metric(&values[stage_index * plane..(stage_index + 1) * plane], &expected);
                json!({"stage_index":stage_index,"elements":plane,"relative_rms":rms,"cosine":cosine})
            }).collect::<Vec<_>>();
            let traced = traced.to_f32().unwrap();
            let mismatches = traced.iter().zip(&actual).filter(|(a,b)| a.to_bits() != b.to_bits()).count();
            let ordinary = actual.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
            let (trace_rms, _) = numerical_metric(&traced, &ordinary);
            println!("{}", json!({"private_retained_attention_stage_metrics":{"patches":n,"stages":stage_metrics,"trace_to_ordinary_relative_rms":trace_rms,"trace_ordinary_mismatch_count":mismatches}}));
            assert_eq!(mismatches, 0, "private attention trace perturbs ordinary arithmetic");
        }
        json!({"operator_index":index,"elements":actual.len(),"relative_rms":rms,"cosine":cosine})
    }).collect()
}

#[test]
#[ignore = "requires an explicitly selected retained artifact, source-bound fixture and GPU lease"]
fn retained_lily_tower_source_reference() {
    let artifact = std::path::PathBuf::from(
        std::env::var_os("STARK_LILY_MODEL_PATH")
            .expect("set private retained artifact path"),
    );
    let reference = std::path::PathBuf::from(
        std::env::var_os("STARK_LILY_VISION_REFERENCE_PATH")
            .expect("set private retained tower reference path"),
    );
    let manifest_bytes = fs::read(reference.join("manifest.json"))
        .expect("private tower manifest unavailable");
    assert!(manifest_bytes.len() <= 1024 * 1024);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .expect("private tower manifest invalid");
    assert_eq!(manifest["kind"], "RetainedLilyTowerSourceReference");
    assert_eq!(manifest["source_revision"], "2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397");
    assert_eq!(
        manifest["source_sha256"]["modeling_qwen4_exp.py"],
        "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
    );
    assert_eq!(manifest["gate"], json!({"relative_rms_max":0.02,"cosine_min":0.999}));
    let contract_bytes = fs::read(artifact.join("stark-image-artifact.json"))
        .expect("private retained contract unavailable");
    assert_eq!(hex(&sha256(&contract_bytes)), manifest["artifact_manifest_sha256"]);
    let contract: Value = serde_json::from_slice(&contract_bytes)
        .expect("private retained contract invalid");
    assert_eq!(contract, manifest["artifact_contract"]);
    assert_eq!(
        contract["artifact_sha256"],
        "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4"
    );
    for (file, key) in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("preprocessor_config.json", "processor_sha256"),
        ("chat_template.jinja", "template_sha256"),
    ] {
        assert!(fs::metadata(artifact.join(file)).unwrap().len() <= 32 * 1024 * 1024);
        assert_eq!(
            hex(&sha256(&fs::read(artifact.join(file)).unwrap())),
            contract[key]
        );
    }
    let config_bytes = fs::read(artifact.join("config.json")).unwrap();
    let raw_config: Value = serde_json::from_slice(&config_bytes).unwrap();
    assert_eq!(manifest["vision_config"], raw_config["vision_config"]);
    let config = Qwen4ExpConfig::from_json(&config_bytes)
        .unwrap_or_else(|_| panic!("private tower config admission failed"));
    let vision = config.vision.as_ref().expect("private retained tower config missing");
    assert_eq!((vision.depth, vision.hidden_size, vision.num_heads), (27, 1152, 16));
    let context = MetalContext::new()
        .unwrap_or_else(|_| panic!("private tower Metal setup failed"));
    let checkpoint = Checkpoint::open(&artifact)
        .unwrap_or_else(|_| panic!("private tower header admission failed"));
    let loader =
        Loader::new(&context, checkpoint, config.quantization, &[], Box::new(|_| 4));
    let weights = super::super::vision_weights::load(&loader, vision)
        .unwrap_or_else(|_| panic!("private tower weights failed"));
    assert_eq!(weights.bytes(), 897_862_112);
    let hashes = &manifest["source_visual_tensor_sha256"];
    for (name, tensor) in [
        ("patch_embed.proj.weight", &weights.patch_proj_w),
        ("patch_embed.proj.bias", &weights.patch_proj_b),
        ("pos_embed.weight", &weights.pos_embed),
        ("merger.norm.weight", &weights.merger_norm_w),
        ("merger.norm.bias", &weights.merger_norm_b),
        ("merger.linear_fc1.weight", &weights.merger_fc1_w),
        ("merger.linear_fc1.bias", &weights.merger_fc1_b),
        ("merger.linear_fc2.weight", &weights.merger_fc2_w),
        ("merger.linear_fc2.bias", &weights.merger_fc2_b),
    ] {
        check_weight(hashes, name, tensor);
    }
    for (index, block) in weights.blocks.iter().enumerate() {
        for (name, tensor) in [
            ("norm1.weight", &block.norm1_w),
            ("norm1.bias", &block.norm1_b),
            ("attn.qkv.weight", &block.qkv_w),
            ("attn.qkv.bias", &block.qkv_b),
            ("attn.proj.weight", &block.proj_w),
            ("attn.proj.bias", &block.proj_b),
            ("norm2.weight", &block.norm2_w),
            ("norm2.bias", &block.norm2_b),
            ("mlp.linear_fc1.weight", &block.fc1_w),
            ("mlp.linear_fc1.bias", &block.fc1_b),
            ("mlp.linear_fc2.weight", &block.fc2_w),
            ("mlp.linear_fc2.bias", &block.fc2_b),
        ] {
            check_weight(hashes, &format!("blocks.{index}.{name}"), tensor);
        }
    }
    let mut scratch = None;
    let mut metrics = Vec::new();
    let mut operators = Vec::new();
    for (case_index, case) in
        manifest["cases"].as_array().expect("private cases missing").iter().enumerate()
    {
        let name = case["name"].as_str().unwrap();
        assert!(matches!(name, "square_256" | "stress_32"));
        let folder = reference.join(name);
        let grid = case["grid"].as_array().unwrap();
        assert_eq!(grid[0], 1);
        let gh = usize::try_from(grid[1].as_u64().unwrap()).unwrap();
        let gw = usize::try_from(grid[2].as_u64().unwrap()).unwrap();
        assert!(gh * gw <= 256);
        let rgb = checked_file(
            &folder,
            &json!({"file":"rgb.u8","sha256":case["rgb_sha256"]}),
            256 * 256 * 3,
        );
        assert_eq!(rgb.len(), gh * gw * 16 * 16 * 3);
        let pixels = checked_tensor(&folder, &case["pixels"]);
        assert_eq!(pixels.len(), gh * gw * vision.patch_dim());
        let pixels = pixels.into_iter().map(f32::from).collect::<Vec<_>>();
        operators.push(json!({"case_index":case_index,"metrics":operator_metrics(&context, vision, &weights, &folder, case, gh, gw)}));
        for stage in case["stages"].as_array().unwrap() {
            let depth = usize::try_from(stage["blocks"].as_u64().unwrap()).unwrap();
            assert!(matches!(depth, 0 | 1 | 27));
            let expected = checked_tensor(&folder, &stage["merged"]);
            let output = forward_with(
                &context,
                vision,
                &weights,
                &mut scratch,
                &pixels,
                gh,
                gw,
                depth,
            )
            .unwrap_or_else(|_| panic!("private retained tower execution failed"));
            let actual = output
                .merged
                .to_f32()
                .unwrap_or_else(|_| panic!("private retained tower readback failed"));
            let (rms, cosine) = numerical_metric(&actual, &expected);
            metrics.push(json!({"case_index":case_index,"blocks":depth,"elements":actual.len(),"relative_rms":rms,"cosine":cosine}));
        }
    }
    println!("{}", json!({"private_retained_tower_metrics":metrics}));
    println!("{}", json!({"private_retained_operator_metrics":operators}));
    // Independent same-input source control: this is tight enough to reject
    // the former unnormalized-BF16-P online policy (~0.4% RMS) while retaining
    // room for F32 reduction order and BF16 halfway rounding across devices.
    for (case, result) in manifest["cases"].as_array().unwrap().iter().zip(&operators) {
        if let Some(records) = case["operators"].as_array() {
            for (record, metric) in
                records.iter().zip(result["metrics"].as_array().unwrap())
            {
                if record["operator"] == "attention" {
                    assert!(
                        metric["relative_rms"].as_f64().unwrap() <= 0.001
                            && metric["cosine"].as_f64().unwrap() >= 0.999999,
                        "private retained attention exceeds same-input source gate"
                    );
                }
            }
        }
    }
    assert!(
        metrics.iter().all(|metric| metric["relative_rms"].as_f64().unwrap() <= 0.02
            && metric["cosine"].as_f64().unwrap() >= 0.999),
        "private retained tower exceeds frozen source gate"
    );
}
