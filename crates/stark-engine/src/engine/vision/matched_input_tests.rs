//! Private same-input decoder primitive diagnostics; never a production API.

use std::path::{Path, PathBuf};

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{
        BufferMut, CommandBufferExecutable, CommandBufferPending, Kernels,
        kernel::{
            VisionCastBF16Kernel, VisionMatmulBF16Kernel,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    encodable_block::normalization::{Normalization, PostLayerScalar, ShortcutMode},
};

fn bytes(
    root: &Path,
    definition: &Value,
) -> Vec<u8> {
    let data = std::fs::read(root.join(definition["file"].as_str().unwrap())).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&data)), definition["sha256"].as_str().unwrap());
    data
}

fn record<B: Backend>(
    actual: &B::GlobalBuffer,
    expected: &[u8],
    output: &Path,
) -> Value {
    let actual = actual.as_slice::<bf16>();
    assert_eq!(actual.len() * 2, expected.len());
    let (mut squared_error, mut squared_expected, mut mismatches) = (0.0f64, 0.0f64, 0usize);
    let mut raw = Vec::with_capacity(expected.len());
    for (actual, expected) in actual.iter().zip(expected.chunks_exact(2)) {
        let expected = bf16::from_bits(u16::from_le_bytes(expected.try_into().unwrap()));
        let a = f64::from(actual.to_f32());
        let e = f64::from(expected.to_f32());
        assert!(a.is_finite() && e.is_finite());
        squared_error += (a - e).powi(2);
        squared_expected += e * e;
        mismatches += usize::from(actual.to_bits() != expected.to_bits());
        raw.extend_from_slice(&actual.to_bits().to_le_bytes());
    }
    std::fs::write(output, raw).unwrap();
    json!({"relative_rms":(squared_error/squared_expected).sqrt(),"bf16_mismatches":mismatches,"elements":actual.len()})
}

fn run<B: Backend>(backend: &str) {
    let artifact = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set exact private artifact"));
    let reference = PathBuf::from(
        std::env::var_os("STARK_VISION_MATCHED_INPUT_REFERENCE_PATH").expect("set matched-input private reference"),
    );
    let tag = std::env::var("STARK_VISION_PRIMITIVE_TAG").unwrap_or_else(|_| "baseline".to_owned());
    assert!(tag.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'));
    let context = B::Context::new().unwrap();
    let config: Value = serde_json::from_slice(&std::fs::read(artifact.join("config.json")).unwrap()).unwrap();
    // Keep this diagnostic bound to the exact admitted artifact, even though
    // it constructs only individual operators and never exposes a loader.
    for (name, hash) in ARTIFACT_FILES {
        assert!(file_matches(&mut File::open(artifact.join(name)).unwrap(), hash).unwrap());
    }
    let weights_file = File::open(artifact.join("model.safetensors")).unwrap();
    let loader = ParameterLoader::<B>::new(&weights_file, &context).unwrap();
    let loader = if std::env::var_os("STARK_VISION_RMS_REFERENCE_PROFILE").is_some() {
        loader.with_numerical_profile(crate::parameters::NumericalProfile::Qwen35ReferenceBF16)
    } else {
        loader
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(reference.join("manifest.json")).unwrap()).unwrap();
    let output = reference.join(format!("native-{backend}-{tag}"));
    std::fs::create_dir_all(&output).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let cast = <B::Kernels as Kernels>::VisionCastBF16Kernel::new(&context).unwrap();
    let exact = <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(&context).unwrap();
    let mut metrics = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let index = case["case_index"].as_u64().unwrap();
        let folder = reference.join(case["folder"].as_str().unwrap());
        for stage in case["stages"].as_array().unwrap() {
            let name = stage["name"].as_str().unwrap();
            let input_data = bytes(&folder, &stage["input"]);
            let expected = bytes(&folder, &stage["expected"]);
            let m = stage["input"]["shape"][0].as_u64().unwrap() as u32;
            let k = stage["input"]["shape"][1].as_u64().unwrap() as u32;
            let n = stage["expected"]["shape"][1].as_u64().unwrap() as u32;
            let mut input = context.create_buffer(input_data.len()).unwrap();
            (&mut input).copyin(&input_data);
            let tree = loader.tree().subtree(stage["tree"].as_str().unwrap());
            if stage["kind"] == "norm" {
                let definition = config["decoder_config"]["transformer_config"]["layer_configs"][0]
                    [stage["config"].as_str().unwrap()]
                .clone();
                let norm = Normalization::<B>::new(
                    k,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    DataType::BF16,
                    &serde_json::from_value(definition).unwrap(),
                    &tree,
                    &context,
                )
                .unwrap();
                let mut command = context.create_command_buffer(None, None).unwrap();
                let result = norm.encode(&input, 0, m, None::<&mut B::GlobalBuffer>, &mut command).unwrap();
                let mut readback = context.create_buffer(expected.len()).unwrap();
                command.encode_copy(&result, &mut readback);
                command.end_encoding().submit().wait_until_completed().unwrap();
                let values = record::<B>(&readback, &expected, &output.join(format!("case_{index}_{name}.bf16")));
                metrics.push(json!({"case_index":index,"stage":name,"mode":"production_norm","metrics":values}));
            } else {
                let weights = tree
                    .subtree("weights")
                    .leaf("weights")
                    .unwrap()
                    .validate(&[n, k], DataType::BF16)
                    .unwrap()
                    .read_buffer()
                    .unwrap();
                for mode in ["production_bf16", "production_f32_then_cast", "reference_32_stream"] {
                    let mut command = context.create_command_buffer(None, None).unwrap();
                    let mut readback = context.create_buffer(expected.len()).unwrap();
                    if mode == "production_bf16" {
                        let mut kernel = <B::Kernels as Kernels>::MatmulKernel::new(
                            &context,
                            DataType::BF16,
                            DataType::BF16,
                            DataType::BF16,
                        )
                        .unwrap();
                        kernel
                            .encode(
                                MatmulArguments {
                                    a: MatmulA::FullPrecision {
                                        values: &input,
                                        offset: 0,
                                    },
                                    b: MatmulB::FullPrecision {
                                        b: &weights,
                                    },
                                    b_leading_dimension: None,
                                    b_transpose: true,
                                    d: &mut readback,
                                    d_transform: MatmulDOps::none(),
                                    gather_indices: None::<&B::GlobalBuffer>,
                                    m,
                                    n,
                                    k,
                                },
                                &mut command,
                            )
                            .unwrap();
                    } else {
                        let mut full_precision = context.create_buffer(expected.len() * 2).unwrap();
                        if mode == "production_f32_then_cast" {
                            let mut kernel = <B::Kernels as Kernels>::MatmulKernel::new(
                                &context,
                                DataType::BF16,
                                DataType::BF16,
                                DataType::F32,
                            )
                            .unwrap();
                            kernel
                                .encode(
                                    MatmulArguments {
                                        a: MatmulA::FullPrecision {
                                            values: &input,
                                            offset: 0,
                                        },
                                        b: MatmulB::FullPrecision {
                                            b: &weights,
                                        },
                                        b_leading_dimension: None,
                                        b_transpose: true,
                                        d: &mut full_precision,
                                        d_transform: MatmulDOps::none(),
                                        gather_indices: None::<&B::GlobalBuffer>,
                                        m,
                                        n,
                                        k,
                                    },
                                    &mut command,
                                )
                                .unwrap();
                        } else {
                            exact.encode(&input, &weights, &mut full_precision, m, n, k, k, 1, 32, &mut command);
                        }
                        cast.encode(&full_precision, &mut readback, m * n, &mut command);
                    }
                    command.end_encoding().submit().wait_until_completed().unwrap();
                    let values =
                        record::<B>(&readback, &expected, &output.join(format!("case_{index}_{name}_{mode}.bf16")));
                    metrics.push(json!({"case_index":index,"stage":name,"mode":mode,"metrics":values}));
                }
                if name == "mlp_up" {
                    use crate::backends::common::{
                        gpu_types::ActivationType,
                        kernel::{GatedActMul, GatedActMulSettings},
                    };
                    let down = case["stages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|definition| definition["name"] == "mlp_down")
                        .unwrap();
                    let expected_gate = bytes(&folder, &down["input"]);
                    let mut projected = context.create_buffer(expected.len()).unwrap();
                    (&mut projected).copyin(&expected);
                    let mut gate_output = context.create_buffer(expected_gate.len()).unwrap();
                    let gate = GatedActMul::<B>::full_precision(
                        &context,
                        DataType::BF16,
                        true,
                        false,
                        GatedActMulSettings::default(),
                    )
                    .unwrap();
                    let mut command = context.create_command_buffer(None, None).unwrap();
                    gate.encode_fp(
                        &projected,
                        None::<&B::GlobalBuffer>,
                        &mut gate_output,
                        None::<&B::GlobalBuffer>,
                        n / 2,
                        m,
                        0,
                        0,
                        ActivationType::SILU,
                        &mut command,
                    );
                    command.end_encoding().submit().wait_until_completed().unwrap();
                    let measured = record::<B>(
                        &gate_output,
                        &expected_gate,
                        &output.join(format!("case_{index}_mlp_gate_silu.bf16")),
                    );
                    metrics.push(json!({"case_index":index,"stage":"mlp_gate","mode":"production_gated_silu","metrics":measured}));
                }
            }
        }
    }
    std::fs::write(output.join("metrics.json"), serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();
    println!("{}", json!({"matched_input_primitive_metrics":metrics}));
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_decoder_matched_input_primitives_cpu() {
    run::<crate::backends::cpu::Cpu>("cpu");
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_decoder_matched_input_primitives_metal() {
    run::<crate::backends::metal::Metal>("metal");
}

fn attention<B: Backend>(backend: &str) {
    use crate::{
        backends::common::kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig},
        encodable_block::mixer::attention::{KVCacheView, reference::ReferenceBF16Attention},
    };

    let artifact = PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").unwrap());
    let root = PathBuf::from(std::env::var_os("STARK_VISION_MATCHED_ATTENTION_REFERENCE_PATH").unwrap());
    let config: Value = serde_json::from_slice(&std::fs::read(artifact.join("config.json")).unwrap()).unwrap();
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let context = B::Context::new().unwrap();
    let output = root.join(format!("native-{backend}"));
    std::fs::create_dir_all(&output).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut metrics = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let case_index = case["case_index"].as_u64().unwrap();
        let folder = root.join(case["folder"].as_str().unwrap());
        for stage in case["stages"].as_array().unwrap() {
            let layer = stage["layer"].as_u64().unwrap() as usize;
            let heads = stage["query_heads"].as_u64().unwrap() as u32;
            let groups = stage["kv_heads"].as_u64().unwrap() as u32;
            let head_dim = stage["head_dim"].as_u64().unwrap() as u32;
            let rows = stage["suffix_len"].as_u64().unwrap() as u32;
            let scale = stage["scale"].as_f64().unwrap() as f32;
            let query_data = bytes(&folder, &stage["queries"]);
            let key_data = bytes(&folder, &stage["keys"]);
            let value_data = bytes(&folder, &stage["values"]);
            let expected = bytes(&folder, &stage["output"]);
            let mut queries = context.create_buffer(query_data.len()).unwrap();
            let mut keys = context.create_buffer(key_data.len()).unwrap();
            let mut values = context.create_buffer(value_data.len()).unwrap();
            (&mut queries).copyin(&query_data);
            (&mut keys).copyin(&key_data);
            (&mut values).copyin(&value_data);
            let reference = ReferenceBF16Attention::<B>::new(
                &context,
                &serde_json::from_value(
                    config["decoder_config"]["transformer_config"]["layer_configs"][layer]["mixer_config"].clone(),
                )
                .unwrap(),
            )
            .unwrap();
            let generic = <B::Kernels as Kernels>::AttentionKernel::new(
                &context,
                AttentionKernelConfig {
                    head_dim,
                    num_groups: groups,
                    num_q_heads: heads,
                    has_sinks: false,
                    is_kv_cache_ring: false,
                    is_causal: true,
                    sliding_window_size: None,
                    scale: Some(scale),
                    data_type: DataType::BF16,
                    materialize_prefill_scores: false,
                    source_bf16_prefill: false,
                },
            )
            .unwrap();
            for mode in ["production_fused", "reference_full", "reference_prefix_decode"] {
                let mut command = context.create_command_buffer(None, None).unwrap();
                let mut readback = context.create_buffer(expected.len()).unwrap();
                if mode == "production_fused" {
                    let result = generic
                        .encode(
                            AttentionArguments {
                                queries: &queries,
                                keys: &keys,
                                values: &values,
                                suffix_length: rows,
                                trie: None::<&B::GlobalBuffer>,
                                sinks: None,
                                cache: KVCacheView::full(0),
                            },
                            &mut command,
                        )
                        .unwrap();
                    command.encode_copy(&result, &mut readback);
                } else {
                    let parts = if mode == "reference_full" {
                        vec![rows]
                    } else {
                        vec![31, rows - 32, 1]
                    };
                    let mut prefix = 0u32;
                    for count in parts {
                        let mut chunk_data =
                            Vec::with_capacity(count as usize * heads as usize * head_dim as usize * 2);
                        for head in 0..heads {
                            let begin = (head * rows + prefix) as usize * head_dim as usize * 2;
                            let end = begin + count as usize * head_dim as usize * 2;
                            chunk_data.extend_from_slice(&query_data[begin..end]);
                        }
                        let chunk = command.allocate_constant_from_slice(&chunk_data).unwrap();
                        let result = reference.encode(&chunk, &keys, &values, count, prefix, &mut command).unwrap();
                        let begin = prefix as usize * heads as usize * head_dim as usize * 2;
                        let end = begin + count as usize * heads as usize * head_dim as usize * 2;
                        command.encode_copy(&result, (&mut readback).subrange_mut(begin..end));
                        prefix += count;
                    }
                }
                command.end_encoding().submit().wait_until_completed().unwrap();
                let measured = record::<B>(
                    &readback,
                    &expected,
                    &output.join(format!("case_{case_index}_layer_{layer}_{mode}.bf16")),
                );
                if mode != "production_fused" {
                    assert!(measured["relative_rms"].as_f64().unwrap() < 0.0005, "same-input BF16 attention drift");
                }
                metrics.push(json!({"case_index":case_index,"layer":layer,"mode":mode,"metrics":measured}));
            }
        }
    }
    std::fs::write(output.join("metrics.json"), serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();
    println!("{}", json!({"matched_attention_metrics":metrics}));
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_decoder_matched_attention_metal() {
    attention::<crate::backends::metal::Metal>("metal");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_decoder_matched_attention_cpu() {
    attention::<crate::backends::cpu::Cpu>("cpu");
}

fn all_mlp_gates<B: Backend>(backend: &str) {
    use crate::backends::common::{
        gpu_types::ActivationType,
        kernel::{GatedActMul, GatedActMulSettings, ReferenceGatedSiLUBF16Kernel},
        vision_math::vision_exp,
    };
    let root = PathBuf::from(std::env::var_os("STARK_VISION_MLP_GATE_REFERENCE_PATH").unwrap());
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let context = B::Context::new().unwrap();
    let output = root.join(format!("native-{backend}"));
    std::fs::create_dir_all(&output).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let kernel =
        GatedActMul::<B>::full_precision(&context, DataType::BF16, true, false, GatedActMulSettings::default())
            .unwrap();
    let reference_kernel = <B::Kernels as Kernels>::ReferenceGatedSiLUBF16Kernel::new(&context).unwrap();
    let mut metrics = Vec::new();
    for stage in manifest["files"].as_array().unwrap() {
        let case_index = stage["case_index"].as_u64().unwrap();
        let layer = stage["layer"].as_u64().unwrap();
        let rows = stage["rows"].as_u64().unwrap() as u32;
        let width = stage["gated_dim"].as_u64().unwrap() as u32;
        let packed = bytes(&root, &stage["packed"]);
        let expected = bytes(&root, &stage["expected"]);
        let mut input = context.create_buffer(packed.len()).unwrap();
        (&mut input).copyin(&packed);
        let mut actual = context.create_buffer(expected.len()).unwrap();
        let mut command = context.create_command_buffer(None, None).unwrap();
        kernel.encode_fp(
            &input,
            None::<&B::GlobalBuffer>,
            &mut actual,
            None::<&B::GlobalBuffer>,
            width,
            rows,
            0,
            0,
            ActivationType::SILU,
            &mut command,
        );
        command.end_encoding().submit().wait_until_completed().unwrap();
        let native =
            record::<B>(&actual, &expected, &output.join(format!("case_{case_index}_layer_{layer}_native.bf16")));
        let mut command = context.create_command_buffer(None, None).unwrap();
        reference_kernel.encode(&input, &mut actual, width, rows, &mut command);
        command.end_encoding().submit().wait_until_completed().unwrap();
        let reference =
            record::<B>(&actual, &expected, &output.join(format!("case_{case_index}_layer_{layer}_reference.bf16")));
        assert_eq!(reference["bf16_mismatches"].as_u64(), Some(0), "source reference SiLU mismatch");
        let values = packed
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
            .collect::<Vec<_>>();
        let mut source = Vec::with_capacity(expected.len());
        for row in 0..rows as usize {
            for column in 0..width as usize {
                let begin = row * 2 * width as usize;
                let value = values[begin + column].to_f32();
                let gate = values[begin + width as usize + column].to_f32();
                let activated = bf16::from_f32(gate / (1.0 + vision_exp(-gate))).to_f32();
                source.extend_from_slice(&bf16::from_f32(value * activated).to_bits().to_le_bytes());
            }
        }
        let source_mismatches = source.chunks_exact(2).zip(expected.chunks_exact(2)).filter(|(a, b)| a != b).count();
        metrics.push(json!({"case_index":case_index,"layer":layer,"native":native,"reference":reference,"source_sleef_bf16_mismatches":source_mismatches}));
    }
    std::fs::write(output.join("metrics.json"), serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();
    println!("{}", json!({"all_mlp_gate_metrics":metrics}));
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn qwen35_vision_decoder_all_mlp_gates_metal() {
    all_mlp_gates::<crate::backends::metal::Metal>("metal");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn qwen35_vision_decoder_all_mlp_gates_cpu() {
    all_mlp_gates::<crate::backends::cpu::Cpu>("cpu");
}
