use std::io::{Read, Seek, SeekFrom, Write};

use half::bf16;
use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{BufferRef, CommandBufferExecutable, CommandBufferPending, Context},
    parameters::{NumericalProfile, ParameterLoader},
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
};

fn fixture() -> std::fs::File {
    let mut header = serde_json::Map::new();
    let mut metadata = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, rows, columns) in [
        ("experts.up_projection.weights", 512usize, 64usize),
        ("experts.down_projection.weights", 256, 64),
        ("shared_expert.up_projection.weights", 128, 64),
        ("shared_expert.down_projection.weights", 64, 64),
        ("router.weights", 4, 64),
        ("shared_gate.weights", 1, 64),
    ] {
        let mut values = vec![bf16::ZERO; rows * columns];
        match name {
            "experts.up_projection.weights" => {
                for expert in 0..4 {
                    for channel in 0..64 {
                        values[(expert * 128 + channel) * 64 + channel] =
                            bf16::from_f32([0.25, -0.5, 0.75, -1.0][expert]);
                        values[(expert * 128 + 64 + channel) * 64 + channel] = bf16::ONE;
                    }
                }
            },
            "experts.down_projection.weights" => {
                for expert in 0..4 {
                    for channel in 0..64 {
                        values[(expert * 64 + channel) * 64 + channel] = bf16::from_f32([1.0, 1.0, -1.0, -1.0][expert]);
                    }
                }
            },
            "shared_expert.up_projection.weights" => {
                for channel in 0..64 {
                    values[channel * 64 + channel] = bf16::from_f32(0.5);
                    values[(64 + channel) * 64 + channel] = bf16::ONE;
                }
            },
            "shared_expert.down_projection.weights" => {
                for channel in 0..64 {
                    values[channel * 64 + channel] = bf16::ONE;
                }
            },
            "router.weights" => {
                for (expert, value) in [0.5, 2.0, 1.0, 3.0].into_iter().enumerate() {
                    values[expert * 64] = bf16::from_f32(value);
                }
            },
            "shared_gate.weights" => values[0] = bf16::from_f32(0.5),
            _ => unreachable!(),
        }
        let begin = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        header.insert(
            format!("{name}.weights"),
            json!({"dtype":"BF16","shape":[rows,columns],"data_offsets":[begin,payload.len()]}),
        );
        metadata.insert(
            format!("{name}.spec"),
            json!(json!({"type":"FullPrecisionSpec","layout":"output_input"}).to_string()),
        );
    }
    header.insert("__metadata__".into(), Value::Object(metadata));
    let mut encoded = serde_json::to_vec(&header).unwrap();
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let path = std::env::temp_dir().join(format!(
        "stark-qwen-bf16-moe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&encoded).unwrap();
    file.write_all(&payload).unwrap();
    file
}

fn reference_config() -> MixtureOfExpertsConfig {
    // The MLP superclass adds required tags to both configs. LinearConfig is
    // untagged; its empty router/gate objects match the emitted converter graph.
    serde_json::from_value(json!({"type":"MixtureOfExpertsConfig",
        "expert_config":{"type":"DenseMLPConfig","linear_config":{},
        "activation":{"type":"SiLU","alpha":1.0},"has_up_biases":false,"has_down_biases":false,
        "up_clipping":null,"gate_clipping":null},"router_config":{},
        "routing_function":{"type":"SoftmaxRouting"},"num_routed_experts":4,
        "num_active_routed_experts":3,"router_has_biases":false,"num_shared_experts":1,
        "expert_hidden_dim":64,"gate_config":{}}))
    .unwrap()
}

fn full_qwen_reference<B: Backend>() {
    let context = B::Context::new().unwrap();
    let file = fixture();
    let loader = ParameterLoader::new(&file, context.as_ref())
        .unwrap()
        .with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
    let config = reference_config();
    let model = QwenMoe::<B>::new(&config, 64, 64, &context, &loader.tree(), DataType::BF16).unwrap();
    loader.tree().assert_all_tensors_validated().unwrap();
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let mut input = commands.allocate_scratch_for_shape(&[4, 64], DataType::BF16).unwrap();
    let values: Vec<bf16> =
        [1.0f32, 0.5, -0.5, -1.0].into_iter().flat_map(|x| std::iter::repeat_n(bf16::from_f32(x), 64)).collect();
    let source = create_buffer_with_data::<B, bf16>(&context, &values);
    commands.encode_copy(&source, &mut input);
    let output = model.encode(input, 4, &mut commands).unwrap();
    let mut result = context.create_buffer(4 * 64 * 2).unwrap();
    commands.encode_copy(&output, &mut result);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, bf16>(&result);
    let round = |x| bf16::from_f32(x).to_f32();
    for (row, x) in [1.0f32, 0.5, -0.5, -1.0].into_iter().enumerate() {
        let logits = [0.5f32, 2.0, 1.0, 3.0].map(|value| round(value * x));
        let mut selected: Vec<usize> = (0..4).collect();
        selected.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
        selected.truncate(3);
        let maximum = logits[selected[0]];
        let all_sum: f32 = logits.iter().map(|logit| (logit - maximum).exp()).sum();
        let selected_sum: f32 = selected.iter().map(|expert| (logits[*expert] - maximum).exp() / all_sum).sum();
        selected.sort_unstable();
        let silu = round(x / (1.0 + (-x).exp()));
        let mut routed = 0.0;
        for expert in selected {
            let up = round([0.25, -0.5, 0.75, -1.0][expert] * x);
            let activation = round(silu * up);
            let down = round(activation * [1.0, 1.0, -1.0, -1.0][expert]);
            let weight = round(((logits[expert] - maximum).exp() / all_sum) / selected_sum);
            routed = round(routed + round(down * weight));
        }
        let shared = round(silu * round(0.5 * x));
        let shared_gate = round(1.0 / (1.0 + (-round(0.5 * x)).exp()));
        let expected = round(routed + round(shared * shared_gate));
        assert!(
            actual[row * 64..(row + 1) * 64].iter().all(|value| (value.to_f32() - expected).abs() <= 0.00390625),
            "Qwen BF16 MoE reference row {row}"
        );
    }
}

#[stark_test]
fn qwen_bf16_moe_reference_profile_executes_router_selected_banks_and_shared_gate() {
    for_each_backend!(|B| full_qwen_reference::<B>());
}

/// A small native expert-head integration fixture. This uses the same routed /
/// shared bank layout as the converter contract, not official model weights.
fn predictor_fixture(zero_mlp: bool) -> std::fs::File {
    let mut mlp = fixture();
    mlp.seek(SeekFrom::Start(0)).unwrap();
    let mut bytes = Vec::new();
    mlp.read_to_end(&mut bytes).unwrap();
    let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let original: Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
    let mut payload = bytes[8 + length..].to_vec();
    if zero_mlp {
        payload.fill(0);
    }
    let mut header = serde_json::Map::new();
    let mut metadata = serde_json::Map::new();
    for (name, record) in original.as_object().unwrap() {
        if name == "__metadata__" {
            for (name, spec) in record.as_object().unwrap() {
                metadata.insert(format!("mtp.transformer.layers.0.mlp.{name}"), spec.clone());
            }
        } else {
            header.insert(format!("mtp.transformer.layers.0.mlp.{name}"), record.clone());
        }
    }
    for (name, width) in [
        ("embedding_norm", 64),
        ("hidden_norm", 64),
        ("transformer.output_norm", 64),
        ("transformer.layers.0.pre_mixer_norm", 64),
        ("transformer.layers.0.pre_mlp_norm", 64),
        ("transformer.layers.0.mixer.query_norm", 128),
        ("transformer.layers.0.mixer.key_norm", 128),
    ] {
        let begin = payload.len();
        payload.resize(begin + width * 4, 0);
        header.insert(
            format!("mtp.{name}.scales"),
            json!({"dtype":"F32","shape":[width],"data_offsets":[begin,payload.len()]}),
        );
    }
    for (name, rows, columns) in [
        ("fusion.weights", 64usize, 128usize),
        ("transformer.layers.0.mixer.qkvg_projection.weights", 512, 64),
        ("transformer.layers.0.mixer.out_projection.weights", 64, 128),
    ] {
        let mut weights = vec![bf16::ZERO; rows * columns];
        if name == "fusion.weights" {
            for row in 0..64 {
                weights[row * 128 + row] = bf16::from_f32(0.5);
                weights[row * 128 + 64 + row] = bf16::from_f32(0.25);
            }
        } else if name.ends_with("qkvg_projection.weights") {
            // Nonzero Q/K/V make cache-only priming depend on the actual
            // preceding rows; zero gate logits retain the ordinary sigmoid.
            for channel in 0..64 {
                weights[channel * 64 + channel] = bf16::from_f32(0.25);
                weights[(128 + channel) * 64 + channel] = bf16::from_f32(0.25);
                weights[(256 + channel) * 64 + channel] = bf16::from_f32(0.5);
            }
        } else {
            for channel in 0..64 {
                weights[channel * 128 + channel] = bf16::from_f32(0.125);
            }
        }
        let begin = payload.len();
        payload.extend(weights.iter().flat_map(|value| value.to_le_bytes()));
        header.insert(
            format!("mtp.{name}.weights"),
            json!({"dtype":"BF16","shape":[rows,columns],"data_offsets":[begin,payload.len()]}),
        );
        metadata.insert(
            format!("mtp.{name}.spec"),
            json!(json!({"type":"FullPrecisionSpec","layout":"output_input"}).to_string()),
        );
    }
    header.insert("__metadata__".into(), Value::Object(metadata));
    let mut encoded = serde_json::to_vec(&header).unwrap();
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let path = std::env::temp_dir().join(format!(
        "stark-qwen-bf16-expert-predictor-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&encoded).unwrap();
    file.write_all(&payload).unwrap();
    file
}

fn expert_predictor_component<B: Backend>() -> Vec<f32> {
    use crate::{
        backends::common::gpu_types::trie::TrieNode,
        config::qwen_mtp::QwenMtpConfig,
        encodable_block::{batch_topology::BatchTopology, qwen_mtp::QwenMtp},
    };

    let context = B::Context::new().unwrap();
    let norm = json!({"epsilon":1e-6,"scale_offset":1.0,"upcast_mode":"full_layer", "subtract_mean":false,
        "has_scale":true,"has_biases":false});
    let config: QwenMtpConfig = serde_json::from_value(json!({"model_dim":64,"target_layer":1,
        "input_norm_config":norm,"transformer_config":{"model_dim":64,"hidden_dim":64,"output_norm_config":norm,
        "layer_configs":[{"pre_mixer_norm_config":norm,"pre_mlp_norm_config":norm,"has_post_layer_scalar":false,
            "mixer_conv_config":null,"post_mixer_norm_config":null,"mlp_conv_config":null,"post_mlp_norm_config":null,
            "hidden_dim":null,"ple_config":null,"kv_source_layer_index":null,"rope_config":null,
            "mixer_config":{"type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},
                "num_heads":1,"num_groups":1,"head_dim":128,"is_causal":true,"has_sinks":false,"has_qkvg_biases":false,
                "has_out_biases":false,"has_gate":true,"normalize_values":false,"is_kv_sharing":false,
                "query_norm_config":norm,"key_norm_config":norm,"scale":null,"sliding_window_size":null,
                "logit_soft_cap":null,"reference_bf16":true},"mlp_config":reference_config()}]}}))
    .unwrap();
    let embedding = (0..256).map(|i| bf16::from_f32(((i * 7 % 17) as f32 - 8.0) * 0.125)).collect::<Vec<_>>();
    let hidden = (0..256).map(|i| bf16::from_f32(((i * 11 % 19) as f32 - 9.0) * 0.0625)).collect::<Vec<_>>();
    let mut outputs = Vec::new();
    for zero_mlp in [false, true] {
        let file = predictor_fixture(zero_mlp);
        let loader = ParameterLoader::new(&file, context.as_ref())
            .unwrap()
            .with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
        let model = QwenMtp::<B>::new(&config, 2, &loader.tree().subtree("mtp"), &context).unwrap();
        loader.tree().assert_all_tensors_validated().unwrap();
        let mut state = model.create_state(16, &context).unwrap();
        let chain = |length| {
            (0..length)
                .map(|height| TrieNode {
                    height,
                    trie_start: height,
                    trie_end: length - 1,
                })
                .collect::<Vec<_>>()
        };
        let nodes = chain(4);
        let prefix_nodes = chain(3);
        let single_node = chain(1);
        state.prepare(0, 4).unwrap();
        let e = create_buffer_with_data::<B, bf16>(&context, &embedding);
        let h = create_buffer_with_data::<B, bf16>(&context, &hidden);
        let mut command = context.create_command_buffer(None, None).unwrap();
        let full = model.encode(&e, &h, &BatchTopology::new(&nodes, true), &mut state, &mut command).unwrap();
        let mut result = context.create_buffer(256 * 2).unwrap();
        command.encode_copy(&full.normalized, &mut result);
        state.encode_accept(&[0, 1, 2, 3], &mut command).unwrap();
        command.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&result).into_iter().map(f32::from).collect::<Vec<_>>();
        assert!(actual.iter().all(|value| value.is_finite()));
        assert_eq!(state.context_length(), 4);
        if !zero_mlp {
            let mut primed = model.create_state(16, &context).unwrap();
            primed.prepare(0, 3).unwrap();
            let mut command = context.create_command_buffer(None, None).unwrap();
            model
                .encode_cache_only(
                    e.subrange(..3 * 64 * 2),
                    h.subrange(..3 * 64 * 2),
                    &BatchTopology::new(&prefix_nodes, true),
                    &mut primed,
                    &mut command,
                )
                .unwrap();
            primed.encode_accept(&[0, 1, 2], &mut command).unwrap();
            command.end_encoding().submit().wait_until_completed().unwrap();
            primed.prepare(3, 1).unwrap();
            let mut command = context.create_command_buffer(None, None).unwrap();
            let last = model
                .encode(
                    e.subrange(3 * 64 * 2..),
                    h.subrange(3 * 64 * 2..),
                    &BatchTopology::new(&single_node, true),
                    &mut primed,
                    &mut command,
                )
                .unwrap();
            let mut last_result = context.create_buffer(64 * 2).unwrap();
            command.encode_copy(&last.normalized, &mut last_result);
            primed.encode_accept(&[0], &mut command).unwrap();
            command.end_encoding().submit().wait_until_completed().unwrap();
            assert_eq!(primed.context_length(), 4);
            assert!(
                buffer_to_vec::<B, bf16>(&last_result)
                    .iter()
                    .zip(&actual[3 * 64..])
                    .all(|(a, b)| (a.to_f32() - b).abs() <= 0.015625),
                "expert predictor cache-only priming differs"
            );
        }
        outputs.push(actual);
    }
    assert!(outputs[0].iter().zip(&outputs[1]).any(|(a, b)| (a - b).abs() > 0.01), "expert head MLP did not run");
    outputs.remove(0)
}

#[stark_test]
fn qwen_bf16_expert_predictor_executes_and_cache_only_priming_preserves_output() {
    let expected = expert_predictor_component::<crate::backends::cpu::Cpu>();
    assert_eq!(expected.len(), 256);
    #[cfg(backend = "metal")]
    {
        let output = expert_predictor_component::<crate::backends::metal::Metal>();
        assert!(
            output.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= 0.03125),
            "CPU/Metal synthetic expert predictor differs"
        );
    }
}

#[cfg(backend = "cpu")]
#[stark_test]
fn qwen_bf16_moe_reference_scratch_stays_bounded_across_rows_and_routes() {
    use crate::backends::cpu::Cpu;

    let context = <Cpu as Backend>::Context::new().unwrap();
    let file = fixture();
    let loader = ParameterLoader::new(&file, context.as_ref())
        .unwrap()
        .with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
    let config = reference_config();
    let model = QwenMoe::<Cpu>::new(&config, 64, 64, &context, &loader.tree(), DataType::BF16).unwrap();
    let pool = context.create_allocation_pool();
    let mut commands = context.create_command_buffer(None, Some(pool.clone())).unwrap();
    let mut input = commands.allocate_scratch_for_shape(&[32, 64], DataType::BF16).unwrap();
    let source = create_buffer_with_data::<Cpu, bf16>(&context, &vec![bf16::ONE; 32 * 64]);
    commands.encode_copy(&source, &mut input);
    let output = model.encode(input, 32, &mut commands).unwrap();
    // Count every scratch request in this command buffer, including retained
    // routes, outputs and activation buffers. Materializing up/down matrices
    // for each of the 96 routes would alone request over 2 MiB here.
    assert!(pool.requested_bytes() < 512 * 1024, "reference MoE scratch expanded into weight matrices");
    let mut result = context.create_buffer(32 * 64 * 2).unwrap();
    commands.encode_copy(&output, &mut result);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    assert!(buffer_to_vec::<Cpu, bf16>(&result).iter().all(|value| value.is_finite()));
}
