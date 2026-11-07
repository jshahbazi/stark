use std::io::Write;

use half::bf16;
use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{BufferMut, CommandBufferExecutable, CommandBufferPending, Context},
    parameters::{NumericalProfile, ParameterLoader},
    tests::helpers::{buffer_to_vec, for_each_backend},
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

fn full_qwen_reference<B: Backend>() {
    let context = B::Context::new().unwrap();
    let file = fixture();
    let loader =
        ParameterLoader::new(&file, &context).unwrap().with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
    let config = serde_json::from_value(json!({"expert_config":{"linear_config":{},
        "activation":{"type":"SiLU","alpha":1.0},"has_up_biases":false,"has_down_biases":false,
        "up_clipping":null,"gate_clipping":null},"router_config":{},
        "routing_function":{"type":"SoftmaxRouting"},"num_routed_experts":4,
        "num_active_routed_experts":3,"router_has_biases":false,"num_shared_experts":1,
        "expert_hidden_dim":64,"gate_config":{}}))
    .unwrap();
    let model = QwenMoe::<B>::new(&config, 64, 64, &context, &loader.tree(), DataType::BF16).unwrap();
    loader.tree().assert_all_tensors_validated().unwrap();
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let mut input = commands.allocate_scratch_for_shape(&[4, 64], DataType::BF16).unwrap();
    let values: Vec<bf16> =
        [1.0f32, 0.5, -0.5, -1.0].into_iter().flat_map(|x| std::iter::repeat_n(bf16::from_f32(x), 64)).collect();
    (&mut input).copyin(bytemuck::cast_slice(&values));
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

#[cfg(backend = "cpu")]
#[stark_test]
fn qwen_bf16_moe_reference_scratch_stays_bounded_across_rows_and_routes() {
    use crate::backends::cpu::Cpu;

    let context = <Cpu as Backend>::Context::new().unwrap();
    let file = fixture();
    let loader =
        ParameterLoader::new(&file, &context).unwrap().with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
    let config = serde_json::from_value(json!({"expert_config":{"linear_config":{},
        "activation":{"type":"SiLU","alpha":1.0},"has_up_biases":false,"has_down_biases":false,
        "up_clipping":null,"gate_clipping":null},"router_config":{},
        "routing_function":{"type":"SoftmaxRouting"},"num_routed_experts":4,
        "num_active_routed_experts":3,"router_has_biases":false,"num_shared_experts":1,
        "expert_hidden_dim":64,"gate_config":{}}))
    .unwrap();
    let model = QwenMoe::<Cpu>::new(&config, 64, 64, &context, &loader.tree(), DataType::BF16).unwrap();
    let pool = context.create_allocation_pool();
    let mut commands = context.create_command_buffer(None, Some(pool.clone())).unwrap();
    let mut input = commands.allocate_scratch_for_shape(&[32, 64], DataType::BF16).unwrap();
    (&mut input).copyin(bytemuck::cast_slice(&vec![bf16::ONE; 32 * 64]));
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
