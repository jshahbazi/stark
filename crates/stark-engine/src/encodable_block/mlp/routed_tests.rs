use std::io::Write;

use half::bf16;
use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
    parameters::ParameterLoader,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
};

fn banks(packed: bool) -> std::fs::File {
    let (experts, model, hidden) = (4usize, 64usize, 64usize);
    let mut header = serde_json::Map::new();
    let mut metadata = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, rows, columns, up) in [
        ("up_projection.weights", experts * 2 * hidden, model, true),
        ("down_projection.weights", experts * model, hidden, false),
    ] {
        let mut values = vec![0u8; rows * columns];
        for expert in 0..experts {
            for channel in 0..64 {
                if up {
                    values[((expert * 128 + channel) * columns) + channel] = (expert + 1) as u8;
                    values[((expert * 128 + 64 + channel) * columns) + channel] = 4;
                } else {
                    values[((expert * 64 + channel) * columns) + channel] = 4;
                }
            }
        }
        let mut add = |suffix: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>| {
            let begin = payload.len();
            payload.extend_from_slice(&bytes);
            header.insert(
                format!("{name}.{suffix}"),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[begin,payload.len()]}),
            );
        };
        if packed {
            let bytes: Vec<u8> = values.chunks_exact(2).map(|pair| pair[0] | (pair[1] << 4)).collect();
            add("weights", "U8", &[rows, columns / 2], bytes);
            for (plane, value) in [("scales", 0.25f32), ("biases", 0.0)] {
                add(plane, "BF16", &[1, rows], (0..rows).flat_map(|_| bf16::from_f32(value).to_le_bytes()).collect());
            }
            metadata.insert(
                format!("{name}.spec"),
                json!(json!({"type":"MLXSpec","layout":"output_input","bits":4,"group_size":64}).to_string()),
            );
        } else {
            add(
                "weights",
                "BF16",
                &[rows, columns],
                values.iter().flat_map(|value| bf16::from_f32(f32::from(*value) * 0.25).to_le_bytes()).collect(),
            );
            metadata.insert(
                format!("{name}.spec"),
                json!(json!({"type":"FullPrecisionSpec","layout":"output_input"}).to_string()),
            );
        }
    }
    header.insert("__metadata__".into(), Value::Object(metadata));
    let mut encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let path = std::env::temp_dir().join(format!(
        "stark-moe-bank-{}-{}",
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

fn selected_banks<B: Backend>() {
    for packed in [false, true] {
        let context = B::Context::new().unwrap();
        let file = banks(packed);
        let loader = ParameterLoader::new(&file, &*context).unwrap();
        let config = DenseMLPConfig::unclipped(
            serde_json::from_value(json!({})).unwrap(),
            serde_json::from_value(json!({"type":"SiLU","alpha":1.0})).unwrap(),
            false,
            false,
        );
        let experts = RoutedExperts::<B>::new(
            &context,
            64,
            64,
            4,
            2,
            &config,
            &loader.tree(),
            true,
            RoutingPrecision {
                round_logits_bf16: true,
                round_weights_bf16: true,
                round_products_bf16: true,
                ..Default::default()
            },
        )
        .unwrap();
        let input = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ONE; 128]);
        // Equal leading experts exercise stable expert IDs, and a second row
        // picks different experts to detect row/route or flattened-bank aliasing.
        let logits = create_buffer_with_data::<B, f32>(&context, &[3.0, -2.0, -2.0, 3.0, -2.0, 3.0, 3.0, -2.0]);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let output = experts.encode(&input, &logits, 2, &mut commands).unwrap();
        let mut result = context.create_buffer(128 * 2).unwrap();
        commands.encode_copy(&output, &mut result);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&result);
        let silu = bf16::from_f32(1.0 / (1.0 + (-1.0f32).exp())).to_f32();
        let expected = |a: f32, b: f32| {
            let a = bf16::from_f32(bf16::from_f32(silu * a).to_f32() * 0.5).to_f32();
            let b = bf16::from_f32(bf16::from_f32(silu * b).to_f32() * 0.5).to_f32();
            bf16::from_f32(a + b).to_f32()
        };
        for (row, expected) in [expected(0.25, 1.0), expected(0.5, 0.75)].into_iter().enumerate() {
            for value in &actual[row * 64..(row + 1) * 64] {
                assert!((value.to_f32() - expected).abs() <= 0.004, "packed={packed}, row={row}");
            }
        }
    }
}

#[stark_test]
fn routed_experts_select_only_the_requested_full_and_packed_bank_rows() {
    for_each_backend!(|B| selected_banks::<B>());
}
