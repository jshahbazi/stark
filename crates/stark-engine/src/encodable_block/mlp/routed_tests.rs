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
    banks_with_experts(packed, 4)
}

fn banks_with_experts(
    packed: bool,
    experts: usize,
) -> std::fs::File {
    assert!((1..=8).contains(&experts));
    let (model, hidden) = (64usize, 64usize);
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

fn bf16_reference_banks<B: Backend>() {
    let context = B::Context::new().unwrap();
    let file = banks(false);
    let loader = ParameterLoader::new(&file, &*context)
        .unwrap()
        .with_numerical_profile(crate::parameters::NumericalProfile::Qwen35ReferenceBF16);
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
        4,
        &config,
        &loader.tree(),
        true,
        RoutingPrecision {
            round_logits_bf16: true,
            round_weights_bf16: true,
            round_products_bf16: true,
            sort_expert_ids: true,
            round_accumulator_bf16: true,
        },
    )
    .unwrap();
    let input = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ONE; 128]);
    // Ranking differs from expert ID order and differs between tokens. This
    // exercises full bank gathers, route/weight pairing and BF16 partial sums.
    let logits = [0.5f32, 2.0, 1.0, 3.0, 2.0, 3.0, 0.5, 1.0];
    let router = create_buffer_with_data::<B, f32>(&context, &logits);
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let output = experts.encode(&input, &router, 2, &mut commands).unwrap();
    let mut result = context.create_buffer(128 * 2).unwrap();
    commands.encode_copy(&output, &mut result);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, bf16>(&result);
    let round = |x| bf16::from_f32(x).to_f32();
    let silu = round(1.0 / (1.0 + (-1.0f32).exp()));
    for row in 0..2 {
        let source = &logits[row * 4..(row + 1) * 4];
        let denominator: f32 = source.iter().map(|x| (x - 3.0).exp()).sum();
        let mut expected = 0.0f32;
        for (expert, logit) in source.iter().enumerate() {
            let weight = round((logit - 3.0).exp() / denominator);
            let activated = round(silu * ((expert + 1) as f32 * 0.25));
            expected = round(expected + round(activated * weight));
        }
        for value in &actual[row * 64..(row + 1) * 64] {
            assert!((value.to_f32() - expected).abs() <= 0.004, "reference full bank row {row}");
        }
    }
}

#[stark_test]
fn full_bf16_expert_banks_apply_reference_routes_products_and_partial_sums() {
    for_each_backend!(|B| bf16_reference_banks::<B>());
}

/// Synthetic affine4/group64 operator attribution, not a full-model memory or
/// transformer parity proof. Run under the external 30-second resource guard.
#[cfg(backend = "metal")]
#[ignore]
#[stark_test]
fn routed_affine_command_heap_curve_and_completed_allocator_reuse() {
    use crate::{backends::metal::Metal, parameters::NumericalProfile};

    let started = std::time::Instant::now();
    let context = <Metal as Backend>::Context::new().unwrap();
    let file = banks_with_experts(true, 8);
    let loader =
        ParameterLoader::new(&file, &*context).unwrap().with_numerical_profile(NumericalProfile::Qwen35AffineBF16);
    let config = DenseMLPConfig::unclipped(
        serde_json::from_value(json!({})).unwrap(),
        serde_json::from_value(json!({"type":"SiLU","alpha":1.0})).unwrap(),
        false,
        false,
    );
    let experts = RoutedExperts::<Metal>::new(
        &context,
        64,
        64,
        8,
        8,
        &config,
        &loader.tree(),
        true,
        RoutingPrecision {
            round_logits_bf16: true,
            round_weights_bf16: true,
            round_products_bf16: true,
            sort_expert_ids: true,
            round_accumulator_bf16: true,
        },
    )
    .unwrap();
    let mut curve: Vec<Value> = Vec::new();
    let mut previous_probe = None;
    let mut baseline_256 = None;
    let mut baseline_1024 = None;
    let mut split_1024 = Vec::new();
    // The four final chunks test exact operator row splitting as well as
    // bounded reuse of the largest previously allocated command heap.
    for (index, rows) in [16usize, 64, 256, 1024, 256, 256, 256, 256].into_iter().enumerate() {
        assert!(started.elapsed().as_secs() < 25, "command heap probe exceeded internal deadline");
        let row_start = index.saturating_sub(4) * 256;
        let input = create_buffer_with_data::<Metal, bf16>(
            &context,
            &(row_start..row_start + rows)
                .flat_map(|row| {
                    (0..64).map(move |channel| {
                        // Two exact BF16 base-32 digits identify every absolute
                        // row through 1023, independently of router periodicity.
                        let digit = match channel {
                            0 => row % 32,
                            1 => row / 32,
                            _ => (row * (channel * 2 + 1) + channel) % 32,
                        };
                        bf16::from_f32(1.0 + digit as f32 / 32.0)
                    })
                })
                .collect::<Vec<_>>(),
        );
        let logits = create_buffer_with_data::<Metal, f32>(
            &context,
            &(row_start..row_start + rows)
                .flat_map(|row| (0..8).map(move |expert| ((expert + row) % 8) as f32 * 0.25))
                .collect::<Vec<_>>(),
        );
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let probe = commands.validation_heap_probe();
        let reused = previous_probe.as_ref().map(|previous| probe.same_allocator(previous));
        if index > 0 {
            assert_eq!(reused, Some(true), "completed allocator was not reused");
        }
        let heap_before = probe.allocated_bytes();
        let output = experts.encode(&input, &logits, rows as u32, &mut commands).unwrap();
        let mut readback = context.create_buffer(rows * 64 * 2).unwrap();
        commands.encode_copy(&output, &mut readback);
        let (constant_bytes, scratch_bytes) = commands.validation_payload_reserved_bytes();
        let heap_encoded = probe.allocated_bytes();
        // Constants and route scratch are no longer CPU-live at end_encoding.
        drop(output);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        // The completion callback sends feedback before publishing its cache
        // entry. Wait for that publication before asking new() to reset/reuse.
        while !probe.is_cached_after_completion() {
            assert!(started.elapsed().as_secs() < 25, "allocator completion cache publication timed out");
            std::thread::yield_now();
        }
        let heap_completed = probe.allocated_bytes();
        let actual = buffer_to_vec::<Metal, bf16>(&readback);
        assert!(actual.iter().all(|value| value.to_f32().is_finite()));
        if index == 2 {
            baseline_256 = Some(actual.clone());
        } else if index == 3 {
            baseline_1024 = Some(actual.clone());
        } else if index >= 4 {
            if index == 4 {
                assert!(actual == *baseline_256.as_ref().unwrap(), "operator result changed after allocator reset");
            }
            assert!(
                actual == baseline_1024.as_ref().unwrap()[row_start * 64..(row_start + rows) * 64],
                "operator row offsets changed under split execution"
            );
            split_1024.extend_from_slice(&actual);
            assert!(
                heap_encoded <= curve[3]["heap_encoded_bytes"].as_u64().unwrap(),
                "command heap grew on equal smaller work"
            );
        }
        curve.push(json!({
            "rows":rows,"minimum_route_dispatches":rows*8*6,
            "heap_before_bytes":heap_before,"heap_encoded_bytes":heap_encoded,
            "heap_completed_bytes":heap_completed,"constant_reserved_bytes":constant_bytes,
            "scratch_reserved_bytes":scratch_bytes,"reused_completed_allocator":reused,
        }));
        previous_probe = Some(probe);
    }
    assert!(split_1024 == *baseline_1024.as_ref().unwrap(), "split affine expert rows differ");
    // An earlier periodic fixture could hide this wrong absolute-row offset.
    assert!(
        split_1024[..64] != baseline_1024.as_ref().unwrap()[248 * 64..249 * 64],
        "wrong-row-offset negative control did not distinguish outputs"
    );
    let output = std::env::var_os("STARK_TEST_COMMAND_HEAP_RECEIPT").expect("set private aggregate receipt path");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    serde_json::to_writer_pretty(
        options.open(output).unwrap(),
        &json!({
            "schema":1,"completed":true,"performance_eligible":false,
            "synthetic":true,"model_dim":64,"hidden_dim":64,"experts":8,"active":8,
            "weights":"affine4-group64","activation":"BF16","accumulator":"BF16-reference-routed",
            "operator_split_exact":true,"wrong_row_offset_negative":true,"allocator_reset_reuse":true,
            "reset_shrink_claim":false,"full_model_memory_claim":false,
            "elapsed_seconds":started.elapsed().as_secs_f64(),"curve":curve,
        }),
    )
    .unwrap();
}
