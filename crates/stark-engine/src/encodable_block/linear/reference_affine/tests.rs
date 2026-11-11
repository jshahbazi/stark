use half::bf16;
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{
        Buffer, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::QuantizationMode,
        kernel::matmul::{QuantParamsLayout, QuantizedB, QuantizedCorrection},
    },
    encodable_block::linear::ReferenceBF16Matmul,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
};

struct Bank {
    codes: Vec<u8>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
    dense: Vec<bf16>,
    params: QuantParams,
}

fn quantized<'a, B: Backend>(
    codes: &'a B::GlobalBuffer,
    scales: &'a B::GlobalBuffer,
    biases: &'a B::GlobalBuffer,
    group_size: u32,
    signed_codes: bool,
    params: QuantParams,
) -> MatmulB<&'a B::GlobalBuffer> {
    MatmulB::Quantized(QuantizedB {
        codes,
        scales,
        correction: QuantizedCorrection::Biases(biases),
        params,
        mode: QuantizationMode::U4,
        group_size,
        signed_codes,
    })
}

fn bank(
    rows: u32,
    k: u32,
    bits: u32,
    layout: QuantParamsLayout,
) -> Bank {
    let params = QuantParams::new(layout, rows, k / 64);
    let shape = params.scale_shape();
    let mut result = Bank {
        codes: vec![0; (rows * k * bits / 8) as usize],
        scales: vec![bf16::NAN; (shape[0] * shape[1]) as usize],
        biases: vec![bf16::NAN; (shape[0] * shape[1]) as usize],
        dense: vec![bf16::ZERO; (rows * k) as usize],
        params,
    };
    let stride = params.scale_strides();
    let row_bytes = (k * bits / 8) as usize;
    for row in 0..rows {
        for inner in 0..k {
            let group = (row * stride.output_stride + inner / 64 * stride.group_stride) as usize;
            let scale = bf16::from_f32(0.1 + row as f32 / 32.0 + (inner / 64) as f32 / 64.0);
            let bias = bf16::from_f32(-0.2 - row as f32 / 16.0);
            result.scales[group] = scale;
            result.biases[group] = bias;
            let code = (row * 19 + inner * 7 + inner / 11) % (1 << bits);
            let bit = inner * bits;
            let offset = row as usize * row_bytes + (bit / 8) as usize;
            result.codes[offset] |= (code << (bit % 8)) as u8;
            if bit % 8 + bits > 8 {
                result.codes[offset + 1] |= (code >> (8 - bit % 8)) as u8;
            }
            result.dense[(row * k + inner) as usize] = bf16::from_f32(code as f32 * scale.to_f32() + bias.to_f32());
        }
    }
    result
}

#[stark_test]
fn affine_reference_plan_bounds_work_and_complete_k() {
    let plan = ReferenceAffinePlan::new(128, 34816, 5120).unwrap();
    assert!(plan.dispatches > 1 && plan.dispatches <= MAX_REFERENCE_DISPATCHES);
    assert!(u64::from(plan.tile_rows) * u64::from(plan.tile_columns) * u64::from(plan.k) <= MAX_TERMS_PER_DISPATCH);
    let mut outputs = 0_u64;
    for row in (0..plan.m).step_by(plan.tile_rows as usize) {
        for column in (0..plan.n).step_by(plan.tile_columns as usize) {
            outputs += u64::from(plan.tile_rows.min(plan.m - row)) * u64::from(plan.tile_columns.min(plan.n - column));
        }
    }
    assert_eq!(outputs, 128 * 34816);
    // Cover one full native prefill and the admitted top-32 expert expansion.
    assert!(ReferenceAffinePlan::new(1024, 34816, 5120).is_ok());
    assert!(ReferenceAffinePlan::new(8192, 1024, 2048).is_ok());
    assert!(ReferenceAffinePlan::new(32768, 1024, 2048).is_ok());
    for shape in [(0, 1, 64), (1, 0, 64), (1, 1, 0), (1, 1, 63), (32769, 1, 64), (4096, 1048576, 131072)] {
        assert!(ReferenceAffinePlan::new(shape.0, shape.1, shape.2).is_err());
    }
    assert!(ReferenceAffinePlan::with_limits(8, 9, 64, 63, 1 << 20, 1024).is_err());
    assert!(ReferenceAffinePlan::with_limits(8, 9, 64, 64, 1 << 20, 1).is_err());
}

fn source_boundaries<B: Backend>() {
    let context = B::Context::new().unwrap();
    let (m, rows, k) = (3, 7, 128);
    let input_values = (0..m * k).map(|index| bf16::from_f32((index % 31) as f32 / 16.0 - 1.0)).collect::<Vec<_>>();
    let mut prefixed = vec![bf16::from_f32(-99.0); 2];
    prefixed.extend(input_values);
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), &prefixed);
    let baseline = ReferenceBF16Matmul::<B>::new(context.as_ref()).unwrap();
    for bits in [0, 4, 5, 6, 8] {
        for layout in [QuantParamsLayout::OutputGroup, QuantParamsLayout::GroupOutput] {
            let data = bank(rows, k, bits.max(4), layout);
            let codes = create_buffer_with_data::<B, u8>(context.as_ref(), &data.codes);
            let scales = create_buffer_with_data::<B, bf16>(context.as_ref(), &data.scales);
            let biases = create_buffer_with_data::<B, bf16>(context.as_ref(), &data.biases);
            let dense = create_buffer_with_data::<B, bf16>(context.as_ref(), &data.dense);
            for gathered in [false, true] {
                let n = if gathered {
                    5
                } else {
                    rows
                };
                let selected = (0..m)
                    .flat_map(|row| (0..n).map(move |column| (row * 3 + column * 2) % rows))
                    .collect::<Vec<u32>>();
                let selected = create_buffer_with_data::<B, u32>(context.as_ref(), &selected);
                for dtype in [DataType::BF16, DataType::F32] {
                    let reference = ReferenceAffineBF16Matmul::<B>::new(context.as_ref(), dtype).unwrap();
                    let mut commands = context.create_command_buffer(None, None).unwrap();
                    let b = if bits == 0 {
                        MatmulB::FullPrecision {
                            b: &dense,
                        }
                    } else {
                        MatmulB::Quantized(QuantizedB {
                            codes: &codes,
                            scales: &scales,
                            correction: QuantizedCorrection::Biases(&biases),
                            params: data.params,
                            mode: match bits {
                                4 => QuantizationMode::U4,
                                5 => QuantizationMode::U5,
                                6 => QuantizationMode::U6,
                                8 => QuantizationMode::U8,
                                _ => unreachable!(),
                            },
                            group_size: 64,
                            signed_codes: false,
                        })
                    };
                    let actual =
                        reference.encode(&input, 4, b, gathered.then_some(&selected), m, n, k, &mut commands).unwrap();
                    let expected = if gathered {
                        baseline
                            .encode_gathered(&input, 4, &dense, &selected, m, n, rows, k, dtype, &mut commands)
                            .unwrap()
                    } else {
                        baseline.encode_output(&input, 4, &dense, m, n, k, dtype, &mut commands).unwrap()
                    };
                    let mut actual_out = context.create_buffer((m * n) as usize * dtype.size_in_bytes()).unwrap();
                    let mut expected_out = context.create_buffer(actual_out.size()).unwrap();
                    commands.encode_copy(&actual, &mut actual_out);
                    commands.encode_copy(&expected, &mut expected_out);
                    commands.end_encoding().submit().wait_until_completed().unwrap();
                    if dtype == DataType::BF16 {
                        assert_eq!(buffer_to_vec::<B, bf16>(&actual_out), buffer_to_vec::<B, bf16>(&expected_out));
                    } else {
                        assert_eq!(buffer_to_vec::<B, f32>(&actual_out), buffer_to_vec::<B, f32>(&expected_out));
                    }
                }
            }
        }
    }
}

#[stark_test]
fn affine_reference_matches_source_for_banks_widths_strides_and_router_widening() {
    for_each_backend!(|B| source_boundaries::<B>());
}

fn cancellation_tree<B: Backend>() {
    let context = B::Context::new().unwrap();
    let mut input = vec![bf16::ZERO; 128];
    input[0] = bf16::from_f32(67108864.0);
    input[4] = bf16::ONE;
    input[8] = bf16::from_f32(-67108864.0);
    // The first row's source tree returns 1; sequential F32 contraction
    // returns 0. The second row must round 1+1/256 to BF16 before F32 storage.
    input[64] = bf16::ONE;
    input[65] = bf16::from_f32(1.0 / 256.0);
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), &input);
    let full = create_buffer_with_data::<B, bf16>(context.as_ref(), &[bf16::ONE; 64]);
    for bits in [0, 4, 5, 6, 8] {
        let codes = create_buffer_with_data::<B, u8>(context.as_ref(), &vec![0; (64 * bits / 8).max(1)]);
        let params = QuantParams::new(QuantParamsLayout::GroupOutput, 1, 1);
        let shape = params.scale_shape();
        let plane = vec![bf16::ONE; (shape[0] * shape[1]) as usize];
        let plane = create_buffer_with_data::<B, bf16>(context.as_ref(), &plane);
        let b = if bits == 0 {
            MatmulB::FullPrecision {
                b: &full,
            }
        } else {
            MatmulB::Quantized(QuantizedB {
                codes: &codes,
                scales: &plane,
                correction: QuantizedCorrection::Biases(&plane),
                params,
                mode: match bits {
                    4 => QuantizationMode::U4,
                    5 => QuantizationMode::U5,
                    6 => QuantizationMode::U6,
                    8 => QuantizationMode::U8,
                    _ => unreachable!(),
                },
                group_size: 64,
                signed_codes: false,
            })
        };
        let reference = ReferenceAffineBF16Matmul::<B>::new(context.as_ref(), DataType::F32).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let output = reference.encode(&input, 0, b, None::<&B::GlobalBuffer>, 2, 1, 64, &mut commands).unwrap();
        let mut actual = context.create_buffer(8).unwrap();
        commands.encode_copy(&output, &mut actual);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(buffer_to_vec::<B, f32>(&actual), vec![1.0, 1.0]);
    }
}

#[stark_test]
fn affine_reference_preserves_source_tree_and_rounds_before_router_widening() {
    for_each_backend!(|B| cancellation_tree::<B>());
}

fn explicit_source_policy<B: Backend>() {
    let context = B::Context::new().unwrap();
    let codes = create_buffer_with_data::<B, u8>(context.as_ref(), &[0; 32]);
    let plane = create_buffer_with_data::<B, bf16>(context.as_ref(), &[bf16::ONE; 4]);
    let params = QuantParams::new(QuantParamsLayout::GroupOutput, 1, 1);
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), &[bf16::ONE; 64]);
    let reference = ReferenceAffineBF16Matmul::<B>::new(context.as_ref(), DataType::BF16).unwrap();
    let mut commands = context.create_command_buffer(None, None).unwrap();
    assert!(!has_source_affine_policy::<B>(&quantized::<B>(&codes, &plane, &plane, 64, false, params)));
    for params in [params.with_materialized_affine_input_sum(true), params.with_source_output_widths(&[1]).unwrap()] {
        assert!(has_source_affine_policy::<B>(&quantized::<B>(&codes, &plane, &plane, 64, false, params)));
        assert!(
            reference
                .encode(
                    &input,
                    0,
                    quantized::<B>(&codes, &plane, &plane, 64, false, params),
                    None::<&B::GlobalBuffer>,
                    1,
                    1,
                    64,
                    &mut commands
                )
                .is_err()
        );
    }
}

#[stark_test]
fn affine_reference_excludes_explicit_source_numerical_markers() {
    for_each_backend!(|B| explicit_source_policy::<B>());
}

fn malformed<B: Backend>() {
    let context = B::Context::new().unwrap();
    let reference = ReferenceAffineBF16Matmul::<B>::new(context.as_ref(), DataType::F32).unwrap();
    let data = bank(7, 128, 4, QuantParamsLayout::GroupOutput);
    let codes = create_buffer_with_data::<B, u8>(context.as_ref(), &data.codes);
    let short_codes = create_buffer_with_data::<B, u8>(context.as_ref(), &data.codes[..data.codes.len() - 1]);
    let scales = create_buffer_with_data::<B, bf16>(context.as_ref(), &data.scales);
    let short_scales = create_buffer_with_data::<B, bf16>(context.as_ref(), &data.scales[..data.scales.len() - 2]);
    let biases = create_buffer_with_data::<B, bf16>(context.as_ref(), &data.biases);
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), &vec![bf16::ONE; 3 * 128]);
    let short_indices = create_buffer_with_data::<B, u32>(context.as_ref(), &[0; 14]);
    let unaligned_indices = create_buffer_with_data::<B, u32>(context.as_ref(), &[0; 16]);
    let unaligned_input = create_buffer_with_data::<B, bf16>(context.as_ref(), &vec![bf16::ONE; 3 * 128 + 1]);
    let mut commands = context.create_command_buffer(None, None).unwrap();
    for (code_buffer, scale_buffer, group_size, signed, params, offset) in [
        (&short_codes, &scales, 64, false, data.params, 0),
        (&codes, &short_scales, 64, false, data.params, 0),
        (&codes, &scales, 32, false, data.params, 0),
        (&codes, &scales, 64, true, data.params, 0),
        (&codes, &scales, 64, false, QuantParams::new(QuantParamsLayout::GroupOutput, 9, 2), 0),
        (&codes, &scales, 64, false, data.params, 1),
        (&codes, &scales, 64, false, data.params, 2),
    ] {
        assert!(
            reference
                .encode(
                    &input,
                    offset,
                    quantized::<B>(code_buffer, scale_buffer, &biases, group_size, signed, params),
                    None::<&B::GlobalBuffer>,
                    3,
                    5,
                    128,
                    &mut commands
                )
                .is_err()
        );
    }
    assert!(
        reference
            .encode(
                &input,
                0,
                quantized::<B>(&codes, &scales, &biases, 64, false, data.params),
                Some(&short_indices),
                3,
                5,
                128,
                &mut commands
            )
            .is_err()
    );
    assert!(
        reference
            .encode(
                (&unaligned_input).subrange(1..),
                0,
                quantized::<B>(&codes, &scales, &biases, 64, false, data.params),
                None::<&B::GlobalBuffer>,
                3,
                5,
                128,
                &mut commands
            )
            .is_err()
    );
    assert!(
        reference
            .encode(
                &input,
                0,
                quantized::<B>(&codes, &scales, &biases, 64, false, data.params),
                Some((&unaligned_indices).subrange(1..)),
                3,
                5,
                128,
                &mut commands
            )
            .is_err()
    );
    let mut selected = vec![0_u32; 15];
    selected[0] = u32::MAX;
    let selected = create_buffer_with_data::<B, u32>(context.as_ref(), &selected);
    let output = reference
        .encode(
            &input,
            0,
            quantized::<B>(&codes, &scales, &biases, 64, false, data.params),
            Some(&selected),
            3,
            5,
            128,
            &mut commands,
        )
        .unwrap();
    let mut readback = context.create_buffer(15 * 4).unwrap();
    commands.encode_copy(&output, &mut readback);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    let values = buffer_to_vec::<B, f32>(&readback);
    assert!(values[0].is_nan() && values[1..].iter().all(|value| value.is_finite()));
}

#[stark_test]
fn affine_reference_rejects_malformed_bounds_and_masks_invalid_bank_rows() {
    for_each_backend!(|B| malformed::<B>());
}

fn mixed_partitions<B: Backend>() {
    use std::io::Write;

    use serde_json::{Value, json};

    use crate::{
        config::weight_matrix::{AnyWeightMatrixSpec, Layout, mlx_spec::MLXSpec},
        encodable_block::linear::{Linear, LinearMatmul, row_partitioned::RowPartitionedLinear},
        parameters::{NumericalProfile, ParameterLoader},
    };
    let context = B::Context::new().unwrap();
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut expected_weights = Vec::new();
    for (index, rows, bits) in [(0, 5, 4), (1, 7, 8)] {
        let data = bank(rows, 128, bits, QuantParamsLayout::GroupOutput);
        expected_weights.extend_from_slice(&data.dense);
        let plane = data.params.scale_shape();
        for (suffix, dtype, shape, bytes) in [
            ("weights", "U8", vec![rows, 128 * bits / 8], data.codes),
            ("scales", "BF16", plane.to_vec(), data.scales.iter().flat_map(|v| v.to_le_bytes()).collect()),
            ("biases", "BF16", plane.to_vec(), data.biases.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ] {
            let begin = payload.len();
            payload.extend_from_slice(&bytes);
            header.insert(
                format!("segment_{index}.{suffix}"),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[begin,payload.len()]}),
            );
        }
    }
    let spec = json!({"type":"MLXRowPartitionedSpec","layout":"output_input","segments":[
        {"rows":5,"bits":4,"group_size":64},{"rows":7,"bits":8,"group_size":64}]});
    header.insert("__metadata__".into(), json!({"spec":spec.to_string()}));
    let mut encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let path = std::env::temp_dir().join(format!(
        "stark-affine-partitions-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&encoded).unwrap();
    file.write_all(&payload).unwrap();
    let loader = ParameterLoader::<B>::new(&file, context.as_ref())
        .unwrap()
        .with_numerical_profile(NumericalProfile::Qwen35AffineBF16);
    // Numerical markers are independent of mixed-width storage. The same
    // admitted matrix selects the new encoder only while unmarked; both
    // explicit source-policy variants retain the existing backend route.
    for marked in [false, true] {
        let mut spec = MLXSpec::new(4, 64, Layout::OutputInput);
        spec.materialize_affine_input_sum = marked;
        let mut route = LinearMatmul::<B>::load(
            context.as_ref(),
            AnyWeightMatrixSpec::MLXSpec(spec),
            128,
            5,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            &loader.tree().subtree("segment_0"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(route.uses_reference_affine(), !marked);
        route.set_source_output_widths(&[5]).unwrap();
        assert_eq!(route.uses_reference_affine(), !marked);
    }
    let AnyWeightMatrixSpec::MLXRowPartitionedSpec(spec) = loader.tree().metadata("spec").unwrap() else {
        panic!("mixed fixture spec differs")
    };
    let matrix = RowPartitionedLinear::load(
        context.as_ref(),
        spec,
        128,
        12,
        false,
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
        &loader.tree(),
    )
    .unwrap();
    loader.tree().assert_all_tensors_validated().unwrap();
    let baseline = ReferenceBF16Matmul::<B>::new(context.as_ref()).unwrap();
    let weights = create_buffer_with_data::<B, bf16>(context.as_ref(), &expected_weights);
    let values = (0..3 * 128).map(|i| bf16::from_f32((i % 19) as f32 / 8.0 - 1.0)).collect::<Vec<_>>();
    let source = create_buffer_with_data::<B, bf16>(context.as_ref(), &values);
    let mut commands = context.create_command_buffer(None, None).unwrap();
    let expected = baseline.encode(&source, 0, &weights, 3, 12, 128, &mut commands).unwrap();
    let mut input = commands.allocate_scratch_for_shape(&[3, 128], DataType::BF16).unwrap();
    commands.encode_copy(&source, &mut input);
    let actual = matrix.encode(input, 3, &mut commands).unwrap();
    let mut actual_out = context.create_buffer(3 * 12 * 2).unwrap();
    let mut expected_out = context.create_buffer(3 * 12 * 2).unwrap();
    commands.encode_copy(&actual, &mut actual_out);
    commands.encode_copy(&expected, &mut expected_out);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    assert_eq!(buffer_to_vec::<B, bf16>(&actual_out), buffer_to_vec::<B, bf16>(&expected_out));
}

#[stark_test]
fn affine_reference_preserves_mixed_row_partition_outputs() {
    for_each_backend!(|B| mixed_partitions::<B>());
}

fn embedding_factory<B: Backend>() {
    use std::io::Write;

    use serde_json::json;

    use crate::{
        config::embedding::AnyEmbeddingConfig,
        encodable_block::embedding::Embedding,
        parameters::{NumericalProfile, ParameterLoader},
    };
    let context = B::Context::new().unwrap();
    let (vocab, k, m) = (7, 128, 3);
    // Real 27B topology: Q8 input_output input lookup, independent Q6
    // output_input readout. Also exercise tied input_output readout itself.
    for tied in [false, true] {
        let input = bank(vocab, k, 8, QuantParamsLayout::OutputGroup);
        let readout = bank(vocab, k, 6, QuantParamsLayout::GroupOutput);
        let mut metadata = serde_json::Map::new();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (prefix, data, bits, layout) in if tied {
            vec![("embedding", &input, 8, "input_output")]
        } else {
            vec![("input_embedding", &input, 8, "input_output"), ("output_embedding", &readout, 6, "output_input")]
        } {
            metadata.insert(
                format!("{prefix}.spec"),
                json!(json!({"type":"MLXSpec","bits":bits,"group_size":64,"layout":layout}).to_string()),
            );
            for (suffix, dtype, shape, bytes) in [
                ("weights", "U8", vec![vocab, k * bits / 8], data.codes.clone()),
                (
                    "scales",
                    "BF16",
                    data.params.scale_shape().to_vec(),
                    data.scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                (
                    "biases",
                    "BF16",
                    data.params.scale_shape().to_vec(),
                    data.biases.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            ] {
                let begin = payload.len();
                payload.extend_from_slice(&bytes);
                header.insert(
                    format!("{prefix}.{suffix}"),
                    json!({"dtype":dtype,"shape":shape,"data_offsets":[begin,payload.len()]}),
                );
            }
        }
        header.insert("__metadata__".into(), metadata.into());
        let mut encoded = serde_json::to_vec(&header).unwrap();
        encoded.resize(encoded.len().next_multiple_of(8), b' ');
        let path = std::env::temp_dir().join(format!(
            "stark-affine-embedding-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&encoded).unwrap();
        file.write_all(&payload).unwrap();
        let loader = ParameterLoader::<B>::new(&file, context.as_ref())
            .unwrap()
            .with_numerical_profile(NumericalProfile::Qwen35AffineBF16);
        let config: AnyEmbeddingConfig = serde_json::from_value(json!({"type":if tied {"TiedEmbeddingConfig"} else {"UntiedEmbeddingConfig"}, "input_scale":null,"logit_soft_cap":null,"logit_scale":null})).unwrap();
        let (embedding, factors) =
            Embedding::new(context.as_ref(), vocab, k, &config, &loader.tree(), DataType::BF16).unwrap();
        assert!(factors.is_none());
        loader.tree().assert_all_tensors_validated().unwrap();
        let values = (0..m * k).map(|i| bf16::from_f32((i % 23) as f32 / 8.0 - 1.0)).collect::<Vec<_>>();
        let values = create_buffer_with_data::<B, bf16>(context.as_ref(), &values);
        let tokens = create_buffer_with_data::<B, u32>(context.as_ref(), &[6, 0, 3]);
        let weights = create_buffer_with_data::<B, bf16>(
            context.as_ref(),
            if tied {
                &input.dense
            } else {
                &readout.dense
            },
        );
        let baseline = ReferenceBF16Matmul::<B>::new(context.as_ref()).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let expected = baseline.encode(&values, 0, &weights, m, vocab, k, &mut commands).unwrap();
        let actual =
            embedding.encode_readout(m, &values, vocab, None::<&B::GlobalBuffer>, false, &mut commands).unwrap();
        let lookup = embedding.encode_lookup(&tokens, m, &mut commands).unwrap();
        let mut actual_out = context.create_buffer((m * vocab * 2) as usize).unwrap();
        let mut expected_out = context.create_buffer((m * vocab * 2) as usize).unwrap();
        let mut lookup_out = context.create_buffer((m * k * 2) as usize).unwrap();
        commands.encode_copy(&actual, &mut actual_out);
        commands.encode_copy(&expected, &mut expected_out);
        commands.encode_copy(&lookup, &mut lookup_out);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(buffer_to_vec::<B, bf16>(&actual_out), buffer_to_vec::<B, bf16>(&expected_out));
        let lookup_expected = [6, 0, 3]
            .into_iter()
            .flat_map(|token| input.dense[token * k as usize..(token + 1) * k as usize].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(buffer_to_vec::<B, bf16>(&lookup_out), lookup_expected);
    }
}

#[stark_test]
fn affine_reference_embedding_factory_preserves_real_input_output_layouts() {
    for_each_backend!(|B| embedding_factory::<B>());
}
