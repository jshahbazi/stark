//! Model-free independent pinned Torch Flash controls. No checkpoint vectors.

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{Kernels, kernel::Gemma4VisionFlashQKKernel},
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

fn check<B: Backend>() {
    let root = std::path::PathBuf::from(std::env::var_os("STARK_GEMMA4_FLASH_FIXTURE_PATH").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["scope"], "model-free-larger-vision-flash");
    assert_eq!(manifest["torch_source_revision"], "5c4886908584029761b579af026dcfb627c84070");
    assert_eq!(manifest["torch_version"].as_str().unwrap().split('+').next(), Some("2.14.1"));
    assert_eq!(manifest["oracle"], "aten._scaled_dot_product_flash_attention_for_cpu");
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    let context = B::Context::new().unwrap();
    let attention = flash::FlashAttention::<B>::new(&context).unwrap();
    let mut reports = Vec::new();
    for (case_index, case) in cases.iter().enumerate() {
        let keys = [31u32, 511, 513, 1025, 2394, 2520][case_index];
        assert_eq!(case["valid_key_rows"], keys);
        assert_eq!(case["source_padded_key_rows"], 2520);
        assert_eq!(case["query_rows"], 8);
        assert_eq!(case["heads"], 2);
        assert_eq!(case["logical_head_dim"], 72);
        assert_eq!(case["physical_head_dim"], 128);
        let load = |key: &str, count: usize| {
            let record = &case[key];
            let name = record["file"].as_str().unwrap();
            assert!(name.len() < 128 && !name.contains('/') && !name.contains('\\'));
            let bytes = std::fs::read(root.join(name)).unwrap();
            assert_eq!(bytes.len(), count * 2);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
            bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())))
                .collect::<Vec<_>>()
        };
        let q = create_buffer_with_data::<B, bf16>(&context, &load("queries", 2 * 8 * 128));
        let k = create_buffer_with_data::<B, bf16>(&context, &load("keys", keys as usize * 2 * 128));
        let v = create_buffer_with_data::<B, bf16>(&context, &load("values", keys as usize * 2 * 128));
        let expected = load("expected", 8 * 2 * 72);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let mut queries = commands.allocate_scratch_for_shape(&[2, 8, 128], DataType::BF16).unwrap();
        let mut key_buffer = commands.allocate_scratch_for_shape(&[keys, 2, 128], DataType::BF16).unwrap();
        let mut values = commands.allocate_scratch_for_shape(&[keys, 2, 128], DataType::BF16).unwrap();
        commands.encode_copy(&q, &mut queries);
        commands.encode_copy(&k, &mut key_buffer);
        commands.encode_copy(&v, &mut values);
        let output = attention.encode(&queries, &key_buffer, &values, 8, keys, 2, &mut commands).unwrap();
        let mut readback = context.create_buffer(8 * 2 * 72 * 2).unwrap();
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&readback);
        let different = actual.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        reports.push(
            serde_json::json!({"case_index":case_index,"valid_key_rows":keys,"different_bf16_elements":different}),
        );
    }
    let passed = reports.iter().all(|case| case["different_bf16_elements"] == 0);
    let report =
        serde_json::json!({"passed":passed,"scope":"model-free-larger-vision-flash","backend":B::NAME,"cases":reports});
    std::fs::write(root.join(format!("native-flash-{}.json", B::NAME)), serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    println!("{report}");
    assert!(passed, "canonical larger vision Flash differs");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn gemma4_larger_flash_canonical_cpu_fixture() {
    check::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma4_larger_flash_canonical_metal_fixture() {
    check::<crate::backends::metal::Metal>();
}

fn qk_check<B: Backend>() {
    let root = std::path::PathBuf::from(std::env::var_os("STARK_GEMMA4_FLASH_FIXTURE_PATH").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["torch_source_revision"], "5c4886908584029761b579af026dcfb627c84070");
    let witness = &manifest["qk_witness"];
    assert_eq!(witness["oracle"], "single-unmasked-key-pinned-flash-logsumexp");
    assert_eq!(witness["query_rows"], 256);
    assert_eq!(witness["key_rows"], 1);
    assert_eq!(witness["heads"], 1);
    assert_eq!(witness["source_padded_key_rows"], 2520);
    assert!(witness["legacy32_f32_differences"].as_u64().unwrap() > 0);
    let load = |key: &str, bytes: usize| {
        let record = &witness[key];
        let name = record["file"].as_str().unwrap();
        assert!(name.len() < 128 && !name.contains('/') && !name.contains('\\'));
        let value = std::fs::read(root.join(name)).unwrap();
        assert_eq!(value.len(), bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&value)), record["sha256"].as_str().unwrap());
        value
    };
    let decode = |bytes: Vec<u8>| {
        bytes
            .chunks_exact(2)
            .map(|value| bf16::from_bits(u16::from_le_bytes(value.try_into().unwrap())))
            .collect::<Vec<_>>()
    };
    let context = B::Context::new().unwrap();
    let q = create_buffer_with_data::<B, bf16>(&context, &decode(load("queries", 256 * 128 * 2)));
    let k = create_buffer_with_data::<B, bf16>(&context, &decode(load("keys", 128 * 2)));
    let expected = load("expected", 256 * 4)
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
    let kernel = <B::Kernels as Kernels>::Gemma4VisionFlashQKKernel::new(&context).unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut scores = command.allocate_scratch_for_shape(&[256, 512], DataType::F32).unwrap();
    kernel.encode(&q, &k, &mut scores, 256, 1, 1, 0, &mut command);
    let mut readback = context.create_buffer(256 * 512 * 4).unwrap();
    command.encode_copy(&scores, &mut readback);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, f32>(&readback);
    let different = (0..256).filter(|row| actual[row * 512].to_bits() != expected[*row].to_bits()).count();
    assert!(actual.chunks_exact(512).all(|row| row[1..].iter().all(|value| *value == f32::NEG_INFINITY)));
    let report = serde_json::json!({"scope":"independent-flash-f32-qk-witness","backend":B::NAME,
        "different_f32_elements":different,"legacy32_f32_differences":witness["legacy32_f32_differences"]});
    std::fs::write(root.join(format!("native-flash-qk-{}.json", B::NAME)), serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    println!("{report}");
    assert_eq!(different, 0, "pinned F32 QK source score differs");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn gemma4_larger_flash_qk_canonical_cpu_fixture() {
    qk_check::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma4_larger_flash_qk_canonical_metal_fixture() {
    qk_check::<crate::backends::metal::Metal>();
}
