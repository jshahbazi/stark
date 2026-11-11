//! Independent model-free pinned Torch RMS controls; vectors stay private.

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{Kernels, kernel::Gemma4VisionNormBF16Kernel},
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

fn check<B: Backend>() {
    let root = std::path::PathBuf::from(std::env::var_os("STARK_GEMMA4_NORM_FIXTURE_PATH").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["source_sha256"], "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3");
    assert_eq!(manifest["scope"], "model-free-larger-vision-rms");
    assert_eq!(manifest["transformers_version"], "5.15.1");
    assert_eq!(manifest["torch_version"].as_str().unwrap().split('+').next(), Some("2.14.1"));
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel::new(&context).unwrap();
    let mut reports = Vec::new();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for (index, case) in cases.iter().enumerate() {
        let width = case["width"].as_u64().unwrap() as u32;
        let rows = case["rows"].as_u64().unwrap() as u32;
        assert_eq!(
            width,
            if index < 2 {
                72
            } else {
                1152
            }
        );
        assert_eq!(case["has_scale"].as_bool().unwrap(), index % 2 == 1);
        assert_eq!(case["epsilon"].as_f64().unwrap(), 1e-6);
        assert!(
            matches!(width, 72 | 1152)
                && rows
                    == if width == 72 {
                        4096
                    } else {
                        512
                    }
        );
        assert!(case["legacy_rounding_witnesses"].as_u64().unwrap() > 0);
        let load = |key: &str, length: usize| {
            let record = &case[key];
            let name = record["file"].as_str().unwrap();
            assert!(name.len() < 128 && !name.contains('/') && !name.contains('\\'));
            let bytes = std::fs::read(root.join(name)).unwrap();
            assert_eq!(bytes.len(), length * 2);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), record["sha256"].as_str().unwrap());
            bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())))
                .collect::<Vec<_>>()
        };
        let input = load("input", (rows * width) as usize);
        let gains = load("gains", width as usize);
        let expected = load("expected", (rows * width) as usize);
        let input = create_buffer_with_data::<B, bf16>(&context, &input);
        let gains = create_buffer_with_data::<B, bf16>(&context, &gains);
        let mut command = context.create_command_buffer(None, None).unwrap();
        let mut output = command.allocate_scratch_for_shape(&[rows, width], DataType::BF16).unwrap();
        kernel.encode(
            &input,
            &gains,
            &mut output,
            rows,
            width,
            1e-6,
            u32::from(case["has_scale"].as_bool().unwrap()),
            &mut command,
        );
        let mut readback = context.create_buffer((rows * width * 2) as usize).unwrap();
        command.encode_copy(&output, &mut readback);
        command.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&readback);
        let different = actual.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        reports.push(serde_json::json!({"width":width,"rows":rows,"has_scale":case["has_scale"],"different_bf16_elements":different}));
    }
    let passed = reports.iter().all(|report| report["different_bf16_elements"] == 0);
    let report =
        serde_json::json!({"passed":passed,"backend":B::NAME,"scope":"model-free-larger-vision-rms","cases":reports});
    std::fs::write(root.join(format!("native-norm-{}.json", B::NAME)), serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    println!("{report}");
    assert!(passed, "canonical larger vision RMS differs");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn gemma4_larger_norm_canonical_cpu_fixture() {
    check::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma4_larger_norm_canonical_metal_fixture() {
    check::<crate::backends::metal::Metal>();
}
