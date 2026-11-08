//! Current main-engine tanh-GELU measured independently of Lily dispatch.

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::*;
use crate::{
    backends::common::{Kernels, kernel::VisionGeluBF16Kernel},
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

fn check<B: Backend>() {
    let root = PathBuf::from(std::env::var_os("STARK_GEMMA4_GELU_FIXTURE_PATH").unwrap());
    let manifest_bytes = std::fs::read(root.join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_bytes)),
        "78f542f4bb6f10a760927ef5501ba5d9884d0f1c78bc430ecd5646f77c037644"
    );
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest["kind"], "RetainedLilyGeluDomainReference");
    assert_eq!(manifest["torch"], "2.14.1");
    assert_eq!(manifest["torch_git"], "5c4886908584029761b579af026dcfb627c84070");
    assert_eq!(manifest["finite_inputs"], 65280);
    let load = |key: &str| {
        let r = &manifest[key];
        let name = r["file"].as_str().unwrap();
        assert!(name.len() < 128 && !name.contains('/') && !name.contains('\\'));
        let bytes = std::fs::read(root.join(name)).unwrap();
        assert_eq!(bytes.len(), 65280 * 2);
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), r["sha256"].as_str().unwrap());
        bytes
            .chunks_exact(2)
            .map(|value| bf16::from_bits(u16::from_le_bytes(value.try_into().unwrap())))
            .collect::<Vec<_>>()
    };
    let input = load("input");
    assert!(input.iter().all(|value| value.is_finite()));
    assert_eq!(input.iter().map(|v| v.to_bits()).collect::<std::collections::BTreeSet<_>>().len(), 65280);
    let expected = load("tanh");
    let context = B::Context::new().unwrap();
    let values = create_buffer_with_data::<B, bf16>(&context, &input);
    let kernel = <B::Kernels as Kernels>::VisionGeluBF16Kernel::new(&context).unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut scratch = command.allocate_scratch_for_shape(&[65280], DataType::BF16).unwrap();
    command.encode_copy(&values, &mut scratch);
    kernel.encode(&mut scratch, 65280, 1, &mut command);
    let mut readback = context.create_buffer(65280 * 2).unwrap();
    command.encode_copy(&scratch, &mut readback);
    command.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<B, bf16>(&readback);
    let differing = input
        .iter()
        .zip(actual.iter().zip(&expected))
        .filter_map(|(input, (a, b))| {
            if a.to_bits() != b.to_bits() {
                Some(input.to_bits())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let normal_differences = differing.iter().filter(|bits| (**bits & 0x7f80) != 0).count();
    let output = PathBuf::from(std::env::var_os("STARK_GEMMA4_GELU_REPORT_PATH").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    // Private domain witnesses permit an intersection with captured activations;
    // only aggregate counts and hashes are emitted to the diagnostic log.
    let witness_bytes = serde_json::to_vec(&differing).unwrap();
    std::fs::write(output.join(format!("differing-inputs-{}.json", B::NAME)), &witness_bytes).unwrap();
    let report = json!({"scope":"current-main-vision-tanh-gelu-finite-domain","backend":B::NAME,
        "cases":65280,"different_bf16_elements":differing.len(),"normal_input_differences":normal_differences,
        "private_witness_sha256":format!("{:x}",Sha256::digest(&witness_bytes))});
    std::fs::write(output.join(format!("native-gelu-{}.json", B::NAME)), serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    println!("{report}");
    assert!(differing.is_empty(), "current main tanh GELU differs from pinned finite domain");
}

#[cfg(backend = "cpu")]
#[stark_test]
#[ignore]
fn gemma4_larger_main_gelu_cpu_fixture() {
    check::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
#[ignore]
fn gemma4_larger_main_gelu_metal_fixture() {
    check::<crate::backends::metal::Metal>();
}
