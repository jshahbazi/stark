//! Private, source-bound full-coordinate host-math oracle replay.

use std::path::Path;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use super::vision_sin_cos;

fn reference_file(
    root: &Path,
    manifest: &Value,
    name: &str,
) -> Vec<f32> {
    let definition = &manifest["files"][name];
    let file = definition["file"].as_str().expect("invalid private trig reference filename");
    let data = std::fs::read(root.join(file)).unwrap_or_else(|_| panic!("private trig reference unavailable"));
    let hash = format!("{:x}", Sha256::digest(&data));
    assert!(definition["sha256"].as_str() == Some(hash.as_str()), "private trig reference hash mismatch");
    assert!(data.len() == 65_536 * 4, "incomplete private trig reference domain");
    data.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

#[stark_test]
#[ignore]
fn vision_host_rotary_tables_match_pinned_sleef_full_coordinate_range() {
    let root = std::path::PathBuf::from(
        std::env::var_os("STARK_VISION_TRIG_REFERENCE_PATH").expect("set private trig reference directory"),
    );
    let metadata = std::fs::read(root.join("manifest.json"))
        .unwrap_or_else(|_| panic!("private trig reference manifest unavailable"));
    let manifest: Value =
        serde_json::from_slice(&metadata).unwrap_or_else(|_| panic!("invalid private trig reference manifest"));
    assert!(manifest["reference_threads"].as_u64() == Some(8), "unexpected private trig thread count");
    assert!(manifest["coordinate_count"].as_u64() == Some(4096), "incomplete private trig coordinate range");
    assert!(manifest["frequencies_per_axis"].as_u64() == Some(16), "unexpected private trig head width");
    let angles = reference_file(&root, &manifest, "angles");
    assert!(angles.iter().all(|x| x.is_finite() && (0.0..4096.0).contains(x)), "invalid private trig angle");
    let expected_cosines = reference_file(&root, &manifest, "cosines");
    let expected_sines = reference_file(&root, &manifest, "sines");
    let (cosines, sines) = vision_sin_cos(&angles);
    let cosine_mismatches = cosines.iter().zip(&expected_cosines).filter(|(a, e)| a.to_bits() != e.to_bits()).count();
    let sine_mismatches = sines.iter().zip(&expected_sines).filter(|(a, e)| a.to_bits() != e.to_bits()).count();
    let metrics = json!({
        "coordinate_count":4096,"frequencies_per_axis":16,"f32_elements_per_function":angles.len(),
        "reference_threads":8,"cosine_bit_mismatches":cosine_mismatches,"sine_bit_mismatches":sine_mismatches,
        "reference_manifest_sha256":format!("{:x}",Sha256::digest(&metadata)),
    });
    println!("{metrics}");
    std::fs::write(root.join("native-host-trig-metrics.json"), serde_json::to_vec_pretty(&metrics).unwrap())
        .unwrap_or_else(|_| panic!("private trig aggregate output unavailable"));
    assert!(cosine_mismatches == 0 && sine_mismatches == 0, "private full-coordinate host trig parity failed");
}
