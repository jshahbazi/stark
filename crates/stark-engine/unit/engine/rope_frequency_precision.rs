//! Private, source-bound inspection of static rotary frequencies.

use std::{fs, path::PathBuf};

use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use crate::backends::{
    common::{Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    metal::{Metal, SourceRoPEFrequencyProbeMetalKernel},
};

fn probe() -> Result<(), &'static str> {
    let fixture = PathBuf::from(std::env::var_os("STARK_SOURCE_FREQ_FIXTURE").ok_or("fixture_missing")?);
    let output = PathBuf::from(std::env::var_os("STARK_SOURCE_FREQ_OUTPUT").ok_or("output_missing")?);
    let receipt_bytes = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read_failed")?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes).map_err(|_| "receipt_invalid")?;
    if receipt["engine"] != "mlx"
        || receipt["device"] != "gpu"
        || receipt["identity"]["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
    {
        return Err("source_identity_invalid");
    }
    let bytes = fs::read(fixture.join("global_proportional_frequencies.f32")).map_err(|_| "frequency_read_failed")?;
    let identity = receipt["operations"]
        .as_array()
        .ok_or("operations_missing")?
        .iter()
        .find(|entry| entry["operation"] == "global_proportional_frequencies")
        .ok_or("frequency_missing")?;
    if bytes.len() != 256 * 4
        || identity["elements"] != 256
        || identity["sha256"] != format!("{:x}", Sha256::digest(&bytes))
    {
        return Err("frequency_identity_invalid");
    }
    let reference: Vec<_> = bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect();
    let context = <Metal as Backend>::Context::new().map_err(|_| "context_failed")?;
    let kernel = SourceRoPEFrequencyProbeMetalKernel::new(&context, false).map_err(|_| "kernel_failed")?;
    let mut values = context.create_buffer(256 * 4).map_err(|_| "buffer_failed")?;
    let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands_failed")?;
    kernel.encode(&mut values, None::<&<Metal as Backend>::GlobalBuffer>, 1_000_000.0, 512, 128, &mut commands);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execution_failed")?;
    let actual: Vec<f32> = values.copyout();
    let mut unequal = 0;
    let mut max_relative = 0.0f64;
    for (actual, expected) in actual.iter().zip(&reference) {
        if actual.is_nan() || expected.is_nan() {
            return Err("frequency_nan");
        }
        unequal += usize::from(actual.to_bits() != expected.to_bits());
        if actual.is_finite() && expected.is_finite() {
            max_relative = max_relative.max((f64::from(*actual) - f64::from(*expected)).abs() / f64::from(*expected));
        }
    }
    let raw: Vec<_> = actual.iter().flat_map(|value| value.to_le_bytes()).collect();
    let report = serde_json::json!({"backend":"metal","operator":"global-proportional-frequencies",
        "base":1000000,"head_dim":512,"rotary_dim":128,"elements":256,"unequal_elements":unequal,
        "exact":unequal==0,"max_relative_error":max_relative,
        "source_receipt_sha256":format!("{:x}",Sha256::digest(receipt_bytes)),
        "reference_sha256":format!("{:x}",Sha256::digest(bytes)),
        "native_sha256":format!("{:x}",Sha256::digest(raw))});
    crate::tests::private_output::private_write(&output, &serde_json::to_vec(&report).map_err(|_| "report_invalid")?)?;
    Ok(())
}

#[stark_test]
#[ignore]
fn private_source_rope_frequency_probe() {
    let result = probe();
    assert!(result.is_ok(), "private frequency probe failed: {}", result.err().unwrap_or("unknown"));
}
