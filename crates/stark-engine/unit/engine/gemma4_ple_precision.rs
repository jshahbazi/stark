//! Private, identical-input PLE producer control. Never run by CI.

use std::{
    fs::{self, File},
    path::PathBuf,
};

use half::bf16;
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        metal::Metal,
    },
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::per_layer_embedding::PerLayerEmbedding,
    parameters::ParameterLoader,
};

fn probe() -> Result<(), &'static str> {
    let path = |key| std::env::var_os(key).map(PathBuf::from).ok_or("private_path_missing");
    let model = path("STARK_PLE_MODEL")?;
    let fixture = path("STARK_PLE_SOURCE_FIXTURE")?;
    let cases_path = path("STARK_PLE_CASES")?;
    let report_path = path("STARK_PLE_OUTPUT")?;
    let raw = fs::read(fixture.join("receipt.json")).map_err(|_| "receipt_read_failed")?;
    let receipt: serde_json::Value = serde_json::from_slice(&raw).map_err(|_| "receipt_invalid")?;
    if receipt["device"] != "gpu"
        || receipt["mlx_lm_revision"] != "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
        || receipt["trace_geometry"]["input_tokens"] != 511
    {
        return Err("source_identity_invalid");
    }
    let case_bytes = fs::read(cases_path).map_err(|_| "cases_read_failed")?;
    if receipt["cases_sha256"] != format!("{:x}", Sha256::digest(&case_bytes)) {
        return Err("cases_identity_invalid");
    }
    let cases: serde_json::Value = serde_json::from_slice(&case_bytes).map_err(|_| "cases_invalid")?;
    let mut tokens: Vec<u32> =
        serde_json::from_value(cases["cases"][0]["tokens"].clone()).map_err(|_| "tokens_invalid")?;
    tokens.pop();
    let input_bytes: Vec<_> = tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
    if tokens.len() != 511 || receipt["trace_geometry"]["input_sha256"] != format!("{:x}", Sha256::digest(&input_bytes))
    {
        return Err("prefix_identity_invalid");
    }
    let config_bytes = fs::read(model.join("config.json")).map_err(|_| "config_read_failed")?;
    let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|_| "config_invalid")?;
    let config: DecoderConfig =
        serde_json::from_value(config_json["decoder_config"].clone()).map_err(|_| "decoder_config_invalid")?;
    let ple = config.ple_model_config.as_ref().ok_or("ple_missing")?;
    if ple.num_layers != 35 || ple.ple_dim != 256 || config.transformer_config.model_dim != 1536 {
        return Err("ple_geometry_invalid");
    }
    let read = |name: &str, shape: &[usize]| -> Result<Vec<bf16>, &'static str> {
        let identity = receipt["bos_trace_components"]
            .as_array()
            .ok_or("components_missing")?
            .iter()
            .find(|entry| entry["component"] == name)
            .ok_or("component_missing")?;
        let bytes =
            fs::read(fixture.join("trace-prefill").join(format!("{name}.f32"))).map_err(|_| "component_read_failed")?;
        if identity["shape"] != serde_json::json!(shape)
            || bytes.len() != shape.iter().product::<usize>() * 4
            || identity["sha256"] != format!("{:x}", Sha256::digest(&bytes))
        {
            return Err("component_identity_invalid");
        }
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                let value = f32::from_le_bytes(chunk.try_into().map_err(|_| "component_invalid")?);
                let narrowed = bf16::from_f32(value);
                if !value.is_finite() || value.to_bits() != narrowed.to_f32().to_bits() {
                    return Err("component_not_bf16");
                }
                Ok(narrowed)
            })
            .collect()
    };
    let embedding = read("embedding", &[1, 511, 1536])?;
    let reference = read("ple-input-0", &[1, 511, 256])?;
    let context = <Metal as Backend>::Context::new().map_err(|_| "context_failed")?;
    let weights = File::open(model.join("model.safetensors")).map_err(|_| "weights_read_failed")?;
    let loader = ParameterLoader::<Metal>::new(&weights, &context).map_err(|_| "loader_failed")?;
    let producer = PerLayerEmbedding::<Metal>::new(
        &context,
        ple,
        1536,
        DataType::BF16,
        &loader.tree().subtree("decoder.per_layer_embedding"),
    )
    .map_err(|_| "producer_load_failed")?;
    let mut commands = context
        .create_command_buffer(Some("private identical-input PLE probe"), None)
        .map_err(|_| "commands_failed")?;
    let input = commands.allocate_constant_from_slice(&embedding).map_err(|_| "input_failed")?;
    let token_buffer = commands.allocate_constant_from_slice(&tokens).map_err(|_| "tokens_failed")?;
    let result = producer.encode(&token_buffer, &input, 511, &mut commands).map_err(|_| "encode_failed")?;
    let mut readback = context.create_buffer(511 * 35 * 256 * 2).map_err(|_| "readback_failed")?;
    commands.encode_copy(&result, &mut readback);
    drop(input);
    drop(token_buffer);
    commands.end_encoding().submit().wait_until_completed().map_err(|_| "execution_failed")?;
    let all: Vec<bf16> = readback.copyout();
    if all.len() != 511 * 35 * 256 || all.iter().any(|value| !value.is_finite()) {
        return Err("output_invalid");
    }
    let first: Vec<_> = all.chunks_exact(35 * 256).flat_map(|row| row[..256].iter().copied()).collect();
    let mut unequal = 0usize;
    let mut sum_square = 0.0f64;
    let mut max_abs = 0.0f64;
    for (actual, expected) in first.iter().zip(&reference) {
        let delta = f64::from(actual.to_f32()) - f64::from(expected.to_f32());
        unequal += usize::from(actual.to_bits() != expected.to_bits());
        sum_square += delta * delta;
        max_abs = max_abs.max(delta.abs());
    }
    let hash = |values: &[bf16]| {
        let bytes: Vec<_> = values.iter().flat_map(|value| value.to_f32().to_le_bytes()).collect();
        format!("{:x}", Sha256::digest(bytes))
    };
    let report = serde_json::json!({
        "backend":"metal", "operator":"ple-producer-layer-0", "acceptance":"diagnostic-no-numerical-gate",
        "source_receipt_sha256":format!("{:x}",Sha256::digest(raw)),
        "native_config_sha256":format!("{:x}",Sha256::digest(config_bytes)),
        "input_tokens":511,"elements":first.len(),"unequal_elements":unequal,"exact":unequal==0,
        "rmse":(sum_square/first.len() as f64).sqrt(),"max_abs_error":max_abs,
        "native_sha256":hash(&first),"reference_sha256":hash(&reference),
    });
    crate::tests::private_output::private_write(
        &report_path,
        &serde_json::to_vec(&report).map_err(|_| "report_invalid")?,
    )?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_ple_producer_probe() {
    let result = probe();
    assert!(result.is_ok(), "private PLE producer probe failed: {}", result.err().unwrap_or("unknown"));
}
