//! Matched warm text profiling with private inputs and aggregate-only output.
//! The timed request includes 512 prompt tokens, 128 greedy decode forwards,
//! 128 returned tokens and the final lookahead drain. EOS never stops a run.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::{Path, PathBuf},
    time::Instant,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{LanguageModel, stream::SamplingMethod},
    },
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceWeight {
    filename: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Quantization {
    mode: String,
    bits: Option<u32>,
    group_size: Option<u32>,
    overrides_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: u32,
    input_token_ids: Vec<u32>,
    decode_tokens: u32,
    context_tokens: u32,
    warmup_runs: u32,
    measured_runs: u32,
    source_repo: String,
    source_revision: String,
    source_config_sha256: String,
    source_weights: Vec<SourceWeight>,
    quantization: Quantization,
}

fn hex(
    value: &str,
    count: usize,
) -> bool {
    value.len() == count && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_file(path: &Path) -> Result<String, &'static str> {
    let mut file = File::open(path).map_err(|_| "component_open_failed")?;
    let before = file.metadata().map_err(|_| "component_metadata_failed")?;
    let mut digest = Sha256::new();
    let mut block = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut block).map_err(|_| "component_read_failed")?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    let after = file.metadata().map_err(|_| "component_metadata_failed")?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("component_changed_while_hashing");
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_fixture(fixture: &Fixture) -> Result<(), &'static str> {
    if fixture.schema_version != 1
        || fixture.input_token_ids.len() != 512
        || fixture.decode_tokens != 128
        || fixture.context_tokens < 641
        || fixture.context_tokens > 8192
        || fixture.warmup_runs != 1
        || fixture.measured_runs != 3
        || !hex(&fixture.source_revision, 40)
        || !hex(&fixture.source_config_sha256, 64)
        || fixture.source_weights.is_empty()
        || fixture.source_weights.len() > 64
        || fixture.source_repo.len() > 256
        || fixture.source_repo.split('/').count() != 2
        || !fixture.source_repo.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"/._-".contains(&byte))
    {
        return Err("invalid_profile_fixture");
    }
    for weight in &fixture.source_weights {
        if weight.bytes == 0
            || weight.bytes > 128 * 1024 * 1024 * 1024
            || !hex(&weight.sha256, 64)
            || !weight.filename.ends_with(".safetensors")
            || !weight.filename.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        {
            return Err("invalid_source_weight_identity");
        }
    }
    let quantization = &fixture.quantization;
    match quantization.mode.as_str() {
        "none" if quantization.bits.is_none() && quantization.group_size.is_none() => {},
        "affine"
            if matches!(quantization.bits, Some(4 | 8))
                && matches!(quantization.group_size, Some(16 | 32 | 64 | 128)) => {},
        _ => return Err("unsupported_profile_quantization"),
    }
    if quantization.overrides_sha256.as_ref().is_some_and(|hash| !hex(hash, 64)) {
        return Err("invalid_quantization_override_identity");
    }
    Ok(())
}

fn source_weights_digest(weights: &[SourceWeight]) -> String {
    let mut ordered: Vec<_> = weights.iter().collect();
    ordered.sort_by_key(|weight| &weight.filename);
    let mut digest = Sha256::new();
    for weight in ordered {
        digest.update(weight.filename.as_bytes());
        digest.update([0]);
        digest.update(weight.bytes.to_le_bytes());
        digest.update(weight.sha256.as_bytes());
        digest.update([b'\n']);
    }
    format!("{:x}", digest.finalize())
}

fn verify_receipt(
    model: &Path,
    fixture: &Fixture,
    weights_hash: &str,
    config_hash: &str,
) -> Result<String, &'static str> {
    let path = model.join("conversion.json");
    if fs::metadata(&path).map_err(|_| "missing_conversion_receipt")?.len() > 8 * 1024 * 1024 {
        return Err("conversion_receipt_too_large");
    }
    let receipt: Value = serde_json::from_slice(&fs::read(&path).map_err(|_| "unreadable_conversion_receipt")?)
        .map_err(|_| "invalid_conversion_receipt")?;
    if receipt["status"] != "ok"
        || receipt["output_sha256"] != weights_hash
        || receipt["config_sha256"] != config_hash
        || receipt["source_revision"] != fixture.source_revision
    {
        return Err("converted_model_identity_mismatch");
    }
    let files = receipt["source_files"].as_array().ok_or("missing_source_file_identities")?;
    let has_source = |hash: &str, bytes: Option<u64>| {
        files.iter().any(|file| {
            file["repo"] == fixture.source_repo
                && file["revision"] == fixture.source_revision
                && file["sha256"] == hash
                && bytes.is_none_or(|bytes| file["size"].as_u64() == Some(bytes))
        })
    };
    if !has_source(&fixture.source_config_sha256, None)
        || fixture.source_weights.iter().any(|weight| !has_source(&weight.sha256, Some(weight.bytes)))
    {
        return Err("source_weights_not_bound_to_conversion");
    }
    hash_file(&path)
}

fn one_run(
    model: &LanguageModel<Metal>,
    fixture: &Fixture,
    run: u32,
) -> Result<Value, &'static str> {
    let input: Vec<u64> = fixture.input_token_ids.iter().map(|&token| u64::from(token)).collect();
    let started = Instant::now();
    let mut state = model.create_empty_state(Some(fixture.context_tokens), 0).map_err(|_| "profile_state_failed")?;
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(&input, &mut state, options).map_err(|_| "profile_stream_failed")?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let mut output_digest = Sha256::new();
    let mut first_return_seconds = 0.0;
    for index in 0..fixture.decode_tokens {
        let token = stream.next().ok_or("profile_stopped_early")?.map_err(|_| "profile_generation_failed")?;
        if index == 0 {
            first_return_seconds = started.elapsed().as_secs_f64();
        }
        let token = u32::try_from(token).map_err(|_| "profile_invalid_token_width")?;
        output_digest.update(token.to_le_bytes());
    }
    let last_return_seconds = started.elapsed().as_secs_f64();
    let metrics = stream.metrics();
    if metrics.num_tokens_prefilled != 512
        || metrics.num_decode_forward_passes != 128
        || metrics.num_tokens_returned != 128
    {
        return Err("unexpected_native_profile_work_counts");
    }
    // Dropping the stream drains the pass queued for the last returned token.
    // The extra sampled lookahead token is not part of the output digest.
    drop(stream);
    let total_seconds = started.elapsed().as_secs_f64();
    if state.tokens().len() != 640 {
        return Err("unexpected_native_profile_context_length");
    }
    Ok(json!({
        "run":run, "measured":run >= fixture.warmup_runs, "input_tokens":512, "returned_tokens":128,
        "decode_forward_steps":128, "context_tokens_after":640, "request_total_seconds":total_seconds,
        "stream_setup_seconds":setup_seconds, "first_token_return_seconds":first_return_seconds,
        "remaining_token_returns_seconds":last_return_seconds-first_return_seconds,
        "final_drain_seconds":total_seconds-last_return_seconds, "work_tokens_per_second":640.0/total_seconds,
        "output_sha256":format!("{:x}", output_digest.finalize()),
    }))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run() -> Result<Value, &'static str> {
    if cfg!(feature = "scoring-last-layer-pruning") || cfg!(feature = "attention-kv-only") {
        return Err("profile_requires_generation_feature_set");
    }
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 2 {
        return Err("expected_native_model_and_private_fixture");
    }
    if fs::metadata(&args[1]).map_err(|_| "missing_profile_fixture")?.len() > 1024 * 1024 {
        return Err("profile_fixture_too_large");
    }
    let fixture: Fixture = serde_json::from_slice(&fs::read(&args[1]).map_err(|_| "unreadable_profile_fixture")?)
        .map_err(|_| "invalid_profile_fixture_json")?;
    validate_fixture(&fixture)?;
    if args[0].join("speculator").exists() {
        return Err("profile_requires_ordinary_non_speculative_model");
    }
    let weights_path = args[0].join("model.safetensors");
    let config_path = args[0].join("config.json");
    let weights_hash = hash_file(&weights_path)?;
    let config_hash = hash_file(&config_path)?;
    let receipt_hash = verify_receipt(&args[0], &fixture, &weights_hash, &config_hash)?;
    let engine = Engine::<Metal>::new().map_err(|_| "profile_metal_initialization_failed")?;
    let device = engine.metal_device_identity();
    let started = Instant::now();
    let model = engine.load_language_model(&args[0]).map_err(|_| "profile_native_load_failed")?;
    let load_seconds = started.elapsed().as_secs_f64();
    let mut runs = Vec::new();
    for index in 0..fixture.warmup_runs + fixture.measured_runs {
        runs.push(one_run(&model, &fixture, index)?);
    }
    let repeated = runs.iter().all(|item| item["output_sha256"] == runs[0]["output_sha256"]);
    let mut input_digest = Sha256::new();
    for token in &fixture.input_token_ids {
        input_digest.update(token.to_le_bytes());
    }
    let measured: Vec<_> = runs.iter().filter(|item| item["measured"] == true).collect();
    let median_seconds = median(measured.iter().map(|item| item["request_total_seconds"].as_f64().unwrap()).collect());
    Ok(json!({
        "schema_version":1,"passed":repeated,"backend":"stark-metal","scope":"matched-pp512-tg128-greedy-v1",
        "timing_boundary":"fresh state, 512-token prefill, 128 returned tokens, 128 decode forwards, final lookahead drain; first-return timing includes lookahead scheduling",
        "input_sha256":format!("{:x}",input_digest.finalize()),"input_tokens":512,"decode_forward_steps":128,
        "returned_tokens":128,"context_tokens":fixture.context_tokens,"warmup_runs":1,"measured_runs":3,
        "source_repo":fixture.source_repo,"source_revision":fixture.source_revision,
        "source_config_sha256":fixture.source_config_sha256,"source_weights_digest":source_weights_digest(&fixture.source_weights),
        "source_weights":fixture.source_weights,"quantization":fixture.quantization,"activation_dtype":"bfloat16",
        "native_weights_sha256":weights_hash,"native_config_sha256":config_hash,"conversion_receipt_sha256":receipt_hash,
        "executable_sha256":hash_file(&std::env::current_exe().map_err(|_| "profile_executable_identity_failed")?)?,
        "device":{"name":device.device_name,"gpu_core_count":device.gpu_core_count,"apple_gpu_family":device.apple_gpu_family,"supports_mxu":device.supports_mxu},
        "load_seconds":load_seconds,"peak_memory_bytes":engine.peak_memory_usage(),
        "median_request_total_seconds":median_seconds,"median_work_tokens_per_second":640.0/median_seconds,
        "repeated_outputs_identical":repeated,"runs":runs,
    }))
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let fd = unsafe { libc::dup(libc::STDOUT_FILENO) };
    if fd < 0 {
        std::process::exit(1);
    }
    let mut output = unsafe { File::from_raw_fd(fd) };
    let Ok(null) = OpenOptions::new().write(true).open("/dev/null") else {
        std::process::exit(1);
    };
    if unsafe { libc::dup2(null.as_raw_fd(), libc::STDOUT_FILENO) } < 0
        || unsafe { libc::dup2(null.as_raw_fd(), libc::STDERR_FILENO) } < 0
    {
        std::process::exit(1);
    }
    match std::panic::catch_unwind(run) {
        Ok(Ok(report)) => {
            let _ = writeln!(output, "{report}");
            if report["passed"] != true {
                std::process::exit(1);
            }
        },
        result => {
            let code = match result {
                Ok(Err(code)) => code,
                _ => "native_profile_failed",
            };
            let _ = writeln!(output, "{}", json!({"passed":false,"error":code}));
            std::process::exit(1);
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_digest_matches_python_protocol_and_binds_sizes() {
        let mut weights = vec![
            SourceWeight {
                filename: "model-b.safetensors".into(),
                bytes: 32,
                sha256: "b".repeat(64),
            },
            SourceWeight {
                filename: "model-a.safetensors".into(),
                bytes: 16,
                sha256: "a".repeat(64),
            },
        ];
        assert_eq!(source_weights_digest(&weights), "6e71b27b56cfdeda6a0c9537533e859386a7d3e7a1bc4a2ade7d95a945c37b0e");
        weights.reverse();
        assert_eq!(source_weights_digest(&weights), "6e71b27b56cfdeda6a0c9537533e859386a7d3e7a1bc4a2ade7d95a945c37b0e");
        weights[0].bytes += 1;
        assert_ne!(source_weights_digest(&weights), "6e71b27b56cfdeda6a0c9537533e859386a7d3e7a1bc4a2ade7d95a945c37b0e");
    }

    #[test]
    fn timing_fixture_rejects_changed_work_counts_before_loading_native_model() {
        let mut fixture = Fixture {
            schema_version: 1,
            input_token_ids: vec![1; 512],
            decode_tokens: 128,
            context_tokens: 1024,
            warmup_runs: 1,
            measured_runs: 3,
            source_repo: "test/Fixture".into(),
            source_revision: "a".repeat(40),
            source_config_sha256: "b".repeat(64),
            source_weights: vec![SourceWeight {
                filename: "model.safetensors".into(),
                bytes: 16,
                sha256: "c".repeat(64),
            }],
            quantization: Quantization {
                mode: "none".into(),
                bits: None,
                group_size: None,
                overrides_sha256: None,
            },
        };
        assert!(validate_fixture(&fixture).is_ok());
        fixture.decode_tokens = 127;
        assert!(validate_fixture(&fixture).is_err());
        fixture.decode_tokens = 128;
        fixture.input_token_ids.pop();
        assert!(validate_fixture(&fixture).is_err());
    }
}
