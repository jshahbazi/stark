//! Private cached-trajectory controls. Only hashes/counts enter the report.

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use half::bf16;
use sha2::{Digest, Sha256};
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::trie::TrieNode,
        },
        metal::Metal,
    },
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::Decoder},
    engine::{Engine, language_model::stream::SamplingMethod},
    parameters::ParameterLoader,
    tests::private_output::{create_private_directory, private_write},
};

const EXPECTED_SOURCE: &str = "03f17cff595a19aba064fb7ddf00547b52fd0cdf3c908b77db96f2727aba52a1";
// Current source-compatible kernels match this oracle. Historical immutable
// binaries can still bind their own output through the explicit environment override.
const EXPECTED_NATIVE: &str = EXPECTED_SOURCE;

fn digest_tokens(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn argmax(logits: &[bf16]) -> Result<u32, &'static str> {
    let mut best = None;
    for (i, x) in logits.iter().enumerate() {
        if !x.is_finite() {
            return Err("nonfinite_logits");
        }
        if best.is_none_or(|(_, v)| x.to_f32() > v) {
            best = Some((i as u32, x.to_f32()));
        }
    }
    best.map(|(i, _)| i).ok_or("empty_logits")
}
fn environment(name: &str) -> Result<PathBuf, &'static str> {
    std::env::var_os(name).map(PathBuf::from).ok_or("missing_environment")
}

fn digest_file(path: &Path) -> Result<String, &'static str> {
    let mut file = File::open(path).map_err(|_| "identity_open")?;
    let before = file.metadata().map_err(|_| "identity_metadata")?;
    let mut block = vec![0u8; 8 * 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let count = file.read(&mut block).map_err(|_| "identity_read")?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    let after = file.metadata().map_err(|_| "identity_metadata")?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("identity_changed");
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn run() -> Result<(), &'static str> {
    if cfg!(feature = "scoring-last-layer-pruning") || cfg!(feature = "attention-kv-only") {
        return Err("generation_features_required");
    }
    let model_path = environment("STARK_GEMMA_TRAJECTORY_MODEL")?;
    let cases_path = environment("STARK_GEMMA_TRAJECTORY_CASES")?;
    let fixture_path = environment("STARK_GEMMA_TRAJECTORY_FIXTURE")?;
    let output = environment("STARK_GEMMA_TRAJECTORY_OUTPUT")?;
    let capture_step = std::env::var("STARK_GEMMA_TRAJECTORY_CAPTURE_STEP")
        .ok()
        .map(|x| x.parse::<usize>().map_err(|_| "capture_step"))
        .transpose()?;
    let capture_layer = std::env::var("STARK_GEMMA_TRAJECTORY_CAPTURE_LAYER")
        .ok()
        .map(|x| x.parse::<usize>().map_err(|_| "capture_layer"))
        .transpose()?
        .unwrap_or(8);
    let expected_native =
        std::env::var("STARK_GEMMA_TRAJECTORY_EXPECTED_NATIVE_SHA256").unwrap_or_else(|_| EXPECTED_NATIVE.to_owned());
    if expected_native.len() != 64 || !expected_native.bytes().all(|x| x.is_ascii_hexdigit() && !x.is_ascii_uppercase())
    {
        return Err("expected_native_hash");
    }
    if capture_step.is_some_and(|i| i > 128) || capture_layer >= 35 {
        return Err("capture_step");
    }
    let fixture_bytes = fs::read(&fixture_path).map_err(|_| "fixture_read")?;
    let case_bytes = fs::read(&cases_path).map_err(|_| "cases_read")?;
    let config_bytes = fs::read(model_path.join("config.json")).map_err(|_| "config_read")?;
    let fixture: serde_json::Value = serde_json::from_slice(&fixture_bytes).map_err(|_| "fixture_json")?;
    let cases: serde_json::Value = serde_json::from_slice(&case_bytes).map_err(|_| "cases_json")?;
    let raw: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|_| "config_json")?;
    let prompt: Vec<u32> = serde_json::from_value(fixture["input_token_ids"].clone()).map_err(|_| "prompt_invalid")?;
    let mut selected: Option<&serde_json::Value> = None;
    let mut longest = 0;
    for entry in cases["cases"].as_array().ok_or("cases_missing")? {
        if let Some(tokens) = entry["tokens"].as_array()
            && tokens.len() >= 640
            && tokens.len() > longest
        {
            longest = tokens.len();
            selected = Some(&entry["tokens"]);
        }
    }
    let all: Vec<u32> =
        serde_json::from_value(selected.ok_or("cases_empty")?.clone()).map_err(|_| "sequence_invalid")?;
    let sequence = &all[all.len() - 640..];
    if prompt.len() != 512
        || sequence.len() != 640
        || sequence[..512] != prompt
        || fixture["decode_tokens"] != 128
        || fixture["context_tokens"] != 1024
    {
        return Err("geometry");
    }
    let expected = &sequence[512..];
    if digest_tokens(expected) != EXPECTED_SOURCE {
        return Err("source_sequence_identity");
    }
    let config: DecoderConfig = serde_json::from_value(raw["decoder_config"].clone()).map_err(|_| "decoder_config")?;
    if config.transformer_config.layer_configs.len() != 35 || config.ple_model_config.is_none() {
        return Err("family");
    }
    create_private_directory(&output).map_err(|_| "output_create")?;
    let weights_hash = digest_file(&model_path.join("model.safetensors"))?;
    let executable_hash = digest_file(&std::env::current_exe().map_err(|_| "executable_path")?)?;
    let harness_hash = format!("{:x}", Sha256::digest(include_bytes!("gemma4_cached_trajectory.rs")));
    let ring_kernel_hash = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../../src/backends/metal/kernel/attention/source_single_pass.metal"))
    );
    let cache_host_hash =
        format!("{:x}", Sha256::digest(include_bytes!("../../src/encodable_block/mixer/attention/state.rs")));
    let cached_qk_hash = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../../src/backends/metal/kernel/attention/source_cached_qk.metal"))
    );
    let fallback_host_hash =
        format!("{:x}", Sha256::digest(include_bytes!("../../src/backends/metal/kernel/attention/fallback.rs")));
    let engine = Engine::<Metal>::new().map_err(|_| "engine")?;
    // Reproduce the exact public benchmark boundary before treating a manual
    // decoder trace as evidence for generation.
    let public_hash = {
        let model = engine.load_language_model(&model_path).map_err(|_| "model_load")?;
        let mut state = model.create_empty_state(Some(1024), 0).map_err(|_| "public_state")?;
        let mut options = model.default_stream_options();
        options.sampling_method = SamplingMethod::Greedy;
        let input: Vec<u64> = prompt.iter().map(|&x| u64::from(x)).collect();
        let mut stream = model.stream(&input, &mut state, options).map_err(|_| "public_stream")?;
        let mut returned = Vec::new();
        for _ in 0..128 {
            returned.push(
                u32::try_from(stream.next().ok_or("stream_short")?.map_err(|_| "stream_step")?)
                    .map_err(|_| "token_width")?,
            );
        }
        let metrics = stream.metrics();
        if metrics.num_tokens_prefilled != 512
            || metrics.num_tokens_returned != 128
            || metrics.num_decode_forward_passes != 128
        {
            return Err("public_work_counts");
        }
        drop(stream);
        if state.tokens().len() != 640 {
            return Err("public_cache_length");
        }
        digest_tokens(&returned)
    };
    if public_hash != expected_native {
        return Err("frozen_benchmark_mismatch");
    }
    drop(engine);
    let context = <Metal as Backend>::Context::new().map_err(|_| "manual_context")?;
    let context = &context;
    let file = File::open(model_path.join("model.safetensors")).map_err(|_| "weights")?;
    let loader = ParameterLoader::<Metal>::new(&file, context).map_err(|_| "loader")?;
    let decoder = Decoder::<Metal>::new(context, &config, &loader.tree().subtree("decoder"), DataType::BF16)
        .map_err(|_| "decoder")?;
    loader.tree().assert_all_tensors_validated().map_err(|_| "parameters")?;
    let mut reports = Vec::new();
    for teacher_forced in [false, true] {
        let mut state = decoder.create_empty_state(Some(1024), context).map_err(|_| "state")?;
        state.prepare(0, 512).map_err(|_| "prepare")?;
        let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
        let prefix = commands.allocate_constant_from_slice(&prompt[..511]).map_err(|_| "prefix")?;
        let nodes: Vec<_> = (0..511)
            .map(|i| TrieNode {
                trie_start: i,
                trie_end: 510,
                height: i,
            })
            .collect();
        decoder
            .encode(&prefix, &BatchTopology::new(&nodes, true), None, None, &mut state, &mut commands)
            .map_err(|_| "prime")?;
        state.encode_accept(&(0..511).collect::<Vec<_>>(), &mut commands).map_err(|_| "prime_accept")?;
        drop(prefix);
        commands.end_encoding().submit().wait_until_completed().map_err(|_| "prime_execute")?;
        let mut selected = Vec::new();
        let mut steps = Vec::new();
        let mut previous = prompt[511];
        for step in 0..129 {
            if step > 0 {
                state.prepare(state.context_length(), 1).map_err(|_| "step_prepare")?;
            }
            let token = if step == 0 {
                prompt[511]
            } else if teacher_forced {
                expected[step - 1]
            } else {
                previous
            };
            let mut commands = context.create_command_buffer(None, None).map_err(|_| "commands")?;
            let tokens = commands.allocate_constant_from_slice(&[token]).map_err(|_| "token")?;
            let nodes = [TrieNode {
                trie_start: 0,
                trie_end: 0,
                height: 0,
            }];
            let layer_indices: Vec<u32> = (0..35).collect();
            let embedding = if teacher_forced {
                Some(decoder.embedding().encode_lookup(&tokens, 1, &mut commands).map_err(|_| "embedding")?)
            } else {
                None
            };
            let decoded = decoder
                .encode(
                    &tokens,
                    &BatchTopology::new(&nodes, true),
                    Some((0..1).into()),
                    teacher_forced.then_some(layer_indices.as_slice()),
                    &mut state,
                    &mut commands,
                )
                .map_err(|_| "decode")?;
            let mut sources = Vec::new();
            if let Some(x) = embedding {
                sources.push(("embedding".to_owned(), x));
            }
            if teacher_forced {
                for (i, x) in decoded.hidden_features.ok_or("hidden_features")?.into_vec().into_iter().enumerate() {
                    sources.push((format!("layer_{i:02}"), x));
                }
                sources.push(("final_hidden".to_owned(), decoded.final_hidden.ok_or("final_hidden")?));
            }
            sources.push(("logits".to_owned(), decoded.logits.ok_or("logits")?));
            let mut copies = Vec::new();
            for (name, source) in sources {
                let mut target = context.create_buffer(source.size()).map_err(|_| "copy")?;
                commands.encode_copy(&source, &mut target);
                copies.push((name, target));
            }
            let capture = teacher_forced && capture_step == Some(step);
            let cache_view = if capture {
                let cache = state.attention_state_for_test(capture_layer).ok_or("cache_state")?;
                if cache.data_type != DataType::BF16 {
                    return Err("cache_dtype");
                }
                let view = cache.view();
                let rows = view.prefix_len() + 1;
                let bytes = rows as usize * cache.element_dim as usize * 2;
                for (name, source) in
                    [("native_cache_keys", cache.keys.as_ref()), ("native_cache_values", cache.values.as_ref())]
                {
                    let mut target = context.create_buffer(bytes).map_err(|_| "cache_copy")?;
                    commands.encode_copy(source.subrange(..bytes), &mut target);
                    copies.push((name.to_owned(), target));
                }
                let ring = view.ring_params();
                Some(
                    serde_json::json!({"layer_index":capture_layer,"step":step,"absolute_position":511+step,"rows":rows,"element_dim":cache.element_dim,"prefix_rows":view.prefix_len(),"ring_offset":ring.map(|x|x.ring_offset),"ring_length":ring.map(|x|x.ring_length),"source_ring_origin":view.source_ring_origin()}),
                )
            } else {
                None
            };
            state.encode_accept(&[0], &mut commands).map_err(|_| "step_accept")?;
            drop(tokens);
            commands.end_encoding().submit().wait_until_completed().map_err(|_| "step_execute")?;
            let mut components = Vec::new();
            let mut choice = None;
            let capture_path = output.join(format!("step-{step:03}"));
            if capture {
                create_private_directory(&capture_path).map_err(|_| "capture_directory")?;
            }
            for (name, copy) in copies {
                let values = copy.copyout::<bf16>();
                if name == "logits" {
                    choice = Some(argmax(&values)?);
                }
                if values.iter().any(|x| !x.is_finite()) {
                    return Err("nonfinite_feature");
                }
                let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_f32().to_le_bytes()).collect();
                if capture {
                    private_write(&capture_path.join(format!("{name}.f32")), &bytes).map_err(|_| "capture_write")?;
                }
                components.push(serde_json::json!({"component":name,"elements":values.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
            }
            if let Some(view) = &cache_view {
                private_write(
                    &capture_path.join("cache-view.json"),
                    &serde_json::to_vec(view).map_err(|_| "cache_json")?,
                )
                .map_err(|_| "cache_write")?;
            }
            previous = choice.ok_or("choice_missing")?;
            if step < 128 {
                selected.push(previous);
            }
            if teacher_forced {
                steps.push(serde_json::json!({"step":step,"greedy_agreement":if step<128 {Some(previous==expected[step])} else {None},"components":components,"cache_view":cache_view}));
            }
        }
        if state.context_length() != 640 {
            return Err("manual_cache_length");
        }
        let hash = digest_tokens(&selected);
        if !teacher_forced && hash != public_hash {
            return Err("manual_benchmark_mismatch");
        }
        reports.push(serde_json::json!({"teacher_forced":teacher_forced,"returned_tokens":128,"logit_steps":129,"greedy_sha256":hash,"greedy_agreements":selected.iter().zip(expected).filter(|(a,b)|a==b).count(),"first_greedy_difference":selected.iter().zip(expected).position(|(a,b)|a!=b),"steps":steps}));
    }
    private_write(&output.join("trajectory-native.json"),&serde_json::to_vec(&serde_json::json!({"scope":"cached-trajectory-source-conditioned-v2","public_benchmark_sha256":public_hash,"expected_native_sha256":expected_native,"expected_source_sha256":EXPECTED_SOURCE,"fixture_sha256":format!("{:x}",Sha256::digest(&fixture_bytes)),"cases_sha256":format!("{:x}",Sha256::digest(&case_bytes)),"config_sha256":format!("{:x}",Sha256::digest(&config_bytes)),"weights_sha256":weights_hash,"executable_sha256":executable_hash,"compiled_harness_sha256":harness_hash,"compiled_ring_kernel_sha256":ring_kernel_hash,"compiled_cache_host_sha256":cache_host_hash,"compiled_cached_qk_sha256":cached_qk_hash,"compiled_fallback_host_sha256":fallback_host_hash,"case_selection":"longest-at-least-640-first-tie-last-640","modes":reports})).map_err(|_|"report_json")?).map_err(|_|"report_write")?;
    Ok(())
}

#[stark_test]
#[ignore]
fn gemma4_private_cached_trajectory_probe() {
    let result = run();
    assert!(result.is_ok(), "private cached trajectory probe failed: {}", result.err().unwrap_or("unknown"));
}
