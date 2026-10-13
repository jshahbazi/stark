//! Private, volatile IPC for the scoped Qwen3 import experiment.
//! Full logits and input IDs pass through pipes only; the Python owner persists
//! aggregate metrics and hashes. This worker never decodes or logs model text.
use std::{
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
    time::Instant,
};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{LanguageModel, stream::SamplingMethod},
    },
};

// Reuse the content-free descriptor/panic/capture isolation of stark-score.
#[path = "../src/protocol.rs"]
mod protocol;

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Tokenize {
        text: String,
    },
    Score {
        input_token_ids: Vec<u32>,
        context_tokens: u32,
    },
    Generate {
        input_token_ids: Vec<u32>,
        context_tokens: u32,
        output_tokens: u32,
    },
}

fn hash_u32(values: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn rss_peak() -> Result<u64, &'static str> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err("memory_measurement_failed");
    }
    let peak = unsafe { usage.assume_init() }.ru_maxrss;
    if peak < 0 {
        return Err("memory_measurement_failed");
    }
    // macOS ru_maxrss is bytes; this binary always uses the Metal backend.
    Ok(peak as u64)
}

fn validate(
    model: &LanguageModel<Metal>,
    ids: &[u32],
    context: u32,
    vocab: u32,
) -> Result<(), &'static str> {
    if ids.is_empty()
        || context == 0
        || context > 16384
        || ids.len() > context as usize
        || ids.iter().any(|id| *id >= vocab)
        || model.max_context_length().is_some_and(|maximum| context > maximum)
    {
        return Err("invalid_experiment_request");
    }
    Ok(())
}

fn handle(
    model: &LanguageModel<Metal>,
    request: Request,
    vocab: u32,
) -> Result<Value, &'static str> {
    match request {
        Request::Tokenize {
            text,
        } => {
            if text.is_empty() {
                return Err("invalid_experiment_request");
            }
            let tokens = model.tokenizer().encode(text, false).map_err(|_| "tokenization_failed")?.get_ids().to_vec();
            Ok(json!({"token_ids_sha256": hash_u32(&tokens), "token_ids": tokens}))
        },
        Request::Score {
            input_token_ids,
            context_tokens,
        } => {
            validate(model, &input_token_ids, context_tokens, vocab)?;
            let started = Instant::now();
            let scores =
                model.score_next_token(&input_token_ids, context_tokens).map_err(|_| "native_scoring_failed")?;
            let total_seconds = started.elapsed().as_secs_f64();
            if scores.logits.len() != vocab as usize || scores.logits.iter().any(|value| !value.is_finite()) {
                return Err("invalid_score_values");
            }
            let bits = scores.logits.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            Ok(json!({"logits": scores.logits, "logits_sha256": hash_u32(&bits),
                "input_tokens_sha256": hash_u32(&input_token_ids), "input_tokens": input_token_ids.len(),
                "metrics": {"state_creation_seconds": scores.state_creation_seconds,
                    "prefill_seconds": scores.prefill_seconds, "readout_seconds": scores.readout_seconds,
                    "total_seconds": total_seconds}}))
        },
        Request::Generate {
            input_token_ids,
            context_tokens,
            output_tokens,
        } => {
            validate(model, &input_token_ids, context_tokens, vocab)?;
            if !(2..=128).contains(&output_tokens)
                || input_token_ids.len() + output_tokens as usize > context_tokens as usize
            {
                return Err("invalid_experiment_request");
            }
            let started = Instant::now();
            let mut state = model.create_empty_state(Some(context_tokens), 0).map_err(|_| "native_state_failed")?;
            let state_creation_seconds = started.elapsed().as_secs_f64();
            let input = input_token_ids.iter().map(|id| *id as u64).collect::<Vec<_>>();
            let mut options = model.default_stream_options();
            options.sampling_method = SamplingMethod::Greedy;
            let prefill_started = Instant::now();
            let mut stream = model.stream(&input, &mut state, options).map_err(|_| "native_stream_failed")?;
            let mut tokens = Vec::with_capacity(output_tokens as usize);
            tokens.push(stream.next().ok_or("stream_ended_early")?.map_err(|_| "native_stream_failed")? as u32);
            let time_to_first_token_seconds = prefill_started.elapsed().as_secs_f64();
            let decode_started = Instant::now();
            for _ in 1..output_tokens {
                tokens.push(stream.next().ok_or("stream_ended_early")?.map_err(|_| "native_stream_failed")? as u32);
            }
            let decode_seconds = decode_started.elapsed().as_secs_f64();
            let metrics = stream.metrics();
            let forward_passes = json!({"prefill": metrics.num_prefill_forward_passes,
                "decode": metrics.num_decode_forward_passes, "tokens_prefilled": metrics.num_tokens_prefilled});
            let drain_started = Instant::now();
            drop(stream);
            let drain_seconds = drain_started.elapsed().as_secs_f64();
            Ok(json!({"generated_tokens_sha256": hash_u32(&tokens), "generated_tokens": tokens.len(),
                "input_tokens_sha256": hash_u32(&input_token_ids), "first_token_id": tokens[0],
                "metrics": {"state_creation_seconds": state_creation_seconds,
                    "time_to_first_token_seconds": time_to_first_token_seconds, "decode_seconds": decode_seconds,
                    "decode_tokens": output_tokens - 1, "drain_seconds": drain_seconds,
                    "total_seconds": state_creation_seconds + time_to_first_token_seconds + decode_seconds + drain_seconds},
                "forward_passes": forward_passes}))
        },
    }
}

mod scoring {
    use super::*;
    pub(super) fn run_lines(
        path: &Path,
        enabled: bool,
        digest_mode: bool,
        output: &mut impl Write,
    ) -> Result<(), &'static str> {
        if enabled || digest_mode || path.join("speculator").exists() {
            return Err("unsupported_experiment_mode");
        }
        let config: Value =
            serde_json::from_reader(File::open(path.join("config.json")).map_err(|_| "model_config_unavailable")?)
                .map_err(|_| "invalid_model_config")?;
        let vocab = config["decoder_config"]["vocab_size"]
            .as_u64()
            .filter(|size| *size > 0 && *size <= u32::MAX as u64)
            .ok_or("invalid_model_config")? as u32;
        let started = Instant::now();
        let engine = Engine::<Metal>::new().map_err(|_| "metal_initialization_failed")?;
        let model = engine.load_language_model(path).map_err(|_| "model_load_failed")?;
        reply(
            output,
            &json!({"ready": true, "protocol": "qwen3-gguf-experiment-v1", "load_seconds": started.elapsed().as_secs_f64(),
            "vocab_size": vocab, "process_peak_rss_bytes": rss_peak()?, "engine_peak_allocated_bytes": engine.peak_memory_usage()}),
        )?;
        let mut input = io::stdin().lock();
        loop {
            let mut bytes = Vec::new();
            let count = Read::by_ref(&mut input)
                .take(1_048_577)
                .read_until(b'\n', &mut bytes)
                .map_err(|_| "protocol_read_failed")?;
            if count == 0 {
                return Ok(());
            }
            if bytes.len() > 1_048_576 {
                return Err("request_too_large");
            }
            let request = serde_json::from_slice::<Request>(&bytes).map_err(|_| "invalid_experiment_request")?;
            let mut response = handle(&model, request, vocab)?;
            response["process_peak_rss_bytes"] = json!(rss_peak()?);
            response["engine_peak_allocated_bytes"] = json!(engine.peak_memory_usage());
            reply(output, &response)?;
        }
    }
    fn reply(
        output: &mut impl Write,
        value: &Value,
    ) -> Result<(), &'static str> {
        serde_json::to_writer(&mut *output, value).map_err(|_| "protocol_write_failed")?;
        output.write_all(b"\n").and_then(|_| output.flush()).map_err(|_| "protocol_write_failed")
    }
}

fn main() {
    protocol::main(false);
}
