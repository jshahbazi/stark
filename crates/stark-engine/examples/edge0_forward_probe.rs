//! Explicit diagnostic-only Edge0 forward probe. Token/text/logit payloads
//! require private pipes and must never be redirected to persisted artifacts.
//! Build this identical helper against each sealed engine source separately.

use std::{
    io::{self, BufRead, Read, Write},
    os::fd::AsRawFd,
    path::Path,
    time::Instant,
};

use serde_json::{Value, json};
use stark_engine::engine::hybrid::Model;

#[allow(unexpected_cfgs, reason = "the supervisor seals the candidate-only example cfg and build flags")]
fn execution_metadata(_model: &Model) -> Result<Value, &'static str> {
    #[cfg(stark_edge0_candidate_probe)]
    {
        _model.edge0_execution_metadata().ok_or("execution_metadata_unavailable")
    }
    #[cfg(not(stark_edge0_candidate_probe))]
    {
        // This arm is admitted only against the immutable Phase 3 source;
        // the supervisor separately seals that source, flags and executable.
        Ok(json!({"backend":"edge0-f32-reference","activation_precision":"f32",
            "accumulation_precision":"f32","recurrent_state_precision":"f32","attention_cache_precision":"f32",
            "prefill_max_batch":1,"prefix_reuse":false,"source_binding":"0721afc5",
            "weights":{"resident_enabled":false,"registered_bank_bytes":0,
            "counters":{"batched_matrix_calls":0,"batched_q8_calls":0,"batched_expert_calls":0}}}))
    }
}

fn ids(value: &Value) -> Result<Vec<u32>, &'static str> {
    let values: Vec<u32> = serde_json::from_value(value.clone()).map_err(|_| "invalid_ids")?;
    if values.is_empty() || values.len() > 512 {
        return Err("invalid_ids_count");
    }
    Ok(values)
}

fn run() -> Result<(), &'static str> {
    if !pipe(io::stdin().as_raw_fd()) || !pipe(io::stdout().as_raw_fd()) {
        return Err("private_pipes_required");
    }
    #[cfg(feature = "metal")]
    // SAFETY: startup is single threaded and the supervising process owns the
    // dedicated nonblocking telemetry pipe, as required by this API.
    let _telemetry = unsafe { stark_engine::backends::metal::memory_usage::start_from_environment() };
    let path = std::env::args_os().nth(1).ok_or("model_path_required")?;
    let started = Instant::now();
    let mut model = Model::load(Path::new(&path)).map_err(|_| "load_failed")?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    let mut input = io::stdin().lock();
    let ready = json!({"ready":true,"protocol":1,"load_seconds":started.elapsed().as_secs_f64(),
        "vocab_size":model.vocab_size(),"max_context":model.max_context_length(),
        "request_memory_bytes":model.generation_memory_bytes(512,128,false),
        "eos_ids":model.eos_token_ids(),"execution":execution_metadata(&model)?});
    serde_json::to_writer(&mut output, &ready).map_err(|_| "pipe_write")?;
    writeln!(output).map_err(|_| "pipe_write")?;
    output.flush().map_err(|_| "pipe_write")?;
    loop {
        let mut bytes = Vec::new();
        let count = input.by_ref().take(1024 * 1024 + 1).read_until(b'\n', &mut bytes).map_err(|_| "pipe_read")?;
        if count == 0 {
            break;
        }
        if bytes.len() > 1024 * 1024 || bytes.last() != Some(&b'\n') {
            return Err("request_size");
        }
        let request: Value = serde_json::from_slice(&bytes).map_err(|_| "request_invalid")?;
        let result = match request.get("operation").and_then(Value::as_str) {
            Some("metadata") => execution_metadata(&model)?,
            Some("encode") => {
                let text = request["text"].as_str().ok_or("text_required")?;
                let encoded = model.tokenizer().encode(text, false).map_err(|_| "encode_failed")?;
                json!({"ids":encoded.get_ids()})
            },
            Some("prefill") => {
                let tokens = ids(&request["ids"])?;
                if request["reset"].as_bool().ok_or("reset_required")? {
                    model.reset();
                }
                model.reserve_generation_context(512).map_err(|_| "reserve_failed")?;
                let start = Instant::now();
                model.ingest_prefix(&tokens[..tokens.len() - 1], || Ok(())).map_err(|_| "ingest_failed")?;
                let ingest_seconds = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let logits = model.step(*tokens.last().ok_or("empty_prefix")?).map_err(|_| "step_failed")?;
                let readout_seconds = start.elapsed().as_secs_f64();
                json!({"logits":logits,"ingest_seconds":ingest_seconds,
                    "readout_seconds":readout_seconds,"prefix_tokens":tokens.len()})
            },
            Some("step") => {
                let token = request["id"].as_u64().filter(|value| *value <= u32::MAX as u64).ok_or("id_required")?;
                let start = Instant::now();
                let logits = model.step(token as u32).map_err(|_| "step_failed")?;
                let seconds = start.elapsed().as_secs_f64();
                json!({"logits":logits,"seconds":seconds})
            },
            Some("reject_and_reset") => {
                if model.step(model.vocab_size() as u32).is_ok() {
                    return Err("invalid_token_accepted");
                }
                json!({"rejected":true})
            },
            _ => return Err("operation_invalid"),
        };
        serde_json::to_writer(&mut output, &result).map_err(|_| "pipe_write")?;
        writeln!(output).map_err(|_| "pipe_write")?;
        output.flush().map_err(|_| "pipe_write")?;
    }
    Ok(())
}

fn pipe(fd: std::os::fd::RawFd) -> bool {
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: fstat only inspects this descriptor and initializes metadata on success.
    if unsafe { libc::fstat(fd, metadata.as_mut_ptr()) } != 0 {
        return false;
    }
    // SAFETY: successful fstat above initialized metadata.
    unsafe { metadata.assume_init() }.st_mode & libc::S_IFMT == libc::S_IFIFO
}

fn main() {
    if let Err(code) = run() {
        eprintln!("{code}");
        std::process::exit(1);
    }
}
