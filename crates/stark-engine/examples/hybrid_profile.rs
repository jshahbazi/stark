//! Aggregate native full-checkpoint profile. Never prints model inputs/outputs.

use std::{path::Path, time::Instant};

use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::{Error, Model, activity::InferenceActivity, profile};

// These variants carry fixed source labels only. Never format I/O exceptions,
// model values or metadata into diagnostic output.
fn native_error(error: Error) -> &'static str {
    match error {
        Error::Invalid(label) | Error::Unsupported(label) => label,
        _ => "native execution failed",
    }
}

fn run() -> Result<(), &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("expected model directory, decode steps, replays and optional synthetic prefix length");
    }
    let steps: usize = args[1].parse().map_err(|_| "invalid step count")?;
    let replays: usize = args[2].parse().map_err(|_| "invalid replay count")?;
    if !(2..=256).contains(&steps) || !(1..=8).contains(&replays) {
        return Err("invalid profile bounds");
    }
    let mut model = Model::load(Path::new(&args[0])).map_err(|_| "model admission failed")?;
    let mut input = model.tokenizer().encode("Hello.", false).map_err(|_| "tokenization failed")?.get_ids().to_vec();
    if input.is_empty() {
        return Err("empty profile input");
    }
    if let Some(length) = args.get(3) {
        let length: usize = length.parse().map_err(|_| "invalid prefix length")?;
        if !(2..=65536).contains(&length) {
            return Err("invalid prefix length");
        }
        // Synthetic repeated prefix: performance coverage, not an answer-
        // quality evaluation. IDs remain private and are never persisted.
        input = input.iter().copied().cycle().take(length).collect();
    }
    let mut input_hash = Sha256::new();
    for &token in &input {
        input_hash.update(token.to_le_bytes());
    }
    let input_sha256 = format!("{:x}", input_hash.finalize());
    let activity_requested = std::env::var_os("STARK_PROFILE_ACTIVITY").is_some_and(|value| value == "1");
    for replay in 0..replays {
        let activity = activity_requested.then(InferenceActivity::begin).transpose().map_err(native_error)?;
        model.reset();
        profile::reset();
        profile::enable(true);
        let started = Instant::now();
        let mut logits = Vec::new();
        for (index, &token) in input.iter().enumerate() {
            if index + 1 == input.len() {
                logits = model.step(token).map_err(native_error)?;
            } else {
                model.ingest(token).map_err(native_error)?;
            }
        }
        let prefill_seconds = started.elapsed().as_secs_f64();
        let prefill_profile = profile::snapshot();
        profile::reset();
        let started = Instant::now();
        let mut hash = Sha256::new();
        for _ in 0..steps {
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            // Hash only: inputs, outputs, probabilities and logits stay private.
            hash.update((best as u32).to_le_bytes());
            logits = model.step(best as u32).map_err(native_error)?;
        }
        let decode_seconds = started.elapsed().as_secs_f64();
        profile::enable(false);
        println!(
            "{}",
            serde_json::json!({"replay": replay, "input_tokens": input.len(), "input_sha256": input_sha256,
            "activity_assertion": activity.is_some(),
            "synthetic_repeated_prefix": args.len() == 4, "decode_steps": steps,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "decode_tokens_per_second": steps as f64 / decode_seconds,
            "output_sha256": format!("{:x}",hash.finalize()),
            "prefill_profile": prefill_profile, "decode_profile": profile::snapshot()})
        );
    }
    model.reset();
    Ok(())
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    match std::panic::catch_unwind(run) {
        Ok(Ok(())) => {},
        Ok(Err(message)) => {
            eprintln!("native aggregate profile failed: {message}");
            std::process::exit(1);
        },
        Err(_) => {
            eprintln!("native aggregate profile failed");
            std::process::exit(1);
        },
    }
}
