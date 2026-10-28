//! Aggregate native full-checkpoint profile. Never prints model inputs/outputs.

use std::{path::Path, time::Instant};

use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::{Model, profile};

fn run() -> Result<(), &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("expected model directory, decode steps and replays");
    }
    let steps: usize = args[1].parse().map_err(|_| "invalid step count")?;
    let replays: usize = args[2].parse().map_err(|_| "invalid replay count")?;
    if !(2..=128).contains(&steps) || !(1..=8).contains(&replays) {
        return Err("invalid profile bounds");
    }
    let mut model = Model::load(Path::new(&args[0])).map_err(|_| "model admission failed")?;
    let input = model.tokenizer().encode("Hello.", false).map_err(|_| "tokenization failed")?.get_ids().to_vec();
    if input.is_empty() {
        return Err("empty profile input");
    }
    for replay in 0..replays {
        model.reset();
        profile::reset();
        profile::enable(true);
        let started = Instant::now();
        let mut logits = Vec::new();
        for (index, &token) in input.iter().enumerate() {
            if index + 1 == input.len() {
                logits = model.step(token).map_err(|_| "prefill failed")?;
            } else {
                model.ingest(token).map_err(|_| "prefill failed")?;
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
            logits = model.step(best as u32).map_err(|_| "decode failed")?;
        }
        let decode_seconds = started.elapsed().as_secs_f64();
        profile::enable(false);
        println!(
            "{}",
            serde_json::json!({"replay": replay, "input_tokens": input.len(), "decode_steps": steps,
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
