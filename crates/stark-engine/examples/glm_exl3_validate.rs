//! Aggregate-only artifact qualification for the explicit EXL3 prefix build.
//! Token inputs arrive through an anonymous stdin pipe and remain in memory.

#[cfg(not(stark_glm_exl3_prefix_validation))]
fn main() -> std::process::ExitCode {
    eprintln!("explicit GLM prefix validation build required");
    std::process::ExitCode::FAILURE
}

#[cfg(stark_glm_exl3_prefix_validation)]
mod validation {
    use std::{io::Read, path::Path};

    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use stark_engine::engine::hybrid::{Model, PrefixSnapshot};

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Input {
        first: Vec<u32>,
        second: Vec<u32>,
        decode: usize,
    }

    struct Row {
        logits: Vec<f32>,
        state: [u8; 32],
    }

    fn prefill(
        model: &mut Model,
        ids: &[u32],
    ) -> Result<Vec<f32>, &'static str> {
        let (&last, prefix) = ids.split_last().ok_or("empty input")?;
        model.ingest_prefix(prefix, || Ok(())).map_err(|_| "prefill admission failed")?;
        model.step(last).map_err(|_| "prefill failed")
    }

    fn rows(
        model: &mut Model,
        first: Vec<f32>,
        decode: usize,
    ) -> Result<Vec<Row>, &'static str> {
        let mut rows = Vec::with_capacity(decode + 1);
        let mut logits = first;
        for index in 0..=decode {
            let state = model.validation_state_sha256().map_err(|_| "state hash failed")?;
            let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).ok_or("empty readout")?.0 as u32;
            rows.push(Row {
                logits,
                state,
            });
            if index < decode {
                logits = model.step(next).map_err(|_| "decode failed")?;
            } else {
                break;
            }
        }
        Ok(rows)
    }

    fn snapshot(
        model: &mut Model,
        ids: &[u32],
        boundary: usize,
    ) -> Result<PrefixSnapshot, &'static str> {
        model.reset();
        model.capture_prefix_at(boundary).map_err(|_| "capture admission failed")?;
        let _ = prefill(model, ids)?;
        model.take_prefix_snapshot().ok_or("snapshot missing")
    }

    pub fn run() -> Result<serde_json::Value, &'static str> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 1 {
            return Err("expected model directory");
        }
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(0, &mut stat) } != 0
            || stat.st_mode & libc::S_IFMT != libc::S_IFIFO
            || stat.st_nlink != 0
        {
            return Err("input must use an anonymous pipe");
        }
        let mut bytes = Vec::new();
        std::io::stdin().take(2 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|_| "input read failed")?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err("input exceeds limit");
        }
        let input: Input = serde_json::from_slice(&bytes).map_err(|_| "invalid input")?;
        drop(bytes);
        let mut model = Model::load(Path::new(&args[0])).map_err(|_| "model load failed")?;
        let chunk = model.prefix_chunk_tokens().ok_or("prefix validation unavailable")?;
        if input.decode > 8
            || input.first.len() <= chunk
            || input.second.len() <= chunk
            || input.first.len().max(input.second.len()) + input.decode > model.max_context_length() as usize
            || input.first[..chunk] != input.second[..chunk]
            || input.first[chunk..] == input.second[chunk..]
        {
            return Err("invalid qualification workload");
        }
        let saved = snapshot(&mut model, &input.first, chunk)?;
        model.reset();
        model.restore_prefix(&saved).map_err(|_| "restore admission failed")?;
        let restored = prefill(&mut model, &input.second[chunk..])?;
        let restored = rows(&mut model, restored, input.decode)?;
        model.reset();
        let fresh = prefill(&mut model, &input.second)?;
        let fresh = rows(&mut model, fresh, input.decode)?;
        let mut exact_rows = 0;
        let mut exact_states = 0;
        let mut maximum_absolute_error = 0.0f64;
        let mut readout_digest = Sha256::new();
        let mut state_digest = Sha256::new();
        for (a, b) in restored.iter().zip(&fresh) {
            if a.logits.len() != b.logits.len() {
                return Err("readout shape mismatch");
            }
            exact_rows += usize::from(a.logits.iter().zip(&b.logits).all(|(a, b)| a.to_bits() == b.to_bits()));
            exact_states += usize::from(a.state == b.state);
            for (a, b) in a.logits.iter().zip(&b.logits) {
                maximum_absolute_error = maximum_absolute_error.max((*a as f64 - *b as f64).abs());
                readout_digest.update(a.to_bits().to_le_bytes());
            }
            state_digest.update(a.state);
        }
        // A fresh intervening prompt invalidates the older native snapshot.
        model.reset();
        let stale_rejected =
            model.restore_prefix(&saved).is_err() || prefill(&mut model, &input.second[chunk..]).is_err();
        drop(saved);
        let saved = snapshot(&mut model, &input.second, chunk)?;
        let invalid_token_rejected = model.step(model.vocab_size() as u32).is_err();
        let failed_request_invalidated = model.restore_prefix(&saved).is_err();
        drop(saved);
        let saved = snapshot(&mut model, &input.second, chunk)?;
        model.reset();
        model.restore_prefix(&saved).map_err(|_| "callback restore admission failed")?;
        let callback_rejected = model
            .ingest_prefix(&input.second[chunk..], || {
                Err(stark_engine::engine::hybrid::Error::Invalid("qualification cancellation"))
            })
            .is_err();
        let callback_invalidated = model.restore_prefix(&saved).is_err();
        drop(saved);
        let count = input.decode + 1;
        Ok(serde_json::json!({
            "schema": "stark-glm-exl3-prefix-state-v1", "prefix_tokens": chunk,
            "first_tokens": input.first.len(), "second_tokens": input.second.len(),
            "readout_rows": count, "vocabulary": model.vocab_size(),
            "bit_exact_readout_rows": exact_rows, "bit_exact_state_rows": exact_states,
            "maximum_absolute_error": maximum_absolute_error,
            "restored_readout_bundle_sha256": format!("{:x}", readout_digest.finalize()),
            "restored_state_bundle_sha256": format!("{:x}", state_digest.finalize()),
            "stale_snapshot_rejected": stale_rejected,
            "invalid_token_rejected": invalid_token_rejected,
            "failure_invalidated_snapshot": failed_request_invalidated,
            "callback_rejected": callback_rejected,
            "callback_invalidated_snapshot": callback_invalidated,
            "passed": exact_rows == count && exact_states == count && stale_rejected
                && invalid_token_rejected && failed_request_invalidated && callback_rejected && callback_invalidated
        }))
    }
}

#[cfg(stark_glm_exl3_prefix_validation)]
fn main() -> std::process::ExitCode {
    match validation::run() {
        Ok(value) => {
            println!("{value}");
            if value["passed"] == true {
                std::process::ExitCode::SUCCESS
            } else {
                std::process::ExitCode::FAILURE
            }
        },
        Err(category) => {
            println!(
                "{}",
                serde_json::json!({
                    "schema": "stark-glm-exl3-validation-error-v1", "category": category
                })
            );
            std::process::ExitCode::FAILURE
        },
    }
}
