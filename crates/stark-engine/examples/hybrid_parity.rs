//! Compare private Flint/GLM source-bound fixtures with their native forward paths.
//!
//! The fixture directory contains config.json, tokens.json (an array of token
//! IDs), weights, and reference.f32 (little-endian rows of vocabulary logits).
//! Only aggregate errors, hashes and reset checks are printed.

use std::{fs, path::Path, time::Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::{flint::FlintModel, glm::GlmModel, weights::Weights};

enum Model {
    Flint(FlintModel),
    Glm(GlmModel),
}

impl Model {
    fn step(
        &mut self,
        token: u32,
    ) -> stark_engine::engine::hybrid::Result<Vec<f32>> {
        match self {
            Self::Flint(model) => model.step(token),
            Self::Glm(model) => model.step(token),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Flint(model) => model.reset(),
            Self::Glm(model) => model.reset(),
        }
    }
}

fn run() -> Result<bool, &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err("expected fixture directory and optional absolute/relative tolerance");
    }
    let tolerance = match args.get(1) {
        Some(value) => value.parse::<f64>().map_err(|_| "invalid tolerance")?,
        None => 2e-5,
    };
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance > 0.1 {
        return Err("invalid tolerance");
    }
    let path = Path::new(&args[0]);
    let config: Value = serde_json::from_slice(&fs::read(path.join("config.json")).map_err(|_| "missing config")?)
        .map_err(|_| "invalid config")?;
    let family = config.get("model_type").and_then(Value::as_str).ok_or("missing model type")?;
    if family == "qwen4_exp" {
        return Err("model_conversion_required");
    }
    if !matches!(family, "flint" | "glm5_next") {
        return Err("unsupported model type");
    }
    let tokens: Vec<u32> = serde_json::from_slice(&fs::read(path.join("tokens.json")).map_err(|_| "missing tokens")?)
        .map_err(|_| "invalid token fixture")?;
    if tokens.is_empty() || tokens.len() > 4096 {
        return Err("invalid fixture sequence length");
    }
    let reference_path = path.join("reference.f32");
    let reference_size = fs::metadata(&reference_path).map_err(|_| "missing reference")?.len();
    if reference_size == 0 || reference_size > 256 * 1024 * 1024 || !reference_size.is_multiple_of(4) {
        return Err("invalid reference size");
    }
    let reference_bytes = fs::read(reference_path).map_err(|_| "unreadable reference")?;
    let reference: Vec<_> = reference_bytes.as_chunks::<4>().0.iter().map(|value| f32::from_le_bytes(*value)).collect();
    if !reference.len().is_multiple_of(tokens.len()) || reference.iter().any(|value| !value.is_finite()) {
        return Err("invalid reference layout or values");
    }
    let width = reference.len() / tokens.len();
    let weights = Weights::open(path).map_err(|_| "native weight admission failed")?;
    let mut model = match family {
        "flint" => Model::Flint(FlintModel::new(weights, &config).map_err(|_| "native model admission failed")?),
        "glm5_next" => Model::Glm(GlmModel::new(weights, &config).map_err(|_| "native model admission failed")?),
        _ => return Err("unsupported model type"),
    };
    let started = Instant::now();
    let mut native = Vec::with_capacity(reference.len());
    for &token in &tokens {
        let logits = model.step(token).map_err(|_| "native forward failed")?;
        if logits.len() != width || logits.iter().any(|value| !value.is_finite()) {
            return Err("invalid native readout");
        }
        native.extend(logits);
    }
    let elapsed = started.elapsed().as_secs_f64();
    model.reset();
    let reset_exact = model.step(tokens[0]).map_err(|_| "reset replay failed")? == native[..width];
    let failure_rejected = model.step(u32::MAX).is_err();
    let failure_cleared = model.step(tokens[0]).map_err(|_| "failure recovery failed")? == native[..width];
    let mut max_abs = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut error_square = 0.0_f64;
    let mut reference_square = 0.0_f64;
    let mut close = true;
    for (&expected, &actual) in reference.iter().zip(&native) {
        let expected = f64::from(expected);
        let error = (f64::from(actual) - expected).abs();
        max_abs = max_abs.max(error);
        abs_sum += error;
        error_square += error * error;
        reference_square += expected * expected;
        close &= error <= tolerance + tolerance * expected.abs();
    }
    let argmax = |row: &[f32]| -> usize {
        row.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .map_or(0, |(index, _)| index)
    };
    let agreement =
        reference.chunks_exact(width).zip(native.chunks_exact(width)).filter(|(a, b)| argmax(a) == argmax(b)).count();
    let passed = close && reset_exact && failure_rejected && failure_cleared;
    let mut native_hash = Sha256::new();
    for value in native {
        native_hash.update(value.to_le_bytes());
    }
    let report = serde_json::json!({
        "family": family,
        "backend": if cfg!(feature = "metal") { "metal" } else { "cpu" },
        "positions": tokens.len(), "vocabulary": width,
        "max_abs_error": max_abs, "mean_abs_error": abs_sum / reference.len() as f64,
        "relative_l2_error": if reference_square > 0.0 { (error_square / reference_square).sqrt() } else { error_square.sqrt() },
        "greedy_agreement": agreement, "tolerance": tolerance,
        "reset_replay_exact": reset_exact, "failure_rejected": failure_rejected,
        "failure_clears_state": failure_cleared, "elapsed_seconds": elapsed,
        "reference_sha256": format!("{:x}", Sha256::digest(&reference_bytes)),
        "native_sha256": format!("{:x}", native_hash.finalize()), "passed": passed,
    });
    println!("{report}");
    Ok(passed)
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    match std::panic::catch_unwind(run) {
        Ok(Ok(true)) => {},
        Ok(Ok(false)) => std::process::exit(1),
        Ok(Err(message)) => {
            eprintln!("hybrid parity validation failed: {message}");
            std::process::exit(1);
        },
        Err(_) => {
            eprintln!("hybrid parity validation failed");
            std::process::exit(1);
        },
    }
}
