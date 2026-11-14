//! Compare private, source-bound full-vocabulary references with native Metal.
//! Inputs and logits stay in private fixtures/memory; stdout contains aggregates.

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::fd::{AsRawFd, FromRawFd},
    path::Path,
    time::Instant,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use stark_engine::{backends::metal::Metal, engine::Engine};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    tokens: Vec<u32>,
    context_tokens: u32,
}

fn log_partition(values: &[f32]) -> f64 {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    maximum + values.iter().map(|&value| (f64::from(value) - maximum).exp()).sum::<f64>().ln()
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().fold(0, |best, (index, value)| {
        if *value > values[best] {
            index
        } else {
            best
        }
    })
}

fn run() -> Result<serde_json::Value, &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("expected_model_and_private_fixture_directories");
    }
    let gemma_profile = match args.get(2).map(String::as_str) {
        None => false,
        Some("gemma4-canonical-bf16-v4") => true,
        _ => return Err("unsupported_numerical_profile"),
    };
    let fixture_path = Path::new(&args[1]);
    let input_path = fixture_path.join("cases.json");
    if fs::metadata(&input_path).map_err(|_| "missing_fixture")?.len() > 2 * 1024 * 1024 {
        return Err("fixture_too_large");
    }
    let fixture: Fixture = serde_json::from_slice(&fs::read(input_path).map_err(|_| "unreadable_fixture")?)
        .map_err(|_| "invalid_fixture")?;
    if fixture.cases.is_empty() || fixture.cases.len() > 32 {
        return Err("invalid_case_count");
    }
    let engine = Engine::<Metal>::new().map_err(|_| "metal_initialization_failed")?;
    let model = engine.load_language_model(Path::new(&args[0])).map_err(|_| "native_load_failed")?;
    let mut rows = Vec::new();
    let mut all_passed = true;
    for (index, case) in fixture.cases.iter().enumerate() {
        if case.tokens.is_empty()
            || case.tokens.len() > 8192
            || case.tokens.len() > case.context_tokens as usize
            || case.context_tokens > 16384
        {
            return Err("invalid_case_geometry");
        }
        let reference_path = fixture_path.join(format!("reference-{index}.f32"));
        let size = fs::metadata(&reference_path).map_err(|_| "missing_reference")?.len();
        if size == 0 || size > 16 * 1024 * 1024 || !size.is_multiple_of(4) {
            return Err("invalid_reference_size");
        }
        let bytes = fs::read(reference_path).map_err(|_| "unreadable_reference")?;
        let reference: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|value| f32::from_le_bytes(*value)).collect();
        let started = Instant::now();
        let actual = model.score_next_token(&case.tokens, case.context_tokens).map_err(|error| {
            use stark_engine::engine::language_model::scoring::LanguageModelScoreError;
            match error {
                LanguageModelScoreError::InvalidInput => "native_score_invalid_input",
                LanguageModelScoreError::ContextOverflow => "native_score_context_overflow",
                LanguageModelScoreError::SpeculatorNotSupported => "native_score_speculator",
                LanguageModelScoreError::Backend(_) => "native_score_backend",
                LanguageModelScoreError::Decoder(_) => "native_score_decoder",
                LanguageModelScoreError::State(_) => "native_score_state",
                LanguageModelScoreError::InvalidOutput => "native_score_invalid_output",
            }
        })?;
        let elapsed = started.elapsed().as_secs_f64();
        let native = &actual.logits;
        if native.len() != reference.len() || native.iter().chain(&reference).any(|value| !value.is_finite()) {
            return Err("invalid_logits");
        }
        let maximum =
            reference.iter().zip(native).map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0, f64::max);
        let squared_error: f64 =
            reference.iter().zip(native).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
        let reference_squared: f64 = reference.iter().map(|&value| f64::from(value).powi(2)).sum();
        let rmse = (squared_error / native.len() as f64).sqrt();
        let mean_error =
            reference.iter().zip(native).map(|(&a, &b)| f64::from(b) - f64::from(a)).sum::<f64>() / native.len() as f64;
        let centered_rmse = (rmse * rmse - mean_error * mean_error).max(0.0).sqrt();
        let centered_maximum = reference
            .iter()
            .zip(native)
            .map(|(&a, &b)| (f64::from(b) - f64::from(a) - mean_error).abs())
            .fold(0.0, f64::max);
        let relative_l2 = (squared_error / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
        let reference_mean = reference.iter().map(|&value| f64::from(value)).sum::<f64>() / reference.len() as f64;
        let reference_variance =
            (reference_squared / reference.len() as f64 - reference_mean * reference_mean).max(f64::MIN_POSITIVE);
        let centered_relative_l2 = centered_rmse / reference_variance.sqrt();
        let ref_partition = log_partition(&reference);
        let native_partition = log_partition(native);
        let kl: f64 = reference
            .iter()
            .zip(native)
            .map(|(&a, &b)| {
                let log_p = f64::from(a) - ref_partition;
                log_p.exp() * (log_p - (f64::from(b) - native_partition))
            })
            .sum();
        // Logit distributions are invariant to a common additive offset. Keep
        // raw absolute bounds, and apply the relative bound after centering.
        // Raw relative error and greedy agreement remain separate diagnostics.
        // Canonical HF Gemma BF16/FP32 controls differ by up to 5.405%
        // centered L2; HF BF16 versus MLX reaches 6.068%. Its norm budget is
        // calibrated from these independent source controls. Raw absolute
        // and KL bounds remain fixed; OLMo retains the tighter default.
        let passed = centered_maximum <= 0.75
            && mean_error.abs() <= 0.75
            && kl <= 0.01
            && if gemma_profile {
                centered_rmse <= 0.15 && centered_rmse <= 1.0 / 128.0 + 0.08 * reference_variance.sqrt()
            } else {
                // Combined absolute/relative BF16 tolerance avoids dividing
                // small centered logits by a near-zero reference norm.
                rmse <= 0.1 && centered_rmse <= 1.0 / 128.0 + 0.02 * reference_variance.sqrt()
            };
        all_passed &= passed;
        let mut native_hash = Sha256::new();
        for value in native {
            native_hash.update(value.to_le_bytes());
        }
        let mut token_hash = Sha256::new();
        for value in &case.tokens {
            token_hash.update(value.to_le_bytes());
        }
        rows.push(serde_json::json!({
            "input_tokens": case.tokens.len(), "context_tokens": case.context_tokens,
            "vocabulary": native.len(), "max_abs_error": maximum, "rmse": rmse,
            "mean_error": mean_error, "centered_rmse": centered_rmse,
            "centered_max_abs_error": centered_maximum,
            "centered_relative_l2_error": centered_relative_l2,
            "relative_l2_error": relative_l2, "reference_to_native_kl": kl.max(0.0),
            "greedy_agreement": argmax(&reference) == argmax(native),
            "prefill_seconds": actual.prefill_seconds, "elapsed_seconds": elapsed,
            "input_sha256": format!("{:x}", token_hash.finalize()),
            "reference_sha256": format!("{:x}", Sha256::digest(&bytes)),
            "native_sha256": format!("{:x}", native_hash.finalize()), "passed": passed,
        }));
    }
    Ok(
        serde_json::json!({"backend":"metal", "acceptance": if gemma_profile {"gemma4-canonical-bf16-v4"} else {"bf16-centered-logits-v3"}, "cases": rows, "passed": all_passed}),
    )
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    // Native diagnostics never become validation output or exception dumps.
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
                _ => "native_validation_failed",
            };
            let _ = writeln!(output, "{}", serde_json::json!({"passed":false,"error":code}));
            std::process::exit(1);
        },
    }
}
