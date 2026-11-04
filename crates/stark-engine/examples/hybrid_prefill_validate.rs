//! Compare chunked GLM prefill with scalar ingestion on one loaded checkpoint.
//! Inputs and full vocabulary readouts remain in memory. Only aggregates and
//! hashes are printed. Run with the optimized recipe and candidate selectors.

use std::{fs, path::Path, time::Instant};

use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::{Model, activity::InferenceActivity, profile};

const ABSOLUTE_TOLERANCE: f64 = 1e-3;
const RELATIVE_TOLERANCE: f64 = 1e-3;

fn digest(values: impl IntoIterator<Item = [u8; 4]>) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value);
    }
    format!("{:x}", hash.finalize())
}

fn best(values: &[f32]) -> usize {
    let mut winner = 0;
    for index in 1..values.len() {
        if values[index] > values[winner] {
            winner = index;
        }
    }
    winner
}

fn run() -> Result<bool, &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("expected model directory, corpus file and comma-separated prefix lengths");
    }
    let lengths: Vec<usize> = args[2]
        .split(',')
        .map(|value| value.parse().map_err(|_| "invalid prefix lengths"))
        .collect::<Result<_, _>>()?;
    if lengths.is_empty() || lengths.len() > 16 || lengths.iter().any(|&length| !(2..=4096).contains(&length)) {
        return Err("invalid prefix limits");
    }
    let corpus = fs::read(&args[1]).map_err(|_| "corpus unavailable")?;
    if corpus.len() > 8 * 1024 * 1024 {
        return Err("corpus exceeds limit");
    }
    let text = std::str::from_utf8(&corpus).map_err(|_| "corpus is not UTF-8")?;
    let enabled = |name| std::env::var_os(name).is_some_and(|value| value == "1");
    let wide = std::env::var("STARK_GLM53_PREFILL_WIDE").ok();
    let control = enabled("STARK_GLM53_CONTROL_PREFILL_BATCH");
    let kda = enabled("STARK_GLM53_KDA_REGISTER_PREFILL");
    let _activity = InferenceActivity::begin().map_err(|_| "activity assertion failed")?;
    let mut model = Model::load(Path::new(&args[0])).map_err(|_| "model admission failed")?;
    let encoded = model.tokenizer().encode(text, false).map_err(|_| "corpus tokenization failed")?;
    if encoded.is_empty() || lengths.iter().any(|&length| length + 4 > model.max_context_length() as usize) {
        return Err("invalid corpus or context capacity");
    }
    let mut passed = true;
    for length in lengths {
        let input: Vec<_> = encoded.get_ids().iter().copied().cycle().take(length + 4).collect();
        model.reset();
        let started = Instant::now();
        // Explicit single-token ingestion keeps the old KDA/control chronology
        // even when the candidate's chunk selectors are enabled in this process.
        for &token in &input[..length - 1] {
            model.ingest(token).map_err(|_| "scalar ingestion failed")?;
        }
        let mut expected = vec![model.step(input[length - 1]).map_err(|_| "scalar readout failed")?];
        let scalar_seconds = started.elapsed().as_secs_f64();
        for &token in &input[length..] {
            expected.push(model.step(token).map_err(|_| "scalar continuation failed")?);
        }
        model.reset();
        profile::reset();
        profile::enable(true);
        let started = Instant::now();
        model.ingest_prefix(&input[..length - 1], || Ok(())).map_err(|_| "chunk ingestion failed")?;
        let mut actual = vec![model.step(input[length - 1]).map_err(|_| "chunk readout failed")?];
        let chunk_seconds = started.elapsed().as_secs_f64();
        profile::enable(false);
        let dispatch: std::collections::BTreeMap<_, _> = profile::snapshot()
            .into_iter()
            .filter(|(name, _)| name.contains("prefill") || name.contains("scratch_high_water"))
            .collect();
        let count = |name| dispatch.get(name).map_or(0, |counter| counter.bytes);
        let routes_observed =
            count("native.glm.prefill.chunk_tokens") > 0 || count("native.glm.prefill.wide_tokens") > 0;
        let requested_routes_observed = routes_observed
            && (wide.is_none() || count("native.glm.prefill.wide_tokens") == (length - 1) as u64)
            && (!control || count("glm.control.prefill.batch.tokens") > 0)
            && (!kda || count("glm.kda.prefill.register.tokens") > 0);
        for &token in &input[length..] {
            actual.push(model.step(token).map_err(|_| "chunk continuation failed")?);
        }
        let mut max_abs = 0.0_f64;
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        let mut all_close = true;
        let mut exact = true;
        let mut agreement = 0;
        for (reference, candidate) in expected.iter().zip(&actual) {
            if reference.is_empty() || reference.len() != candidate.len() {
                return Err("readout dimensions changed");
            }
            agreement += usize::from(best(reference) == best(candidate));
            for (&reference, &candidate) in reference.iter().zip(candidate) {
                if !reference.is_finite() || !candidate.is_finite() {
                    return Err("nonfinite readout");
                }
                let error = (f64::from(reference) - f64::from(candidate)).abs();
                max_abs = max_abs.max(error);
                squared_error += error * error;
                squared_reference += f64::from(reference).powi(2);
                all_close &= error <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * f64::from(reference).abs();
                exact &= reference.to_bits() == candidate.to_bits();
            }
        }
        let row_passed = all_close && agreement == expected.len() && requested_routes_observed;
        passed &= row_passed;
        println!(
            "{}",
            serde_json::json!({
                "input_tokens": length, "teacher_forced_continuation": 4,
                "corpus_tokens": encoded.len(), "cyclic_corpus": length + 4 > encoded.len(),
                "requested_routes": {"wide_tile": wide, "batched_control": control, "register_kda": kda},
                "requested_routes_observed": requested_routes_observed,
                "input_sha256": digest(input.iter().map(|value| value.to_le_bytes())),
                "corpus_sha256": format!("{:x}", Sha256::digest(&corpus)),
                "scalar_seconds": scalar_seconds, "chunk_seconds": chunk_seconds,
                "timing_scope": "diagnostic; candidate profiling enabled; not a controlled performance comparison",
                "candidate_dispatch": dispatch,
                "scalar_prefill_tokens_per_second": length as f64 / scalar_seconds,
                "chunk_prefill_tokens_per_second": length as f64 / chunk_seconds,
                "readout_rows": expected.len(), "vocabulary": expected[0].len(),
                "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
                "max_absolute_error": max_abs,
                "relative_l2_error": (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
                "greedy_agreement": agreement, "exact": exact, "all_close": all_close,
                "reference_sha256": digest(expected.iter().flatten().map(|value| value.to_le_bytes())),
                "candidate_sha256": digest(actual.iter().flatten().map(|value| value.to_le_bytes())),
                "passed": row_passed,
            })
        );
    }
    model.reset();
    Ok(passed)
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    match std::panic::catch_unwind(run) {
        Ok(Ok(true)) => {},
        Ok(Ok(false)) => std::process::exit(1),
        Ok(Err(message)) => {
            eprintln!("native prefill validation failed: {message}");
            std::process::exit(1);
        },
        Err(_) => {
            eprintln!("native prefill validation failed");
            std::process::exit(1);
        },
    }
}
