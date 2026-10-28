//! Private adapter qualification against the pinned, untouched Lily oracle.
//!
//! Match /tmp/stark-lily-oracle/README.md: tokenizer 0.22, no special tokens,
//! exact F32/u32 little-endian hashes, Q8 K/V, no speculation or EOS stopping.
//! Timings include the adapter readout; teacher mode also checks fresh-state
//! replay and recovery from rejected input IDs, outside the primary timings.

use std::{
    fs::File,
    io::Write,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use serde::Serialize;
use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::Model;

const CAPACITY: usize = 16_384;
static OUTPUT: OnceLock<Mutex<File>> = OnceLock::new();

fn emit(bytes: &[u8]) {
    if let Some(output) = OUTPUT.get() {
        let mut output = output.lock().unwrap_or_else(|error| error.into_inner());
        let _ = output.write_all(bytes);
        let _ = output.write_all(b"\n");
        let _ = output.flush();
    }
}

fn fail(label: &'static str) -> ! {
    emit(format!("{{\"error\":\"{label}\"}}").as_bytes());
    std::process::exit(1)
}

fn checked<T, E>(
    result: Result<T, E>,
    label: &'static str,
) -> T {
    match result {
        Ok(value) => value,
        Err(_) => fail(label),
    }
}

#[cfg(unix)]
fn quiet_start() {
    use std::os::fd::{AsRawFd, FromRawFd};

    // Run before loading a core or starting its threads. No inherited upstream
    // diagnostic setting may change this qualification run's execution policy.
    let names = std::env::vars_os()
        .map(|(name, _)| name)
        .filter(|name| name.to_string_lossy().starts_with("LILY_"))
        .collect::<Vec<_>>();
    for name in names {
        unsafe { std::env::remove_var(name) };
    }
    std::panic::set_hook(Box::new(|_| fail("panic")));
    let saved = unsafe { libc::dup(1) };
    if saved < 0 {
        std::process::exit(1);
    }
    let saved = unsafe { File::from_raw_fd(saved) };
    if OUTPUT.set(Mutex::new(saved)).is_err() {
        fail("output_initialization");
    }
    let null = checked(File::options().read(true).write(true).open("/dev/null"), "null_output");
    if unsafe { libc::dup2(null.as_raw_fd(), 1) } < 0 || unsafe { libc::dup2(null.as_raw_fd(), 2) } < 0 {
        fail("output_redirection");
    }
}

#[cfg(not(unix))]
fn quiet_start() {
    // The qualified backend and its private oracle require macOS Metal.
    std::process::exit(1);
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Teacher,
    Greedy,
}

struct Options {
    model: PathBuf,
    mode: Mode,
    context: usize,
    steps: usize,
}

fn options() -> Options {
    let mut model = None;
    let mut mode = Mode::Teacher;
    let mut context = 0;
    let mut steps = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| fail("invalid_arguments"));
        match flag.to_str() {
            Some("--model") if model.is_none() => model = Some(PathBuf::from(value)),
            Some("--mode") => {
                mode = match value.to_str() {
                    Some("teacher") => Mode::Teacher,
                    Some("greedy") => Mode::Greedy,
                    _ => fail("invalid_mode"),
                };
            },
            Some("--kv-cache") if value == "q8" => {},
            Some("--context") => {
                context = checked(value.to_string_lossy().parse::<usize>(), "invalid_context");
                if ![0, 4096].contains(&context) {
                    fail("invalid_context");
                }
            },
            Some("--steps") => steps = Some(checked(value.to_string_lossy().parse::<usize>(), "invalid_steps")),
            _ => fail("invalid_arguments"),
        }
    }
    let steps = steps.unwrap_or(if mode == Mode::Greedy {
        256
    } else {
        32
    });
    if steps == 0 || steps > 256 || (mode == Mode::Greedy && steps != 256) {
        fail("invalid_steps");
    }
    Options {
        model: model.unwrap_or_else(|| fail("missing_model")),
        mode,
        context,
        steps,
    }
}

fn digest_ids(ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for id in ids {
        digest.update(id.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Scalar `>` preserves the first (lowest-ID) winner for equal finite logits,
/// including opposite signed zeroes. Model::step already rejects NaN/Inf.
fn hash_logits(
    logits: &[f32],
    vocab: usize,
    digest: &mut Sha256,
) -> Option<u32> {
    if logits.len() != vocab || logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        digest.update(value.to_bits().to_le_bytes());
        if *value > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).ok()
}

enum Readout {
    Logits(Vec<f32>),
    Greedy(u32),
}

fn readout(
    model: &mut Model,
    token: u32,
    mode: Mode,
) -> Readout {
    match mode {
        Mode::Teacher => Readout::Logits(checked(model.step(token), "model_step")),
        Mode::Greedy => Readout::Greedy(checked(model.step_greedy(token), "model_greedy_step")),
    }
}

fn record(
    readout: Readout,
    vocab: usize,
    logits: &mut Sha256,
    greedy: &mut Sha256,
) -> u32 {
    let token = match readout {
        Readout::Logits(values) => hash_logits(&values, vocab, logits).unwrap_or_else(|| fail("invalid_logits")),
        Readout::Greedy(token) if (token as usize) < vocab => token,
        Readout::Greedy(_) => fail("invalid_greedy_readout"),
    };
    greedy.update(token.to_le_bytes());
    token
}

#[derive(Serialize)]
struct Pass {
    logit_sha256: Option<String>,
    greedy_sha256: String,
    teacher_input_sha256: Option<String>,
    prefill_seconds: f64,
    decode_seconds: f64,
}

fn pass(
    model: &mut Model,
    input: &[u32],
    base: &[u32],
    mode: Mode,
    steps: usize,
) -> Pass {
    let started = Instant::now();
    for &token in &input[..input.len() - 1] {
        checked(model.ingest(token), "model_ingest");
    }
    let first = readout(model, input[input.len() - 1], mode);
    let prefill_seconds = started.elapsed().as_secs_f64();
    let mut logits_digest = Sha256::new();
    let mut greedy_digest = Sha256::new();
    let token = record(first, model.vocab_size(), &mut logits_digest, &mut greedy_digest);
    let mut forced = Vec::new();
    let started = Instant::now();
    if mode == Mode::Teacher {
        for index in 1..steps {
            let token = base[(index - 1) % base.len()];
            forced.push(token);
            record(readout(model, token, mode), model.vocab_size(), &mut logits_digest, &mut greedy_digest);
        }
    } else {
        let vocab_size = model.vocab_size();
        let stream = checked(model.greedy_decode(token, steps - 1), "greedy_decode_start");
        let mut count = 0;
        for result in stream {
            let next = checked(result, "greedy_decode");
            record(Readout::Greedy(next), vocab_size, &mut logits_digest, &mut greedy_digest);
            count += 1;
        }
        // The stream has dropped and drained its work, so a cleanup failure
        // cannot be hidden by an otherwise complete primary receipt.
        if model.is_fenced() || count != steps - 1 {
            fail("greedy_decode_cleanup");
        }
    }
    let decode_seconds = started.elapsed().as_secs_f64();
    Pass {
        logit_sha256: (mode == Mode::Teacher).then(|| format!("{:x}", logits_digest.finalize())),
        greedy_sha256: format!("{:x}", greedy_digest.finalize()),
        teacher_input_sha256: (mode == Mode::Teacher).then(|| digest_ids(&forced)),
        prefill_seconds,
        decode_seconds,
    }
}

fn equal_hashes(
    left: &Pass,
    right: &Pass,
) -> bool {
    left.logit_sha256 == right.logit_sha256
        && left.greedy_sha256 == right.greedy_sha256
        && left.teacher_input_sha256 == right.teacher_input_sha256
}

#[derive(Serialize)]
struct Recovery {
    reset_replay_matches: bool,
    invalid_ingest_rejected: bool,
    invalid_ingest_recovery_matches: bool,
    invalid_step_rejected: bool,
    invalid_step_recovery_matches: bool,
    replay_logit_sha256: Option<String>,
    replay_greedy_sha256: String,
}

fn recovery_checks(
    model: &mut Model,
    input: &[u32],
    base: &[u32],
    steps: usize,
    primary: &Pass,
) -> Recovery {
    model.reset();
    let replay = pass(model, input, base, Mode::Teacher, steps);
    let reset_replay_matches = equal_hashes(primary, &replay);

    // Exercise rejection with a pending prompt and with committed recurrent
    // state. Errors must discard all prior logical state without a new load.
    model.reset();
    checked(model.ingest(base[0]), "recovery_pending_ingest");
    let invalid = checked(u32::try_from(model.vocab_size()), "invalid_id_geometry");
    let invalid_ingest_rejected = model.ingest(invalid).is_err();
    if !invalid_ingest_rejected {
        fail("invalid_ingest_accepted");
    }
    let ingest_recovery = pass(model, input, base, Mode::Teacher, steps);
    let invalid_ingest_recovery_matches = equal_hashes(primary, &ingest_recovery);
    let invalid_step_rejected = model.step(invalid).is_err();
    if !invalid_step_rejected {
        fail("invalid_step_accepted");
    }
    let step_recovery = pass(model, input, base, Mode::Teacher, steps);
    let invalid_step_recovery_matches = equal_hashes(primary, &step_recovery);
    if !reset_replay_matches || !invalid_ingest_recovery_matches || !invalid_step_recovery_matches {
        fail("recovery_hash_mismatch");
    }
    Recovery {
        reset_replay_matches,
        invalid_ingest_rejected,
        invalid_ingest_recovery_matches,
        invalid_step_rejected,
        invalid_step_recovery_matches,
        replay_logit_sha256: replay.logit_sha256,
        replay_greedy_sha256: replay.greedy_sha256,
    }
}

#[derive(Serialize)]
struct Receipt {
    mode: &'static str,
    decode_mode: &'static str,
    kv_cache: &'static str,
    input_sha256: String,
    teacher_input_sha256: Option<String>,
    input_count: usize,
    logit_sha256: Option<String>,
    logit_readout_count: usize,
    greedy_sha256: String,
    greedy_count: usize,
    greedy_readout: &'static str,
    load_seconds: f64,
    prefill_seconds: f64,
    decode_seconds: f64,
    decode_steps: usize,
    prefill_timing: &'static str,
    recovery: Option<Recovery>,
}

fn run(options: Options) -> Receipt {
    // Same tokenizer crate/version and encoding flags as the private oracle.
    let tokenizer = checked(tokenizers::Tokenizer::from_file(options.model.join("tokenizer.json")), "tokenizer_load");
    let encoded = checked(tokenizer.encode("Hello.", false), "tokenizer_encode");
    let base = encoded.get_ids();
    if base.is_empty() {
        fail("empty_input");
    }
    let input = if options.context == 0 {
        base.to_vec()
    } else {
        base.iter().copied().cycle().take(options.context).collect::<Vec<_>>()
    };
    let started = Instant::now();
    let mut model = checked(Model::load(&options.model), "model_load");
    if !model.supports_greedy_readout() || model.max_context_length() as usize != CAPACITY {
        fail("mixed_backend_geometry");
    }
    if input.len() + options.steps >= CAPACITY || input.iter().any(|&id| id as usize >= model.vocab_size()) {
        fail("input_geometry");
    }
    let load_seconds = started.elapsed().as_secs_f64();
    let primary = pass(&mut model, &input, base, options.mode, options.steps);
    let recovery =
        (options.mode == Mode::Teacher).then(|| recovery_checks(&mut model, &input, base, options.steps, &primary));
    model.reset();
    Receipt {
        mode: if options.mode == Mode::Teacher {
            "teacher"
        } else {
            "greedy"
        },
        decode_mode: if options.mode == Mode::Teacher {
            "synchronous_teacher"
        } else {
            "safe_encode_ahead"
        },
        kv_cache: "q8",
        input_sha256: digest_ids(&input),
        teacher_input_sha256: primary.teacher_input_sha256,
        input_count: input.len(),
        logit_sha256: primary.logit_sha256,
        logit_readout_count: if options.mode == Mode::Teacher {
            options.steps
        } else {
            0
        },
        greedy_sha256: primary.greedy_sha256,
        greedy_count: options.steps,
        greedy_readout: if options.mode == Mode::Teacher {
            "cpu_lowest_id_argmax"
        } else {
            "gpu_greedy_id"
        },
        load_seconds,
        prefill_seconds: primary.prefill_seconds,
        decode_seconds: primary.decode_seconds,
        decode_steps: options.steps - 1,
        prefill_timing: "adapter_ingest_and_readout",
        recovery,
    }
}

fn main() {
    quiet_start();
    let receipt = run(options());
    emit(&checked(serde_json::to_vec(&receipt), "receipt_encoding"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_hash_has_fixed_little_endian_encoding() {
        let expected = Sha256::digest([0x78, 0x56, 0x34, 0x12, 0x01, 0xef, 0xcd, 0xab]);
        assert_eq!(digest_ids(&[0x1234_5678, 0xabcd_ef01]), format!("{expected:x}"));
    }

    #[test]
    fn full_float_hash_preserves_bits_and_lowest_id_ties() {
        let mut digest = Sha256::new();
        assert_eq!(hash_logits(&[0.0, -0.0, 1.0, 1.0], 4, &mut digest), Some(2));
        let expected = Sha256::digest([0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 63, 0, 0, 128, 63]);
        assert_eq!(format!("{:x}", digest.finalize()), format!("{expected:x}"));
        assert_eq!(hash_logits(&[-0.0, 0.0], 2, &mut Sha256::new()), Some(0));
    }

    #[test]
    fn malformed_readouts_are_rejected_before_hashing() {
        for values in [vec![], vec![1.0], vec![1.0, f32::NAN], vec![f32::INFINITY, 1.0]] {
            let mut digest = Sha256::new();
            assert_eq!(hash_logits(&values, 2, &mut digest), None);
            assert_eq!(format!("{:x}", digest.finalize()), format!("{:x}", Sha256::digest([])));
        }
    }
}
