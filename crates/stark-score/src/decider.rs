//! Decider has its own protocol: a noncausal backbone and a marker readout.
//! Request content and native exception details must never enter diagnostics.
use std::{
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{self, BufRead, Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::{Path, PathBuf},
    time::Instant,
};

use serde::Deserialize;
use serde_json::{Value, json};
use stark_engine::{backends::metal::Metal, engine::Engine};

const MAX_INPUT_TOKENS: usize = 4096;
const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    ResetCache {},
    Tokenize {
        text: String,
        #[serde(default)]
        add_special_tokens: bool,
    },
    Decide {
        input_token_ids: Vec<u32>,
        readout_positions: Vec<u32>,
        option_counts: Vec<usize>,
        #[serde(default)]
        images: Vec<PathBuf>,
        context_tokens: u32,
    },
}

struct Scores {
    logits: Vec<f32>,
    input_tokens: usize,
    state_creation_seconds: f64,
    prefill_seconds: f64,
    readout_seconds: f64,
}

trait Backend {
    fn has_images(&self) -> bool {
        true
    }
    fn vocab_size(&self) -> u32;
    fn max_context(&self) -> u32;
    fn num_options(&self) -> usize;
    fn tokenize(
        &self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str>;
    fn score(
        &mut self,
        tokens: &[u32],
        marker: u32,
        images: &[Vec<u8>],
        context: u32,
    ) -> Result<Scores, &'static str>;
}

struct NativeBackend {
    model: stark_engine::engine::decider::DeciderModel<Metal>,
}

impl Backend for NativeBackend {
    fn has_images(&self) -> bool {
        self.model.has_images()
    }
    fn vocab_size(&self) -> u32 {
        self.model.vocab_size()
    }
    fn max_context(&self) -> u32 {
        self.model.max_context_length().unwrap_or(MAX_INPUT_TOKENS as u32).min(MAX_INPUT_TOKENS as u32)
    }
    fn num_options(&self) -> usize {
        self.model.num_options()
    }
    fn tokenize(
        &self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        self.model
            .tokenizer()
            .encode(text, special)
            .map(|value| value.get_ids().to_vec())
            .map_err(|_| "tokenization_failed")
    }
    fn score(
        &mut self,
        tokens: &[u32],
        marker: u32,
        images: &[Vec<u8>],
        context: u32,
    ) -> Result<Scores, &'static str> {
        let result = if images.is_empty() {
            self.model.score(tokens, marker, context)
        } else {
            let encoded = images.iter().map(Vec::as_slice).collect::<Vec<_>>();
            self.model.score_images(tokens, marker, &encoded, context)
        }
        .map_err(|error| error.0)?;
        Ok(Scores {
            logits: result.logits,
            input_tokens: result.input_tokens,
            state_creation_seconds: result.state_creation_seconds,
            prefill_seconds: result.prefill_seconds,
            readout_seconds: result.readout_seconds,
        })
    }
}

fn read_image(path: &Path) -> Result<Vec<u8>, &'static str> {
    let mut file = File::open(path).map_err(|_| "invalid_image_request")?;
    let metadata = file.metadata().map_err(|_| "invalid_image_request")?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_IMAGE_BYTES {
        return Err("invalid_image_request");
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file).take(MAX_IMAGE_BYTES + 1).read_to_end(&mut bytes).map_err(|_| "invalid_image_request")?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_IMAGE_BYTES {
        return Err("invalid_image_request");
    }
    Ok(bytes)
}

fn respond<B: Backend>(
    backend: &mut B,
    request: Request,
) -> Result<Value, &'static str> {
    match request {
        // There is no retained KV, recurrent state or completed-distribution cache.
        Request::ResetCache {} => Ok(json!({"ok":true,"cache_cleared":true})),
        Request::Tokenize {
            text,
            add_special_tokens,
        } => Ok(json!({"ok":true,"input_token_ids":backend.tokenize(&text, add_special_tokens)?})),
        Request::Decide {
            input_token_ids,
            readout_positions,
            option_counts,
            images,
            context_tokens,
        } => {
            if input_token_ids.is_empty()
                || input_token_ids.len() > backend.max_context() as usize
                || context_tokens == 0
                || context_tokens > backend.max_context()
                || input_token_ids.len() > context_tokens as usize
                || input_token_ids.iter().any(|&id| id >= backend.vocab_size())
                || readout_positions.len() != 1
                || readout_positions[0] as usize != input_token_ids.len() - 1
                || option_counts.len() != 1
                || !(1..=backend.num_options()).contains(&option_counts[0])
            {
                return Err("invalid_decision_request");
            }
            if images.len() > 1 {
                return Err("too_many_images");
            }
            let encoded = images.iter().map(|path| read_image(path)).collect::<Result<Vec<_>, _>>()?;
            let start = Instant::now();
            let scores = backend.score(&input_token_ids, readout_positions[0], &encoded, context_tokens)?;
            if scores.logits.len() != backend.num_options()
                || scores.logits.iter().any(|x| !x.is_finite())
                || scores.input_tokens > context_tokens as usize
            {
                return Err("invalid_native_result");
            }
            let logits = &scores.logits[..option_counts[0]];
            Ok(json!({
                "ok":true,"logits":[logits],"input_tokens":scores.input_tokens,
                "cache_hit":false,"reused_input_tokens":0,
                "metrics": {
                    "state_creation_seconds":scores.state_creation_seconds,
                    "prefill_seconds":scores.prefill_seconds,
                    "readout_seconds":scores.readout_seconds,
                    "total_seconds":start.elapsed().as_secs_f64()
                }
            }))
        },
    }
}

fn write_reply(
    output: &mut impl Write,
    value: &Value,
) -> Result<(), &'static str> {
    serde_json::to_writer(&mut *output, value).map_err(|_| "protocol_write_failed")?;
    output.write_all(b"\n").map_err(|_| "protocol_write_failed")?;
    output.flush().map_err(|_| "protocol_write_failed")
}

// Drain an oversized frame before returning, so the next request is independent.
fn read_frame(input: &mut impl BufRead) -> Result<Option<Vec<u8>>, &'static str> {
    let mut bytes = Vec::new();
    loop {
        let part = input.fill_buf().map_err(|_| "protocol_read_failed")?;
        if part.is_empty() {
            return if bytes.is_empty() {
                Ok(None)
            } else {
                Ok(Some(bytes))
            };
        }
        let count = part.iter().position(|&b| b == b'\n').map_or(part.len(), |i| i + 1);
        let terminated = part[count - 1] == b'\n';
        if bytes.len().saturating_add(count) > MAX_LINE_BYTES {
            input.consume(count);
            if !terminated {
                loop {
                    let rest = input.fill_buf().map_err(|_| "protocol_read_failed")?;
                    if rest.is_empty() {
                        break;
                    }
                    let count = rest.iter().position(|&b| b == b'\n').map_or(rest.len(), |i| i + 1);
                    let done = rest[count - 1] == b'\n';
                    input.consume(count);
                    if done {
                        break;
                    }
                }
            }
            return Err("request_too_large");
        }
        bytes.extend_from_slice(&part[..count]);
        input.consume(count);
        if terminated {
            return Ok(Some(bytes));
        }
    }
}

fn run<B: Backend>(
    backend: &mut B,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    write_reply(
        output,
        &json!({
        "ready":true,"mode":"decider","images":backend.has_images(),"max_images":if backend.has_images() {1} else {0},
            "max_input_tokens":backend.max_context(),"options":backend.num_options(),
            "question_mode":"separate","locate":false,
            "prefix_cache":{"supported":false},"consecutive_cache":{"supported":false},
            "mtp":{"supported":false}
        }),
    )?;
    loop {
        let response = match read_frame(input) {
            Ok(None) => return Ok(()),
            Ok(Some(line)) => match serde_json::from_slice::<Request>(&line) {
                Ok(request) => respond(backend, request),
                Err(_) => Err("invalid_request"),
            },
            Err("request_too_large") => Err("request_too_large"),
            Err(error) => return Err(error),
        };
        write_reply(output, &response.unwrap_or_else(|error| json!({"ok":false,"error":error})))?;
    }
}

fn protocol_output() -> Result<File, &'static str> {
    let null = OpenOptions::new().write(true).open("/dev/null").map_err(|_| "output_setup_failed")?;
    let fd = unsafe { libc::fcntl(libc::STDOUT_FILENO, libc::F_DUPFD_CLOEXEC, 3) };
    if fd < 0 {
        return Err("output_setup_failed");
    }
    let output = unsafe { File::from_raw_fd(fd) };
    if unsafe { libc::dup2(null.as_raw_fd(), libc::STDOUT_FILENO) } < 0
        || unsafe { libc::dup2(null.as_raw_fd(), libc::STDERR_FILENO) } < 0
    {
        return Err("output_setup_failed");
    }
    Ok(output)
}

pub fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let Ok(output) = protocol_output() else {
        std::process::exit(1);
    };
    // Capture selectors must be removed before engine threads exist.
    unsafe {
        for (name, _) in std::env::vars_os() {
            let key = name.to_string_lossy();
            if key.starts_with("STARK_CAPTURE_") || key.starts_with("UZU_CAPTURE_") {
                std::env::remove_var(name);
            }
        }
    }
    let mut output = io::BufWriter::new(output);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut args = std::env::args_os().skip(1);
        if args.next().as_deref() != Some(OsStr::new("--model-path")) {
            return Err("invalid_arguments");
        }
        let path = PathBuf::from(args.next().ok_or("invalid_arguments")?);
        let text_validation = match args.next() {
            None => false,
            Some(flag) if flag == "--text-only-validation" => true,
            _ => return Err("invalid_arguments"),
        };
        if args.next().is_some() {
            return Err("invalid_arguments");
        }
        let engine = Engine::<Metal>::new().map_err(|_| "native_initialization_failed")?;
        let model = if text_validation {
            engine.load_decider_text_validation_model(&path)
        } else {
            engine.load_decider_model(&path)
        }
        .map_err(|error| error.0)?;
        let mut backend = NativeBackend {
            model,
        };
        run(&mut backend, &mut io::stdin().lock(), &mut output)
    }));
    let error = match outcome {
        Ok(Ok(())) => return,
        Ok(Err(error)) => error,
        Err(_) => "native_operation_failed",
    };
    let _ = write_reply(&mut output, &json!({"ready":false,"error":error}));
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake {
        calls: usize,
    }
    impl Backend for Fake {
        fn vocab_size(&self) -> u32 {
            100
        }
        fn max_context(&self) -> u32 {
            64
        }
        fn num_options(&self) -> usize {
            3
        }
        fn tokenize(
            &self,
            _: &str,
            _: bool,
        ) -> Result<Vec<u32>, &'static str> {
            Ok(vec![1, 2])
        }
        fn score(
            &mut self,
            tokens: &[u32],
            _: u32,
            _: &[Vec<u8>],
            _: u32,
        ) -> Result<Scores, &'static str> {
            self.calls += 1;
            Ok(Scores {
                logits: vec![3.0, 2.0, 1.0],
                input_tokens: tokens.len(),
                state_creation_seconds: 0.0,
                prefill_seconds: 0.0,
                readout_seconds: 0.0,
            })
        }
    }
    fn request() -> Value {
        json!({"operation":"decide","input_token_ids":[1,2],"readout_positions":[1],"option_counts":[2],"context_tokens":64})
    }
    #[test]
    fn validates_before_compute_and_recovers_without_retaining_state() {
        let mut backend = Fake {
            calls: 0,
        };
        let mut bad = request();
        bad["readout_positions"] = json!([0]);
        assert_eq!(
            respond(&mut backend, serde_json::from_value(bad).unwrap()).unwrap_err(),
            "invalid_decision_request"
        );
        let mut bad = request();
        bad["input_token_ids"] = json!([1, 100]);
        assert!(respond(&mut backend, serde_json::from_value(bad).unwrap()).is_err());
        let mut bad = request();
        bad["option_counts"] = json!([4]);
        assert!(respond(&mut backend, serde_json::from_value(bad).unwrap()).is_err());
        assert_eq!(backend.calls, 0);
        for _ in 0..2 {
            let reply = respond(&mut backend, serde_json::from_value(request()).unwrap()).unwrap();
            assert_eq!(reply["logits"], json!([[3.0, 2.0]]));
            assert_eq!(reply["cache_hit"], false);
        }
        assert_eq!(backend.calls, 2);
    }
    #[test]
    fn oversized_or_invalid_frames_do_not_corrupt_next_request() {
        let mut input = vec![b' '; MAX_LINE_BYTES + 1];
        input.extend_from_slice(b"\n{\"operation\":\"reset_cache\"}\n");
        let mut output = Vec::new();
        run(
            &mut Fake {
                calls: 0,
            },
            &mut io::Cursor::new(input),
            &mut output,
        )
        .unwrap();
        let replies: Vec<Value> =
            String::from_utf8(output).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(replies[1]["error"], "request_too_large");
        assert_eq!(replies[2]["cache_cleared"], true);
        assert!(serde_json::from_value::<Request>(json!({"operation":"reset_cache","content":"private"})).is_err());
    }
}
