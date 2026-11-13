//! Private-pipe qualification of the worker's locked Rust tokenizer dependency.
//! Text and ID vectors are IPC payloads only. The supervising qualifier must
//! retain them in memory and publish aggregates/hashes; never redirect stdout
//! to a validation artifact. This helper creates no inference model/context.

use std::{
    fs,
    io::{self, BufRead, IsTerminal, Read, Write},
};

use serde_json::{Value, json};
use tokenizers::{
    OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer, pre_tokenizers::PreTokenizerWrapper,
};

fn strings(value: &Value) -> Result<Vec<&str>, &'static str> {
    value
        .as_array()
        .filter(|values| values.len() <= 1024)
        .ok_or("invalid_batch")?
        .iter()
        .map(|value| value.as_str().ok_or("invalid_text"))
        .collect()
}

fn run() -> Result<(), &'static str> {
    if io::stdin().is_terminal() || io::stdout().is_terminal() {
        return Err("private_pipes_required");
    }
    let path = std::env::args_os().nth(1).ok_or("tokenizer_path_required")?;
    let program = fs::read(path).map_err(|_| "tokenizer_unavailable")?;
    if program.len() > 16 * 1024 * 1024 {
        return Err("tokenizer_size");
    }
    let mut tokenizer = Tokenizer::from_bytes(&program).map_err(|_| "tokenizer_invalid")?;
    let mut splitter = split_program(&serde_json::from_slice(&program).map_err(|_| "tokenizer_invalid")?)?;
    let mut input = io::stdin().lock();
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(output, "{{\"ready\":true,\"tokenizers\":\"0.22.2\"}}").map_err(|_| "pipe_write")?;
    output.flush().map_err(|_| "pipe_write")?;
    loop {
        let mut bytes = Vec::new();
        let read = input.by_ref().take(16 * 1024 * 1024 + 1).read_until(b'\n', &mut bytes).map_err(|_| "pipe_read")?;
        if read == 0 {
            break;
        }
        if bytes.len() > 16 * 1024 * 1024 || bytes.last() != Some(&b'\n') {
            return Err("request_size");
        }
        let request: Value = serde_json::from_slice(&bytes).map_err(|_| "request_invalid")?;
        let result = match request.get("operation").and_then(Value::as_str) {
            Some("encode") => {
                let special = request.get("special").and_then(Value::as_bool).ok_or("special_required")?;
                let values = strings(&request["texts"])?;
                let encoded = tokenizer.encode_batch(values, special).map_err(|_| "encode_failed")?;
                json!({"ids": encoded.iter().map(|value| value.get_ids()).collect::<Vec<_>>()})
            },
            Some("decode" | "stream") => {
                let skip = request.get("skip").and_then(Value::as_bool).ok_or("skip_required")?;
                let sequences: Vec<Vec<u32>> =
                    serde_json::from_value(request["ids"].clone()).map_err(|_| "ids_invalid")?;
                if sequences.len() > 1024 || sequences.iter().any(|ids| ids.len() > 16384) {
                    return Err("ids_size");
                }
                let mut texts = Vec::with_capacity(sequences.len());
                for ids in &sequences {
                    if request["operation"] == "stream" {
                        let mut decoder = tokenizer.decode_stream(skip);
                        let mut text = String::new();
                        for &id in ids {
                            if let Some(piece) = decoder.step(id).map_err(|_| "stream_failed")? {
                                text.push_str(&piece);
                            }
                        }
                        texts.push(text);
                    } else {
                        texts.push(tokenizer.decode(ids, skip).map_err(|_| "decode_failed")?);
                    }
                }
                json!({"texts":texts})
            },
            Some("boundaries") => {
                let mut all = Vec::new();
                for text in strings(&request["texts"])? {
                    let mut pieces = PreTokenizedString::from(text);
                    splitter.pre_tokenize(&mut pieces).map_err(|_| "split_failed")?;
                    all.push(
                        pieces
                            .get_splits(OffsetReferential::Original, OffsetType::Byte)
                            .iter()
                            .map(|(_, (start, end), _)| end - start)
                            .collect::<Vec<_>>(),
                    );
                }
                json!({"boundaries":all})
            },
            Some("replace") => {
                let program = &request["program"];
                let replacement = Tokenizer::from_bytes(serde_json::to_vec(program).map_err(|_| "program_invalid")?)
                    .map_err(|_| "program_invalid")?;
                let replacement_splitter = split_program(program)?;
                tokenizer = replacement;
                splitter = replacement_splitter;
                json!({"replaced":true})
            },
            _ => return Err("operation_invalid"),
        };
        serde_json::to_writer(&mut output, &result).map_err(|_| "pipe_write")?;
        writeln!(output).map_err(|_| "pipe_write")?;
        output.flush().map_err(|_| "pipe_write")?;
    }
    Ok(())
}

fn split_program(program: &Value) -> Result<PreTokenizerWrapper, &'static str> {
    serde_json::from_value(program.pointer("/pre_tokenizer/pretokenizers/0").ok_or("split_program_invalid")?.clone())
        .map_err(|_| "split_program_invalid")
}

fn main() {
    if let Err(code) = run() {
        eprintln!("{code}");
        std::process::exit(1);
    }
}
