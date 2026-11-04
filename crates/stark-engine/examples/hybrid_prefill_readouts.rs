//! Private full-vocabulary transport for compare_glm_prefill_readouts.py.
//! Readouts go only to an inherited anonymous pipe, never files or terminals.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    os::fd::FromRawFd,
    path::Path,
    time::Instant,
};

use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::{Model, activity::InferenceActivity, profile};

fn pipe(fd: i32) -> Result<BufWriter<File>, &'static str> {
    if fd <= 2 {
        return Err("invalid transport descriptor");
    }
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 || stat.st_mode & libc::S_IFMT != libc::S_IFIFO || stat.st_nlink != 0
    {
        return Err("transport must be an anonymous pipe");
    }
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || flags & libc::O_ACCMODE != libc::O_WRONLY {
        return Err("transport must be a writable pipe");
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
        return Err("transport inheritance could not be restricted");
    }
    // The parent explicitly transfers this descriptor to this process. Its
    // anonymous-pipe identity and direction have been checked above.
    Ok(BufWriter::new(unsafe { File::from_raw_fd(fd) }))
}

fn run() -> Result<(), &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("expected model, corpus, comma-separated lengths and pipe descriptor");
    }
    let lengths: Vec<usize> = args[2]
        .split(',')
        .map(|value| value.parse().map_err(|_| "invalid prefix lengths"))
        .collect::<Result<_, _>>()?;
    if lengths.is_empty() || lengths.len() > 16 || lengths.iter().any(|&length| !(2..=4096).contains(&length)) {
        return Err("invalid prefix limits");
    }
    let fd = args[3].parse().map_err(|_| "invalid transport descriptor")?;
    let mut transport = pipe(fd)?;
    let corpus = fs::read(&args[1]).map_err(|_| "corpus unavailable")?;
    if corpus.len() > 8 * 1024 * 1024 {
        return Err("corpus exceeds limit");
    }
    let text = std::str::from_utf8(&corpus).map_err(|_| "corpus is not UTF-8")?;
    let _activity = InferenceActivity::begin().map_err(|_| "activity assertion failed")?;
    let mut model = Model::load(Path::new(&args[0])).map_err(|_| "model admission failed")?;
    let encoded = model.tokenizer().encode(text, false).map_err(|_| "corpus tokenization failed")?;
    if encoded.is_empty() || lengths.iter().any(|&length| length + 4 > model.max_context_length() as usize) {
        return Err("invalid corpus or context capacity");
    }
    transport.write_all(b"SPRF001\n").map_err(|_| "transport failed")?;
    transport.write_all(&(lengths.len() as u32).to_le_bytes()).map_err(|_| "transport failed")?;
    for (case, length) in lengths.into_iter().enumerate() {
        let input: Vec<_> = encoded.get_ids().iter().copied().cycle().take(length + 4).collect();
        let mut input_hash = Sha256::new();
        for token in &input {
            input_hash.update(token.to_le_bytes());
        }
        model.reset();
        profile::reset();
        profile::enable(true);
        let started = Instant::now();
        model.ingest_prefix(&input[..length - 1], || Ok(())).map_err(|_| "prefix ingestion failed")?;
        let mut rows = vec![model.step(input[length - 1]).map_err(|_| "prefix readout failed")?];
        let prefill_seconds = started.elapsed().as_secs_f64();
        profile::enable(false);
        let dispatch: std::collections::BTreeMap<_, _> = profile::snapshot()
            .into_iter()
            .filter(|(name, _)| name.contains("prefill") || name.contains("scratch_high_water"))
            .collect();
        for &token in &input[length..] {
            rows.push(model.step(token).map_err(|_| "continuation failed")?);
        }
        let width = rows[0].len();
        if width == 0
            || width > 1_000_000
            || rows.iter().any(|row| row.len() != width || row.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid readout");
        }
        let metadata = serde_json::to_vec(&serde_json::json!({
            "input_tokens": length, "corpus_tokens": encoded.len(),
            "cyclic_corpus": length + 4 > encoded.len(),
            "input_sha256": format!("{:x}", input_hash.finalize()),
            "readout_rows": rows.len(), "vocabulary": width,
            "prefill_seconds": prefill_seconds, "dispatch": dispatch,
        }))
        .map_err(|_| "metadata encoding failed")?;
        if metadata.len() > 65_536 {
            return Err("metadata exceeds limit");
        }
        transport.write_all(&(metadata.len() as u32).to_le_bytes()).map_err(|_| "transport failed")?;
        transport.write_all(&metadata).map_err(|_| "transport failed")?;
        for value in rows.iter().flatten() {
            transport.write_all(&value.to_le_bytes()).map_err(|_| "transport failed")?;
        }
        transport.flush().map_err(|_| "transport failed")?;
        println!("{{\"completed_case\":{},\"input_tokens\":{length}}}", case + 1);
    }
    model.reset();
    Ok(())
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    match std::panic::catch_unwind(run) {
        Ok(Ok(())) => {},
        Ok(Err(message)) => {
            eprintln!("native prefill readout transport failed: {message}");
            std::process::exit(1);
        },
        Err(_) => {
            eprintln!("native prefill readout transport failed");
            std::process::exit(1);
        },
    }
}
