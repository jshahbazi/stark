// SPDX-License-Identifier: Apache-2.0
//! Bind the pinned Sushi STQ projection planes and verify the Engram lookup map.

use std::{env, error::Error, path::Path, time::Instant};

use serde_json::json;
use sha2::{Digest, Sha256};
use stark_engine::engine::hy4::{Session, stq_layout::StqFormat, stq_source::StqSource};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: hy4_stq_inspect <artifact-directory> [rows context chunk decode]")?;
    let source = StqSource::open(Path::new(&path))?;
    let trunk = source.stq_trunk()?;
    let mut pair1 = 0;
    for block in 1..42 {
        let banks = source.stq_routed_experts(block).ok_or("invalid STQ routed expert group")?;
        if banks.iter().any(|bank| bank.format != StqFormat::Pair1K4) {
            return Err("unexpected STQ routed expert format".into());
        }
        pair1 += banks.len();
    }

    let mut dense = 0;
    let mut draft = 0;
    for name in source.tensor_names().filter_map(|name| name.strip_suffix(".trellis")) {
        let bank = source.stq_projection(name).ok_or("invalid STQ dense projection")?;
        match bank.format {
            StqFormat::Spec1K6 => dense += 1,
            StqFormat::Spec1K4 => draft += 1,
            StqFormat::Pair1K4 => return Err("unexpected STQ dense format".into()),
        }
    }
    if pair1 != 123 || dense + draft != 464 || draft == 0 {
        return Err("unexpected STQ projection counts".into());
    }

    let mut engram_rows = 0u64;
    for block in [1, 11, 21, 31] {
        let table = source.stq_engram_table(block).ok_or("invalid STQ Engram table")?;
        engram_rows += u64::from(table.rows);
    }
    let token_map_entries = source.stq_engram_token_map()?.len();
    source.verify_unchanged()?;
    println!(
        "{}",
        json!({
            "shards": source.shard_count(),
            "tensors": source.tensor_count(),
            "trunk_aliases": trunk.weights().len(),
            "pair1_k4_banks": pair1,
            "spec1_k6_projections": dense,
            "spec1_k4_projections": draft,
            "engram_rows": engram_rows,
            "engram_token_map_entries": token_map_entries,
        })
    );
    if let Some(rows) = args.next() {
        let rows: usize = rows.parse()?;
        let context: u32 = args.next().ok_or("missing context")?.parse()?;
        let chunk: usize = args.next().ok_or("missing chunk")?.parse()?;
        let decode: usize = args.next().ok_or("missing decode count")?.parse()?;
        if rows == 0 || rows.checked_add(decode).is_none_or(|n| n >= context as usize) || !matches!(chunk, 1024 | 4096)
        {
            return Err("invalid diagnostic dimensions".into());
        }
        run_trunk(&source, rows, context, chunk, decode)?;
    }
    Ok(())
}

fn run_trunk(
    source: &StqSource,
    rows: usize,
    context: u32,
    chunk: usize,
    decode: usize,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    // The diagnostic is restricted to a privately installed, pinned, immutable
    // artifact. Keep it in place for the duration of the command.
    let mut session = unsafe { Session::load_stq_diagnostic(source, chunk as u32, context) }?;
    let context = session.context_sizing().context_tokens;
    if rows + decode >= context as usize {
        return Err("fitted context does not hold this diagnostic".into());
    }
    let load_seconds = started.elapsed().as_secs_f64();
    let tokens: Vec<u32> = (0..rows).map(|i| 256 + (i * 127 % 8192) as u32).collect();
    session.begin(&tokens, context, 0, 0, None).map_err(|e| format!("begin: {e}"))?;
    let started = Instant::now();
    for window in tokens.chunks(chunk) {
        session.prefill(window.len()).map_err(|e| format!("prefill: {e}"))?;
    }
    let prefill_seconds = started.elapsed().as_secs_f64();
    if !session.logits()?.iter().all(|x| x.is_finite()) {
        return Err("nonfinite STQ logits".into());
    }
    let started = Instant::now();
    let mut digest = Sha256::new();
    for _ in 0..decode {
        let token = session.greedy()?;
        digest.update(token.to_le_bytes());
        session.append(token).map_err(|e| format!("decode: {e}"))?;
    }
    let decode_seconds = started.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({
            "context": context,
            "chunk": chunk,
            "prefill_tokens": rows,
            "decode_steps": decode,
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "output_sha256": format!("{:x}", digest.finalize()),
            "native_stats": session.stats(),
        })
    );
    Ok(())
}
