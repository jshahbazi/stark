// SPDX-License-Identifier: Apache-2.0
//! Bind the pinned Sushi STQ projection planes and verify the Engram lookup map.

use std::{env, error::Error, path::Path, time::Instant};

use serde_json::json;
use sha2::{Digest, Sha256};
use stark_engine::engine::hy4::{
    ImageInput, MAX_DRAFTS, Session, StqOptions, stq_layout::StqFormat, stq_source::StqSource,
};

struct Diagnostic {
    rows: usize,
    context: u32,
    chunk: usize,
    output: usize,
    repeats: usize,
    drafts: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: hy4_stq_inspect <artifact-directory> [rows context chunk output [repeats [drafts]]]")?;
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
        let test = Diagnostic {
            rows: rows.parse()?,
            context: args.next().ok_or("missing context")?.parse()?,
            chunk: args.next().ok_or("missing chunk")?.parse()?,
            output: args.next().ok_or("missing output count")?.parse()?,
            repeats: args.next().map_or(Ok(1), |n| n.parse())?,
            drafts: args.next().map_or(Ok(0), |n| n.parse())?,
        };
        if test.rows == 0
            || test.repeats == 0
            || test.rows.checked_add(test.output).is_none_or(|n| n >= test.context as usize)
            || !matches!(test.chunk, 1024 | 4096)
        {
            return Err("invalid diagnostic dimensions".into());
        }
        let tokenizer = tokenizers::Tokenizer::from_file(Path::new(&path).join("tokenizer.json"))
            .map_err(|_| "cannot open pinned-pack tokenizer")?;
        run_trunk(&source, &tokenizer, test)?;
    }
    Ok(())
}

fn run_trunk(
    source: &StqSource,
    tokenizer: &tokenizers::Tokenizer,
    test: Diagnostic,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    // The diagnostic is restricted to a privately installed, pinned, immutable
    // artifact. Keep it in place for the duration of the command.
    let profile = env::var_os("STARK_HY4_STQ_DIAGNOSTIC_PROFILE").is_some();
    let round_bf16 = env::var_os("STARK_HY4_STQ_DIAGNOSTIC_BF16").is_some();
    let pair1_decode = env::var_os("STARK_HY4_STQ_DIAGNOSTIC_PAIR1_DECODE").is_some();
    let resident = env::var_os("STARK_HY4_STQ_DIAGNOSTIC_RESIDENT").is_some();
    let mut session = unsafe {
        Session::load_stq_diagnostic(
            source,
            StqOptions {
                chunk: test.chunk as u32,
                context_limit: test.context,
                profile,
                round_bf16,
                pair1_decode,
                resident,
                tensor_mla: env::var_os("STARK_HY4_STQ_TENSOR_MLA").is_some(),
                gpu_verify_plan: env::var_os("STARK_HY4_STQ_GPU_VERIFY").is_some(),
                register_mla: env::var_os("STARK_HY4_STQ_REGISTER_MLA").is_some(),
                fused_finish_reduce: env::var_os("STARK_HY4_STQ_FUSED_FINISH").is_some(),
                split_spec1: env::var_os("STARK_HY4_STQ_SPLIT_SPEC1").is_some(),
                verify_register_mla: env::var_os("STARK_HY4_STQ_VERIFY_MLA").is_some(),
                cache_engram_pages: env::var_os("STARK_HY4_STQ_ENGRAM_PAGE_CACHE").is_some(),
                wide_attention: env::var_os("STARK_HY4_STQ_WIDE_ATTENTION").is_some(),
                wide_routed: env::var_os("STARK_HY4_STQ_WIDE_ROUTED").is_some(),
                draft_register_mla: env::var_os("STARK_HY4_STQ_DRAFT_REGISTER_MLA").is_some(),
                wide_ffn: env::var_os("STARK_HY4_STQ_WIDE_FFN").is_some(),
                fast_verify: env::var_os("STARK_HY4_STQ_FAST_VERIFY").is_some(),
                fused_attn_hc: env::var_os("STARK_HY4_STQ_FUSED_ATTN_HC").is_some(),
                batch_draft: env::var_os("STARK_HY4_STQ_BATCH_DRAFT").is_some(),
            },
        )
    }?;
    let context = session.context_sizing().context_tokens;
    if test.rows + test.output >= context as usize || test.drafts > session.max_drafts() {
        return Err("fitted context does not hold this diagnostic".into());
    }
    let load_seconds = started.elapsed().as_secs_f64();
    let tokens: Vec<u32> = (0..test.rows).map(|i| 256 + (i * 127 % 8192) as u32).collect();
    let mut expected = None;
    for repeat in 0..test.repeats {
        session.begin(&tokens, context, 0, test.drafts as u32, None).map_err(|e| format!("begin: {e}"))?;
        let started = Instant::now();
        for window in tokens.chunks(test.chunk) {
            session.prefill(window.len()).map_err(|e| format!("prefill: {e}"))?;
        }
        let prefill_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut output = Vec::with_capacity(test.output);
        if test.output != 0 {
            output.push(session.greedy()?);
        }
        if test.drafts > 0 && output.len() < test.output {
            session.append(output[0]).map_err(|e| format!("draft anchor: {e}"))?;
        }
        while output.len() < test.output {
            if test.drafts == 0 {
                session
                    .append(*output.last().ok_or("missing first greedy token")?)
                    .map_err(|e| format!("decode: {e}"))?;
                output.push(session.greedy()?);
            } else {
                let count = (test.output - output.len()).min(test.drafts + 1);
                let mut verified = [0; MAX_DRAFTS + 1];
                let kept = session.verify(count, &[], &mut verified).map_err(|e| format!("DFly2 verify: {e}"))?;
                output.extend_from_slice(&verified[..kept]);
            }
        }
        let decode_seconds = started.elapsed().as_secs_f64();
        let mut digest = Sha256::new();
        for &token in &output {
            digest.update(token.to_le_bytes());
        }
        if !session.logits()?.iter().all(|x| x.is_finite()) {
            return Err("nonfinite STQ logits".into());
        }
        let digest = digest.finalize();
        if *expected.get_or_insert(digest) != digest {
            return Err("STQ greedy output changed across fresh requests".into());
        }
        let text = tokenizer.decode(&output, false).map_err(|_| "cannot detokenize diagnostic")?;
        println!(
            "{}",
            json!({
                "repeat": repeat,
                "context": context,
                "chunk": test.chunk,
                "prefill_tokens": test.rows,
                "output_tokens": test.output,
                "decode_steps": test.output.saturating_sub(1),
                "draft_depth": test.drafts,
                "bf16_projection_boundaries": round_bf16,
                "pair1_single_token_gemv": pair1_decode,
                "mapped_weight_residency": resident,
                "vision_weights_loaded": session.vision_geometry().patch != 0,
                "load_seconds": load_seconds,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "output_sha256": format!("{digest:x}"),
                "text_sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
                "native_stats": session.stats(),
            })
        );
    }
    if env::var_os("STARK_HY4_STQ_DIAGNOSTIC_LIFECYCLE").is_some() {
        check_lifecycle(&mut session, &tokens, &test, context)?;
    }
    Ok(())
}

fn check_lifecycle(
    session: &mut Session,
    tokens: &[u32],
    test: &Diagnostic,
    context: u32,
) -> Result<(), Box<dyn Error>> {
    if test.rows < test.chunk || !test.rows.is_multiple_of(test.chunk) || test.rows + 1 >= context as usize {
        return Err("lifecycle diagnostic requires whole prefill chunks and room for a suffix".into());
    }
    let draft = test.drafts as u32;
    let mut continuation = tokens.to_vec();
    continuation.push(1616);
    session.retain_prefix(test.rows)?;
    session.begin(&continuation, context, test.rows, draft, None)?;
    session.prefill(1)?;
    let retained = session.logits()?.to_vec();
    session.begin(&continuation, context, 0, draft, None)?;
    for window in continuation.chunks(test.chunk) {
        session.prefill(window.len())?;
    }
    if retained.iter().zip(session.logits()?).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err("STQ prefix replay changed the logits".into());
    }
    let geometry = session.vision_geometry();
    let (width, height) = (96u32, 64u32);
    let count = (height / geometry.patch / geometry.merge) * (width / geometry.patch / geometry.merge + 1);
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i.wrapping_mul(17) % 251) as u8).collect();
    let image = ImageInput::new(&pixels, width, height);
    let mut image_tokens = vec![tokens[0], 120118];
    image_tokens.extend(std::iter::repeat_n(geometry.image_token, count as usize));
    image_tokens.push(120119);
    let pool = session.index_pool_group_tokens() as usize;
    image_tokens.extend_from_slice(&tokens[..pool]);
    session.begin(&image_tokens, context, 0, 0, Some(&image))?;
    session.prefill(image_tokens.len())?;
    if !session.logits()?.iter().all(|x| x.is_finite()) {
        return Err("nonfinite STQ image logits".into());
    }
    if session.retain_prefix(pool).is_ok() || session.begin(&image_tokens, context, pool, 0, None).is_ok() {
        return Err("STQ image allowed prefix reuse".into());
    }
    println!("{}", json!({"prefix_reuse_tokens": test.rows, "replay_logits_equal": true, "image_rows": count}));
    Ok(())
}
