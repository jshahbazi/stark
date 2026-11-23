// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only screen of the affine 8-bit dense and readout kernels on
//! restored real model state. Reports GPU times in alternating pairs and
//! numeric differences; never prints tensor values or tokens.

use std::path::Path;

use half::bf16;
use serde_json::{Value, json};

use super::{prefill::project_rows, schedule_validation::worker_ranges, *};
use crate::engine::inkling_slice::affine8::{self, Affine8, Affine8Weights, Planes};

/// Dense tensors per physical layer, in visit-segment order, then the output
/// projection: (name, rows, columns, input slot). Slot 3 is the attended row.
const DENSE: [(&str, usize, usize, usize); 9] = [
    ("attn.wq_du.weight", 3072, WIDTH, 0),
    ("attn.wk_dv.weight", KV, WIDTH, 1),
    ("attn.wv_dv.weight", KV, WIDTH, 2),
    ("attn.wr_du.weight", HEADS * 16, WIDTH, 0),
    ("attn.gate_attn.weight", 3072, WIDTH, 0),
    ("attn.attn_sink_proj.weight", HEADS, WIDTH, 0),
    ("attn.k_sconv.weight_proj", 32, WIDTH, 1),
    ("attn.v_sconv.weight_proj", 32, WIDTH, 2),
    ("attn.wo_ud.weight", WIDTH, 3072, 3),
];
const OUTPUT: usize = 8;
const ROWS: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// Original BF16 weights and kernels.
    Original,
    /// BF16 kernels on the BF16 copy of the effective affine weights.
    Copy,
    /// Affine 8-bit kernels.
    Affine,
}

struct Dense {
    planes: Vec<Vec<Option<Planes>>>,
    copies: Vec<Vec<Option<Buffer>>>,
    /// Per visit, row and tensor index.
    outputs: Vec<Vec<Vec<Buffer>>>,
}

type Results = Vec<Vec<Vec<Vec<f32>>>>;

fn segments(layer: &Layer) -> usize {
    if layer.dynamic.is_some() {
        8
    } else {
        6
    }
}

fn live(
    layer: &Layer,
    index: usize,
) -> bool {
    index < segments(layer) || index == OUTPUT
}

fn original(
    layer: &Layer,
    index: usize,
) -> &Buffer {
    match index {
        0 => &layer.q,
        1 => &layer.k,
        2 => &layer.v,
        3 => &layer.r,
        4 => &layer.attn_gate,
        5 => &layer.sink,
        6 | 7 => &layer.dynamic.as_ref().expect("global layer")[index - 6],
        _ => &layer.output,
    }
}

fn source_visit(
    visit: usize,
    row: usize,
) -> usize {
    (visit + 12 * row) % PHYSICAL.len()
}

fn input(
    session: &Session,
    visit: usize,
    row: usize,
    slot: usize,
) -> &Buffer {
    let scratch = &session.visits[source_visit(visit, row)].scratch;
    if slot == 3 {
        &scratch.attended
    } else {
        &scratch.qkv_input[slot]
    }
}

fn weight<'a>(
    session: &'a Session,
    dense: &'a Dense,
    arm: Arm,
    physical: usize,
    index: usize,
) -> &'a Buffer {
    match arm {
        Arm::Original => original(&session.layers[physical], index),
        Arm::Copy | Arm::Affine => dense.copies[physical][index].as_ref().expect("loaded copy"),
    }
}

/// The dense stage of one step (one row) or one verifier round (two to four
/// rows), dispatched exactly as the decode paths dispatch it.
fn encode_dense(
    gpu: &mut Stream<'_>,
    session: &Session,
    dense: &Dense,
    arm: Arm,
    rows: usize,
) {
    for (visit, &physical) in PHYSICAL.iter().enumerate() {
        let layer = &session.layers[physical];
        let count = segments(layer);
        let out = &dense.outputs[visit];
        let fused =
            session.prefill_lanes.as_ref().is_some_and(|lanes| lanes.fuses_decode(gpu, &session.visits[visit].scratch));
        if rows == 1 && fused {
            let segs = (0..count)
                .map(|index| (weight(session, dense, arm, physical, index), &out[0][index]))
                .collect::<Vec<_>>();
            Layer::project_visit_segments(gpu, &session.visits[visit].scratch.qkv_input, &segs);
        } else {
            for index in 0..count {
                let (_, outs, cols, slot) = DENSE[index];
                let bindings = (0..rows).map(|row| (binding(input(session, visit, row, slot)), &out[row][index]));
                project_rows(gpu, None, weight(session, dense, arm, physical, index), bindings, cols, outs, true);
            }
        }
        let (_, outs, cols, slot) = DENSE[OUTPUT];
        let bindings = (0..rows).map(|row| (binding(input(session, visit, row, slot)), &out[row][OUTPUT]));
        project_rows(gpu, None, weight(session, dense, arm, physical, OUTPUT), bindings, cols, outs, true);
    }
}

fn bf16s(
    buffer: &Buffer,
    count: usize,
) -> Vec<f32> {
    read_shared::<u16>(buffer, count).into_iter().map(|bits| bf16::from_bits(bits).to_f32()).collect()
}

fn read_dense(
    session: &Session,
    dense: &Dense,
    rows: usize,
) -> Results {
    dense
        .outputs
        .iter()
        .enumerate()
        .map(|(visit, out)| {
            let layer = &session.layers[PHYSICAL[visit]];
            (0..rows)
                .map(|row| {
                    (0..DENSE.len())
                        .map(|index| {
                            if live(layer, index) {
                                bf16s(&out[row][index], DENSE[index].1)
                            } else {
                                Vec::new()
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn read_logits(
    session: &Session,
    rows: usize,
) -> Vec<Vec<f32>> {
    let all = read_shared::<f32>(&session.logits, rows * PADDED_VOCABULARY);
    all.chunks(PADDED_VOCABULARY).map(|row| row[..VOCABULARY].to_vec()).collect()
}

fn set_arm(
    session: &mut Session,
    stash: &mut Option<Affine8Weights>,
    arm: Arm,
) {
    if arm == Arm::Original {
        if let Some(weights) = session.runtime.affine8.take() {
            *stash = Some(weights);
        }
    } else {
        if session.runtime.affine8.is_none() {
            session.runtime.affine8 = stash.take();
        }
        if let Some(weights) = session.runtime.affine8.as_mut() {
            weights.decode = arm == Arm::Affine;
        }
    }
}

pub(super) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

pub(super) fn range(values: &[f64]) -> [f64; 2] {
    [values.iter().copied().fold(f64::INFINITY, f64::min), values.iter().copied().fold(f64::NEG_INFINITY, f64::max)]
}

pub(super) fn timing_summary(pairs: &[(f64, f64)]) -> Value {
    let mut base = pairs.iter().map(|p| p.0).collect::<Vec<_>>();
    let mut candidate = pairs.iter().map(|p| p.1).collect::<Vec<_>>();
    let mut saved = pairs.iter().map(|p| p.0 - p.1).collect::<Vec<_>>();
    let mut ratio = pairs.iter().map(|p| p.1 / p.0).collect::<Vec<_>>();
    json!({
        "pairs": pairs.len(),
        "bf16_ms_range": range(&base), "bf16_ms_median": median(&mut base),
        "affine8_ms_range": range(&candidate), "affine8_ms_median": median(&mut candidate),
        "saved_ms_range": range(&saved), "saved_ms_median_of_pairs": median(&mut saved),
        "ratio_median_of_pairs": median(&mut ratio),
        "affine8_faster_pairs": pairs.iter().filter(|p| p.1 < p.0).count(),
    })
}

/// Difference statistics against a reference.
#[derive(Default)]
struct Error {
    count: usize,
    unequal_bf16: usize,
    squared: f64,
    reference: f64,
    max_abs: f64,
}

impl Error {
    fn add(
        &mut self,
        got: f32,
        want: f64,
    ) {
        self.count += 1;
        let delta = f64::from(got) - want;
        self.squared += delta * delta;
        self.reference += want * want;
        self.max_abs = self.max_abs.max(delta.abs());
        if bf16::from_f64(want).to_bits() != bf16::from_f32(got).to_bits() {
            self.unequal_bf16 += 1;
        }
    }

    fn report(&self) -> Value {
        json!({
            "values": self.count,
            "relative_l2": (self.squared / self.reference.max(f64::MIN_POSITIVE)).sqrt(),
            "max_abs": self.max_abs,
            "outside_bf16_rounding_of_reference": self.unequal_bf16,
        })
    }
}

fn top(values: &[f32]) -> usize {
    transaction_validation::greedy(values) as usize
}

/// FP64 products of each live tensor row with every (visit, row) input that
/// reads it, for BF16-rounded and unrounded dequantized weights.
fn dense_reference(
    session: &Session,
    dense: &Dense,
    rows: usize,
    copy: &Results,
    affine: &Results,
) -> [Error; 3] {
    let mut rounded = Error::default();
    let mut unrounded = Error::default();
    let mut copy_rounded = Error::default();
    for (physical, planes) in dense.planes.iter().enumerate() {
        let visits = PHYSICAL.iter().enumerate().filter(|&(_, &p)| p == physical).map(|(v, _)| v).collect::<Vec<_>>();
        for (index, tensor) in planes.iter().enumerate() {
            let Some(tensor) = tensor else {
                continue;
            };
            let slot = DENSE[index].3;
            let inputs = visits
                .iter()
                .flat_map(|&visit| (0..rows).map(move |row| (visit, row)))
                .map(|(visit, row)| (visit, row, bf16s(input(session, visit, row, slot), DENSE[index].2)))
                .collect::<Vec<_>>();
            let threads = 12;
            let chunk = tensor.rows.div_ceil(threads);
            let products = std::thread::scope(|scope| {
                let handles = (0..threads)
                    .map(|part| {
                        let inputs = &inputs;
                        scope.spawn(move || {
                            let mut out = Vec::new();
                            for out_row in part * chunk..((part + 1) * chunk).min(tensor.rows) {
                                let w_round = tensor.row(out_row, true);
                                let w_exact = tensor.row(out_row, false);
                                for (visit, row, x) in inputs {
                                    let r =
                                        w_round.iter().zip(x).map(|(&w, &x)| f64::from(w) * f64::from(x)).sum::<f64>();
                                    let e =
                                        w_exact.iter().zip(x).map(|(&w, &x)| f64::from(w) * f64::from(x)).sum::<f64>();
                                    out.push((*visit, *row, out_row, r, e));
                                }
                            }
                            out
                        })
                    })
                    .collect::<Vec<_>>();
                handles.into_iter().flat_map(|handle| handle.join().expect("reference thread")).collect::<Vec<_>>()
            });
            for (visit, row, out_row, r, e) in products {
                rounded.add(affine[visit][row][index][out_row], r);
                unrounded.add(affine[visit][row][index][out_row], e);
                copy_rounded.add(copy[visit][row][index][out_row], r);
            }
        }
    }
    [rounded, unrounded, copy_rounded]
}

/// FP64 capped logits of each hidden row with the BF16 effective head.
fn readout_reference(
    head: &Planes,
    hidden: &[Vec<f32>],
) -> Vec<Vec<f64>> {
    let threads = 12;
    let chunk = VOCABULARY.div_ceil(threads);
    let parts = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|part| {
                scope.spawn(move || {
                    let mut out = vec![Vec::new(); hidden.len()];
                    for column in part * chunk..((part + 1) * chunk).min(VOCABULARY) {
                        let w = head.row(column, true);
                        for (row, x) in hidden.iter().enumerate() {
                            let dot = w.iter().zip(x).map(|(&w, &x)| f64::from(w) * f64::from(x)).sum::<f64>();
                            out[row].push(30.0 * (dot / 30.0).tanh());
                        }
                    }
                    out
                })
            })
            .collect::<Vec<_>>();
        handles.into_iter().map(|handle| handle.join().expect("reference thread")).collect::<Vec<_>>()
    });
    (0..hidden.len()).map(|row| parts.iter().flat_map(|part| part[row].iter().copied()).collect()).collect()
}

/// Screen the affine 8-bit decode kernels against the BF16 kernels.
pub fn measure_affine8_dense(
    checkpoint: &Path,
    draft: &Path,
    tensors: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    pairs: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = Session::load(checkpoint, context)?;
    session.load_dspark(draft)?;
    session.set_batched_prefill(true, true)?;
    session.set_source_verifier(true)?;
    session.prepare_dspark_int8_head()?;
    session.set_dspark_enabled(true)?;
    let source_active = session.runtime.source_text_active;
    for range in worker_ranges(prompt, source_active, session.prefill_batch_rows()) {
        session.ingest_many(&tokens[range])?;
    }
    // Four real decode steps: their hidden rows feed the readout, and the last
    // step leaves real inputs in every visit's scratch.
    let mut hidden_rows = Vec::with_capacity(ROWS);
    let mut token = tokens[prompt - 1];
    for _ in 0..ROWS {
        let logits = session.step(token)?;
        hidden_rows.push(read_shared::<u16>(&session.hidden, WIDTH));
        token = transaction_validation::greedy(&logits);
    }

    // Affine tensors from the converter output, with their BF16 copies.
    affine8::register(&mut session.runtime)?;
    let source = Tensors::open(tensors)?;
    let device = session.runtime.device.clone();
    let embed = Planes::read(&source, "model.llm.embed.weight", [STORAGE_VOCABULARY, WIDTH])?;
    let embedding_copy = embed.effective_buffer(&device, STORAGE_VOCABULARY, STORAGE_VOCABULARY)?;
    let embedding = Affine8::new(&device, &embed, STORAGE_VOCABULARY)?;
    drop(embed);
    let unembed = Planes::read(&source, "model.llm.unembed.weight", [STORAGE_VOCABULARY, WIDTH])?;
    let mut head_copy = unembed.effective_buffer(&device, PADDED_VOCABULARY, PADDED_VOCABULARY)?;
    let head = Affine8::new(&device, &unembed, PADDED_VOCABULARY)?;
    let mut weights = Affine8Weights::new(embedding, head);
    let mut dense = Dense {
        planes: Vec::new(),
        copies: Vec::new(),
        outputs: (0..PHYSICAL.len())
            .map(|_| {
                (0..ROWS)
                    .map(|_| DENSE.iter().map(|&(_, rows, _, _)| zeroed(&device, rows * 2)).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?,
    };
    for (physical, layer) in session.layers.iter().enumerate() {
        let mut planes = Vec::new();
        let mut copies = Vec::new();
        for (index, &(name, rows, cols, _)) in DENSE.iter().enumerate() {
            if !live(layer, index) {
                planes.push(None);
                copies.push(None);
                continue;
            }
            let tensor = Planes::read(&source, &format!("model.llm.layers.{physical}.{name}"), [rows, cols])?;
            let copy = tensor.effective_buffer(&device, rows, rows)?;
            weights.insert(&copy, Affine8::new(&device, &tensor, rows)?);
            planes.push(Some(tensor));
            copies.push(Some(copy));
        }
        dense.planes.push(planes);
        dense.copies.push(copies);
    }

    // GPU dequantization must reproduce the load-time BF16 copies exactly.
    let mut dequantized = [0usize; 2];
    {
        let mut tensors = vec![(&weights.head, &head_copy), (&weights.embedding, &embedding_copy)];
        for copy in dense.copies.iter().flatten().flatten() {
            tensors.push((weights.tensor_for(copy).ok_or("missing affine tensor")?, copy));
        }
        for (tensor, copy) in tensors {
            let out = zeroed(&device, tensor.rows * tensor.cols * 2)?;
            execute(&session.runtime, |gpu| tensor.encode_dequantize(gpu, &out))?;
            let a = read_shared::<u16>(&out, tensor.rows * tensor.cols);
            let b = read_shared::<u16>(copy, tensor.rows * tensor.cols);
            dequantized[0] += a.len();
            dequantized[1] += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        }
    }

    // The 8-bit embedding lookup against the BF16 lookup on the copy.
    let mut embedding_unequal = 0usize;
    {
        let [a, b] = [zeroed(&device, WIDTH * 4)?, zeroed(&device, WIDTH * 4)?];
        for &token in tokens.iter().take(64) {
            execute(&session.runtime, |gpu| {
                gpu.one("ink_embedding", &[&embedding_copy, &session.embed_norm, &a], token, (1, 1), WIDTH / 4);
                gpu.one(
                    "ink_a8_embedding",
                    &[&weights.embedding.tensor, &session.embed_norm, &b],
                    [token, STORAGE_VOCABULARY as u32],
                    (1, 1),
                    WIDTH / 4,
                );
            })?;
            let (x, y) = (read_shared::<u32>(&a, WIDTH), read_shared::<u32>(&b, WIDTH));
            embedding_unequal += x.iter().zip(&y).filter(|(x, y)| x != y).count();
        }
    }

    // Readout rows: the hidden rows of the four real steps.
    for (row, values) in hidden_rows.iter().enumerate() {
        let out = unsafe {
            std::slice::from_raw_parts_mut(session.hidden.contents().as_ptr().cast::<u16>().add(row * WIDTH), WIDTH)
        };
        out.copy_from_slice(values);
    }
    let hidden = hidden_rows
        .iter()
        .map(|row| row.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let reference_logits = readout_reference(&unembed, &hidden);
    drop(unembed);

    let mut stash = Some(weights);
    let mut dense_checks = Vec::new();
    let mut readout_checks = Vec::new();
    for rows in 1..=ROWS {
        let mut results = Vec::new();
        for arm in [Arm::Original, Arm::Copy, Arm::Affine] {
            set_arm(&mut session, &mut stash, arm);
            execute(&session.runtime, |gpu| encode_dense(gpu, &session, &dense, arm, rows))?;
            results.push(read_dense(&session, &dense, rows));
        }
        let (base, copy, affine) = (&results[0], &results[1], &results[2]);
        let mut unequal = 0usize;
        let mut total = 0usize;
        let mut quantization = Error::default();
        for ((a, b), c) in base.iter().zip(copy).zip(affine) {
            for ((a, b), c) in a.iter().zip(b).zip(c) {
                for ((a, b), c) in a.iter().zip(b).zip(c) {
                    total += b.len();
                    unequal += b.iter().zip(c).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    for (&want, &got) in a.iter().zip(c) {
                        quantization.add(got, f64::from(want));
                    }
                }
            }
        }
        let [rounded, unrounded, copy_rounded] = dense_reference(&session, &dense, rows, copy, affine);
        dense_checks.push(json!({
            "rows": rows,
            "values": total,
            "affine8_unequal_to_bf16_kernel_on_copy": unequal,
            "affine8_vs_fp64_bf16_effective_weights": rounded.report(),
            "bf16_kernel_on_copy_vs_fp64_bf16_effective_weights": copy_rounded.report(),
            "affine8_vs_fp64_unrounded_dequantized_weights": unrounded.report(),
            "affine8_vs_original_bf16_weights": quantization.report(),
        }));

        let mut logits = Vec::new();
        for arm in [Arm::Original, Arm::Copy, Arm::Affine] {
            set_arm(&mut session, &mut stash, arm);
            if arm == Arm::Copy {
                std::mem::swap(&mut session.unembedding, &mut head_copy);
            }
            execute(&session.runtime, |gpu| session.encode_output_rows(gpu, rows))?;
            if arm == Arm::Copy {
                std::mem::swap(&mut session.unembedding, &mut head_copy);
            }
            logits.push(read_logits(&session, rows));
        }
        let mut affine_reference = Error::default();
        let mut copy_reference = Error::default();
        let mut quantized = Error::default();
        let mut copy_unequal = 0usize;
        let mut top1 = [0usize; 3];
        for row in 0..rows {
            for column in 0..VOCABULARY {
                affine_reference.add(logits[2][row][column], reference_logits[row][column]);
                copy_reference.add(logits[1][row][column], reference_logits[row][column]);
                quantized.add(logits[2][row][column], f64::from(logits[0][row][column]));
                copy_unequal += usize::from(logits[1][row][column].to_bits() != logits[2][row][column].to_bits());
            }
            let reference_top = reference_logits[row]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(index, _)| index);
            top1[0] += usize::from(top(&logits[2][row]) == reference_top);
            top1[1] += usize::from(top(&logits[2][row]) == top(&logits[1][row]));
            top1[2] += usize::from(top(&logits[2][row]) == top(&logits[0][row]));
        }
        readout_checks.push(json!({
            "rows": rows,
            "affine8_vs_fp64_bf16_effective_head": affine_reference.report(),
            "bf16_tensor_kernel_on_copy_vs_fp64_bf16_effective_head": copy_reference.report(),
            "affine8_unequal_to_bf16_tensor_kernel_on_copy": copy_unequal,
            "affine8_vs_original_bf16_head": quantized.report(),
            "top1_agree_fp64_reference": top1[0],
            "top1_agree_bf16_kernel_on_copy": top1[1],
            "top1_agree_original_head": top1[2],
        }));
    }

    // Timing: original BF16 against affine 8-bit, alternating order per pair.
    let mut dense_timing = Vec::new();
    let mut readout_timing = Vec::new();
    for rows in 1..=ROWS {
        let mut dense_pairs = Vec::with_capacity(pairs);
        let mut readout_pairs = Vec::with_capacity(pairs);
        for pair in 0..pairs {
            let order = if pair % 2 == 0 {
                [Arm::Original, Arm::Affine]
            } else {
                [Arm::Affine, Arm::Original]
            };
            let mut dense_ms = [0.0; 2];
            let mut readout_ms = [0.0; 2];
            for arm in order {
                set_arm(&mut session, &mut stash, arm);
                let slot = usize::from(arm == Arm::Affine);
                dense_ms[slot] =
                    execute(&session.runtime, |gpu| encode_dense(gpu, &session, &dense, arm, rows))?.gpu_ms;
                readout_ms[slot] = execute(&session.runtime, |gpu| session.encode_output_rows(gpu, rows))?.gpu_ms;
            }
            dense_pairs.push((dense_ms[0], dense_ms[1]));
            readout_pairs.push((readout_ms[0], readout_ms[1]));
        }
        let mut combined =
            dense_pairs.iter().zip(&readout_pairs).map(|(d, r)| (d.0 + r.0) - (d.1 + r.1)).collect::<Vec<_>>();
        dense_timing.push(json!({"rows": rows, "dense": timing_summary(&dense_pairs)}));
        readout_timing.push(json!({
            "rows": rows,
            "readout": timing_summary(&readout_pairs),
            "dense_plus_readout_saved_ms_range": range(&combined),
            "dense_plus_readout_saved_ms_median_of_pairs": median(&mut combined),
        }));
    }
    // Threadgroup size sweep of the affine kernels (row assignment only).
    let mut sweep = Vec::new();
    set_arm(&mut session, &mut stash, Arm::Affine);
    for threads in [64usize, 128, 256, 512] {
        if let Some(weights) = session.runtime.affine8.as_mut() {
            weights.threads = [threads, threads];
        }
        for rows in 1..=ROWS {
            let mut dense_ms = Vec::new();
            let mut readout_ms = Vec::new();
            for _ in 0..pairs.max(4) / 2 {
                dense_ms.push(
                    execute(&session.runtime, |gpu| encode_dense(gpu, &session, &dense, Arm::Affine, rows))?.gpu_ms,
                );
                readout_ms.push(execute(&session.runtime, |gpu| session.encode_output_rows(gpu, rows))?.gpu_ms);
            }
            sweep.push(json!({
                "threads": threads, "rows": rows,
                "dense_ms_median": median(&mut dense_ms), "readout_ms_median": median(&mut readout_ms),
            }));
        }
    }
    if let Some(weights) = session.runtime.affine8.as_mut() {
        weights.threads = Affine8Weights::new_threads();
    }
    set_arm(&mut session, &mut stash, Arm::Original);

    Ok(json!({
        "affine8_thread_sweep": sweep,
        "prompt_tokens": prompt,
        "context": context,
        "source_text_active": source_active,
        "visits": PHYSICAL.len(),
        "dequantized_values": dequantized[0],
        "gpu_dequantize_unequal_to_load_copy": dequantized[1],
        "embedding_lookups": 64.min(tokens.len()),
        "affine8_embedding_unequal_to_bf16_lookup_on_copy": embedding_unequal,
        "dense_correctness": dense_checks,
        "readout_correctness": readout_checks,
        "dense_timing": dense_timing,
        "readout_timing": readout_timing,
        "affine8_resident_bytes": stash.as_ref().map(Affine8Weights::bytes),
    }))
}

/// Teacher-forced rows scored by the quality comparison: the last `TAIL`
/// prompt tokens (the last of them is the seed) are stepped one at a time,
/// then the three teacher tokens.
const TAIL: usize = 32;

pub(super) fn load_worker_session(
    checkpoint: &Path,
    draft: Option<&Path>,
    context: usize,
) -> Result<Session> {
    let mut session = Session::load(checkpoint, context)?;
    if let Some(draft) = draft {
        session.load_dspark(draft)?;
    }
    session.set_batched_prefill(true, true)?;
    if draft.is_some() {
        session.set_source_verifier(true)?;
        session.prepare_dspark_int8_head()?;
        session.set_dspark_enabled(true)?;
    }
    Ok(session)
}

pub(super) fn ingest(
    session: &mut Session,
    tokens: &[u32],
    prompt: usize,
) -> Result<f64> {
    let started = std::time::Instant::now();
    for range in worker_ranges(prompt, session.runtime.source_text_active, session.prefill_batch_rows()) {
        session.ingest_many(&tokens[range])?;
    }
    Ok(1000.0 * started.elapsed().as_secs_f64())
}

/// Logits of the teacher-forced rows: prefill of the first `prompt - TAIL`
/// tokens on the worker schedule, then one step per remaining prompt token
/// and per teacher token.
fn forced_rows(
    checkpoint: &Path,
    tokens: &[u32],
    prompt: usize,
    context: usize,
) -> Result<(Vec<Vec<f32>>, usize)> {
    let mut session = load_worker_session(checkpoint, None, context)?;
    let resident = session.runtime.device.current_allocated_size();
    let head = prompt - TAIL;
    ingest(&mut session, tokens, head + 1)?;
    let mut rows = Vec::new();
    for &token in &tokens[head..prompt + 3] {
        rows.push(session.step(token)?);
    }
    Ok((rows, resident))
}

/// Per-row relative logit L2 and top-1 agreement of the affine8 artifact
/// against the original artifact (and the pinned capture where present).
pub fn measure_affine8_quality(
    original: &Path,
    affine: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
) -> Result<Value> {
    if prompt <= TAIL + 1 || prompt + 3 > context {
        return Err("Inkling affine8 quality needs a longer prompt");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let (base, base_resident) = forced_rows(original, &tokens, prompt, context)?;
    let (candidate, candidate_resident) = forced_rows(affine, &tokens, prompt, context)?;
    let score = |got: &[f32], want: &[f32]| {
        let (mut delta, mut norm, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
        for (&g, &w) in got.iter().zip(want) {
            let d = f64::from(g) - f64::from(w);
            delta += d * d;
            norm += f64::from(w) * f64::from(w);
            max_abs = max_abs.max(d.abs());
        }
        ((delta / norm).sqrt(), max_abs, top(got) == top(want))
    };
    let mut relative = Vec::new();
    let mut max_abs = 0.0f64;
    let mut agree = 0;
    // Teacher-forced next tokens: the following input token of each row.
    let mut agree_next_base = 0;
    let mut agree_next_candidate = 0;
    for (index, (got, want)) in candidate.iter().zip(&base).enumerate() {
        let (l2, abs, same) = score(got, want);
        relative.push(l2);
        max_abs = max_abs.max(abs);
        agree += usize::from(same);
        if let Some(&next) = tokens.get(prompt - TAIL + index + 1) {
            agree_next_base += usize::from(top(want) == next as usize);
            agree_next_candidate += usize::from(top(got) == next as usize);
        }
    }
    let pinned = oracle.join("row-0.safetensors");
    let mut pinned_report = Value::Null;
    if pinned.exists() && Tensors::open(&pinned)?.values::<u32>("start", "U32", &[1])?[0] as usize + 1 == prompt {
        let mut rows = Vec::new();
        for row in 0..4 {
            let want = Tensors::open(&oracle.join(format!("row-{row}.safetensors")))?.values::<f32>(
                "full.logits",
                "F32",
                &[1, 1, VOCABULARY],
            )?;
            let index = TAIL - 1 + row;
            let (l2_affine, _, top_affine) = score(&candidate[index], &want);
            let (l2_original, _, top_original) = score(&base[index], &want);
            rows.push(json!({
                "row": row,
                "affine8_relative_l2": l2_affine, "affine8_top1_agree": top_affine,
                "original_relative_l2": l2_original, "original_top1_agree": top_original,
            }));
        }
        pinned_report = json!(rows);
    }
    let mut sorted = relative.clone();
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "rows": relative.len(),
        "relative_logit_l2_median": median(&mut sorted),
        "relative_logit_l2_max": relative.iter().copied().fold(0.0, f64::max),
        "relative_logit_l2_per_row": relative,
        "max_abs_logit_difference": max_abs,
        "top1_agree": agree,
        "teacher_next_token_top1_original": agree_next_base,
        "teacher_next_token_top1_affine8": agree_next_candidate,
        "teacher_rows_scored": relative.len() - 1,
        "against_pinned_capture": pinned_report,
        "original_session_metal_bytes_after_load": base_resident,
        "affine8_session_metal_bytes_after_load": candidate_resident,
    }))
}

/// Paired step, verify-round and prefill timings of the two artifacts loaded
/// side by side, alternating order, from the same restored prompt state.
pub fn measure_affine8_timing(
    original: &Path,
    affine: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    pairs: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut sessions =
        [load_worker_session(original, Some(draft), context)?, load_worker_session(affine, Some(draft), context)?];
    let mut prefill = vec![Vec::new(), Vec::new()];
    let mut bases = Vec::new();
    let mut anchors = Vec::new();
    for (index, session) in sessions.iter_mut().enumerate() {
        prefill[index].push(ingest(session, &tokens, prompt)?);
        let logits = session.step(tokens[prompt - 1])?;
        anchors.push(transaction_validation::greedy(&logits));
        bases.push(session.snapshot()?);
    }
    // Prefill pairs: reset and ingest the whole worker schedule again.
    for pair in 0..pairs.min(6) {
        let order = if pair % 2 == 0 {
            [0, 1]
        } else {
            [1, 0]
        };
        for index in order {
            sessions[index].reset();
            prefill[index].push(ingest(&mut sessions[index], &tokens, prompt)?);
        }
    }
    let mut steps = Vec::new();
    let mut rounds = vec![Vec::new(); ROWS - 1];
    for pair in 0..pairs {
        let order = if pair % 2 == 0 {
            [0, 1]
        } else {
            [1, 0]
        };
        let mut step = [0.0; 2];
        let mut round = [[0.0; 2]; ROWS - 1];
        for index in order {
            let session = &mut sessions[index];
            session.restore(&bases[index])?;
            let started = std::time::Instant::now();
            session.step(anchors[index])?;
            step[index] = 1000.0 * started.elapsed().as_secs_f64();
            for rows in 2..=ROWS {
                session.restore(&bases[index])?;
                let inputs = std::iter::once(anchors[index])
                    .chain(tokens[prompt..prompt + rows - 1].iter().copied())
                    .collect::<Vec<_>>();
                let started = std::time::Instant::now();
                let txn = session.begin_verify(&inputs)?;
                txn.logits(rows - 1)?;
                round[rows - 2][index] = 1000.0 * started.elapsed().as_secs_f64();
                drop(txn);
            }
        }
        steps.push((step[0], step[1]));
        for rows in 0..ROWS - 1 {
            rounds[rows].push((round[rows][0], round[rows][1]));
        }
    }
    let prefill_pairs = prefill[0].iter().copied().zip(prefill[1].iter().copied()).collect::<Vec<_>>();
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "source_text_active": [sessions[0].runtime.source_text_active, sessions[1].runtime.source_text_active],
        "note": "wall milliseconds per command; base = original artifact, candidate = affine8 artifact",
        "prefill_ms": timing_summary(&prefill_pairs),
        "step_ms": timing_summary(&steps),
        "verify_round_ms": rounds.iter().enumerate().map(|(i, pairs)| json!({"rows": i + 2, "timing": timing_summary(pairs)})).collect::<Vec<_>>(),
    }))
}

/// One side of a quality comparison: an artifact target and whether prefill
/// runs the bit-exact reference kernels instead of the default ones.
pub struct QualitySide<'a> {
    pub checkpoint: &'a Path,
    pub exact_prefill: bool,
    /// A decode change switched off on this side (see `--paired-control`).
    pub control_off: Option<&'a str>,
}

/// Log-softmax of one logit row in FP64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total = logits.iter().map(|&x| (f64::from(x) - peak).exp()).sum::<f64>().ln();
    logits.iter().map(|&x| f64::from(x) - peak - total).collect()
}

/// Teacher-forced rows: the worker schedule over the first `prompt - window`
/// tokens, then one step per remaining prompt token and the first two teacher
/// tokens. Every row has the following token of the text as its target.
fn teacher_rows(
    side: &QualitySide<'_>,
    tokens: &[u32],
    prompt: usize,
    context: usize,
    window: usize,
    mut each: impl FnMut(usize, &[f32]),
) -> Result<()> {
    let mut session = load_worker_session(side.checkpoint, None, context)?;
    session.runtime.exact_prefill = side.exact_prefill;
    if let Some(control) = side.control_off {
        super::decode_stages::set_control(&mut session, control, false)?;
    }
    let head = prompt - window;
    ingest(&mut session, tokens, head + 1)?;
    for (row, &token) in tokens[head..prompt + 2].iter().enumerate() {
        let logits = session.step(token)?;
        each(row, &logits);
    }
    Ok(())
}

/// Natural-text quality comparison of two sides on teacher-forced rows.
/// Aggregates only: accuracy against the text, top-1 agreement (overall and
/// on confident reference rows), KL divergence from the reference and the
/// change in log-probability of the actual next token.
pub fn measure_quality_compare(
    reference: QualitySide<'_>,
    candidate: QualitySide<'_>,
    oracle: &Path,
    prompt: usize,
    context: usize,
    window: usize,
) -> Result<Value> {
    if window == 0 || prompt <= window + 1 || prompt + 3 > context {
        return Err("Inkling quality comparison needs a prompt longer than the window");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let head = prompt - window;
    let target = |row: usize| tokens[head + row + 1] as usize;
    // Reference: keep the log-distributions in memory only.
    let mut reference_rows: Vec<(Vec<f32>, usize)> = Vec::new();
    teacher_rows(&reference, &tokens, prompt, context, window, |_, logits| {
        reference_rows.push((logits.to_vec(), top(logits)));
    })?;
    let rows = reference_rows.len();
    let (mut correct_ref, mut correct_cand, mut agree) = (0usize, 0usize, 0usize);
    let mut confident = [(0usize, 0usize); 2];
    let (mut kl_sum, mut kl_max, mut logp_delta) = (0.0f64, 0.0f64, 0.0f64);
    let mut failure = None;
    teacher_rows(&candidate, &tokens, prompt, context, window, |row, logits| {
        let Some((want, want_top)) = reference_rows.get(row) else {
            failure = Some("Inkling quality comparison row count differs");
            return;
        };
        let got_top = top(logits);
        let p = log_softmax(want);
        let q = log_softmax(logits);
        let next = target(row);
        correct_ref += usize::from(*want_top == next);
        correct_cand += usize::from(got_top == next);
        let same = usize::from(got_top == *want_top);
        agree += same;
        let top_probability = p[*want_top].exp();
        for (slot, threshold) in [0.5, 0.9].into_iter().enumerate() {
            if top_probability >= threshold {
                confident[slot].0 += 1;
                confident[slot].1 += same;
            }
        }
        let kl = p.iter().zip(&q).map(|(&lp, &lq)| lp.exp() * (lp - lq)).sum::<f64>().max(0.0);
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        logp_delta += (p[next] - q[next]).abs();
    })?;
    if let Some(error) = failure {
        return Err(error);
    }
    let side = |s: &QualitySide<'_>| {
        json!({
            "target": s.checkpoint.file_name().and_then(|n| n.to_str()),
            "affine8": s.checkpoint.to_str().is_some_and(|p| p.contains("affine8")),
            "exact_prefill": s.exact_prefill,
            "control_off": s.control_off,
        })
    };
    let rate = |hit: usize, total: usize| {
        if total == 0 {
            f64::NAN
        } else {
            hit as f64 / total as f64
        }
    };
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "window": window,
        "rows": rows,
        "reference": side(&reference),
        "candidate": side(&candidate),
        "next_token_accuracy_reference": rate(correct_ref, rows),
        "next_token_accuracy_candidate": rate(correct_cand, rows),
        "top1_agreement": rate(agree, rows),
        "confident_rows_p50": confident[0].0,
        "top1_agreement_p50": rate(confident[0].1, confident[0].0),
        "confident_rows_p90": confident[1].0,
        "top1_agreement_p90": rate(confident[1].1, confident[1].0),
        "kl_reference_to_candidate_mean": kl_sum / rows.max(1) as f64,
        "kl_reference_to_candidate_max": kl_max,
        "next_token_logprob_abs_change_mean": logp_delta / rows.max(1) as f64,
    }))
}
