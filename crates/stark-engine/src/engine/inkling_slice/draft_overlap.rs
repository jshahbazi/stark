// SPDX-License-Identifier: Apache-2.0
//! Aggregate acceptance estimates from target-sampled teacher trajectories.

use serde_json::{Value, json};

use super::{super::dspark::DraftBlock, *};

const DEPTH: usize = 3;
const ANCHORS: usize = 96;
const TEMPERATURES: [f32; 4] = [0.5, 0.75, 1.0, 1.25];

fn uniform(state: &mut u64) -> f64 {
    // Match the ordinary Inkling stochastic worker's SplitMix64 draws.
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn weights(
    scores: &[f32],
    temperature: f32,
) -> Result<Vec<f64>> {
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("Inkling draft overlap had a nonfinite score");
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(scores.iter().map(|&score| (((score - max) / temperature) as f64).exp()).collect())
}

fn draw(
    scores: &[f32],
    weights: &[f64],
    uniform: f64,
) -> usize {
    let mut ids = (0..scores.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    let target = uniform * ids.iter().map(|&id| weights[id]).sum::<f64>();
    let mut cumulative = 0.0;
    for &id in &ids {
        cumulative += weights[id];
        if target < cumulative {
            return id;
        }
    }
    *ids.last().unwrap()
}

fn greedy(scores: &[f32]) -> usize {
    scores.iter().enumerate().fold(0, |best, (id, &score)| {
        if score > scores[best] {
            id
        } else {
            best
        }
    })
}

struct TeacherRow {
    token: u32,
    uniform: f64,
    p: Vec<f64>,
}

#[derive(Default)]
struct Estimate {
    marginal: [f64; DEPTH],
    joint: [f64; DEPTH],
    common_uniform: [usize; DEPTH],
}

pub fn measure_draft_overlap(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    mut seed: u64,
) -> Result<Value> {
    const PREFIX: usize = 4096;
    let fixture = Tensors::open(&oracle.join("tokens.safetensors"))?;
    let prompt = fixture.values::<u32>("tokens", "U32", &[PREFIX + 3])?;
    let mut session = Session::load(checkpoint, 4352)?;
    session.load_dspark(draft)?;
    session.set_batched_prefill(true, true)?;
    session.set_source_verifier(true)?;
    session.set_dspark_enabled(true)?;
    let block = DraftBlock::new(&mut session.runtime)?;
    packed_prefill::register(
        &mut session.runtime,
        concat!("#define INK_DRAFT_EXPORT_LOGITS\n", include_str!("dspark.metal")),
        &["ink_draft_export_scores"],
    )?;
    let scores = zeroed(&session.runtime.device, DEPTH * PADDED_VOCABULARY * size_of::<f32>())?;
    for (start, end) in [(0, 2048), (2048, 4092), (4092, 4095)] {
        session.ingest_many(&prompt[start..end])?;
    }
    let mut logits = session.step(prompt[4095])?;
    let reference = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let expected = reference.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
    if !logits.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()) {
        return Err("Inkling draft overlap prefill seed differed from pinned logits");
    }
    let base = session.snapshot()?;
    let mut teacher = Vec::new();
    for _ in 0..ANCHORS + DEPTH {
        let u = uniform(&mut seed);
        let mut p = weights(&logits, 1.0)?;
        let token = draw(&logits, &p, u) as u32;
        let mass = p.iter().sum::<f64>();
        p.iter_mut().for_each(|probability| *probability /= mass);
        teacher.push(TeacherRow {
            token,
            uniform: u,
            p,
        });
        logits = session.step(token)?;
    }
    session.restore(&base)?;

    let mut estimates = TEMPERATURES.map(|_| Estimate::default());
    let mut greedy_matches = [0; DEPTH];
    let mut controls = 0;
    for i in 0..ANCHORS {
        let anchor = teacher[i].token;
        let next = &teacher[i + 1..=i + DEPTH];
        let actual_previous = [next[0].token, next[1].token];
        let export = |previous: &[u32]| -> Result<Vec<f32>> {
            session.active_dspark().ok_or("Inkling draft overlap needs DSpark")?.teacher_proposal_logits(
                &session.runtime,
                anchor,
                session.position,
                &session.embedding,
                &session.embed_norm,
                &session.unembedding,
                previous,
                &block,
                &scores,
            )?;
            Ok(read_shared(&scores, DEPTH * PADDED_VOCABULARY))
        };
        let rows = |buffer: &[f32]| {
            buffer.chunks(PADDED_VOCABULARY).map(|row| &row[..VOCABULARY]).map(greedy).collect::<Vec<_>>()
        };
        if i % 16 == 0 {
            let proposed = session.propose_verify(anchor, DEPTH)?;
            let control = export(&proposed[..DEPTH - 1])?;
            if rows(&control).into_iter().ne(proposed.into_iter().map(|id| id as usize)) {
                return Err("Inkling draft overlap score export disagreed with the production draft");
            }
            controls += DEPTH;
        }
        let exported = export(&actual_previous)?;
        let mut greedy_prefix = true;
        let mut common_uniform = [true; TEMPERATURES.len()];
        let mut ratio = [1.0_f64; TEMPERATURES.len()];
        for (row, target) in next.iter().enumerate() {
            let start = row * PADDED_VOCABULARY;
            let q_scores = &exported[start..start + VOCABULARY];
            let token = target.token as usize;
            greedy_prefix &= greedy(q_scores) == token;
            greedy_matches[row] += usize::from(greedy_prefix);
            for (j, (&temperature, estimate)) in TEMPERATURES.iter().zip(&mut estimates).enumerate() {
                let q = weights(q_scores, temperature)?;
                let q_mass = q.iter().sum::<f64>();
                estimate.marginal[row] += target.p.iter().zip(&q).map(|(&p, q)| p.min(q / q_mass)).sum::<f64>();
                ratio[j] *= (q[token] / q_mass / target.p[token]).min(1.0);
                estimate.joint[row] += ratio[j];
                common_uniform[j] &= draw(q_scores, &q, target.uniform) == token;
                estimate.common_uniform[row] += usize::from(common_uniform[j]);
            }
        }
        session.ingest_many(&[anchor])?;
    }

    let reports = TEMPERATURES
        .iter()
        .zip(estimates)
        .map(|(temperature, estimate)| {
            json!({
                "draft_temperature":temperature,
                "marginal_overlap":estimate.marginal.map(|v| v / ANCHORS as f64),
                "joint_rejection_estimate":estimate.joint.map(|v| v / ANCHORS as f64),
                "common_uniform_accept_counts":estimate.common_uniform,
                "expected_useful_per_round":1.0 + estimate.joint.iter().sum::<f64>() / ANCHORS as f64,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "prefix":PREFIX,"anchors":ANCHORS,"mask_depth":DEPTH,"target_temperature":1.0,
        "pinned_seed_match":true,"production_draft_argmax_controls":controls,
        "ordinary_greedy_accept_counts":greedy_matches,"estimates":reports,
    }))
}
