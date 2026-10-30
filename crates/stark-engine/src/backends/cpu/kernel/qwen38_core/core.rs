//! Independent scalar controls for Qwen3.8 Uzu architecture kernels.
// Recurrence/HC equations follow Lily e22d58fa7cc488493fb1b5d810a198bd29730aeb
// Apache-2.0; see root LICENSE.qwen38-reference, NOTICE.qwen38-reference and
// docs/QWEN38_UZU_PROVENANCE.md.
// Copyright 2026 Perplexity AI, Inc. Copyright 2026 Fabio Greter.
// Gated delta equations also derive from MLX-LM (MIT):
// Copyright (c) 2023-2025 Apple Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
use half::bf16;
use stark_engine_macros::kernel;

#[kernel(Qwen38HcNorm)]
pub fn hc_norm(
    input: *const bf16,
    weight: *const f32,
    output: *mut bf16,
    width: u32,
    streams: u32,
    rows: u32,
    epsilon: f32,
) {
    unsafe {
        for segment in 0..(rows * streams) as usize {
            let base = segment * width as usize;
            let weight_base = segment % streams as usize * width as usize;
            let sum: f32 = (0..width as usize)
                .map(|i| {
                    let x = (*input.add(base + i)).to_f32();
                    x * x
                })
                .sum();
            let inverse = (sum / width as f32 + epsilon).sqrt().recip();
            for i in 0..width as usize {
                *output.add(base + i) =
                    bf16::from_f32((*input.add(base + i)).to_f32() * inverse * (1.0 + *weight.add(weight_base + i)));
            }
        }
    }
}

#[kernel(Qwen38SiluScaled)]
pub fn silu_scaled(
    input: *const bf16,
    output: *mut bf16,
    count: u32,
    scale: f32,
) {
    unsafe {
        for i in 0..count as usize {
            let x = (*input.add(i)).to_f32() * scale;
            *output.add(i) = bf16::from_f32(x / (1.0 + (-x).exp()));
        }
    }
}

#[kernel(Qwen38HcMix)]
pub fn hc_mix(
    normalized: *const bf16,
    logits: *const bf16,
    output: *mut bf16,
    width: u32,
    streams: u32,
    rows: u32,
) {
    unsafe {
        for i in 0..(rows * width) as usize {
            let row = i / width as usize;
            let column = i % width as usize;
            let mut sum = 0.0;
            for stream in 0..streams as usize {
                let index = (row * streams as usize + stream) * width as usize + column;
                let gate = 1.0 / (1.0 + (-(*logits.add(index)).to_f32()).exp());
                sum += gate * (*normalized.add(index)).to_f32();
            }
            *output.add(i) = bf16::from_f32(sum / streams as f32);
        }
    }
}

#[kernel(Qwen38HcInject)]
pub fn hc_inject(
    input: *const bf16,
    branch: *const bf16,
    logits: *const bf16,
    output: *mut bf16,
    width: u32,
    streams: u32,
    rows: u32,
) {
    unsafe {
        for i in 0..(rows * width * streams) as usize {
            let row = i / (width * streams) as usize;
            let logit = (*logits.add(i / width as usize)).to_f32() / streams as f32;
            let gate = 2.0 / (1.0 + (-logit).exp());
            *output.add(i) = bf16::from_f32(
                (*input.add(i)).to_f32() + (*branch.add(row * width as usize + i % width as usize)).to_f32() * gate,
            );
        }
    }
}

#[kernel(Qwen38Broadcast)]
pub fn broadcast(
    input: *const bf16,
    output: *mut bf16,
    width: u32,
    streams: u32,
    rows: u32,
) {
    unsafe {
        for i in 0..(rows * width * streams) as usize {
            *output.add(i) = *input.add(i / (width * streams) as usize * width as usize + i % width as usize);
        }
    }
}

#[kernel(Qwen38SiluMultiply)]
pub fn silu_multiply(
    gate: *const bf16,
    up: *const bf16,
    output: *mut bf16,
    count: u32,
) {
    unsafe {
        for i in 0..count as usize {
            let x = (*gate.add(i)).to_f32();
            *output.add(i) = bf16::from_f32((x / (1.0 + (-x).exp())) * (*up.add(i)).to_f32());
        }
    }
}

#[kernel(Qwen38GdnConv)]
pub fn gdn_conv(
    input: *const bf16,
    weight: *const f32,
    window: *mut bf16,
    output: *mut bf16,
    channels: u32,
    taps: u32,
) {
    assert!(taps >= 2);
    unsafe {
        for channel in 0..channels as usize {
            let taps = taps as usize;
            let base = channel * (taps - 1);
            let mut sum = 0.0;
            for tap in 0..taps - 1 {
                sum += (*window.add(base + tap)).to_f32() * *weight.add(channel * taps + tap);
            }
            sum += (*input.add(channel)).to_f32() * *weight.add(channel * taps + taps - 1);
            for tap in 0..taps - 2 {
                *window.add(base + tap) = *window.add(base + tap + 1);
            }
            *window.add(base + taps - 2) = *input.add(channel);
            *output.add(channel) = bf16::from_f32(sum / (1.0 + (-sum).exp()));
        }
    }
}

#[kernel(Qwen38GdnStep)]
pub fn gdn_step(
    qkv: *const bf16,
    z: *const bf16,
    a: *const bf16,
    b: *const bf16,
    a_log: *const f32,
    dt_bias: *const f32,
    norm: *const f32,
    state: *mut f32,
    output: *mut bf16,
    key_heads: u32,
    value_heads: u32,
    epsilon: f32,
    sigmoid_gate: u32,
) {
    assert!(key_heads > 0 && value_heads.is_multiple_of(key_heads));
    unsafe {
        for head in 0..value_heads as usize {
            let key_head = head / (value_heads / key_heads) as usize;
            let key_dim = key_heads as usize * 128;
            let mut q = [0.0f32; 128];
            let mut k = [0.0f32; 128];
            for i in 0..128 {
                q[i] = (*qkv.add(key_head * 128 + i)).to_f32();
                k[i] = (*qkv.add(key_dim + key_head * 128 + i)).to_f32();
            }
            let q_inverse = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12).recip();
            let k_inverse = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12).recip();
            for i in 0..128 {
                q[i] = (q[i] * q_inverse) * 0.08838834764831845f32;
                k[i] *= k_inverse;
            }
            let value = (*a.add(head)).to_f32() + *dt_bias.add(head);
            let softplus = if value > 20.0 {
                value
            } else {
                (1.0 + value.exp()).ln()
            };
            let decay = (-(*a_log.add(head)).exp() * softplus).exp();
            let beta = 1.0 / (1.0 + (-(*b.add(head)).to_f32()).exp());
            let mut raw = [0.0f32; 128];
            for column in 0..128 {
                let base = head * 128 * 128 + column;
                let mut prediction = 0.0;
                for row in 0..128 {
                    prediction += k[row] * *state.add(base + row * 128) * decay;
                }
                let delta = ((*qkv.add(2 * key_dim + head * 128 + column)).to_f32() - prediction) * beta;
                let mut value = 0.0;
                for row in 0..128 {
                    let updated = *state.add(base + row * 128) * decay + k[row] * delta;
                    *state.add(base + row * 128) = updated;
                    value += q[row] * updated;
                }
                raw[column] = bf16::from_f32(value).to_f32();
            }
            let inverse = (raw.iter().map(|x| x * x).sum::<f32>() / 128.0 + epsilon).sqrt().recip();
            for i in 0..128 {
                let value = (*z.add(head * 128 + i)).to_f32();
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                let activation = if sigmoid_gate != 0 {
                    sigmoid
                } else {
                    value * sigmoid
                };
                *output.add(head * 128 + i) = bf16::from_f32(*norm.add(i) * raw[i] * inverse * activation);
            }
        }
    }
}

#[kernel(Qwen38MoeRoute)]
pub fn moe_route(
    logits: *const f32,
    indices: *mut u32,
    scores: *mut f32,
    experts: u32,
    selected: u32,
    normalize: u32,
    rows: u32,
) {
    assert!(selected > 0 && selected <= experts);
    unsafe {
        for row in 0..rows as usize {
            let logits = logits.add(row * experts as usize);
            let indices = indices.add(row * selected as usize);
            let scores = scores.add(row * selected as usize);
            for slot in 0..selected as usize {
                let mut best = f32::NEG_INFINITY;
                let mut best_id = 0;
                for expert in 0..experts {
                    let used = (0..slot).any(|prior| *indices.add(prior) == expert);
                    if !used && *logits.add(expert as usize) > best {
                        best = *logits.add(expert as usize);
                        best_id = expert;
                    }
                }
                *indices.add(slot) = best_id;
            }
            let maximum = *logits.add(*indices as usize);
            let total = if normalize != 0 {
                (0..selected as usize)
                    .map(|slot| (*logits.add(*indices.add(slot) as usize) - maximum).exp())
                    .sum::<f32>()
            } else {
                (0..experts as usize).map(|expert| (*logits.add(expert) - maximum).exp()).sum::<f32>()
            };
            for slot in 0..selected as usize {
                *scores.add(slot) = (*logits.add(*indices.add(slot) as usize) - maximum).exp() / total;
            }
        }
    }
}

#[kernel(Qwen38MoeIndices)]
pub fn moe_indices(
    selected_experts: *const u32,
    gate_indices: *mut u32,
    down_indices: *mut u32,
    intermediate: u32,
    hidden: u32,
    selected: u32,
    rows: u32,
    index_count: u32,
) {
    assert!(index_count >= rows * selected * intermediate.max(hidden));
    unsafe {
        for i in 0..(rows * selected * intermediate) as usize {
            *gate_indices.add(i) =
                *selected_experts.add(i / intermediate as usize) * intermediate + i as u32 % intermediate;
        }
        for i in 0..(rows * selected * hidden) as usize {
            *down_indices.add(i) = *selected_experts.add(i / hidden as usize) * hidden + i as u32 % hidden;
        }
    }
}

#[kernel(Qwen38MoeCombine)]
pub fn moe_combine(
    experts: *const bf16,
    scores: *const f32,
    shared: *const bf16,
    shared_gate: *const f32,
    output: *mut bf16,
    hidden: u32,
    selected: u32,
    rows: u32,
) {
    unsafe {
        for i in 0..(rows * hidden) as usize {
            let row = i / hidden as usize;
            let column = i % hidden as usize;
            let mut sum = 0.0;
            for slot in 0..selected as usize {
                sum += *scores.add(row * selected as usize + slot)
                    * (*experts.add((row * selected as usize + slot) * hidden as usize + column)).to_f32();
            }
            let gate = 1.0 / (1.0 + (-*shared_gate.add(row)).exp());
            sum += gate * (*shared.add(i)).to_f32();
            *output.add(i) = bf16::from_f32(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[stark_engine_macros::stark_test]
    fn batched_hyper_connections_repeat_weights_and_keep_token_rows_separate() {
        let (rows, streams, width) = (17_usize, 3_usize, 5_usize);
        let input = (0..rows * streams * width).map(|i| bf16::from_f32((i as f32 - 80.0) / 97.0)).collect::<Vec<_>>();
        let weights = (0..streams * width).map(|i| i as f32 / 31.0 - 0.2).collect::<Vec<_>>();
        let logits = (0..input.len()).map(|i| bf16::from_f32((i % 23) as f32 / 13.0 - 0.5)).collect::<Vec<_>>();
        let injection = (0..rows * streams).map(|i| bf16::from_f32(i as f32 / 37.0 - 0.4)).collect::<Vec<_>>();
        let mut normalized = vec![bf16::ZERO; input.len()];
        let mut mixed = vec![bf16::ZERO; rows * width];
        let mut injected = vec![bf16::ZERO; input.len()];
        let mut broadcasted = vec![bf16::ZERO; input.len()];
        hc_norm(
            input.as_ptr(),
            weights.as_ptr(),
            normalized.as_mut_ptr(),
            width as u32,
            streams as u32,
            rows as u32,
            1e-6,
        );
        hc_mix(normalized.as_ptr(), logits.as_ptr(), mixed.as_mut_ptr(), width as u32, streams as u32, rows as u32);
        hc_inject(
            input.as_ptr(),
            mixed.as_ptr(),
            injection.as_ptr(),
            injected.as_mut_ptr(),
            width as u32,
            streams as u32,
            rows as u32,
        );
        broadcast(mixed.as_ptr(), broadcasted.as_mut_ptr(), width as u32, streams as u32, rows as u32);
        for row in 0..rows {
            let base = row * streams * width;
            let mut norm = vec![bf16::ZERO; streams * width];
            let mut mix = vec![bf16::ZERO; width];
            let mut inject = vec![bf16::ZERO; streams * width];
            let mut repeated = vec![bf16::ZERO; streams * width];
            hc_norm(input[base..].as_ptr(), weights.as_ptr(), norm.as_mut_ptr(), width as u32, streams as u32, 1, 1e-6);
            hc_mix(norm.as_ptr(), logits[base..].as_ptr(), mix.as_mut_ptr(), width as u32, streams as u32, 1);
            hc_inject(
                input[base..].as_ptr(),
                mix.as_ptr(),
                injection[row * streams..].as_ptr(),
                inject.as_mut_ptr(),
                width as u32,
                streams as u32,
                1,
            );
            broadcast(mix.as_ptr(), repeated.as_mut_ptr(), width as u32, streams as u32, 1);
            for (actual, expected) in [
                (&normalized[base..base + streams * width], norm.as_slice()),
                (&mixed[row * width..(row + 1) * width], mix.as_slice()),
                (&injected[base..base + streams * width], inject.as_slice()),
                (&broadcasted[base..base + streams * width], repeated.as_slice()),
            ] {
                assert!(
                    actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "batched HC row indexing changed arithmetic"
                );
            }
        }
    }

    #[stark_engine_macros::stark_test]
    fn batched_routes_indices_and_combine_preserve_global_expert_rows() {
        let (rows, experts, selected, hidden, intermediate) = (17_usize, 7_usize, 3_usize, 5_usize, 4_usize);
        let logits = (0..rows * experts).map(|i| ((i * 13 + i / experts) % 17) as f32 / 8.0).collect::<Vec<_>>();
        let mut ids = vec![0; rows * selected];
        let mut scores = vec![0.0; rows * selected];
        let mut gates = vec![0; rows * selected * intermediate];
        let mut down = vec![0; rows * selected * hidden];
        moe_route(
            logits.as_ptr(),
            ids.as_mut_ptr(),
            scores.as_mut_ptr(),
            experts as u32,
            selected as u32,
            1,
            rows as u32,
        );
        moe_indices(
            ids.as_ptr(),
            gates.as_mut_ptr(),
            down.as_mut_ptr(),
            intermediate as u32,
            hidden as u32,
            selected as u32,
            rows as u32,
            (rows * selected * hidden.max(intermediate)) as u32,
        );
        let values = (0..rows * selected * hidden).map(|i| bf16::from_f32((i % 31) as f32 / 19.0)).collect::<Vec<_>>();
        let shared = (0..rows * hidden).map(|i| bf16::from_f32(i as f32 / 101.0)).collect::<Vec<_>>();
        let shared_gate = (0..rows).map(|i| i as f32 / 11.0 - 1.0).collect::<Vec<_>>();
        let mut output = vec![bf16::ZERO; rows * hidden];
        moe_combine(
            values.as_ptr(),
            scores.as_ptr(),
            shared.as_ptr(),
            shared_gate.as_ptr(),
            output.as_mut_ptr(),
            hidden as u32,
            selected as u32,
            rows as u32,
        );
        for row in 0..rows {
            let mut row_ids = vec![0; selected];
            let mut row_scores = vec![0.0; selected];
            let mut row_output = vec![bf16::ZERO; hidden];
            moe_route(
                logits[row * experts..].as_ptr(),
                row_ids.as_mut_ptr(),
                row_scores.as_mut_ptr(),
                experts as u32,
                selected as u32,
                1,
                1,
            );
            assert!(ids[row * selected..(row + 1) * selected] == row_ids, "router crossed token rows");
            assert!(
                scores[row * selected..(row + 1) * selected]
                    .iter()
                    .zip(&row_scores)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "router normalization order changed"
            );
            for (slot, &expert) in row_ids.iter().enumerate() {
                for column in 0..intermediate {
                    assert!(
                        gates[(row * selected + slot) * intermediate + column]
                            == expert * intermediate as u32 + column as u32,
                        "gate bank row lost its global expert ID"
                    );
                }
                for column in 0..hidden {
                    assert!(
                        down[(row * selected + slot) * hidden + column] == expert * hidden as u32 + column as u32,
                        "down bank row lost its global expert ID"
                    );
                }
            }
            moe_combine(
                values[row * selected * hidden..].as_ptr(),
                row_scores.as_ptr(),
                shared[row * hidden..].as_ptr(),
                shared_gate[row..].as_ptr(),
                row_output.as_mut_ptr(),
                hidden as u32,
                selected as u32,
                1,
            );
            assert!(
                output[row * hidden..(row + 1) * hidden]
                    .iter()
                    .zip(row_output)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "expert combine crossed token rows"
            );
        }
    }

    #[stark_engine_macros::stark_test]
    fn router_ties_are_stable_and_normalization_covers_selected_or_all_experts() {
        let logits = [3.0, 3.0, 1.0, -2.0];
        let mut indices = [u32::MAX; 2];
        let mut scores = [0.0; 2];
        moe_route(logits.as_ptr(), indices.as_mut_ptr(), scores.as_mut_ptr(), 4, 2, 1, 1);
        assert_eq!(indices, [0, 1]);
        assert_eq!(scores, [0.5, 0.5]);
        moe_route(logits.as_ptr(), indices.as_mut_ptr(), scores.as_mut_ptr(), 4, 2, 0, 1);
        let expected = 1.0_f64 / (2.0 + (-2.0_f64).exp() + (-5.0_f64).exp());
        assert!(scores.iter().all(|&score| (f64::from(score) - expected).abs() < 1e-7));
        let selected = [7, 2];
        let mut gates = [0; 6];
        let mut down = [0; 8];
        moe_indices(selected.as_ptr(), gates.as_mut_ptr(), down.as_mut_ptr(), 3, 4, 2, 1, 8);
        assert_eq!(gates, [21, 22, 23, 6, 7, 8]);
        assert_eq!(down, [28, 29, 30, 31, 8, 9, 10, 11]);
    }

    #[stark_engine_macros::stark_test]
    fn convolution_preserves_channel_major_taps_and_causal_history() {
        let weights = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
        let mut history = [bf16::ZERO; 6];
        let mut out = [bf16::ZERO; 2];
        let mut host_history = [[0.0_f32; 3]; 2];
        for input in [[0.25, -0.5], [1.0, 0.25], [-0.75, 0.5], [0.0, -1.0]] {
            let input = input.map(bf16::from_f32);
            gdn_conv(input.as_ptr(), weights.as_ptr(), history.as_mut_ptr(), out.as_mut_ptr(), 2, 4);
            for channel in 0..2 {
                let sum = (0..3).map(|tap| host_history[channel][tap] * weights[channel * 4 + tap]).sum::<f32>()
                    + input[channel].to_f32() * weights[channel * 4 + 3];
                let expected = bf16::from_f32(sum / (1.0 + (-sum).exp()));
                assert_eq!(out[channel].to_bits(), expected.to_bits());
                host_history[channel].rotate_left(1);
                host_history[channel][2] = input[channel].to_f32();
                for tap in 0..3 {
                    assert_eq!(history[channel * 3 + tap].to_f32(), host_history[channel][tap]);
                }
            }
        }
    }

    #[stark_engine_macros::stark_test]
    fn recurrent_delta_uses_decayed_state_and_shared_key_head_across_value_heads() {
        let mut qkv = vec![bf16::ZERO; 512];
        qkv[0] = bf16::ONE;
        qkv[128] = bf16::ONE;
        qkv[256..384].fill(bf16::from_f32(0.5));
        qkv[384..].fill(bf16::from_f32(0.25));
        let z = [bf16::ZERO; 256];
        let controls = [bf16::ZERO; 2];
        let alog = [0.0; 2];
        let bias = [0.0; 2];
        let norm = [1.0; 128];
        let mut state = vec![0.0; 2 * 128 * 128];
        let mut output = [bf16::ZERO; 256];
        for expected in [[0.25, 0.125], [0.3125, 0.15625]] {
            gdn_step(
                qkv.as_ptr(),
                z.as_ptr(),
                controls.as_ptr(),
                controls.as_ptr(),
                alog.as_ptr(),
                bias.as_ptr(),
                norm.as_ptr(),
                state.as_mut_ptr(),
                output.as_mut_ptr(),
                1,
                2,
                1e-6,
                1,
            );
            for head in 0..2 {
                assert!(
                    state[head * 16384..head * 16384 + 128].iter().all(|&value| (value - expected[head]).abs() < 1e-7)
                );
                assert!(state[head * 16384 + 128..(head + 1) * 16384].iter().all(|&value| value == 0.0));
            }
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }
}
