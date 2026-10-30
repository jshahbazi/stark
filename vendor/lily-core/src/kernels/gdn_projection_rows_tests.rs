// SPDX-License-Identifier: Apache-2.0
//! Synthetic exact controls for the private short-batch GDN input fusion.
use super::*;
use crate::kernels::gdn::conv_window_rollback;
use crate::sha256::{Sha256, hex};

fn word(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn values(seed: &mut u32, count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|_| (word(seed) % 4097) as f32 * scale / 4096.0 - scale / 2.0)
        .collect()
}

struct Guarded {
    storage: Tensor,
    value: Tensor,
}

impl Guarded {
    fn new(ctx: &MetalContext, shape: &[usize], data: &[f32]) -> Result<Self> {
        let count: usize = shape.iter().product();
        ensure!(count == data.len(), "guarded fixture geometry");
        let storage = Tensor::zeros(ctx, &[count + 4], DType::BF16)?;
        storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
        let value = storage.view(2, shape)?;
        let bf: Vec<_> = data.iter().copied().map(half::bf16::from_f32).collect();
        value.write_bytes(bytemuck::cast_slice(&bf))?;
        Ok(Self { storage, value })
    }

    fn zero(ctx: &MetalContext, shape: &[usize]) -> Result<Self> {
        Self::new(ctx, shape, &vec![0.0; shape.iter().product()])
    }

    fn canaries(&self) -> Result<()> {
        let bytes = self.storage.raw_bytes();
        ensure!(
            bytes[..4].iter().chain(&bytes[bytes.len() - 4..]).all(|&b| b == 0xa5),
            "GDN projection fusion wrote outside its view"
        );
        Ok(())
    }
}

fn weights(
    ctx: &MetalContext,
    seed: &mut u32,
    n: usize,
    k: usize,
) -> Result<QuantWeights> {
    let allocated = n + 2;
    let codes: Vec<_> = (0..allocated * k / 8).map(|_| word(seed)).collect();
    let scales = values(seed, allocated * k / 64, 0.01);
    let biases = values(seed, allocated * k / 64, 0.125);
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[allocated, k / 8],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[allocated, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[allocated, k / 64])?,
        group_size: 64,
        bits: 4,
    }
    .view_rows(2, n)
}

fn compare(
    expected: &[u8],
    actual: &[u8],
    field: &str,
    reference_hash: &mut Sha256,
    candidate_hash: &mut Sha256,
    count: &mut usize,
) -> Result<()> {
    ensure!(expected == actual, "GDN projection rows mismatch: {field}");
    reference_hash.update(expected);
    candidate_hash.update(actual);
    *count += expected.len();
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn fused_gdn_projection_rows_match_decode_and_all_rollback_prefixes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x3b71_293d;
    let mut cases = 0;
    let mut rollback_prefix_cases = 0;
    let mut compared_bytes = 0;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    // The small geometry splits both the convolution and Z boundaries
    // within output-channel pairs. Every tensor/weight is an offset view.
    for (k, c, v, h) in [(2560, 10240, 6144, 48), (128, 191, 63, 2)] {
        let n = c + v + 2 * h;
        let w = weights(&ctx, &mut seed, n, k)?;
        for m in 1..=4 {
            for slot in 0..2 {
                for capture_enabled in [false, true] {
                    for style in 0..4 {
                        let mut x_values = values(&mut seed, m * k, 4.0);
                        let mut conv_values = values(&mut seed, 4 * c, 1.0);
                        for (i, value) in x_values.iter_mut().enumerate() {
                            match style {
                                1 => *value = [0.0, -0.0][i % 2],
                                2 => *value = [2.0, -2.0, 0.5, -0.5, 0.0, -0.0][i % 6],
                                3 => {
                                    *value = [64.0, -64.0, 0.000_001, -0.000_001][i % 4]
                                }
                                _ => {}
                            }
                        }
                        if style == 3 {
                            for (i, value) in conv_values.iter_mut().enumerate() {
                                *value = [40.0, -40.0, 0.0, -0.0][i % 4];
                            }
                        }
                        let input = Guarded::new(&ctx, &[m, k], &x_values)?;
                        let conv_w = Guarded::new(&ctx, &[4, c], &conv_values)?;
                        let initial = [
                            values(&mut seed, 3 * c, 2.0),
                            values(&mut seed, 3 * c, 2.0),
                        ];
                        let windows = [
                            Guarded::new(&ctx, &[c, 3], &initial[0])?,
                            Guarded::new(&ctx, &[c, 3], &initial[1])?,
                        ];
                        let original_window = windows[slot].value.raw_bytes().to_vec();
                        let ordinary_window =
                            Guarded::new(&ctx, &[c, 3], &initial[slot])?;
                        let reference_raw = Guarded::zero(&ctx, &[m, n])?;
                        let reference_conv: Vec<_> = (0..m)
                            .map(|_| Guarded::zero(&ctx, &[c]))
                            .collect::<Result<_>>()?;
                        let mut prefix_windows = vec![original_window.clone()];
                        for row in 0..m {
                            let pass = ctx.begin_concurrent()?;
                            gemv_q4_gdn_conv_step(
                                &ctx,
                                &pass,
                                &w,
                                &input.value.view(row * k, &[k])?,
                                &reference_raw.value.view(row * n, &[n])?,
                                &ordinary_window.value,
                                &conv_w.value,
                                &reference_conv[row].value,
                            )?;
                            pass.end()?.commit()?.wait()?;
                            prefix_windows
                                .push(ordinary_window.value.raw_bytes().to_vec());
                        }
                        let qkv = Guarded::zero(&ctx, &[m, c])?;
                        let z = Guarded::zero(&ctx, &[m, v])?;
                        let a = Guarded::zero(&ctx, &[m, h])?;
                        let b = Guarded::zero(&ctx, &[m, h])?;
                        let conv_out = Guarded::zero(&ctx, &[m, c])?;
                        let capture = Guarded::new(&ctx, &[m, c], &vec![42.0; m * c])?;
                        let capture_before = capture.value.raw_bytes().to_vec();
                        let rollbacks: Vec<_> = (0..=m)
                            .map(|_| Guarded::zero(&ctx, &[c, 3]))
                            .collect::<Result<_>>()?;
                        let pass = ctx.begin_concurrent()?;
                        gemv_q4_gdn_conv_rows_exact(
                            &ctx,
                            &pass,
                            &w,
                            &input.value,
                            &qkv.value,
                            &z.value,
                            &a.value,
                            &b.value,
                            &windows[slot].value,
                            &windows[1 - slot].value,
                            &conv_w.value,
                            &conv_out.value,
                            capture_enabled.then_some(&capture.value),
                        )?;
                        pass.level_barrier(&[
                            &qkv.value,
                            &capture.value,
                            &windows[1 - slot].value,
                        ])?;
                        for prefix in 0..=m {
                            conv_window_rollback(
                                &ctx,
                                &pass,
                                &windows[slot].value,
                                if capture_enabled {
                                    &capture.value
                                } else {
                                    &qkv.value
                                },
                                &rollbacks[prefix].value,
                                prefix,
                            )?;
                        }
                        pass.end()?.commit()?.wait()?;
                        for (out, offset, width, field) in [
                            (&qkv, 0, c, "qkv"),
                            (&z, c, v, "z"),
                            (&a, c + v, h, "a"),
                            (&b, c + v + h, h, "b"),
                        ] {
                            for row in 0..m {
                                compare(
                                    &reference_raw.value.raw_bytes()[2
                                        * (row * n + offset)
                                        ..2 * (row * n + offset + width)],
                                    &out.value.raw_bytes()
                                        [2 * row * width..2 * (row + 1) * width],
                                    field,
                                    &mut reference_hash,
                                    &mut candidate_hash,
                                    &mut compared_bytes,
                                )?;
                            }
                        }
                        for row in 0..m {
                            compare(
                                reference_conv[row].value.raw_bytes(),
                                &conv_out.value.raw_bytes()
                                    [2 * row * c..2 * (row + 1) * c],
                                "convolution",
                                &mut reference_hash,
                                &mut candidate_hash,
                                &mut compared_bytes,
                            )?;
                        }
                        compare(
                            &original_window,
                            windows[slot].value.raw_bytes(),
                            "preserved input window",
                            &mut reference_hash,
                            &mut candidate_hash,
                            &mut compared_bytes,
                        )?;
                        compare(
                            ordinary_window.value.raw_bytes(),
                            windows[1 - slot].value.raw_bytes(),
                            "final window",
                            &mut reference_hash,
                            &mut candidate_hash,
                            &mut compared_bytes,
                        )?;
                        compare(
                            if capture_enabled {
                                qkv.value.raw_bytes()
                            } else {
                                &capture_before
                            },
                            capture.value.raw_bytes(),
                            "capture",
                            &mut reference_hash,
                            &mut candidate_hash,
                            &mut compared_bytes,
                        )?;
                        for prefix in 0..=m {
                            compare(
                                &prefix_windows[prefix],
                                rollbacks[prefix].value.raw_bytes(),
                                "rollback window",
                                &mut reference_hash,
                                &mut candidate_hash,
                                &mut compared_bytes,
                            )?;
                            rollback_prefix_cases += 1;
                        }
                        for tensor in [
                            &input,
                            &conv_w,
                            &windows[0],
                            &windows[1],
                            &ordinary_window,
                            &reference_raw,
                            &qkv,
                            &z,
                            &a,
                            &b,
                            &conv_out,
                            &capture,
                        ]
                        .into_iter()
                        .chain(reference_conv.iter())
                        .chain(rollbacks.iter())
                        {
                            tensor.canaries()?;
                            ensure!(
                                tensor.value.to_f32()?.iter().all(|x| x.is_finite()),
                                "GDN fixture output is nonfinite"
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "GDN projection rows exact: cases={cases} rollback_prefix_cases={rollback_prefix_cases} compared_bytes={compared_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
}

struct StreamingOutputs {
    stack: Tensor,
    qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    capture: Tensor,
    conv: Tensor,
    windows: Vec<Tensor>,
}

impl StreamingOutputs {
    fn new(ctx: &MetalContext, m: usize, layers: usize) -> Result<Self> {
        let bf = DType::BF16;
        Ok(Self {
            stack: Tensor::zeros(ctx, &[m, 16480], bf)?,
            qkv: Tensor::zeros(ctx, &[m, 10240], bf)?,
            z: Tensor::zeros(ctx, &[m, 6144], bf)?,
            a: Tensor::zeros(ctx, &[m, 48], bf)?,
            b: Tensor::zeros(ctx, &[m, 48], bf)?,
            capture: Tensor::zeros(ctx, &[m, 10240], bf)?,
            conv: Tensor::zeros(ctx, &[m, 10240], bf)?,
            windows: (0..layers)
                .map(|_| Tensor::zeros(ctx, &[10240, 3], bf))
                .collect::<Result<_>>()?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn streaming_route(
    ctx: &MetalContext,
    weights: &[QuantWeights],
    input: &Tensor,
    windows: &[Tensor],
    conv_w: &Tensor,
    output: &StreamingOutputs,
    fused: bool,
) -> Result<f64> {
    use crate::kernels::elementwise::{copy_words, split_cols_bf16};
    use crate::kernels::gdn::conv1d_prefill;
    let pass = ctx.begin_concurrent()?;
    for (index, w) in weights.iter().enumerate() {
        if fused {
            gemv_q4_gdn_conv_rows_exact(
                ctx,
                &pass,
                w,
                input,
                &output.qkv,
                &output.z,
                &output.a,
                &output.b,
                &windows[index],
                &output.windows[index],
                conv_w,
                &output.conv,
                Some(&output.capture),
            )?;
        } else {
            gemv_quant_rows_decode_exact(ctx, &pass, w, input, &output.stack)?;
            pass.level_barrier(&[&output.stack])?;
            split_cols_bf16(
                ctx,
                &pass,
                &output.stack,
                &[&output.qkv, &output.z, &output.a, &output.b],
            )?;
            pass.level_barrier(&[&output.qkv, &output.z, &output.a, &output.b])?;
            copy_words(ctx, &pass, &output.qkv, &output.capture)?;
            conv1d_prefill(
                ctx,
                &pass,
                &windows[index],
                &output.windows[index],
                &output.qkv,
                conv_w,
                &output.conv,
            )?;
        }
        pass.level_barrier(&[
            &output.qkv,
            &output.z,
            &output.a,
            &output.b,
            &output.capture,
            &output.conv,
            &output.windows[index],
        ])?;
    }
    let completed = pass.commit()?.wait_retain()?;
    let timing = completed.timing()?;
    let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "GDN projection rows timing unavailable"
    );
    Ok(seconds)
}

fn compare_streaming(
    reference: &StreamingOutputs,
    candidate: &StreamingOutputs,
) -> Result<()> {
    for (a, b) in [
        (&reference.qkv, &candidate.qkv),
        (&reference.z, &candidate.z),
        (&reference.a, &candidate.a),
        (&reference.b, &candidate.b),
        (&reference.capture, &candidate.capture),
        (&reference.conv, &candidate.conv),
    ]
    .into_iter()
    .chain(reference.windows.iter().zip(candidate.windows.iter()))
    {
        ensure!(a.raw_bytes() == b.raw_bytes(), "GDN projection timing output differs");
    }
    Ok(())
}

#[test]
#[ignore = "physical Apple Silicon; synthetic alternating component timings only"]
fn fused_gdn_projection_rows_streaming_timing() -> Result<()> {
    use std::time::{Duration, Instant};
    let ctx = MetalContext::new()?;
    let mut seed = 0x4d38_9a71;
    let layers = 36;
    let template = weights(&ctx, &mut seed, 16480, 2560)?;
    // Identical contents in distinct allocations keep fixture preparation
    // bounded while forcing streaming physical weight addresses.
    let weights: Vec<_> = (0..layers)
        .map(|_| {
            Ok(QuantWeights {
                codes: Tensor::from_bytes(
                    &ctx,
                    template.codes.raw_bytes(),
                    template.codes.shape(),
                    DType::U32,
                )?,
                scales: Tensor::from_bytes(
                    &ctx,
                    template.scales.raw_bytes(),
                    template.scales.shape(),
                    DType::BF16,
                )?,
                biases: Tensor::from_bytes(
                    &ctx,
                    template.biases.raw_bytes(),
                    template.biases.shape(),
                    DType::BF16,
                )?,
                bits: 4,
                group_size: 64,
            })
        })
        .collect::<Result<_>>()?;
    let weight_bytes = weights
        .iter()
        .map(|w| w.codes.byte_len() + w.scales.byte_len() + w.biases.byte_len())
        .sum::<usize>();
    let windows = (0..layers)
        .map(|_| {
            Tensor::from_f32_as_bf16(
                &ctx,
                &values(&mut seed, 3 * 10240, 2.0),
                &[10240, 3],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let conv_w = Tensor::from_f32_as_bf16(
        &ctx,
        &values(&mut seed, 4 * 10240, 1.0),
        &[4, 10240],
    )?;
    for m in 2..=4 {
        let input = Tensor::from_f32_as_bf16(
            &ctx,
            &values(&mut seed, m * 2560, 4.0),
            &[m, 2560],
        )?;
        let outputs = [
            StreamingOutputs::new(&ctx, m, layers)?,
            StreamingOutputs::new(&ctx, m, layers)?,
        ];
        streaming_route(&ctx, &weights, &input, &windows, &conv_w, &outputs[0], false)?;
        streaming_route(&ctx, &weights, &input, &windows, &conv_w, &outputs[1], true)?;
        compare_streaming(&outputs[0], &outputs[1])?;
        let warmup = Instant::now();
        let mut warmup_round = 0;
        while warmup.elapsed() < Duration::from_millis(500) {
            let route = warmup_round % 2;
            streaming_route(
                &ctx,
                &weights,
                &input,
                &windows,
                &conv_w,
                &outputs[route],
                route == 1,
            )?;
            warmup_round += 1;
        }
        let mut samples = [Vec::new(), Vec::new()];
        for round in 0..18 {
            for offset in 0..2 {
                let route = (round + offset) % 2;
                samples[route].push(
                    streaming_route(
                        &ctx,
                        &weights,
                        &input,
                        &windows,
                        &conv_w,
                        &outputs[route],
                        route == 1,
                    )? * 1e6
                        / layers as f64,
                );
            }
            compare_streaming(&outputs[0], &outputs[1])?;
        }
        let means = samples.each_ref().map(|v| v.iter().sum::<f64>() / v.len() as f64);
        let medians = samples.each_ref().map(|v| {
            let mut x = v.clone();
            x.sort_by(f64::total_cmp);
            (x[8] + x[9]) / 2.0
        });
        eprintln!(
            "{}",
            serde_json::json!({
                "scope":"synthetic_gdn_projection_conv_scatter_streaming",
                "rows":m,"layers":layers,"weight_bytes":weight_bytes,"pairs":18,
                "reference_mean_gpu_us_per_layer":means[0],"fused_mean_gpu_us_per_layer":means[1],
                "reference_median_gpu_us_per_layer":medians[0],"fused_median_gpu_us_per_layer":medians[1],
                "mean_speedup":means[0]/means[1],"byte_control":true,
                "reference_gpu_us_per_layer":samples[0],"fused_gpu_us_per_layer":samples[1],
            })
        );
    }
    Ok(())
}
