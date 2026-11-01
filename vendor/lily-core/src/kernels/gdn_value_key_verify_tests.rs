// SPDX-License-Identifier: Apache-2.0
// Synthetic, in-memory exact controls; never print tensor contents.
use super::*;

fn values(n: usize, seed: usize, divisor: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + i / 7 * 11 + seed) % 251) as i32 - 125) as f32 / divisor)
        .collect()
}

#[test]
#[ignore = "requires physical Metal; exact ordinary ValueKey verification control"]
fn official_value_key_verification_rows_match_decode_and_captures() -> Result<()> {
    let ctx = MetalContext::new()?;
    const C: usize = 10240;
    const V: usize = 6144;
    const H: usize = 48;
    const S: usize = H * 128 * 128;
    let a_log = Tensor::from_f32(&ctx, &values(H, 17, 100.0), &[H])?;
    let dt = Tensor::from_f32_as_bf16(&ctx, &values(H, 31, 90.0), &[H])?;
    let norm = Tensor::from_f32(&ctx, &values(128, 43, 80.0), &[128])?;
    let conv_w = Tensor::from_f32_as_bf16(&ctx, &values(4 * C, 5, 250.0), &[4, C])?;
    let scale = 1.0f32 / 128.0f32.sqrt();
    for rows in 1..=4 {
        let x = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * C, 7 + rows, 73.0),
            &[rows, C],
        )?;
        let a = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * H, 9 + rows, 200.0),
            &[rows, H],
        )?;
        let b = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * H, 11 + rows, 50.0),
            &[rows, H],
        )?;
        let z = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * V, 13 + rows, 20.0),
            &[rows, V],
        )?;
        let history = values(3 * C, 19 + rows, 60.0);
        let initial_state = values(S, 23 + rows, 1000.0);
        let wi = Tensor::from_f32_as_bf16(&ctx, &history, &[C, 3])?;
        let wo = Tensor::zeros(&ctx, &[C, 3], DType::BF16)?;
        let wr = Tensor::from_f32_as_bf16(&ctx, &history, &[C, 3])?;
        let state = Tensor::from_f32(&ctx, &initial_state, &[H, 128, 128])?;
        let reference_state = Tensor::from_f32(&ctx, &initial_state, &[H, 128, 128])?;
        let conv = Tensor::zeros(&ctx, &[rows, C], DType::BF16)?;
        let conv_ref = Tensor::zeros(&ctx, &[rows, C], DType::BF16)?;
        let raw = Tensor::zeros(&ctx, &[rows, V], DType::BF16)?;
        let raw_ref = Tensor::zeros(&ctx, &[rows, V], DType::BF16)?;
        let out = Tensor::zeros(&ctx, &[rows, V], DType::BF16)?;
        let out_ref = Tensor::zeros(&ctx, &[rows, V], DType::BF16)?;
        let mid = (rows > 1)
            .then(|| Tensor::zeros(&ctx, &[rows - 1, H, 128, 128], DType::F32))
            .transpose()?;
        let pass = ctx.begin_concurrent()?;
        conv1d_verify_value_key_exact(&ctx, &pass, &wi, &wo, &x, &conv_w, &conv)?;
        pass.level_barrier(&[&conv, &wo])?;
        gdn_verify_value_key_exact(
            &ctx,
            &pass,
            &conv,
            &a,
            &b,
            &a_log,
            &dt,
            &state,
            &z,
            &norm,
            &raw,
            &out,
            scale,
            1e-6,
            mid.as_ref(),
        )?;
        pass.commit()?.wait()?;
        let mut expected_mid = Vec::new();
        for row in 0..rows {
            let qr = conv_ref.view(row * C, &[C])?;
            let rr = raw_ref.view(row * V, &[V])?;
            let or = out_ref.view(row * V, &[V])?;
            let pass = ctx.begin_concurrent()?;
            conv1d_step_with_layout(
                &ctx,
                &pass,
                &wr,
                &x.view(row * C, &[C])?,
                &conv_w,
                &qr,
                true,
                GdnStateLayout::ValueKey,
            )?;
            pass.level_barrier(&[&qr, &wr])?;
            gdn_step_gated_fused_with_layout(
                &ctx,
                &pass,
                &qr,
                &a.view(row * H, &[H])?,
                &b.view(row * H, &[H])?,
                &a_log,
                &dt,
                &reference_state,
                &z.view(row * V, &[V])?,
                &norm,
                &or,
                scale,
                16,
                1e-6,
                GdnGate::Sigmoid,
                true,
                GdnStateLayout::ValueKey,
                &rr,
            )?;
            pass.commit()?.wait()?;
            if row + 1 < rows {
                expected_mid.extend_from_slice(reference_state.raw_bytes());
            }
        }
        ensure!(
            wi.raw_bytes()
                == Tensor::from_f32_as_bf16(&ctx, &history, &[C, 3])?.raw_bytes(),
            "VK input history modified"
        );
        ensure!(wo.raw_bytes() == wr.raw_bytes(), "VK conv final window mismatch");
        ensure!(conv.raw_bytes() == conv_ref.raw_bytes(), "VK conv output mismatch");
        ensure!(raw.raw_bytes() == raw_ref.raw_bytes(), "VK recurrence raw mismatch");
        ensure!(out.raw_bytes() == out_ref.raw_bytes(), "VK norm-gate output mismatch");
        ensure!(
            state.raw_bytes() == reference_state.raw_bytes(),
            "VK final state mismatch"
        );
        if let Some(mid) = mid {
            ensure!(mid.raw_bytes() == expected_mid, "VK capture state mismatch");
        }
    }
    Ok(())
}
