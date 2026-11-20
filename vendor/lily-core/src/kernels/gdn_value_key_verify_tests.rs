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
    let mut replay_cases = 0;
    // Four rows is today's verify limit; nine checks the row count stays a
    // parameter.
    for (draw, rows) in (0..3usize).flat_map(|draw| [1, 2, 3, 4, 9].map(move |rows| (draw, rows))) {
        let x = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * C, 7 + rows + 1000 * draw, 73.0),
            &[rows, C],
        )?;
        let a = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * H, 9 + rows + 1000 * draw, 200.0),
            &[rows, H],
        )?;
        let b = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * H, 11 + rows + 1000 * draw, 50.0),
            &[rows, H],
        )?;
        let z = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * V, 13 + rows + 1000 * draw, 20.0),
            &[rows, V],
        )?;
        let history = values(3 * C, 19 + rows + 1000 * draw, 60.0);
        let initial_state = values(S, 23 + rows + 1000 * draw, 1000.0);
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
        // The verify pass writes its final state to a separate buffer and
        // keeps `state` (pre-verify) for replay rollback.
        let state_after = Tensor::zeros(&ctx, &[H, 128, 128], DType::F32)?;
        let in_place = Tensor::from_f32(&ctx, &initial_state, &[H, 128, 128])?;
        let replayed: Vec<_> = (0..rows)
            .map(|_| Tensor::zeros(&ctx, &[H, 128, 128], DType::F32))
            .collect::<Result<_>>()?;
        let sink = Tensor::zeros(&ctx, &[rows, V], DType::BF16)?;
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
            &state_after,
            &z,
            &norm,
            &raw,
            &out,
            scale,
            1e-6,
        )?;
        pass.level_barrier(&[&raw, &out, &state_after])?;
        // In place (state_in aliases state_out) gives the same final state.
        gdn_verify_value_key_exact(
            &ctx, &pass, &conv, &a, &b, &a_log, &dt, &in_place, &in_place, &z, &norm,
            &raw, &out, scale, 1e-6,
        )?;
        pass.level_barrier(&[&raw, &out, &in_place])?;
        // Replay rollback to every accepted prefix from the retained state.
        for (n, target) in replayed.iter().enumerate().map(|(i, t)| (i + 1, t)) {
            gdn_replay_rows(
                &ctx,
                &pass,
                GdnStateLayout::ValueKey,
                &GdnReplayRows {
                    qkv: &conv.view(0, &[n, C])?,
                    a: &a.view(0, &[n, H])?,
                    b: &b.view(0, &[n, H])?,
                },
                &a_log,
                &dt,
                &norm,
                &state,
                target,
                &sink,
                scale,
                16,
                1e-6,
                GdnGate::Sigmoid,
            )?;
            pass.level_barrier(&[target, &sink])?;
        }
        pass.commit()?.wait()?;
        ensure!(
            state.raw_bytes()
                == Tensor::from_f32(&ctx, &initial_state, &[H, 128, 128])?.raw_bytes(),
            "VK verify modified its input state"
        );
        let mut expected_prefix = Vec::new();
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
            expected_prefix.push(reference_state.raw_bytes().to_vec());
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
            state_after.raw_bytes() == reference_state.raw_bytes(),
            "VK final state mismatch"
        );
        ensure!(
            in_place.raw_bytes() == reference_state.raw_bytes(),
            "VK in-place final state mismatch"
        );
        for (n, (target, expected)) in replayed.iter().zip(&expected_prefix).enumerate() {
            ensure!(
                target.raw_bytes() == &expected[..],
                "VK replay state mismatch rows={rows} accepted_prefix={}",
                n + 1
            );
            replay_cases += 1;
        }
    }
    eprintln!("VK verify replay exact: replay_cases={replay_cases}");
    Ok(())
}
