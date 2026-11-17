// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
#[ignore = "requires an Apple GPU with Metal 4 tensor operations"]
fn selected_gdn_chunk_capture_matches_standalone_prefix() -> Result<()> {
    const H: usize = 48;
    const HK: usize = 16;
    const DIM: usize = GDN_HEAD_DIM;
    const CUT: usize = 2048;
    const SCALE: f32 = 0.27;
    let ctx = MetalContext::new()?;
    let mut seed = 0xc39d_c567u32;
    let mut values = |len: usize, scale: f32, bias: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed % 4096) as f32 * (scale / 4096.0) + bias
            })
            .collect()
    };

    let a_log = Tensor::from_f32(&ctx, &values(H, 1.0, -6.0), &[H])?;
    let dt_bias = Tensor::from_f32_as_bf16(&ctx, &values(H, 0.5, -0.25), &[H])?;
    let state_shape = [H, DIM, DIM];
    let initial = values(H * DIM * DIM, 0.02, -0.01);
    let new_state = || Tensor::from_f32(&ctx, &initial, &state_shape);
    let mut compared_bytes = 0;

    // A selected block at the end of the WY scan with one serial token,
    // a selected block before more WY work and a tail, and a full 4K pass.
    for rows in [CUT + 1, CUT + 65, 2 * CUT] {
        let width = (2 * HK + H) * DIM;
        let qkv = Tensor::from_f32_as_bf16(
            &ctx,
            &values(rows * width, 0.3, -0.15),
            &[rows, width],
        )?;
        let a =
            Tensor::from_f32_as_bf16(&ctx, &values(rows * H, 1.0, -0.5), &[rows, H])?;
        let b =
            Tensor::from_f32_as_bf16(&ctx, &values(rows * H, 1.0, -0.5), &[rows, H])?;
        let qk = Tensor::zeros(&ctx, &[rows, 2 * HK * DIM], DType::BF16)?;
        let decay = Tensor::zeros(&ctx, &[rows, H], DType::F32)?;
        let beta = Tensor::zeros(&ctx, &[rows, H], DType::F32)?;
        let w = Tensor::zeros(&ctx, &[rows, H, DIM], DType::BF16)?;
        let u = Tensor::zeros(&ctx, &[rows, H, DIM], DType::BF16)?;
        let p = Tensor::zeros(&ctx, &[rows, H, GDN_CHUNK], DType::BF16)?;
        let g = Tensor::zeros(&ctx, &[rows, H], DType::F32)?;
        let staging = GdnRegscanStaging {
            qk_norm: &qk,
            decay: &decay,
            beta: &beta,
            chunk: Some(GdnChunkStaging { w: &w, u: &u, p: &p, g: &g }),
        };
        let ordinary = new_state()?;
        let captured = new_state()?;
        let reference = new_state()?;
        let expected_out = Tensor::zeros(&ctx, &[rows, H, DIM], DType::BF16)?;
        let actual_out = Tensor::zeros(&ctx, &[rows, H, DIM], DType::BF16)?;
        let prefix_out = Tensor::zeros(&ctx, &[CUT, H, DIM], DType::BF16)?;
        let padded = Tensor::zeros(&ctx, &[H * DIM * DIM + 8], DType::F32)?;
        padded.write_bytes(&vec![0xa5; padded.byte_len()])?;
        let snapshot = padded.view(4, &state_shape)?;

        let pass = ctx.begin_concurrent()?;
        gdn_prefill_scan_named(
            &ctx,
            &pass,
            &qkv,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &staging,
            &ordinary,
            &expected_out,
            SCALE,
            HK,
            None,
            GDN_CHUNK_SCAN,
        )?;
        pass.end()?.commit()?.wait()?;

        let pass = ctx.begin_concurrent()?;
        gdn_prefill_capture_key_value(
            &ctx,
            &pass,
            &qkv,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &staging,
            &captured,
            &actual_out,
            SCALE,
            HK,
            CUT,
            &snapshot,
        )?;
        pass.end()?.commit()?.wait()?;

        let w_prefix = w.view(0, &[CUT, H, DIM])?;
        let u_prefix = u.view(0, &[CUT, H, DIM])?;
        let p_prefix = p.view(0, &[CUT, H, GDN_CHUNK])?;
        let g_prefix = g.view(0, &[CUT, H])?;
        let staging_prefix = GdnRegscanStaging {
            qk_norm: &qk.view(0, &[CUT, 2 * HK * DIM])?,
            decay: &decay.view(0, &[CUT, H])?,
            beta: &beta.view(0, &[CUT, H])?,
            chunk: Some(GdnChunkStaging {
                w: &w_prefix,
                u: &u_prefix,
                p: &p_prefix,
                g: &g_prefix,
            }),
        };
        let pass = ctx.begin_concurrent()?;
        gdn_prefill_scan_named(
            &ctx,
            &pass,
            &qkv.view(0, &[CUT, width])?,
            &a.view(0, &[CUT, H])?,
            &b.view(0, &[CUT, H])?,
            &a_log,
            &dt_bias,
            &staging_prefix,
            &reference,
            &prefix_out,
            SCALE,
            HK,
            None,
            GDN_CHUNK_SCAN,
        )?;
        pass.end()?.commit()?.wait()?;

        ensure!(
            actual_out.raw_bytes() == expected_out.raw_bytes(),
            "GDN capture changed output at {rows} rows"
        );
        ensure!(
            captured.raw_bytes() == ordinary.raw_bytes(),
            "GDN capture changed terminal state at {rows} rows"
        );
        ensure!(
            snapshot.raw_bytes() == reference.raw_bytes(),
            "GDN capture did not match a standalone prefix at {rows} rows"
        );
        let bytes = padded.raw_bytes();
        ensure!(
            bytes[..16].iter().chain(&bytes[bytes.len() - 16..]).all(|&b| b == 0xa5),
            "GDN capture wrote outside its output view"
        );

        // Retain the same 64-row WY/serial-tail partition as the unsplit
        // scan. Ordinary short calls may select the serial-only route.
        let suffix = rows - CUT;
        let resume_out = Tensor::zeros(&ctx, &[suffix, H, DIM], DType::BF16)?;
        let suffix_staging = GdnRegscanStaging {
            qk_norm: &qk.view(0, &[suffix, 2 * HK * DIM])?,
            decay: &decay.view(0, &[suffix, H])?,
            beta: &beta.view(0, &[suffix, H])?,
            chunk: Some(GdnChunkStaging {
                w: &w.view(0, &[suffix, H, DIM])?,
                u: &u.view(0, &[suffix, H, DIM])?,
                p: &p.view(0, &[suffix, H, GDN_CHUNK])?,
                g: &g.view(0, &[suffix, H])?,
            }),
        };
        let pass = ctx.begin_concurrent()?;
        gdn_prefill_scan_named(
            &ctx,
            &pass,
            &qkv.view(CUT * width, &[suffix, width])?,
            &a.view(CUT * H, &[suffix, H])?,
            &b.view(CUT * H, &[suffix, H])?,
            &a_log,
            &dt_bias,
            &suffix_staging,
            &snapshot,
            &resume_out,
            SCALE,
            HK,
            None,
            GDN_CHUNK_SCAN,
        )?;
        pass.end()?.commit()?.wait()?;
        ensure!(
            resume_out.raw_bytes()
                == expected_out.view(CUT * H * DIM, &[suffix, H, DIM])?.raw_bytes()
                && snapshot.raw_bytes() == ordinary.raw_bytes(),
            "GDN continuation changed the selected prefix result at {rows} rows"
        );
        compared_bytes += actual_out.byte_len()
            + captured.byte_len()
            + snapshot.byte_len()
            + resume_out.byte_len()
            + snapshot.byte_len();
    }
    println!(
        "{{\"gdn_selected_capture_exact_cases\":3,\"compared_bytes\":{compared_bytes}}}"
    );
    Ok(())
}
