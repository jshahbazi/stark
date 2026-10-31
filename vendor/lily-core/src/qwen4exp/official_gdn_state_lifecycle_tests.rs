// SPDX-License-Identifier: Apache-2.0
// Synthetic physical-layout ownership and lifecycle checks; no model is loaded.
use super::*;
use crate::kernels::gdn::{conv1d_step_with_layout, gdn_step_raw_value_key_mlx};
use crate::sha256::{hex, sha256};

const H: usize = 48;
const D: usize = 128;
const C: usize = 10240;

fn state(
    ctx: &MetalContext,
    layout: GdnStateLayout,
    seed: usize,
) -> Result<DecodeState> {
    let dense: Vec<f32> = (0..H * D * D)
        .map(|i| (((i * 11 + seed) % 257) as i32 - 128) as f32 / 997.0)
        .collect();
    let window = |which| -> Result<Tensor> {
        let values: Vec<f32> = (0..3 * C)
            .map(|i| ((i + seed + which) as i32 % 31 - 15) as f32 / 32.0)
            .collect();
        Tensor::from_f32_as_bf16(ctx, &values, &[C, 3])
    };
    Ok(DecodeState {
        pos: 7,
        rope_delta: 0,
        capacity: 128,
        layers: vec![LayerState::Gdn {
            state: Tensor::from_f32(ctx, &dense, &[H, D, D])?,
            layout,
            conv_windows: [window(0)?, window(7)?],
        }],
        conv_slot: 1,
        ple: None,
        mtp: None,
        spec: None,
        kv_format: KvFormat::Bf16,
        kv_heads: 1,
        head_dim: 128,
        ratio: 4,
    })
}

fn recurrent(state: &DecodeState) -> (&Tensor, &[Tensor; 2], GdnStateLayout) {
    match &state.layers[0] {
        LayerState::Gdn { state, conv_windows, layout } => {
            (state, conv_windows, *layout)
        }
        LayerState::Attn { .. } => unreachable!("synthetic recurrent state"),
    }
}

fn digest(t: &Tensor) -> String {
    hex(&sha256(t.raw_bytes()))
}

#[test]
fn official_value_key_reset_snapshot_restore_preserves_layout_and_decode_continuation()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let reference = state(&ctx, GdnStateLayout::ValueKey, 17)?;
    let mut resumed = state(&ctx, GdnStateLayout::ValueKey, 17)?;
    let snapshot = resumed.snapshot(&ctx)?;
    ensure!(snapshot.gdn[0].2 == GdnStateLayout::ValueKey, "snapshot lost VK policy");
    let expected_state = digest(recurrent(&reference).0);
    let expected_window = digest(&recurrent(&reference).1[1]);
    resumed.reset()?;
    ensure!(
        recurrent(&resumed).0.raw_bytes().iter().all(|v| *v == 0),
        "reset did not zero F32 state"
    );
    ensure!(
        recurrent(&resumed).1.iter().all(|t| t.raw_bytes().iter().all(|v| *v == 0)),
        "reset did not zero conv windows"
    );
    ensure!(
        recurrent(&resumed).2 == GdnStateLayout::ValueKey
            && resumed.pos == 0
            && resumed.conv_slot == 0,
        "reset changed physical policy or position"
    );
    resumed.restore(&ctx, &snapshot)?;
    ensure!(
        digest(recurrent(&resumed).0) == expected_state
            && digest(&recurrent(&resumed).1[0]) == expected_window,
        "restore changed recurrent bytes"
    );
    ensure!(
        resumed.pos == 7
            && resumed.conv_slot == 0
            && recurrent(&resumed).2 == GdnStateLayout::ValueKey,
        "restore metadata or policy mismatch"
    );

    // Cross-policy restore must refuse before issuing any blit or metadata
    // advance, even though both square tensors have the same shape and dtype.
    let mut wrong = state(&ctx, GdnStateLayout::KeyValue, 91)?;
    let before = (
        digest(recurrent(&wrong).0),
        digest(&recurrent(&wrong).1[0]),
        digest(&recurrent(&wrong).1[1]),
        wrong.pos,
        wrong.conv_slot,
    );
    ensure!(
        wrong.restore(&ctx, &snapshot).is_err(),
        "cross-policy snapshot restore was accepted"
    );
    ensure!(
        before
            == (
                digest(recurrent(&wrong).0),
                digest(&recurrent(&wrong).1[0]),
                digest(&recurrent(&wrong).1[1]),
                wrong.pos,
                wrong.conv_slot
            ),
        "rejected restore mutated buffers or metadata"
    );

    // The first actual conv/recurrence after restore must equal continuation
    // from the independent original state, despite restore selecting slot0.
    let xvalues: Vec<f32> =
        (0..C).map(|i| (i as i32 % 29 - 14) as f32 / 16.0).collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &xvalues, &[C])?;
    let weights: Vec<f32> =
        (0..4 * C).map(|i| [0.25, -0.5, 0.375, 1.0][i / C]).collect();
    let weights = Tensor::from_f32_as_bf16(&ctx, &weights, &[4, C])?;
    let a = Tensor::zeros(&ctx, &[H], DType::BF16)?;
    let b = Tensor::from_f32_as_bf16(&ctx, &vec![-6.84375; H], &[H])?;
    let a_log = Tensor::from_f32(&ctx, &vec![0.5; H], &[H])?;
    let dt = Tensor::zeros(&ctx, &[H], DType::BF16)?;
    let raws = [
        Tensor::zeros(&ctx, &[H, D], DType::BF16)?,
        Tensor::zeros(&ctx, &[H, D], DType::BF16)?,
    ];
    let convs = [
        Tensor::zeros(&ctx, &[C], DType::BF16)?,
        Tensor::zeros(&ctx, &[C], DType::BF16)?,
    ];
    let pass = ctx.begin_concurrent()?;
    for (i, current) in [&reference, &resumed].into_iter().enumerate() {
        let (tensor, windows, layout) = recurrent(current);
        conv1d_step_with_layout(
            &ctx,
            &pass,
            &windows[current.conv_slot],
            &x,
            &weights,
            &convs[i],
            true,
            layout,
        )?;
        pass.level_barrier(&[&convs[i], &windows[current.conv_slot]])?;
        gdn_step_raw_value_key_mlx(
            &ctx,
            &pass,
            &convs[i],
            &a,
            &b,
            &a_log,
            &dt,
            tensor,
            &raws[i],
            1.0 / 128.0f32.sqrt(),
            16,
        )?;
    }
    pass.commit()?.wait()?;
    ensure!(
        raws[0].raw_bytes() == raws[1].raw_bytes()
            && recurrent(&reference).0.raw_bytes() == recurrent(&resumed).0.raw_bytes(),
        "restored decode continuation differs"
    );
    Ok(())
}
