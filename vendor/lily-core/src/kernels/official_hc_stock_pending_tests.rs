// SPDX-License-Identifier: Apache-2.0
// Source-bound synthetic WR1 controls. No arrays leave RAM.
fn pending_branch(frame: usize, second: bool) -> Vec<bf16> {
    (0..H)
        .map(|i| {
            let value = if second {
                (((i * 11 + frame * 17) % 257) as f32 - 128.0) / 128.0
            } else {
                (((i * 7 + frame * 13) % 193) as f32 - 96.0) / 64.0
            };
            bf16::from_f32(value)
        })
        .collect()
}

fn pending_gate(frame: usize) -> [bf16; G] {
    let values = [0.72265625, 1.140625, 1.6640625, 0.30859375];
    std::array::from_fn(|g| bf16::from_f32(values[(g + frame) % G]))
}

fn pending_stage(seed: u32, inject: bool, frame: usize, stage: &str, bytes: &[u8]) {
    println!(
        "{}",
        serde_json::json!({
            "scope": "stock_hc_pending_stage", "seed": seed, "inject": inject,
            "frame": frame, "stage": stage, "bytes": bytes.len(), "sha256": digest(bytes),
        })
    );
}

#[test]
fn official_stock_pending_write_pair_selection_respects_skip_routes() {
    for bits in 0u32..64 {
        let args: [bool; 6] = std::array::from_fn(|i| bits & (1 << i) != 0);
        assert_eq!(
            write_pair_selected(args[0], args[1], args[2], args[3], args[4], args[5]),
            bits == 63,
        );
    }
    for official in [false, true] {
        for rows in [1, 2] {
            for read_enabled in [false, true] {
                for injection_enabled in [false, true] {
                    let selected =
                        full_read_selected(official, rows, true, read_enabled);
                    assert_eq!(
                        write_pair_selected(
                            selected,
                            selected,
                            true,
                            true,
                            injection_enabled,
                            true
                        ),
                        official && rows == 1 && read_enabled && injection_enabled,
                    );
                }
            }
        }
    }
}

#[test]
fn official_stock_wr1_matches_materialized_wr0_eight_frames_with_offsets() -> Result<()>
{
    let ctx = MetalContext::new()?;
    for seed in [0x48434631, 0x48434632] {
        for inject in [false, true] {
            let f = fixture(&ctx, seed)?;
            let prepared = OfficialHcReadWeights::prepare(
                &ctx,
                &f.norm,
                &f.down,
                &f.up,
                inject.then_some(&f.inject),
                H,
                G,
                1e-6,
                true,
            )?
            .ok_or_else(|| anyhow::anyhow!("pending descriptor declined"))?;
            let reference = Guarded::new(&ctx, K, DType::BF16)?;
            let canonical = Guarded::new(&ctx, K, DType::BF16)?;
            let written = Guarded::new(&ctx, K, DType::BF16)?;
            let norm_ref = Guarded::new(&ctx, K, DType::BF16)?;
            let norm_new = Guarded::new(&ctx, K, DType::BF16)?;
            let parts_ref = Guarded::new(&ctx, PARTS, DType::F32)?;
            let parts_new = Guarded::new(&ctx, PARTS, DType::F32)?;
            let mixed_ref = Guarded::new(&ctx, H, DType::BF16)?;
            let mixed_new = Guarded::new(&ctx, H, DType::BF16)?;
            let gate_ref = Guarded::new(&ctx, G, DType::BF16)?;
            let gate_new = Guarded::new(&ctx, G, DType::BF16)?;
            let incoming_gate = Guarded::new(&ctx, G, DType::BF16)?;
            let branch = Guarded::new(&ctx, H, DType::BF16)?;
            reference.value.write_bytes(f.x.raw_bytes())?;
            canonical.value.write_bytes(f.x.raw_bytes())?;
            for frame in 0..8 {
                let before = canonical.value.raw_bytes().to_vec();
                let a = pending_branch(frame, false);
                let gates = pending_gate(frame);
                branch.value.write_bytes(bytemuck::cast_slice(&a))?;
                incoming_gate.value.write_bytes(bytemuck::cast_slice(&gates))?;
                gate_ref.value.write_bytes(bytemuck::cast_slice(&gates))?;
                // Production reuses the old attention gate buffer as the
                // next MLP gate output. The ND -> U2V barrier owns this reuse.
                gate_new.value.write_bytes(bytemuck::cast_slice(&gates))?;
                let pass = ctx.begin_concurrent()?;
                inject_after_read(
                    &ctx,
                    &pass,
                    &reference.value,
                    &branch.value,
                    &incoming_gate.value,
                    H,
                    G,
                    true,
                    false,
                    InjectionInputKind::PrecomputedGate,
                )?;
                pass.level_barrier(&[&reference.value])?;
                prepared.read(
                    &ctx,
                    &pass,
                    &reference.value,
                    &f.norm,
                    &norm_ref.value,
                    &parts_ref.value,
                    &mixed_ref.value,
                    &gate_ref.value,
                )?;
                prepared.read_after_write(
                    &ctx,
                    &pass,
                    &canonical.value,
                    &f.norm,
                    &norm_new.value,
                    &parts_new.value,
                    &mixed_new.value,
                    &gate_new.value,
                    &written.value,
                    PendingHcWrite {
                        branch: &branch.value,
                        gate: &gate_new.value,
                        kind: InjectionInputKind::PrecomputedGate,
                    },
                )?;
                pass.commit()?.wait()?;
                equal(
                    written.value.raw_bytes(),
                    reference.value.raw_bytes(),
                    "WR1 materialized stream",
                )?;
                equal(
                    canonical.value.raw_bytes(),
                    &before,
                    "WR1 mutated input stream",
                )?;
                equal(
                    norm_new.value.raw_bytes(),
                    norm_ref.value.raw_bytes(),
                    "WR1 normalized stream",
                )?;
                equal(
                    &valid_parts(&parts_new.value, false),
                    &valid_parts(&parts_ref.value, false),
                    "WR1 down partials",
                )?;
                equal(
                    mixed_new.value.raw_bytes(),
                    mixed_ref.value.raw_bytes(),
                    "WR1 mixed output",
                )?;
                equal(
                    gate_new.value.raw_bytes(),
                    gate_ref.value.raw_bytes(),
                    "WR1 gate output or untouched no-inject input",
                )?;
                if inject {
                    equal(
                        &valid_parts(&parts_new.value, true),
                        &valid_parts(&parts_ref.value, true),
                        "WR1 inject partials",
                    )?;
                }
                let old: &[bf16] = bytemuck::cast_slice(&before);
                let expected: Vec<bf16> = (0..K)
                    .map(|j| {
                        let delta =
                            bf16::from_f32(a[j % H].to_f32() * gates[j / H].to_f32());
                        bf16::from_f32(old[j].to_f32() + delta.to_f32())
                    })
                    .collect();
                equal(
                    written.value.raw_bytes(),
                    bytemuck::cast_slice(&expected),
                    "WR1 explicit two-round product/add",
                )?;
                if frame == 0 {
                    ensure!(
                        (0..K).any(|j| {
                            let unrounded_product =
                                a[j % H].to_f32() * gates[j / H].to_f32();
                            bf16::from_f32(old[j].to_f32() + unrounded_product)
                                != expected[j]
                        }),
                        "pending fixture has no product-rounding witness"
                    );
                }
                pending_stage(
                    seed,
                    inject,
                    frame,
                    "written_bf16",
                    written.value.raw_bytes(),
                );
                pending_stage(
                    seed,
                    inject,
                    frame,
                    "normalized_bf16",
                    norm_new.value.raw_bytes(),
                );
                pending_stage(
                    seed,
                    inject,
                    frame,
                    "down_stream_partials_f32",
                    &valid_parts(&parts_new.value, false),
                );
                pending_stage(
                    seed,
                    inject,
                    frame,
                    "mixed_bf16",
                    mixed_new.value.raw_bytes(),
                );
                if inject {
                    pending_stage(
                        seed,
                        inject,
                        frame,
                        "inject_stream_partials_f32",
                        &valid_parts(&parts_new.value, true),
                    );
                    pending_stage(
                        seed,
                        inject,
                        frame,
                        "inject_gate_bf16",
                        gate_new.value.raw_bytes(),
                    );
                }

                // Replace the old final write in the pair without adding a
                // copy. Carry canonical output into the next frame.
                branch
                    .value
                    .write_bytes(bytemuck::cast_slice(&pending_branch(frame, true)))?;
                let pass = ctx.begin_concurrent()?;
                inject_after_read(
                    &ctx,
                    &pass,
                    &reference.value,
                    &branch.value,
                    &gate_ref.value,
                    H,
                    G,
                    true,
                    false,
                    InjectionInputKind::PrecomputedGate,
                )?;
                write_gate_to(
                    &ctx,
                    &pass,
                    &written.value,
                    &canonical.value,
                    true,
                    PendingHcWrite {
                        branch: &branch.value,
                        gate: &gate_new.value,
                        kind: InjectionInputKind::PrecomputedGate,
                    },
                )?;
                pass.commit()?.wait()?;
                equal(
                    canonical.value.raw_bytes(),
                    reference.value.raw_bytes(),
                    "WR1 pair canonical output",
                )?;
                pending_stage(
                    seed,
                    inject,
                    frame,
                    "canonical_bf16",
                    canonical.value.raw_bytes(),
                );
                for guarded in [
                    &reference,
                    &canonical,
                    &written,
                    &norm_ref,
                    &norm_new,
                    &parts_ref,
                    &parts_new,
                    &mixed_ref,
                    &mixed_new,
                    &gate_ref,
                    &gate_new,
                    &incoming_gate,
                    &branch,
                ] {
                    guarded.check()?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn official_stock_pending_write_rejects_aliases_raw_gates_and_multirow() -> Result<()> {
    let ctx = MetalContext::new()?;
    let f = fixture(&ctx, 0x48434631)?;
    let prepared = OfficialHcReadWeights::prepare(
        &ctx,
        &f.norm,
        &f.down,
        &f.up,
        Some(&f.inject),
        H,
        G,
        1e-6,
        true,
    )?
    .ok_or_else(|| anyhow::anyhow!("pending descriptor declined"))?;
    let normalized = Tensor::zeros(&ctx, &[1, K], DType::BF16)?;
    let parts = Tensor::zeros(&ctx, &[1, PARTS], DType::F32)?;
    let mixed = Tensor::zeros(&ctx, &[1, H], DType::BF16)?;
    let gate = Tensor::zeros(&ctx, &[1, G], DType::BF16)?;
    let branch = Tensor::zeros(&ctx, &[1, H], DType::BF16)?;
    let written = Tensor::zeros(&ctx, &[1, K], DType::BF16)?;
    let pending = PendingHcWrite {
        branch: &branch,
        gate: &gate,
        kind: InjectionInputKind::PrecomputedGate,
    };
    let pass = ctx.begin_concurrent()?;
    ensure!(
        prepared
            .read_after_write(
                &ctx,
                &pass,
                &f.x,
                &f.norm,
                &normalized,
                &parts,
                &mixed,
                &gate,
                &f.x,
                pending,
            )
            .is_err(),
        "WR1 accepted in-place stream alias"
    );
    ensure!(
        prepared
            .read_after_write(
                &ctx,
                &pass,
                &f.x,
                &f.norm,
                &normalized,
                &parts,
                &mixed,
                &gate,
                &normalized,
                pending,
            )
            .is_err(),
        "WR1 accepted normalized-output alias"
    );
    let raw = PendingHcWrite { kind: InjectionInputKind::RawLogits, ..pending };
    ensure!(
        prepared
            .read_after_write(
                &ctx,
                &pass,
                &f.x,
                &f.norm,
                &normalized,
                &parts,
                &mixed,
                &gate,
                &written,
                raw,
            )
            .is_err(),
        "WR1 accepted raw logits as a finished gate"
    );
    let multirow = Tensor::zeros(&ctx, &[2, K], DType::BF16)?;
    ensure!(
        prepared
            .read_after_write(
                &ctx,
                &pass,
                &multirow,
                &f.norm,
                &normalized,
                &parts,
                &mixed,
                &gate,
                &written,
                pending,
            )
            .is_err(),
        "WR1 accepted multirow input"
    );
    let shared = Tensor::zeros(&ctx, &[K + 2], DType::BF16)?;
    ensure!(
        write_gate_to(
            &ctx,
            &pass,
            &shared.view(0, &[1, K])?,
            &shared.view(2, &[1, K])?,
            true,
            pending,
        )
        .is_err(),
        "gate output accepted partial residual overlap"
    );
    ensure!(
        write_gate_to(&ctx, &pass, &f.x, &written, true, raw).is_err(),
        "gate output accepted raw logits"
    );
    ensure!(
        write_gate_to(&ctx, &pass, &f.x, &written, false, pending).is_err(),
        "gate output accepted legacy policy"
    );
    Ok(())
}
