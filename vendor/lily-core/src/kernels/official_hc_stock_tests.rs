// SPDX-License-Identifier: Apache-2.0
// Synthetic native controls bound to actual stock C API outputs.
// Stock receipt SHA256: 0053013d874c0082e2a4bfe48770f1a1176a006c9d296f67587af92b5ace818b
// Stock controller SHA256: 3d1b30414a7a23959ad0c214e3bc802f47b634b9a47c80797fef5e7ceb337927
// Arrays stay in RAM. Diagnostics contain hashes/counts only.
use super::*;
use crate::sha256::{Sha256, hex};

include!("official_hc_stock_pending_tests.rs");

fn digest(raw: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(raw);
    hex(&h.finish())
}

#[derive(Default)]
struct StageChecks {
    checked: usize,
    failed: usize,
}
impl StageChecks {
    fn record(
        &mut self,
        raw: &[u8],
        expected: &str,
        seed: u32,
        inject: bool,
        stage: &str,
    ) {
        let actual = digest(raw);
        let matched = actual == expected;
        self.checked += 1;
        if !matched {
            self.failed += 1;
        }
        // Aggregate identities only; never emit or persist synthetic arrays.
        println!(
            "{}",
            serde_json::json!({
                "scope": "stock_hc_stage_hash",
                "seed": seed, "inject": inject, "stage": stage,
                "bytes": raw.len(), "actual_sha256": actual,
                "expected_sha256": expected, "exact": matched,
            })
        );
    }
}

fn equal(a: &[u8], b: &[u8], stage: &str) -> Result<()> {
    ensure!(
        a == b,
        "{stage}: actual_sha256={}, expected_sha256={}",
        digest(a),
        digest(b),
    );
    Ok(())
}

struct Rng(u32);
impl Rng {
    fn word(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn bf16(&mut self, n: usize, scale: f64, offset: f64) -> Vec<bf16> {
        (0..n)
            .map(|_| {
                let frac = (self.word() % 4096) as f64 / 4096.0;
                // Match the source-bound Python fixture's F64 affine -> F32 -> BF16.
                bf16::from_f32((frac * scale + offset) as f32)
            })
            .collect()
    }
    fn codes(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.word()).collect()
    }
}

struct Fixture {
    x: Tensor,
    norm: Tensor,
    down: QuantWeights,
    up: QuantWeights,
    inject: QuantWeights,
    hashes: Vec<(&'static str, String)>,
}

fn fixture(ctx: &MetalContext, seed: u32) -> Result<Fixture> {
    let mut rng = Rng(seed);
    let x = rng.bf16(K, 4.0, -2.0);
    let norm = rng.bf16(K, 1.0, 0.5);
    let dc = rng.codes(R * K / 4);
    let ds = rng.bf16(R * K / 64, 0.0095, 0.0005);
    let db = rng.bf16(R * K / 64, 1.0, -0.5);
    let uc = rng.codes(K * R / 4);
    let us = rng.bf16(K * R / 64, 0.0095, 0.0005);
    let ub = rng.bf16(K * R / 64, 1.0, -0.5);
    let iw = rng.bf16(G * K, 0.1, -0.05);
    let it: Vec<bf16> = (0..K)
        .flat_map(|k| {
            let iw = &iw;
            (0..G).map(move |g| iw[g * K + k])
        })
        .collect();
    let hashes = vec![
        ("x", digest(bytemuck::cast_slice(&x))),
        ("nw", digest(bytemuck::cast_slice(&norm))),
        ("dw", digest(bytemuck::cast_slice(&dc))),
        ("ds", digest(bytemuck::cast_slice(&ds))),
        ("db", digest(bytemuck::cast_slice(&db))),
        ("uw", digest(bytemuck::cast_slice(&uc))),
        ("us", digest(bytemuck::cast_slice(&us))),
        ("ub", digest(bytemuck::cast_slice(&ub))),
        ("iw_native", digest(bytemuck::cast_slice(&iw))),
        ("iw_transpose", digest(bytemuck::cast_slice(&it))),
    ];
    let q = |n: usize, k: usize, codes: &[u32], scales: &[bf16], biases: &[bf16]| {
        // Nonzero aligned row views also exercise descriptor buffer offsets.
        let mut padded = vec![0u32; 2 * k / 4];
        padded.extend_from_slice(codes);
        Ok::<_, anyhow::Error>(QuantWeights {
            codes: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&padded),
                &[n + 2, k / 4],
                DType::U32,
            )?
            .view(2 * k / 4, &[n, k / 4])?,
            scales: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(scales),
                &[n, k / 64],
                DType::BF16,
            )?,
            biases: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(biases),
                &[n, k / 64],
                DType::BF16,
            )?,
            group_size: 64,
            bits: 8,
        })
    };
    Ok(Fixture {
        x: Tensor::from_bytes(ctx, bytemuck::cast_slice(&x), &[1, K], DType::BF16)?,
        norm: Tensor::from_bytes(ctx, bytemuck::cast_slice(&norm), &[K], DType::BF16)?,
        down: q(R, K, &dc, &ds, &db)?,
        up: q(K, R, &uc, &us, &ub)?,
        inject: QuantWeights::bf16(Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&iw),
            &[G, K],
            DType::BF16,
        )?)?,
        hashes,
    })
}

struct Guarded {
    storage: Tensor,
    value: Tensor,
}
impl Guarded {
    fn new(ctx: &MetalContext, n: usize, dtype: DType) -> Result<Self> {
        let storage = Tensor::zeros(ctx, &[n + 8], dtype)?;
        storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
        let value = storage.view(4, &[1, n])?;
        Ok(Self { storage, value })
    }
    fn check(&self) -> Result<()> {
        let pad = 4 * self.storage.dtype().size();
        let bytes = self.storage.raw_bytes();
        ensure!(
            bytes[..pad].iter().chain(&bytes[bytes.len() - pad..]).all(|b| *b == 0xa5),
            "stock HC kernel wrote outside output view"
        );
        Ok(())
    }
}

fn valid_parts(parts: &Tensor, inject: bool) -> Vec<u8> {
    let raw = parts.raw_bytes();
    (0..G)
        .flat_map(|g| {
            let start = if inject { (g * (R + G) + R) * 4 } else { g * (R + G) * 4 };
            let len = if inject { G * 4 } else { R * 4 };
            raw[start..start + len].iter().copied()
        })
        .collect()
}

#[test]
fn official_stock_read_matches_actual_stock_stage_hashes_and_owns_buffers() -> Result<()>
{
    let ctx = MetalContext::new()?;
    let goldens: serde_json::Value = serde_json::from_str(STOCK_GOLDENS)?;
    let mut checks = StageChecks::default();
    for row in goldens.as_array().ok_or_else(|| anyhow::anyhow!("golden shape"))? {
        let seed =
            row["seed"].as_u64().ok_or_else(|| anyhow::anyhow!("golden seed"))? as u32;
        let with_inject =
            row["inject"].as_bool().ok_or_else(|| anyhow::anyhow!("golden inject"))?;
        let f = fixture(&ctx, seed)?;
        for (name, got) in &f.hashes {
            ensure!(
                row["fixture_input_hashes"][*name].as_str() == Some(got.as_str()),
                "stock HC fixture identity mismatch: {name}, sha256={got}"
            );
        }
        let prepared = OfficialHcReadWeights::prepare(
            &ctx,
            &f.norm,
            &f.down,
            &f.up,
            with_inject.then_some(&f.inject),
            H,
            G,
            1e-6,
            true,
        )?
        .ok_or_else(|| anyhow::anyhow!("stock HC descriptor declined"))?;
        ensure!(
            prepared.selected_for(true, 1, true),
            "one-row stock descriptor declined"
        );
        ensure!(
            !prepared.selected_for(true, 2, true)
                && !prepared.selected_for(true, 17, true)
                && !prepared.selected_for(false, 1, true),
            "stock read admitted fallback rows/policy"
        );
        // Derived tensors must retain their own codes/up handles after the
        // original projection descriptors are dropped.
        let Fixture { x, norm, down, up, inject, hashes: _ } = f;
        drop(down);
        drop(up);
        drop(inject);
        let normalized = Guarded::new(&ctx, K, DType::BF16)?;
        let parts = Guarded::new(&ctx, PARTS, DType::F32)?;
        let mixed = Guarded::new(&ctx, H, DType::BF16)?;
        let gate = Guarded::new(&ctx, G, DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        prepared.read(
            &ctx,
            &pass,
            &x,
            &norm,
            &normalized.value,
            &parts.value,
            &mixed.value,
            &gate.value,
        )?;
        pass.commit()?.wait()?;
        let stage = &row["stages"];
        for (name, raw) in [
            ("normalized_bf16", normalized.value.raw_bytes().to_vec()),
            ("down_stream_partials_f32", valid_parts(&parts.value, false)),
            ("mixed_bf16", mixed.value.raw_bytes().to_vec()),
        ] {
            let expected = stage[name]["sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("golden stage"))?;
            checks.record(&raw, expected, seed, with_inject, name);
        }
        if with_inject {
            checks.record(
                &valid_parts(&parts.value, true),
                stage["inject_stream_partials_f32"]["sha256"].as_str().unwrap(),
                seed,
                with_inject,
                "inject_stream_partials_f32",
            );
            checks.record(
                gate.value.raw_bytes(),
                stage["inject_gate_bf16"]["sha256"].as_str().unwrap(),
                seed,
                with_inject,
                "inject_gate_bf16",
            );
        } else {
            ensure!(
                gate.value.raw_bytes().iter().all(|b| *b == 0xa5),
                "no-inject output was written"
            );
            ensure!(
                valid_parts(&parts.value, true).iter().all(|b| *b == 0xa5),
                "unused no-inject partials were written"
            );
        }
        for tensor in [&normalized, &parts, &mixed, &gate] {
            tensor.check()?;
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "scope": "stock_hc_stage_hash_summary", "checked": checks.checked,
            "failed": checks.failed, "all_exact": checks.failed == 0,
        })
    );
    ensure!(checks.checked == 16, "not all stock HC stages were checked");
    ensure!(
        checks.failed == 0,
        "stock HC stage hashes differ: failed_stages={}, checked_stages={}",
        checks.failed,
        checks.checked
    );
    Ok(())
}

#[test]
fn official_stock_gate_write_preserves_frames_and_raw_logit_fallback() -> Result<()> {
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
    .ok_or_else(|| anyhow::anyhow!("descriptor declined"))?;
    let normalized = Tensor::zeros(&ctx, &[1, K], DType::BF16)?;
    let parts = Tensor::zeros(&ctx, &[1, PARTS], DType::F32)?;
    let mixed = Tensor::zeros(&ctx, &[1, H], DType::BF16)?;
    let gate = Tensor::zeros(&ctx, &[1, G], DType::BF16)?;
    let branch_values: Vec<f32> =
        (0..H).map(|i| ((i % 97) as f32 - 48.0) / 32.0).collect();
    let branch = Tensor::from_f32_as_bf16(&ctx, &branch_values, &[1, H])?;
    let hyper = Tensor::zeros(&ctx, &[1, K], DType::BF16)?;
    hyper.write_bytes(f.x.raw_bytes())?;
    for _frame in 0..2 {
        // Carry the prior frame's residual into the next full HC read.
        let original = hyper.to_f32()?;
        let pass = ctx.begin_concurrent()?;
        prepared.read(
            &ctx,
            &pass,
            &hyper,
            &f.norm,
            &normalized,
            &parts,
            &mixed,
            &gate,
        )?;
        pass.level_barrier(&[&gate, &mixed])?;
        inject_after_read(
            &ctx,
            &pass,
            &hyper,
            &branch,
            &gate,
            H,
            G,
            true,
            false,
            InjectionInputKind::PrecomputedGate,
        )?;
        pass.commit()?.wait()?;
        let gate_values = gate.to_f32()?;
        let branch_values = branch.to_f32()?;
        let expected: Vec<bf16> = (0..K)
            .map(|j| {
                let delta = bf16::from_f32(branch_values[j % H] * gate_values[j / H]);
                bf16::from_f32(original[j] + delta.to_f32())
            })
            .collect();
        equal(
            hyper.raw_bytes(),
            bytemuck::cast_slice(&expected),
            "prepared gate frame write",
        )?;
        // A later raw-logit row must retain the old operation chain. The
        // explicit enum makes this independent of earlier gate buffer contents.
        let legacy = Tensor::zeros(&ctx, &[2, K], DType::BF16)?;
        let through_wrapper = Tensor::zeros(&ctx, &[2, K], DType::BF16)?;
        let repeated_hyper = [f.x.raw_bytes(), f.x.raw_bytes()].concat();
        legacy.write_bytes(&repeated_hyper)?;
        through_wrapper.write_bytes(&repeated_hyper)?;
        let repeated_branch = Tensor::from_bytes(
            &ctx,
            &[branch.raw_bytes(), branch.raw_bytes()].concat(),
            &[2, H],
            DType::BF16,
        )?;
        let logits = Tensor::from_f32_as_bf16(
            &ctx,
            &[-1.25, 0.5, 1.75, -0.125, 0.75, -0.25, -1.0, 1.5],
            &[2, G],
        )?;
        let pass = ctx.begin_concurrent()?;
        super::super::hc_inject_bf16_with_rounding(
            &ctx,
            &pass,
            &legacy,
            &repeated_branch,
            &logits,
            H,
            G,
            true,
            false,
        )?;
        inject_after_read(
            &ctx,
            &pass,
            &through_wrapper,
            &repeated_branch,
            &logits,
            H,
            G,
            true,
            false,
            InjectionInputKind::RawLogits,
        )?;
        pass.commit()?.wait()?;
        equal(
            through_wrapper.raw_bytes(),
            legacy.raw_bytes(),
            "raw-logit fallback after prepared gate",
        )?;
    }
    Ok(())
}

#[test]
fn official_stock_verifier_rows_match_decode_and_finished_gate_write() -> Result<()> {
    let ctx = MetalContext::new()?;
    let f = fixture(&ctx, 0x48435631)?;
    let base = f.x.to_f32()?;
    let mut compared_rows = 0;
    for inject in [false, true] {
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
        .ok_or_else(|| anyhow::anyhow!("verifier stock HC descriptor declined"))?;
        for rows in 1..=4 {
            let input: Vec<f32> = (0..rows * K)
                .map(|i| {
                    let row = i / K;
                    base[i % K] * (1.0 + row as f32 * 0.125)
                        + ((i * 7 % 17) as f32 - 8.0) / 64.0
                })
                .collect();
            let hyper = Tensor::from_f32_as_bf16(&ctx, &input, &[rows, K])?;
            let reference_hyper = Tensor::from_f32_as_bf16(&ctx, &input, &[rows, K])?;
            let branch_values: Vec<f32> = (0..rows * H)
                .map(|i| (((i * 11 + i / H * 31) % 101) as i32 - 50) as f32 / 32.0)
                .collect();
            let branch = Tensor::from_f32_as_bf16(&ctx, &branch_values, &[rows, H])?;
            let normalized = Guarded::new(&ctx, rows * K, DType::BF16)?;
            let parts = Guarded::new(&ctx, rows * PARTS, DType::F32)?;
            let mixed = Guarded::new(&ctx, rows * H, DType::BF16)?;
            let gate = Guarded::new(&ctx, rows * G, DType::BF16)?;
            for value in [&normalized.value, &parts.value, &mixed.value, &gate.value] {
                value.write_bytes(&vec![0; value.byte_len()])?;
            }
            let rn = Tensor::zeros(&ctx, &[rows, K], DType::BF16)?;
            let rp = Tensor::zeros(&ctx, &[rows, PARTS], DType::F32)?;
            let rm = Tensor::zeros(&ctx, &[rows, H], DType::BF16)?;
            let rg = Tensor::zeros(&ctx, &[rows, G], DType::BF16)?;
            let pass = ctx.begin_concurrent()?;
            prepared.read_rows(
                &ctx,
                &pass,
                &hyper,
                &f.norm,
                &normalized.value,
                &parts.value,
                &mixed.value,
                &gate.value,
            )?;
            for row in 0..rows {
                prepared.read(
                    &ctx,
                    &pass,
                    &reference_hyper.view(row * K, &[1, K])?,
                    &f.norm,
                    &rn.view(row * K, &[1, K])?,
                    &rp.view(row * PARTS, &[1, PARTS])?,
                    &rm.view(row * H, &[1, H])?,
                    &rg.view(row * G, &[1, G])?,
                )?;
            }
            pass.level_barrier(&[&mixed.value, &gate.value, &rm, &rg])?;
            inject_after_read(
                &ctx,
                &pass,
                &hyper,
                &branch,
                &gate.value,
                H,
                G,
                true,
                false,
                InjectionInputKind::PrecomputedGate,
            )?;
            for row in 0..rows {
                inject_after_read(
                    &ctx,
                    &pass,
                    &reference_hyper.view(row * K, &[1, K])?,
                    &branch.view(row * H, &[1, H])?,
                    &rg.view(row * G, &[1, G])?,
                    H,
                    G,
                    true,
                    false,
                    InjectionInputKind::PrecomputedGate,
                )?;
            }
            pass.commit()?.wait()?;
            for (candidate, reference, label) in [
                (&normalized.value, &rn, "verifier normalized rows"),
                (&parts.value, &rp, "verifier HC partial rows"),
                (&mixed.value, &rm, "verifier HC mixed rows"),
                (&gate.value, &rg, "verifier HC finished gate rows"),
                (&hyper, &reference_hyper, "verifier finished-gate row write"),
            ] {
                equal(candidate.raw_bytes(), reference.raw_bytes(), label)?;
            }
            for guarded in [&normalized, &parts, &mixed, &gate] {
                guarded.check()?;
            }
            compared_rows += rows;
        }
    }
    println!(
        "official_hc_exact_verifier compared_rows={compared_rows} differing_bytes=0"
    );
    Ok(())
}

#[test]
fn official_stock_read_declines_other_descriptors_and_rejects_multirow_buffers()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let f = fixture(&ctx, 0x48434631)?;
    ensure!(
        OfficialHcReadWeights::prepare(
            &ctx,
            &f.norm,
            &f.down,
            &f.up,
            Some(&f.inject),
            H,
            G,
            1e-6,
            false,
        )?
        .is_none(),
        "legacy policy created a stock descriptor"
    );
    ensure!(
        OfficialHcReadWeights::prepare(
            &ctx,
            &f.norm,
            &f.down,
            &f.up,
            Some(&f.inject),
            H / 2,
            G,
            1e-6,
            true,
        )?
        .is_none(),
        "incompatible geometry created a stock descriptor"
    );
    let bad_storage = Tensor::zeros(&ctx, &[K * R / 4 + 1], DType::U32)?;
    let bad_offset_up = QuantWeights {
        codes: bad_storage.view(1, &[K, R / 4])?,
        scales: f.up.scales.clone(),
        biases: f.up.biases.clone(),
        group_size: 64,
        bits: 8,
    };
    ensure!(
        OfficialHcReadWeights::prepare(
            &ctx,
            &f.norm,
            &f.down,
            &bad_offset_up,
            None,
            H,
            G,
            1e-6,
            true,
        )?
        .is_none(),
        "unaligned otherwise-valid up descriptor selected vectorized stock HC"
    );
    let prepared = OfficialHcReadWeights::prepare(
        &ctx, &f.norm, &f.down, &f.up, None, H, G, 1e-6, true,
    )?
    .ok_or_else(|| anyhow::anyhow!("final no-inject descriptor declined"))?;
    let x = Tensor::zeros(&ctx, &[2, K], DType::BF16)?;
    let normalized = Tensor::zeros(&ctx, &[1, K], DType::BF16)?;
    let parts = Tensor::zeros(&ctx, &[1, PARTS], DType::F32)?;
    let mixed = Tensor::zeros(&ctx, &[1, H], DType::BF16)?;
    let gate = Tensor::zeros(&ctx, &[1, G], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    ensure!(
        prepared
            .read(&ctx, &pass, &x, &f.norm, &normalized, &parts, &mixed, &gate)
            .is_err(),
        "multirow stock read was accepted"
    );
    ensure!(
        inject_after_read(
            &ctx,
            &pass,
            &x,
            &mixed,
            &gate,
            H,
            G,
            true,
            false,
            InjectionInputKind::PrecomputedGate
        )
        .is_err(),
        "mismatched prepared-gate write rows were accepted"
    );
    pass.commit()?.wait()?;
    Ok(())
}

#[test]
fn skipped_hc_read_preserves_raw_logits_without_process_environment() -> Result<()> {
    // Reproduce the decode early-return condition with an otherwise admitted
    // official one-row descriptor, without touching OnceLock environment state.
    let admitted = full_read_selected(true, 1, true, true);
    ensure!(
        InjectionInputKind::from_full_read(admitted)
            == InjectionInputKind::PrecomputedGate,
        "enabled admitted read lost its finished-gate contract"
    );
    let skipped = full_read_selected(true, 1, true, false);
    ensure!(
        !skipped
            && InjectionInputKind::from_full_read(skipped)
                == InjectionInputKind::RawLogits,
        "skipped HC read reinterpreted an untouched injection buffer as a gate"
    );
    for (official, rows, prepared, read_enabled) in [
        (true, 2, true, true),
        (true, 17, true, true),
        (false, 1, true, true),
        (true, 1, false, true),
        (true, 2, true, false),
    ] {
        let selected = full_read_selected(official, rows, prepared, read_enabled);
        ensure!(
            InjectionInputKind::from_full_read(selected)
                == InjectionInputKind::RawLogits,
            "fallback HC read selected finished-gate injection"
        );
    }
    Ok(())
}

const STOCK_GOLDENS: &str = r###"[{"seed":1212368433,"inject":false,"fixture_input_hashes":{"db":"8c2fd62109f62747eac5e7b57703403c96768e966c8db6ecb290b9e579c988e5","ds":"f4489a24511a9a570ebfbf3f849ba4e7525418f5f18a30d6fda8834f1d76d09d","dw":"fde99a66e51beaa9ce6e6cbb7fac52fcb8134fd4940d7dc683c530f26d5f6901","iw_native":"6e88d0a56796067ad9c1d9879356dc2f4a37cf14c3288a9de6d23282bb00680c","iw_transpose":"b624dac2784575844d7f322e6415788e6e096b97b3268e4ef6e540c7459e3a3c","nw":"f66e7499b37a12b122e0b83fba2e99e46008e7851ba388a1213f8807c0f77076","ub":"a05184a9cdebc3569c03afa8f90da25b4d5da00d5d177d12c1a975574e4ac48f","us":"e6628e3e2801b91a43f1ca4e328d78027ad94e868bddd775a1ebb169487ad881","uw":"ab66d7c30cd93208fd82e835f79f425e58741f2a18e9f97450b59805d5f513d1","x":"026722e34a96eef6b3807c57be893203b58673cf128284f626d50d86025fd92a"},"stages":{"down_stream_partials_f32":{"sha256":"200ec53714ae07f301bcaa7cdb9dc529ad24cd5092326c599c88043e9ecf57a0"},"mixed_bf16":{"sha256":"bbd7da04189751a8e501c1dabdd8f43e96ea7627d4437b8b7d045e7c3a73e3e9"},"normalized_bf16":{"sha256":"a48e69a7e30e3bd1e1f08055983a63d9f48b20b644db877818a7f17b0e6617dc"}}},{"seed":1212368433,"inject":true,"fixture_input_hashes":{"db":"8c2fd62109f62747eac5e7b57703403c96768e966c8db6ecb290b9e579c988e5","ds":"f4489a24511a9a570ebfbf3f849ba4e7525418f5f18a30d6fda8834f1d76d09d","dw":"fde99a66e51beaa9ce6e6cbb7fac52fcb8134fd4940d7dc683c530f26d5f6901","iw_native":"6e88d0a56796067ad9c1d9879356dc2f4a37cf14c3288a9de6d23282bb00680c","iw_transpose":"b624dac2784575844d7f322e6415788e6e096b97b3268e4ef6e540c7459e3a3c","nw":"f66e7499b37a12b122e0b83fba2e99e46008e7851ba388a1213f8807c0f77076","ub":"a05184a9cdebc3569c03afa8f90da25b4d5da00d5d177d12c1a975574e4ac48f","us":"e6628e3e2801b91a43f1ca4e328d78027ad94e868bddd775a1ebb169487ad881","uw":"ab66d7c30cd93208fd82e835f79f425e58741f2a18e9f97450b59805d5f513d1","x":"026722e34a96eef6b3807c57be893203b58673cf128284f626d50d86025fd92a"},"stages":{"down_stream_partials_f32":{"sha256":"200ec53714ae07f301bcaa7cdb9dc529ad24cd5092326c599c88043e9ecf57a0"},"inject_gate_bf16":{"sha256":"537ae55ab8ecc8cba3ad17f307358c792e3e82ddefbb1f34da714eda916724a4"},"inject_stream_partials_f32":{"sha256":"e09cbe43c69bbdca43f22a1d818d54447a958febf94e5fc8e4f1ccfb57f30821"},"mixed_bf16":{"sha256":"bbd7da04189751a8e501c1dabdd8f43e96ea7627d4437b8b7d045e7c3a73e3e9"},"normalized_bf16":{"sha256":"a48e69a7e30e3bd1e1f08055983a63d9f48b20b644db877818a7f17b0e6617dc"}}},{"seed":1212368434,"inject":false,"fixture_input_hashes":{"db":"9704098a630d8fb2151d9605475e511bac1ff6330ab2e6e80440caa59778b984","ds":"df68b4ee92336b4b1185bf11b53fe7910345df073a12f04faf9e9c06a4d61ec8","dw":"8124d17c792424f527f6b9c579bb2c03c5cec3dcf69c41b8a7829128dd043898","iw_native":"f2b45741879dc052b98fd0d6984d18975b13421a8fec923e4cd9945f70b7b639","iw_transpose":"41802d0d89266e3d3465ac5918929c0991157672134b207a10c25f651ab4f75d","nw":"748684e29284ff83ad1e65bdef61ca668ae80fcef331e9f62b689abe1f962388","ub":"52bfd4b45749985a3f4efa3160e92a065ee9281018443b34737f198a2b7f1143","us":"ad0a06e857efbeff5d4b3de676e2b3b79783ae188b802b4c57e003d0fbfb4fba","uw":"d9e7ba54dcfef18636a14e3f0ba64578335b793cf844c7e189ca9c53cbe9f9ba","x":"31a990119063156dee578fb444bd5a968d51fc03feb613d76b7abf544e419560"},"stages":{"down_stream_partials_f32":{"sha256":"c863b221a02bc3f0fd403ac1fddff0bd549c0dd40e06bd7f0948b84a9f965f39"},"mixed_bf16":{"sha256":"87031efe5f8b7801905e76ebc79109c34278df816d71491b03ae5c3fe5970fce"},"normalized_bf16":{"sha256":"c9f6d361c99d5534f93fdca006fe9a84fd8613740c37a65fe9308bbac35858db"}}},{"seed":1212368434,"inject":true,"fixture_input_hashes":{"db":"9704098a630d8fb2151d9605475e511bac1ff6330ab2e6e80440caa59778b984","ds":"df68b4ee92336b4b1185bf11b53fe7910345df073a12f04faf9e9c06a4d61ec8","dw":"8124d17c792424f527f6b9c579bb2c03c5cec3dcf69c41b8a7829128dd043898","iw_native":"f2b45741879dc052b98fd0d6984d18975b13421a8fec923e4cd9945f70b7b639","iw_transpose":"41802d0d89266e3d3465ac5918929c0991157672134b207a10c25f651ab4f75d","nw":"748684e29284ff83ad1e65bdef61ca668ae80fcef331e9f62b689abe1f962388","ub":"52bfd4b45749985a3f4efa3160e92a065ee9281018443b34737f198a2b7f1143","us":"ad0a06e857efbeff5d4b3de676e2b3b79783ae188b802b4c57e003d0fbfb4fba","uw":"d9e7ba54dcfef18636a14e3f0ba64578335b793cf844c7e189ca9c53cbe9f9ba","x":"31a990119063156dee578fb444bd5a968d51fc03feb613d76b7abf544e419560"},"stages":{"down_stream_partials_f32":{"sha256":"c863b221a02bc3f0fd403ac1fddff0bd549c0dd40e06bd7f0948b84a9f965f39"},"inject_gate_bf16":{"sha256":"b5837964c41e4a25fa2b79ad237c5068a4d02e11df9032085693b208a32a6ca8"},"inject_stream_partials_f32":{"sha256":"30b9ca5c4848e19adef98a7b62a55ec390c8ee34d297fa8fb4014d5c19246d36"},"mixed_bf16":{"sha256":"87031efe5f8b7801905e76ebc79109c34278df816d71491b03ae5c3fe5970fce"},"normalized_bf16":{"sha256":"c9f6d361c99d5534f93fdca006fe9a84fd8613740c37a65fe9308bbac35858db"}}}]"###;
