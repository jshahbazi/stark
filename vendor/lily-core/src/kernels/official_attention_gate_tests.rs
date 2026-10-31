// Original Stark controls against a private stock-MLX C API receipt.
// Synthetic arrays remain in RAM. Failures expose aggregate counts/hashes.
use super::*;
use crate::sha256::{Sha256, hex};
use half::bf16;

const ADVERSARIAL_BITS: [u16; 32] = [
    0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x3f00, 0xbf00, 0x40db, 0xc0db,
    0x3b80, 0xbb80, 0x0080, 0x8080, 0x0001, 0x8001, 0x7f7f, 0xff7f, 0x4100, 0xc100,
    0x4180, 0xc180, 0x3fc0, 0xbfc0, 0x3e80, 0xbe80, 0x42b0, 0xc2b0, 0x3f81, 0xbf81,
    0x3fff, 0xbfff,
];

fn digest(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finish())
}
fn fixtures() -> Vec<(&'static str, Vec<u16>, Vec<bf16>, Vec<usize>)> {
    let mut out = Vec::new();
    for (name, rows) in [("decode_adversarial", 1usize), ("prefill_adversarial", 3)] {
        let n = rows * 16 * 256;
        let gates = (0..n).map(|i| ADVERSARIAL_BITS[i % 32]).collect();
        let values = (0..n)
            .map(|i| {
                bf16::from_f32(if i % 17 == 0 {
                    -0.0
                } else {
                    ((i * 29 % 257) as i32 - 128) as f32 / 64.0
                })
            })
            .collect();
        out.push((name, gates, values, vec![rows, 16, 256]));
    }
    let gates: Vec<u16> =
        (0u32..65536).map(|i| i as u16).filter(|i| i & 0x7f80 != 0x7f80).collect();
    let values = (0..gates.len())
        .map(|i| {
            bf16::from_f32(if i % 17 == 0 {
                -0.0
            } else {
                ((i * 29 % 257) as i32 - 128) as f32 / 64.0
            })
        })
        .collect();
    let n = gates.len();
    out.push(("all_finite_nonuniform", gates, values, vec![1, n]));
    out
}
fn receipt() -> Result<serde_json::Value> {
    let path = std::env::var_os("STARK_QWEN38_ATTN_GATE_ORACLE").ok_or_else(|| {
        anyhow::anyhow!("set private stock attention gate receipt path")
    })?;
    let path = std::path::Path::new(&path);
    let meta = std::fs::metadata(path)?;
    ensure!(
        meta.is_file() && meta.len() <= 1024 * 1024,
        "invalid private attention gate receipt"
    );
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        value["schema"] == "stark-private-stock-attention-gate-oracle-v1"
            && value["status"] == "complete"
            && value["device"] == "gpu"
            && value["model_loaded"] == false
            && value["raw_arrays_persisted"] == false
            && value["all_repeats_exact"] == true,
        "unqualified stock attention gate receipt"
    );
    ensure!(
        value["native_fixture_sha256"]
            == digest(include_bytes!("official_attention_gate_tests.rs"))
            && value["native_shader_sha256"]
                == digest(include_bytes!("metal/official_attention_gate.metal"))
            && value["table_producer_shader_sha256"]
                == digest(include_bytes!("metal/official_moe.metal")),
        "attention gate native source identity mismatch"
    );
    ensure!(
        value["transformer_sha256"]
            == "b2879bdd3f15bef2607df9aeb205977468fccde0e5f3d9a2ba1010384fe53489"
            && value["stock_asset_sha256"]["lib/libmlx.dylib"]
                == "fc504cec0f86f4a20b111d60dd1112df178bfd7d9b7978d4d735fa1a9c9bf8d4"
            && value["stock_asset_sha256"]["lib/libmlxc.dylib"]
                == "83ebedc48c226376af608b939e17256550d4baa1ad0cc276195c676fdc55c501"
            && value["stock_asset_sha256"]["lib/mlx.metallib"]
                == "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1",
        "attention gate stock reference identity mismatch"
    );
    Ok(value)
}

#[test]
#[ignore = "requires physical Metal and the source-bound private stock gate receipt"]
fn official_attention_gate_matches_stock_bf16_sigmoid_then_product() -> Result<()> {
    let oracle = receipt()?;
    let ctx = MetalContext::new()?;
    let table = crate::kernels::moe::official_moe_sigmoid_table(&ctx)?;
    let finite: Vec<u8> = (0u32..65536)
        .filter(|i| i & 0x7f80 != 0x7f80)
        .flat_map(|i| {
            table.raw_bytes()[i as usize * 2..i as usize * 2 + 2].iter().copied()
        })
        .collect();
    ensure!(
        digest(&finite)
            == "c487ccb3208e0ad603a280dd5b17a28910f9487929a0ba9bdfb1f782b3722b32",
        "reused table no longer matches actual stock all-finite sigmoid"
    );
    let mut mutation_witnesses = 0usize;
    for (name, gates, values, shape) in fixtures() {
        let row = oracle["fixtures"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing gate fixtures"))?
            .iter()
            .find(|r| r["name"] == name)
            .ok_or_else(|| anyhow::anyhow!("missing gate fixture"))?;
        let gb = bytemuck::cast_slice::<u16, u8>(&gates);
        let xb = bytemuck::cast_slice::<bf16, u8>(&values);
        ensure!(
            row["gate_bf16_sha256"] == digest(gb)
                && row["value_bf16_sha256"] == digest(xb),
            "gate input identity mismatch"
        );
        let g = Tensor::from_bytes(&ctx, gb, &shape, DType::BF16)?;
        let x = Tensor::from_bytes(&ctx, xb, &shape, DType::BF16)?;
        let out = Tensor::zeros(&ctx, &shape, DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        sigmoid_mul_bf16_with_rounding(&ctx, &pass, &g, &x, &out, Some(&table), true)?;
        pass.commit_wait()?;
        ensure!(
            row["output_bf16_sha256"] == digest(out.raw_bytes()),
            "attention gate differs from actual stock BF16 primitive/product"
        );
        for ((&bits, x), a) in gates.iter().zip(&values).zip(out.to_f32()?) {
            let g = bf16::from_bits(bits).to_f32();
            if g.abs() < 10.0 && x.to_f32() != 0.0 {
                let wrong = bf16::from_f32(x.to_f32() / (1.0 + (-g).exp())).to_f32();
                mutation_witnesses += usize::from(a.to_bits() != wrong.to_bits());
            }
        }
    }
    ensure!(
        mutation_witnesses > 0,
        "gate fixture does not distinguish old F32 sigmoid-through-product"
    );
    Ok(())
}

#[test]
#[ignore = "requires physical Metal"]
fn official_attention_gate_refuses_f32_missing_table_and_preserves_legacy() -> Result<()>
{
    let ctx = MetalContext::new()?;
    let table = crate::kernels::moe::official_moe_sigmoid_table(&ctx)?;
    let g = Tensor::from_f32_as_bf16(&ctx, &[1.0, -2.0, 0.5, -0.0], &[4])?;
    let x = Tensor::from_f32_as_bf16(&ctx, &[1.25, -0.75, 1.5, -2.0], &[4])?;
    let fg = Tensor::from_f32(&ctx, &[1.0, -2.0, 0.5, -0.0], &[4])?;
    let old = Tensor::zeros(&ctx, &[4], DType::BF16)?;
    let same = Tensor::zeros(&ctx, &[4], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    ensure!(
        sigmoid_mul_bf16_with_rounding(&ctx, &pass, &fg, &x, &same, Some(&table), true)
            .is_err(),
        "official gate accepted promoting F32 input"
    );
    ensure!(
        sigmoid_mul_bf16_with_rounding(&ctx, &pass, &g, &x, &same, None, true).is_err(),
        "official gate accepted missing table"
    );
    sigmoid_mul_bf16(&ctx, &pass, &g, &x, &old)?;
    sigmoid_mul_bf16_with_rounding(&ctx, &pass, &g, &x, &same, None, false)?;
    pass.commit_wait()?;
    ensure!(old.raw_bytes() == same.raw_bytes(), "legacy attention gate changed");
    Ok(())
}
