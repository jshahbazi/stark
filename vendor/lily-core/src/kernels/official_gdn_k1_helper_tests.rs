// SPDX-License-Identifier: Apache-2.0
// Exact actual-stock custom K1 hashes; aggregate-only synthetic diagnostics.
use super::*;
use crate::sha256::{hex, sha256};

#[test]
fn official_gdn_k1_integer_barriers_match_stock_entire_finite_bf16_domain() -> Result<()>
{
    // Actual CAPI report 7ad8bed3b1b8397758a26b4a96668f8d4366642e3d9a68956d3b78b70157ba55.
    // All-finite K1 custom sigmoid differs from primitive on exactly -6.84375;
    // therefore matching the older 32-value primitive control is insufficient.
    let bits: Vec<u16> = (0..=u16::MAX).filter(|v| v & 0x7f80 != 0x7f80).collect();
    ensure!(bits.len() == 65280, "finite BF16 domain count");
    let input: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    ensure!(
        hex(&sha256(&input))
            == "81100c8586b90228fcdfc535d5f2dc1222be8d798bad124b03afd8df05858fc2",
        "finite BF16 domain hash"
    );
    let ctx = MetalContext::new()?;
    let x = Tensor::from_bytes(&ctx, &input, &[bits.len()], DType::BF16)?;
    let outputs = (0..6)
        .map(|_| Tensor::zeros(&ctx, &[bits.len()], DType::BF16))
        .collect::<Result<Vec<_>>>()?;
    let pipeline = ctx.pipeline(
        "official_gdn_k1_helper_exhaustive",
        include_str!("metal/official_gdn_k1_helper.metal"),
        MslVersion::V3_1,
    )?;
    let pass = ctx.begin_concurrent()?;
    pass.dispatch_at(
        &pipeline,
        &[
            x.binding(),
            outputs[0].binding(),
            outputs[1].binding(),
            outputs[2].binding(),
            outputs[3].binding(),
            outputs[4].binding(),
            outputs[5].binding(),
        ],
        &[],
        Grid::Threads { grid: (bits.len(), 1, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.commit()?.wait()?;
    let hashes: Vec<String> =
        outputs.iter().map(|t| hex(&sha256(t.raw_bytes()))).collect();
    let expected_sig =
        "6d8c7069994a4298cd6aaf5e4ab8ef5464176df027e072082314de5ec6e7bd6b";
    let expected_silu =
        "b35ebfe3e2574fa39079c24c08ea3d6139682976980da7c1c19759dbc36391bf";
    // Include all six aggregate hashes on failure, without printing inputs or
    // output arrays. Diagnostic variants never substitute for MODE0 acceptance.
    ensure!(
        hashes[0] == expected_sig && hashes[1] == expected_silu,
        "K1 all-finite helper mismatch: default_sig_sha256={} default_silu_sha256={} precise_sig_sha256={} precise_silu_sha256={} fast_sig_sha256={} fast_silu_sha256={}",
        hashes[0],
        hashes[1],
        hashes[2],
        hashes[3],
        hashes[4],
        hashes[5]
    );
    Ok(())
}
