"""Prepare the isolated generic exact-projection reuse experiment patch."""

from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

RUST = '''
/// Diagnostic reuse of packed weights across independent verifier inputs.
/// The Q4/BF16 path keeps the stock two-output-channel, 16-lane order; Q8
/// keeps the ordinary SIMD32 reduction and requested output precision.
fn gemv_quant_rows_reuse_candidate(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(x.dtype() == DType::BF16 && x.shape().len() == 2,
        "projection reuse needs BF16 rows");
    let m = x.shape()[0];
    ensure!((2..=4).contains(&m) && x.shape()[1] == k && y.numel() == m * n,
        "projection reuse geometry");
    let family = match (w.bits, y.dtype()) {
        (4, DType::BF16) if n.is_multiple_of(2) => "q4_bf16",
        (8, DType::BF16) => "q8_bf16",
        (8, DType::F32) => "q8_f32",
        _ => anyhow::bail!("unsupported projection reuse policy"),
    };
    let name = match (family, m) {
        ("q4_bf16", 2) => "stark_projection_reuse_q4_bf16_m2",
        ("q4_bf16", 3) => "stark_projection_reuse_q4_bf16_m3",
        ("q4_bf16", 4) => "stark_projection_reuse_q4_bf16_m4",
        ("q8_bf16", 2) => "stark_projection_reuse_q8_bf16_m2",
        ("q8_bf16", 3) => "stark_projection_reuse_q8_bf16_m3",
        ("q8_bf16", 4) => "stark_projection_reuse_q8_bf16_m4",
        ("q8_f32", 2) => "stark_projection_reuse_q8_f32_m2",
        ("q8_f32", 3) => "stark_projection_reuse_q8_f32_m3",
        ("q8_f32", 4) => "stark_projection_reuse_q8_f32_m4",
        _ => unreachable!(),
    };
    // Preserve generic Q4/Q8 PlatformDefault arithmetic policy.
    let pipeline = ctx.pipeline(name, DECODE_ROWS_SOURCE, MslVersion::V3_1)?;
    ensure!(pipeline.thread_execution_width() == 32,
        "projection reuse requires SIMD32");
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(),
            x.binding(), y.binding()],
        &[&u32_bytes(k), &u32_bytes(w.group_size), &u32_bytes(n)],
        Grid::Threadgroups {
            groups: (if w.bits == 4 { n / 2 } else { n }, 1, 1),
            threadgroup: (32, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "stark_projection_reuse_tests.rs"]
mod stark_projection_reuse_tests;

'''

METAL = '''
// Diagnostic generic projection reuse. Q4 uses the already-qualified packed
// helper; Q8 retains GEMV_Q8_BODY's per-input expression and SIMD32 reduction.
template <uint M>
inline void stark_projection_reuse_q4_body(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* input,
    device bfloat* output, uint K, uint GS, uint N, uint group, uint lane) {
    const uint sublane = lane % 16;
    const uint channel = group * 2 + lane / 16;
    float sums[M];
    stark_gemv_q4_packed_rows_sum<M>(
        codes, scales, biases, input, channel, sublane, K, GS, sums);
    if (sublane == 0) {
        #pragma clang loop unroll(full)
        for (uint m = 0; m < M; ++m)
            output[(ulong)m * N + channel] = bfloat(sums[m]);
    }
}

template <uint M, typename CT>
inline void stark_projection_reuse_q8_body(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* input,
    device CT* output, uint K, uint GS, uint N, uint row, uint lane) {
    const uint words = K / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
    const uint groups = K / GS;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words);
    float sums[M] = {};
    for (uint i = lane; i < blocks; i += 32) {
        const uint g = i / bpg;
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        const uint4 w4 = wrow[i];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < M; ++m) {
            device const bfloat4* xv =
                (device const bfloat4*)(input + (ulong)m * K);
            sums[m] += dot_word_q8(w4.x, s, b, xv, 4 * i)
                + dot_word_q8(w4.y, s, b, xv, 4 * i + 1)
                + dot_word_q8(w4.z, s, b, xv, 4 * i + 2)
                + dot_word_q8(w4.w, s, b, xv, 4 * i + 3);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0; m < M; ++m) {
        const float sum = simd_sum(sums[m]);
        if (lane == 0) output[(ulong)m * N + row] = CT(sum);
    }
}

#define STARK_PROJECTION_REUSE_Q4(M) \\
kernel void stark_projection_reuse_q4_bf16_m##M( \\
    device const uint* codes [[buffer(0)]], \\
    device const bfloat* scales [[buffer(1)]], \\
    device const bfloat* biases [[buffer(2)]], \\
    device const bfloat* input [[buffer(3)]], \\
    device bfloat* output [[buffer(4)]], \\
    constant uint& K [[buffer(5)]], constant uint& GS [[buffer(6)]], \\
    constant uint& N [[buffer(7)]], \\
    uint group [[threadgroup_position_in_grid]], \\
    uint lane [[thread_index_in_threadgroup]]) { \\
    stark_projection_reuse_q4_body<M>( \\
        codes, scales, biases, input, output, K, GS, N, group, lane); \\
}

#define STARK_PROJECTION_REUSE_Q8(M, CT, SUFFIX) \\
kernel void stark_projection_reuse_q8_##SUFFIX##_m##M( \\
    device const uint* codes [[buffer(0)]], \\
    device const bfloat* scales [[buffer(1)]], \\
    device const bfloat* biases [[buffer(2)]], \\
    device const bfloat* input [[buffer(3)]], \\
    device CT* output [[buffer(4)]], \\
    constant uint& K [[buffer(5)]], constant uint& GS [[buffer(6)]], \\
    constant uint& N [[buffer(7)]], \\
    uint row [[threadgroup_position_in_grid]], \\
    uint lane [[thread_index_in_threadgroup]]) { \\
    stark_projection_reuse_q8_body<M, CT>( \\
        codes, scales, biases, input, output, K, GS, N, row, lane); \\
}
STARK_PROJECTION_REUSE_Q4(2)
STARK_PROJECTION_REUSE_Q4(3)
STARK_PROJECTION_REUSE_Q4(4)
STARK_PROJECTION_REUSE_Q8(2, bfloat, bf16)
STARK_PROJECTION_REUSE_Q8(3, bfloat, bf16)
STARK_PROJECTION_REUSE_Q8(4, bfloat, bf16)
STARK_PROJECTION_REUSE_Q8(2, float, f32)
STARK_PROJECTION_REUSE_Q8(3, float, f32)
STARK_PROJECTION_REUSE_Q8(4, float, f32)
#undef STARK_PROJECTION_REUSE_Q4
#undef STARK_PROJECTION_REUSE_Q8
'''

TEST = '''// SPDX-License-Identifier: Apache-2.0
//! Exact finite component controls for projection reuse; no values are logged.
use super::*;

fn word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
#[ignore = "physical Apple Silicon; actual projection and vocabulary geometry"]
fn projection_reuse_matches_ordinary_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x7a53_98f1;
    let mut compared_bytes = 0usize;
    let mut cases = 0usize;
    for (bits, k, n, dtype, patterns) in [
        (4, 128, 18, DType::BF16, 4),
        (8, 128, 17, DType::BF16, 4),
        (8, 128, 17, DType::F32, 4),
        (4, 6144, 2560, DType::BF16, 1), // GDN output
        (8, 2560, 1280, DType::BF16, 1), // shared gate/up
        (8, 640, 2560, DType::BF16, 1), // shared down
        (8, 2560, 13312, DType::BF16, 1), // attention joined QKV
        (8, 6144, 2560, DType::BF16, 1), // attention output
        (8, 2560, 640, DType::BF16, 1), // attention indexer QK
        (8, 2560, 512, DType::F32, 1), // router
        (8, 2560, 248320, DType::F32, 1), // vocabulary
    ] {
        let storage_n = n + 2;
        let packed = 32 / bits;
        let codes: Vec<u32> = (0..storage_n * k / packed)
            .map(|_| word(&mut seed)).collect();
        let scales: Vec<f32> = (0..storage_n * k / 64)
            .map(|_| (word(&mut seed) % 127 + 1) as f32 / 65536.0).collect();
        let biases: Vec<f32> = (0..storage_n * k / 64)
            .map(|_| (word(&mut seed) % 257) as f32 / 4096.0 - 0.03125).collect();
        let w = QuantWeights {
            codes: Tensor::from_bytes(&ctx, bytemuck::cast_slice(&codes),
                &[storage_n, k / packed], DType::U32)?,
            scales: Tensor::from_f32_as_bf16(&ctx, &scales, &[storage_n, k / 64])?,
            biases: Tensor::from_f32_as_bf16(&ctx, &biases, &[storage_n, k / 64])?,
            group_size: 64, bits,
        }.view_rows(2, n)?;
        for m in 2..=4 {
            for pattern in 0..patterns {
                let values: Vec<f32> = (0..(m + 2) * k).map(|i| match pattern {
                    0 => (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0,
                    1 => if i % 2 == 0 { 0.0 } else { -0.0 },
                    2 => half::bf16::from_bits([1, 0x8001, 0x3f80, 0xbf80][i % 4]).to_f32(),
                    _ => if i % 2 == 0 { 256.0 } else { -256.0 },
                }).collect();
                let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m + 2, k])?
                    .view(k, &[m, k])?;
                let guard = Tensor::zeros(&ctx, &[m * n + 16], dtype)?;
                guard.write_bytes(&vec![0xa5; guard.byte_len()])?;
                let candidate = guard.view(8, &[m, n])?;
                // Separate row allocations admit odd BF16 widths as well.
                let reference: Vec<_> = (0..m).map(|_| Tensor::zeros(&ctx, &[n], dtype))
                    .collect::<Result<_>>()?;
                let pass = ctx.begin_concurrent()?;
                for row in 0..m {
                    gemv_quant(&ctx, &pass, &w, &x.view(row * k, &[k])?,
                        &reference[row])?;
                }
                gemv_quant_rows_reuse_candidate(&ctx, &pass, &w, &x, &candidate)?;
                pass.end()?.commit()?.wait()?;
                let candidate_bytes = candidate.raw_bytes();
                let row_bytes = n * dtype.size();
                ensure!(reference.iter().enumerate().all(|(row, expected)|
                    expected.raw_bytes() == &candidate_bytes[row * row_bytes..(row + 1) * row_bytes]),
                    "projection reuse changed exact output");
                let bytes = guard.raw_bytes();
                let pad = 8 * dtype.size();
                ensure!(bytes[..pad].iter().chain(&bytes[bytes.len() - pad..])
                    .all(|&b| b == 0xa5), "projection reuse changed guards");
                compared_bytes += m * n * dtype.size();
                cases += 1;
            }
        }
    }
    println!("{{\\"projection_reuse\\":\\"exact\\",\\"cases\\":{cases},\\"compared_bytes\\":{compared_bytes}}}");
    Ok(())
}
'''

path = "vendor/lily-core/src/kernels/quant.rs"
original = (ROOT / path).read_text()
needle = '''    include_str!("metal/quant.metal"),
    "\\n",
    include_str!("metal/stark_decode_rows.metal"),'''
replacement = '''    include_str!("metal/quant.metal"),
    "\\n",
    include_str!("metal/stark_packed_rows.metal"),
    "\\n",
    include_str!("metal/stark_decode_rows.metal"),'''
assert original.count(needle) == 1
changed = original.replace(needle, replacement)
needle = '''    ensure!(y.numel() == m * n, "decode row output geometry");
    let requested_rows ='''
replacement = '''    ensure!(y.numel() == m * n, "decode row output geometry");
    // Explicit experimental selector; the production default is unchanged.
    if m > 1
        && (w.bits == 8 || (w.bits == 4 && y.dtype() == DType::BF16 && n.is_multiple_of(2)))
        && matches!(y.dtype(), DType::BF16 | DType::F32)
        && std::env::var_os("STARK_LILY_VERIFY_PROJECTION_REUSE")
            .is_some_and(|value| value == "1")
    {
        return gemv_quant_rows_reuse_candidate(ctx, pass, w, x, y);
    }
    let requested_rows ='''
assert changed.count(needle) == 1
changed = changed.replace(needle, replacement)
anchor = '#[cfg(test)]\n#[path = "quant_decode_rows_tests.rs"]'
assert changed.count(anchor) == 1
changed = changed.replace(anchor, RUST + anchor)

metal_path = "vendor/lily-core/src/kernels/metal/stark_decode_rows.metal"
test_path = "vendor/lily-core/src/kernels/stark_projection_reuse_tests.rs"
metal_old = (ROOT / metal_path).read_text()
changes = [(path, original, changed), (metal_path, metal_old, metal_old + METAL),
           (test_path, "", TEST)]
patch = "".join("".join(difflib.unified_diff(old.splitlines(keepends=True),
    new.splitlines(keepends=True), fromfile="a/" + name if old else "/dev/null",
    tofile="b/" + name)) for name, old, new in changes)
patch_path = OUT / "generic-projection-reuse-diagnostic.patch"
patch_path.write_text(patch)
metadata = {
    "selector": "STARK_LILY_VERIFY_PROJECTION_REUSE=1",
    "default": "unchanged existing row-axis projections",
    "scope": "generic exact 2..4 input-row projections; Q8 F32/BF16 or even-output Q4 BF16",
    "excluded": ["HC fused projections", "GDN input fused projection/conv", "Q4 F32", "odd-output Q4 BF16", "single-row"],
    "production_sources_modified": False,
    "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
    "owning_source_sha256": {name: hashlib.sha256(old.encode()).hexdigest()
                              for name, old, _ in changes if old},
    "numerics": "Q4 existing exact packed-2-output helper and BF16 boundary; Q8 ordinary dot/reduction and requested F32/BF16 output",
    "controls": "60 finite cases, offset inputs/weights/outputs, guards; actual GDNout/shared/attention/indexer/router/head shapes",
    "risks": ["compiler may schedule longer accumulator bodies differently",
              "register pressure may reduce occupancy",
              "existing cache reuse may limit weight traffic savings"],
}
(OUT / "generic-projection-reuse-candidate.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
