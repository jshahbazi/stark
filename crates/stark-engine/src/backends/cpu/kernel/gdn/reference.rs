// Pinned PyTorch SumKernel.cpp F32 128-wide reduction and Activation.cpp SiLU.
// Revision 5c4886908584029761b579af026dcfb627c84070; BSD terms/copyrights in
// scripts/licenses/distribution/pytorch-LICENSE. Legacy GDN kernels are separate.
use half::bf16;
use stark_engine_macros::kernel;

use crate::backends::common::vision_math::{F32Pair, vision_exp};

// Positive-range SLEEF log1pf_u10, used only below softplus's x > 20 cutoff.
// SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68 simdsp.c xlog1pf/df.h;
// Boost Software License notice is in scripts/licenses/distribution/sleef-LICENSE.txt.
fn log1p_exp_positive(d: f32) -> f32 {
    let exponent = (((((d + 1.0) * (1.0f32 / 0.75)).to_bits() >> 23) & 255) as i32) - 127;
    let t = f32::from_bits(((127 - exponent) as u32) << 23);
    let m = d.mul_add(t, t - 1.0);
    let mut s = F32Pair {
        hi: 0.6931471824645996f32,
        lo: -1.904654323148236e-9f32,
    }
    .mul_f(exponent as f32);
    let denominator = F32Pair::fast_two_sum(2.0, m);
    let reciprocal = denominator.hi.recip();
    let hi = m * reciprocal;
    let u = reciprocal.mul_add(m, -hi);
    let v = (-denominator.lo).mul_add(reciprocal, (-denominator.hi).mul_add(reciprocal, 1.0));
    let x = F32Pair {
        hi,
        lo: hi.mul_add(v, 0.0f32.mul_add(reciprocal, u)),
    };
    let x2 = x.hi * x.hi;
    let polynomial = 0.3027294874f32.mul_add(x2, 0.3996108174f32).mul_add(x2, 0.6666694880f32);
    let y = F32Pair {
        hi: x.hi * 2.0,
        lo: x.lo * 2.0,
    };
    let sum = s.hi + y.hi;
    s = F32Pair {
        hi: sum,
        lo: (((s.hi - sum) + y.hi) + s.lo) + y.lo,
    };
    s = s.add_f((x2 * x.hi) * polynomial);
    s.hi + s.lo
}

#[kernel(ReferenceGDNDecay)]
pub fn reference_gdn_decay(
    controls: *const bf16,
    a_log: *const f32,
    dt_bias: *const f32,
    output: *mut f32,
    rows: u32,
    heads: u32,
    conv_dim: u32,
    value_dim: u32,
    row_stride: u32,
    write_log: u32,
) {
    for row in 0..rows as usize {
        for head in 0..heads as usize {
            let x = unsafe {
                (*controls
                    .add(row * row_stride as usize + conv_dim as usize + value_dim as usize + heads as usize + head))
                .to_f32()
                    + *dt_bias.add(head)
            };
            let softplus = if x > 20.0 {
                x
            } else {
                log1p_exp_positive(vision_exp(x))
            };
            let g = -vision_exp(unsafe { *a_log.add(head) }) * softplus;
            unsafe {
                *output.add(row * heads as usize + head) = if write_log != 0 {
                    g
                } else {
                    vision_exp(g)
                };
            }
        }
    }
}

unsafe fn square_sum_128(input: *const bf16) -> f32 {
    let mut registers = [[0.0f32; 4]; 4];
    for block in 0..8 {
        for register in 0..4 {
            for lane in 0..4 {
                let x = unsafe { (*input.add(block * 16 + register * 4 + lane)).to_f32() };
                registers[register][lane] += x * x;
            }
        }
    }
    let mut lanes = registers[0];
    for register in 1..4 {
        for lane in 0..4 {
            lanes[lane] += registers[register][lane];
        }
    }
    let mut sum = 0.0f32;
    for lane in lanes {
        sum += lane;
    }
    sum
}

#[kernel(ReferenceGDNQKL2)]
pub fn reference_gdn_qk_l2(
    input: *const bf16,
    q_norm: *mut f32,
    k_norm: *mut f32,
    key_dim: u32,
    rows: u32,
    row_stride: u32,
) {
    for row in 0..rows as usize {
        for head in 0..key_dim as usize / 128 {
            let base = row * row_stride as usize + head * 128;
            let q = unsafe { input.add(base) };
            let k = unsafe { input.add(base + key_dim as usize) };
            let q_inverse = (unsafe { square_sum_128(q) } + 1e-6).sqrt().recip();
            let k_inverse = (unsafe { square_sum_128(k) } + 1e-6).sqrt().recip();
            for channel in 0..128 {
                let output = row * key_dim as usize + head * 128 + channel;
                unsafe {
                    *q_norm.add(output) = ((*q.add(channel)).to_f32() * q_inverse) * 0.0883883461356163f32;
                    *k_norm.add(output) = (*k.add(channel)).to_f32() * k_inverse;
                }
            }
        }
    }
}

#[kernel(ReferenceGDNNormGate)]
pub fn reference_gdn_norm_gate(
    core: *mut bf16,
    controls: *const bf16,
    scales: *const f32,
    rows: u32,
    num_heads: u32,
    value_dim: u32,
    conv_dim: u32,
    row_stride: u32,
    epsilon: f32,
) {
    for row in 0..rows as usize {
        for head in 0..num_heads as usize {
            let base = row * value_dim as usize + head * 128;
            let mean = unsafe { square_sum_128(core.add(base)) } / 128.0;
            let inverse = (mean + epsilon).sqrt().recip();
            for channel in 0..128 {
                unsafe {
                    let normalized = bf16::from_f32((*core.add(base + channel)).to_f32() * inverse).to_f32();
                    let gain = bf16::from_f32(*scales.add(channel)).to_f32();
                    let affine = bf16::from_f32(normalized * gain).to_f32();
                    let z =
                        (*controls.add(row * row_stride as usize + conv_dim as usize + head * 128 + channel)).to_f32();
                    let silu = z / (1.0 + vision_exp(-z));
                    *core.add(base + channel) = bf16::from_f32(affine * silu);
                }
            }
        }
    }
}
