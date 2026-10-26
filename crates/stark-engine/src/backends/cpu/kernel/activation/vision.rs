// Vision-only numerical compatibility with PyTorch v2.14.1 Gelu.h and the
// pinned SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68 tanhf_u10/df helpers.
// BSD/Boost notices are retained in scripts/licenses/distribution.

use half::bf16;
use stark_engine_macros::kernel;

use crate::backends::common::vision_math::F32Pair;

fn pair_add(
    x: F32Pair,
    y: F32Pair,
) -> F32Pair {
    let hi = x.hi + y.hi;
    F32Pair {
        hi,
        lo: (((x.hi - hi) + y.hi) + x.lo) + y.lo,
    }
}

fn reciprocal(d: F32Pair) -> F32Pair {
    let hi = d.hi.recip();
    F32Pair {
        hi,
        lo: hi * (-d.lo).mul_add(hi, (-d.hi).mul_add(hi, 1.0)),
    }
}

fn divide(
    n: F32Pair,
    d: F32Pair,
) -> F32Pair {
    let t = d.hi.recip();
    let hi = n.hi * t;
    let u = t.mul_add(n.hi, -hi);
    let v = (-d.lo).mul_add(t, (-d.hi).mul_add(t, 1.0));
    F32Pair {
        hi,
        lo: hi.mul_add(v, n.lo.mul_add(t, u)),
    }
}

fn exp_pair(d: F32Pair) -> F32Pair {
    let q = ((d.hi + d.lo) * 1.4426950408889634f32).round_ties_even();
    let mut s = d.add2_f(q * -0.693145751953125f32);
    s = s.add2_f(q * -1.428606765330187e-6f32);
    let mut u = 0.1980960224e-3f32;
    u = u.mul_add(s.hi, 0.1394256484e-2f32);
    u = u.mul_add(s.hi, 0.8333456703e-2f32);
    u = u.mul_add(s.hi, 0.4166637361e-1f32);
    let mut t = s.mul_f(u).add2_f(0.16666665941423424f32);
    t = s.mul(t).add2_f(0.5);
    t = s.add2(s.square().mul(t));
    t = F32Pair::f_add(1.0, t);
    // tanh calls this only for |x| <= 8.66434; scaling is exact and normal.
    let scale = f32::from_bits(((q as i32 + 127) as u32) << 23);
    F32Pair {
        hi: t.hi * scale,
        lo: t.lo * scale,
    }
}

fn tanh(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x.abs() > 8.664339742f32 {
        return 1.0f32.copysign(x);
    }
    let d = exp_pair(F32Pair {
        hi: x.abs(),
        lo: 0.0,
    });
    let e = reciprocal(d);
    let numerator = pair_add(
        d,
        F32Pair {
            hi: -e.hi,
            lo: -e.lo,
        },
    );
    let result = divide(numerator, pair_add(d, e));
    (result.hi + result.lo).copysign(x)
}

// The pinned NEON erf uses this Arm polynomial, rather than scalar libm erff.
fn erf_exp(x: f32) -> f32 {
    if x < -87.33654f32 {
        return 0.0;
    }
    let n = (x * 1.4426950408889634f32).round();
    let r = (-n).mul_add(f32::from_bits(0x3f31_7200), x);
    let r = (-n).mul_add(f32::from_bits(0x35bf_be8e), r);
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    let r2 = r * r;
    let p = r.mul_add(f32::from_bits(0x3c07_2010), f32::from_bits(0x3d2b_9f17));
    let q = r.mul_add(f32::from_bits(0x3e2a_af33), f32::from_bits(0x3eff_fedb));
    let q = p.mul_add(r2, q);
    let p = f32::from_bits(0x3f7f_fff6) * r;
    let polynomial = q.mul_add(r2, p);
    scale.mul_add(polynomial, scale)
}

fn erf(x: f32) -> f32 {
    let t = 0.3275911f32.mul_add(x.abs(), 1.0).recip();
    let r = 1.061405429f32.mul_add(t, -1.453152027);
    let r = r.mul_add(t, 1.421413741);
    let r = r.mul_add(t, -0.284496736);
    let r = r.mul_add(t, 0.254829592);
    let exponent = erf_exp(-(x * x));
    let negative_exp_t = t * -exponent;
    negative_exp_t.mul_add(r, 1.0).copysign(x)
}

#[kernel(VisionGeluBF16)]
pub fn vision_gelu_bf16(
    values: *mut bf16,
    length: u32,
    approximate: u32,
) {
    for index in 0..length as usize {
        let x = unsafe { (*values.add(index)).to_f32() };
        let cdf = if approximate != 0 {
            let cube = x * x * x;
            tanh(0.7978845608028654f32 * (x + 0.044715f32 * cube))
        } else {
            erf(x * std::f32::consts::FRAC_1_SQRT_2)
        };
        let output = (0.5f32 * x) * (1.0f32 + cdf);
        unsafe { *values.add(index) = bf16::from_f32(output) };
    }
}
