// Copyright Naoki Shibata and contributors 2010 - 2024.
// Distributed under the Boost Software License, Version 1.0.
// See scripts/licenses/distribution/sleef-LICENSE.txt.
//
// Scalar-lane adaptation of SLEEF ADVSIMD sinf/cosf u10 and double-float
// arithmetic at 5a1d179df9cf652951b59010a2d2075372d67f68. This implements
// only the finite image-coordinate range admitted by the vision patch budget.
// Explicit mul_add operations match SLEEF; other operations must not contract.

#[derive(Clone, Copy)]
pub(crate) struct F32Pair {
    pub(crate) hi: f32,
    pub(crate) lo: f32,
}

impl F32Pair {
    pub(crate) fn normalize(self) -> Self {
        let hi = self.hi + self.lo;
        Self {
            hi,
            lo: (self.hi - hi) + self.lo,
        }
    }

    pub(crate) fn fast_two_sum(
        x: f32,
        y: f32,
    ) -> Self {
        let hi = x + y;
        Self {
            hi,
            lo: (x - hi) + y,
        }
    }

    pub(crate) fn two_sum(
        x: f32,
        y: f32,
    ) -> Self {
        let hi = x + y;
        let v = hi - x;
        Self {
            hi,
            lo: (x - (hi - v)) + (y - v),
        }
    }

    pub(crate) fn add_f(
        self,
        y: f32,
    ) -> Self {
        let hi = self.hi + y;
        Self {
            hi,
            lo: ((self.hi - hi) + y) + self.lo,
        }
    }

    pub(crate) fn f_add(
        x: f32,
        y: Self,
    ) -> Self {
        let hi = x + y.hi;
        Self {
            hi,
            lo: ((x - hi) + y.hi) + y.lo,
        }
    }

    pub(crate) fn add2_f(
        self,
        y: f32,
    ) -> Self {
        let hi = self.hi + y;
        let v = hi - self.hi;
        let t = (self.hi - (hi - v)) + (y - v);
        Self {
            hi,
            lo: t + self.lo,
        }
    }

    pub(crate) fn add2(
        self,
        y: Self,
    ) -> Self {
        let hi = self.hi + y.hi;
        let v = hi - self.hi;
        let t = (self.hi - (hi - v)) + (y.hi - v);
        Self {
            hi,
            lo: t + (self.lo + y.lo),
        }
    }

    pub(crate) fn product(
        x: f32,
        y: f32,
    ) -> Self {
        let hi = x * y;
        Self {
            hi,
            lo: x.mul_add(y, -hi),
        }
    }

    pub(crate) fn mul_f(
        self,
        y: f32,
    ) -> Self {
        let hi = self.hi * y;
        Self {
            hi,
            lo: self.lo.mul_add(y, self.hi.mul_add(y, -hi)),
        }
    }

    pub(crate) fn mul(
        self,
        y: Self,
    ) -> Self {
        let hi = self.hi * y.hi;
        Self {
            hi,
            lo: self.hi.mul_add(y.lo, self.lo.mul_add(y.hi, self.hi.mul_add(y.hi, -hi))),
        }
    }

    pub(crate) fn mul_to_f(
        self,
        y: Self,
    ) -> f32 {
        self.hi.mul_add(y.hi, self.lo.mul_add(y.hi, self.hi * y.lo))
    }

    pub(crate) fn square(self) -> Self {
        let hi = self.hi * self.hi;
        Self {
            hi,
            lo: (self.hi + self.hi).mul_add(self.lo, self.hi.mul_add(self.hi, -hi)),
        }
    }
}

fn remainder_quarter(x: f32) -> (f32, i32) {
    let y = (x * 4.0).round_ties_even();
    (x - y * 0.25, (y - x.round_ties_even() * 4.0) as i32)
}

fn remainder_pi(x: f32) -> (F32Pair, i32) {
    // All admitted coordinates are below 4096, so SLEEF's exponent-indexed
    // Payne-Hanek table always selects its first four source constants.
    let value = F32Pair::product(x, 0.159154892);
    let (hi, q) = remainder_quarter(value.hi);
    let value = F32Pair {
        hi,
        lo: value.lo,
    }
    .normalize()
    .add2(F32Pair::product(x, 5.112411827e-8));
    let (hi, q2) = remainder_quarter(value.hi);
    let value = F32Pair {
        hi,
        lo: value.lo,
    }
    .normalize();
    let low = F32Pair {
        hi: 3.626141271e-15,
        lo: -2.036222915e-22,
    }
    .mul_f(x);
    let value = value.add2(low).normalize().mul(F32Pair {
        hi: 3.1415927410125732422 * 2.0,
        lo: -8.7422776573475857731e-8 * 2.0,
    });
    if x.abs() < 0.7 {
        (
            F32Pair {
                hi: x,
                lo: 0.0,
            },
            q + q2,
        )
    } else {
        (value, q + q2)
    }
}

fn polynomial(t: F32Pair) -> f32 {
    let square = t.square();
    let u = 2.6083159809786593541503e-6f32.mul_add(square.hi, -0.0001981069071916863322258);
    let u = u.mul_add(square.hi, 0.00833307858556509017944336);
    let x = F32Pair::fast_two_sum(-0.166666597127914428710938, u * square.hi).mul(square);
    t.mul_to_f(F32Pair::f_add(1.0, x))
}

fn trig_lane(
    d: f32,
    large: bool,
    cosine: bool,
) -> f32 {
    let (s, q) = if !large {
        if cosine {
            let dq = d.mul_add(std::f32::consts::FRAC_1_PI, -0.5).round_ties_even().mul_add(2.0, 1.0);
            let s = F32Pair::two_sum(d, dq * (-3.1414794921875 * 0.5));
            let s = s.add2_f(dq * (-0.00011315941810607910156 * 0.5));
            (s.add2_f(dq * (-1.9841872589410058936e-9 * 0.5)), dq.round_ties_even() as i32)
        } else {
            let u = (d * std::f32::consts::FRAC_1_PI).round_ties_even();
            let v = u.mul_add(-3.1414794921875, d);
            let s = F32Pair::two_sum(v, u * -0.00011315941810607910156);
            (s.add_f(u * -1.9841872589410058936e-9), u as i32)
        }
    } else {
        let (mut s, q) = remainder_pi(d);
        let q = q & 3;
        let positive = s.hi > 0.0;
        let result_q = if cosine {
            (q * 2
                + if positive {
                    8
                } else {
                    7
                })
                >> 1
        } else {
            (q * 2
                + if positive {
                    2
                } else {
                    1
                })
                >> 2
        };
        if q & 1
            == if cosine {
                0
            } else {
                1
            }
        {
            let negative_sign = if cosine {
                !positive
            } else {
                s.hi.is_sign_negative()
            };
            let sign = if negative_sign {
                -1.0
            } else {
                1.0
            };
            s = s.add2(F32Pair {
                hi: (3.1415927410125732422 * -0.5) * sign,
                lo: (-8.7422776573475857731e-8 * -0.5) * sign,
            });
        }
        (s.normalize(), result_q)
    };
    let result = polynomial(s);
    let flip = if cosine {
        q & 2 == 0
    } else {
        q & 1 == 1
    };
    if !cosine && d == 0.0 {
        d
    } else if flip {
        -result
    } else {
        result
    }
}

/// Evaluate the same independent four-lane sin/cos calls as the pinned CPU
/// reference. The range branch is chosen per group, including its smaller lanes.
pub(crate) fn vision_sin_cos(angles: &[f32]) -> (Vec<f32>, Vec<f32>) {
    debug_assert!(angles.len().is_multiple_of(4));
    debug_assert!(angles.iter().all(|x| x.is_finite() && x.abs() < 4096.0));
    let mut cosines = Vec::with_capacity(angles.len());
    let mut sines = Vec::with_capacity(angles.len());
    for group in angles.chunks(4) {
        let large = group.iter().any(|d| d.abs() >= 125.0);
        for &d in group {
            cosines.push(trig_lane(d, large, true));
            sines.push(trig_lane(d, large, false));
        }
    }
    (cosines, sines)
}

#[cfg(test)]
mod table_tests;
