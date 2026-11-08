//! Larger-Gemma vision only. Pinned Torch CPU FlashAttentionKernel.cpp,
//! vec128_float_neon.h and functional_base.h at PyTorch
//! 5c4886908584029761b579af026dcfb627c84070; BSD notice in distribution licenses.

use half::bf16;
use stark_engine_macros::kernel;

const TILE: usize = 512;
const WIDTH: usize = 72;
const PADDED: usize = 128;

fn fast_exp(x: f32) -> f32 {
    if x < f32::from_bits(0xc2af_5dc1) {
        return 0.0;
    }
    if x > f32::from_bits(0x42b0_c0a5) {
        return f32::INFINITY;
    }
    let n = (x * f32::from_bits(0x3fb8_aa3b)).round();
    let r = (-n).mul_add(f32::from_bits(0x3f31_7218), x);
    let q = r.mul_add(f32::from_bits(0x3e2a_c976), f32::from_bits(0x3f00_be9a));
    let p = q.mul_add(r * r, 1.0 + r);
    f32::from_bits(p.to_bits().wrapping_add((n as i32 as u32) << 23))
}

#[kernel(Gemma4VisionFlashQK)]
pub fn gemma4_vision_flash_qk(
    queries: *const bf16,
    keys: *const bf16,
    scores: *mut f32,
    query_rows: u32,
    key_rows: u32,
    heads: u32,
    key_start: u32,
) {
    for head in 0..heads as usize {
        for row in 0..query_rows as usize {
            let query = (head * query_rows as usize + row) * PADDED;
            for column in 0..TILE {
                let index = (head * query_rows as usize + row) * TILE + column;
                let key_row = key_start as usize + column;
                if key_row >= key_rows as usize {
                    unsafe { *scores.add(index) = f32::NEG_INFINITY };
                    continue;
                }
                let key = (key_row * heads as usize + head) * PADDED;
                // CPUBlas gemm_no_downcast uses generic gemm_transa_ and
                // sum()'s four partials. BF16-output Linear's 32-stream
                // compute_dot specialization does not apply to F32 scores.
                let mut sums = [0.0f32; 4];
                for start in (0..WIDTH).step_by(4) {
                    for lane in 0..4 {
                        let channel = start + lane;
                        let (q, k) =
                            unsafe { ((*queries.add(query + channel)).to_f32(), (*keys.add(key + channel)).to_f32()) };
                        sums[lane] = q.mul_add(k, sums[lane]);
                    }
                }
                unsafe { *scores.add(index) = ((sums[0] + sums[1]) + sums[2]) + sums[3] };
            }
        }
    }
}

#[kernel(Gemma4VisionFlashExpBF16)]
pub fn gemma4_vision_flash_exp_bf16(
    scores: *const f32,
    exponents: *mut bf16,
    state: *mut f32,
    query_rows: u32,
    heads: u32,
    key_start: u32,
) {
    for row in 0..(query_rows * heads) as usize {
        let base = row * TILE;
        let previous_max = if key_start == 0 {
            f32::NEG_INFINITY
        } else {
            unsafe { *state.add(row * 3) }
        };
        let previous_sum = if key_start == 0 {
            0.0
        } else {
            unsafe { *state.add(row * 3 + 1) }
        };
        let mut maximum = previous_max;
        for column in 0..TILE {
            maximum = maximum.max(unsafe { *scores.add(base + column) });
        }
        let factor = (previous_max - maximum).exp();
        let mut partial = [0.0f32; 8];
        // Source virtual padding is 2520 keys (multiple of eight): all valid
        // entries use vector fexp, including a compressed ragged key tail.
        for column in 0..TILE {
            let value = fast_exp(unsafe { *scores.add(base + column) } - maximum);
            partial[column % 8] += value;
            unsafe { *exponents.add(base + column) = bf16::from_f32(value) };
        }
        let lanes =
            [partial[0] + partial[4], partial[1] + partial[5], partial[2] + partial[6], partial[3] + partial[7]];
        let total = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
        unsafe {
            *state.add(row * 3) = maximum;
            *state.add(row * 3 + 1) = factor.mul_add(previous_sum, total);
            *state.add(row * 3 + 2) = factor;
        }
    }
}

/// Model-free diagnostic for the exact finish reciprocal; unused in production.
#[kernel(Gemma4VisionFlashReciprocalF32)]
pub fn gemma4_vision_flash_reciprocal_f32(
    state: *const f32,
    inverse: *mut f32,
    query_rows: u32,
    heads: u32,
) {
    for row in 0..(query_rows * heads) as usize {
        unsafe { *inverse.add(row) = (*state.add(row * 3 + 1)).recip() };
    }
}

/// Source scalar expf observer. CPU production retains its libm implementation.
#[kernel(Gemma4VisionFlashScalarExpF32)]
pub fn gemma4_vision_flash_scalar_exp_f32(
    input: *const f32,
    output: *mut f32,
    count: u32,
) {
    for row in 0..count as usize {
        unsafe { *output.add(row) = (*input.add(row)).exp() };
    }
}

#[kernel(Gemma4VisionFlashPVF32)]
pub fn gemma4_vision_flash_pv_f32(
    exponents: *const bf16,
    values: *const bf16,
    state: *const f32,
    accumulator: *mut f32,
    query_rows: u32,
    key_rows: u32,
    heads: u32,
    key_start: u32,
) {
    for head in 0..heads as usize {
        for row in 0..query_rows as usize {
            let plane = head * query_rows as usize + row;
            for channel in 0..WIDTH {
                let mut sums = [0.0f32; 4];
                for column in 0..TILE {
                    let key = key_start as usize + column;
                    if key < key_rows as usize {
                        let (p, v) = unsafe {
                            (
                                (*exponents.add(plane * TILE + column)).to_f32(),
                                (*values.add((key * heads as usize + head) * PADDED + channel)).to_f32(),
                            )
                        };
                        sums[column % 4] = p.mul_add(v, sums[column % 4]);
                    }
                }
                let current = ((sums[0] + sums[1]) + sums[2]) + sums[3];
                let index = plane * WIDTH + channel;
                let previous = if key_start == 0 {
                    0.0
                } else {
                    unsafe { *accumulator.add(index) * *state.add(plane * 3 + 2) }
                };
                unsafe { *accumulator.add(index) = previous + current };
            }
        }
    }
}

#[kernel(Gemma4VisionFlashFinishBF16)]
pub fn gemma4_vision_flash_finish_bf16(
    accumulator: *const f32,
    state: *const f32,
    output: *mut bf16,
    query_rows: u32,
    heads: u32,
) {
    for head in 0..heads as usize {
        for row in 0..query_rows as usize {
            let plane = head * query_rows as usize + row;
            let inverse = unsafe { (*state.add(plane * 3 + 1)).recip() };
            for channel in 0..WIDTH {
                unsafe {
                    *output.add((row * heads as usize + head) * WIDTH + channel) =
                        bf16::from_f32(*accumulator.add(plane * WIDTH + channel) * inverse)
                };
            }
        }
    }
}
