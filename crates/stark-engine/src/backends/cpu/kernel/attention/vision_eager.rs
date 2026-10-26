use half::bf16;
use stark_engine_macros::kernel;

// SLEEF expf_u10, revision 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors, Boost Software License 1.0;
// see scripts/licenses/distribution/sleef-LICENSE.txt.
fn vision_exp(x: f32) -> f32 {
    if x < -104.0 {
        return 0.0;
    }
    if x > 100.0 {
        return f32::INFINITY;
    }
    let q = (x * 1.4426950408889634f32).round_ties_even();
    let s = q.mul_add(-0.693145751953125f32, x);
    let s = q.mul_add(-1.428606765330187e-6f32, s);
    let mut u = 0.00019852761761285365f32;
    u = u.mul_add(s, 0.0013930435525253415f32);
    u = u.mul_add(s, 0.008333360776305199f32);
    u = u.mul_add(s, 0.041666485369205475f32);
    u = u.mul_add(s, 0.1666666716337204f32);
    u = u.mul_add(s, 0.5);
    u = 1.0 + (s * s).mul_add(u, s);
    let half = (q as i32) >> 1;
    let first = f32::from_bits(((half + 127) as u32) << 23);
    let second = f32::from_bits(((q as i32 - half + 127) as u32) << 23);
    (u * first) * second
}

/// The pinned FP32 softmax reduces four lanes, folds crosswise, and multiplies
/// by the reciprocal before the eager attention's single BF16 conversion.
#[kernel(VisionSoftmaxBF16)]
pub fn vision_softmax_bf16(
    values: *mut bf16,
    row_length: u32,
    rows: u32,
) {
    for row in 0..rows as usize {
        let start = row * row_length as usize;
        let mut maximum = f32::NEG_INFINITY;
        for column in 0..row_length as usize {
            maximum = maximum.max(unsafe { (*values.add(start + column)).to_f32() });
        }
        let mut partial = [0.0f32; 4];
        for column in 0..row_length as usize {
            partial[column % 4] += vision_exp(unsafe { (*values.add(start + column)).to_f32() } - maximum);
        }
        let total = if row_length >= 4 {
            (partial[0] + partial[2]) + (partial[1] + partial[3])
        } else {
            partial[..row_length as usize].iter().copied().reduce(|sum, x| sum + x).unwrap_or(0.0)
        };
        let inverse_total = total.recip();
        for column in 0..row_length as usize {
            let probability = vision_exp(unsafe { (*values.add(start + column)).to_f32() } - maximum) * inverse_total;
            unsafe {
                *values.add(start + column) = bf16::from_f32(probability);
            }
        }
    }
}

/// Interleave a contiguous [row, head_dim] result into [row, heads, head_dim].
#[kernel(VisionHeadCopyBF16)]
pub fn vision_head_copy_bf16(
    input: *const bf16,
    output: *mut bf16,
    length: u32,
    head_dim: u32,
    model_dim: u32,
    head_index: u32,
) {
    for index in 0..length as usize {
        let destination = index / head_dim as usize * model_dim as usize
            + head_index as usize * head_dim as usize
            + index % head_dim as usize;
        unsafe {
            *output.add(destination) = *input.add(index);
        }
    }
}
