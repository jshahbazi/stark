// FP32 reduction schedules adapted from PyTorch commit
// 5c4886908584029761b579af026dcfb627c84070, BlasKernel.cpp,
// ReducedPrecisionFloatGemvFastPathKernel.cpp and functional_base.h.
// See scripts/licenses/distribution/pytorch-LICENSE and ggml-LICENSE for
// upstream BSD/MIT terms and copyrights.

use half::bf16;
use stark_engine_macros::kernel;

/// Indexed equivalent of the aligned contiguous reference dot below. Read the
/// selected bank row directly; scratch storage never contains weight matrices.
#[kernel(ReferenceGatherMatmulBF16)]
pub fn reference_gather_matmul_bf16(
    input: *const bf16,
    weights: *const bf16,
    indices: *const u32,
    output: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    bank_rows: u32,
) {
    for row in 0..m as usize {
        for column in 0..n as usize {
            let selected = unsafe { *indices.add(row * n as usize + column) };
            let result = if selected < bank_rows {
                let mut sums = [[0.0f32; 4]; 8];
                for start in (0..k as usize).step_by(32) {
                    for (register, lanes) in sums.iter_mut().enumerate() {
                        for (lane, sum) in lanes.iter_mut().enumerate() {
                            let inner = start + register * 4 + lane;
                            let (a, b) = unsafe {
                                (
                                    (*input.add(row * k as usize + inner)).to_f32(),
                                    (*weights.add(selected as usize * k as usize + inner)).to_f32(),
                                )
                            };
                            *sum = a.mul_add(b, *sum);
                        }
                    }
                }
                for offset in [4, 2, 1] {
                    for register in 0..offset {
                        for lane in 0..4 {
                            sums[register][lane] += sums[register + offset][lane];
                        }
                    }
                }
                (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3])
            } else {
                f32::NAN
            };
            unsafe { *output.add(row * n as usize + column) = result };
        }
    }
}

/// Reproduce the pinned BF16 operation's FP32 accumulation before its final cast.
/// Contiguous dot products use eight four-lane registers; the non-transposed
/// strided and Conv3d paths use four independent scalar partial sums.
#[kernel(VisionMatmulBF16)]
pub fn vision_matmul_bf16(
    input: *const bf16,
    weights: *const bf16,
    output: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    b_leading_dimension: u32,
    transpose_b: u32,
    streams: u32,
) {
    let (m, n, k, leading) = (m as usize, n as usize, k as usize, b_leading_dimension as usize);
    for row in 0..m {
        for column in 0..n {
            let product = |inner: usize| unsafe {
                let a = (*input.add(row * k + inner)).to_f32();
                let b_index = if transpose_b != 0 {
                    column * leading + inner
                } else {
                    inner * leading + column
                };
                (a, (*weights.add(b_index)).to_f32())
            };
            let mut result;
            if streams == 32 {
                let mut sums = [[0.0f32; 4]; 8];
                let aligned = k / 32 * 32;
                for start in (0..aligned).step_by(32) {
                    for (register, lanes) in sums.iter_mut().enumerate() {
                        for (lane, sum) in lanes.iter_mut().enumerate() {
                            let (a, b) = product(start + register * 4 + lane);
                            *sum = a.mul_add(b, *sum);
                        }
                    }
                }
                for offset in [4, 2, 1] {
                    for register in 0..offset {
                        for lane in 0..4 {
                            sums[register][lane] += sums[register + offset][lane];
                        }
                    }
                }
                result = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);
                let mut tail = [0.0f32; 4];
                let vector_aligned = k / 8 * 8;
                for start in (aligned..vector_aligned).step_by(8) {
                    for half in 0..2 {
                        for (lane, sum) in tail.iter_mut().enumerate() {
                            let (a, b) = product(start + half * 4 + lane);
                            *sum = a.mul_add(b, *sum);
                        }
                    }
                }
                result += (tail[0] + tail[1]) + (tail[2] + tail[3]);
                for inner in vector_aligned..k {
                    let (a, b) = product(inner);
                    result = a.mul_add(b, result);
                }
            } else {
                let mut sums = [0.0f32; 4];
                let aligned = k / 4 * 4;
                for start in (0..aligned).step_by(4) {
                    for (lane, sum) in sums.iter_mut().enumerate() {
                        let (a, b) = product(start + lane);
                        *sum = a.mul_add(b, *sum);
                    }
                }
                for inner in aligned..k {
                    let (a, b) = product(inner);
                    sums[0] = a.mul_add(b, sums[0]);
                }
                result = ((sums[0] + sums[1]) + sums[2]) + sums[3];
            }
            unsafe { *output.add(row * n + column) = result };
        }
    }
}

/// Diagnostic packed counterpart of the pinned eight-by-four BF16 dot.
#[kernel(QwenAffineReferenceMatmulBF16)]
pub fn qwen_affine_reference_matmul_bf16(
    input: *const bf16,
    codes: *const u8,
    scales: *const bf16,
    biases: *const bf16,
    output: *mut bf16,
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
) {
    let (m, n, k, bits) = (m as usize, n as usize, k as usize, bits as usize);
    for row in 0..m {
        for column in 0..n {
            if !matches!(bits, 4 | 5 | 6 | 8) || !k.is_multiple_of(64) {
                unsafe { *output.add(row * n + column) = bf16::NAN };
                continue;
            }
            let product = |inner: usize| unsafe {
                let bit = inner * bits;
                let shift = bit % 8;
                let offset = column * (k * bits / 8) + bit / 8;
                let mut packed = u32::from(*codes.add(offset));
                if shift + bits > 8 {
                    packed |= u32::from(*codes.add(offset + 1)) << 8;
                }
                let code = (packed >> shift) & ((1 << bits) - 1);
                let group = column * scale_output_stride as usize + inner / 64 * scale_group_stride as usize;
                let value = code as f32 * (*scales.add(group)).to_f32() + (*biases.add(group)).to_f32();
                ((*input.add(row * k + inner)).to_f32(), bf16::from_f32(value).to_f32())
            };
            let mut sums = [[0.0_f32; 4]; 8];
            for start in (0..k).step_by(32) {
                for (register, lanes) in sums.iter_mut().enumerate() {
                    for (lane, sum) in lanes.iter_mut().enumerate() {
                        let (a, b) = product(start + register * 4 + lane);
                        *sum = a.mul_add(b, *sum);
                    }
                }
            }
            for offset in [4, 2, 1] {
                for register in 0..offset {
                    for lane in 0..4 {
                        sums[register][lane] += sums[register + offset][lane];
                    }
                }
            }
            let value = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);
            unsafe { *output.add(row * n + column) = bf16::from_f32(value) };
        }
    }
}
