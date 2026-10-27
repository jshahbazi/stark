// F32 row-sum cascade adapted from PyTorch commit
// 5c4886908584029761b579af026dcfb627c84070, aten/src/ATen/native/cpu/SumKernel.cpp.
// CPU mean division follows ReduceOps.cpp; sqrt/reciprocal order follows
// aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h at the same revision.
// See scripts/licenses/distribution/pytorch-LICENSE for BSD terms/copyrights.

use half::bf16;
use stark_engine_macros::kernel;

/// Exact pinned BF16 RMS boundaries for ordinary, final and packed Q/K norms.
#[kernel(ReferenceRMSBF16)]
pub fn reference_rms_bf16(
    #[optional(!in_place)] input: Option<*const bf16>,
    scales: *const f32,
    output: *mut bf16,
    #[optional(copy_to_shortcut)] shortcut: Option<*mut bf16>,
    batch_size: u32,
    element_count: u32,
    row_stride: u32,
    epsilon: f32,
    scale_offset: f32,
    head_offset: u32,
    head_count: u32,
    #[specialize] in_place: bool,
    #[specialize] copy_to_shortcut: bool,
    #[specialize] residual_add: bool,
) {
    assert!(element_count >= 16 && element_count.is_multiple_of(16));
    assert!(copy_to_shortcut || !residual_add);
    assert_eq!(shortcut.is_some(), copy_to_shortcut);
    let input = if in_place {
        output as *const bf16
    } else {
        input.unwrap()
    };
    let width = element_count as usize;
    let count = width / 16;
    let ceil_log2 = usize::BITS - (count - 1).leading_zeros();
    let power = 4usize.max(ceil_log2 as usize / 4);
    let step = 1usize << power;
    let mask = step - 1;
    for row in 0..batch_size as usize {
        for head in 0..head_count as usize {
            let base = row * row_stride as usize + (head_offset as usize + head) * width;
            if let Some(shortcut) = shortcut {
                for channel in 0..width {
                    let mut value = unsafe { *input.add(base + channel) };
                    if residual_add {
                        value = bf16::from_f32(value.to_f32() + unsafe { (*shortcut.add(base + channel)).to_f32() });
                    }
                    unsafe {
                        *shortcut.add(base + channel) = value;
                    }
                }
            }
            let source = if residual_add {
                shortcut.unwrap() as *const bf16
            } else {
                input
            };
            let mut levels = [[[0.0f32; 4]; 4]; 4];
            let mut index = 0usize;
            while index + step <= count {
                for _ in 0..step {
                    for register in 0..4 {
                        for lane in 0..4 {
                            let x = unsafe { (*source.add(base + index * 16 + register * 4 + lane)).to_f32() };
                            levels[0][register][lane] += x * x;
                        }
                    }
                    index += 1;
                }
                for level in 1..4 {
                    for register in 0..4 {
                        for lane in 0..4 {
                            levels[level][register][lane] += levels[level - 1][register][lane];
                            levels[level - 1][register][lane] = 0.0;
                        }
                    }
                    if index & (mask << (level * power)) != 0 {
                        break;
                    }
                }
            }
            while index < count {
                for register in 0..4 {
                    for lane in 0..4 {
                        let x = unsafe { (*source.add(base + index * 16 + register * 4 + lane)).to_f32() };
                        levels[0][register][lane] += x * x;
                    }
                }
                index += 1;
            }
            for level in 1..4 {
                for register in 0..4 {
                    for lane in 0..4 {
                        levels[0][register][lane] += levels[level][register][lane];
                    }
                }
            }
            let mut lanes = levels[0][0];
            for register in 1..4 {
                for lane in 0..4 {
                    lanes[lane] += levels[0][register][lane];
                }
            }
            let mut sum = 0.0f32;
            for lane in lanes {
                sum += lane;
            }
            let mean = sum / element_count as f32;
            let inverse = (mean + epsilon).sqrt().recip();
            for channel in 0..width {
                let x = unsafe { (*source.add(base + channel)).to_f32() };
                let normalized = x * inverse;
                let gain = unsafe { *scales.add(channel) } + scale_offset;
                unsafe {
                    *output.add(base + channel) = bf16::from_f32(normalized * gain);
                }
            }
        }
    }
}
