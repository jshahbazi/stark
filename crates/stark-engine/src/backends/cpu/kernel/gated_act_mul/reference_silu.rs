use half::bf16;
use stark_engine_macros::kernel;

use crate::backends::common::vision_math::vision_exp;

/// PyTorch 5c4886908584029761b579af026dcfb627c84070 Activation.cpp/NEON SiLU:
/// x/(1+exp(-x)), cast BF16, then BF16 product. BSD terms are preserved in
/// scripts/licenses/distribution/pytorch-LICENSE.
/// SLEEF u10 source attribution is recorded with the shared exponential helper.
#[kernel(ReferenceGatedSiLUBF16)]
pub fn reference_gated_silu_bf16(
    packed: *const bf16,
    output: *mut bf16,
    gated_dim: u32,
    rows: u32,
) {
    let width = gated_dim as usize;
    for row in 0..rows as usize {
        for column in 0..width {
            let base = row * 2 * width;
            let value = unsafe { (*packed.add(base + column)).to_f32() };
            let gate = unsafe { (*packed.add(base + width + column)).to_f32() };
            let activated = bf16::from_f32(gate / (1.0 + vision_exp(-gate))).to_f32();
            unsafe {
                *output.add(row * width + column) = bf16::from_f32(value * activated);
            }
        }
    }
}
