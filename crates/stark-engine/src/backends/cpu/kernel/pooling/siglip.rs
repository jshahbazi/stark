//! Gemma 3 spatial average pooling, before the learned soft-embedding RMS norm.

use half::bf16;
use stark_engine_macros::kernel;

#[kernel(SiglipPoolBF16)]
pub fn siglip_pool_bf16(
    input: *const bf16,
    output: *mut bf16,
    width: u32,
    side: u32,
    pool: u32,
) {
    let (width, side, pool) = (width as usize, side as usize, pool as usize);
    let out_side = side / pool;
    for y in 0..out_side {
        for x in 0..out_side {
            for channel in 0..width {
                let mut sum = 0.0f32;
                for dy in 0..pool {
                    for dx in 0..pool {
                        let offset = ((y * pool + dy) * side + x * pool + dx) * width + channel;
                        sum += unsafe { (*input.add(offset)).to_f32() };
                    }
                }
                unsafe {
                    *output.add((y * out_side + x) * width + channel) = bf16::from_f32(sum / (pool * pool) as f32)
                };
            }
        }
    }
}
