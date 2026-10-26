// Numerical compatibility implementation of the BF16 RowwiseMoments reduction
// schedule in PyTorch v2.14.1, aten/src/ATen/native/cpu/moments_utils.h and
// layer_norm_kernel.cpp. The upstream BSD notices are retained in
// scripts/licenses/distribution/pytorch-LICENSE. The text-model normalization is independent.

use half::bf16;
use stark_engine_macros::kernel;

// VisionTowerConfig bounds a row to fewer than 2^20 elements. Sixteen levels
// therefore cover every supported row, including the four-lane BF16 schedule.
const MOMENT_LEVELS: usize = 16;

#[derive(Clone, Copy, Default)]
struct Moments {
    count: u32,
    mean: [f32; 4],
    squared_deviation: [f32; 4],
}

impl Moments {
    fn combine(
        &mut self,
        other: Self,
    ) {
        let old_count = self.count;
        let count = old_count + other.count;
        let fraction = if count == 0 {
            0.0
        } else {
            other.count as f32 / count as f32
        };
        for lane in 0..4 {
            let delta = other.mean[lane] - self.mean[lane];
            let weighted_delta = fraction * delta;
            let old_delta = delta * old_count as f32;
            let deviations = self.squared_deviation[lane] + other.squared_deviation[lane];
            self.mean[lane] += weighted_delta;
            self.squared_deviation[lane] = old_delta.mul_add(weighted_delta, deviations);
        }
        self.count = count;
    }
}

/// FP32 Welford moments with the pinned BF16 reference's two four-lane halves,
/// sixteen-vector chunks and pairwise cascade. Explicit multiply-add operations
/// match the reference; the final affine operations deliberately stay separate.
#[kernel(VisionLayerNormBF16)]
pub fn vision_layer_norm_bf16(
    input: *const bf16,
    weight: *const bf16,
    bias: *const bf16,
    output: *mut bf16,
    rows: u32,
    width: u32,
    epsilon: f32,
) {
    let width = width as usize;
    let vectors = width / 8;
    let chunks = vectors.div_ceil(16);
    let depth = if chunks == 0 {
        0
    } else {
        chunks.next_power_of_two().trailing_zeros() as usize
    };
    assert!(depth < MOMENT_LEVELS);

    for row in 0..rows as usize {
        let row_input = unsafe { input.add(row * width) };
        let mut stack = [Moments::default(); MOMENT_LEVELS];
        for chunk in 0..chunks {
            let count = 16.min(vectors - chunk * 16);
            let mut halves = [Moments::default(); 2];
            for index in 0..count {
                let fraction = 1.0 / (index + 1) as f32;
                for (half, moments) in halves.iter_mut().enumerate() {
                    for lane in 0..4 {
                        let value = unsafe { (*row_input.add(chunk * 128 + index * 8 + half * 4 + lane)).to_f32() };
                        let delta = value - moments.mean[lane];
                        moments.mean[lane] = delta.mul_add(fraction, moments.mean[lane]);
                        let centered = value - moments.mean[lane];
                        moments.squared_deviation[lane] = delta.mul_add(centered, moments.squared_deviation[lane]);
                    }
                }
            }
            halves[0].count = count as u32;
            halves[1].count = count as u32;
            stack[0].combine(halves[0]);
            stack[0].combine(halves[1]);
            let mut mask = chunk + 1;
            let mut level = 1;
            while level < depth && mask & 1 == 0 {
                let previous = stack[level - 1];
                stack[level].combine(previous);
                stack[level - 1] = Moments::default();
                mask >>= 1;
                level += 1;
            }
        }
        for level in 1..depth {
            let moments = stack[level];
            stack[0].combine(moments);
        }

        // Finish any scalar tail, then combine the four independent lane sums.
        let mut count = 0u32;
        let mut mean = 0.0f32;
        let mut squared_deviation = 0.0f32;
        for index in vectors * 8..width {
            let value = unsafe { (*row_input.add(index)).to_f32() };
            let delta = value - mean;
            count += 1;
            mean += delta / count as f32;
            squared_deviation = delta.mul_add(value - mean, squared_deviation);
        }
        let lane_count = (vectors * 2) as u32;
        for lane in 0..4 {
            let total = count + lane_count;
            let fraction = if total == 0 {
                0.0
            } else {
                lane_count as f32 / total as f32
            };
            let delta = stack[0].mean[lane] - mean;
            // The pinned scalar AddMoments contracts these expressions; its
            // vector combine and final affine use separate operations.
            mean = fraction.mul_add(delta, mean);
            let weighted_square = delta * delta * fraction;
            let correction = weighted_square.mul_add(count as f32, stack[0].squared_deviation[lane]);
            squared_deviation += correction;
            count = total;
        }
        let inverse_std = (squared_deviation / width as f32 + epsilon).sqrt().recip();
        let offset = -inverse_std * mean;
        for index in 0..width {
            unsafe {
                let value = (*row_input.add(index)).to_f32();
                let scaled = value * inverse_std;
                let centered = scaled + offset;
                let affine = centered * (*weight.add(index)).to_f32();
                *output.add(row * width + index) = bf16::from_f32(affine + (*bias.add(index)).to_f32());
            }
        }
    }
}
