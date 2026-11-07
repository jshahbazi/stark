//! Native Gemma 4 vision primitives, independently implementing the equations
//! in MLX-VLM 53ebd1647cb83549b482a031ea70ba9d9ea60a04 (MIT) and Google's
//! Gemma 4 E2B checkpoint (Apache-2.0). No reference runtime is invoked.

use half::bf16;
use stark_engine_macros::kernel;

#[kernel(Gemma4VisionClampBF16)]
pub fn gemma4_vision_clamp_bf16(
    input: *const bf16,
    output: *mut bf16,
    length: u32,
    minimum: f32,
    maximum: f32,
) {
    for i in 0..length as usize {
        unsafe { *output.add(i) = bf16::from_f32((*input.add(i)).to_f32().clamp(minimum, maximum)) };
    }
}

#[kernel(Gemma4VisionNormBF16)]
pub fn gemma4_vision_norm_bf16(
    input: *const bf16,
    weight: *const bf16,
    output: *mut bf16,
    rows: u32,
    width: u32,
    epsilon: f32,
    has_scale: u32,
) {
    let width = width as usize;
    for row in 0..rows as usize {
        let mut sum = 0.0f32;
        for j in 0..width {
            let value = unsafe { (*input.add(row * width + j)).to_f32() };
            sum += value * value;
        }
        let scale = (sum / width as f32 + epsilon).sqrt().recip();
        for j in 0..width {
            unsafe {
                let value = (*input.add(row * width + j)).to_f32() * scale;
                let gain = if has_scale == 0 {
                    1.0
                } else {
                    (*weight.add(j)).to_f32()
                };
                *output.add(row * width + j) = bf16::from_f32(value * gain);
            }
        }
    }
}

#[kernel(Gemma4VisionPrepareBF16)]
pub fn gemma4_vision_prepare_bf16(
    q: *const bf16,
    k: *const bf16,
    v: *const bf16,
    queries: *mut bf16,
    keys: *mut bf16,
    values: *mut bf16,
    cosines: *const bf16,
    sines: *const bf16,
    rows: u32,
    num_heads: u32,
    head_dim: u32,
) {
    let (rows, heads, width) = (rows as usize, num_heads as usize, head_dim as usize);
    for row in 0..rows {
        for head in 0..heads {
            for channel in 0..width {
                let axis_width = width / 2;
                let half = axis_width / 2;
                let axis = channel / axis_width;
                let local = channel % axis_width;
                let paired = axis * axis_width + (local + half) % axis_width;
                let base = (row * heads + head) * width;
                let rotate = |source: *const bf16| unsafe {
                    let cosine = (*cosines.add(row * width + channel)).to_f32();
                    let sine = (*sines.add(row * width + channel)).to_f32();
                    let value = (*source.add(base + channel)).to_f32();
                    let pair = (*source.add(base + paired)).to_f32()
                        * if local < half {
                            -1.0
                        } else {
                            1.0
                        };
                    let a = bf16::from_f32(value * cosine).to_f32();
                    let b = bf16::from_f32(pair * sine).to_f32();
                    bf16::from_f32(a + b)
                };
                unsafe {
                    *queries.add((head * rows + row) * width + channel) = rotate(q);
                    *keys.add(base + channel) = rotate(k);
                    *values.add(base + channel) = *v.add(base + channel);
                }
            }
        }
    }
}

#[kernel(Gemma4VisionProductBF16)]
pub fn gemma4_vision_product_bf16(
    left: *const bf16,
    right: *const bf16,
    output: *mut bf16,
    length: u32,
) {
    for i in 0..length as usize {
        unsafe { *output.add(i) = bf16::from_f32((*left.add(i)).to_f32() * (*right.add(i)).to_f32()) };
    }
}

#[kernel(Gemma4VisionPoolBF16)]
pub fn gemma4_vision_pool_bf16(
    input: *const bf16,
    output: *mut bf16,
    grid_width: u32,
    grid_height: u32,
    width: u32,
) {
    let (gw, gh, width) = (grid_width as usize, grid_height as usize, width as usize);
    for y in 0..gh / 3 {
        for x in 0..gw / 3 {
            for channel in 0..width {
                let mut sum = 0.0f32;
                for dy in 0..3 {
                    for dx in 0..3 {
                        sum +=
                            unsafe { (*input.add(((y * 3 + dy) * gw + x * 3 + dx) * width + channel)).to_f32() } / 9.0;
                    }
                }
                let pooled = bf16::from_f32(sum).to_f32();
                unsafe {
                    *output.add((y * (gw / 3) + x) * width + channel) = bf16::from_f32(pooled * (width as f32).sqrt())
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn gemma_vision_norm_matches_independent_float64_rms() {
        let input = [bf16::from_f32(1.25), bf16::from_f32(-2.5), bf16::from_f32(0.75), bf16::from_f32(4.0)];
        let weight = [bf16::from_f32(0.5), bf16::from_f32(2.0), bf16::from_f32(1.0), bf16::from_f32(-1.0)];
        let mut output = [bf16::ZERO; 4];
        gemma4_vision_norm_bf16(input.as_ptr(), weight.as_ptr(), output.as_mut_ptr(), 1, 4, 1e-6, 1);
        let rms = (input.iter().map(|value| f64::from(value.to_f32()).powi(2)).sum::<f64>() / 4.0 + 1e-6).sqrt();
        for i in 0..4 {
            let expected = bf16::from_f64(f64::from(input[i].to_f32()) / rms * f64::from(weight[i].to_f32()));
            assert!(
                (output[i].to_f32() - expected.to_f32()).abs() < 0.008,
                "Gemma RMS differs from independent reference"
            );
        }
    }

    #[stark_test]
    fn gemma_vision_axial_rope_does_not_mix_spatial_partitions() {
        let q = (0..64).map(|i| bf16::from_f32((i + 1) as f32 / 64.0)).collect::<Vec<_>>();
        let k = q.clone();
        let v = q.clone();
        let mut queries = vec![bf16::ZERO; 64];
        let mut keys = queries.clone();
        let mut values = queries.clone();
        let mut cosine = vec![bf16::ONE; 64];
        let mut sine = vec![bf16::ZERO; 64];
        for i in 32..64 {
            let angle = 7.0 / 100f32.powf(((i % 16) * 2) as f32 / 32.0);
            cosine[i] = bf16::from_f32(angle.cos());
            sine[i] = bf16::from_f32(angle.sin());
        }
        gemma4_vision_prepare_bf16(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            queries.as_mut_ptr(),
            keys.as_mut_ptr(),
            values.as_mut_ptr(),
            cosine.as_ptr(),
            sine.as_ptr(),
            1,
            1,
            64,
        );
        assert!(queries[..32] == q[..32], "X partition changed under Y-only rotation");
        assert!(values == v, "RoPE changed values");
        let a = bf16::from_f32(q[32].to_f32() * cosine[32].to_f32());
        let b = bf16::from_f32(-q[48].to_f32() * sine[32].to_f32());
        assert!(queries[32] == bf16::from_f32(a.to_f32() + b.to_f32()), "Y partition uses an incorrect paired channel");
    }

    #[stark_test]
    fn gemma_vision_pool_keeps_rectangular_patch_groups() {
        let input = (0..3)
            .flat_map(|_| {
                (0..6).flat_map(|x| {
                    [bf16::from_f32(if x < 3 {
                        3.0
                    } else {
                        9.0
                    }); 4]
                })
            })
            .collect::<Vec<_>>();
        let mut output = [bf16::ZERO; 8];
        gemma4_vision_pool_bf16(input.as_ptr(), output.as_mut_ptr(), 6, 3, 4);
        assert!(output[..4].iter().all(|v| v.to_f32() == 6.0));
        assert!(output[4..].iter().all(|v| v.to_f32() == 18.0));
    }
}
