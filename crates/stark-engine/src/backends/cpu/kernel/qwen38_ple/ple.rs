//! CPU references for the standard-backend Qwen3.8 PLE kernels.

use half::bf16;
use stark_engine_macros::kernel;

#[kernel(Qwen38PleGather)]
pub fn qwen38_ple_gather(
    codes: *const u8,
    scales: *const bf16,
    biases: *const bf16,
    output: *mut bf16,
    head_dim: u32,
    group_size: u32,
    heads: u32,
    rows: u32,
) {
    for head in 0..(heads * rows) as usize {
        for column in 0..head_dim as usize {
            let code_at = head * (head_dim as usize / 2) + column / 2;
            let parameter = head * (head_dim / group_size) as usize + column / group_size as usize;
            unsafe {
                let code = (*codes.add(code_at) >> (4 * (column % 2))) & 15;
                *output.add(head * head_dim as usize + column) =
                    bf16::from_f32(code as f32 * (*scales.add(parameter)).to_f32() + (*biases.add(parameter)).to_f32());
            }
        }
    }
}

#[kernel(Qwen38PleGate)]
pub fn qwen38_ple_gate(
    key: *const bf16,
    query: *const bf16,
    value: *const bf16,
    output: *mut bf16,
    width: u32,
    streams: u32,
    rows: u32,
) {
    for segment in 0..(streams * rows) as usize {
        let base = segment * width as usize;
        let mut sum = 0.0f32;
        for i in 0..width as usize {
            unsafe {
                sum += (*key.add(base + i)).to_f32() * (*query.add(base + i)).to_f32();
            }
        }
        sum *= 1.0 / (width as f32).sqrt();
        let magnitude = sum.abs().max(1e-6).sqrt();
        let signed_root = if sum < 0.0 {
            -magnitude
        } else {
            magnitude
        };
        let gate = 1.0 / (1.0 + (-signed_root).exp());
        let value_base = segment / streams as usize * width as usize;
        for i in 0..width as usize {
            unsafe {
                *output.add(base + i) = bf16::from_f32(gate * (*value.add(value_base + i)).to_f32());
            }
        }
    }
}

#[kernel(Qwen38PleConv)]
pub fn qwen38_ple_conv(
    window: *mut bf16,
    input: *const bf16,
    weight: *const f32,
    base: *const bf16,
    hyper: *mut bf16,
    width: u32,
    kernel_size: u32,
    dilation: u32,
    rows: u32,
) {
    let history_size = ((kernel_size - 1) * dilation) as usize;
    for column in 0..width as usize {
        let mut history = [0.0f32; 16];
        for (i, value) in history[..history_size].iter_mut().enumerate() {
            unsafe {
                *value = (*window.add(column * history_size + i)).to_f32();
            }
        }
        for row in 0..rows as usize {
            let at = row * width as usize + column;
            unsafe {
                let current = (*input.add(at)).to_f32();
                let mut sum = current * *weight.add(column * kernel_size as usize + kernel_size as usize - 1);
                for tap in 0..kernel_size as usize - 1 {
                    sum += history[history_size - (kernel_size as usize - 1 - tap) * dilation as usize]
                        * *weight.add(column * kernel_size as usize + tap);
                }
                let conv = sum / (1.0 + (-sum).exp());
                *hyper.add(at) = bf16::from_f32((*hyper.add(at)).to_f32() + (*base.add(at)).to_f32() + conv);
                history.copy_within(1..history_size, 0);
                history[history_size - 1] = current;
            }
        }
        for (i, value) in history[..history_size].iter().enumerate() {
            unsafe {
                *window.add(column * history_size + i) = bf16::from_f32(*value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn staged_q4_rows_preserve_nibble_and_group_order() {
        let codes = [0x10u8, 0x32, 0x54, 0x76];
        let scales = [2.0, 3.0, 1.0, 1.0].map(bf16::from_f32);
        let biases = [-1.0, 4.0, 0.0, 0.0].map(bf16::from_f32);
        let mut output = [bf16::NAN; 8];
        qwen38_ple_gather(codes.as_ptr(), scales.as_ptr(), biases.as_ptr(), output.as_mut_ptr(), 4, 2, 1, 2);
        assert_eq!(output, [-1.0, 1.0, 10.0, 13.0, 4.0, 5.0, 6.0, 7.0].map(bf16::from_f32));
    }

    #[stark_test]
    fn channel_major_dilated_convolution_and_chunking_match() {
        let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0].map(bf16::from_f32);
        let weights = [2.0f32, 3.0, 5.0, 7.0];
        let base = [bf16::ZERO; 6];
        let mut whole = [bf16::ZERO; 6];
        let mut whole_history = [bf16::ZERO; 4];
        qwen38_ple_conv(
            whole_history.as_mut_ptr(),
            input.as_ptr(),
            weights.as_ptr(),
            base.as_ptr(),
            whole.as_mut_ptr(),
            2,
            2,
            2,
            3,
        );
        let expected = [3.0f32, 70.0, 6.0, 140.0, 11.0, 260.0].map(|v| bf16::from_f32(v / (1.0 + (-v).exp())));
        assert_eq!(whole, expected);
        assert_eq!(whole_history, [2.0, 3.0, 20.0, 30.0].map(bf16::from_f32));
        for split in 1..3 {
            let mut chunked = [bf16::ZERO; 6];
            let mut history = [bf16::ZERO; 4];
            qwen38_ple_conv(
                history.as_mut_ptr(),
                input.as_ptr(),
                weights.as_ptr(),
                base.as_ptr(),
                chunked.as_mut_ptr(),
                2,
                2,
                2,
                split,
            );
            unsafe {
                qwen38_ple_conv(
                    history.as_mut_ptr(),
                    input.as_ptr().add(split as usize * 2),
                    weights.as_ptr(),
                    base.as_ptr().add(split as usize * 2),
                    chunked.as_mut_ptr().add(split as usize * 2),
                    2,
                    2,
                    2,
                    3 - split,
                );
            }
            assert_eq!(chunked, whole);
            assert_eq!(history, whole_history);
        }
    }
}
