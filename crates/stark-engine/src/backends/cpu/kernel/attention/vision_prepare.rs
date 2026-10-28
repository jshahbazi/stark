use half::bf16;
use stark_engine_macros::kernel;

/// Preserve the pinned vision operation's two FP32 products and separate add.
/// Q becomes head-major; K and V retain the patch-major head layout.
#[kernel(VisionPrepareBF16)]
pub fn vision_prepare_bf16(
    qkv: *const bf16,
    queries: *mut bf16,
    keys: *mut bf16,
    values: *mut bf16,
    cosines: *const f32,
    sines: *const f32,
    rows: u32,
    num_heads: u32,
    head_dim: u32,
) {
    let (rows, heads, width) = (rows as usize, num_heads as usize, head_dim as usize);
    let model_width = heads * width;
    for row in 0..rows {
        for head in 0..heads {
            for channel in 0..width {
                let paired = if channel < width / 2 {
                    channel + width / 2
                } else {
                    channel - width / 2
                };
                let source = row * 3 * model_width + head * width;
                let position = row * width + channel;
                let rotate = |offset: usize| unsafe {
                    let value = (*qkv.add(source + offset + channel)).to_f32();
                    let pair = (*qkv.add(source + offset + paired)).to_f32();
                    let signed_pair = if channel < width / 2 {
                        -pair
                    } else {
                        pair
                    };
                    let first = value * *cosines.add(position);
                    let second = signed_pair * *sines.add(position);
                    bf16::from_f32(first + second)
                };
                unsafe {
                    *queries.add(head * rows * width + row * width + channel) = rotate(0);
                    *keys.add(row * model_width + head * width + channel) = rotate(model_width);
                    *values.add(row * model_width + head * width + channel) =
                        *qkv.add(source + 2 * model_width + channel);
                }
            }
        }
    }
}
