use half::bf16;
use stark_engine_macros::kernel;

/// Preserve the BF16 residual sum before applying the final layer scale.
#[kernel(MaterializedBF16AddScale)]
pub fn materialized_bf16_add_scale(
    #[optional(!in_place)] input: Option<*const bf16>,
    bias: *const bf16,
    output: *mut bf16,
    num_cols: u32,
    length: u32,
    scale: f32,
    #[specialize] in_place: bool,
) {
    let input = if in_place {
        output as *const bf16
    } else {
        input.unwrap()
    };
    for position in 0..length as usize {
        unsafe {
            let sum =
                bf16::from_f32((*input.add(position)).to_f32() + (*bias.add(position % num_cols as usize)).to_f32());
            *output.add(position) = bf16::from_f32((sum.to_f32() + 0.0) * scale);
        }
    }
}
