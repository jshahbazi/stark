use half::bf16;
use num_traits::Float;
use stark_engine_macros::kernel;

use crate::array::ArrayElement;

#[kernel(LogitTransform)]
#[variants(T, f32, bf16)]
pub fn logit_transform<T: ArrayElement + Float>(
    logits: *mut T,
    length: u32,
    scale: f32,
    soft_cap: f32,
    #[specialize] has_soft_cap: bool,
    #[specialize] materialize_soft_cap: bool,
) {
    let length = length as usize;
    unsafe {
        for position in 0..length {
            let mut value = (*logits.add(position)).to_f32().unwrap() * scale;
            if has_soft_cap {
                if materialize_soft_cap {
                    value = T::from(value).unwrap().to_f32().unwrap();
                    let divided = T::from(value / soft_cap).unwrap().to_f32().unwrap();
                    let activated = T::from(divided.tanh()).unwrap().to_f32().unwrap();
                    value = activated * soft_cap;
                } else {
                    value = (value / soft_cap).tanh() * soft_cap;
                }
            }
            *logits.add(position) = T::from(value).unwrap();
        }
    }
}
