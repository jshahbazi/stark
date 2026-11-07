//! Reference routing and accumulation for the bounded selected-expert path.

use half::bf16;
use stark_engine_macros::kernel;

#[kernel(MoeRoute)]
pub fn moe_route(
    logits: *const f32,
    ids: *mut u32,
    weights: *mut f32,
    rows: u32,
    experts: u32,
    active: u32,
    round_weights: u32,
    round_logits: u32,
) {
    for row in 0..rows as usize {
        let experts = experts as usize;
        let active = active as usize;
        let source = unsafe { std::slice::from_raw_parts(logits.add(row * experts), experts) };
        let values: Vec<f32> = source
            .iter()
            .map(|x| {
                if round_logits != 0 {
                    bf16::from_f32(*x).to_f32()
                } else {
                    *x
                }
            })
            .collect();
        let x = &values;
        let ids = unsafe { std::slice::from_raw_parts_mut(ids.add(row * active), active) };
        let weights = unsafe { std::slice::from_raw_parts_mut(weights.add(row * active), active) };
        let valid = (1..=1024).contains(&experts)
            && (1..=32).contains(&active)
            && active <= experts
            && x.iter().all(|x| x.is_finite());
        let maximum = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = x.iter().map(|x| (*x - maximum).exp()).sum();
        let mut selected_sum = 0.0;
        for rank in 0..active {
            let mut best = f32::NEG_INFINITY;
            let mut selected = 0;
            for (expert, &value) in x.iter().enumerate() {
                if !ids[..rank].contains(&(expert as u32)) && value > best {
                    best = value;
                    selected = expert as u32;
                }
            }
            ids[rank] = if valid {
                selected
            } else {
                0
            };
            weights[rank] = (best - maximum).exp() / denominator;
            selected_sum += weights[rank];
        }
        for weight in weights {
            *weight /= selected_sum;
            if round_weights != 0 {
                *weight = bf16::from_f32(*weight).to_f32();
            }
            if !valid {
                *weight = f32::NAN;
            }
        }
    }
}

#[kernel(MoeGatherRows)]
pub fn moe_gather_rows(
    ids: *const u32,
    indices: *mut u32,
    route_offset: u32,
    rows_per_expert: u32,
    experts: u32,
    length: u32,
) {
    let expert = unsafe { *ids.add(route_offset as usize) };
    let expert = if expert < experts {
        expert
    } else {
        0
    };
    for position in 0..length as usize {
        unsafe {
            *indices.add(position) = expert * rows_per_expert + position as u32;
        }
    }
}

#[kernel(MoeAccumulate)]
pub fn moe_accumulate(
    input: *const bf16,
    weights: *const f32,
    accumulator: *mut f32,
    output: *mut bf16,
    length: u32,
    weight_offset: u32,
    output_offset: u32,
    initialize: u32,
    finalize: u32,
    round_product: u32,
) {
    let weight = unsafe { *weights.add(weight_offset as usize) };
    for position in 0..length as usize {
        let out = output_offset as usize + position;
        unsafe {
            let mut product = (*input.add(position)).to_f32() * weight;
            if round_product != 0 {
                product = bf16::from_f32(product).to_f32();
            }
            let value = if initialize != 0 {
                0.0
            } else {
                *accumulator.add(out)
            } + product;
            *accumulator.add(out) = value;
            if finalize != 0 {
                *output.add(out) = bf16::from_f32(value);
            }
        }
    }
}

#[kernel(MoeSharedAdd)]
pub fn moe_shared_add(
    expert: *const bf16,
    shared: *const bf16,
    gate: *const bf16,
    output: *mut bf16,
    num_cols: u32,
    length: u32,
) {
    for position in 0..length as usize {
        unsafe {
            let gate = (*gate.add(position / num_cols as usize)).to_f32();
            let scale = bf16::from_f32(1.0 / (1.0 + (-gate).exp())).to_f32();
            let value = bf16::from_f32((*shared.add(position)).to_f32() * scale).to_f32();
            *output.add(position) = bf16::from_f32((*expert.add(position)).to_f32() + value);
        }
    }
}
