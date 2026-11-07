//! Reference routing and accumulation for the bounded selected-expert path.

use half::bf16;
use stark_engine_macros::kernel;

#[kernel(GemmaMoeRouterScaleBF16)]
pub fn gemma_moe_router_scale_bf16(
    input: *const bf16,
    scales: *const bf16,
    output: *mut bf16,
    width: u32,
    length: u32,
    root_scale: f32,
) {
    for index in 0..length as usize {
        unsafe {
            let learned = bf16::from_f32((*input.add(index)).to_f32() * (*scales.add(index % width as usize)).to_f32());
            *output.add(index) = bf16::from_f32(learned.to_f32() * root_scale);
        }
    }
}

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
    round_accumulator: u32,
) {
    let weight = unsafe { *weights.add(weight_offset as usize) };
    for position in 0..length as usize {
        let out = output_offset as usize + position;
        unsafe {
            let mut product = (*input.add(position)).to_f32() * weight;
            if round_product != 0 {
                product = bf16::from_f32(product).to_f32();
            }
            let mut value = if initialize != 0 {
                0.0
            } else {
                *accumulator.add(out)
            } + product;
            if round_accumulator != 0 {
                value = bf16::from_f32(value).to_f32();
            }
            *accumulator.add(out) = value;
            if finalize != 0 {
                *output.add(out) = bf16::from_f32(value);
            }
        }
    }
}

/// Apply learned expert gains after top-k renormalization, and optionally sort
/// the selected entries by ID for source-ordered BF16 accumulation.
#[kernel(MoeAdjustRoutes)]
pub fn moe_adjust_routes(
    ids: *mut u32,
    weights: *mut f32,
    scales: *const f32,
    rows: u32,
    experts: u32,
    active: u32,
    sort_ids: u32,
) {
    for row in 0..rows as usize {
        let base = row * active as usize;
        for rank in 0..active as usize {
            unsafe {
                let id = *ids.add(base + rank);
                *weights.add(base + rank) = if id < experts {
                    *weights.add(base + rank) * *scales.add(id as usize)
                } else {
                    f32::NAN
                };
            }
        }
        if sort_ids != 0 {
            for rank in 1..active as usize {
                let mut index = rank;
                while index > 0 && unsafe { *ids.add(base + index) < *ids.add(base + index - 1) } {
                    unsafe {
                        std::ptr::swap(ids.add(base + index), ids.add(base + index - 1));
                        std::ptr::swap(weights.add(base + index), weights.add(base + index - 1));
                    }
                    index -= 1;
                }
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

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn learned_expert_gains_follow_renormalization_and_id_sort_keeps_weights_paired() {
        let mut ids = [3, 1, 2];
        let mut weights = [0.2f32, 0.3, 0.5];
        let scales = [-2.0f32, 0.5, 2.0, 4.0];
        moe_adjust_routes(ids.as_mut_ptr(), weights.as_mut_ptr(), scales.as_ptr(), 1, 4, 3, 1);
        assert_eq!(ids, [1, 2, 3]);
        assert_eq!(weights, [0.15, 1.0, 0.8]);
    }

    #[stark_test]
    fn gemma_expert_accumulation_materializes_each_bf16_partial_sum() {
        let weights = [1.0f32; 3];
        let inputs = [bf16::from_f32(256.0), bf16::ONE, bf16::from_f32(-256.0)];
        let mut results = [bf16::NAN; 2];
        for round in 0..2u32 {
            let mut accumulator = [f32::NAN];
            for rank in 0..3 {
                moe_accumulate(
                    &inputs[rank],
                    weights.as_ptr(),
                    accumulator.as_mut_ptr(),
                    &mut results[round as usize],
                    1,
                    rank as u32,
                    0,
                    u32::from(rank == 0),
                    u32::from(rank == 2),
                    1,
                    round,
                );
            }
        }
        assert_eq!(results[0], bf16::ONE);
        assert_eq!(results[1], bf16::ZERO);
    }
}
