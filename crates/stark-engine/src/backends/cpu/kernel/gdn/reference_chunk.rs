// Pinned HF Qwen3.5 torch_chunk_gated_delta_rule (64-token blocks), revision
// 2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397; Apache-2.0 distribution notice.
use half::bf16;
use stark_engine_macros::kernel;

use crate::backends::common::vision_math::vision_exp;

#[kernel(ReferenceGDNChunkPrepare)]
pub fn reference_gdn_chunk_prepare(
    q_input: *const f32,
    k_input: *const f32,
    beta: *const f32,
    log_decay: *const f32,
    values: *const bf16,
    q: *mut f32,
    k: *mut f32,
    kb: *mut f32,
    vb: *mut f32,
    cumulative: *mut f32,
    rows: u32,
    heads: u32,
    groups: u32,
    key_dim: u32,
    value_dim: u32,
    row_stride: u32,
) {
    let _ = value_dim;
    for plane in 0..rows.div_ceil(64) as usize * heads as usize {
        let chunk = plane / heads as usize;
        let head = plane % heads as usize;
        let group = head / (heads / groups) as usize;
        let mut sum = 0.0f64;
        for i in 0..64 {
            let row = chunk * 64 + i;
            let active = row < rows as usize;
            sum += f64::from(if active {
                unsafe { *log_decay.add(row * heads as usize + head) }
            } else {
                0.0
            });
            unsafe { *cumulative.add(plane * 64 + i) = sum as f32 };
            let b = if active {
                unsafe { *beta.add(row * heads as usize + head) }
            } else {
                0.0
            };
            for channel in 0..128 {
                let dst = (plane * 64 + i) * 128 + channel;
                let kval = if active {
                    unsafe { *k_input.add(row * key_dim as usize + group * 128 + channel) }
                } else {
                    0.0
                };
                unsafe {
                    *q.add(dst) = if active {
                        *q_input.add(row * key_dim as usize + group * 128 + channel)
                    } else {
                        0.0
                    };
                    *k.add(dst) = kval;
                    *kb.add(dst) = kval * b;
                    *vb.add(dst) = if active {
                        (*values.add(row * row_stride as usize + 2 * key_dim as usize + head * 128 + channel)).to_f32()
                            * b
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

#[kernel(ReferenceGDNChunkGram)]
pub fn reference_gdn_chunk_gram(
    q: *const f32,
    k: *const f32,
    kb: *const f32,
    cumulative: *const f32,
    triangular: *mut f32,
    attention: *mut f32,
    planes: u32,
) {
    for plane in 0..planes as usize {
        for row in 0..64 {
            for column in 0..64 {
                let index = plane * 4096 + row * 64 + column;
                let (mut kk, mut qk) = (0.0f32, 0.0f32);
                if column <= row {
                    for inner in 0..128 {
                        unsafe {
                            let b = *k.add((plane * 64 + column) * 128 + inner);
                            kk = (*kb.add((plane * 64 + row) * 128 + inner)).mul_add(b, kk);
                            qk = (*q.add((plane * 64 + row) * 128 + inner)).mul_add(b, qk);
                        }
                    }
                    let pairwise =
                        vision_exp(unsafe { *cumulative.add(plane * 64 + row) - *cumulative.add(plane * 64 + column) });
                    kk *= pairwise;
                    qk *= pairwise;
                }
                unsafe {
                    *triangular.add(index) = kk;
                    *attention.add(index) = qk;
                }
            }
        }
    }
}

#[kernel(ReferenceGDNChunkSolve)]
pub fn reference_gdn_chunk_solve(
    triangular: *const f32,
    kb: *const f32,
    vb: *const f32,
    cumulative: *const f32,
    u: *mut f32,
    w: *mut f32,
    planes: u32,
) {
    for plane in 0..planes as usize {
        for channel in 0..128 {
            for row in 0..64 {
                let dst = (plane * 64 + row) * 128 + channel;
                let mut left = unsafe { *vb.add(dst) };
                let mut right = unsafe { *kb.add(dst) } * vision_exp(unsafe { *cumulative.add(plane * 64 + row) });
                for column in 0..row {
                    unsafe {
                        let coefficient = -*triangular.add(plane * 4096 + row * 64 + column);
                        left = coefficient.mul_add(*u.add((plane * 64 + column) * 128 + channel), left);
                        right = coefficient.mul_add(*w.add((plane * 64 + column) * 128 + channel), right);
                    }
                }
                unsafe {
                    *u.add(dst) = left;
                    *w.add(dst) = right;
                }
            }
        }
    }
}

#[kernel(ReferenceGDNChunkNewValues)]
pub fn reference_gdn_chunk_new_values(
    u: *const f32,
    w: *const f32,
    state: *const f32,
    vnew: *mut f32,
    heads: u32,
    chunk: u32,
) {
    for head in 0..heads as usize {
        for row in 0..64 {
            for dv in 0..128 {
                let plane = chunk as usize * heads as usize + head;
                let mut product = 0.0f32;
                for dk in 0..128 {
                    unsafe {
                        product = (*w.add((plane * 64 + row) * 128 + dk))
                            .mul_add(*state.add((head * 128 + dv) * 128 + dk), product);
                    }
                }
                unsafe {
                    *vnew.add((head * 64 + row) * 128 + dv) = *u.add((plane * 64 + row) * 128 + dv) - product;
                }
            }
        }
    }
}

#[kernel(ReferenceGDNChunkOutput)]
pub fn reference_gdn_chunk_output(
    q: *const f32,
    attention: *const f32,
    cumulative: *const f32,
    state: *const f32,
    vnew: *const f32,
    output: *mut bf16,
    heads: u32,
    chunk: u32,
    rows: u32,
) {
    for head in 0..heads as usize {
        for row in 0..64 {
            if chunk as usize * 64 + row >= rows as usize {
                continue;
            }
            let plane = chunk as usize * heads as usize + head;
            let decay = vision_exp(unsafe { *cumulative.add(plane * 64 + row) });
            for dv in 0..128 {
                let (mut inter, mut intra) = (0.0f32, 0.0f32);
                for dk in 0..128 {
                    unsafe {
                        let query = *q.add((plane * 64 + row) * 128 + dk) * decay;
                        inter = query.mul_add(*state.add((head * 128 + dv) * 128 + dk), inter);
                    }
                }
                for inner in 0..64 {
                    unsafe {
                        intra = (*attention.add(plane * 4096 + row * 64 + inner))
                            .mul_add(*vnew.add((head * 64 + inner) * 128 + dv), intra);
                    }
                }
                unsafe {
                    *output.add(((chunk as usize * 64 + row) * heads as usize + head) * 128 + dv) =
                        bf16::from_f32(inter + intra);
                }
            }
        }
    }
}

#[kernel(ReferenceGDNChunkState)]
pub fn reference_gdn_chunk_state(
    k: *const f32,
    cumulative: *const f32,
    vnew: *const f32,
    state: *mut f32,
    heads: u32,
    chunk: u32,
) {
    for head in 0..heads as usize {
        let plane = chunk as usize * heads as usize + head;
        let end = unsafe { *cumulative.add(plane * 64 + 63) };
        for dv in 0..128 {
            for dk in 0..128 {
                let mut update = 0.0f32;
                for row in 0..64 {
                    unsafe {
                        let key =
                            *k.add((plane * 64 + row) * 128 + dk) * vision_exp(end - *cumulative.add(plane * 64 + row));
                        update = key.mul_add(*vnew.add((head * 64 + row) * 128 + dv), update);
                    }
                }
                let index = (head * 128 + dv) * 128 + dk;
                unsafe {
                    *state.add(index) = *state.add(index) * vision_exp(end) + update;
                }
            }
        }
    }
}
