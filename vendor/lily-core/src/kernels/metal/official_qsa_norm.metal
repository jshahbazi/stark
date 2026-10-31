// Adapted by Stark from the vendored Lily QSA kernels; see UPSTREAM.md.
static inline void official_qsa_rope_pair_bf16(bfloat lo, bfloat hi, float angle,
                                      device bfloat* out_lo, device bfloat* out_hi) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const float c = float(bfloat(cos(angle)));
    const float s = float(bfloat(sin(angle)));
    const float l = float(lo);
    const float h = float(hi);
    *out_lo = bfloat(float(bfloat(l * c)) - float(bfloat(h * s)));
    *out_hi = bfloat(float(bfloat(h * c)) + float(bfloat(l * s)));
}

kernel void qsa_prep_q_bf16_mlx(device const bfloat* qk  [[buffer(0)]],  // [M, (NH+1)*D]
                            device const bfloat* w   [[buffer(1)]],  // [D]
                            device bfloat*       q   [[buffer(2)]],  // [M, NH, D]
                            constant uint&       D   [[buffer(3)]],
                            constant uint&       NH  [[buffer(4)]],
                            constant uint&       rot [[buffer(5)]],
                            constant uint&       base_pos [[buffer(6)]],
                            constant float&      theta [[buffer(7)]],
                            constant float&      eps [[buffer(8)]],
                            constant int&        rope_delta [[buffer(9)]],
                            uint seg  [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    const uint row = seg / NH;
    const uint h = seg % NH;
    const ulong src = (ulong)row * (NH + 1) * D + (ulong)h * D;
    const ulong dst = (ulong)seg * D;
    const float value = float(qk[src + tid]);
    official_rms_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float angle = qsa_rope_position(base_pos + row, rope_delta)
                          * pow(theta, -2.0f * float(tid) / float(rot));
        official_qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           q + dst + tid, q + dst + half_rot + tid);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }
}

kernel void qsa_prep_q_mrope_bf16_mlx(device const bfloat* qk  [[buffer(0)]],  // [M, (NH+1)*D]
                                  device const bfloat* w   [[buffer(1)]],  // [D]
                                  device bfloat*       q   [[buffer(2)]],  // [M, NH, D]
                                  device const uint*   positions [[buffer(3)]],  // [rows, 3]
                                  constant uint&       D   [[buffer(4)]],
                                  constant uint&       NH  [[buffer(5)]],
                                  constant uint&       rot [[buffer(6)]],
                                  constant uint&       base_pos [[buffer(7)]],
                                  constant float&      theta [[buffer(8)]],
                                  constant float&      eps [[buffer(9)]],
                                  constant uint&       pos_base [[buffer(10)]],
                                  constant uint&       sec_h_end [[buffer(11)]],
                                  constant uint&       sec_w_end [[buffer(12)]],
                                  uint seg  [[threadgroup_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]],
                                  uint sg   [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    const uint row = seg / NH;
    const uint h = seg % NH;
    const ulong src = (ulong)row * (NH + 1) * D + (ulong)h * D;
    const ulong dst = (ulong)seg * D;
    official_rms_row(float(qk[src + tid]), w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const uint axis = qsa_mrope_axis(tid, sec_h_end, sec_w_end);
        const uint pos = positions[(base_pos + row - pos_base) * 3 + axis];
        const float angle = float(pos) * pow(theta, -2.0f * float(tid) / float(rot));
        official_qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           q + dst + tid, q + dst + half_rot + tid);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }
}

kernel void qsa_block_keys_bf16_mlx(device const bfloat* cache [[buffer(0)]],  // [max_seq, D]
                                device const bfloat* w     [[buffer(1)]],  // [D]
                                device bfloat*       blk   [[buffer(2)]],  // [max_blocks, D]
                                constant uint&       D     [[buffer(3)]],
                                constant uint&       ratio [[buffer(4)]],
                                constant uint&       first_block [[buffer(5)]],
                                constant uint&       rot   [[buffer(6)]],
                                constant float&      theta [[buffer(7)]],
                                constant float&      eps   [[buffer(8)]],
                                constant uint&       count [[buffer(9)]],  // blocks to build (grid may exceed it)
                                constant int&        rope_delta [[buffer(10)]],
                                uint tg   [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    if (tg >= count) {
        return;  // uniform per threadgroup: no barrier is skipped unevenly
    }
    const uint b = first_block + tg;
    float pooled = 0.0f;
    for (uint i = 0; i < ratio; ++i) {
        pooled += float(cache[(ulong)(b * ratio + i) * D + tid]);
    }
    // The reference pools in f32 and rounds the mean to bf16 before the norm.
    const float value = float(bfloat(pooled / float(ratio)));
    official_rms_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const ulong dst = (ulong)b * D;
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float angle = qsa_rope_position(b * ratio, rope_delta)
                          * pow(theta, -2.0f * float(tid) / float(rot));
        official_qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           blk + dst + tid, blk + dst + half_rot + tid);
    } else if (tid >= rot) {
        blk[dst + tid] = normed[tid];
    }
}

kernel void qsa_block_keys_mrope_bf16_mlx(device const bfloat* cache [[buffer(0)]],  // [max_seq, D]
                                      device const bfloat* w     [[buffer(1)]],  // [D]
                                      device bfloat*       blk   [[buffer(2)]],  // [max_blocks, D]
                                      device const uint*   positions [[buffer(3)]],  // [rows, 3]
                                      constant uint&       D     [[buffer(4)]],
                                      constant uint&       ratio [[buffer(5)]],
                                      constant uint&       first_block [[buffer(6)]],
                                      constant uint&       rot   [[buffer(7)]],
                                      constant float&      theta [[buffer(8)]],
                                      constant float&      eps   [[buffer(9)]],
                                      constant uint&       count [[buffer(10)]],
                                      constant uint&       pos_base [[buffer(11)]],
                                      constant uint&       sec_h_end [[buffer(12)]],
                                      constant uint&       sec_w_end [[buffer(13)]],
                                      uint tg   [[threadgroup_position_in_grid]],
                                      uint tid  [[thread_index_in_threadgroup]],
                                      uint sg   [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    if (tg >= count) {
        return;  // uniform per threadgroup: no barrier is skipped unevenly
    }
    const uint b = first_block + tg;
    float pooled = 0.0f;
    for (uint i = 0; i < ratio; ++i) {
        pooled += float(cache[(ulong)(b * ratio + i) * D + tid]);
    }
    // The reference pools in f32 and rounds the mean to bf16 before the norm.
    const float value = float(bfloat(pooled / float(ratio)));
    official_rms_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const ulong dst = (ulong)b * D;
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const uint axis = qsa_mrope_axis(tid, sec_h_end, sec_w_end);
        const uint pos = positions[(b * ratio - pos_base) * 3 + axis];
        const float angle = float(pos) * pow(theta, -2.0f * float(tid) / float(rot));
        official_qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           blk + dst + tid, blk + dst + half_rot + tid);
    } else if (tid >= rot) {
        blk[dst + tid] = normed[tid];
    }
}
