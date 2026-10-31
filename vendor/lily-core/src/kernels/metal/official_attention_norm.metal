// Adapted by Stark from the vendored Lily attention kernels; see UPSTREAM.md.
// Explicit official BF16-gain policy; original entrypoints stay unchanged.
kernel void q_norm_rope_split_decode_bf16_mlx(
    device const bfloat* qg   [[buffer(0)]],
    device const bfloat* w    [[buffer(1)]],
    device bfloat*       q    [[buffer(2)]],
    device bfloat*       gate [[buffer(3)]],
    constant uint&       D    [[buffer(4)]],
    constant uint&       rot  [[buffer(5)]],
    constant uint&       pos  [[buffer(6)]],
    constant float&      theta [[buffer(7)]],
    constant float&      eps   [[buffer(8)]],
    constant int&        rope_delta [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms = 0.0f;
    threadgroup bfloat normed[256];
    const ulong src = (ulong)head * 2 * D;
    const ulong dst = (ulong)head * D;
    const float value = float(qg[src + tid]);
    gate[dst + tid] = qg[src + D + tid];
    official_rms_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = rope_position(pos, rope_delta) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        q[dst + tid] = bfloat(lo * c - hi * s);
        q[dst + half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }
}

kernel void k_norm_rope_scatter_decode_bf16_mlx(
    device const bfloat* k     [[buffer(0)]],
    device const bfloat* w     [[buffer(1)]],
    device bfloat*       cache [[buffer(2)]],
    constant uint&       D     [[buffer(3)]],
    constant uint&       rot   [[buffer(4)]],
    constant uint&       pos   [[buffer(5)]],
    constant float&      theta [[buffer(6)]],
    constant float&      eps   [[buffer(7)]],
    constant uint&       max_seq [[buffer(8)]],
    constant int&        rope_delta [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms = 0.0f;
    threadgroup bfloat normed[256];
    const ulong src = (ulong)head * D;
    const ulong dst = ((ulong)head * max_seq + pos) * D;
    const float value = float(k[src + tid]);
    official_rms_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = rope_position(pos, rope_delta) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        cache[dst + tid] = bfloat(lo * c - hi * s);
        cache[dst + half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        cache[dst + tid] = normed[tid];
    }
}

kernel void k_norm_rope_scatter_decode_q8_mlx(
    device const bfloat* k      [[buffer(0)]],
    device const bfloat* w      [[buffer(1)]],
    device char*         cache  [[buffer(2)]],
    device half*         scales [[buffer(3)]],
    constant uint&       D      [[buffer(4)]],
    constant uint&       rot    [[buffer(5)]],
    constant uint&       pos    [[buffer(6)]],
    constant float&      theta  [[buffer(7)]],
    constant float&      eps    [[buffer(8)]],
    constant uint&       max_seq [[buffer(9)]],
    constant int&        rope_delta [[buffer(10)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms = 0.0f;
    threadgroup bfloat normed[256];
    threadgroup bfloat roped[256];
    const ulong src = (ulong)head * D;
    const ulong row = (ulong)head * max_seq + pos;
    const float value = float(k[src + tid]);
    official_rms_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = rope_position(pos, rope_delta) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        roped[tid] = bfloat(lo * c - hi * s);
        roped[half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        roped[tid] = normed[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv_q8_store(float(roped[tid]), cache + row * D + tid,
                scales + row * (D / KV_Q8_GROUP) + tid / KV_Q8_GROUP, lane);
}
