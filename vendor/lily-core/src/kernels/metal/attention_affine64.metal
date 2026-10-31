// Adapted by Stark from the vendored Lily attention kernels; see UPSTREAM.md.
// All threads of one 256-thread group write one D=256 cache row.
kernel void scatter_kv_affine64(device uchar* cache [[buffer(0)]],
                                 device bfloat* metadata [[buffer(1)]],
                                 device const bfloat* rows [[buffer(2)]],
                                 constant uint& D [[buffer(3)]],
                                 constant uint& max_seq [[buffer(4)]],
                                 constant uint& base_pos [[buffer(5)]],
                                 constant uint& kvh [[buffer(6)]],
                                 uint row_index [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float minima[8];
    threadgroup float maxima[8];
    const uint h = row_index % kvh;
    const uint m = row_index / kvh;
    const ulong row = (ulong)h * max_seq + base_pos + m;
    kv_affine64_store(float(rows[(ulong)row_index * D + tid]),
        cache + row * D + tid,
        metadata + row * KV_AFFINE64_META_PER_ROW(D) + 2u * (tid / 64u),
        minima, maxima, tid, sg, lane);
}

kernel void kv_dequant_affine64(device const uchar* cache [[buffer(0)]],
                                 device const bfloat* metadata [[buffer(1)]],
                                 device bfloat* out [[buffer(2)]],
                                 constant uint& D [[buffer(3)]],
                                 constant uint& max_seq [[buffer(4)]],
                                 constant uint& len [[buffer(5)]],
                                 constant uint& out_cap [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint dim = gid % D;
    const uint token = (gid / D) % len;
    const uint head = gid / (D * len);
    const ulong row = (ulong)head * max_seq + token;
    out[((ulong)head * out_cap + token) * D + dim] = kv_affine64_at(
        cache + row * D, metadata + row * KV_AFFINE64_META_PER_ROW(D), dim);
}

kernel void k_norm_rope_scatter_decode_affine64(
    device const bfloat* k      [[buffer(0)]],
    device const bfloat* w      [[buffer(1)]],
    device uchar*         cache  [[buffer(2)]],
    device bfloat*         scales [[buffer(3)]],
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
    threadgroup float partial[8];
    threadgroup float minima[8];
    threadgroup float maxima[8];
    threadgroup float inv_rms = 0.0f;
    threadgroup bfloat normed[256];
    threadgroup bfloat roped[256];
    const ulong src = (ulong)head * D;
    const ulong row = (ulong)head * max_seq + pos;
    const float value = float(k[src + tid]);
    float acc = value * value;
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(float(bfloat(value * inv_rms)) * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    kv_affine64_store(float(roped[tid]), cache + row * D + tid,
        scales + row * KV_AFFINE64_META_PER_ROW(D) + 2u * (tid / 64u),
        minima, maxima, tid, sg, lane);
}

kernel void sdpa_decode_split_affine64(device const bfloat* q        [[buffer(0)]],  // [NQ, D]
                                 device const uchar*   k_cache  [[buffer(1)]],
                                 device const bfloat*   k_scales [[buffer(2)]],
                                 device const uchar*   v_cache  [[buffer(3)]],
                                 device const bfloat*   v_scales [[buffer(4)]],
                                 device float*        partials [[buffer(5)]],  // [NQ, S, D]
                                 device float*        stats    [[buffer(6)]],  // [NQ, S, 2]
                                 constant uint&       D        [[buffer(7)]],
                                 constant uint&       max_seq  [[buffer(8)]],
                                 constant uint&       len      [[buffer(9)]],
                                 constant uint&       splits   [[buffer(10)]],
                                 constant uint&       group    [[buffer(11)]],
                                 constant float&      scale    [[buffer(12)]],
                                 uint2 tg  [[threadgroup_position_in_grid]],
                                 uint tid  [[thread_index_in_threadgroup]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[SDPA_SPLIT];
    threadgroup float q_s[TG];
    threadgroup float part[TG / 32];
    threadgroup float red;
    threadgroup float v_stage[(TG / 32) * 256];

    const uint hq = tg.x;
    const uint split = tg.y;
    const uint groups = D / KV_Q8_GROUP;
    const ulong head_row = (ulong)(hq / group) * max_seq;
    device const uchar* k_head = k_cache + head_row * D;
    device const uchar* v_head = v_cache + head_row * D;
    device const bfloat* ks_head = k_scales + head_row * groups;
    device const bfloat* vs_head = v_scales + head_row * groups;
    const uint base = split * SDPA_SPLIT;
    const uint count = base < len ? min(uint(SDPA_SPLIT), len - base) : 0;

    if (count == 0) {
        if (tid == 0) {
            stats[(hq * splits + split) * 2] = -INFINITY;
            stats[(hq * splits + split) * 2 + 1] = 0.0f;
        }
        return;
    }

    if (tid < D) {
        q_s[tid] = float(q[hq * D + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    const float4 qa(q_s[lane * 8], q_s[lane * 8 + 1], q_s[lane * 8 + 2],
                    q_s[lane * 8 + 3]);
    const float4 qb(q_s[lane * 8 + 4], q_s[lane * 8 + 5], q_s[lane * 8 + 6],
                    q_s[lane * 8 + 7]);
    for (uint p = sg; p < count; p += TG / 32) {
        const ulong t = base + p;
        const uint4 kq = kv_affine64_load8(k_head + t * D, ks_head + t * groups, lane * 8);
        // Preserve the BF16 reader's F32 accumulation expression/order.
        float dot = 0.0f;
        dot += metal::dot(float4(as_type<bfloat4>(kq.xy)), qa);
        dot += metal::dot(float4(as_type<bfloat4>(kq.zw)), qb);
        dot = simd_sum(dot);
        float s = dot * scale;
        if (lane == 0) {
            scores[p] = s;
        }
        local_max = max(local_max, s);
    }
    local_max = simd_max(local_max);
    if (lane == 0) { part[sg] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < TG / 32; ++i) { m = max(m, part[i]); }
        red = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float chunk_max = red;

    float e = 0.0f;
    if (tid < count) {
        e = exp(scores[tid] - chunk_max);
        scores[tid] = e;
    }
    float local_sum = simd_sum(e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { part[sg] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) { s += part[i]; }
        stats[(hq * splits + split) * 2] = chunk_max;
        stats[(hq * splits + split) * 2 + 1] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint p = sg; p < count; p += TG / 32) {
        const ulong t = base + p;
        const float wgt = scores[p];
        const uint4 vq = kv_affine64_load8(v_head + t * D, vs_head + t * groups, lane * 8);
        acc0 += wgt * float4(as_type<bfloat4>(vq.xy));
        acc1 += wgt * float4(as_type<bfloat4>(vq.zw));
    }
    for (uint j = 0; j < 4; ++j) {
        v_stage[sg * D + lane * 8 + j] = acc0[j];
        v_stage[sg * D + lane * 8 + 4 + j] = acc1[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < D; d += TG) {
        float acc = 0.0f;
        for (uint s = 0; s < TG / 32; ++s) {
            acc += v_stage[s * D + d];
        }
        partials[((ulong)hq * splits + split) * D + d] = acc;
    }
}


kernel void k_norm_rope_scatter_decode_affine64_mlx(
    device const bfloat* k      [[buffer(0)]],
    device const bfloat* w      [[buffer(1)]],
    device uchar*         cache  [[buffer(2)]],
    device bfloat*         scales [[buffer(3)]],
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
    threadgroup float minima[8];
    threadgroup float maxima[8];
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
    kv_affine64_store(float(roped[tid]), cache + row * D + tid,
        scales + row * KV_AFFINE64_META_PER_ROW(D) + 2u * (tid / 64u),
        minima, maxima, tid, sg, lane);
}
