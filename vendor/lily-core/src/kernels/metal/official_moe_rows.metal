// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Modified for Stark: the qualified MLX-Serve-derived single-row bodies in
// official_moe.metal, with an independent verifier-row address axis only.
// See official_moe.metal, NOTICE and LICENSE.mlx-serve for provenance.

kernel void official_moe_gate_up_a4_rows(device const uint* gq [[buffer(0)]],
    device const bfloat* gs [[buffer(1)]], device const bfloat* gb [[buffer(2)]],
    device const uint* uq [[buffer(3)]], device const bfloat* us [[buffer(4)]],
    device const bfloat* ub [[buffer(5)]], device const bfloat* x [[buffer(6)]],
    device const uint* inds [[buffer(7)]], device const bfloat* table [[buffer(8)]],
    device bfloat* y [[buffer(9)]], constant uint& K [[buffer(10)]],
    constant uint& N [[buffer(11)]], constant uint& GS [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint n = group.x * 2 + sg;
    const uint e = group.y;
    const uint batch = group.z;
    const ulong row = ulong(inds[ulong(batch) * 10 + e]) * N + n;
    const uint KW2 = K / 16;
    const uint KG = K / GS;
    device const uint2* gw = (device const uint2*)gq + row * KW2 + lane;
    device const uint2* uw = (device const uint2*)uq + row * KW2 + lane;
    float ag = 0.0f, au = 0.0f;
    for (uint k = 0; k < K; k += 512) {
        const uint kk = k + lane * 16;
        if (kk < K) {
            float xr[16], xs = 0.0f;
            for (uint i = 0; i < 16; ++i) { xr[i] = float(x[ulong(batch) * K + kk + i]); xs += xr[i]; }
            const ulong at = row * KG + kk / GS;
            ag += float(gs[at]) * official_moe_dot16(gw[k / 16], xr) + float(gb[at]) * xs;
            au += float(us[at]) * official_moe_dot16(uw[k / 16], xr) + float(ub[at]) * xs;
        }
    }
    const float g = simd_sum(ag), u = simd_sum(au);
    if (lane == 0) {
#pragma METAL fp contract(off)
        const bfloat gt = bfloat(g);
        const bfloat activated = bfloat(float(gt) * float(table[as_type<ushort>(gt)]));
        y[(ulong(batch) * 10 + e) * N + n] = bfloat(float(activated) * float(bfloat(u)));
    }
}

kernel void official_moe_down_a4_rows(device const uint* dq [[buffer(0)]],
    device const bfloat* ds [[buffer(1)]], device const bfloat* db [[buffer(2)]],
    device const bfloat* act [[buffer(3)]], device const uint* inds [[buffer(4)]],
    device const float* scores [[buffer(5)]], device const bfloat* shared [[buffer(6)]],
    device const bfloat* glog [[buffer(7)]], device bfloat* y [[buffer(8)]],
    constant uint& I [[buffer(9)]], constant uint& H [[buffer(10)]],
    constant uint& GS [[buffer(11)]], constant uint& TOPK [[buffer(12)]],
    constant uint& HAS_SHARED [[buffer(13)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint row0 = group.x * 2 + sg;
    const uint batch = group.y;
    const uint KW2 = I / 16, KG = I / GS;
    float acc = 0.0f;
    for (uint e = 0; e < TOPK; ++e) {
        const ulong row = ulong(inds[ulong(batch) * TOPK + e]) * H + row0;
        device const uint2* dw = (device const uint2*)dq + row * KW2 + lane;
        const float sc = float(bfloat(scores[ulong(batch) * TOPK + e]));
        for (uint k = 0; k < I; k += 512) {
            const uint kk = k + lane * 16;
            if (kk < I) {
                float xr[16], xs = 0.0f;
                for (uint i = 0; i < 16; ++i) { xr[i] = float(act[(ulong(batch) * TOPK + e) * I + kk + i]); xs += xr[i]; }
                const ulong at = row * KG + kk / GS;
                acc += sc * (float(ds[at]) * official_moe_dot16(dw[k / 16], xr) + float(db[at]) * xs);
            }
        }
    }
    const float v = simd_sum(acc);
    if (lane == 0) {
#pragma METAL fp contract(off)
        bfloat out = bfloat(v);
        if (HAS_SHARED) {
            const bfloat sig = official_moe_sigmoid(glog[batch]);
            const bfloat gated = bfloat(float(sig) * float(shared[ulong(batch) * H + row0]));
            out = bfloat(float(out) + float(gated));
        }
        y[ulong(batch) * H + row0] = out;
    }
}
