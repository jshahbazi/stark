// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Modified for Stark. Routing, affine4 dots and eight-partial reduction are
// adapted from MLX-Serve 046b647686419df06c6c7132fb1d15d76185c3e5.
// The stable BF16 sigmoid follows MLX 64ea011cb65f14d9ce2737e60db9a4ae91ed7441,
// Copyright (c) 2023-2024 Apple Inc. See NOTICE and LICENSE.mlx-serve.
// Separate entry points: original Lily Metal bodies are unchanged.
#include <metal_stdlib>
using namespace metal;

static inline bfloat official_moe_sigmoid(bfloat x) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat ex = bfloat(precise::exp(abs(float(x))));
    const bfloat den = bfloat(1.0f + float(ex));
    const bfloat y = bfloat(1.0f / float(den));
    return float(x) < 0.0f ? y : bfloat(1.0f - float(y));
}

// Evaluated once per scratch owner, outside the token loop. Qualification
// compares all finite BF16 entries against the stock MLX primitive table.
kernel void official_moe_sigmoid_table(device bfloat* table [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    if (i < 65536) table[i] = official_moe_sigmoid(as_type<bfloat>(ushort(i)));
}

template <bool FoldGate>
static inline void official_moe_router_body(device const bfloat* logits,
    device uint* inds, device float* scores, device const bfloat* gx,
    device const bfloat* gw, device bfloat* glog, uint E, uint K, uint GK,
    uint row, uint tid, uint sg, uint lane, threadgroup float* rk,
    threadgroup uint* sel, threadgroup float* gp) {
#pragma METAL fp math_mode(safe)
    const ulong rbase = ulong(row) * E;
    for (uint e = tid; e < E; e += 256) rk[e] = float(logits[rbase + e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (FoldGate && tid >= 32) {
        float ga = 0.0f;
        for (uint k = tid - 32; k < GK; k += 224)
            ga += float(gx[ulong(row) * GK + k]) * float(gw[k]);
        ga = simd_sum(ga);
        if (lane == 0) gp[sg] = ga;
    }
    if (tid < 32) {
        // Served full-E softmax tree: virtual threads own four consecutive
        // values, then SIMD partials occupy lanes 0..SIMDS-1.
        const uint VT = (E + 3) / 4;
        const uint SIMDS = (VT + 31) / 32;
        float lane_max = -INFINITY;
        for (uint g = 0; g < SIMDS; ++g) {
            const uint vt = g * 32 + lane;
            float m = -INFINITY;
            for (uint i = 0; i < 4; ++i) {
                const uint e = vt * 4 + i;
                if (e < E) m = max(m, rk[e]);
            }
            const float mg = simd_max(m);
            if (lane == g) lane_max = mg;
        }
        const float smax = simd_max(lane_max);
        float lane_sum = 0.0f;
        for (uint g = 0; g < SIMDS; ++g) {
            const uint vt = g * 32 + lane;
            float acc = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                const uint e = vt * 4 + i;
                acc += e < E ? fast::exp(rk[e] - smax) : 0.0f;
            }
            const float pg = simd_sum(acc);
            if (lane == g) lane_sum = pg;
        }
        const float snorm = 1.0f / simd_sum(lane_sum);
        float mine[16];
        for (uint i = 0; i < 16; ++i) {
            const uint e = lane + 32 * i;
            mine[i] = e < E ? rk[e] : -INFINITY;
        }
        for (uint r = 0; r < K; ++r) {
            float best = -INFINITY;
            uint bslot = 0xFFFFFFFFu;
            for (uint i = 0; i < 16; ++i) {
                if (mine[i] > best) { best = mine[i]; bslot = i; }
            }
            const float gmax = simd_max(best);
            const uint cand = best == gmax ? lane + 32 * bslot : 0xFFFFFFFFu;
            const uint gidx = min(simd_min(cand), E - 1);
            if (gidx % 32 == lane) mine[gidx / 32] = -INFINITY;
            if (lane == 0) sel[r] = gidx;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float w = 0.0f;
        uint myidx = 0;
        if (lane < K) {
            myidx = sel[lane];
            w = float(bfloat(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
        }
        float tot = 0.0f;
        for (uint j = 0; j < K; ++j)
            tot = float(bfloat(simd_shuffle(w, ushort(j)) + tot));
        w = w / float(bfloat(tot));
        if (lane < K) {
            inds[ulong(row) * K + lane] = myidx;
            // Preserve native F32 storage while storing the exact BF16 score.
            scores[ulong(row) * K + lane] = float(bfloat(w));
        }
    }
    if (FoldGate) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float gt = 0.0f;
            for (uint g = 1; g < 8; ++g) gt += gp[g];
            glog[row] = bfloat(gt);
        }
    }
}

#define OFFICIAL_MOE_ROUTER(NAME, FOLD) \
kernel void NAME(device const bfloat* logits [[buffer(0)]], \
    device uint* inds [[buffer(1)]], device float* scores [[buffer(2)]], \
    device const bfloat* gx [[buffer(3)]], device const bfloat* gw [[buffer(4)]], \
    device bfloat* glog [[buffer(5)]], constant uint& E [[buffer(6)]], \
    constant uint& K [[buffer(7)]], constant uint& GK [[buffer(8)]], \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float rk[512]; threadgroup uint sel[32]; threadgroup float gp[8]; \
    official_moe_router_body<FOLD>(logits, inds, scores, gx, gw, glog, E, K, GK, \
                                  row, tid, sg, lane, rk, sel, gp); \
}
OFFICIAL_MOE_ROUTER(official_moe_router_bf16, false)
OFFICIAL_MOE_ROUTER(official_moe_router_gate_bf16, true)

// This is the served affine4 x16 dot's precise source ordering. Float
// contraction and simd_sum equivalence remain GPU-oracle qualification items.
static inline float official_moe_dot16(uint2 w, thread const float* x) {
#pragma METAL fp math_mode(safe)
    float a = 0.0f, b = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        a += x[j] * float((w.x >> (4 * j)) & 15u);
        b += x[8 + j] * float((w.y >> (4 * j)) & 15u);
    }
    return a + b;
}

kernel void official_moe_gate_up_a4(device const uint* gq [[buffer(0)]],
    device const bfloat* gs [[buffer(1)]], device const bfloat* gb [[buffer(2)]],
    device const uint* uq [[buffer(3)]], device const bfloat* us [[buffer(4)]],
    device const bfloat* ub [[buffer(5)]], device const bfloat* x [[buffer(6)]],
    device const uint* inds [[buffer(7)]], device const bfloat* table [[buffer(8)]],
    device bfloat* y [[buffer(9)]], constant uint& K [[buffer(10)]],
    constant uint& N [[buffer(11)]], constant uint& GS [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint n = group.x * 2 + sg;
    const uint e = group.y;
    const ulong row = ulong(inds[e]) * N + n;
    const uint KW2 = K / 16;
    const uint KG = K / GS;
    device const uint2* gw = (device const uint2*)gq + row * KW2 + lane;
    device const uint2* uw = (device const uint2*)uq + row * KW2 + lane;
    float ag = 0.0f, au = 0.0f;
    for (uint k = 0; k < K; k += 512) {
        const uint kk = k + lane * 16;
        if (kk < K) {
            float xr[16], xs = 0.0f;
            for (uint i = 0; i < 16; ++i) { xr[i] = float(x[kk + i]); xs += xr[i]; }
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
        y[ulong(e) * N + n] = bfloat(float(activated) * float(bfloat(u)));
    }
}

kernel void official_moe_down_a4(device const uint* dq [[buffer(0)]],
    device const bfloat* ds [[buffer(1)]], device const bfloat* db [[buffer(2)]],
    device const bfloat* act [[buffer(3)]], device const uint* inds [[buffer(4)]],
    device const float* scores [[buffer(5)]], device const bfloat* shared [[buffer(6)]],
    device const bfloat* glog [[buffer(7)]], device bfloat* y [[buffer(8)]],
    constant uint& I [[buffer(9)]], constant uint& H [[buffer(10)]],
    constant uint& GS [[buffer(11)]], constant uint& TOPK [[buffer(12)]],
    constant uint& HAS_SHARED [[buffer(13)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint row0 = group * 2 + sg;
    const uint KW2 = I / 16, KG = I / GS;
    float acc = 0.0f;
    for (uint e = 0; e < TOPK; ++e) {
        const ulong row = ulong(inds[e]) * H + row0;
        device const uint2* dw = (device const uint2*)dq + row * KW2 + lane;
        const float sc = float(bfloat(scores[e]));
        for (uint k = 0; k < I; k += 512) {
            const uint kk = k + lane * 16;
            if (kk < I) {
                float xr[16], xs = 0.0f;
                for (uint i = 0; i < 16; ++i) { xr[i] = float(act[ulong(e) * I + kk + i]); xs += xr[i]; }
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
            const bfloat sig = official_moe_sigmoid(glog[0]);
            const bfloat gated = bfloat(float(sig) * float(shared[row0]));
            out = bfloat(float(out) + float(gated));
        }
        y[row0] = out;
    }
}

kernel void official_moe_silu_mul(device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]], device const bfloat* table [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant uint& N [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (i >= N) return;
    const bfloat g = gate[i];
    const bfloat act = bfloat(float(g) * float(table[as_type<ushort>(g)]));
    y[i] = bfloat(float(act) * float(up[i]));
}

// The canonical K10/H2560 prefill rule, including short tails: score*down
// rounds to BF16, eight BF16 partials consume e%8, then partials fold 0..7.
kernel void official_moe_prefill_combine(device const bfloat* ed [[buffer(0)]],
    device const uint* slots [[buffer(1)]], device const float* scores [[buffer(2)]],
    device const bfloat* shared [[buffer(3)]], device const bfloat* gate [[buffer(4)]],
    device bfloat* y [[buffer(5)]], constant uint& H [[buffer(6)]],
    constant uint& TOPK [[buffer(7)]], uint2 i [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (i.x >= H) return;
    bfloat partial[8];
    for (uint p = 0; p < 8; ++p) partial[p] = bfloat(0.0f);
    for (uint e = 0; e < TOPK; ++e) {
        const uint at = i.y * TOPK + e;
        const bfloat product = bfloat(float(ed[ulong(slots[at]) * H + i.x]) * float(bfloat(scores[at])));
        partial[e % 8] = bfloat(float(product) + float(partial[e % 8]));
    }
    bfloat total = partial[0];
    for (uint p = 1; p < 8; ++p) total = bfloat(float(partial[p]) + float(total));
    const bfloat sig = official_moe_sigmoid(gate[i.y]);
    const bfloat gated = bfloat(float(sig) * float(shared[ulong(i.y) * H + i.x]));
    y[ulong(i.y) * H + i.x] = bfloat(float(total) + float(gated));
}
