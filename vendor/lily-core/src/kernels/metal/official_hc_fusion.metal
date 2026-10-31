
// SPDX-License-Identifier: Apache-2.0
// Modified by Stark: exact-boundary fusion of the materialized official HC
// read. This source is concatenated AFTER the unchanged skinny.metal source,
// and reuses its unpack/load helpers. The original register-A reduction order
// is retained; normalization remains materialized by official_hc.metal.

// One register-A Q8 pair at M=1: identical block/word/metadata association to
// gemm_skinny_q8_reg_body<1, bfloat>. Every 16-element block first accumulates
// qx/xs across both eight-element word pairs, then applies scale/bias once.
static inline float2 official_hc_skinny_pair(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* a,
    uint K, uint GS, uint row0, uint lane) {
    const uint row1 = row0 + 1;
    const uint words = K / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;
    device const uint4* w0 = (device const uint4*)(codes + ulong(row0) * words);
    device const uint4* w1 = (device const uint4*)(codes + ulong(row1) * words);
    float acc0[1] = {0.0f};
    float acc1[1] = {0.0f};
    for (uint blk = lane; blk < blocks; blk += 32) {
        const uint g = blk / bpg;
        const float s0 = float(scales[row0 * groups + g]);
        const float b0 = float(biases[row0 * groups + g]);
        const float s1 = float(scales[row1 * groups + g]);
        const float b1 = float(biases[row1 * groups + g]);
        const uint4 v0 = w0[blk];
        const uint4 v1 = w1[blk];
        float qx0[1] = {0.0f};
        float qx1[1] = {0.0f};
        float xs[1] = {0.0f};
        for (uint wp = 0; wp < 2; ++wp) {
            const float4 q0lo = skinny_unpack_q8(v0[2 * wp]);
            const float4 q0hi = skinny_unpack_q8(v0[2 * wp + 1]);
            const float4 q1lo = skinny_unpack_q8(v1[2 * wp]);
            const float4 q1hi = skinny_unpack_q8(v1[2 * wp + 1]);
            for (uint i = 0; i < 1; ++i) {
                float4 xlo, xhi;
                skinny_load_x8(a + ulong(i) * K + blk * 16 + wp * 8, a_vec, xlo, xhi);
                xs[i] += dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
                qx0[i] += dot(q0lo, xlo) + dot(q0hi, xhi);
                qx1[i] += dot(q1lo, xlo) + dot(q1hi, xhi);
            }
        }
        for (uint i = 0; i < 1; ++i) {
            acc0[i] += s0 * qx0[i] + b0 * xs[i];
            acc1[i] += s1 * qx1[i] + b1 * xs[i];
        }
    }
    return float2(simd_sum(acc0[0]), simd_sum(acc1[0]));
}

// The F32 projection sum enters the safe helper BEFORE the BF16 cast.
// This prevents fast math in the dot scope from eliding projection rounding
// when the same value feeds the activation.
static inline bfloat2 official_hc_finish_down(float sum) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat projected = bfloat(sum);
    const bfloat scaled = bfloat(float(projected) * 0.25f);
    const bfloat gate = bfloat(1.0f / (1.0f + exp(-float(scaled))));
    const bfloat act = bfloat(float(scaled) * float(gate));
    return bfloat2(projected, act);
}

kernel void official_hc_down_act_q8_m1(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* hn [[buffer(3)]],
    device bfloat* down [[buffer(4)]],
    device bfloat* act [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // The original register-A dispatch has two simdgroups, two rows each.
    const uint row0 = (tg * 2 + sg) * 2;
    const float2 sums = official_hc_skinny_pair(codes, scales, biases, hn, 10240, 64, row0, lane);
    if (lane == 0) {
        const bfloat2 first = official_hc_finish_down(sums.x);
        const bfloat2 second = official_hc_finish_down(sums.y);
        down[row0] = first.x;
        act[row0] = first.y;
        down[row0 + 1] = second.x;
        act[row0 + 1] = second.y;
    }
}

static inline bfloat official_hc_finish_mix(
    thread const float* logits, device const bfloat* hn, uint col) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    float sum = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        const bfloat logit = bfloat(logits[g]);
        const bfloat gate = bfloat(1.0f / (1.0f + exp(-float(logit))));
        const bfloat product = bfloat(float(gate) * float(hn[g * 2560 + col]));
        sum += float(product);
    }
    return bfloat(float(bfloat(sum)) * float(bfloat(0.25f)));
}

kernel void official_hc_up_mix_q8_m1(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* act [[buffer(3)]],
    device const bfloat* hn [[buffer(4)]],
    device bfloat* mixed [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // Four simdgroups each own two adjacent columns, preserving the original
    // pairing of adjacent up-projection rows inside every stream.
    const uint col = (tg * 4 + sg) * 2;
    float first[4];
    float second[4];
    for (uint g = 0; g < 4; ++g) {
        const float2 sums = official_hc_skinny_pair(
            codes, scales, biases, act, 320, 64, g * 2560 + col, lane);
        first[g] = sums.x;
        second[g] = sums.y;
    }
    if (lane == 0) {
        mixed[col] = official_hc_finish_mix(first, hn, col);
        mixed[col + 1] = official_hc_finish_mix(second, hn, col + 1);
    }
}

// Synthetic test-only dispatch entry: adversarial F32 projection boundaries
// and signed zeros exercise the same production helper against a separately
// materialized BF16 input. No production graph calls this entry point.
kernel void official_hc_down_tail_rounding_control(
    device const float* sums [[buffer(0)]],
    device bfloat* down [[buffer(1)]],
    device bfloat* act [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const bfloat2 result = official_hc_finish_down(sums[gid]);
    down[gid] = result.x;
    act[gid] = result.y;
}
