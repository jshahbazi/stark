// SPDX-License-Identifier: Apache-2.0
// Stark: row-batched entry points retaining ordinary decode's exact bodies.
// Compiled after moe.metal, whose router and down/combine helpers are reused.

#define STARK_MOE_ROUTER_DECODE_ROWS(EPL)                                    \
kernel void stark_moe_router_decode_rows_e##EPL(                             \
    device const float* logits [[buffer(0)]],                                \
    device uint* indices [[buffer(1)]],                                      \
    device float* scores [[buffer(2)]],                                      \
    constant uint& E [[buffer(3)]],                                          \
    constant uint& K [[buffer(4)]],                                          \
    constant uint& renorm [[buffer(5)]],                                     \
    uint row [[threadgroup_position_in_grid]],                               \
    uint tid [[thread_index_in_threadgroup]],                                \
    uint tg_size [[threads_per_threadgroup]],                                \
    uint sg [[simdgroup_index_in_threadgroup]],                              \
    uint lane [[thread_index_in_simdgroup]]) {                               \
    threadgroup float probs[MOE_MAX_E];                                     \
    threadgroup float red[32];                                              \
    threadgroup float sel_val[MOE_ROUTER_MAX_K];                             \
    threadgroup uint sel_idx[MOE_ROUTER_MAX_K];                              \
    moe_router_topk_body<EPL>(logits + (ulong)row * E,                        \
        indices + (ulong)row * K, scores + (ulong)row * K,                   \
        E, K, renorm, tid, tg_size, sg, lane, probs, red, sel_val, sel_idx); \
}

STARK_MOE_ROUTER_DECODE_ROWS(8)
STARK_MOE_ROUTER_DECODE_ROWS(16)
STARK_MOE_ROUTER_DECODE_ROWS(32)

kernel void stark_moe_down_combine_decode_rows(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device const uint* indices [[buffer(4)]],
    device const float* scores [[buffer(5)]],
    device const bfloat* shared [[buffer(6)]],
    device const float* gate [[buffer(7)]],
    device bfloat* out [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& GS [[buffer(10)]],
    constant uint& H [[buffer(11)]],
    constant uint& S [[buffer(12)]],
    constant uint& has_shared [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xchg[4 * 2 * (MOE_MAX_K / 4 + 1)];
    const uint row = tg.x;
    moe_down_combine_body<16, 4>(codes, scales, biases,
        x + (ulong)row * S * K, indices + (ulong)row * S,
        scores + (ulong)row * S, shared + (ulong)row * H, gate + row,
        out + (ulong)row * H, K, GS, H, S, has_shared, tg.y, sg, lane, xchg);
}

kernel void stark_moe_gate_up_decode_rows(
    device const uint* gate_codes [[buffer(0)]],
    device const bfloat* gate_scales [[buffer(1)]],
    device const bfloat* gate_biases [[buffer(2)]],
    device const uint* up_codes [[buffer(3)]],
    device const bfloat* up_scales [[buffer(4)]],
    device const bfloat* up_biases [[buffer(5)]],
    device const bfloat* x [[buffer(6)]],
    device const uint* indices [[buffer(7)]],
    device bfloat* y [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& GS [[buffer(10)]],
    constant uint& N [[buffer(11)]],
    constant uint& TOPK [[buffer(12)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint row = tg.y;
    const uint slot = tg.x * TOPK + tg.z;
    const ulong grow = (ulong)indices[slot] * N + row;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    device const uint4* gate_wrow =
        (device const uint4*)(gate_codes + grow * words);
    device const uint4* up_wrow =
        (device const uint4*)(up_codes + grow * words);
    device const bfloat4* xv = (device const bfloat4*)(x + (ulong)(slot / TOPK) * K);
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint i = lane; i < blocks; i += 32) {
        uint g = i / bpg;
        uint4 gate_w4 = gate_wrow[i];
        uint4 up_w4 = up_wrow[i];
        float gate_dot = 0.0f;
        float up_dot = 0.0f;
        float x_sum = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            float4 xlo = float4(xv[2 * (4 * i + j)]);
            float4 xhi = float4(xv[2 * (4 * i + j) + 1]);
            uint gw = gate_w4[j];
            uint uw = up_w4[j];
            float4 gqlo = float4(float((gw >> 0) & 0xF), float((gw >> 4) & 0xF),
                                 float((gw >> 8) & 0xF), float((gw >> 12) & 0xF));
            float4 gqhi = float4(float((gw >> 16) & 0xF), float((gw >> 20) & 0xF),
                                 float((gw >> 24) & 0xF), float((gw >> 28) & 0xF));
            float4 uqlo = float4(float((uw >> 0) & 0xF), float((uw >> 4) & 0xF),
                                 float((uw >> 8) & 0xF), float((uw >> 12) & 0xF));
            float4 uqhi = float4(float((uw >> 16) & 0xF), float((uw >> 20) & 0xF),
                                 float((uw >> 24) & 0xF), float((uw >> 28) & 0xF));
            gate_dot += dot(gqlo, xlo) + dot(gqhi, xhi);
            up_dot += dot(uqlo, xlo) + dot(uqhi, xhi);
            x_sum += dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
        }
        gate_sum += float(gate_scales[grow * groups + g]) * gate_dot
                  + float(gate_biases[grow * groups + g]) * x_sum;
        up_sum += float(up_scales[grow * groups + g]) * up_dot
                + float(up_biases[grow * groups + g]) * x_sum;
    }
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (lane == 0) {
        gate_sum = float(bfloat(gate_sum));
        up_sum = float(bfloat(up_sum));
        y[(ulong)slot * N + row] =
            bfloat((gate_sum / (1.0f + exp(-gate_sum))) * up_sum);
    }
}

// Row-axis wrapper around the existing ordinary fused body. The BF16 branch
// boundary, F32 weighted combine and subsequent HC expression are unchanged.
kernel void stark_moe_down_combine_inject_decode_rows(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device const uint* indices [[buffer(4)]],
    device const float* scores [[buffer(5)]],
    device const bfloat* shared [[buffer(6)]],
    device const float* gate [[buffer(7)]],
    device bfloat* out [[buffer(8)]],
    device bfloat* hyper [[buffer(9)]],
    device const bfloat* inj [[buffer(10)]],
    constant uint& K [[buffer(11)]],
    constant uint& GS [[buffer(12)]],
    constant uint& H [[buffer(13)]],
    constant uint& S [[buffer(14)]],
    constant uint& has_shared [[buffer(15)]],
    constant uint& G [[buffer(16)]],
    constant float& inv_g [[buffer(17)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xchg[4 * 2 * (MOE_MAX_K / 4 + 1)];
    const uint row = tg.x;
    moe_down_combine_inject_body<16, 4>(codes, scales, biases,
        x + (ulong)row * S * K, indices + (ulong)row * S,
        scores + (ulong)row * S, shared + (ulong)row * H, gate + row,
        out + (ulong)row * H, hyper + (ulong)row * G * H,
        inj + (ulong)row * G, G, inv_g, K, GS, H, S, has_shared,
        tg.y, sg, lane, xchg);
}
