// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

// Keep the original FP32 tap order and BF16 cast after convolution.
static bfloat ink_verify_kv_conv(device const bfloat *current_row, device const bfloat *weight,
    device const bfloat *dynamic, device bfloat *history, uint channel, uint position, bool changing) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint Width = 1024;
    const bfloat raw = current_row[channel];
    const float current = float(raw);
    float result = 0.f;
    for (uint tap = 0; tap < 4; ++tap) {
        const uint lag = changing ? tap : 3 - tap;
        if (position < lag) continue;
        const float value = lag == 0 ? current : float(history[((position-lag) % 3) * Width + channel]);
        const float coefficient = changing ? float(weight[tap * Width + channel]) + float(dynamic[(channel / 128) * 4 + tap])
                                          : float(weight[channel * 4 + tap]);
        result += coefficient * value;
    }
    history[(position % 3) * Width + channel] = raw;
    return bfloat((result + current) * (changing ? 1.f / 3.f : 0.447213595499958f));
}

#ifdef INK_VERIFY_FORK
// A fork row is evaluated at its logical depth while its speculative cache
// and history write remains in the physical last-row slot. Snapshot history
// arguments are captured before this kernel mutates any ring entries.
static bfloat ink_verify_kv_conv_fork(
    device const bfloat *current_row, device const bfloat *weight,
    device const bfloat *dynamic, device bfloat *history,
    device const bfloat *ancestor0, device const bfloat *ancestor1,
    device const bfloat *ancestor2, bfloat4 old0, bfloat4 old1, bfloat4 old2,
    uint channel, uint component, uint logical_position, uint physical_position,
    uint logical_depth, bool sibling, bool changing
) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint Width = 1024;
    const bfloat raw = current_row[channel];
    const float current = float(raw);
    float result = 0.f;
    for (uint tap = 0; tap < 4; ++tap) {
        const uint lag = changing ? tap : 3 - tap;
        if (logical_position < lag) continue;
        float value;
        if (lag == 0) {
            value = current;
        } else if (sibling && lag <= logical_depth) {
            const uint ancestor = logical_depth - lag;
            device const bfloat *source = ancestor == 0 ? ancestor0
                : ancestor == 1 ? ancestor1 : ancestor2;
            value = float(source[channel]);
        } else if (sibling) {
            const uint slot = (logical_position - lag) % 3;
            value = slot == 0 ? float(old0[component])
                : slot == 1 ? float(old1[component]) : float(old2[component]);
        } else {
            value = float(history[((logical_position - lag) % 3) * Width + channel]);
        }
        const float coefficient = changing
            ? float(weight[tap * Width + channel]) + float(dynamic[(channel / 128) * 4 + tap])
            : float(weight[channel * 4 + tap]);
        result += coefficient * value;
    }
    history[(physical_position % 3) * Width + channel] = raw;
    return bfloat((result + current) * (changing ? 1.f / 3.f : 0.447213595499958f));
}
#endif

kernel void ink_verify_kv_pack(
    device const bfloat *k0 [[buffer(0)]], device const bfloat *k1 [[buffer(1)]],
    device const bfloat *k2 [[buffer(2)]], device const bfloat *k3 [[buffer(3)]],
    device const bfloat *v0 [[buffer(4)]], device const bfloat *v1 [[buffer(5)]],
    device const bfloat *v2 [[buffer(6)]], device const bfloat *v3 [[buffer(7)]],
    device const bfloat *dk0 [[buffer(8)]], device const bfloat *dk1 [[buffer(9)]],
    device const bfloat *dk2 [[buffer(10)]], device const bfloat *dk3 [[buffer(11)]],
    device const bfloat *dv0 [[buffer(12)]], device const bfloat *dv1 [[buffer(13)]],
    device const bfloat *dv2 [[buffer(14)]], device const bfloat *dv3 [[buffer(15)]],
    device const bfloat *wk [[buffer(16)]], device const bfloat *wv [[buffer(17)]],
    device bfloat *hk [[buffer(18)]], device bfloat *hv [[buffer(19)]],
    device bfloat *keys [[buffer(20)]], device bfloat *values [[buffer(21)]],
    constant uint4 &shape [[buffer(22)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint ordinal = 4 * group + sg;
    const bool value_head = ordinal >= 8;
    const uint channel = (ordinal % 8) * 128 + lane * 4;
    device const bfloat *w = value_head ? wv : wk;
    device bfloat *history = value_head ? hv : hk;
    device bfloat *cache = value_head ? values : keys;
    device const bfloat *raw[] = {value_head ? v0 : k0, value_head ? v1 : k1, value_head ? v2 : k2, value_head ? v3 : k3};
    device const bfloat *dynamic[] = {value_head ? dv0 : dk0, value_head ? dv1 : dk1, value_head ? dv2 : dk2, value_head ? dv3 : dk3};
#ifdef INK_VERIFY_FORK
    const uint sibling_depth = (shape.w >> 8) & 0xff;
    const bfloat4 old0 = *reinterpret_cast<device const bfloat4 *>(history + channel);
    const bfloat4 old1 = *reinterpret_cast<device const bfloat4 *>(history + 1024 + channel);
    const bfloat4 old2 = *reinterpret_cast<device const bfloat4 *>(history + 2048 + channel);
#endif
    for (uint row = 0; row < shape.y; ++row) {
#ifdef INK_VERIFY_FORK
        const bool sibling = row + 1 == shape.y;
        const uint logical_depth = sibling ? sibling_depth : row;
        const uint position = shape.x + logical_depth;
        const uint physical_position = shape.x + row;
        const uint physical_row = row;
#else
        const uint position = shape.x + row;
#endif
        float4 vals;
        float norm = 0.f;
        for (uint j = 0; j < 4; ++j) {
#ifdef INK_VERIFY_FORK
            const float value = float(ink_verify_kv_conv_fork(
                raw[physical_row], w, dynamic[physical_row], history,
                raw[0], raw[1], raw[2], old0, old1, old2,
                channel + j, j, position, physical_position, logical_depth,
                sibling, (shape.w & 1) != 0
            ));
#else
            const float value = float(ink_verify_kv_conv(raw[row], w, dynamic[row], history, channel + j, position, shape.w != 0));
#endif
            if (value_head) norm = fma(value,value,norm);
            else {
#pragma clang fp contract(off) reassociate(off)
                norm += value * value;
            }
            vals[j] = value;
        }
        norm = simd_sum(norm);
        const float scale = value_head ? precise::rsqrt(norm / 128.0f + 1e-6f)
                                      : min(1.0f, 22.62741699797f / (precise::sqrt(norm) + 1e-6f));
#ifdef INK_VERIFY_FORK
        const uint slot = (physical_position % shape.z) * 1024 + channel;
#else
        const uint slot = (position % shape.z) * 1024 + channel;
#endif
        *reinterpret_cast<device bfloat4 *>(cache + slot) = bfloat4(vals * scale);
    }
}

kernel void ink_verify_qr_pack(
    device bfloat *q0 [[buffer(0)]], device bfloat *q1 [[buffer(1)]], device bfloat *q2 [[buffer(2)]], device bfloat *q3 [[buffer(3)]],
    device bfloat *r0 [[buffer(4)]], device bfloat *r1 [[buffer(5)]], device bfloat *r2 [[buffer(6)]], device bfloat *r3 [[buffer(7)]],
    device const bfloat *q_gain [[buffer(8)]], device const bfloat *r_gain [[buffer(9)]],
    constant uint2 &shape [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
#pragma clang fp contract(off) reassociate(off)
    const uint ordinal = 4 * group.x + sg;
    const uint head = ordinal % 24;
    const uint token = group.y;
    device bfloat *qs[] = {q0,q1,q2,q3};
    device bfloat *rs[] = {r0,r1,r2,r3};
    const uint position = shape.x + token;
    if (ordinal >= 24) {
        device bfloat *row = rs[token] + head * 16;
        device const bfloat *gain = r_gain + head * 16;
        const float value = lane < 16 ? float(row[lane]) * float(gain[lane]) : 0.f;
        float norm = 0.f;
        if (lane == 0) {
            for (uint i = 0; i < 16; ++i) {
                const float source = float(row[i]) * float(gain[i]);
                norm += source * source;
            }
        }
        float scale = min(1.0f, 8.0f / (precise::sqrt(simd_broadcast(norm,0)) + 1e-6f));
        if (shape.y == 0 && position >= 8192) scale *= ink_long_position_gain(position);
        if (lane < 16) row[lane] = bfloat(value * scale);
    } else {
        device bfloat *row = qs[token] + head * 128 + 4 * lane;
        device const bfloat *gain = q_gain + head * 128 + 4 * lane;
        float4 vals;
        float norm = 0.f;
        for (uint j = 0; j < 4; ++j) {
            const float value = float(row[j]) * float(gain[j]);
            norm += value * value;
            vals[j] = value;
        }
        float scale = min(1.0f, 22.62741699797f / (precise::sqrt(simd_sum(norm)) + 1e-6f));
        if (shape.y == 0 && position >= 8192) scale *= ink_long_position_gain(position);
        for (uint j = 0; j < 4; ++j) row[j] = bfloat(vals[j] * scale);
    }
}

// Diagnostic only (visit capture): the K/V convolution outputs of
// ink_verify_kv_conv for one row, without storing the history slot.
kernel void ink_verify_kv_conv_probe(
    device const bfloat *k [[buffer(0)]], device const bfloat *v [[buffer(1)]],
    device const bfloat *dk [[buffer(2)]], device const bfloat *dv [[buffer(3)]],
    device const bfloat *wk [[buffer(4)]], device const bfloat *wv [[buffer(5)]],
    device const bfloat *hk [[buffer(6)]], device const bfloat *hv [[buffer(7)]],
    device bfloat *ck [[buffer(8)]], device bfloat *cv [[buffer(9)]],
    constant uint2 &position_changing [[buffer(10)]], uint2 tid [[thread_position_in_grid]]
) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint Width = 1024;
    const bool value_head = tid.y != 0;
    const uint channel = tid.x;
    const uint position = position_changing.x;
    const bool changing = position_changing.y != 0;
    device const bfloat *weight = value_head ? wv : wk;
    device const bfloat *dynamic = value_head ? dv : dk;
    device const bfloat *history = value_head ? hv : hk;
    const float current = float((value_head ? v : k)[channel]);
    float result = 0.f;
    for (uint tap = 0; tap < 4; ++tap) {
        const uint lag = changing ? tap : 3 - tap;
        if (position < lag) continue;
        const float value = lag == 0 ? current : float(history[((position-lag) % 3) * Width + channel]);
        const float coefficient = changing ? float(weight[tap * Width + channel]) + float(dynamic[(channel / 128) * 4 + tap])
                                          : float(weight[channel * 4 + tap]);
        result += coefficient * value;
    }
    (value_head ? cv : ck)[channel] = bfloat((result + current) * (changing ? 1.f / 3.f : 0.447213595499958f));
}

// One-row ink_verify_kv_pack and ink_verify_qr_pack in one launch of 16
// workgroups: workgroups 0-3 run the K/V body and 4-15 the Q/R body, each
// copied unchanged with its row fixed to zero.
kernel void ink_verify_heads_pack_one(
    device const bfloat *k [[buffer(0)]], device const bfloat *v [[buffer(1)]],
    device const bfloat *dk [[buffer(2)]], device const bfloat *dv [[buffer(3)]],
    device const bfloat *wk [[buffer(4)]], device const bfloat *wv [[buffer(5)]],
    device bfloat *hk [[buffer(6)]], device bfloat *hv [[buffer(7)]],
    device bfloat *keys [[buffer(8)]], device bfloat *values [[buffer(9)]],
    device bfloat *q [[buffer(10)]], device bfloat *r [[buffer(11)]],
    device const bfloat *q_gain [[buffer(12)]], device const bfloat *r_gain [[buffer(13)]],
    constant uint4 &shape [[buffer(14)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    // shape: position, cache slots, dynamic convolution, attention window.
    if (group < 4) {
        const uint ordinal = 4 * group + sg;
        const bool value_head = ordinal >= 8;
        const uint channel = (ordinal % 8) * 128 + lane * 4;
        device const bfloat *w = value_head ? wv : wk;
        device bfloat *history = value_head ? hv : hk;
        device bfloat *cache = value_head ? values : keys;
        device const bfloat *raw = value_head ? v : k;
        device const bfloat *dynamic = value_head ? dv : dk;
        const uint position = shape.x;
        float4 vals;
        float norm = 0.f;
        for (uint j = 0; j < 4; ++j) {
            const float value = float(ink_verify_kv_conv(raw, w, dynamic, history, channel + j, position, shape.z != 0));
            if (value_head) norm = fma(value,value,norm);
            else {
#pragma clang fp contract(off) reassociate(off)
                norm += value * value;
            }
            vals[j] = value;
        }
        norm = simd_sum(norm);
        const float scale = value_head ? precise::rsqrt(norm / 128.0f + 1e-6f)
                                      : min(1.0f, 22.62741699797f / (precise::sqrt(norm) + 1e-6f));
        const uint slot = (position % shape.y) * 1024 + channel;
        *reinterpret_cast<device bfloat4 *>(cache + slot) = bfloat4(vals * scale);
        return;
    }
    {
#pragma clang fp contract(off) reassociate(off)
    const uint ordinal = 4 * (group - 4) + sg;
    const uint head = ordinal % 24;
    const uint position = shape.x;
    if (ordinal >= 24) {
        device bfloat *row = r + head * 16;
        device const bfloat *gain = r_gain + head * 16;
        const float value = lane < 16 ? float(row[lane]) * float(gain[lane]) : 0.f;
        float norm = 0.f;
        if (lane == 0) {
            for (uint i = 0; i < 16; ++i) {
                const float source = float(row[i]) * float(gain[i]);
                norm += source * source;
            }
        }
        float scale = min(1.0f, 8.0f / (precise::sqrt(simd_broadcast(norm,0)) + 1e-6f));
        if (shape.w == 0 && position >= 8192) scale *= ink_long_position_gain(position);
        if (lane < 16) row[lane] = bfloat(value * scale);
    } else {
        device bfloat *row = q + head * 128 + 4 * lane;
        device const bfloat *gain = q_gain + head * 128 + 4 * lane;
        float4 vals;
        float norm = 0.f;
        for (uint j = 0; j < 4; ++j) {
            const float value = float(row[j]) * float(gain[j]);
            norm += value * value;
            vals[j] = value;
        }
        float scale = min(1.0f, 22.62741699797f / (precise::sqrt(simd_sum(norm)) + 1e-6f));
        if (shape.w == 0 && position >= 8192) scale *= ink_long_position_gain(position);
        for (uint j = 0; j < 4; ++j) row[j] = bfloat(vals[j] * scale);
    }
    }
}
