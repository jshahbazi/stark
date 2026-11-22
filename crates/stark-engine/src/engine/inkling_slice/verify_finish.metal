// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

// One thread owns one channel of the causal three-slot history. Keep all
// BF16 boundaries and visit each provisional row in increasing position.
kernel void ink_verify_finish_experts(
    device const bfloat *down [[buffer(0)]],
    device bfloat *merged [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *history [[buffer(3)]],
    device float *p0 [[buffer(4)]], device float *p1 [[buffer(5)]],
    device float *p2 [[buffer(6)]], device float *p3 [[buffer(7)]],
    constant uint2 &range [[buffer(8)]],
    uint channel [[thread_position_in_grid]]
) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint Width = 2304;
    for (uint row = 0; row < range.y; ++row) {
        float sum = 0.0f;
        for (uint route = 0; route < 8; ++route)
            sum += float(down[(row * 8 + route) * Width + channel]);
        const bfloat current_bf16 = bfloat(sum);
        const float current = float(current_bf16);
        merged[row * Width + channel] = current_bf16;
        const uint pos = range.x + row;
        float result = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            const uint lag = 3 - tap;
            if (pos < lag) continue;
            const float value = lag == 0 ? current : float(history[((pos-lag) % 3) * Width + channel]);
            result += float(weight[channel * 4 + tap]) * value;
        }
        history[(pos % 3) * Width + channel] = current_bf16;
        const bfloat ffn = bfloat((result + current) * 0.447213595499958f);
        device float *partial = row == 0 ? p0 : row == 1 ? p1 : row == 2 ? p2 : p3;
        partial[channel] = float(bfloat(partial[channel] + float(ffn)));
    }
}

kernel void ink_verify_finish_attention(
    device const bfloat *weight [[buffer(0)]],
    device bfloat *history [[buffer(1)]],
    device const bfloat *x0 [[buffer(2)]], device const bfloat *x1 [[buffer(3)]],
    device const bfloat *x2 [[buffer(4)]], device const bfloat *x3 [[buffer(5)]],
    device float *p0 [[buffer(6)]], device float *p1 [[buffer(7)]],
    device float *p2 [[buffer(8)]], device float *p3 [[buffer(9)]],
    device float *b0 [[buffer(10)]], device float *b1 [[buffer(11)]],
    device float *b2 [[buffer(12)]], device float *b3 [[buffer(13)]],
    constant uint3 &range [[buffer(14)]],
    uint channel [[thread_position_in_grid]]
) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint Width = 2304;
    for (uint row = 0; row < range.y; ++row) {
        device const bfloat *input = row == 0 ? x0 : row == 1 ? x1 : row == 2 ? x2 : x3;
        device float *partial = row == 0 ? p0 : row == 1 ? p1 : row == 2 ? p2 : p3;
        device float *block = row == 0 ? b0 : row == 1 ? b1 : row == 2 ? b2 : b3;
        const uint pos = range.x + row;
        const float current = float(input[channel]);
        float result = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            const uint lag = 3 - tap;
            if (pos < lag) continue;
            const float value = lag == 0 ? current : float(history[((pos - lag) % 3) * Width + channel]);
            result += float(weight[channel * 4 + tap]) * value;
        }
        history[(pos % 3) * Width + channel] = input[channel];
        const bfloat attention = bfloat((result + current) * 0.447213595499958f);
        if (range.z != 0) block[channel] = partial[channel];
        partial[channel] = (range.z != 0 ? 0.0f : partial[channel]) + float(attention);
    }
}
