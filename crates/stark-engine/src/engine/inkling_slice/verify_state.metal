// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

kernel void ink_save_anchor_convolution(
    device const ushort *key [[buffer(0)]], device const ushort *value [[buffer(1)]],
    device const ushort *attended [[buffer(2)]], device const ushort *expert [[buffer(3)]],
    device ushort *saved [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < 1024) saved[i] = key[i];
    else if (i < 2048) saved[i] = value[i - 1024];
    else if (i < 4352) saved[i] = attended[i - 2048];
    else saved[i] = expert[i - 4352];
}

// Per-row visit preparation for every verifier row in one launch. Each thread
// owns one channel and performs the separate copies in their original order.
// Mode 0 saves the prelude, mode 1 starts a later pass and mode 2 captures a
// BF16 draft tap at word offset `tap`.
kernel void ink_verify_prepare_rows(
    device float *p0 [[buffer(0)]], device float *p1 [[buffer(1)]],
    device float *p2 [[buffer(2)]], device float *p3 [[buffer(3)]],
    device float *s0 [[buffer(4)]], device float *s1 [[buffer(5)]],
    device float *s2 [[buffer(6)]], device float *s3 [[buffer(7)]],
    device const float *b0 [[buffer(8)]], device const float *b1 [[buffer(9)]],
    device const float *b2 [[buffer(10)]], device const float *b3 [[buffer(11)]],
    device float *v0 [[buffer(12)]], device float *v1 [[buffer(13)]],
    device float *v2 [[buffer(14)]], device float *v3 [[buffer(15)]],
    device bfloat *f0 [[buffer(16)]], device bfloat *f1 [[buffer(17)]],
    device bfloat *f2 [[buffer(18)]], device bfloat *f3 [[buffer(19)]],
    constant uint2 &mode_tap [[buffer(20)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint Width = 2304;
    device float *partial[] = {p0, p1, p2, p3};
    device float *prelude[] = {s0, s1, s2, s3};
    device const float *blocks[] = {b0, b1, b2, b3};
    device float *previous[] = {v0, v1, v2, v3};
    device bfloat *features[] = {f0, f1, f2, f3};
    const uint row = group.y;
    const uint i = group.x * 128 + tid;
    if (mode_tap.x == 0) {
        prelude[row][i] = partial[row][i];
    } else if (mode_tap.x == 1) {
        previous[row][i] = blocks[row][2 * Width + i];
        previous[row][Width + i] = partial[row][i];
        partial[row][i] = prelude[row][i];
    } else {
        features[row][mode_tap.y + i] = bfloat(partial[row][i]);
    }
}
