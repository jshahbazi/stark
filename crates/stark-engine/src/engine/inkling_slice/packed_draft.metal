// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

kernel void ink_packed_draft_tap(
    device const float *partial [[buffer(0)]], device bfloat *taps [[buffer(1)]],
    constant uint &tap [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
    taps[(at.y * 4 + tap) * 2304 + at.x] = bfloat(partial[at.y * 2304 + at.x]);
}

kernel void ink_packed_draft_hidden(
    device const bfloat *hidden [[buffer(0)]], device bfloat *taps [[buffer(1)]],
    uint2 at [[thread_position_in_grid]]) {
    taps[(at.y * 4 + 3) * 2304 + at.x] = hidden[at.y * 2304 + at.x];
}
