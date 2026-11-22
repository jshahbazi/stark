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
