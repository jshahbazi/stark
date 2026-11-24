// SPDX-License-Identifier: Apache-2.0
// Sort the short verifier's routes by (expert, original route). At most
// 32 routes fit in one SIMD group, preserving order within an expert.
#include <metal_stdlib>
using namespace metal;

kernel void ink_verify_bucket_short(
    device const uint *ids [[buffer(0)]],
    device uint *order [[buffer(1)]],
    device uint *runs [[buffer(2)]],
    device uint *dispatch [[buffer(3)]],
    constant uint2 &shape [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint routes = shape.x;
    uint key = lane < routes ? (ids[lane] << 5) | lane : UINT_MAX;
    for (uint size = 2; size <= 32; size *= 2) {
        for (uint offset = size / 2; offset != 0; offset /= 2) {
            const uint other = simd_shuffle_xor(key, ushort(offset));
            const bool use_min = ((lane & size) == 0) == ((lane & offset) == 0);
            key = use_min ? min(key, other) : max(key, other);
        }
    }
    const uint expert = key >> 5;
    const uint preceding = simd_shuffle_up(expert, 1);
    const bool head = lane < routes && (lane == 0 || preceding != expert);
    // Each head contributes a distinct bit, so the unsigned sum is a mask.
    const uint mask = simd_sum(head ? 1u << lane : 0u);
    const uint live = popcount(mask);
    const uint prior = (1u << lane) - 1u;
    if (lane < routes) order[lane] = key & 31u;
    if (head) {
        const uint after = mask & ~(prior | (1u << lane));
        const uint end = after ? ctz(after) : routes;
        const uint run = popcount(mask & prior);
        runs[3 * run] = lane;
        runs[3 * run + 1] = end - lane;
        runs[3 * run + 2] = expert;
    }
    for (uint run = live + lane; run < 128; run += 32) {
        runs[3 * run] = 0;
        runs[3 * run + 1] = 0;
        runs[3 * run + 2] = 0;
    }
    if (lane == 0) {
        dispatch[0] = shape.y;
        dispatch[1] = live;
        dispatch[2] = 1;
    }
}
