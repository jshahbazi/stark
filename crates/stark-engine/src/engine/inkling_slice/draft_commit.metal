// SPDX-License-Identifier: Apache-2.0
//! Packed draft-cache commit for prefill: the per-row `commit_from` work for
//! every row of a command in a few dispatches.
#include <metal_stdlib>
using namespace metal;

constant uint ink_commit_width = 2304;
constant uint ink_commit_kv = 1024;

// The four-tap context mix of `ink_draft_mix`, for every row. Taps hold four
// BF16 rows of `ink_commit_width` per token row.
kernel void ink_draft_commit_mix(
    device const bfloat *taps [[buffer(0)]], device const bfloat *weights [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &layer [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
#pragma clang fp contract(off) reassociate(off)
    const device bfloat *features = taps + at.y * 4 * ink_commit_width;
    const uint i = at.x;
    out[at.y * ink_commit_width + i] =
        (float(features[i]) * float(weights[layer * 4])
         + float(features[ink_commit_width + i]) * float(weights[layer * 4 + 1]))
        + (float(features[2 * ink_commit_width + i]) * float(weights[layer * 4 + 2])
           + float(features[3 * ink_commit_width + i]) * float(weights[layer * 4 + 3]));
}

// Store projected values of consecutive positions into the draft ring.
// p = (first position, ring capacity).
kernel void ink_draft_commit_values(
    device const bfloat *values [[buffer(0)]], device bfloat *cache [[buffer(1)]],
    constant uint2 &p [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
    cache[((p.x + at.y) % p.y) * ink_commit_kv + at.x] = values[at.y * ink_commit_kv + at.x];
}
