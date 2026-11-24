// SPDX-License-Identifier: Apache-2.0
// Diagnostic bandwidth ceilings for the decode stage table. Never used by
// inference. Mode 0 only loads (XOR-reduced so the loads stay live), mode 1
// loads and sums every byte as a float.

#include <metal_stdlib>
using namespace metal;

static inline float ink_probe_bytes(uint4 v) {
    return dot(float4(as_type<uchar4>(v.x)) + float4(as_type<uchar4>(v.y)) + float4(as_type<uchar4>(v.z)) +
                   float4(as_type<uchar4>(v.w)),
               float4(1.0f));
}

// args: x = uint4 count, y = mode.
kernel void ink_probe_read(
    device const uint4 *data [[buffer(0)]], device uint *out [[buffer(1)]], constant uint2 &args [[buffer(2)]],
    uint tid [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    uint4 acc = 0;
    float sum = 0.0f;
    if (args.y == 0) {
        for (uint i = tid; i < args.x; i += threads) acc ^= data[i];
    } else {
        for (uint i = tid; i < args.x; i += threads) sum += ink_probe_bytes(data[i]);
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u || sum == -1.0f) out[0] = tid;
}

// The planes of selected experts. args: x = uint4 per expert, y = expert
// count, z = mode, w = first id. Grid: (chunks, experts).
kernel void ink_probe_gather(
    device const uint4 *data [[buffer(0)]], device const uint *ids [[buffer(1)]], device uint *out [[buffer(2)]],
    constant uint4 &args [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lid [[thread_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
    const uint width = threads.x;
    device const uint4 *base = data + size_t(ids[args.w + group.y]) * args.x;
    uint4 acc = 0;
    float sum = 0.0f;
    const uint stride = groups.x * width;
    if (args.z == 0) {
        for (uint i = group.x * width + lid; i < args.x; i += stride) acc ^= base[i];
    } else {
        for (uint i = group.x * width + lid; i < args.x; i += stride) sum += ink_probe_bytes(base[i]);
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u || sum == -1.0f) out[0] = lid;
}
