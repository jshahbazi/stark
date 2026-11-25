// Original Stark validation kernel. Copyright 2026 Stark contributors.
// Licensed under the Apache License, Version 2.0; see ../../../LICENSE.
// This file is separate from the pinned, unmodified Lily math kernels.
#include <metal_stdlib>
using namespace metal;

// Monotone integer ranks preserve every finite IEEE comparison, including ties
// between signed zeros, independently of the GPU's denormal handling.
static inline uint runner_rank(uint bits) {
    if ((bits << 1) == 0) bits = 0;
    return (bits >> 31) ? ~bits : (bits ^ 0x80000000u);
}

kernel void stark_runner_partial(
    device const uint* logits [[buffer(0)]],
    device const uint* primary [[buffer(1)]],
    device uint2* parts [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint2 local_best[8];
    uint rank = 0, id = 0xffffffffu;
    for (uint i = g * 256 + tid; i < count; i += groups * 256) {
        uint r = runner_rank(logits[i]);
        if (i != primary[0] && (r > rank || (r == rank && i < id))) {
            rank = r;
            id = i;
        }
    }
    uint best = simd_max(rank);
    uint first = simd_min(rank == best ? id : 0xffffffffu);
    if (lane == 0) local_best[sg] = uint2(best, first);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint2 p = local_best[0];
        for (uint i = 1; i < 8; ++i) {
            uint2 q = local_best[i];
            if (q.x > p.x || (q.x == p.x && q.y < p.y)) p = q;
        }
        parts[g] = p;
    }
}

kernel void stark_runner_finish(
    device const float* logits [[buffer(0)]],
    device const uint* primary [[buffer(1)]],
    device const uint2* parts [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant uint& groups [[buffer(4)]]) {
    uint2 p = parts[0];
    for (uint i = 1; i < groups; ++i) {
        uint2 q = parts[i];
        if (q.x > p.x || (q.x == p.x && q.y < p.y)) p = q;
    }
    out[0] = p.y;
    out[1] = as_type<uint>(logits[primary[0]] - logits[p.y]);
}

kernel void stark_reject_nonfinite_readout_f32(
    device const uint* logits_bits [[buffer(0)]],
    device atomic_uint* next_token [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count && (logits_bits[index] & 0x7f800000u) == 0x7f800000u) {
        // Integer inspection remains exact under the model's fast-math policy.
        atomic_store_explicit(next_token, 0xffffffffu, memory_order_relaxed);
    }
}

// Marks a device argmax whose selected logit is a zero or subnormal. Device
// comparisons may flush subnormals, so only then can the selection differ
// from the host's IEEE first maximum; the host then redoes the selection.
// A selected normal value is exact: every flushed operand is below it, and
// comparisons between normal values are exact.
kernel void stark_flag_tiny_selection_f32(
    device const uint* logits_bits [[buffer(0)]],
    device const uint* selected [[buffer(1)]],
    device atomic_uint* flag [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        const uint id = selected[0];
        if (id >= count || (logits_bits[id] & 0x7f800000u) == 0u) {
            atomic_fetch_or_explicit(flag, 1u, memory_order_relaxed);
        }
    }
}
