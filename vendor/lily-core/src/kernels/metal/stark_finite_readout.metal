// Original Stark validation kernel. Copyright 2026 Stark contributors.
// Licensed under the Apache License, Version 2.0; see ../../../LICENSE.
// This file is separate from the pinned, unmodified Lily math kernels.
#include <metal_stdlib>
using namespace metal;

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
