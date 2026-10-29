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
