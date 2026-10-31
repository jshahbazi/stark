// Copyright (c) 2023-2024 Apple Inc.
// Adapted by Stark from MLX 64ea011cb65f14d9ce2737e60db9a4ae91ed7441,
// quantized.h affine_quantize. See LICENSE.mlx and UPSTREAM.md.
#include <metal_stdlib>
using namespace metal;

#define KV_AFFINE64_GROUP 64u
#define KV_AFFINE64_META_PER_ROW(D) ((D) / 32u)

// Pinned MLX C-API oracle: matched-mlx-affine8-oracle-v1.json,
// SHA256 40655a784a2576e74f8fe2b206146b229bec40f1b8adb573ed972fa2fc0ddec2.
// The reference rounds once after an f32 code*stored_scale+stored_bias.
// Rounding the product to bf16 before adding the bias does not match.
static inline bfloat kv_affine64_dequant_code(uchar code, bfloat scale, bfloat bias) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    // Integer RNE materializes the BF16 payload even when an inlined reader
    // immediately expands it to F32 under its original fast-math policy.
    // Native bfloat->float roundtrip casts can otherwise be optimized away.
    const float value = float(code) * float(scale) + float(bias);
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return as_type<bfloat>(ushort((bits >> 16) | 0x0040u));
    }
    const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return as_type<bfloat>(ushort(rounded >> 16));
}

static inline bfloat kv_affine64_at(device const uchar* row,
                                   device const bfloat* metadata,
                                   uint dim) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint pair = 2u * (dim / KV_AFFINE64_GROUP);
    return kv_affine64_dequant_code(row[dim], metadata[pair], metadata[pair + 1u]);
}

static inline bfloat4 kv_affine64_load4(device const uchar* row,
                                       device const bfloat* metadata,
                                       uint dim) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return bfloat4(kv_affine64_at(row, metadata, dim),
                   kv_affine64_at(row, metadata, dim + 1u),
                   kv_affine64_at(row, metadata, dim + 2u),
                   kv_affine64_at(row, metadata, dim + 3u));
}

static inline uint4 kv_affine64_load8(device const uchar* row,
                                     device const bfloat* metadata,
                                     uint dim) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return uint4(as_type<uint2>(kv_affine64_load4(row, metadata, dim)),
                 as_type<uint2>(kv_affine64_load4(row, metadata, dim + 4u)));
}

// The reductions are min/max only. Each pair of 32-lane simdgroups covers
// one 64-value group; unlike the upstream two-values-per-lane schedule,
// this also fits the existing 256-thread fused norm/RoPE writer.
static inline void kv_affine64_store(float x,
                                     device uchar* value,
                                     device bfloat* pair,
                                     threadgroup float* minima,
                                     threadgroup float* maxima,
                                     uint tid, uint sg, uint lane) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const float low = simd_min(x);
    const float high = simd_max(max(x, 0.0f));
    if (lane == 0) { minima[sg] = low; maxima[sg] = high; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint group = tid / KV_AFFINE64_GROUP;
    const float w_min = min(minima[2u * group], minima[2u * group + 1u]);
    const float w_max = max(maxima[2u * group], maxima[2u * group + 1u]);
    float scale = max((w_max - w_min) / 255.0f, 1e-7f);
    const bool side = abs(w_min) > abs(w_max);
    scale = side ? scale : -scale;
    const float edge = side ? w_min : w_max;
    const float q0 = round(edge / scale);
    const bool at_zero = q0 == 0.0f;
    scale = at_zero ? scale : edge / q0;
    const float bias = at_zero ? 0.0f : edge;
    // Codes intentionally use the unrounded f32 scale/bias. Only stored
    // metadata is bf16. round(), rather than rint(), matches MLX.
    *value = uchar(min(round((x - bias) / scale), 255.0f));
    if (tid % KV_AFFINE64_GROUP == 0) {
        pair[0] = bfloat(scale);
        pair[1] = bfloat(bias);
    }
}
