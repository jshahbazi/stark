// SPDX-License-Identifier: MIT
// Adapted from ggml-org/llama.cpp, revision
// dde879bc4f954b809892fd0d660515f42536394a,
// ggml/src/ggml-metal/kernels/mul_mv_mma.metal (generic F32 kernel,
// lane mapping, clamped tails and K-slice reduction).
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Original Stark adaptations: checked same-weight [token,K] views, existing
// byte-safe hybrid_quant dequantization, sticky status, and uniform original
// block-dot fallback for extreme inputs. Activations and every MMA fragment
// and accumulator remain F32. MMA/K-slice reduction changes sum association.
// The Apple lane mapping is implementation-defined: basis/tail tests on the
// admitted 32-lane device/compiler are required before any runtime admission.
// Runtime admission is restricted to the component-qualified Q6/T4 and
// Q5/Q6/T8 shapes on Apple M5 Max; other uses remain test-only.
// Q8_0/T8 is a separate opt-in candidate on exact GLM dense shapes. Q8's
// 32-value blocks retain byte-safe half-scale and signed-byte reads. Its
// extreme fallback preserves the selected Safe scalar or block chronology.

constant uint hybrid_kquant_mma_nsg [[function_constant(80)]];

inline uint hybrid_kquant_mma_fm(uint lane) {
    return ((lane / 4) & 4) + ((lane / 2) % 4);
}
inline uint hybrid_kquant_mma_fn(uint lane) {
    return ((lane / 4) & 2) * 2 + (lane % 2) * 2;
}

template<uint NT, uint RT>
inline void hybrid_kquant_mma_impl(device const uchar *codes_base,
    device const float *input_base, device float *output_base,
    constant uint *p, device atomic_uint *failure,
    threadgroup float *red, threadgroup atomic_uint *reference,
    uint2 tile, uint lane, uint sg) {
    const uint nsg = hybrid_kquant_mma_nsg;
    const uint rows = p[0], columns = p[1], tokens = p[2];
    device const uchar *codes = codes_base + p[3];
    device const float *input = input_base + p[4];
    device float *output = output_base + p[5];
    const uint fm = hybrid_kquant_mma_fm(lane);
    const uint fn = hybrid_kquant_mma_fn(lane);
    const uint first_row = tile.x * (8 * NT);
    const uint first_token = tile.y * (8 * RT);
    const uint tid = sg * 32 + lane;
    const ulong row_bytes = ulong(columns / hybrid_packed_block) * hybrid_packed_block_bytes;
    if (tid == 0) atomic_store_explicit(reference, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 accum[RT][NT];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < RT; ++rt) {
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
            accum[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }
    bool extreme = false;
    bool nonfinite = false;
    for (uint chunk = sg; chunk < columns / 64; chunk += nsg) {
        simdgroup_float8x8 b[RT][8];
        #pragma clang loop unroll(full)
        for (uint rt = 0; rt < RT; ++rt) {
            const uint token0 = min(first_token + rt * 8 + fn, tokens - 1);
            const uint token1 = min(first_token + rt * 8 + fn + 1, tokens - 1);
            const uint at = chunk * 64 + 16 * (fm / 2) + 8 * (fm % 2);
            const ulong base0 = ulong(token0) * columns + at;
            const ulong base1 = ulong(token1) * columns + at;
            #pragma clang loop unroll(full)
            for (uint s = 0; s < 8; ++s) {
                const float x0 = input[base0 + s];
                const float x1 = input[base1 + s];
                extreme = extreme || abs(x0) > 1e20f || abs(x1) > 1e20f;
                nonfinite = nonfinite || !isfinite(x0) || !isfinite(x1);
                b[rt][s].thread_elements()[0] = x0;
                b[rt][s].thread_elements()[1] = x1;
            }
        }
        // A lane's two elements address logical K columns fn and fn+1.
        // Each lane dequantizes the contiguous 16-value segment starting at
        // chunk*64 + (fn/2)*16. The first/second MMA elements take its first
        // and second eight values. First is a multiple of 16, so it never
        // crosses a Q8_0 block of 32 or a Q5/Q6 block of 256 values.
        // Existing Stark byte reads preserve Q8_0/Q5/Q6
        // storage and avoid upstream typed-pointer alignment assumptions.
        const uint ci = chunk * 4 + fn / 2;
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
            const uint row = min(first_row + t * 8 + fm, rows - 1);
            const uint first = ci * 16;
            device const uchar *block = codes + ulong(row) * row_bytes
                + ulong(first / hybrid_packed_block) * hybrid_packed_block_bytes;
            float4x4 weights;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16; ++j) {
                weights[j / 4][j % 4] = hybrid_quant(block, hybrid_packed_kind, first % hybrid_packed_block + j);
            }
            #pragma clang loop unroll(full)
            for (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 a;
                a.thread_elements()[0] = weights[s / 4][s % 4];
                a.thread_elements()[1] = weights[s / 4 + 2][s % 4];
                #pragma clang loop unroll(full)
                for (uint rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(accum[rt][t], a, b[rt][s], accum[rt][t]);
                }
            }
        }
    }
    if (extreme || nonfinite) atomic_store_explicit(reference, 1u, memory_order_relaxed);
    if (nonfinite) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(reference, memory_order_relaxed)) {
        if (hybrid_packed_kind == 8 && p[6] == 0) {
            // The configured scalar Q8 source uses 256 lanes per output:
            // eight simd_sum partials followed by their sequential sum.
            // Runtime Q8 admission supplies 8 or 16 SIMD groups. Reuse red
            // for one or two independent outputs and keep all groups in
            // the barriers, including clamped row/token tails.
            const uint parallel_outputs = nsg / 8;
            for (uint base = 0; base < (8 * NT) * (8 * RT); base += parallel_outputs) {
                const uint flat = base + sg / 8;
                const uint row = first_row + flat % (8 * NT);
                const uint token = first_token + flat / (8 * NT);
                const bool real = flat < (8 * NT) * (8 * RT) && row < rows && token < tokens;
                float sum = 0.0f;
                if (real) {
                    for (uint col = (sg % 8) * 32 + lane; col < columns; col += 256) {
                        device const uchar *block = codes + ulong(row) * row_bytes
                            + ulong(col / hybrid_packed_block) * hybrid_packed_block_bytes;
                        sum += hybrid_quant(block, hybrid_packed_kind, col % hybrid_packed_block)
                            * input[ulong(token) * columns + col];
                    }
                }
                const float partial = simd_sum(sum);
                if (lane == 0) red[sg] = partial;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg % 8 == 0 && lane == 0 && real) {
                    float total = 0.0f;
                    for (uint i = 0; i < 8; ++i) total += red[sg + i];
                    if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
                    output[ulong(token) * rows + row] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            return;
        }
        // Uniform across all SIMD groups, including clamped output tails.
        // Re-evaluate each real output with the original block helper so an
        // MMA overflow cannot conceal or invent the zero*MAX contract.
        for (uint flat = sg; flat < (8 * NT) * (8 * RT); flat += nsg) {
            const uint row = first_row + flat % (8 * NT);
            const uint token = first_token + flat / (8 * NT);
            if (row < rows && token < tokens) {
                const float total = hybrid_packed_block_dot(codes, input,
                    ulong(row) * row_bytes, ulong(token) * columns, columns, lane);
                if (lane == 0) {
                    if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
                    output[ulong(token) * rows + row] = total;
                }
            }
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < RT; ++rt) {
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
            red[((sg * RT + rt) * NT + t) * 64 + 2 * lane] = accum[rt][t].thread_elements()[0];
            red[((sg * RT + rt) * NT + t) * 64 + 2 * lane + 1] = accum[rt][t].thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < RT * NT * 64; index += nsg * 32) {
        float sum = 0.0f;
        for (uint split = 0; split < nsg; ++split) sum += red[split * RT * NT * 64 + index];
        const uint rt = index / (NT * 64);
        const uint t = (index / 64) % NT;
        const uint owner_lane = (index % 64) / 2;
        const uint row = first_row + t * 8 + hybrid_kquant_mma_fm(owner_lane);
        const uint token = first_token + rt * 8 + hybrid_kquant_mma_fn(owner_lane) + index % 2;
        if (row < rows && token < tokens) {
            if (!isfinite(sum)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            output[ulong(token) * rows + row] = sum;
        }
    }
}

#define HYBRID_KQUANT_MMA(NT, RT) \
kernel void hybrid_kquant_mma_nt##NT##_rt##RT( \
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]], \
    device atomic_uint *failure [[buffer(4)]], threadgroup float *red [[threadgroup(0)]], \
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup atomic_uint reference; \
    hybrid_kquant_mma_impl<NT, RT>(codes, input, output, p, failure, red, &reference, tile, lane, sg); \
}
HYBRID_KQUANT_MMA(1, 1)
HYBRID_KQUANT_MMA(2, 1)
HYBRID_KQUANT_MMA(4, 1)
HYBRID_KQUANT_MMA(1, 2)
HYBRID_KQUANT_MMA(2, 2)
HYBRID_KQUANT_MMA(4, 2)
#undef HYBRID_KQUANT_MMA
