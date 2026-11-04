// SPDX-License-Identifier: MIT
// Adapted from ggml-org/llama.cpp revision
// dde879bc4f954b809892fd0d660515f42536394a,
// ggml/src/ggml-metal/kernels/mul_mv_mma.metal.
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Stark's wide scheduling retains the existing F32 fragment mapping, K-split
// chronology, byte-safe GGUF decoding and per-eight-token extreme fallback.
// A decoded weight fragment is reused across 32/64/128 input tokens. Input
// fragments stream through one matrix, and one eight-token reduction reuses
// threadgroup scratch, bounding source-level register and shared storage.

template<uint NT, uint RT>
inline void hybrid_kquant_dense_wide_impl(device const uchar *codes_base,
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
    const ulong row_bytes = ulong(columns / 256) * hybrid_packed_block_bytes;
    for (uint rt = tid; rt < RT; rt += nsg * 32) {
        atomic_store_explicit(&reference[rt], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 accum[RT][NT];
    bool extreme[RT];
    bool nonfinite[RT];
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < RT; ++rt) {
        extreme[rt] = false;
        nonfinite[rt] = false;
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
            accum[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }
    // Preserve the admitted T8 kernel's K partition and each accumulator's
    // instruction order. Only independent token accumulators are interleaved.
    for (uint chunk = sg; chunk < columns / 64; chunk += nsg) {
        const uint ci = chunk * 4 + fn / 2;
        const uint at = chunk * 64 + 16 * (fm / 2) + 8 * (fm % 2);
        #pragma clang loop unroll(full)
        for (uint t = 0; t < NT; ++t) {
            const uint row = min(first_row + t * 8 + fm, rows - 1);
            const uint first = ci * 16;
            device const uchar *block = codes + ulong(row) * row_bytes
                + ulong(first / 256) * hybrid_packed_block_bytes;
            float4x4 weights;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16; ++j) {
                weights[j / 4][j % 4] = hybrid_quant(block, hybrid_packed_kind, first % 256 + j);
            }
            #pragma clang loop unroll(full)
            for (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 a;
                a.thread_elements()[0] = weights[s / 4][s % 4];
                a.thread_elements()[1] = weights[s / 4 + 2][s % 4];
                #pragma clang loop unroll(full)
                for (uint rt = 0; rt < RT; ++rt) {
                    const uint token0 = min(first_token + rt * 8 + fn, tokens - 1);
                    const uint token1 = min(first_token + rt * 8 + fn + 1, tokens - 1);
                    const float x0 = input[ulong(token0) * columns + at + s];
                    const float x1 = input[ulong(token1) * columns + at + s];
                    if (t == 0) {
                        extreme[rt] = extreme[rt] || abs(x0) > 1e20f || abs(x1) > 1e20f;
                        nonfinite[rt] = nonfinite[rt] || !isfinite(x0) || !isfinite(x1);
                    }
                    simdgroup_float8x8 b;
                    b.thread_elements()[0] = x0;
                    b.thread_elements()[1] = x1;
                    simdgroup_multiply_accumulate(accum[rt][t], a, b, accum[rt][t]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < RT; ++rt) {
        if (extreme[rt] || nonfinite[rt]) {
            atomic_store_explicit(&reference[rt], 1u, memory_order_relaxed);
        }
        if (nonfinite[rt]) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint rt = 0; rt < RT; ++rt) {
        // The reference choice stays scoped to one T8 input group. A large
        // input in another group must not change this group's reduction tree.
        if (atomic_load_explicit(&reference[rt], memory_order_relaxed)) {
            for (uint flat = sg; flat < (8 * NT) * 8; flat += nsg) {
                const uint row = first_row + flat % (8 * NT);
                const uint token = first_token + rt * 8 + flat / (8 * NT);
                if (row < rows && token < tokens) {
                    const float total = hybrid_packed_block_dot(codes, input,
                        ulong(row) * row_bytes, ulong(token) * columns, columns, lane);
                    if (lane == 0) {
                        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
                        output[ulong(token) * rows + row] = total;
                    }
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint t = 0; t < NT; ++t) {
                red[(sg * NT + t) * 64 + 2 * lane] = accum[rt][t].thread_elements()[0];
                red[(sg * NT + t) * 64 + 2 * lane + 1] = accum[rt][t].thread_elements()[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint index = tid; index < NT * 64; index += nsg * 32) {
                float sum = 0.0f;
                for (uint split = 0; split < nsg; ++split) sum += red[split * NT * 64 + index];
                const uint t = index / 64;
                const uint owner_lane = (index % 64) / 2;
                const uint row = first_row + t * 8 + hybrid_kquant_mma_fm(owner_lane);
                const uint token = first_token + rt * 8 + hybrid_kquant_mma_fn(owner_lane) + index % 2;
                if (row < rows && token < tokens) {
                    if (!isfinite(sum)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
                    output[ulong(token) * rows + row] = sum;
                }
            }
        }
        // All groups either used the reference route or consumed this rt's
        // reduction before another rt overwrites the shared backing.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define HYBRID_KQUANT_DENSE_WIDE(NT, RT) \
kernel void hybrid_kquant_dense_wide_nt##NT##_rt##RT( \
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]], \
    device atomic_uint *failure [[buffer(4)]], threadgroup float *red [[threadgroup(0)]], \
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup atomic_uint reference[RT]; \
    hybrid_kquant_dense_wide_impl<NT, RT>(codes, input, output, p, failure, red, reference, tile, lane, sg); \
}
HYBRID_KQUANT_DENSE_WIDE(4, 4)
HYBRID_KQUANT_DENSE_WIDE(2, 8)
HYBRID_KQUANT_DENSE_WIDE(1, 16)
#undef HYBRID_KQUANT_DENSE_WIDE
