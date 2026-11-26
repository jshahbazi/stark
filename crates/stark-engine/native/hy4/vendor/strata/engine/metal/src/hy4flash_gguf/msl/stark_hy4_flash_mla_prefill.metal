// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// The cooperative tile layout follows llama.cpp ggml-metal/kernels/fa_tensor.metal
// at 2cbca389f515df455412dafa07e2b1359cc72a6d plus AngelSlim Hy4 patch
// 27810270f635fa70c1c8a665bc1a35b4352e6f3c5846ddb793098e6706f9db35.
// MIT notice: vendor/strata/engine/third_party/ggml/LICENSE.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct StarkHy4PrefillMlaArgs {
    uint n_tok, q_stride, kv_stride, o_stride, window;
    float scale;
};

static inline int stark_hy4_mla_first_key(int p, uint window) {
    return window ? max(0, p - int(window) + 1) : 0;
}

// The host uses this kernel only for contiguous causal or sliding visibility.
// Each workgroup reads one head across 16 queries. All 16 rows share each
// device K/V tile, and keep Q in threadgroup memory across that traversal.
// Q and P are F16 operands, with F32 scores, statistics, and output.
kernel void stark_hy4_mla_prefill_tensor(
    device const float* query [[buffer(0)]], device const half* cache [[buffer(1)]],
    device const float* sinks [[buffer(2)]], device const int* positions [[buffer(3)]],
    device float* output [[buffer(4)]], constant StarkHy4PrefillMlaArgs& a [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr int Q = 16, C = 64, DK = 576, DV = 512, SG = 8, Threads = SG * 32;
    const uint token0 = group.x * Q, head = group.y;
    const int rows = min(Q, int(a.n_tok - token0));
    const int first_pos = positions[token0], last_pos = positions[token0 + rows - 1];
    const int first_key = stark_hy4_mla_first_key(first_pos, a.window) / C * C;
    const int last_key = last_pos + 1;
    const int full_low = stark_hy4_mla_first_key(last_pos, a.window);
    const float sink = sinks[head];
    query += ulong(token0) * a.q_stride + head * DK;
    output += ulong(token0) * a.o_stride + head * DV;

    alignas(16) threadgroup half staged_q[Q * DK];
    threadgroup float scores[Q * C], rescale[Q];
    threadgroup half probability[Q * C];
    threadgroup atomic_uint rescaled;
    for (int i = tid; i < Q * DK / 4; i += Threads) {
        const int row = i / (DK / 4), d = i % (DK / 4);
        ((threadgroup half4*)staged_q)[i] = row < rows ?
            half4(((device const float4*)(query + ulong(row) * a.q_stride))[d]) : half4(0);
    }

    constexpr auto qk_desc = matmul2d_descriptor(Q, C, DK, false, true, false,
                                                 matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(Q, DV, C, false, false, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<SG>> qk;
    matmul2d<pv_desc, execution_simdgroups<SG>> pv;
    auto q = tensor(staged_q, dextents<int32_t, 2>(DK, Q));
    auto s = tensor(scores, dextents<int32_t, 2>(C, Q));
    auto p = tensor(probability, dextents<int32_t, 2>(C, Q));
    auto v_view = tensor((device half*)cache, dextents<int32_t, 2>(DV, C), array<int, 2>{1, int(a.kv_stride)});
    auto result = pv.template get_destination_cooperative_tensor<decltype(p), decltype(v_view), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < result.get_capacity(); ++i) {
        if (result.is_valid_element(i)) result[i] = 0.0f;
    }

    float peak[Q / SG] = {sink, sink}, mass[Q / SG] = {1.0f, 1.0f};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = first_key; k < last_key; k += C) {
        const int columns = min(C, last_key - k);
        const bool all_visible = k >= full_low && k + C - 1 <= first_pos;
        device half* latent = (device half*)(cache + ulong(k) * a.kv_stride);
        if (tid == 0) atomic_store_explicit(&rescaled, 0u, memory_order_relaxed);
        auto key_tile = tensor(latent, dextents<int32_t, 2>(DK, columns), array<int, 2>{1, int(a.kv_stride)});
        qk.run(q, key_tile, s);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int j = 0; j < Q / SG; ++j) {
            const int r = j * SG + sg;
            const int pos = r < rows ? positions[token0 + r] : -1;
            const int lo = stark_hy4_mla_first_key(pos, a.window);
            float logits[C / 32];
            float m = peak[j];
            #pragma clang loop unroll(full)
            for (int i = 0; i < C / 32; ++i) {
                const int c = lane + 32 * i;
                const bool visible = r < rows && (all_visible || (k + c >= lo && k + c <= pos));
                logits[i] = visible ? scores[r * C + c] * a.scale : -INFINITY;
                m = max(m, logits[i]);
            }
            m = simd_max(m);
            const float factor = precise::exp(peak[j] - m);
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < C / 32; ++i) {
                const float e = precise::exp(logits[i] - m);
                probability[r * C + lane + 32 * i] = half(e);
                total += e;
            }
            mass[j] = mass[j] * factor + simd_sum(total);
            if (lane == 0) {
                rescale[r] = factor;
                if (m > peak[j] && k != first_key)
                    atomic_store_explicit(&rescaled, 1u, memory_order_relaxed);
            }
            peak[j] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (atomic_load_explicit(&rescaled, memory_order_relaxed) != 0u) {
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < result.get_capacity(); ++i) {
                if (result.is_valid_element(i)) result[i] *= rescale[result.get_multidimensional_index(i)[1]];
            }
        }
        auto value_tile = tensor(latent, dextents<int32_t, 2>(DV, columns), array<int, 2>{1, int(a.kv_stride)});
        pv.run(p, value_tile, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < Q / SG; ++j) rescale[j * SG + sg] = mass[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < result.get_capacity(); ++i) {
        if (result.is_valid_element(i)) result[i] /= rescale[result.get_multidimensional_index(i)[1]];
    }
    auto out = tensor(output, dextents<int32_t, 2>(DV, rows), array<int, 2>{1, int(a.o_stride)});
    result.store(out);
}
