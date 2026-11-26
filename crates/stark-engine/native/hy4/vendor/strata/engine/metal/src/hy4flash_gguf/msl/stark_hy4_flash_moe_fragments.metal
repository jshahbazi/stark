// SPDX-License-Identifier: Apache-2.0
// Gathered fragment layout adapted from Stark's inkling_slice/grouped_nax.metal.
// It follows mlx-serve 2fe9be834a601c70ef1a597563026340f0e722fe, oMLX d6b2b92
// (Apache 2.0), and MLX steel/gemm/nax.h (Copyright (c) 2025 Apple Inc.,
// MIT; LICENSE.mlx-reference). The packed formats and plan are Strata's.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

constant constexpr uint STARK_MOE_FRAG_M = 64, STARK_MOE_FRAG_N = 64;
constant constexpr uint STARK_MOE_FRAG_K = 64, STARK_MOE_FRAG_LD = 72;
constant constexpr uint STARK_MOE_FRAG_WEIGHT = STARK_MOE_FRAG_N * STARK_MOE_FRAG_LD;

// One 16-weight block fragment. The bytes are loaded before the tensor
// instructions and expanded into the other weight tile after them. A packed
// K-block and all sixteen-value boundaries are only two-byte aligned.
struct StarkMoePackedFragment {
    ushort low[8], high[8];
    float scale, minimum;
    uint shift, high_bit;
};

template <bool Q3>
static inline void stark_moe_fragment_fetch(device const uchar* row, uint k, thread StarkMoePackedFragment& packed) {
    const uint within = k & 255;
    device const uchar* low;
    if (Q3) {
        device const uchar* block = row + (k >> 8) * 110;
        device const uchar* scales = block + 96;
        const uint group = within >> 4, offset = (group & 1) * 16;
        const uint low_scale = group < 8 ? (scales[group] & 15) : (scales[group - 8] >> 4);
        const uint high_scale = (scales[8 + (group & 3)] >> (2 * (group >> 2))) & 3;
        packed.scale = float(*(device const half*)(block + 108)) * float(int(low_scale | (high_scale << 4)) - 32);
        packed.shift = 2 * ((group >> 1) & 3);
        packed.high_bit = 1 << (group >> 1);
        low = block + 32 + 32 * (group >> 3) + offset;
        const device ushort* high = (const device ushort*)(block + offset);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) packed.high[i] = high[i];
    } else {
        device const uchar* block = row + (k >> 8) * 144;
        const uint chunk = within >> 6, high_nibble = (within >> 5) & 1;
        uint scale, minimum;
        mm_scale_min_k4(2 * chunk + high_nibble, block + 4, scale, minimum);
        packed.scale = float(*(device const half*)block) * float(scale);
        packed.minimum = float(*(device const half*)(block + 2)) * float(minimum);
        packed.shift = high_nibble * 4;
        low = block + 16 + 32 * chunk + (within & 31);
    }
    const device ushort* codes = (const device ushort*)low;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) packed.low[i] = codes[i];
}

template <bool Q3>
static inline void stark_moe_fragment_stage(const thread StarkMoePackedFragment& packed, threadgroup half* tile) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const ushort bits = packed.low[i];
        const uint2 code = (uint2(bits & 255u, bits >> 8) >> packed.shift) & (Q3 ? 3u : 15u);
        float2 decoded;
        if (Q3) {
            const ushort high = packed.high[i];
            const uint2 flags = uint2(high & 255u, high >> 8) & packed.high_bit;
            const int2 q = int2(code) - select(int2(4), int2(0), flags != uint2(0));
            decoded = packed.scale * float2(q);
        } else {
            decoded = packed.scale * float2(code) - packed.minimum;
        }
        ((threadgroup half2*)tile)[i] = half2(decoded);
    }
}

template <bool GateUp, typename Fragment>
static inline void stark_moe_fragment_store(thread Fragment& acc, device float* output, uint col0,
                                             uint first_entry, uint first_row, uint rows, uint width,
                                             float limit, device const int* route_weights) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); i += GateUp ? 2 : 1) {
        const auto at = acc.get_multidimensional_index(i);
        const uint row = first_row + at[1], col = col0 + at[0] / (GateUp ? 2 : 1);
        if (row >= rows) continue;
        const uint entry = first_entry + row;
        if (GateUp) {
            // A lane owns groups of four consecutive output channels, so
            // the gate and up fragment values are adjacent in its registers.
            const float gate = min(acc[i], limit), up = clamp(acc[i + 1], -limit, limit);
            output[ulong(entry) * width + col] = hy4_flash_silu(gate) * up;
        } else {
            output[ulong(entry) * width + col] = acc[i] * as_type<float>(route_weights[entry]);
        }
    }
}

// Four SIMD groups own four 32x32 regions of a 64x64 matrix. Gate and up
// alternate in the column dimension, giving 32 SwiGLU outputs per group.
// Both products gather activations directly into F16 register fragments
// and use one running F32 accumulator. Only weights use shared memory.
template <bool GateUp>
kernel void stark_hy4_moe_fragments(
    device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], device const int* plan [[buffer(3)]],
    constant Hy4FlashMmArgs& a [[buffer(4)]], device const int2* tiles [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr uint M = STARK_MOE_FRAG_M, N = STARK_MOE_FRAG_N, K = STARK_MOE_FRAG_K, LD = STARK_MOE_FRAG_LD;
    constexpr uint Stage = STARK_MOE_FRAG_WEIGHT, OutTile = GateUp ? N / 2 : N;
    if (group.y >= a.n_tile) return;
    const int2 tile = tiles[group.y];
    if (tile.x < 0) return;
    const uint first_entry = plan[a.off_gstart + tile.x] + tile.y;
    const uint rows = min(M, uint(plan[a.off_gstart + tile.x + 1]) - first_entry);
    if (rows == 0) return;

    const uint expert = plan[a.off_gexp + tile.x];
    const uint width = GateUp ? a.n_latent : a.n_ff, out_width = GateUp ? a.n_ff : a.n_latent;
    const uint row_bytes = GateUp ? a.gu_row : a.d_row;
    const ulong stride = GateUp ? a.gu_stride : a.d_stride;
    const uint col0 = group.x * OutTile, sg_row = (sg / 2) * 32, sg_col = (sg % 2) * 32;
    const uint row = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint col = (((lane >> 2) & 2) | (lane & 1)) * 4;
    const uint k_part = (tid % 4) * 16, load_row = tid / 4;
    const bool q3 = GateUp && hy4_flash_mm_gu_type == T_Q3_K;

    device const uchar* packed_rows[2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        const uint r = load_row + i * 32;
        const ulong offset = GateUp ? ((r & 1) ? a.up_off : a.gate_off) : a.down_off;
        packed_rows[i] = weights + expert * stride + offset + ulong(col0 + r / (GateUp ? 2 : 1)) * row_bytes;
    }
    device const float* input_rows[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) {
            // Padded rows repeat a live row. Their outputs are discarded,
            // and a row's result only depends on its own activation.
            const uint entry = first_entry + min(sg_row + i * 16 + row + j * 8, rows - 1);
            const uint source = GateUp ? uint(plan[a.off_etok + entry]) : entry;
            input_rows[i][j] = input + ulong(source) * width + col;
        }
    }

    constexpr auto descriptor =
        matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> product;
    auto left = product.template get_left_input_cooperative_tensor<half, half, float>();
    auto right = product.template get_right_input_cooperative_tensor<half, half, float>();
    auto top = product.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(left)>,
        remove_addrspace_t<decltype(right)>, float>();
    auto bottom = product.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(left)>,
        remove_addrspace_t<decltype(right)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < top.get_capacity(); ++i) {
        top[i] = 0.0f;
        bottom[i] = 0.0f;
    }

    alignas(16) threadgroup half staged[2 * Stage];
    StarkMoePackedFragment pending[2];
    auto fetch = [&](uint k) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            if (q3) stark_moe_fragment_fetch<true>(packed_rows[i], k + k_part, pending[i]);
            else stark_moe_fragment_fetch<false>(packed_rows[i], k + k_part, pending[i]);
        }
    };
    auto stage = [&](uint buffer) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            threadgroup half* dst = staged + buffer * Stage + (load_row + i * 32) * LD + k_part;
            if (q3) stark_moe_fragment_stage<true>(pending[i], dst);
            else stark_moe_fragment_stage<false>(pending[i], dst);
        }
    };
    fetch(0);
    stage(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < width; k += K) {
        const uint buffer = (k / K) & 1;
        const bool more = k + K < width;
        if (more) fetch(k + K);
        threadgroup const half* current = staged + buffer * Stage + sg_col * LD;
        if (sg_row < rows) {
            #pragma clang loop unroll(full)
            for (uint step = 0; step < K; step += 16) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    const half4 v = *(threadgroup const half4*)(current + (row + i * 8) * LD + step + col);
                    #pragma clang loop unroll(full)
                    for (uint d = 0; d < 4; ++d) right[i * 4 + d] = v[d];
                }
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2; ++i) {
                    if (sg_row + i * 16 < rows) {
                        #pragma clang loop unroll(full)
                        for (uint j = 0; j < 2; ++j) {
                            const half4 v = half4(*(device const float4*)(input_rows[i][j] + k + step));
                            #pragma clang loop unroll(full)
                            for (uint d = 0; d < 4; ++d) left[j * 4 + d] = v[d];
                        }
                        if (i == 0) product.run(left, right, top);
                        else product.run(left, right, bottom);
                    }
                }
            }
        }
        if (more) {
            stage(buffer ^ 1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const uint column = col0 + sg_col / (GateUp ? 2 : 1);
    device const int* route_weights = plan + a.off_ew;
    stark_moe_fragment_store<GateUp>(top, output, column, first_entry, sg_row, rows, out_width, a.limit, route_weights);
    stark_moe_fragment_store<GateUp>(bottom, output, column, first_entry, sg_row + 16, rows, out_width, a.limit,
                                     route_weights);
}

template [[host_name("stark_hy4_moe_fragments_gu")]] kernel decltype(stark_hy4_moe_fragments<true>)
    stark_hy4_moe_fragments<true>;
template [[host_name("stark_hy4_moe_fragments_down")]] kernel decltype(stark_hy4_moe_fragments<false>)
    stark_hy4_moe_fragments<false>;
