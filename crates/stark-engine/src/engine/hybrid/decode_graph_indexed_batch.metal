// SPDX-License-Identifier: Apache-2.0
// Original Stark multi-token IQ2_XS/IQ3_XXS schedule. The existing vector8
// dequantization and multiply/add chronology are preserved independently for
// each route. Each decoded weight is shared by up to four routes to one expert.
// GGML layouts/tables retain LICENSE.ggml-hybrid provenance. The grouped-route
// design was informed by llama.cpp d81235049384534c167caea52b85a694f6103d14,
// ggml/src/ggml-metal/ggml-metal-ops.cpp, ggml_metal_op_mul_mat_id. No upstream
// kernel body is copied here.

// One bounded threadgroup packs all routes. Each expert's list is stable in
// token/selected-slot order; descriptors contain only populated groups.
// p: output rows, input columns, experts, selected/token, tokens, batched input,
// total routes, routes/descriptor. Maps hold [count, route...] in an experts x
// (routes+1) array. The checked host submits width4 for vector8 and8 for MMA.
kernel void hybrid_graph_route_copy(
    device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
    constant uint *p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) output[i] = input[i];
}

kernel void hybrid_graph_iq_batch_route_pack(
    device const uint *indices [[buffer(0)]],
    device uint *maps [[buffer(1)]],
    device uint *descriptors [[buffer(2)]],
    device uint *descriptor_count [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint stride = p[6] + 1;
    for (uint expert = tid; expert < p[2]; expert += 256) {
        const uint base = expert * stride;
        uint count = 0;
        for (uint route = 0; route < p[6]; ++route) {
            const uint selected = indices[route];
            if (expert == 0 && selected >= p[2])
                atomic_store_explicit(failure, 1u, memory_order_relaxed);
            if (selected == expert) maps[base + 1 + count++] = route;
        }
        maps[base] = count;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint emitted = 0;
        for (uint expert = 0; expert < p[2]; ++expert) {
            const uint base = expert * stride;
            const uint count = maps[base];
            for (uint first = 0; first < count; first += p[7]) {
                descriptors[3 * emitted] = expert;
                descriptors[3 * emitted + 1] = base + 1 + first;
                descriptors[3 * emitted + 2] = min(p[7], count - first);
                ++emitted;
            }
        }
        descriptor_count[0] = emitted;
    }
}

kernel void hybrid_graph_iq_indexed_batch(
    device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    device const uint *maps [[buffer(4)]],
    device const uint *descriptors [[buffer(5)]],
    device const uint *descriptor_count [[buffer(6)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (grid.y >= descriptor_count[0]) return;
    const uint first_row = grid.x * 4 + simd_group * 2;
    if (first_row >= p[0]) return;
    const uint expert = descriptors[3 * grid.y];
    const uint first = descriptors[3 * grid.y + 1];
    const uint count = descriptors[3 * grid.y + 2];
    uint routes[4] = {0, 0, 0, 0};
    ulong input_offsets[4] = {0, 0, 0, 0};
    for (uint token = 0; token < count; ++token) {
        routes[token] = maps[first + token];
        input_offsets[token] = ulong(p[5] ? routes[token] : routes[token] / p[3]) * p[1];
    }
    const ulong row_bytes = ulong(p[1] >> 8) * hybrid_packed_block_bytes;
    float sums[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint col = lane * 8; col < p[1]; col += 256) {
        float values[4][8];
        for (uint token = 0; token < count; ++token) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) values[token][j] = input[input_offsets[token] + col + j];
        }
        const uint part = (col & 255) >> 5;
        const uint l = (col & 31) >> 3;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            if (first_row + r >= p[0]) continue;
            device const uchar *block = codes + (ulong(expert) * p[0] + first_row + r) * row_bytes
                                              + ulong(col >> 8) * hybrid_packed_block_bytes;
            uint signs;
            ulong grid_values;
            float scale;
            if (hybrid_packed_kind == 17) {
                const uint q = hybrid_u16(block + 2 + part * 8 + l * 2);
                const uint scales = block[66 + part];
                grid_values = iq2xs_grid[q & 511];
                signs = ksigns[q >> 9];
                scale = hybrid_half(block) * (0.5f + float((scales >> (l < 2 ? 0 : 4)) & 15)) * 0.25f;
            } else {
                const uint ss = hybrid_u32(block + 66 + part * 4);
                signs = ksigns[(ss >> (7 * l)) & 127];
                const uint at = 2 + part * 8 + 2 * l;
                grid_values = ulong(iq3xxs_grid[block[at]]) | (ulong(iq3xxs_grid[block[at + 1]]) << 32);
                scale = hybrid_half(block) * (0.5f + float(ss >> 28)) * 0.5f;
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const float weight = scale * float((grid_values >> (8 * j)) & 255ul)
                                     * ((signs & (1 << j)) ? -1.0f : 1.0f);
                for (uint token = 0; token < count; ++token) sums[r][token] += weight * values[token][j];
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        for (uint token = 0; token < count; ++token) {
            const float total = simd_sum(sums[r][token]);
            if (lane == 0 && first_row + r < p[0]) output[ulong(routes[token]) * p[0] + first_row + r] = total;
        }
    }
}
