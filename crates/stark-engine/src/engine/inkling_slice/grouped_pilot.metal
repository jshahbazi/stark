// SPDX-License-Identifier: MIT
// Grouped projection adapted from Ollama mlx/nvfp4_matmul.go at
// 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8.
/*
MIT License

Copyright (c) Ollama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Independent row axis over Stark's existing Inkling router arithmetic.
kernel void ink_group_router(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device float *logits [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint WIDTH = 2304;
    constexpr uint EXPERTS = 128;
    const device bfloat *row = input + group.y * WIDTH;
    const device bfloat *weight = weights + group.x * WIDTH;
    float dot = 0.0f;
    for (uint i = lane; i < WIDTH; i += 32) {
        dot = fma(float(row[i]), float(weight[i]), dot);
    }
    const float sum = simd_sum(dot);
    if (lane == 0) logits[group.y * EXPERTS + group.x] = sum;
}

kernel void ink_group_select(
    device const float *all_logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device uint *all_ids [[buffer(2)]],
    device float *all_weights [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const device float *logits = all_logits + row * 128;
    device uint *ids = all_ids + row * 8;
    device float *weights = all_weights + row * 8;
    threadgroup float choices[128];
    choices[lane] = ink_sigmoid(logits[lane]) + bias[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;

    float scores[8];
    float max_score = -INFINITY;
    for (uint route = 0; route < 8; ++route) {
        uint selected = 0;
        for (uint expert = 1; expert < 128; ++expert) {
            if (choices[expert] > choices[selected]) selected = expert;
        }
        choices[selected] = -INFINITY;
        ids[route] = selected;
        const float x = logits[selected];
        const float exponential = exp(-abs(x));
        const float one_plus = 1.0f + exponential;
        const float softplus = one_plus == 1.0f ? exponential : log(one_plus) * (exponential / (one_plus - 1.0f));
        scores[route] = min(x, 0.0f) - softplus;
        max_score = max(max_score, scores[route]);
    }
    float total = 0.0f;
    for (uint route = 0; route < 8; ++route) {
        scores[route] = exp(scores[route] - max_score);
        total += scores[route];
    }
    for (uint route = 0; route < 8; ++route) weights[route] = scores[route] * (8.0f / total);
}

// Make the sorted permutation and tile runs without a CPU readback. Each
// expert preserves its source route order and owns disjoint output ranges.
kernel void ink_group_bucket(
    device const uint *ids [[buffer(0)]],
    device uint *order [[buffer(1)]],
    device uint *runs [[buffer(2)]],
    constant uint &routes [[buffer(3)]],
    uint expert [[thread_index_in_threadgroup]]) {
    constexpr uint EXPERTS = 128;
    constexpr uint TILE = 32;
    const uint run_slots = (routes + TILE - 1) / TILE + EXPERTS - 1;
    threadgroup uint route_prefix[EXPERTS], run_prefix[EXPERTS];
    uint count = 0;
    for (uint route = 0; route < routes; ++route) count += uint(ids[route] == expert);
    const uint tiles = (count + TILE - 1) / TILE;
    route_prefix[expert] = count;
    run_prefix[expert] = tiles;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < EXPERTS; stride *= 2) {
        const uint prior_routes = expert >= stride ? route_prefix[expert - stride] : 0;
        const uint prior_runs = expert >= stride ? run_prefix[expert - stride] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        route_prefix[expert] += prior_routes;
        run_prefix[expert] += prior_runs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint start = route_prefix[expert] - count;
    uint cursor = start;
    for (uint route = 0; route < routes; ++route) {
        if (ids[route] == expert) order[cursor++] = route;
    }
    const uint first_run = run_prefix[expert] - tiles;
    for (uint tile = 0; tile < tiles; ++tile) {
        const uint offset = tile * TILE;
        const uint record = (first_run + tile) * 3;
        runs[record] = start + offset;
        runs[record + 1] = min(TILE, count - offset);
        runs[record + 2] = expert;
    }
    for (uint tile = run_prefix[EXPERTS - 1] + expert; tile < run_slots; tile += EXPERTS) {
        runs[tile * 3] = 0;
        runs[tile * 3 + 1] = 0;
        runs[tile * 3 + 2] = 0;
    }
}

kernel void ink_group_copy_row(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    output[column] = input[column];
}

kernel void ink_group_combine(
    device const bfloat *down [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
    constexpr uint WIDTH = 2304;
    float sum = 0.0f;
    for (uint route = 0; route < 8; ++route) {
        sum += float(down[(position.y * 8 + route) * WIDTH + position.x]);
    }
    output[position.y * WIDTH + position.x] = bfloat(sum);
}

inline int nvfp4_binary_quantum(float f) {
    uint raw = as_type<uint>(abs(f));
    if (raw == 0)
        return 127;
    return int(raw >> 23) - 150 + int(ctz((raw & 0x7fffffu) | 0x800000u));
}
// Bound exact accumulation by 0.999 * 2^(quantum + 24). Adjusting the
// exponent of 0.999 avoids ldexp in the tile loop. Zero forces native
// replay if an intermediate could be subnormal; exponents above 104
// put the bound beyond the largest finite FP32 value.
inline float nvfp4_exact_limit(int quantum) {
    uint bits = as_type<uint>(0.999f) + (uint(quantum + 24) << 23);
    return quantum < -126 ? 0.f : quantum > 104 ? INFINITY : as_type<float>(bits);
}

// Quantization has already selected the four- or six-range reconstruction.
// Each reconstructed E2M1 times E4M3 value fits exactly in BF16.
kernel void ink_group_ingress(
    device const float *source [[buffer(0)]],
    device const uint *order [[buffer(1)]],
    device bfloat *activation [[buffer(2)]],
    device int *quantum [[buffer(3)]],
    device float *maxima [[buffer(4)]],
    constant uint2 &shape [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint width = shape.x;
    if (position.x >= width / 32) return;
    const uint sorted = position.y;
    const uint row = shape.y != 0 ? order[sorted] / 8 : sorted;
    const uint block = position.x * 32;
    int exponent = 127;
    float peak = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        const float value = source[row * width + block + i];
        activation[sorted * width + block + i] = bfloat(value);
        exponent = min(exponent, nvfp4_binary_quantum(value));
        peak = max(peak, abs(value));
    }
    quantum[sorted * (width / 32) + position.x] = exponent;
    maxima[sorted * (width / 32) + position.x] = peak;
}

kernel void ink_group_qmm(
    device const bfloat *activation [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uint *runs [[buffer(3)]],
    device const int *quantum [[buffer(4)]],
    device const float *maxima [[buffer(5)]],
    device float *result [[buffer(6)]],
#ifdef INK_GROUP_QMM_PROBE
    device atomic_uint *stats [[buffer(7)]],
    constant uint3 &probe [[buffer(8)]],
#else
    constant uint &Width [[buffer(7)]],
#endif
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
#ifdef INK_GROUP_QMM_PROBE
    const uint Width = probe.x, mode = probe.y;
    stats += probe.z;
    uint failedBlocks = 0, replayEntries = 0;
    threadgroup uint counts[8];
#endif
    constexpr uint Channels = 2304;
    constexpr uint BM = 32, BN = 64, BK = 32, Workers = 128;

    uint worker = thread_index_in_threadgroup, lane = thread_index_in_simdgroup,
         simd = simdgroup_index_in_threadgroup;
    uint tile = threadgroup_position_in_grid.y, start = runs[tile * 3], rows = runs[tile * 3 + 1],
         expert = runs[tile * 3 + 2];
    if (rows == 0)
        return;
    uint col0 = threadgroup_position_in_grid.x * BN;
    uint tileRow = (simd / (BN / 32)) * 16, tileColumn = (simd % (BN / 32)) * 32;
    threadgroup bfloat decodedWeights[BN * BK];
    threadgroup float final[BM * BN];
#ifdef INK_GROUP_QMM_SHARED_SCALES
    threadgroup float scalePeak[Workers];
    threadgroup int scaleQuantum[Workers];
#endif
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using ActivationTile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    using WeightTile = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    using ResultTile = metal::tensor<threadgroup float, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto accumulator =
        product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    for (uint i = 0; i < accumulator.get_capacity(); ++i)
        accumulator[i] = 0;
    // The minimum binary quantum of all products so far also divides every
    // possible FP32 prefix, including prefixes rounded to a coarser power of two.
    int smallest = 127;
    for (uint offset = 0; offset < Width; offset += BK) {
#ifdef INK_GROUP_QMM_SHARED_SCALES
        const uint channel = col0 + worker / 2, at = offset + (worker % 2) * 16;
        const uint wr = expert * Channels + channel;
        const float gain = ink_e4m3_positive(scales[wr * (Width / 16) + at / 16]);
        scalePeak[worker] = abs(gain) * 6.f;
        scaleQuantum[worker] = nvfp4_binary_quantum(gain) - 1;
        for (uint i = 0; i < 2; ++i) {
            const uint word = weights[wr * (Width / 8) + at / 8 + i];
            for (uint j = 0; j < 8; ++j)
                decodedWeights[worker * 16 + i * 8 + j] = bfloat(ink_fp4_code((word >> (4 * j)) & 15u) * gain);
        }
#else
        for (uint i = worker; i < BN * (BK / 8); i += Workers) {
            uint channel = col0 + i / (BK / 8), at = offset + (i % (BK / 8)) * 8;
            uint wr = expert * Channels + channel;
            uint word = 0;
            float gain = 0;
            if (Channels % BN == 0 || channel < Channels) {
                word = weights[wr * (Width / 8) + at / 8];
                gain = ink_e4m3_positive(scales[wr * (Width / 16) + at / 16]);
            }
            for (uint j = 0; j < 8; ++j)
                decodedWeights[i * 8 + j] = bfloat(ink_fp4_code((word >> (4 * j)) & 15u) * gain);
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tileRow < rows) {
            int valid = int(min(16u, rows - tileRow));
            float before[16] = {};
            uint replay = 0;
#ifdef INK_GROUP_QMM_PROBE
            if (mode != 2) {
#endif
            uint block = (start + tileRow + min(lane, 15u)) * (Width / BK) + offset / BK;
            float rowPeak = lane < uint(valid) ? maxima[block] : 0.f;
            float inputPeak = simd_max(rowPeak);
            int rowQuantum = lane < uint(valid) ? quantum[block] : 127;
            int minInputQuantum = simd_min(rowQuantum);
#ifdef INK_GROUP_QMM_SHARED_SCALES
            const uint scale = (tileColumn + lane) * 2;
            int weightQuantum = min(127, min(scaleQuantum[scale], scaleQuantum[scale + 1]));
            float weightPeak = max(scalePeak[scale], scalePeak[scale + 1]);
#else
            int weightQuantum = 127;
            float weightPeak = 0;
            for (uint h = 0; h < BK / 16; ++h) {
                uint channel = col0 + tileColumn + lane;
                float gain = 0;
                if (Channels % BN == 0 || channel < Channels)
                    gain = ink_e4m3_positive(scales[(expert * Channels + channel) * (Width / 16) + offset / 16 + h]);
                weightQuantum = min(weightQuantum, nvfp4_binary_quantum(gain) - 1);
                weightPeak = max(weightPeak, abs(gain) * 6.f);
            }
#endif
            smallest = min(smallest, minInputQuantum + simd_min(weightQuantum));
            float columnPeak = weightPeak;
            int columnQuantum = weightQuantum;
            weightPeak = simd_max(weightPeak);
            float partialPeak = 0;
            for (uint i = 0; i < accumulator.get_capacity(); ++i)
                partialPeak = max(partialPeak, abs(accumulator[i]));
            partialPeak = simd_max(partialPeak);
            // Every add is exact if every possible intermediate sum is an
            // integer smaller than 2^24 in this binary quantum. The maximum
            // absolute prefix plus the block L1 bound covers both reduction orders.
            float upper = partialPeak + float(BK) * inputPeak * weightPeak;
            // Native FP32 arithmetic flushes subnormal intermediates; the
            // matrix unit can preserve them. Replay if a prefix can be subnormal.
            bool safe = upper < nvfp4_exact_limit(smallest);
            // A tile can fail the common bound while individual entries pass.
            if (!safe) {
                for (uint i = 0; i < accumulator.get_capacity(); ++i) {
                    auto loc = accumulator.get_multidimensional_index(i);
                    uint row = loc[1], column = loc[0];
                    float lhs = simd_shuffle(rowPeak, row), rhs = simd_shuffle(columnPeak, column);
                    int productQuantum =
                        simd_shuffle(rowQuantum, row) + simd_shuffle(columnQuantum, column);
                    float prev = accumulator[i];
                    before[i] = prev;
                    float bound = abs(prev) + float(BK) * lhs * rhs;
                    int quantum = min(nvfp4_binary_quantum(prev), productQuantum);
                    float limit = nvfp4_exact_limit(quantum);
                    if (row < uint(valid) && !(bound < limit))
                        replay |= 1u << i;
                }
            }
#ifdef INK_GROUP_QMM_PROBE
                failedBlocks += !safe;
                replayEntries += popcount(replay);
                if (mode == 1) replay = 0;
            }
#endif
            auto lhs = ActivationTile((device bfloat *)activation + (start + tileRow) * Width + offset,
                                      Shape(BK, valid), Strides{1, int(Width)});
            auto rhs = WeightTile(decodedWeights + tileColumn * BK, Shape(BK, 32), Strides{1, BK});
            product.run(lhs, rhs, accumulator);
            // Reproduce the native FP32 reduction on the ambiguous outputs.
            while (replay) {
                uint i = ctz(replay);
                replay &= replay - 1;
                auto loc = accumulator.get_multidimensional_index(i);
                uint row = loc[1], column = loc[0];
                float value = before[i];
                for (uint h = 0; h < BK; ++h) {
                    value = fma(float(activation[(start + tileRow + row) * Width + offset + h]),
                                float(decodedWeights[(tileColumn + column) * BK + h]), value);
                }
                accumulator[i] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#ifdef INK_GROUP_QMM_PROBE
    const uint replayCount = simd_sum(replayEntries);
    if (lane == 0) {
        counts[simd * 2] = failedBlocks;
        counts[simd * 2 + 1] = replayCount;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (worker == 0) {
        if (col0 == 0) {
            atomic_fetch_add_explicit(stats, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(stats + 1, rows, memory_order_relaxed);
        }
        if (mode != 2) {
            atomic_fetch_add_explicit(stats + 2, (rows > 16 ? 4 : 2) * (Width / BK), memory_order_relaxed);
            atomic_fetch_add_explicit(stats + 3, counts[0] + counts[2] + counts[4] + counts[6], memory_order_relaxed);
            atomic_fetch_add_explicit(stats + 4, counts[1] + counts[3] + counts[5] + counts[7], memory_order_relaxed);
        }
    }
#endif
    accumulator.store(ResultTile(final + tileRow * BN + tileColumn, Shape(32, 16), Strides{1, BN}));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = worker; i < rows * BN; i += Workers) {
        uint channel = col0 + i % BN;
        if (Channels % BN == 0 || channel < Channels)
            result[(start + i / BN) * Channels + channel] = final[i];
    }
}

kernel void ink_group_gate_up(
    device const float *inner [[buffer(0)]],
    device const float *amax [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const float *weights [[buffer(3)]],
    device const uint *order [[buffer(4)]],
    device const float *expert_max [[buffer(5)]],
    device bfloat *middle [[buffer(6)]],
    constant float &denominator [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]) {
    constexpr uint MID = 1152;
    if (position.x >= MID) return;
    const uint sorted = position.y;
    const uint route = order[sorted];
    const uint column = sorted * (2 * MID) + 2 * position.x;
    const float outer = (amax[route / 8] * expert_max[ids[route]]) / (1536.0f * denominator);
    const float gate = float(bfloat(inner[column] * outer));
    const float up = float(bfloat(inner[column + 1] * outer));
#ifdef INK_GROUP_SOURCE_EPILOGUE
    // Match the F32 unary operations in MLX's compiled GateUpEpilogue.
    // unary_ops.h, fc344c986c4169d99f76b5d95ae6b4a58a359a63 (MIT).
    const float low = 1.0f / (1.0f + precise::exp(abs(gate)));
    const float sigmoid = gate < 0.0f ? low : 1.0f - low;
    const float activation = (4.0f * precise::tanh(gate * 0.25f)) * sigmoid * (25.0f * precise::tanh(up / 25.0f));
#else
    const float activation = (4.0f * tanh(gate * 0.25f)) * ink_sigmoid(gate) * (25.0f * tanh(up / 25.0f));
#endif
    middle[sorted * MID + position.x] = bfloat(activation * weights[route]);
}

kernel void ink_group_down_epilogue(
    device const float *inner [[buffer(0)]],
    device const float *amax [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const uint *order [[buffer(3)]],
    device const float *expert_max [[buffer(4)]],
    device bfloat *down [[buffer(5)]],
    constant float &denominator [[buffer(6)]],
    uint2 position [[thread_position_in_grid]]) {
    constexpr uint WIDTH = 2304;
    const uint sorted = position.y;
    const uint route = order[sorted];
    const float outer = (amax[sorted] * expert_max[ids[route]]) / (1536.0f * denominator);
    down[route * WIDTH + position.x] = bfloat(inner[sorted * WIDTH + position.x] * outer);
}
