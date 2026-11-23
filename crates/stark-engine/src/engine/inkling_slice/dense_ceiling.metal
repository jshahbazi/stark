// SPDX-License-Identifier: Apache-2.0
//! Diagnostic decomposition of the default dense prefill product (128 x 64
//! tile, BK 64, eight SIMD groups of 32 x 32). Timing only; the outputs are
//! not meaningful. Requires dense_fast.metal in the same source.
//!
//! - ink_dceil_tensor: tensor work only. Both operand blocks are loaded into
//!   threadgroup memory once and every K step reuses them.
//! - ink_dceil_loads_device: the per-SIMD-group device reads of the
//!   device-tensor-view form (32 rows x BK activations, 32 columns x BK
//!   weights per step), as vector loads without tensor work.
//! - ink_dceil_loads_staged: the cooperative 128 x BK and 64 x BK staging
//!   of the staged form, with its barriers, without tensor work.
//! - ink_dceil_store: the output store alone.

constant constexpr uint DC_BM = 128, DC_BN = 64, DC_BK = 64, DC_BKP = DC_BK + 8;

kernel void ink_dceil_tensor(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint worker [[thread_index_in_threadgroup]]) {
    threadgroup bfloat a_tile[DC_BM * DC_BKP];
    threadgroup bfloat b_tile[DC_BN * DC_BKP];
    const uint width = shape.x, columns = shape.y;
    const uint tileRow0 = group.y * DC_BM, tileCol0 = group.x * DC_BN;
    for (uint task = worker; task < (DC_BM + DC_BN) * (DC_BK / 8); task += 256) {
        const bool is_a = task < DC_BM * (DC_BK / 8);
        const uint local = is_a ? task : task - DC_BM * (DC_BK / 8);
        const uint line = local / (DC_BK / 8), kk = (local % (DC_BK / 8)) * 8;
        if (is_a)
            *reinterpret_cast<threadgroup uint4 *>(a_tile + line * DC_BKP + kk) =
                *reinterpret_cast<device const uint4 *>(input + (tileRow0 + line) * width + kk);
        else
            *reinterpret_cast<threadgroup uint4 *>(b_tile + line * DC_BKP + kk) =
                *reinterpret_cast<device const uint4 *>(weight + (tileCol0 + line) * width + kk);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rowOff = (simd / 2) * 32, colOff = (simd % 2) * 32;
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using Staged = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, DC_BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto upper = product.template get_destination_cooperative_tensor<Staged, Staged, float>();
    auto lower = product.template get_destination_cooperative_tensor<Staged, Staged, float>();
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        upper[i] = 0;
        lower[i] = 0;
    }
    auto rhs = Staged(b_tile + colOff * DC_BKP, Shape(DC_BK, 32), Strides{1, int(DC_BKP)});
    auto top = Staged(a_tile + rowOff * DC_BKP, Shape(DC_BK, 16), Strides{1, int(DC_BKP)});
    auto bottom = Staged(a_tile + (rowOff + 16) * DC_BKP, Shape(DC_BK, 16), Strides{1, int(DC_BKP)});
    for (uint k = 0; k < width; k += DC_BK) {
        product.run(top, rhs, upper);
        product.run(bottom, rhs, lower);
    }
    const uint row0 = tileRow0 + rowOff, col0 = tileCol0 + colOff;
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        const auto at = upper.get_multidimensional_index(i);
        output[(row0 + at[1]) * columns + col0 + at[0]] = bfloat(upper[i]);
        output[(row0 + 16 + at[1]) * columns + col0 + at[0]] = bfloat(lower[i]);
    }
}

kernel void ink_dceil_loads_device(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint width = shape.x, columns = shape.y;
    const uint row0 = group.y * DC_BM + (simd / 2) * 32, col0 = group.x * DC_BN + (simd % 2) * 32;
    uint4 sum = 0;
    for (uint k = 0; k < width; k += DC_BK) {
        // 32 lines x 64 K = 256 uint4 per operand; eight per lane.
        for (uint task = lane; task < 256; task += 32) {
            const uint line = task / 8, kk = (task % 8) * 8;
            sum ^= *reinterpret_cast<device const uint4 *>(input + (row0 + line) * width + k + kk);
            sum ^= *reinterpret_cast<device const uint4 *>(weight + (col0 + line) * width + k + kk);
        }
    }
    const uint folded = sum.x ^ sum.y ^ sum.z ^ sum.w;
    output[(row0 + lane) * columns + col0] = as_type<bfloat>(ushort(folded));
}

kernel void ink_dceil_loads_staged(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint worker [[thread_index_in_threadgroup]]) {
    threadgroup bfloat a_tile[DC_BM * DC_BKP];
    threadgroup bfloat b_tile[DC_BN * DC_BKP];
    const uint width = shape.x, columns = shape.y;
    const uint tileRow0 = group.y * DC_BM, tileCol0 = group.x * DC_BN;
    uint4 sum = 0;
    for (uint k = 0; k < width; k += DC_BK) {
        for (uint task = worker; task < (DC_BM + DC_BN) * (DC_BK / 8); task += 256) {
            const bool is_a = task < DC_BM * (DC_BK / 8);
            const uint local = is_a ? task : task - DC_BM * (DC_BK / 8);
            const uint line = local / (DC_BK / 8), kk = (local % (DC_BK / 8)) * 8;
            if (is_a)
                *reinterpret_cast<threadgroup uint4 *>(a_tile + line * DC_BKP + kk) =
                    *reinterpret_cast<device const uint4 *>(input + (tileRow0 + line) * width + k + kk);
            else
                *reinterpret_cast<threadgroup uint4 *>(b_tile + line * DC_BKP + kk) =
                    *reinterpret_cast<device const uint4 *>(weight + (tileCol0 + line) * width + k + kk);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum ^= *reinterpret_cast<threadgroup const uint4 *>(a_tile + (worker % DC_BM) * DC_BKP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint folded = sum.x ^ sum.y ^ sum.z ^ sum.w;
    output[(tileRow0 + worker % DC_BM) * columns + tileCol0 + worker / DC_BM] = as_type<bfloat>(ushort(folded));
}

kernel void ink_dceil_store(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint columns = shape.y;
    const uint row0 = group.y * DC_BM + (simd / 2) * 32, col0 = group.x * DC_BN + (simd % 2) * 32;
    for (uint row = 0; row < 32; ++row)
        output[(row0 + row) * columns + col0 + lane] = bfloat(float(row + lane));
}
