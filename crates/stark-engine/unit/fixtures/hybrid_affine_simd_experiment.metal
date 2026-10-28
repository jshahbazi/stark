// Independent SIMD-row experiment for Stark's native affine GEMV.
// This code is original to Stark; it is not imported from another engine.
// The scalar reference schedule follows packed_matvec.metal.
#include <metal_stdlib>
using namespace metal;

struct Params { uint rows, columns, bits, group, scale_kind, rows_per_batch; };
constant uint affine_bits [[function_constant(10)]];
constant uint affine_group [[function_constant(11)]];
constant uint affine_dtype [[function_constant(12)]];

inline ushort read16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }
inline uint read32(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
inline float value(device const uchar *p) {
    if (affine_dtype == 0) return as_type<float>(read32(p));
    if (affine_dtype == 30) return as_type<float>(uint(read16(p)) << 16);
    return float(as_type<half>(read16(p)));
}

kernel void affine_baseline(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant Params &p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint item = affine_dtype == 0 ? 4 : 2;
    const uint packed_shift = affine_bits == 4 ? 1 : 0;
    const uint group_shift = affine_group == 128 ? 7 : (affine_group == 64 ? 6 : 5);
    const ulong row_element = ulong(row) * p.columns;
    const ulong code_offset = row_element >> packed_shift;
    const ulong plane_offset = row_element >> group_shift;
    const ulong input_offset = p.rows_per_batch ? ulong(row / p.rows_per_batch) * p.columns : 0;
    float sum = 0;
    for (uint col = lane; col < p.columns; col += 256) {
        const uint shift = affine_bits == 4 ? (col & 1) * 4 : 0;
        const uint q = (codes[code_offset + (col >> packed_shift)] >> shift) & ((1u << affine_bits) - 1);
        const ulong group = plane_offset + (col >> group_shift);
        const float w = value(scales + group * item) * float(q) + value(biases + group * item);
        sum += w * input[input_offset + col];
    }
    threadgroup float partial[8];
    const float total = simd_sum(sum);
    if (simd_lane == 0) partial[simd_group] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float result = 0;
        for (uint i = 0; i < 8; ++i) result += partial[i];
        output[row] = result;
    }
}

kernel void affine_simd_scalar(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant Params &p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 8 + simd_group;
    if (row >= p.rows) return;
    const uint item = affine_dtype == 0 ? 4 : 2;
    const uint packed_shift = affine_bits == 4 ? 1 : 0;
    const uint group_shift = affine_group == 128 ? 7 : (affine_group == 64 ? 6 : 5);
    const ulong row_element = ulong(row) * p.columns;
    const ulong code_offset = row_element >> packed_shift;
    const ulong plane_offset = row_element >> group_shift;
    const ulong input_offset = p.rows_per_batch ? ulong(row / p.rows_per_batch) * p.columns : 0;
    float sum = 0;
    for (uint col = lane; col < p.columns; col += 32) {
        const uint shift = affine_bits == 4 ? (col & 1) * 4 : 0;
        const uint q = (codes[code_offset + (col >> packed_shift)] >> shift) & ((1u << affine_bits) - 1);
        const ulong group = plane_offset + (col >> group_shift);
        const float w = value(scales + group * item) * float(q) + value(biases + group * item);
        sum += w * input[input_offset + col];
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}

kernel void affine_simd_vector8(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant Params &p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 8 + simd_group;
    if (row >= p.rows) return;
    const uint item = affine_dtype == 0 ? 4 : 2;
    const uint packed_shift = affine_bits == 4 ? 1 : 0;
    const uint group_shift = affine_group == 128 ? 7 : (affine_group == 64 ? 6 : 5);
    const ulong row_element = ulong(row) * p.columns;
    const ulong code_offset = row_element >> packed_shift;
    const ulong plane_offset = row_element >> group_shift;
    const ulong input_offset = p.rows_per_batch ? ulong(row / p.rows_per_batch) * p.columns : 0;
    float sum = 0;
    for (uint col = lane * 8; col < p.columns; col += 256) {
        const ulong group = plane_offset + (col >> group_shift);
        const float scale = value(scales + group * item);
        const float bias = value(biases + group * item);
        device const uchar *packed = codes + code_offset + (col >> packed_shift);
        const uint low = read32(packed);
        const uint high = affine_bits == 8 ? read32(packed + 4) : 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            const uint word = affine_bits == 4 || j < 4 ? low : high;
            const uint shift = affine_bits == 4 ? j * 4 : (j & 3) * 8;
            const uint q = (word >> shift) & ((1u << affine_bits) - 1);
            const float w = scale * float(q) + bias;
            sum += w * input[input_offset + col + j];
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}
