// Original Stark component variant. Each SIMD group keeps the unchanged
// vector8 dot; only the number of independent rows in a threadgroup varies.
// GGML layout/table provenance remains in LICENSE.ggml-hybrid.
constant uint hybrid_iq_rows_per_group [[function_constant(70)]];

kernel void hybrid_graph_iq_indexed_rows(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = grid.x * hybrid_iq_rows_per_group + simd_group;
    if (row >= p[0]) return;
    const uint batch = grid.y;
    const uint expert = indices[batch];
    if (expert >= p[2]) {
        if (lane == 0) {
            output[batch * p[0] + row] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    const ulong at = (ulong(expert) * p[0] + row) * (p[1] >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    const float total = hybrid_iq_dot_vector8(codes, input, at, input_offset, p[1], lane);
    if (lane == 0) output[batch * p[0] + row] = total;
}
