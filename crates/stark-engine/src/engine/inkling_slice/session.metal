// SPDX-License-Identifier: Apache-2.0

kernel void ink_embedding(
    device const bfloat *embedding [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device float *residual [[buffer(2)]],
#ifdef INK_VERIFY_DRAFT_EMBEDDING
    device const uint *proposals [[buffer(3)]], constant uint2 &row_vocabulary [[buffer(4)]],
#else
    constant uint &token [[buffer(3)]],
#endif
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sums[18];
    threadgroup float scale;
    const uint column = tid * 4;
#ifdef INK_VERIFY_DRAFT_EMBEDDING
    const uint proposed = proposals[row_vocabulary.x];
    const uint token = proposed < row_vocabulary.y ? proposed : 0u;
#endif
    const uint row = token * ink_width;
    const float4 values = float4(*reinterpret_cast<device const bfloat4 *>(embedding + row + column));
    float total = 0;
    for (uint i = 0; i < 4; ++i) total += values[i] * values[i];
    total = simd_sum(total);
    if (lane == 0) sums[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(lane < 18 ? sums[lane] : 0.0f);
        if (lane == 0) scale = metal::precise::rsqrt(total / ink_width + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        residual[column + i] = float(bfloat(float(weight[column + i]) * (values[i] * scale)));
    }
}

kernel void ink_embedding_override(
    device const bfloat *embedding [[buffer(0)]], device float *residual [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    residual[column] = float(embedding[column]);
}

kernel void ink_capture_feature(
    device const float *residual [[buffer(0)]], device bfloat *feature [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    feature[column] = bfloat(residual[column]);
}

kernel void ink_unembed_f32(
    device const bfloat *x [[buffer(0)]], device const bfloat *weights [[buffer(1)]],
    device float *logits [[buffer(2)]], constant uint &columns [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = 4 * group + sg;
    if (row >= columns) return;
    const device bfloat *weight = weights + row * ink_width;
    float4 accum = 0;
    for (uint col = lane * 4; col < ink_width; col += 128) {
        accum = fma(float4(*reinterpret_cast<device const bfloat4 *>(x + col)),
                    float4(*reinterpret_cast<device const bfloat4 *>(weight + col)), accum);
    }
    const float value = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
    if (lane == 0) logits[row] = 30.0f * metal::precise::tanh(value / 30.0f);
}

kernel void ink_logits_cap(device float *logits [[buffer(0)]], constant uint &columns [[buffer(1)]],
                            uint column [[thread_position_in_grid]]) {
    if (column < columns) logits[column] = 30.0f * metal::precise::tanh(logits[column] / 30.0f);
}
