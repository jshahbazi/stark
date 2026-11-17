// SPDX-License-Identifier: Apache-2.0
// Edge0 direct-route scheduling over unchanged GGML Q8_0/Q3_K/IQ4_NL bytes.
// Decoding uses the source-bound hybrid_quant helper from packed_matvec.metal;
// see LICENSE.ggml-hybrid. edge0_graph.metal stays byte-identical.
//
// D writes its stored column-scale quotient in the same dispatch. The row
// reduction repeats edge0_graph_packed_single_impl: the 256-column stride, the
// SIMD-group sum and the ordered eight-group F32 sum. The quotient is the F32
// division edge0_graph_column_scale applies to the same stored value, so the
// written bits match the two-dispatch schedule under the safe math library.
template<uint kind, uint block, uint bytes>
inline void edge0_direct_scaled_single_impl(
    device const uchar *codes, device const float *input, device float *output,
    device const float *stored_s, constant uint *p, device atomic_uint *failure,
    uint row, uint tid, uint lane, uint sg, threadgroup float *partial) {
    const uint rows=p[0], columns=p[1];
    if (row>=rows) return;
    float sum=0.0f;
    const ulong row_bytes=ulong(columns/block)*bytes;
    const ulong row_offset=ulong(row)*row_bytes;
    for (uint col=tid;col<columns;col+=256) {
        const float weight=hybrid_quant(codes+row_offset+ulong(col/block)*bytes,kind,col%block);
        sum+=weight*input[col];
    }
    float total=simd_sum(sum);
    if (lane==0) partial[sg]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        float result=0.0f;
        for (uint group=0;group<8;++group) result+=partial[group];
        if (!isfinite(result)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        const float scale=stored_s[row];
        const float value=result/scale;
        if (!(scale>0) || !isfinite(scale) || !isfinite(value))
            atomic_store_explicit(failure,1u,memory_order_relaxed);
        output[row]=value;
    }
}

#define EDGE0_DIRECT_SCALED_SINGLE(NAME,KIND,BLOCK,BYTES) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], \
    device const float *stored_s [[buffer(3)]], \
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(30)]], \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partial[8]; \
    edge0_direct_scaled_single_impl<KIND,BLOCK,BYTES>( \
        codes,input,output,stored_s,p,failure,row,tid,lane,sg,partial); \
}
EDGE0_DIRECT_SCALED_SINGLE(edge0_direct_q8_scaled_single,8,32,34)
EDGE0_DIRECT_SCALED_SINGLE(edge0_direct_q3_scaled_single,11,256,110)
EDGE0_DIRECT_SCALED_SINGLE(edge0_direct_iq4_scaled_single,20,32,18)
#undef EDGE0_DIRECT_SCALED_SINGLE

// Diagnostic schedule verification. Both schedules must write identical F32
// bits; a mismatch sets the command's sticky failure status.
kernel void edge0_direct_require_identical(
    device const uint *left [[buffer(0)]], device const uint *right [[buffer(1)]],
    constant uint *p [[buffer(2)]], device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=p[0]) return;
    if (left[i]!=right[i]) atomic_store_explicit(failure,1u,memory_order_relaxed);
}
