// SPDX-License-Identifier: Apache-2.0
// Edge0 token-batched rows over unchanged GGML Q8_0/Q3_K/IQ4_NL bytes.
// Decoding uses the source-bound hybrid_quant helper from packed_matvec.metal;
// see LICENSE.ggml-hybrid. edge0_graph.metal and edge0_direct.metal stay
// byte-identical.
//
// One weight decode feeds TOKENS independent sums. Every token repeats
// edge0_graph_packed_single_impl: the 256-column stride, the SIMD-group sum
// and the ordered eight-group F32 sum, so the written bits match one-token
// dispatches. The format and the token count are compile-time constants.
// Parameters are rows, columns, TOKENS input element offsets and TOKENS
// output element offsets; token rows need not be contiguous.
template<uint kind, uint block, uint bytes, uint tokens, bool scaled>
inline void edge0_tokens_packed_impl(
    device const uchar *codes, device const float *input, device float *output,
    device const float *stored_s, constant uint *p, device atomic_uint *failure,
    uint row, uint tid, uint lane, uint sg, threadgroup float *partial) {
    const uint rows=p[0], columns=p[1];
    if (row>=rows) return;
    float sum[tokens];
    uint source[tokens];
    for (uint token=0;token<tokens;++token) { sum[token]=0.0f; source[token]=p[2+token]; }
    const ulong row_bytes=ulong(columns/block)*bytes;
    const ulong row_offset=ulong(row)*row_bytes;
    for (uint col=tid;col<columns;col+=256) {
        const float weight=hybrid_quant(codes+row_offset+ulong(col/block)*bytes,kind,col%block);
        for (uint token=0;token<tokens;++token) sum[token]+=weight*input[source[token]+col];
    }
    for (uint token=0;token<tokens;++token) {
        float total=simd_sum(sum[token]);
        if (lane==0) partial[token*8+sg]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) for (uint token=0;token<tokens;++token) {
        float result=0.0f;
        for (uint group=0;group<8;++group) result+=partial[token*8+group];
        if (!isfinite(result)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        if (scaled) {
            const float scale=stored_s[row];
            const float value=result/scale;
            if (!(scale>0) || !isfinite(scale) || !isfinite(value))
                atomic_store_explicit(failure,1u,memory_order_relaxed);
            result=value;
        }
        output[p[2+tokens+token]+row]=result;
    }
}

#define EDGE0_TOKENS_PACKED(NAME,KIND,BLOCK,BYTES,TOKENS) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], \
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]], \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partial[8*TOKENS]; \
    edge0_tokens_packed_impl<KIND,BLOCK,BYTES,TOKENS,false>( \
        codes,input,output,input,p,failure,row,tid,lane,sg,partial); \
}
#define EDGE0_TOKENS_SCALED(NAME,KIND,BLOCK,BYTES,TOKENS) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], \
    device const float *stored_s [[buffer(3)]], \
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(30)]], \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partial[8*TOKENS]; \
    edge0_tokens_packed_impl<KIND,BLOCK,BYTES,TOKENS,true>( \
        codes,input,output,stored_s,p,failure,row,tid,lane,sg,partial); \
}
#define EDGE0_TOKENS_COUNTS(MACRO,PREFIX,KIND,BLOCK,BYTES) \
MACRO(PREFIX##_2,KIND,BLOCK,BYTES,2) \
MACRO(PREFIX##_3,KIND,BLOCK,BYTES,3) \
MACRO(PREFIX##_4,KIND,BLOCK,BYTES,4) \
MACRO(PREFIX##_5,KIND,BLOCK,BYTES,5) \
MACRO(PREFIX##_6,KIND,BLOCK,BYTES,6) \
MACRO(PREFIX##_7,KIND,BLOCK,BYTES,7) \
MACRO(PREFIX##_8,KIND,BLOCK,BYTES,8)
EDGE0_TOKENS_COUNTS(EDGE0_TOKENS_PACKED,edge0_tokens_q8,8,32,34)
EDGE0_TOKENS_COUNTS(EDGE0_TOKENS_PACKED,edge0_tokens_q3,11,256,110)
EDGE0_TOKENS_COUNTS(EDGE0_TOKENS_PACKED,edge0_tokens_iq4,20,32,18)
EDGE0_TOKENS_COUNTS(EDGE0_TOKENS_SCALED,edge0_tokens_q8_scaled,8,32,34)
// Float matrices of the hyper-connection mixes and the router. Their one-token
// form repeats hybrid_packed_matvec, which has no format-constant Edge0 kernel.
#define EDGE0_TOKENS_FLOAT(PREFIX,KIND,BYTES) \
EDGE0_TOKENS_PACKED(PREFIX##_1,KIND,1,BYTES,1) \
EDGE0_TOKENS_COUNTS(EDGE0_TOKENS_PACKED,PREFIX,KIND,1,BYTES)
EDGE0_TOKENS_FLOAT(edge0_tokens_f32,0,4)
EDGE0_TOKENS_FLOAT(edge0_tokens_f16,1,2)
EDGE0_TOKENS_FLOAT(edge0_tokens_bf16,30,2)
#undef EDGE0_TOKENS_FLOAT
#undef EDGE0_TOKENS_PACKED
#undef EDGE0_TOKENS_SCALED

// Fused Q3_K gate/up rows for TOKENS inputs of one expert, repeating
// edge0_graph_gate_up_swiglu_single for every token.
template<uint tokens>
inline void edge0_tokens_gate_up_impl(
    device const uchar *gate, device const uchar *up,
    device const float *input, device float *output,
    constant uint *p, device atomic_uint *failure,
    uint row, uint tid, uint lane, uint sg,
    threadgroup float *partial_gate, threadgroup float *partial_up) {
    const uint rows=p[0], columns=p[1];
    if (row>=rows) return;
    float gate_sum[tokens], up_sum[tokens];
    uint source[tokens];
    for (uint token=0;token<tokens;++token) {
        gate_sum[token]=0.0f;
        up_sum[token]=0.0f;
        source[token]=p[2+token];
    }
    const ulong row_offset=ulong(row)*(columns/256)*110;
    for (uint col=tid;col<columns;col+=256) {
        const ulong offset=row_offset+ulong(col/256)*110;
        const float gw=hybrid_quant(gate+offset,11,col%256);
        const float uw=hybrid_quant(up+offset,11,col%256);
        for (uint token=0;token<tokens;++token) {
            const float value=input[source[token]+col];
            gate_sum[token]+=gw*value;
            up_sum[token]+=uw*value;
        }
    }
    for (uint token=0;token<tokens;++token) {
        float g=simd_sum(gate_sum[token]), u=simd_sum(up_sum[token]);
        if (lane==0) { partial_gate[token*8+sg]=g; partial_up[token*8+sg]=u; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) for (uint token=0;token<tokens;++token) {
        float gate_total=0.0f, up_total=0.0f;
        for (uint group=0;group<8;++group) {
            gate_total+=partial_gate[token*8+group];
            up_total+=partial_up[token*8+group];
        }
        float result=hybrid_ffn_activation(gate_total,up_total,0.0f);
        if (!isfinite(result)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        output[p[2+tokens+token]+row]=result;
    }
}

#define EDGE0_TOKENS_GATE_UP(NAME,KIND,BLOCK,BYTES,TOKENS) \
kernel void NAME( \
    device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]], \
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]], \
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(30)]], \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partial_gate[8*TOKENS], partial_up[8*TOKENS]; \
    edge0_tokens_gate_up_impl<TOKENS>( \
        gate,up,input,output,p,failure,row,tid,lane,sg,partial_gate,partial_up); \
}
EDGE0_TOKENS_COUNTS(EDGE0_TOKENS_GATE_UP,edge0_tokens_gate_up,11,256,110)
#undef EDGE0_TOKENS_GATE_UP
#undef EDGE0_TOKENS_COUNTS
