// SPDX-License-Identifier: Apache-2.0
// Edge0's opt-in F32 graph over unchanged GGML Q8_0/Q3_K/IQ4_NL bytes.
// Decoding uses the source-bound hybrid_quant helper from packed_matvec.metal;
// see LICENSE.ggml-hybrid. This file changes scheduling, not storage formats.
// Each matrix row decodes its weights once for up to eight token inputs.

kernel void edge0_graph_packed_batch(
    device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint rows=p[0], columns=p[1], kind=p[2], block=p[3], bytes=p[4], tokens=p[5];
    if (row >= rows) return;
    float sum[8];
    for (uint token=0;token<8;++token) sum[token]=0.0f;
    const ulong row_bytes=ulong(columns/block)*bytes;
    const ulong row_offset=ulong(row)*row_bytes;
    for (uint col=tid;col<columns;col+=256) {
        const float weight=hybrid_quant(codes+row_offset+ulong(col/block)*bytes,kind,col%block);
        for (uint token=0;token<tokens;++token)
            sum[token]+=weight*input[ulong(token)*columns+col];
    }
    threadgroup float partial[64];
    for (uint token=0;token<tokens;++token) {
        float total=simd_sum(sum[token]);
        if (lane==0) partial[token*8+sg]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) for (uint token=0;token<tokens;++token) {
        float total=0;
        for (uint group=0;group<8;++group) total+=partial[token*8+group];
        if (!isfinite(total)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        output[ulong(token)*rows+row]=total;
    }
}

// The two stored Q3_K rows consume the same selected group's activation.
// Keep both F32 reductions unchanged, then apply SiLU to gate and multiply up.
kernel void edge0_graph_gate_up_swiglu_batch(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint rows=p[0], columns=p[1], tokens=p[2];
    if (row>=rows) return;
    float gate_sum[8], up_sum[8];
    for (uint token=0;token<8;++token) { gate_sum[token]=0; up_sum[token]=0; }
    const ulong row_offset=ulong(row)*(columns/256)*110;
    for (uint col=tid;col<columns;col+=256) {
        const ulong offset=row_offset+ulong(col/256)*110;
        const float gw=hybrid_quant(gate+offset,11,col%256);
        const float uw=hybrid_quant(up+offset,11,col%256);
        for (uint token=0;token<tokens;++token) {
            const float value=input[ulong(token)*columns+col];
            gate_sum[token]+=gw*value;
            up_sum[token]+=uw*value;
        }
    }
    threadgroup float partial_gate[64], partial_up[64];
    for (uint token=0;token<tokens;++token) {
        float g=simd_sum(gate_sum[token]), u=simd_sum(up_sum[token]);
        if (lane==0) { partial_gate[token*8+sg]=g; partial_up[token*8+sg]=u; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) for (uint token=0;token<tokens;++token) {
        float g=0,u=0;
        for (uint group=0;group<8;++group) {
            g+=partial_gate[token*8+group];
            u+=partial_up[token*8+group];
        }
        float result=hybrid_ffn_activation(g,u,0.0f);
        if (!isfinite(result)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        output[ulong(token)*rows+row]=result;
    }
}

kernel void edge0_graph_column_scale(
    device const float *input [[buffer(0)]],
    device const float *stored_s [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=p[0]*p[1]) return;
    const float scale=stored_s[i%p[0]];
    const float value=input[i]/scale;
    if (!(scale>0) || !isfinite(scale) || !isfinite(value))
        atomic_store_explicit(failure,1u,memory_order_relaxed);
    output[i]=value;
}

kernel void edge0_graph_weighted_sum(
    device const float *experts [[buffer(0)]],
    device const float *probabilities [[buffer(1)]],
    device const float *shared [[buffer(2)]],
    device const float *shared_gate [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint *p [[buffer(5)]],
    device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    const uint width=p[0], selected=p[1], tokens=p[2];
    if (i>=width*tokens) return;
    const uint token=i/width, h=i%width;
    float sum=0;
    // U has already been applied separately to every original route slot.
    for (uint route=0;route<selected;++route)
        sum+=probabilities[token*selected+route]*experts[(ulong(token)*selected+route)*width+h];
    const float gate=hybrid_ffn_sigmoid(shared_gate[token]);
    sum+=gate*shared[i];
    if (!isfinite(sum)) atomic_store_explicit(failure,1u,memory_order_relaxed);
    output[i]=sum;
}
