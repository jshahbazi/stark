// SPDX-License-Identifier: Apache-2.0
// Independent original-plane Q2 implementation, described from Edge0-AI/Edge0
// b7900cbb322926ccebdcd0c0c5a1280497ecff22 tools/mtp_rt.py::blob_of and
// include/strata/kernels/cpu/expert.hpp. See LICENSE.strata-reference.
// Four adjacent columns occupy each byte's bits 0/2/4/6. QK=64, d*(q-1).
// This is not GGML Q2_0: scales occupy separate original-record planes.

inline float edge0_mtp_q2_scale(device const uchar *p) {
    // Byte loads preserve correctness for unaligned raw transport subspans.
    const ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

inline float edge0_mtp_q2_value(device const uchar *codes, float scale, uint column) {
    return scale * (float((codes[column / 4] >> (2 * (column % 4))) & 3) - 1.0f);
}

kernel void edge0_mtp_q2_gate_up(
    device const uchar *record [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint hidden=p[0], intermediate=p[1], tokens=p[2], scale_plane=p[4];
    if (row>=intermediate) return;
    const uint blocks=hidden/64;
    const uint gate_row=2*row, up_row=gate_row+1;
    float g[8], u[8];
    for (uint token=0;token<8;++token) { g[token]=0.0f; u[token]=0.0f; }
    for (uint col=tid;col<hidden;col+=256) {
        const uint block=col/64;
        const float gd=edge0_mtp_q2_scale(record+scale_plane+(gate_row*blocks+block)*2);
        const float ud=edge0_mtp_q2_scale(record+scale_plane+(up_row*blocks+block)*2);
        const float gw=edge0_mtp_q2_value(record+(gate_row*blocks+block)*16,gd,col%64);
        const float uw=edge0_mtp_q2_value(record+(up_row*blocks+block)*16,ud,col%64);
        if (!isfinite(gd) || !isfinite(ud) || !isfinite(gw) || !isfinite(uw))
            atomic_store_explicit(failure,1u,memory_order_relaxed);
        for (uint token=0;token<tokens;++token) {
            const float x=input[token*hidden+col];
            const float gp=gw*x, up=uw*x;
            g[token]+=gp; u[token]+=up;
            if (!isfinite(x) || !isfinite(gp) || !isfinite(up) || !isfinite(g[token]) || !isfinite(u[token]))
                atomic_store_explicit(failure,1u,memory_order_relaxed);
        }
    }
    threadgroup float partial_gate[64], partial_up[64];
    for (uint token=0;token<tokens;++token) {
        const float gs=simd_sum(g[token]), us=simd_sum(u[token]);
        if (!isfinite(gs) || !isfinite(us)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        if (lane==0) { partial_gate[token*8+sg]=gs; partial_up[token*8+sg]=us; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) for (uint token=0;token<tokens;++token) {
        float gs=0.0f, us=0.0f;
        for (uint group=0;group<8;++group) {
            gs+=partial_gate[token*8+group]; us+=partial_up[token*8+group];
            if (!isfinite(gs) || !isfinite(us)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        }
        const float result=hybrid_ffn_activation(gs,us,0.0f);
        if (!isfinite(result)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        output[token*intermediate+row]=result;
    }
}

kernel void edge0_mtp_q2_down(
    device const uchar *record [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint hidden=p[0], intermediate=p[1], tokens=p[2], code_plane=p[3], scale_plane=p[5];
    if (row>=hidden) return;
    const uint blocks=intermediate/64;
    float sum[8];
    for (uint token=0;token<8;++token) sum[token]=0.0f;
    for (uint col=tid;col<intermediate;col+=256) {
        const uint block=row*blocks+col/64;
        const float d=edge0_mtp_q2_scale(record+scale_plane+block*2);
        const float weight=edge0_mtp_q2_value(record+code_plane+block*16,d,col%64);
        if (!isfinite(d) || !isfinite(weight)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        for (uint token=0;token<tokens;++token) {
            const float x=input[token*intermediate+col], product=weight*x;
            sum[token]+=product;
            if (!isfinite(x) || !isfinite(product) || !isfinite(sum[token]))
                atomic_store_explicit(failure,1u,memory_order_relaxed);
        }
    }
    threadgroup float partial[64];
    for (uint token=0;token<tokens;++token) {
        const float total=simd_sum(sum[token]);
        if (!isfinite(total)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        if (lane==0) partial[token*8+sg]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) for (uint token=0;token<tokens;++token) {
        float total=0.0f;
        for (uint group=0;group<8;++group) {
            total+=partial[token*8+group];
            if (!isfinite(total)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        }
        output[token*hidden+row]=total;
    }
}
