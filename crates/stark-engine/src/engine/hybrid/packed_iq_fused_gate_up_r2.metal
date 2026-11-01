// SPDX-License-Identifier: Apache-2.0
// Optional two-row-per-SIMD variant of Stark's retained IQ2 gate/up fusion.
// Its normal dot schedule is adapted from ggml-org/llama.cpp b11429, commit
// d81235049384534c167caea52b85a694f6103d14,
// ggml/src/ggml-metal/kernels/mul_mv.metal.
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Function constant 71 MUST be 2 so the exact vector8 fallback covers the same
// four-row TG tile. The ordinary per-row arithmetic is unchanged.

// p=[rows,columns,experts,batched,clip_limit_bits]. Both effective code offsets
// must be even. Bounds comes from a fresh hybrid_graph_iq_input_bound dispatch.
// Gate/up diagnostics are retained to qualify all intermediate obligations.
kernel void hybrid_graph_iq2_fused_gate_up_r2(
    device const uchar *gate_codes [[buffer(0)]], device const uchar *up_codes [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]], device const uint *indices [[buffer(5)]],
    device const uint *bounds [[buffer(6)]], device float *gate [[buffer(7)]],
    device float *up [[buffer(8)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint tid=sg*32+lane;
    const uint expert=indices[grid.y];
    if (expert>=p[2]) {
        atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
        uint row=grid.x*4+tid;
        if (tid<4 && row<p[0]) {
            const uint at=grid.y*p[0]+row;
            gate[at]=0.0f;up[at]=0.0f;output[at]=0.0f;
        }
        return;
    }
    if (bounds[p[3]?grid.y:0]!=0) {
        // This is the unchanged vector8 chronological dot for each bank.
        // Helpers may return for tail SIMD groups; all lanes still join here.
        hybrid_iq_multirow_rows(gate_codes,input,gate,p,indices,failure,grid,lane,sg);
        hybrid_iq_multirow_rows(up_codes,input,up,p,indices,failure,grid,lane,sg);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint row=grid.x*4+tid;
        if (tid<4 && row<p[0]) {
            const uint at=grid.y*p[0]+row;
            const float g=gate[at],u=up[at];
            const float value=hybrid_ffn_activation(g,u,as_type<float>(p[4]));
            if (!isfinite(g)||!isfinite(u)||!isfinite(value))
                atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
            output[at]=value;
        }
        return;
    }
    threadgroup ulong table[512];
    threadgroup uchar signs[128];
    for (uint i=tid;i<512;i+=64) table[i]=iq2xs_grid[i];
    for (uint i=tid;i<128;i+=64) signs[i]=ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint first_row=grid.x*4+sg*2;
    const ulong row_bytes=ulong(p[1]>>8)*74;
    const ulong input_offset=p[3]?ulong(grid.y)*p[1]:0;
    float gate_sum[2]={0.0f,0.0f};
    float up_sum[2]={0.0f,0.0f};
    for (uint col=lane*32;col<p[1];col+=1024) {
        float values[32];
        #pragma clang loop unroll(full)
        for (uint j=0;j<32;++j) values[j]=input[input_offset+col+j];
        const uint part=(col&255)>>5;
        #pragma clang loop unroll(full)
        for (uint r=0;r<2;++r) {
            if (first_row+r>=p[0]) continue;
            const ulong at=(ulong(expert)*p[0]+first_row+r)*row_bytes+ulong(col>>8)*74;
            gate_sum[r]+=hybrid_iq2_fused_part(gate_codes+at,values,part,
                (threadgroup const uchar *)table,signs);
            up_sum[r]+=hybrid_iq2_fused_part(up_codes+at,values,part,
                (threadgroup const uchar *)table,signs);
        }
    }
    #pragma clang loop unroll(full)
    for (uint r=0;r<2;++r) {
        const float g=simd_sum(gate_sum[r])*0.25f;
        const float u=simd_sum(up_sum[r])*0.25f;
        if (lane==0 && first_row+r<p[0]) {
            const uint at=grid.y*p[0]+first_row+r;
            const float value=hybrid_ffn_activation(g,u,as_type<float>(p[4]));
            if (!isfinite(g)||!isfinite(u)||!isfinite(value))
                atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
            gate[at]=g;up[at]=u;output[at]=value;
        }
    }
}
