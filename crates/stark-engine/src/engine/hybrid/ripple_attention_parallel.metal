// SPDX-License-Identifier: Apache-2.0
// Independent Ripple attention components. F32 state and arithmetic;
// SIMD/K partition reductions change association; host selection is bounded.
// Envelope guards preserve the original serial overflow contract for extremes.

kernel void ripple_graph_attention_scores_parallel(device const float *queries [[buffer(0)]],
    device const float *keys [[buffer(1)]], device float *scores [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint count=p[0],heads=p[1],kv_heads=p[2],dim=p[3],capacity=p[4],start=p[5];
    if (!count || count>capacity || count>4096 || start>=capacity || !kv_heads || heads%kv_heads || dim!=128) {
        if (lane==0) atomic_store_explicit(failure,1u,memory_order_relaxed);
        return;
    }
    uint head=task/count,token=task%count;
    if (head>=heads) return;
    uint slot=start+token;
    if (slot>=capacity) slot-=capacity;
    uint key_base=slot*kv_heads*dim+head/(heads/kv_heads)*dim;
    uint d=lane*4;
    // Scalar constructors preserve valid four-byte-aligned buffer subviews.
    float4 q=float4(queries[head*dim+d],queries[head*dim+d+1],queries[head*dim+d+2],queries[head*dim+d+3]);
    float4 k=float4(keys[key_base+d],keys[key_base+d+1],keys[key_base+d+2],keys[key_base+d+3]);
    bool nonfinite=any(!isfinite(q)) || any(!isfinite(k));
    bool extreme=any(abs(q)>float4(1e15f)) || any(abs(k)>float4(1e15f));
    if (simd_any(nonfinite)) atomic_store_explicit(failure,1u,memory_order_relaxed);
    if (simd_any(extreme || nonfinite)) {
        if (lane==0) {
            float score=0.0f;
            for (uint j=0;j<dim;++j) score+=queries[head*dim+j]*keys[key_base+j];
            score*=1.0f/sqrt(float(dim));
            scores[head*capacity+token]=score;
            if (!isfinite(score)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        }
        return;
    }
    float sum=0.0f;
    sum+=q.x*k.x;sum+=q.y*k.y;sum+=q.z*k.z;sum+=q.w*k.w;
    float score=simd_sum(sum)*(1.0f/sqrt(float(dim)));
    if (lane==0) {
        scores[head*capacity+token]=score;
        if (!isfinite(score)) atomic_store_explicit(failure,1u,memory_order_relaxed);
    }
}

kernel void ripple_graph_attention_output_partial(device const float *probabilities [[buffer(0)]],
    device const float *values [[buffer(1)]], device float *partials [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[thread_position_in_grid]]) {
    uint count=p[0],heads=p[1],kv_heads=p[2],dim=p[3],capacity=p[4],start=p[5],tile=p[7];
    if (!count || count>capacity || start>=capacity || count>4096 || !kv_heads || heads%kv_heads || dim!=128 || (tile!=64 && tile!=128)) {
        atomic_store_explicit(failure,1u,memory_order_relaxed);return;
    }
    uint splits=(count+tile-1)/tile;
    uint d=task%dim,split=(task/dim)%splits,head=(task/dim)/splits;
    if (head>=heads) return;
    uint kv_base=head/(heads/kv_heads)*dim+d;
    float value=0.0f;
    bool fallback=false;
    uint end=min(count,(split+1)*tile);
    for (uint token=split*tile;token<end;++token) {
        uint slot=start+token;if (slot>=capacity) slot-=capacity;
        float probability=probabilities[head*capacity+token];
        float v=values[slot*kv_heads*dim+kv_base];
        bool finite=isfinite(probability) && isfinite(v);
        if (!finite) atomic_store_explicit(failure,1u,memory_order_relaxed);
        bool eligible=finite && abs(probability)<=1e15f && abs(v)<=1e15f;
        fallback=fallback || !eligible;
        if (!fallback) value+=probability*v;
    }
    // <=4096 products bounded by 1e30 imply every original/partial absolute
    // sum <=4.096e33. For extremes, the reducer must evaluate the original dot:
    // a partition result alone could conceal or invent serial overflow.
    partials[task]=fallback ? NAN : value;
}

kernel void ripple_graph_attention_output_reduce(device const float *partials [[buffer(0)]],
    device const float *probabilities [[buffer(1)]], device const float *values [[buffer(2)]],
    device const float *gates [[buffer(3)]], device float *output [[buffer(4)]],
    constant uint *p [[buffer(5)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[thread_position_in_grid]]) {
    uint count=p[0],heads=p[1],kv_heads=p[2],dim=p[3],capacity=p[4],start=p[5],tile=p[7];
    if (!count || count>capacity || start>=capacity || count>4096 || !kv_heads || heads%kv_heads || dim!=128 || (tile!=64 && tile!=128)) {
        atomic_store_explicit(failure,1u,memory_order_relaxed);return;
    }
    uint head=task/dim,d=task%dim;
    if (head>=heads) return;
    uint splits=(count+tile-1)/tile;
    float value=0.0f;
    bool fallback=false;
    for (uint split=0;split<splits;++split) {
        float partial=partials[(head*splits+split)*dim+d];
        if (!isfinite(partial)) fallback=true;
        else if (!fallback) value+=partial;
    }
    if (fallback) {
        value=0.0f;
        uint kv_base=head/(heads/kv_heads)*dim+d;
        for (uint token=0;token<count;++token) {
            uint slot=start+token;if (slot>=capacity) slot-=capacity;
            value+=probabilities[head*capacity+token]*values[slot*kv_heads*dim+kv_base];
        }
    }
    if (p[6]) {
        float gate=gates[task];
        if (!isfinite(gate)) atomic_store_explicit(failure,1u,memory_order_relaxed);
        value*=ripple_graph_sigmoid(gate);
    }
    output[task]=value;
    if (!isfinite(value)) atomic_store_explicit(failure,1u,memory_order_relaxed);
}
