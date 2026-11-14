// Test-only observer over the production source-order attention body.
kernel void vision_attention_source_trace_bf16(
    device const bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]],
    device float* stages [[buffer(2)]], constant uint& N [[buffer(3)]],
    constant float& scale [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    source_attention_body<true>(qkv, out, stages, N, scale, group, sg, lane);
}
