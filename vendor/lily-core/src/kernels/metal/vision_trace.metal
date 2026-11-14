// Test-only observation wrapper over the production attention arithmetic.
// This source is appended only by cfg(test); no production selector exists.
#if __METAL_VERSION__ >= 400
kernel void vision_attn_trace_nax_d72(
    device bfloat* qkv [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    device float* stages [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]) {
    threadgroup float s_tile[32 * 64];
    threadgroup bfloat p_tile[32 * 64];
    threadgroup float row_max[32], row_sum[32];
    vision_attn_body<72, 32, 64, 4, true>(
        qkv, out, N, H, scale, s_tile, p_tile, row_max, row_sum, stages,
        tg, tid, tg_size);
}
#endif
