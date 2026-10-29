// Test-only original Lily gated-step wrapper captured before Stark changes.
// Source: e22d58fa7cc488493fb1b5d810a198bd29730aeb, src/kernels/metal/gdn.metal.
// Apache-2.0; see LICENSE, NOTICE and UPSTREAM.md.
// Only the instantiated kernel name changes. Recurrence helper comes from gdn.metal.
#define GDN_STEP_GATED_WRAPPER(NAME, STATE_T)                                 \
kernel void NAME(device const bfloat* q       [[buffer(0)]],                  \
                 device const bfloat* k       [[buffer(1)]],                  \
                 device const bfloat* v       [[buffer(2)]],                  \
                 device const bfloat* a       [[buffer(3)]],                  \
                 device const bfloat* b       [[buffer(4)]],                  \
                 device const float*  a_log   [[buffer(5)]],                  \
                 device const bfloat* dt_bias [[buffer(6)]],                  \
                 device STATE_T*      state   [[buffer(7)]],                  \
                 device const bfloat* z       [[buffer(8)]],                  \
                 device const float*  norm_w  [[buffer(9)]],                  \
                 device bfloat*       out     [[buffer(10)]],                 \
                 constant float&      scale   [[buffer(11)]],                 \
                 constant uint&       vpk     [[buffer(12)]],                 \
                 constant float&      eps     [[buffer(13)]],                 \
                 constant uint&       gate_act [[buffer(14)]],                \
                 uint h    [[threadgroup_position_in_grid]],                  \
                 uint tid  [[thread_index_in_threadgroup]],                   \
                 uint sg   [[simdgroup_index_in_threadgroup]],                \
                 uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup float q_norm[DIM];                                            \
    threadgroup float k_norm[DIM];                                            \
    threadgroup float part_q[DIM / 32];                                       \
    threadgroup float part_k[DIM / 32];                                       \
    threadgroup float gates[4];                                               \
    threadgroup bfloat raw[DIM];                                              \
    threadgroup float norm_part[DIM / 32];                                    \
    threadgroup float inv_rms = 0.0f;                                         \
    const float o = gdn_step_body(q, k, v, a, b, a_log, dt_bias, state,      \
                                  scale, vpk, h, tid, sg, lane, q_norm,       \
                                  k_norm, part_q, part_k, gates);             \
    raw[tid] = bfloat(o);                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const float rv = float(raw[tid]);                                         \
    const float ss = simd_sum(rv * rv);                                       \
    if (lane == 0) {                                                          \
        norm_part[sg] = ss;                                                   \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    if (tid == 0) {                                                           \
        float total = 0.0f;                                                   \
        for (uint i = 0; i < DIM / 32; ++i) {                                \
            total += norm_part[i];                                            \
        }                                                                     \
        inv_rms = rsqrt(total / float(DIM) + eps);                            \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const float gate = float(z[h * DIM + tid]);                               \
    const float act = gate_activation(gate, gate_act);                        \
    out[h * DIM + tid] = bfloat(norm_w[tid] * rv * inv_rms * act);            \
}

GDN_STEP_GATED_WRAPPER(stark_gdn_step_gated_reference, float)
#undef GDN_STEP_GATED_WRAPPER
