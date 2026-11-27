// SPDX-License-Identifier: MIT

// Attention output RMSNorm, HC residual update and FFN flat RMSNorm for the 3072-wide STQ trunk.
// Each thread accumulates the same positions in the same order as the two standalone RMSNorms.
// The residual is retained for the FFN mix and its post gate. The weighted attention output is transient.
kernel void stark_hy4_stq_attn_hc(device const float* attn [[buffer(0)]],
                                device const float* weight [[buffer(1)]],
                                device const float* residual [[buffer(2)]],
                                device const float* post [[buffer(3)]],
                                device float* middle [[buffer(4)]],
                                device float* flat [[buffer(5)]],
                                constant float& eps [[buffer(6)]],
                                uint row [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint width = 3072;
    constexpr uint flat_width = HY4_FLASH_HC * width;
    constexpr uint columns = width / HY4_FLASH_TG;
    device const float* ar = attn + ulong(row) * width;
    device const float* rr = residual + ulong(row) * flat_width;
    device const float* pr = post + ulong(row) * HY4_FLASH_HC;
    device float* mid = middle + ulong(row) * flat_width;
    device float* fr = flat + ulong(row) * flat_width;

    float y[columns];
    float ss = 0.0f;
    #pragma unroll
    for (uint i = 0; i < columns; ++i) {
        const uint d = tid + i * HY4_FLASH_TG;
        y[i] = ar[d];
        ss += y[i] * y[i];
    }
    threadgroup float attn_part[HY4_FLASH_TG_SG];
    const float scale = precise::rsqrt(hy4_flash_tg_sum(ss, attn_part, sg, lane) / float(width) + eps);
    #pragma unroll
    for (uint i = 0; i < columns; ++i)
        y[i] = (y[i] * scale) * weight[tid + i * HY4_FLASH_TG];

    ss = 0.0f;
    #pragma unroll
    for (uint stream = 0; stream < HY4_FLASH_HC; ++stream) {
        const float gate = pr[stream];
        #pragma unroll
        for (uint i = 0; i < columns; ++i) {
            const uint d = stream * width + tid + i * HY4_FLASH_TG;
            const float value = rr[d] + y[i] * gate;
            mid[d] = value;
            ss += value * value;
        }
    }
    threadgroup float flat_part[HY4_FLASH_TG_SG];
    const float flat_scale = precise::rsqrt(hy4_flash_tg_sum(ss, flat_part, sg, lane) / float(flat_width) + eps);
    for (uint d = tid; d < flat_width; d += HY4_FLASH_TG)
        fr[d] = mid[d] * flat_scale;
}
