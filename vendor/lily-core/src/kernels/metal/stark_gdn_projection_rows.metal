// SPDX-License-Identifier: Apache-2.0
// Stark private candidate: exact short Q4 projection/convolution/scatter.
// quant.metal and stark_packed_rows.metal precede this source.
template <uint M>
static inline void stark_q4_gdn_conv_rows_body(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* input,
    device bfloat* qkv, device bfloat* z, device bfloat* a, device bfloat* b,
    device const bfloat* window_in, device bfloat* window_out,
    device const bfloat* conv_w, device bfloat* conv_out,
    device bfloat* capture,
    uint K, uint GS, uint C, uint V, uint H, uint KD, uint capture_enabled,
    uint group, uint lane) {
    constexpr uint LANES = 16;
    const uint sublane = lane % LANES;
    const uint channel = group * 2 + lane / LANES;
    float sums[M];
    stark_gemv_q4_packed_rows_sum<M>(
        codes, scales, biases, input, channel, sublane, K, GS, sums);
    if (sublane != 0) {
        return;
    }
    if (channel < C) {
        // These are exact BF16 window values represented as F32. Retaining
        // them between rows removes only repeated loads/stores.
        uint taps = KD - 1;
        float win[8];
        for (uint t = 0; t < taps; ++t) {
            win[t] = float(window_in[channel * taps + t]);
        }
        for (uint m = 0; m < M; ++m) {
            const bfloat projected = bfloat(sums[m]);
            qkv[(ulong)m * C + channel] = projected;
            if (capture_enabled != 0) {
                capture[(ulong)m * C + channel] = projected;
            }
            float acc = 0.0f;
            for (uint t = 0; t < taps; ++t) {
                acc += win[t] * float(conv_w[t * C + channel]);
            }
            const float xc = float(projected);
            acc += xc * float(conv_w[taps * C + channel]);
            conv_out[(ulong)m * C + channel] =
                bfloat(acc / (1.0f + exp(-acc)));
            for (uint t = 0; t + 1 < taps; ++t) {
                win[t] = win[t + 1];
            }
            win[taps - 1] = xc;
        }
        for (uint t = 0; t < taps; ++t) {
            window_out[channel * taps + t] = bfloat(win[t]);
        }
        return;
    }
    uint col = channel - C;
    if (col < V) {
        for (uint m = 0; m < M; ++m) {
            z[(ulong)m * V + col] = bfloat(sums[m]);
        }
        return;
    }
    col -= V;
    if (col < H) {
        for (uint m = 0; m < M; ++m) {
            a[(ulong)m * H + col] = bfloat(sums[m]);
        }
        return;
    }
    col -= H;
    for (uint m = 0; m < M; ++m) {
        b[(ulong)m * H + col] = bfloat(sums[m]);
    }
}

#define STARK_GDN_PROJECTION_ROWS(NAME, M) \
kernel void NAME( \
    device const uint* codes [[buffer(0)]], \
    device const bfloat* scales [[buffer(1)]], \
    device const bfloat* biases [[buffer(2)]], \
    device const bfloat* input [[buffer(3)]], \
    device bfloat* qkv [[buffer(4)]], \
    device bfloat* z [[buffer(5)]], \
    device bfloat* a [[buffer(6)]], \
    device bfloat* b [[buffer(7)]], \
    device const bfloat* window_in [[buffer(8)]], \
    device bfloat* window_out [[buffer(9)]], \
    device const bfloat* conv_w [[buffer(10)]], \
    device bfloat* conv_out [[buffer(11)]], \
    device bfloat* capture [[buffer(12)]], \
    constant uint& K [[buffer(13)]], \
    constant uint& GS [[buffer(14)]], \
    constant uint& C [[buffer(15)]], \
    constant uint& V [[buffer(16)]], \
    constant uint& H [[buffer(17)]], \
    constant uint& capture_enabled [[buffer(18)]], \
    constant uint& KD [[buffer(19)]], \
    uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) { \
    stark_q4_gdn_conv_rows_body<M>( \
        codes, scales, biases, input, qkv, z, a, b, window_in, window_out, \
        conv_w, conv_out, capture, K, GS, C, V, H, KD, capture_enabled, group, lane); \
}
STARK_GDN_PROJECTION_ROWS(stark_q4_gdn_conv_rows_m1, 1)
STARK_GDN_PROJECTION_ROWS(stark_q4_gdn_conv_rows_m2, 2)
STARK_GDN_PROJECTION_ROWS(stark_q4_gdn_conv_rows_m3, 3)
STARK_GDN_PROJECTION_ROWS(stark_q4_gdn_conv_rows_m4, 4)
#undef STARK_GDN_PROJECTION_ROWS
