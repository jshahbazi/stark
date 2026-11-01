// SPDX-License-Identifier: Apache-2.0
// Stark private candidate: share Q4 code loads across independent token dots.
// Requires quant.metal and preserves gemv_q4_packed_sum<2>'s arithmetic order.
template <uint M>
static inline void stark_gemv_q4_packed_rows_sum(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* input,
    uint channel, uint sublane, uint K, uint GS,
    thread float (&sums)[M]) {
    constexpr uint LANES = 16;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    device const uint4* wrow =
        (device const uint4*)(codes + (ulong)channel * words);
    _Pragma("clang loop unroll(full)")
    for (uint m = 0; m < M; ++m) {
        sums[m] = 0.0f;
    }
    for (uint i = sublane; i < blocks; i += LANES) {
        const uint g = i / bpg;
        const float s = float(scales[channel * groups + g]);
        const float b = float(biases[channel * groups + g]);
        const uint4 w4 = wrow[i];
        _Pragma("clang loop unroll(full)")
        for (uint m = 0; m < M; ++m) {
            device const bfloat4* xv =
                (device const bfloat4*)(input + (ulong)m * K);
            sums[m] += dot_word_q4(w4.x, s, b, xv, 4 * i)
                + dot_word_q4(w4.y, s, b, xv, 4 * i + 1)
                + dot_word_q4(w4.z, s, b, xv, 4 * i + 2)
                + dot_word_q4(w4.w, s, b, xv, 4 * i + 3);
        }
    }
    for (uint off = LANES / 2; off > 0; off >>= 1) {
        _Pragma("clang loop unroll(full)")
        for (uint m = 0; m < M; ++m) {
            sums[m] += simd_shuffle_down(sums[m], off);
        }
    }
}
