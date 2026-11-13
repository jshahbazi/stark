// Stark-authored, CPU-only fixture bridge; no model/tensor-file I/O.
// Quantization/dequantization authority: GGML at llama.cpp revision
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-quants.c.
// Upstream copyright/license remains in LICENSE.ggml-hybrid.
// Matvec, nonlinearities, routing and reductions are independently evaluated
// in F64 by scripts/edge0_native_fixture.py, never by a Rust implementation.
#include "ggml-quants.h"
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" size_t edge0_forward_row_bytes(int type, size_t n) {
    if (type == 8 && n % 32 == 0) return n / 32 * sizeof(block_q8_0);
    if (type == 11 && n % 256 == 0) return n / 256 * sizeof(block_q3_K);
    if (type == 20 && n % 32 == 0) return n / 32 * sizeof(block_iq4_nl);
    return 0;
}

extern "C" int edge0_forward_pack(int type, const float* x, size_t n, void* out) {
    if (!x || !out || !edge0_forward_row_bytes(type, n)) return 1;
    if (type == 8) quantize_row_q8_0_ref(x, (block_q8_0*)out, n);
    else if (type == 11) quantize_row_q3_K_ref(x, (block_q3_K*)out, n);
    else if (type == 20) quantize_row_iq4_nl_ref(x, (block_iq4_nl*)out, n);
    return 0;
}

extern "C" int edge0_forward_unpack(int type, const void* x, size_t n, float* out) {
    if (!x || !out || !edge0_forward_row_bytes(type, n)) return 1;
    if (type == 8) dequantize_row_q8_0((const block_q8_0*)x, out, n);
    else if (type == 11) dequantize_row_q3_K((const block_q3_K*)x, out, n);
    else if (type == 20) dequantize_row_iq4_nl((const block_iq4_nl*)x, out, n);
    return 0;
}

// Pinned GGML CPU activation controls. Type 15 is Q8_K, consumed by
// Q3_K experts; type 8 is Q8_0, consumed by IQ4_NL experts and Q8_0 rows.
extern "C" int edge0_forward_activation(int type, const float* x, size_t n, float* out) {
    if (!x || !out) return 1;
    if (type == 8 && n % 32 == 0) {
        std::vector<block_q8_0> q(n / 32);
        quantize_row_q8_0_ref(x, q.data(), n);
        dequantize_row_q8_0(q.data(), out, n);
    } else if (type == 15 && n % 256 == 0) {
        std::vector<block_q8_K> q(n / 256);
        quantize_row_q8_K_ref(x, q.data(), n);
        dequantize_row_q8_K(q.data(), out, n);
    } else return 2;
    return 0;
}
