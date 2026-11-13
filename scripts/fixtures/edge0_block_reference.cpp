// Stark-authored CPU-only block reference bridge. No model or inference API.
// GGML authority: ggml-org/llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-quants.c (Q4_0, IQ4_NL, Q2_0); MIT license in LICENSE.ggml-hybrid.
// Strata Q2_0's independent source definition is tools/mtp_pack.py::dequant at
// Edge0-AI/Edge0 b7900cbb322926ccebdcd0c0c5a1280497ecff22 (MIT Strata reference).
// FP8 authority is that revision's src/kernels/ple_fp8_parity.cpp -> PleTable ->
// src/kernels/ngram.cpp::fp8_e4m3_dequant_row, including its signed-zero NaN-byte
// policy. Link an exact, attributed CPU subset of that definition; do not
// substitute a generic E4M3 type or a guessed GGML enum.
//
// Build against the already pinned CPU-only GGML build, with the separately
// recorded/hashed Strata FP8 reference subset (no CUDA, model or table I/O):
// clang++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off -dynamiclib THIS.cpp
//   PINNED_FP8_SUBSET.cpp -I SOURCE/ggml/include -I SOURCE/ggml/src
//   -L CPU_BUILD/bin -lggml-base -Wl,-rpath,ABS_CPU_BUILD/bin -o REFERENCE.dylib
// Keep synthetic fixtures and raw F32 captures ignored. Publish hashes/counts.

#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" void edge0_pinned_fp8(const uint8_t * row, float scale, float * output);

extern "C" int edge0_reference_block(int kind, const uint8_t * input, size_t bytes,
                                      float * output, size_t output_count) {
    if (!input || !output || bytes != 18) return 1;
    if (kind == 2 && output_count == 32) {
        block_q4_0 block;
        static_assert(sizeof block == 18);
        std::memcpy(&block, input, sizeof block);
        dequantize_row_q4_0(&block, output, 32);
    } else if (kind == 20 && output_count == 32) {
        block_iq4_nl block;
        static_assert(sizeof block == 18);
        std::memcpy(&block, input, sizeof block);
        dequantize_row_iq4_nl(&block, output, 32);
    } else if (kind == 42 && output_count == 64) {
        block_q2_0 block;
        static_assert(sizeof block == 18);
        std::memcpy(&block, input, sizeof block);
        dequantize_row_q2_0(&block, output, 64);
    } else {
        return 2;
    }
    return 0;
}

extern "C" int edge0_reference_fp8_row(const uint8_t * input, size_t bytes, float scale,
                                        float * output, size_t output_count) {
    if (!input || !output || bytes != 160 || output_count != 160) return 1;
    edge0_pinned_fp8(input, scale, output);
    return 0;
}
