// SPDX-License-Identifier: Apache-2.0
// Shared by every query and relative-head normalization (serial, packed
// prefill and verifier) so that all paths round the long-position gain the
// same way. Prepended to each translation unit that uses it.
#ifndef INK_LONG_POSITION_GAIN
#define INK_LONG_POSITION_GAIN
#include <metal_stdlib>
using namespace metal;

// Global-attention query and relative-head gain for positions >= 8192:
// 1 + 0.1 * log((position + 1) / 8192), rounded once. The explicit fma keeps
// the result independent of the caller's floating-point contraction pragmas;
// the serial reference measured as this single-rounding form.
inline float ink_long_position_gain(uint position) {
    return fma(0.1f, log(float(position + 1) / 8192.0f), 1.0f);
}
#endif
