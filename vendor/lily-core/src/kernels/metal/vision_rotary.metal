// Vision-only retained rotary application with measured host F32 coefficients.
// Original Lily Apache-2.0 provenance and notices remain in UPSTREAM.md/NOTICE.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

kernel void vision_qkv_rope_bf16_compat(
    device bfloat* qkv [[buffer(0)]],
    device const float* cosines [[buffer(1)]],
    device const float* sines [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x, row = gid.y;
    const uint half_h = H / 2, half_d = D / 2;
    if (j >= H) return;
    const uint which = j / half_h, within = j % half_h;
    const uint head = within / half_d, i = within % half_d;
    const ulong base = ulong(row) * 3 * H;
    const uint col0 = which * H + head * D + i, col1 = col0 + half_d;
    const float x0 = float(qkv[base + col0]), x1 = float(qkv[base + col1]);
    const float c = cosines[ulong(row) * half_d + i];
    const float s = sines[ulong(row) * half_d + i];
    qkv[base + col0] = bfloat(x0 * c - x1 * s);
    qkv[base + col1] = bfloat(x1 * c + x0 * s);
}

