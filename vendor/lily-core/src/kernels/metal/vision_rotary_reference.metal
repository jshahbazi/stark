// Test-only safe/uncontracted witness over the retained rotary body.
// The vision tower's own kernels (docs/architecture.md "The vision tower",
// tools/reference/VISION.md "The tower"): the position-embedding blend, the
// 2-D rotary over the fused qkv rows, and full bidirectional attention over
// every patch of one image. LayerNorm lives in norm.metal, the bias and GELU
// passes in elementwise.metal, the projections are the dense bf16 GEMM.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// Decodes a block-major patch row into its grid coordinates: row
// r = ((bh * (gw / 2) + bw) * 2 + ih) * 2 + iw holds the patch at grid
// position (2 bh + ih, 2 bw + iw). VISION.md "Patchify".
static inline uint2 vision_grid_coords(uint r, uint gw) {
    const uint blocks_w = gw / 2;
    const uint iw = r % 2;
    const uint ih = (r / 2) % 2;
    const uint bw = (r / 4) % blocks_w;
    const uint bh = r / (4 * blocks_w);
    return uint2(bh * 2 + ih, bw * 2 + iw);
}

// One axis of the align_corners bilinear resample of the `side` x `side`
// position table to an axis of length `n`: taps floor(src) and floor(src) + 1
// clamped into the table, weights 1 - |src - tap|. `src = i * (side - 1) /
// max(n - 1, 1)` in float32, the closed form of linspace(0, side-1, n)[i]
// that vision_utils._interpolation_axis_taps_weights uses.
static inline void vision_pos_axis_taps(uint i, uint n, uint side,
                                        thread uint2& taps, thread float2& w) {
    const float src = float(i) * float(side - 1) / float(max(n - 1u, 1u));
    const float f = floor(src);
    const int t0 = int(f);
    taps = uint2(uint(clamp(t0, 0, int(side) - 1)), uint(clamp(t0 + 1, 0, int(side) - 1)));
    const float d0 = abs(src - f);
    const float d1 = abs(src - f - 1.0f);
    w = float2(max(1.0f - d0, 0.0f), max(1.0f - d1, 0.0f));
}

// x[r, c] = bf16(x[r, c] + bf16(sum_k w_k * table[tap_k, c])) over the
// patch-embedding output x [N, H] (the GEMM with its bias epilogue): the
// resampled position embedding, blended in float32 from the bf16 table and
// cast to bf16 before the add, as the reference does
// (`pos_embeds.to(hidden_states.dtype)`). Taps and weights depend only on the
// patch's grid coordinates, so every thread of a row recomputes them; that is
// cheaper than a host-side table and needs no extra buffer. Grid (H, N).
kernel void vision_patch_pos_bf16(device bfloat*       x     [[buffer(0)]],
                                  device const bfloat* table [[buffer(1)]],
                                  constant uint&       H     [[buffer(2)]],
                                  constant uint&       gh    [[buffer(3)]],
                                  constant uint&       gw    [[buffer(4)]],
                                  constant uint&       side  [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= H) {
        return;
    }
    const uint2 rc = vision_grid_coords(r, gw);
    uint2 ht, wt;
    float2 hw, ww;
    vision_pos_axis_taps(rc.x, gh, side, ht, hw);
    vision_pos_axis_taps(rc.y, gw, side, wt, ww);
    float blend = 0.0f;
    blend += hw.x * ww.x * float(table[(ulong)(ht.x * side + wt.x) * H + c]);
    blend += hw.x * ww.y * float(table[(ulong)(ht.x * side + wt.y) * H + c]);
    blend += hw.y * ww.x * float(table[(ulong)(ht.y * side + wt.x) * H + c]);
    blend += hw.y * ww.y * float(table[(ulong)(ht.y * side + wt.y) * H + c]);
    const ulong idx = (ulong)r * H + c;
    x[idx] = bfloat(float(x[idx]) + float(bfloat(blend)));
}

// The 2-D rotary in place over the qkv Linear's output [N, 3H] (q | k | v,
// each `heads` heads of `D`). Each thread owns one rotate_half pair
// (i, i + D/2) of one head of one row: for q and k the rotation in float32
// with angle
//   rot[i] = (i < D/4 ? row : col) * inv_freq[i % (D/4)],
//   inv_freq[j] = theta^(-2 j / (D/2)),
// which is `emb = cat(rot, rot)` over the head applied with rotate_half
// pairing (i, i + D/2) (VISION.md "Rotary"); v is left alone. Grid
// (H, N). Precise sin/cos: angles reach a few hundred radians.
kernel void vision_qkv_rope_reference_bf16(device bfloat*       qkv   [[buffer(0)]],
                                 constant uint&       H     [[buffer(1)]],
                                 constant uint&       D     [[buffer(2)]],
                                 constant uint&       gw    [[buffer(3)]],
                                 constant float&      theta [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint r = gid.y;
    const uint half_h = H / 2;
    if (j >= 2 * half_h) {
        return;
    }
    const uint which = j / half_h;          // 0 q, 1 k
    const uint within = j % half_h;
    const uint half_d = D / 2;
    const uint head = within / half_d;
    const uint i = within % half_d;
    const uint col0 = which * H + head * D + i;
    const uint col1 = col0 + half_d;
    const ulong base = (ulong)r * 3 * H;
    const float x0 = float(qkv[base + col0]);
    const float x1 = float(qkv[base + col1]);
    const uint2 rc = vision_grid_coords(r, gw);
    const uint quarter = half_d / 2;
    const uint fi = i % quarter;
    const float pos = float(i < quarter ? rc.x : rc.y);
    const float inv_freq =
        1.0f / precise::pow(theta, float(2 * fi) / float(half_d));
    const float ang = pos * inv_freq;
    const float c = precise::cos(ang);
    const float s = precise::sin(ang);
    qkv[base + col0] = bfloat(x0 * c - x1 * s);
    qkv[base + col1] = bfloat(x1 * c + x0 * s);
}

kernel void vision_rotary_stages_reference(
    device const float* source_angles [[buffer(0)]],
    device float* stages [[buffer(1)]],
    device float* inverse_frequencies [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& gw [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, row = gid.y;
    const uint half_d = D / 2, quarter = half_d / 2;
    if (i >= half_d || row >= N) return;
    const uint2 rc = vision_grid_coords(row, gw);
    const uint fi = i % quarter;
    const float inverse = 1.0f / precise::pow(theta, float(2 * fi) / float(half_d));
    const float position = float(i < quarter ? rc.x : rc.y);
    const float angle = position * inverse;
    const ulong plane = ulong(N) * half_d;
    const ulong index = ulong(row) * half_d + i;
    if (row == 0 && i < quarter) inverse_frequencies[i] = inverse;
    stages[index] = angle;
    stages[plane + index] = precise::cos(angle);
    stages[2 * plane + index] = precise::sin(angle);
    stages[3 * plane + index] = precise::cos(source_angles[index]);
    stages[4 * plane + index] = precise::sin(source_angles[index]);
}

kernel void vision_rotary_apply_reference(
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

