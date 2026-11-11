// Vision-only measured safe/uncontracted retained position body.
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
kernel void vision_patch_pos_bf16_compat(device bfloat*       x     [[buffer(0)]],
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

