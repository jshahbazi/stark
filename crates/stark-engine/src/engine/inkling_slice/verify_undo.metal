// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

constexpr constant uint Kv = 1024 * 2;
constexpr constant uint Wide = 2304 * 2;
constexpr constant uint Conv = 2 * Kv + 2 * Wide;
constexpr constant uint Saved = 8 * Kv + 3 * Conv;
constexpr constant uint Vector = sizeof(uint4);

inline void ink_undo_transfer(device const uint4 *from, device uint4 *to, uint bytes, uint lane) {
    for (uint i = lane; i < bytes / Vector; i += 128) to[i] = from[i];
}

inline void ink_undo_copy(device uint4 *keys, device uint4 *values,
    device uint4 *k_history, device uint4 *v_history, device uint4 *a_history, device uint4 *e_history,
    device uint4 *saved, device const uint4 *anchors, uint4 p, uint segment, uint lane) {
    const uint position = p.x, rows = p.y, capacity = p.z, kept = p.w;
    if (segment < 8) {
        const uint row = segment % 4;
        const uint pos = position + row;
        if (row >= rows || row < kept || pos < capacity) return;
        device uint4 *cache = (segment < 4 ? keys : values) + (pos % capacity) * (Kv / Vector);
        device uint4 *before = saved + segment * (Kv / Vector);
        if (kept == 0) ink_undo_transfer(cache, before, Kv, lane);
        else ink_undo_transfer(before, cache, Kv, lane);
        return;
    }

    const uint channel = segment - 8;
    device uint4 *histories[] = {k_history, v_history, a_history, e_history};
    const uint offsets[] = {0, Kv, Kv * 2, Kv * 2 + Wide};
    const uint stride = channel < 2 ? Kv : Wide;
    const uint offset = offsets[channel];
    device uint4 *history = histories[channel];
    device uint4 *before = saved + (8 * Kv + offset * 3) / Vector;
    if (kept == 0) {
        ink_undo_transfer(history, before, stride * 3, lane);
    } else {
        for (uint row = kept; row < rows; ++row) {
            const uint slot = ((position + row) % 3) * (stride / Vector);
            device const uint4 *source = row == 3 ? anchors + offset / Vector : before + slot;
            ink_undo_transfer(source, history + slot, stride, lane);
        }
    }
}

struct InkUndoVisit {
    device uint4 *keys;
    device uint4 *values;
    device uint4 *k_history;
    device uint4 *v_history;
    device uint4 *a_history;
    device uint4 *e_history;
    ulong capacity;
};
kernel void ink_verify_undo_all(
    device const InkUndoVisit *visits [[buffer(0)]], device uint4 *saved [[buffer(1)]],
    device const uint4 *anchors [[buffer(2)]], constant uint4 &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    device const InkUndoVisit &v = visits[group.y];
    ink_undo_copy(v.keys, v.values, v.k_history, v.v_history, v.a_history, v.e_history,
        saved + group.y * (Saved / Vector), anchors + group.y * (Conv / Vector),
        uint4(p.x, p.y, uint(v.capacity), p.w), group.x, lane);
}
