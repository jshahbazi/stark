// SPDX-License-Identifier: Apache-2.0
// Included with verify_undo.metal. The short packet has no KV wrap. All four
// convolution histories and both KV planes hold the sibling at the last
// physical row. Commit it at its logical depth and undo the rejected suffix.
kernel void ink_verify_commit_sibling(
    device const InkUndoVisit *visits [[buffer(0)]], device const uint4 *saved [[buffer(1)]],
    device const uint4 *anchors [[buffer(2)]], constant uint4 &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    device const InkUndoVisit &v = visits[group.y];
    const uint last = p.y - 1, depth = p.w;
    if (group.x < 2) {
        device uint4 *cache = group.x == 0 ? v.keys : v.values;
        ink_undo_transfer(cache + (p.x + last) * (Kv / Vector),
            cache + (p.x + depth) * (Kv / Vector), Kv, lane);
        return;
    }
    if (group.x >= 6) return;
    const uint plane = group.x - 2;
    device uint4 *histories[] = {v.k_history, v.v_history, v.a_history, v.e_history};
    const uint offsets[] = {0, Kv, 2 * Kv, 2 * Kv + Wide};
    const uint stride = (plane < 2 ? Kv : Wide) / Vector;
    const uint offset = offsets[plane] / Vector;
    device uint4 *history = histories[plane];
    device const uint4 *before = saved + group.y * (Saved / Vector) + 8 * Kv / Vector + 3 * offset;
    device const uint4 *anchor = anchors + group.y * (Conv / Vector) + offset;
    for (uint i = lane; i < stride; i += 128) {
        const uint4 sibling = history[((p.x + last) % 3) * stride + i];
        for (uint row = depth + 1; row <= last; ++row) {
            const uint slot = ((p.x + row) % 3) * stride + i;
            history[slot] = row == 3 ? anchor[i] : before[slot];
        }
        history[((p.x + depth) % 3) * stride + i] = sibling;
    }
}
