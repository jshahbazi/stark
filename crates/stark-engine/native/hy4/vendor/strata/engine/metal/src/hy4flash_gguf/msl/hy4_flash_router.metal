// engine/metal/src/hy4flash_gguf/msl/hy4_flash_router.metal - the MoE router of a hy4-flash trunk block (semantics.md §2).
//
//   probs  = sigmoid(ffn_gate_inp . cur)                 the FULL-WIDTH 3072 cur, 512 logits
//   biased = probs + exp_probs_b                         what the SELECTION ranks
//   sel    = the top `topk` (16) of biased, ties to the SMALLER expert id
//   w[r]   = probs[sel[r]] / max(sum_r probs[sel[r]], 2^-14) * 3.998
//
// Two things make this its own kernel rather than a reuse of moe.metal's moe_route, for the same reason the MoE
// GEMVs are:
//
//   * moe_route is softmax over the logits and top TEN by probability, both written as compile-time facts
//     (`route_tok<NL>`, `rank < 10`).  hy4-flash is sigmoid, top SIXTEEN, and the rank key is the BIASED probability
//     while the weight is the UNBIASED one - which of the two feeds the weight is not a detail (semantics.md §2);
//   * the sum of the selected probabilities is taken IN SELECTION ORDER here (rank 0 then 1 then ...).  moe_route
//     accumulates it on whichever lane won and then folds the lanes with an xor tree, which is a different order.
//     hy4-flash needs one order that holds for every window size.
//
// One thread group per token, so nothing depends on how many tokens the dispatch carries: the sigmoid pass is
// spread over HY4_FLASH_TG threads, and simdgroup 0 alone runs the rank loop and the weight tail.
//
// The floor is on the SUM, not on the weights (hy4_block.py latent_moe: w / maximum(w.sum(), 6.103515625e-5)).

#define HY4_FLASH_ROUTER_MAXE 1024   // experts a token can be routed over (512 here); one threadgroup array each
#define HY4_FLASH_ROUTER_MAXK 64     // top-k slots (16 here)

struct Hy4FlashRouterArgs {
    uint n_expert;
    uint topk;
    uint l_stride;     // floats between tokens of the logits
    uint id_stride;    // ints between tokens of ids
    uint w_stride;     // floats between tokens of w
    uint n_tok;
    float w_scale;     // expert_weights_scale (3.998)
    float floor;       // 2^-14 = 6.103515625e-5
};

// Test hook (the convention of gemv.metal's GEMV_PROBE): 1 = ties go to the LARGER expert id.  The tie test's
// negative control - with it the "all scores equal -> 0..topk-1" case must fail.  Unset in production.
constant int HY4_FLASH_ROUTER_PROBE [[function_constant(50)]];
constant int hy4_flash_router_probe = is_function_constant_defined(HY4_FLASH_ROUTER_PROBE) ? HY4_FLASH_ROUTER_PROBE : 0;

kernel void hy4_flash_router(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device int* ids [[buffer(2)]], device float* w [[buffer(3)]],
                       constant Hy4FlashRouterArgs& a [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float tb[HY4_FLASH_ROUTER_MAXE];   // probs + bias: the ranking key, overwritten with -inf once taken
    threadgroup float tp[HY4_FLASH_ROUTER_MAXE];   // the unbiased probs: the weights' source
    threadgroup float sp[HY4_FLASH_ROUTER_MAXK];   // the selected probability, by rank
    threadgroup int si[HY4_FLASH_ROUTER_MAXK];     // the selected expert id, by rank
    device const float* lr = logits + (ulong)tgp.x * a.l_stride;
    for (uint e = tid; e < a.n_expert; e += HY4_FLASH_TG) {
        const float p = hy4_flash_sigmoid(lr[e]);
        tp[e] = p;
        tb[e] = p + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    const bool hi = hy4_flash_router_probe == 1;
    const int none = hi ? -1 : 0x7FFFFFFF;   // the id a lane with no candidate carries, so a tie never picks it
    for (uint r = 0; r < a.topk; ++r) {
        float best = -INFINITY;
        int bi = none;
        // ascending e, strict >: within a lane's stride the FIRST of equal values wins, i.e. the smaller id
        for (uint e = lane; e < a.n_expert; e += 32) {
            const float v = tb[e];
            if (v > best || (v == best && hi)) {
                best = v;
                bi = int(e);
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ob = simd_shuffle_xor(best, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (ob > best || (ob == best && (hi ? oi > bi : oi < bi))) {
                best = ob;
                bi = oi;
            }
        }
        // A row whose scores never supply a candidate must never index
        // the temporary arrays with the fallback id.
        if (bi < 0 || uint(bi) >= a.n_expert) {
            if (lane == 0) {
                device int* ir = ids + (ulong)tgp.x * a.id_stride;
                device float* wr = w + (ulong)tgp.x * a.w_stride;
                for (uint j = 0; j < a.topk; ++j) { ir[j] = -1; wr[j] = 0.0f; }
            }
            return;
        }
        if (lane == 0) {
            si[r] = bi;
            sp[r] = tp[bi];
            tb[bi] = -INFINITY;   // taken; the next rank must not see it
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        float s = 0.0f;
        for (uint r = 0; r < a.topk; ++r) s += sp[r];   // selection order
        s = max(s, a.floor);
        device int* ir = ids + (ulong)tgp.x * a.id_stride;
        device float* wr = w + (ulong)tgp.x * a.w_stride;
        for (uint r = 0; r < a.topk; ++r) {
            ir[r] = si[r];
            wr[r] = sp[r] / s * a.w_scale;   // the reference's two steps, in its order
        }
    }
}

// Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu. MIT; see
// repository-root LICENSE.sushi-reference.
// Experimental 512-expert / top-16 path, adapted from Sushi's Hy4 route
// selection at beamivalice/sushi commit d9e2f9f741799ec367ab00a81d219b5d6b0b37c6
// (src/hy4flash/hy4_route.zig). Each simdgroup sorts 32 total-order keys,
// then three SIMD merge levels reduce 16 local lists to the global top 16.
// Probability evaluation, selection-order sum, normalization and scaling keep
// the incumbent order. Non-NaN scores outrank NaNs; if fewer than topk scores
// are orderable, the whole row uses the incumbent invalid-row result.
static inline ulong hy4_flash_route_sg_shuffle(ulong v, ushort l) {
    return (ulong(metal::simd_shuffle(uint(v >> 32), l)) << 32) | ulong(metal::simd_shuffle(uint(v), l));
}
static inline ulong hy4_flash_route_sg_xor(ulong v, ushort j) {
    return (ulong(metal::simd_shuffle_xor(uint(v >> 32), j)) << 32) |
           ulong(metal::simd_shuffle_xor(uint(v), j));
}
static inline ulong hy4_flash_route_sort32(ulong key, uint lane) {
    for (uint k = 2; k <= 32; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const ulong other = hy4_flash_route_sg_xor(key, ushort(j));
            key = (((lane & j) == 0) == ((lane & k) == 0)) ? metal::min(key, other) : metal::max(key, other);
        }
    }
    return key;
}
static inline ulong hy4_flash_route_merge32(ulong key, uint lane) {
    for (uint j = 16; j > 0; j >>= 1) {
        const ulong other = hy4_flash_route_sg_xor(key, ushort(j));
        key = ((lane & j) == 0) ? metal::min(key, other) : metal::max(key, other);
    }
    return key;
}
static inline ulong hy4_flash_route_best16of4(threadgroup const ulong* lists, uint lane) {
    const ulong a = hy4_flash_route_merge32(lane < 16 ? lists[lane] : lists[16 + (31 - lane)], lane);
    const ulong b = hy4_flash_route_merge32(lane < 16 ? lists[32 + lane] : lists[48 + (31 - lane)], lane);
    const ulong b_rev = hy4_flash_route_sg_shuffle(b, ushort(31 - lane));
    return hy4_flash_route_merge32(lane < 16 ? a : b_rev, lane);
}

kernel void hy4_flash_router_parallel_512x16(
    device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
    device int* ids [[buffer(2)]], device float* w [[buffer(3)]],
    constant Hy4FlashRouterArgs& a [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float tp[512];
    threadgroup ulong cand[256];
    threadgroup ulong cand2[64];
    device const float* lr = logits + (ulong)tgp.x * a.l_stride;
    for (uint e = tid; e < 512; e += 512) tp[e] = hy4_flash_sigmoid(lr[e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint e = uint(sg) * 32u + uint(lane);
    const float score = tp[e] + bias[e];
    ulong key = ~0ul;
    if (!metal::isnan(score) && score != -INFINITY) {
        uint bits = as_type<uint>(score == 0.0f ? 0.0f : score);
        const uint ascending = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        key = (ulong(~ascending) << 32) | ulong(e);
    } else {
        // NaNs and -infinity follow selectable values; the incumbent's strict
        // rank comparison never chooses either when its initial best is -inf.
        key = (0xFFFFFFFF00000000ul | ulong(e));
    }
    key = hy4_flash_route_sort32(key, uint(lane));
    if (lane < 16) cand[uint(sg) * 16u + uint(lane)] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < 4) {
        const ulong best = hy4_flash_route_best16of4(cand + uint(sg) * 64u, uint(lane));
        if (lane < 16) cand2[uint(sg) * 16u + uint(lane)] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;

    const ulong best = hy4_flash_route_best16of4(cand2, uint(lane));
    // Invalid keys sort after selectable values. Every lane executes every
    // shuffle; lane zero alone reduces probabilities in the incumbent order.
    bool invalid = false;
    float sum = 0.0f;
    for (ushort r = 0; r < 16; ++r) {
        const ulong ranked = hy4_flash_route_sg_shuffle(best, r);
        invalid = invalid || uint(ranked >> 32) == 0xFFFFFFFFu;
        sum += tp[uint(ranked)];
    }
    device int* ir = ids + (ulong)tgp.x * a.id_stride;
    device float* wr = w + (ulong)tgp.x * a.w_stride;
    if (invalid) {
        if (lane == 0) {
            for (uint j = 0; j < 16; ++j) { ir[j] = -1; wr[j] = 0.0f; }
        }
        return;
    }
    sum = max(sum, a.floor);
    for (ushort r = 0; r < 16; ++r) {
        const uint expert = uint(hy4_flash_route_sg_shuffle(best, r));
        if (lane == r) {
            ir[r] = int(expert);
            wr[r] = tp[expert] / sum * a.w_scale;
        }
    }
}
