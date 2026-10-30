// Qwen3.8 QSA algorithm adapted to Stark's standard Uzu kernel DSL.
// Source: lily-qwen3.8-flash-next e22d58fa7cc488493fb1b5d810a198bd29730aeb.
// Copyright 2026 Perplexity AI, Inc.; Copyright 2026 Fabio Greter.
// Apache-2.0: see LICENSE.qwen38-reference and NOTICE.qwen38-reference.
#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
using namespace metal;
#define TG 256
#define QSA_SELECT_TG 512
#define QSA_SPLIT 64
#define QSA_HPP 4
#define QSA_KB 4
#define KV_Q8_GROUP 32
static inline void qsa_rope_pair_bf16(bfloat lo, bfloat hi, float angle,
                                      device bfloat* out_lo, device bfloat* out_hi) {
    const float c = float(bfloat(cos(angle)));
    const float s = float(bfloat(sin(angle)));
    const float l = float(lo);
    const float h = float(hi);
    *out_lo = bfloat(float(bfloat(l * c)) - float(bfloat(h * s)));
    *out_hi = bfloat(float(bfloat(h * c)) + float(bfloat(l * s)));
}

// The rotary position of a sequence index: the index plus `rope_delta` (0
// for text; VISION.md `rope_deltas` for tokens generated after an image).
// Cache slots and block indices stay sequence indices; only the angle takes
// the delta.
static inline float qsa_rope_position(uint seq_index, int rope_delta) {
    return float(int(seq_index) + rope_delta);
}


static inline uint qsa_visible_blocks(uint pos, uint ratio) { return (pos + 1) / ratio; }
static inline void kv_q8_store(float x, device char* value, device half* scale, uint lane) {
    const float amax = simd_max(abs(x));
    const half d = half(min(amax / 127.0f, 65504.0f));
    const float inverse = float(d) != 0.0f ? 1.0f / float(d) : 0.0f;
    *value = char(clamp(rint(x * inverse), -127.0f, 127.0f));
    if (lane == 0) *scale = d;
}


PUBLIC KERNEL(Qwen38QsaPrepareQuery)(
    const device bfloat* qg,
    const device float* w,
    device bfloat* q,
    device bfloat* gate,
    const constant uint& query_heads,
    const constant uint& rot,
    const constant uint& pos,
    const constant float& theta,
    const constant float& eps,
    threadgroup float partial[8],
    threadgroup float inv_rms[1],
    threadgroup bfloat normed[256],
    uint head GROUPS(query_heads),
    uint tid THREADS(256),
    const ThreadContext thread_context
) {
    const uint sg = thread_context.simdgroup_index;
    const uint lane = thread_context.simd_lane_id;
    const uint D = 256;



    const ulong src = (ulong)head * 2 * D;
    const ulong dst = (ulong)head * D;
    const float value = float(qg[src + tid]);
    float acc = value * value;
    gate[dst + tid] = qg[src + D + tid];
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partial[i];
        }
        inv_rms[0] = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms[0] * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = float(pos) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        q[dst + tid] = bfloat(lo * c - hi * s);
        q[dst + half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }

}


PUBLIC KERNEL(Qwen38QsaPrepareKey)(
    const device bfloat* k,
    const device float* w,
    device char* cache,
    device half* scales,
    const constant uint& kv_heads,
    const constant uint& max_seq,
    const constant uint& rot,
    const constant uint& pos,
    const constant float& theta,
    const constant float& eps,
    threadgroup float partial[8],
    threadgroup float inv_rms[1],
    threadgroup bfloat normed[256],
    threadgroup bfloat roped[256],
    uint head GROUPS(kv_heads),
    uint tid THREADS(256),
    const ThreadContext thread_context
) {
    const uint sg = thread_context.simdgroup_index;
    const uint lane = thread_context.simd_lane_id;
    const uint D = 256;




    const ulong src = (ulong)head * D;
    const ulong row = (ulong)head * max_seq + pos;
    const float value = float(k[src + tid]);
    float acc = value * value;
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partial[i];
        }
        inv_rms[0] = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms[0] * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = float(pos) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        roped[tid] = bfloat(lo * c - hi * s);
        roped[half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        roped[tid] = normed[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv_q8_store(float(roped[tid]), cache + row * D + tid,
                scales + row * (D / KV_Q8_GROUP) + tid / KV_Q8_GROUP, lane);

}


PUBLIC KERNEL(Qwen38QsaStoreValue)(
    const device bfloat* rows,
    device char* cache,
    device half* scales,
    const constant uint& kv_heads,
    const constant uint& max_seq,
    const constant uint& pos,
    uint head GROUPS(kv_heads),
    uint tid THREADS(256),
    const ThreadContext thread_context
) {
    const uint D=256, kvh=kv_heads, base_pos=pos, gid=head*256+tid;
    const uint lane=thread_context.simd_lane_id;

    uint d = gid % D;
    uint h = (gid / D) % kvh;
    uint m = gid / (D * kvh);
    const ulong row = (ulong)h * max_seq + base_pos + m;
    kv_q8_store(float(rows[gid]), cache + row * D + d,
                scales + row * (D / KV_Q8_GROUP) + d / KV_Q8_GROUP, lane);

}


PUBLIC KERNEL(Qwen38QsaPrepareIndex)(
    const device bfloat* qk,
    const device float* w,
    device bfloat* q,
    device bfloat* raw,
    const constant uint& index_heads,
    const constant uint& rot,
    const constant uint& pos,
    const constant float& theta,
    const constant float& eps,
    threadgroup float partial[8],
    threadgroup float inv_rms[1],
    threadgroup bfloat normed[256],
    uint head GROUPS(index_heads),
    uint tid THREADS(128),
    const ThreadContext thread_context
) {
    const uint sg = thread_context.simdgroup_index;
    const uint lane = thread_context.simd_lane_id;
    const uint D=128, seg=head, base_pos=pos;
    const int rope_delta=0;




    const uint row = seg / index_heads;
    const uint h = seg % index_heads;
    const ulong src = (ulong)row * (index_heads + 1) * D + (ulong)h * D;
    const ulong dst = (ulong)seg * D;
    const float value = float(qk[src + tid]);
    float acc = simd_sum(value * value);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (D + 31) / 32; ++i) {
            total += partial[i];
        }
        inv_rms[0] = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms[0] * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float angle = qsa_rope_position(base_pos + row, rope_delta)
                          * pow(theta, -2.0f * float(tid) / float(rot));
        qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           q + dst + tid, q + dst + half_rot + tid);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }

    if (head == 0) raw[ulong(pos) * 128 + tid] = qk[ulong(index_heads) * 128 + tid];

}


PUBLIC KERNEL(Qwen38QsaBlockKeys)(
    const device bfloat* cache,
    const device float* w,
    device bfloat* blk,
    const constant uint& ratio,
    const constant uint& first_block,
    const constant uint& rot,
    const constant float& theta,
    const constant float& eps,
    threadgroup float partial[8],
    threadgroup float inv_rms[1],
    threadgroup bfloat normed[256],
    uint tg GROUPS(1),
    uint tid THREADS(128),
    const ThreadContext thread_context
) {
    const uint sg = thread_context.simdgroup_index;
    const uint lane = thread_context.simd_lane_id;
    const uint D=128,count=1;
    const int rope_delta=0;




    if (tg >= count) {
        return;  // uniform per threadgroup: no barrier is skipped unevenly
    }
    const uint b = first_block + tg;
    float pooled = 0.0f;
    for (uint i = 0; i < ratio; ++i) {
        pooled += float(cache[(ulong)(b * ratio + i) * D + tid]);
    }
    // The reference pools in f32 and rounds the mean to bf16 before the norm.
    const float value = float(bfloat(pooled / float(ratio)));
    float acc = simd_sum(value * value);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (D + 31) / 32; ++i) {
            total += partial[i];
        }
        inv_rms[0] = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms[0] * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong dst = (ulong)b * D;
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float angle = qsa_rope_position(b * ratio, rope_delta)
                          * pow(theta, -2.0f * float(tid) / float(rot));
        qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           blk + dst + tid, blk + dst + half_rot + tid);
    } else if (tid >= rot) {
        blk[dst + tid] = normed[tid];
    }

}


PUBLIC KERNEL(Qwen38QsaScores)(
    const device bfloat* q,
    const device bfloat* blk,
    device float* scores,
    const constant uint& index_heads,
    const constant uint& nb_max,
    threadgroup float qs[512],
    uint tile GROUPS(nb_max.div_ceil(256)),
    uint tid THREADS(256)
) {
    const uint D=128;
    const uint2 tg=uint2(tile,0);
    const float inv_sqrt_d=0.08838834764831845f;
  // index_heads * D <= 512

    const uint qi = tg.y;
    for (uint i = tid; i < index_heads * D; i += TG) {
        qs[i] = float(q[(ulong)qi * index_heads * D + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b = tg.x * TG + tid;
    if (b >= nb_max) {
        return;
    }
    const uint nb = nb_max;
    float score = -INFINITY;
    if (b < nb) {
        device const uint4* krow = (device const uint4*)(blk + (ulong)b * D);
        float dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = 0; i * 8 < D; ++i) {
            const uint4 kq = krow[i];
            const float4 ka = float4(as_type<bfloat4>(kq.xy));
            const float4 kb = float4(as_type<bfloat4>(kq.zw));
            for (uint h = 0; h < index_heads; ++h) {
                const float4 qa(qs[h * D + i * 8], qs[h * D + i * 8 + 1],
                                qs[h * D + i * 8 + 2], qs[h * D + i * 8 + 3]);
                const float4 qb(qs[h * D + i * 8 + 4], qs[h * D + i * 8 + 5],
                                qs[h * D + i * 8 + 6], qs[h * D + i * 8 + 7]);
                dots[h] += dot(ka, qa) + dot(kb, qb);
            }
        }
        score = 0.0f;
        for (uint h = 0; h < index_heads; ++h) {
            score += max(dots[h], 0.0f);
        }
        score *= inv_sqrt_d;
    }
    scores[(ulong)qi * nb_max + b] = score;

}

// Non-negative float scores order like their bit patterns.
static inline uint qsa_key(float s) {
    return as_type<uint>(max(s, 0.0f));
}

// Exclusive prefix sum of one value per thread over the whole threadgroup
// (the sum must fit 32 bits); returns this thread's rank and writes the
// total. Consecutive scans need a threadgroup barrier between them (`sums`
// is rewritten by the next call).
template <uint NT>
static inline uint qsa_scan(uint value, threadgroup uint* sums, thread uint& total,
                            uint sg, uint lane) {
    const uint local = simd_prefix_exclusive_sum(value);
    const uint sg_total = simd_sum(value);
    if (lane == 0) {
        sums[sg] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Lane s holds simdgroup s's total; the scan over the lanes gives every
    // simdgroup's offset and the total in one simd step.
    const uint mine = lane < NT / 32 ? sums[lane] : 0u;
    const uint before = simd_shuffle(simd_prefix_exclusive_sum(mine), sg);
    total = simd_sum(mine);
    return before + local;
}

// Counts one per active lane into hist[bin] with one atomic per distinct
// bin per simdgroup: the top radix digit of the scores (sign and exponent)
// takes a handful of values, so per-lane atomics would serialize on them.
static inline void qsa_hist_add(threadgroup atomic_uint* hist, uint bin, bool active,
                                uint lane) {
    while (true) {
        const uint leader = simd_min(active ? bin : 0xFFFFFFFFu);
        if (leader == 0xFFFFFFFFu) {
            break;
        }
        const bool same = active && bin == leader;
        const ulong votes = (simd_vote::vote_t)simd_ballot(same);
        if (same) {
            active = false;
            if (lane == uint(ctz(votes))) {
                atomic_fetch_add_explicit(&hist[leader], popcount(votes),
                                          memory_order_relaxed);
            }
        }
    }
}

// Largest per-thread block count of the selection (32K tokens at ratio 4
// in one chunk).
#define QSA_SELECT_CACHE_MAX 32

// The selection of one query over `nb` blocks with `CACHE` consecutive
// blocks per thread: a chunk is CACHE * QSA_SELECT_TG blocks, the first
// chunk's keys stay in registers across the passes and later chunks are
// re-read from the score row. See qsa_select_blocks.
template <uint CACHE, uint NT = QSA_SELECT_TG>
static void qsa_select_body(device const float* row, device uint* out, uint nb,
                            uint k_max, threadgroup atomic_uint* hist,
                            threadgroup uint* sums, threadgroup uint* prefix_s,
                            threadgroup uint* k_rem_s, uint tid, uint sg, uint lane) {
    constexpr uint CHUNK = CACHE * NT;
    const uint t0 = tid * CACHE;
    uint keys[CACHE];
    for (uint i = 0; i < CACHE; ++i) {
        const uint b = t0 + i;
        keys[i] = b < nb ? qsa_key(row[b]) : 0u;
    }
    static_assert(NT >= 256 && NT % 32 == 0, "at least one selection thread per radix bin");
    if (tid < 256) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&hist[256 + tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Radix select (MSB first) for the k_max-th largest key. The top digit
    // (sign and exponent) crowds a few bins: it is counted with one atomic
    // per distinct bin per simdgroup. Later digits spread over the bins
    // and only the keys under the prefix count.
    uint prefix = 0;
    uint mask = 0;
    uint k_rem = k_max;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24 - 8 * pass;
        threadgroup atomic_uint* h = hist + (pass & 1) * 256;
        for (uint b0 = 0; b0 < nb; b0 += CHUNK) {
            for (uint i = 0; i < CACHE; ++i) {
                const uint b = b0 + t0 + i;
                const uint key = b0 == 0 ? keys[i] : (b < nb ? qsa_key(row[b]) : 0u);
                const bool active = b < nb && (key & mask) == prefix;
                const uint bin = (key >> shift) & 0xFFu;
                if (pass == 0) {
                    qsa_hist_add(h, bin, active, lane);
                } else if (active) {
                    atomic_fetch_add_explicit(&h[bin], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t holds bin 255 - t: the scan gives the count above each
        // bin, and exactly one bin straddles the k_rem-th key.
        // Threads past the 256 bins hold nothing (their `c` is 0, so they
        // never straddle the k_rem-th key).
        const uint c = tid < 256 ? atomic_load_explicit(&h[255 - tid], memory_order_relaxed) : 0u;
        uint total;
        const uint above = qsa_scan<NT>(c, sums, total, sg, lane);
        if (tid < 256 && above < k_rem && above + c >= k_rem) {
            *prefix_s = prefix | ((255 - tid) << shift);
            *k_rem_s = k_rem - above;
        }
        if (tid < 256) {
            atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = *prefix_s;
        k_rem = *k_rem_s;
        mask |= 0xFFu << shift;
    }
    const uint threshold = prefix;
    // Elements above the threshold are all taken; exactly k_rem ties fill up.
    const uint ties_to_take = k_rem;

    // Compaction in ascending block order, one scan per chunk: the low half
    // of the scanned value counts a thread's keys above the threshold, the
    // high half its ties (each at most CHUNK).
    uint sel_base = 0;
    uint tie_base = 0;
    for (uint b0 = 0; b0 < nb; b0 += CHUNK) {
        uint n_above = 0;
        uint n_tie = 0;
        for (uint i = 0; i < CACHE; ++i) {
            const uint b = b0 + t0 + i;
            const uint key = b0 == 0 ? keys[i] : (b < nb ? qsa_key(row[b]) : 0u);
            n_above += (b < nb && key > threshold) ? 1u : 0u;
            n_tie += (b < nb && key == threshold) ? 1u : 0u;
        }
        uint totals;
        const uint ranks =
            qsa_scan<NT>((n_tie << 16) | n_above, sums, totals, sg, lane);
        const uint quota = ties_to_take > tie_base ? ties_to_take - tie_base : 0u;
        // Taken keys before this thread's: the scanned counts; within them,
        // the thread walks its blocks in order.
        uint a = sel_base + (ranks & 0xFFFFu);
        uint t = ranks >> 16;
        for (uint i = 0; i < CACHE; ++i) {
            const uint b = b0 + t0 + i;
            const uint key = b0 == 0 ? keys[i] : (b < nb ? qsa_key(row[b]) : 0u);
            if (b < nb) {
                if (key > threshold) {
                    out[a + min(t, quota)] = b;
                    ++a;
                } else if (key == threshold) {
                    if (t < quota) {
                        out[a + t] = b;
                    }
                    ++t;
                }
            }
        }
        sel_base += (totals & 0xFFFFu) + min(totals >> 16, quota);
        tie_base += totals >> 16;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}



PUBLIC KERNEL(Qwen38QsaSelect)(
    const device float* scores,
    device uint* sel,
    device uint* n_sel,
    const constant uint& nb_max,
    const constant uint& k_max,
    threadgroup atomic_uint hist[512],
    threadgroup uint sums[16],
    threadgroup uint prefix_s[1],
    threadgroup uint k_rem_s[1],
    uint qi GROUPS(1),
    uint tid THREADS(512),
    const ThreadContext thread_context
) {
    const uint sg = thread_context.simdgroup_index;
    const uint lane = thread_context.simd_lane_id;

    // Two histograms: the one the next pass counts into is cleared while
    // the current pass picks its digit.





    const uint nb = nb_max;
    device const float* row = scores + (ulong)qi * nb_max;
    device uint* out = sel + (ulong)qi * k_max;

    if (nb <= k_max) {
        for (uint b = tid; b < nb; b += QSA_SELECT_TG) {
            out[b] = b;
        }
        if (tid == 0) {
            n_sel[qi] = nb;
        }
        return;
    }
    if (nb <= 4 * QSA_SELECT_TG) {
        qsa_select_body<4>(row, out, nb, k_max, hist, sums, prefix_s, k_rem_s,
                           tid, sg, lane);
    } else if (nb <= 8 * QSA_SELECT_TG) {
        qsa_select_body<8>(row, out, nb, k_max, hist, sums, prefix_s, k_rem_s,
                           tid, sg, lane);
    } else if (nb <= 16 * QSA_SELECT_TG) {
        qsa_select_body<16>(row, out, nb, k_max, hist, sums, prefix_s, k_rem_s,
                            tid, sg, lane);
    } else {
        qsa_select_body<QSA_SELECT_CACHE_MAX>(row, out, nb, k_max, hist, sums,
                                              prefix_s, k_rem_s, tid, sg, lane);
    }
    if (tid == 0) {
        n_sel[qi] = k_max;
    }

}

static inline uint qsa_token_at(device const uint* sel_row, uint nblk, uint ratio,
                                uint tail_start, uint i) {
    const uint in_blocks = nblk * ratio;
    return i < in_blocks ? sel_row[i / ratio] * ratio + (i % ratio)
                         : tail_start + (i - in_blocks);
}



PUBLIC KERNEL(Qwen38QsaAttention)(
    const device bfloat* q,
    const device char* k_cache,
    const device char* v_cache,
    const device half* k_scales,
    const device half* v_scales,
    const device uint* sel,
    const device uint* n_sel,
    device float* partials,
    device float* stats,
    const constant uint& max_seq,
    const constant uint& group,
    const constant uint& splits,
    const constant uint& k_max,
    const constant uint& ratio,
    const constant uint& base_pos,
    const constant float& scale,
    const constant uint& query_heads,
    const constant uint& head_groups,
    const constant uint& kv_heads,
    threadgroup float scores[256],
    threadgroup float part[32],
    threadgroup float part_sum[32],
    threadgroup float red[4],
    threadgroup uint tok[64],
    threadgroup float v_stage[4096],
    uint head_group GROUPS(kv_heads * head_groups),
    uint split_index GROUPS(splits),
    uint tid THREADS(256),
    const ThreadContext thread_context
) {
    const uint sg = thread_context.simdgroup_index;
    const uint lane = thread_context.simd_lane_id;
    const uint D=256,split=64;
    const uint3 tg=uint3(head_group,split_index,0);







    const uint kh = tg.x / head_groups;
    const uint hg = tg.x - kh * head_groups;
    const uint split_idx = tg.y;
    const uint qi = tg.z;
    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint total = nblk * ratio + (pos + 1 - tail_start);
    const uint slot0 = split_idx * split;
    const uint count = slot0 < total ? min(split, total - slot0) : 0;
    const uint h_begin = hg * QSA_HPP;
    const uint h_end = head_groups == 1 ? group : min(h_begin + QSA_HPP, group);
    device const uint* sel_row = sel + (ulong)qi * k_max;
    device const char* k_head = k_cache + (ulong)kh * max_seq * D;
    device const char* v_head = v_cache + (ulong)kh * max_seq * D;
    device const half* ks_head = k_scales + (ulong)kh * max_seq * (D / 32);
    device const half* vs_head = v_scales + (ulong)kh * max_seq * (D / 32);

    if (count == 0) {
        if (tid == 0) {
            for (uint h = h_begin; h < h_end; ++h) {
                const ulong hq = (ulong)qi * query_heads + (ulong)kh * group + h;
                stats[(hq * splits + split_idx) * 2] = -INFINITY;
                stats[(hq * splits + split_idx) * 2 + 1] = 0.0f;
            }
        }
        return;
    }
    // The split's cache positions, one dependent load per thread.
    for (uint p = tid; p < count; p += TG) {
        tok[p] = qsa_token_at(sel_row, nblk, ratio, tail_start, slot0 + p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h0 = h_begin; h0 < h_end; h0 += QSA_HPP) {
        const uint gh = min(uint(QSA_HPP), h_end - h0);
        float4 qa[QSA_HPP];
        float4 qb[QSA_HPP];
        for (uint h = 0; h < gh; ++h) {
            device const bfloat* qh =
                q + ((ulong)qi * query_heads + (ulong)kh * group + h0 + h) * D + lane * 8;
            qa[h] = float4(float(qh[0]), float(qh[1]), float(qh[2]), float(qh[3]));
            qb[h] = float4(float(qh[4]), float(qh[5]), float(qh[6]), float(qh[7]));
        }
        float local_max[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            local_max[h] = -INFINITY;
        }
        // Simdgroup sg takes tokens sg, sg + 8, ... in order, QSA_KB rows
        // requested per step.
        for (uint p0 = sg; p0 < count; p0 += QSA_KB * ((TG / 32))) {
            uint2 kq[QSA_KB];
            float ksc[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                kq[j] = ((device const uint2*)(k_head + (ulong)token * D))[lane];
                ksc[j] = float(ks_head[(ulong)token * (D / 32) + lane / 4]);
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 ka = float4(as_type<char4>(kq[j].x)) * ksc[j];
                    const float4 kb = float4(as_type<char4>(kq[j].y)) * ksc[j];
                    for (uint h = 0; h < gh; ++h) {
                        const float s = simd_sum(dot(ka, qa[h]) + dot(kb, qb[h])) * scale;
                        if (lane == 0) {
                            scores[(h) * 64 + (p)] = s;
                        }
                        local_max[h] = max(local_max[h], s);
                    }
                }
            }
        }
        for (uint h = 0; h < gh; ++h) {
            const float m = simd_max(local_max[h]);
            if (lane == 0) {
                part[(h) * 8 + (sg)] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float m = -INFINITY;
                for (uint i = 0; i < (TG / 32); ++i) {
                    m = max(m, part[(h) * 8 + (i)]);
                }
                red[h] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = 0; h < gh; ++h) {
            const float chunk_max = red[h];
            float e = 0.0f;
            if (tid < count) {
                e = exp(scores[(h) * 64 + (tid)] - chunk_max);
                scores[(h) * 64 + (tid)] = e;
            }
            const float local_sum = simd_sum(e);
            if (lane == 0) {
                part_sum[(h) * 8 + (sg)] = local_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float s = 0.0f;
                for (uint i = 0; i < (TG / 32); ++i) {
                    s += part_sum[(h) * 8 + (i)];
                }
                const ulong hq = (ulong)qi * query_heads + (ulong)kh * group + h0 + h;
                stats[(hq * splits + split_idx) * 2] = red[h];
                stats[(hq * splits + split_idx) * 2 + 1] = s;
            }
        }

        float4 acc0[QSA_HPP];
        float4 acc1[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            acc0[h] = float4(0.0f);
            acc1[h] = float4(0.0f);
        }
        for (uint p0 = sg; p0 < count; p0 += QSA_KB * ((TG / 32))) {
            uint2 vq[QSA_KB];
            float vsc[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                vq[j] = ((device const uint2*)(v_head + (ulong)token * D))[lane];
                vsc[j] = float(vs_head[(ulong)token * (D / 32) + lane / 4]);
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 va = float4(as_type<char4>(vq[j].x)) * vsc[j];
                    const float4 vb = float4(as_type<char4>(vq[j].y)) * vsc[j];
                    for (uint h = 0; h < gh; ++h) {
                        const float wgt = scores[(h) * 64 + (p)];
                        acc0[h] += wgt * va;
                        acc1[h] += wgt * vb;
                    }
                }
            }
        }
        for (uint hb = 0; hb < gh; hb += 2) {
            const uint nh = min(2u, gh - hb);
            for (uint h = 0; h < nh; ++h) {
                for (uint j = 0; j < 4; ++j) {
                    v_stage[(h) * 2048 + (sg * D + lane * 8 + j)] = acc0[hb + h][j];
                    v_stage[(h) * 2048 + (sg * D + lane * 8 + 4 + j)] = acc1[hb + h][j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = 0; h < nh; ++h) {
                const ulong hq = (ulong)qi * query_heads + (ulong)kh * group + h0 + hb + h;
                for (uint d = tid; d < D; d += TG) {
                    float acc = 0.0f;
                    for (uint s = 0; s < (TG / 32); ++s) {
                        acc += v_stage[(h) * 2048 + (s * D + d)];
                    }
                    partials[(hq * splits + split_idx) * D + d] = acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

}


PUBLIC KERNEL(Qwen38QsaCombine)(
    const device float* partials,
    const device float* stats,
    const device bfloat* gate,
    device bfloat* out,
    const constant uint& query_heads,
    const constant uint& splits,
    threadgroup float m_max[1],
    threadgroup float inv_l[1],
    uint hq GROUPS(query_heads),
    uint tid THREADS(256)
) {
    const uint D=256;

    // Combine split statistics without per-split scratch.



    if (tid == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < splits; ++s) {
            m = max(m, stats[(hq * splits + s) * 2]);
        }
        float l = 0.0f;
        for (uint s = 0; s < splits; ++s) {
            float w = stats[(hq * splits + s) * 2 + 1] *
                exp(stats[(hq * splits + s) * 2] - m);
            l += w;
        }
        m_max[0] = m;
        inv_l[0] = 1.0f / l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float m = m_max[0];
    for (uint d = tid; d < D; d += TG) {
        float acc = 0.0f;
        for (uint s = 0; s < splits; ++s) {
            acc += exp(stats[(hq * splits + s) * 2] - m) *
                partials[((ulong)hq * splits + s) * D + d];
        }
        const float rounded = float(bfloat(acc * inv_l[0]));
        const float gate_value = 1.0f / (1.0f + exp(-float(gate[hq * D + d])));
        out[hq * D + d] = bfloat(gate_value * rounded);
    }

}
