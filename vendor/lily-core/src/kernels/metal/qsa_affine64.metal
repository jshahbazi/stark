// Adapted by Stark from the vendored Lily QSA kernels; see UPSTREAM.md.
kernel void qsa_attn_split_affine64(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                                device const uchar* k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                                device const uchar* v_cache  [[buffer(2)]],
                                device const bfloat* k_scales [[buffer(3)]],
                                device const bfloat* v_scales [[buffer(4)]],
                                device const uint*   sel      [[buffer(5)]],  // [QB, k_max]
                                device const uint*   n_sel    [[buffer(6)]],  // [QB]
                                device float*        partials [[buffer(7)]],  // [QB*NQ, splits, D]
                                device float*        stats    [[buffer(8)]],  // [QB*NQ, splits, 2]
                                constant uint&       D        [[buffer(9)]],
                                constant uint&       max_seq  [[buffer(10)]],
                                constant uint&       group    [[buffer(11)]],
                                constant uint&       splits   [[buffer(12)]],
                                constant uint&       k_max    [[buffer(13)]],
                                constant uint&       ratio    [[buffer(14)]],
                                constant uint&       base_pos [[buffer(15)]],
                                constant float&      scale    [[buffer(16)]],
                                constant uint&       NQ       [[buffer(17)]],
                                constant uint&       split    [[buffer(18)]],
                                constant uint&       head_groups [[buffer(19)]],
                                uint3 tg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[QSA_HPP][QSA_SPLIT];
    threadgroup float part[QSA_HPP][(TG / 32)];
    threadgroup float part_sum[QSA_HPP][(TG / 32)];
    threadgroup float red[QSA_HPP];
    threadgroup uint tok[QSA_SPLIT];
    threadgroup float v_stage[2][((TG / 32)) * 256];

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
    device const uchar* k_head = k_cache + (ulong)kh * max_seq * D;
    device const uchar* v_head = v_cache + (ulong)kh * max_seq * D;
    device const bfloat* ks_head = k_scales + (ulong)kh * max_seq * (D / 32);
    device const bfloat* vs_head = v_scales + (ulong)kh * max_seq * (D / 32);

    if (count == 0) {
        if (tid == 0) {
            for (uint h = h_begin; h < h_end; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h;
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
                q + ((ulong)qi * NQ + (ulong)kh * group + h0 + h) * D + lane * 8;
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
            uint4 kq[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                kq[j] = kv_affine64_load8(k_head + (ulong)token * D, ks_head + (ulong)token * (D / 32), lane * 8);
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 ka = float4(as_type<bfloat4>(kq[j].xy));
                    const float4 kb = float4(as_type<bfloat4>(kq[j].zw));
                    for (uint h = 0; h < gh; ++h) {
                        const float s = simd_sum(dot(ka, qa[h]) + dot(kb, qb[h])) * scale;
                        if (lane == 0) {
                            scores[h][p] = s;
                        }
                        local_max[h] = max(local_max[h], s);
                    }
                }
            }
        }
        for (uint h = 0; h < gh; ++h) {
            const float m = simd_max(local_max[h]);
            if (lane == 0) {
                part[h][sg] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float m = -INFINITY;
                for (uint i = 0; i < (TG / 32); ++i) {
                    m = max(m, part[h][i]);
                }
                red[h] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = 0; h < gh; ++h) {
            const float chunk_max = red[h];
            float e = 0.0f;
            if (tid < count) {
                e = exp(scores[h][tid] - chunk_max);
                scores[h][tid] = e;
            }
            const float local_sum = simd_sum(e);
            if (lane == 0) {
                part_sum[h][sg] = local_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float s = 0.0f;
                for (uint i = 0; i < (TG / 32); ++i) {
                    s += part_sum[h][i];
                }
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + h;
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
            uint4 vq[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                vq[j] = kv_affine64_load8(v_head + (ulong)token * D, vs_head + (ulong)token * (D / 32), lane * 8);
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 va = float4(as_type<bfloat4>(vq[j].xy));
                    const float4 vb = float4(as_type<bfloat4>(vq[j].zw));
                    for (uint h = 0; h < gh; ++h) {
                        const float wgt = scores[h][p];
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
                    v_stage[h][sg * D + lane * 8 + j] = acc0[hb + h][j];
                    v_stage[h][sg * D + lane * 8 + 4 + j] = acc1[hb + h][j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = 0; h < nh; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + hb + h;
                for (uint d = tid; d < D; d += TG) {
                    float acc = 0.0f;
                    for (uint s = 0; s < (TG / 32); ++s) {
                        acc += v_stage[h][s * D + d];
                    }
                    partials[(hq * splits + split_idx) * D + d] = acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#if __METAL_VERSION__ >= 400
kernel void qsa_attn_gqa_nax_affine64(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                             device const uchar* k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                             device const uchar* v_cache  [[buffer(2)]],
                             device const bfloat* k_scales [[buffer(3)]],
                             device const bfloat* v_scales [[buffer(4)]],
                             device const uint*   sel      [[buffer(5)]],  // [QB, k_max]
                             device const uint*   n_sel    [[buffer(6)]],  // [QB]
                             device bfloat*       out      [[buffer(7)]],  // [QB, NQ, D]
                             constant uint&       max_seq  [[buffer(8)]],
                             constant uint&       group    [[buffer(9)]],
                             constant uint&       k_max    [[buffer(10)]],
                             constant uint&       ratio    [[buffer(11)]],
                             constant uint&       base_pos [[buffer(12)]],
                             constant float&      scale    [[buffer(13)]],
                             constant uint&       NQ       [[buffer(14)]],
                             uint2 tg   [[threadgroup_position_in_grid]],  // (KV head, query)
                             uint  sg   [[simdgroup_index_in_threadgroup]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    // Partial scores of the two halves, double-buffered by step parity so
    // one barrier per step suffices.
    threadgroup float xs[2 * 2 * 16 * 32];
    const uint kh = tg.x;
    const uint qi = tg.y;
    using namespace mpp::tensor_ops;
    constexpr uint BK = 32u;

    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint in_blocks = nblk * ratio;
    const uint total = in_blocks + (pos + 1 - tail_start);
    device const uint* sel_row = sel + (ulong)qi * k_max;
    const uint d0 = sg * uint(QSA_GQA_DH);
    device const uchar* k_head = k_cache + (ulong)kh * max_seq * QSA_TILE_D;
    device const uchar* v_head = v_cache + (ulong)kh * max_seq * QSA_TILE_D;
    device const bfloat* ks_head = k_scales + (ulong)kh * max_seq * (QSA_TILE_D / 32);
    device const bfloat* vs_head = v_scales + (ulong)kh * max_seq * (QSA_TILE_D / 32);
    const uint quad = (lane & 1u) | (((lane >> 3) & 1u) << 1);
    const uint slot = ((lane >> 1) & 3u) | (((lane >> 4) & 1u) << 2);
    // Query rows slot and slot + 8 (rows past the group read row 0 and are
    // never written).
    device const bfloat* q_row0 =
        q + ((ulong)qi * NQ + kh * group + (slot < group ? slot : 0u)) * QSA_TILE_D + d0;
    device const bfloat* q_row1 =
        q + ((ulong)qi * NQ + kh * group + (slot + 8u < group ? slot + 8u : 0u)) * QSA_TILE_D + d0;
    // Memory dim of the lane's first element of half h of chunk c.
    auto dim = [&](uint c, uint h) { return 32u * c + 16u * h + 4u * quad; };
    // The 8 values of halves 0 and 1 of chunk c of a row.
    auto row8 = [&](device const bfloat* row, uint c, thread bfloat4& a, thread bfloat4& b) {
        a = *(device const bfloat4*)(row + dim(c, 0));
        b = *(device const bfloat4*)(row + dim(c, 1));
    };
    auto affine_row8 = [&](device const uchar* row, device const bfloat* metadata,
                           uint c, thread bfloat4& a, thread bfloat4& b) {
        a = kv_affine64_load4(row, metadata, d0 + dim(c, 0));
        b = kv_affine64_load4(row, metadata, d0 + dim(c, 1));
    };

    constexpr auto qk_desc = matmul2d_descriptor(
        QSA_GQA_ROWS, 32, 32, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc = matmul2d_descriptor(
        QSA_GQA_ROWS, 32, 32, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    using QL = decltype(qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    using KR = decltype(qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>());
    using PL = decltype(pv_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    using VR = decltype(pv_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>());
    using ST = decltype(qk_op.template get_destination_cooperative_tensor<QL, KR, float>());
    using OT = decltype(pv_op.template get_destination_cooperative_tensor<PL, VR, float>());
    // (Arrays of cooperative tensors are not allowed: named ones.)
    OT o0, o1, o2, o3;
    VR w0, w1, w2, w3;
    ST s0;
    PL p0;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 16u; ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
        o2[i] = 0.0f;
        o3[i] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};
    uint tok[4];

    auto load_q = [&](uint c, thread QL& qL) {
        bfloat4 a0, a1, b0, b1;
        row8(q_row0, c, a0, a1);
        row8(q_row1, c, b0, b1);
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 4; ++e) {
            qL[e] = a0[e];
            qL[4u + e] = b0[e];
            qL[8u + e] = a1[e];
            qL[12u + e] = b1[e];
        }
    };
    // Scores over head-dim chunk c, accumulated into st.
    auto qk_chunk = [&](uint c, thread ST& st) {
        QL qL;
        KR kR;
        load_q(c, qL);
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            bfloat4 k0v, k1v;
            affine_row8(k_head + (ulong)tok[t] * QSA_TILE_D, ks_head + (ulong)tok[t] * (QSA_TILE_D / 32), c, k0v, k1v);
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 4; ++e) {
                kR[4u * t + e] = k0v[e];
                kR[16u + 4u * t + e] = k1v[e];
            }
        }
        qk_op.run(qL, kR, st);
    };
    // The V rows of output chunk c.
    auto load_v = [&](uint c, thread VR& vr) {
        _Pragma("clang loop unroll(full)")
        for (uint t3 = 0; t3 < 2u; ++t3) {
            _Pragma("clang loop unroll(full)")
            for (uint t2 = 0; t2 < 2u; ++t2) {
                bfloat4 v0v, v1v;
                affine_row8(v_head + (ulong)tok[t2 + 2u * t3] * QSA_TILE_D, vs_head + (ulong)tok[t2 + 2u * t3] * (QSA_TILE_D / 32), c, v0v, v1v);
                _Pragma("clang loop unroll(full)")
                for (uint e = 0; e < 4; ++e) {
                    vr[16u * t3 + 4u * t2 + e] = v0v[e];
                    vr[16u * t3 + 8u + 4u * t2 + e] = v1v[e];
                }
            }
        }
    };
    auto rescale = [&](thread OT& ot, float a0, float a1) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            ot[i] *= ((i >> 2) & 1u) ? a1 : a0;
        }
    };
    auto store = [&](uint c, thread OT& ot) {
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2u; ++rr) {
            const uint row = slot + 8u * rr;
            if (row < group) {
                const float inv = 1.0f / l_run[rr];
                device bfloat* o = out + ((ulong)qi * NQ + kh * group + row) * QSA_TILE_D + d0;
                _Pragma("clang loop unroll(full)")
                for (uint h = 0; h < 2u; ++h) {
                    _Pragma("clang loop unroll(full)")
                    for (uint e = 0; e < 4; ++e) {
                        o[dim(c, h) + e] = bfloat(ot[8u * h + 4u * rr + e] * inv);
                    }
                }
            }
        }
    };
    // The cache rows of this lane's keys (key slot + 8t of the step); keys
    // past the list read the query's own row, which is finite.
    auto tokens = [&](uint k0, thread uint* tk) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            const uint j = k0 + slot + 8u * t;
            tk[t] = j < in_blocks ? sel_row[j / ratio] * ratio + (j % ratio)
                                  : (j < total ? tail_start + (j - in_blocks) : pos);
        }
    };

    uint parity = 0;
    uint next[4];
    tokens(0, next);
    for (uint k0 = 0; k0 < total; k0 += BK) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            tok[t] = next[t];
        }
        tokens(k0 + BK, next);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            s0[i] = 0.0f;
        }
        qk_chunk(0, s0);
        qk_chunk(1, s0);
        qk_chunk(2, s0);
        qk_chunk(3, s0);

        // Sum the two head-dim halves (a + b is b + a: both simdgroups hold
        // the same scores afterwards).
        threadgroup float* mine = xs + (parity * 2u + sg) * (16u * 32u);
        threadgroup float* other = xs + (parity * 2u + (1u - sg)) * (16u * 32u);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            mine[i * 32u + lane] = s0[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[16];
        float mx[2] = {m_run[0], m_run[1]};
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            const uint col = 16u * (i >> 3) + 4u * quad + (i & 3u);
            const uint rr = (i >> 2) & 1u;
            const float v = (s0[i] + other[i * 32u + lane]) * scale;
            s[i] = k0 + col < total ? v : -INFINITY;
            mx[rr] = max(mx[rr], s[i]);
        }
        float alpha[2];
        float psum[2] = {0.0f, 0.0f};
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2; ++rr) {
            mx[rr] = max(mx[rr], simd_shuffle_xor(mx[rr], 1));
            mx[rr] = max(mx[rr], simd_shuffle_xor(mx[rr], 8));
            alpha[rr] = mx[rr] == -INFINITY ? 1.0f : exp(m_run[rr] - mx[rr]);
        }
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            const uint rr = (i >> 2) & 1u;
            const float p = s[i] == -INFINITY ? 0.0f : exp(s[i] - mx[rr]);
            p0[i] = bfloat(p);
            psum[rr] += p;
        }
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2; ++rr) {
            psum[rr] += simd_shuffle_xor(psum[rr], 1);
            psum[rr] += simd_shuffle_xor(psum[rr], 8);
            l_run[rr] = l_run[rr] * alpha[rr] + psum[rr];
            m_run[rr] = mx[rr];
        }
        if (alpha[0] != 1.0f || alpha[1] != 1.0f) {
            rescale(o0, alpha[0], alpha[1]);
            rescale(o1, alpha[0], alpha[1]);
            rescale(o2, alpha[0], alpha[1]);
            rescale(o3, alpha[0], alpha[1]);
        }
        load_v(0, w0);
        pv_op.run(p0, w0, o0);
        load_v(1, w1);
        pv_op.run(p0, w1, o1);
        load_v(2, w2);
        pv_op.run(p0, w2, o2);
        load_v(3, w3);
        pv_op.run(p0, w3, o3);
        parity ^= 1u;
    }
    store(0, o0);
    store(1, o1);
    store(2, o2);
    store(3, o3);
}
#endif
