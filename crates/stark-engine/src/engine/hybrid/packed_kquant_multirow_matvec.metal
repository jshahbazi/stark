// Original Stark Q4_K/Q6_K two-row SIMD schedule. Storage conventions retain
// LICENSE.ggml-hybrid provenance. Geometry motivation: llama.cpp b11429,
// d81235049384534c167caea52b85a694f6103d14, ggml-metal-impl.h (2 rows/SIMD,
// 2 SIMD groups). No upstream kernel body is copied. Q4 keeps the existing
// factored row arithmetic/fallback; Q6 keeps the existing block row arithmetic.

kernel void hybrid_kquant_multirow_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = tile * 4 + simd_group * 2;
    if (first_row >= p.rows) return;
    const ulong row_bytes = ulong(p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong first_offset = ulong(first_row) * row_bytes;
    const ulong input_offset = p.group ? ulong(first_row / p.group) * p.columns : 0;
    device const uchar *codes = codes_base + offsets.x;
    // A two-row pair can cross a selected-view batch boundary. Preserve each
    // row's distinct input using the old helper at that infrequent boundary.
    if (p.group && first_row + 1 < p.rows && first_row / p.group != (first_row + 1) / p.group) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            const ulong at = first_offset + r * row_bytes;
            const ulong x = ulong((first_row + r) / p.group) * p.columns;
            const float total = hybrid_packed_kind == 12
                ? hybrid_packed_factored_dot(codes, input, at, x, p.columns, lane)
                : hybrid_packed_block_dot(codes, input, at, x, p.columns, lane);
            if (lane == 0) output[first_row + r] = total;
        }
        return;
    }
    float sums[2] = {0.0f, 0.0f};
    bool reference_required = false;
    for (uint first = 0; first < p.columns; first += 256) {
        device const uchar *first_block = codes + first_offset + ulong(first >> 8) * hybrid_packed_block_bytes;
        if (hybrid_packed_kind == 12) {
            const uint within = lane * 8;
            const uint group = within >> 5;
            const uint l = within & 31;
            const uint part = group >> 1;
            const uint hi = group & 1;
            float values[8];
            float input_sum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const float x = input[input_offset + first + within + j];
                values[j] = x;
                reference_required = reference_required || abs(x) > 1e20f;
                input_sum += x;
            }
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 2; ++r) {
                if (first_row + r >= p.rows) continue;
                device const uchar *block = first_block + r * row_bytes;
                const uint2 sm = hybrid_scale_min(block + 4, group);
                float code_dot = 0.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    const uint packed = block[16 + part * 32 + l + j];
                    const uint q = (packed >> (hi * 4)) & 15;
                    code_dot += float(q) * values[j];
                }
                sums[r] += hybrid_half(block) * float(sm.x) * code_dot
                    - hybrid_half(block + 2) * float(sm.y) * input_sum;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint half_block = 0; half_block < 2; ++half_block) {
                float values[4];
                #pragma clang loop unroll(full)
                for (uint part = 0; part < 4; ++part) {
                    values[part] = input[input_offset + first + half_block * 128 + part * 32 + lane];
                }
                #pragma clang loop unroll(full)
                for (uint r = 0; r < 2; ++r) {
                    if (first_row + r >= p.rows) continue;
                    device const uchar *block = first_block + r * row_bytes;
                    const float d = hybrid_half(block + 208);
                    const uint low0 = block[half_block * 64 + lane];
                    const uint low1 = block[half_block * 64 + lane + 32];
                    const uint high = block[128 + half_block * 32 + lane];
                    #pragma clang loop unroll(full)
                    for (uint part = 0; part < 4; ++part) {
                        const uint low = part & 1 ? low1 : low0;
                        const int q = int(((low >> (part >= 2 ? 4 : 0)) & 15)
                            | (((high >> (part * 2)) & 3) << 4)) - 32;
                        const float scale = float(as_type<char>(block[192 + half_block * 8 + lane / 16 + part * 2]));
                        const float weight = d * scale * float(q);
                        sums[r] += weight * values[part];
                    }
                }
            }
        }
    }
    if (hybrid_packed_kind == 12 && simd_any(reference_required)) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            if (first_row + r >= p.rows) continue;
            const float total = hybrid_packed_block_dot(codes, input, first_offset + r * row_bytes, input_offset, p.columns, lane);
            if (lane == 0) output[first_row + r] = total;
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first_row + r < p.rows) output[first_row + r] = total;
    }
}
