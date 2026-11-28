// SPDX-License-Identifier: Apache-2.0
// Model-free checks for the pinned native geometry and segmented row hashes.
#include "stq_geometry.h"

#include "ref_quant.h"
#include "strata/hy4flash_gguf/engram.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>

namespace {

using strata::hy4_flash::Attention;
using strata::hy4_flash::EngramHashParams;
using strata::hy4_flash::Geometry;

void check_layers(const Geometry& geo) {
    constexpr std::array sparse{0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41};
    assert(geo.layers.size() == 42 && geo.block_count == 42 && geo.trunk_blocks == 42);
    assert(geo.n_mtp_blocks == 0 && geo.dense_blocks == 1 && geo.n_expert_ordinals() == 41);
    assert(geo.hidden_dim == 3072 && geo.hc_width() == 12288 && geo.vocab_size == 120832);
    assert(geo.expert_dim == 1536 && geo.expert_ff == 1536 && geo.n_experts == 512 && geo.top_k == 16);
    assert(geo.q_lora_rank == 1024 && geo.kv_lora_rank == 512 && geo.kv_a_width == 576);
    assert(geo.qk_head_dim == 256 && geo.v_head_dim == 256 && geo.rope_dim == 64);
    assert(geo.rope_theta_sparse == 1e7f && geo.rope_theta_sliding == 1e4f && geo.sliding_window == 512);
    assert(geo.idx_heads == 16 && geo.idx_key_dim == 128 && geo.idx_kpool == 4 && geo.idx_top_k == 2048);
    assert(geo.idx_select_tail);
    for (int32_t block = 0; block < geo.block_count; ++block) {
        const auto& layer = geo.block(block);
        const bool indexed = std::find(sparse.begin(), sparse.end(), block) != sparse.end();
        assert(layer.block_id == block && layer.expert_ordinal == block - 1 && !layer.is_mtp);
        assert(layer.attention == (indexed ? Attention::Sparse : Attention::Sliding));
        assert(layer.n_heads == (indexed ? 32 : 64) && layer.has_indexer == indexed);
    }
}

void check_engram(const Geometry& geo) {
    constexpr std::array<uint64_t, 4> row_counts{164994380, 164999008, 165004086, 165008942};
    // Independently derived from config.json's bucket counts and NumPy PCG64
    // with the pinned map's compressed vocabulary. Digests use local row IDs.
    constexpr std::array<uint64_t, 4> row_digests{
        0x744a873f9531bf86, 0xd03b8d44694e016c, 0x425d32526b7a5dcc, 0x7f0740449f867cf8,
    };
    assert(geo.engram_table_type == ref::BF16 && geo.engram_row_bytes == 192);
    assert(geo.engram_predecessors() == 2 && geo.engram_heads == 16 && geo.engram_lookup_dim() == 1536);
    assert(geo.engram_pad_token == 120002 && geo.engram_tables.size() == row_counts.size());

    std::vector<int32_t> synthetic_map(geo.engram_pad_token + 1);
    synthetic_map[geo.engram_pad_token] = 17;
    synthetic_map[1] = 19;
    synthetic_map[2] = 23;
    synthetic_map[3] = 29;
    constexpr int32_t tokens[] = {1, 2, 3};
    constexpr int32_t positions[] = {0, 1, 2};
    constexpr int32_t predecessors[] = {-1, -1, -1, 1, 1, 2};

    uint64_t base = 0;
    for (std::size_t i = 0; i < geo.engram_tables.size(); ++i) {
        const auto& table = geo.engram_tables[i];
        assert(table.block_id == 1 + static_cast<int32_t>(10 * i));
        assert(geo.block(table.block_id).engram_table == static_cast<int32_t>(i));
        assert(table.row_base == base && table.rows == row_counts[i]);
        assert(table.head_offsets.front() == base);
        assert(table.head_offsets.back() + table.head_bucket_counts.back() == base + table.rows);
        base += table.rows;

        EngramHashParams params;
        params.ngram_size = geo.engram_ngram_size;
        params.heads_per_ngram = geo.engram_heads_per_ngram;
        params.columns = geo.engram_heads;
        params.pad_token = geo.engram_pad_token;
        params.multipliers = table.multipliers;
        params.primes = table.head_bucket_counts;
        params.offsets = table.head_offsets;
        for (auto& offset : params.offsets) offset -= table.row_base;
        params.token_map = synthetic_map.data();
        params.token_map_len = static_cast<int64_t>(synthetic_map.size());

        std::vector<int64_t> ids;
        std::vector<float> valid;
        std::string error;
        assert(strata::hy4_flash::engram_row_ids(params, tokens, positions, predecessors, nullptr, 3, ids, valid, error));
        assert(ids.size() == 48 && valid.size() == ids.size());
        uint64_t digest = 14695981039346656037ULL;
        for (std::size_t j = 0; j < ids.size(); ++j) {
            assert(ids[j] >= 0 && static_cast<uint64_t>(ids[j]) < table.rows && valid[j] == 1.0f);
            digest = (digest ^ static_cast<uint64_t>(ids[j])) * 1099511628211ULL;
        }
        assert(digest == row_digests[i]);
    }
    assert(base == 660006416 && geo.engram_total_rows == base);
}

}  // namespace

int main() {
    Geometry geo;
    assert(stark::hy4::stq::make_geometry(0, geo).empty() && geo.max_context == 1048576);
    assert(stark::hy4::stq::make_geometry(-1, geo).size() > 0 && geo.max_context == 1048576);
    assert(stark::hy4::stq::make_geometry(1, geo).size() > 0 && geo.max_context == 1048576);
    assert(stark::hy4::stq::make_geometry(1048577, geo).size() > 0 && geo.max_context == 1048576);
    assert(stark::hy4::stq::make_geometry(4096, geo).empty() && geo.max_context == 4096);
    check_layers(geo);
    check_engram(geo);
    std::puts("STQ geometry and Engram row fixtures passed");
}
