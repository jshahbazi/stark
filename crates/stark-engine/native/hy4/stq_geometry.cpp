// SPDX-License-Identifier: Apache-2.0
#include "stq_geometry.h"

#include "ref_quant.h"

#include <algorithm>
#include <array>
#include <utility>

namespace stark::hy4::stq {
namespace {

// beamster/Hy4-Flash-Sushi-STQ-4bpw at 61528ba635a623f127c5a7b1e1f4ac2c2b0d4885.
// StqSource pins config.json SHA-256 c1c5af8449a8e7db0dc5c5e8aa768c80673c94604997cfe13c6ee6aa98e9a3e3.
constexpr int32_t kTrunkBlocks = 42;
constexpr int32_t kLearnedContext = 1'048'576;
constexpr std::array kSparseBlocks{0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41};

struct EngramSpec {
    int32_t block;
    std::array<uint64_t, 16> buckets;
    std::array<uint64_t, 3> multipliers;
};

// Bucket counts follow the config's 2-gram then 3-gram head order. The
// multipliers follow Sushi v1.8.1 src/engram.zig and hy4flash/hy4_model.zig:
// NumPy default_rng(seed + 10007 * (block + 1)), with odd values from
// 2 * integers(0, (INT64_MAX / compressed_vocab) / 2) + 1. The four pinned
// I32 lookup tensors each have 120818 entries and compressed_vocab = 92620.
// Each hashes to 87c7b2e1c2fc227ba7fdec4dd7a867533072eb2b73a85cc54189174ccfaf5f3f.
constexpr std::array<EngramSpec, 4> kEngram{{
    {1,
     {10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147,
      10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259},
     {81853575799317, 39690085502883, 24229915018165}},
    {11,
     {10312277, 10312283, 10312301, 10312343, 10312373, 10312397, 10312409, 10312453,
      10312457, 10312483, 10312507, 10312513, 10312531, 10312543, 10312567, 10312571},
     {19457948045719, 44729556510419, 84779352922315}},
    {21,
     {10312613, 10312619, 10312633, 10312657, 10312681, 10312699, 10312723, 10312747,
      10312759, 10312789, 10312811, 10312823, 10312867, 10312873, 10312889, 10312903},
     {72974230057307, 21530602121221, 55998057243097}},
    {31,
     {10312909, 10312919, 10312927, 10312957, 10312979, 10313011, 10313021, 10313057,
      10313063, 10313117, 10313123, 10313129, 10313161, 10313179, 10313189, 10313201},
     {29128509562787, 16745155128523, 22474588302733}},
}};

}  // namespace

std::string make_geometry(int32_t context, strata::hy4_flash::Geometry& out) {
    using namespace strata::hy4_flash;
    if (context < 0 || context == 1 || context > kLearnedContext) {
        return "STQ context must be between 2 and the learned 1048576 positions, or zero for the learned limit";
    }

    Geometry geo;
    geo.hidden_dim = 3072;
    geo.dense_ff = 13824;
    geo.vocab_size = 120832;
    geo.hc_mult = 4;
    geo.hc_eps = 1e-6f;
    geo.hc_magnitude = 2.0f;
    geo.expert_dim = 1536;
    geo.expert_ff = 1536;
    geo.n_experts = 512;
    geo.top_k = 16;
    geo.n_shared_experts = 1;
    geo.swiglu_clamp = 10.0f;
    geo.rope_dim = 64;
    geo.rope_theta_sparse = 1e7f;
    geo.rope_theta_sliding = 1e4f;
    geo.routed_scale = 3.998f;
    geo.rms_norm_eps = 1e-5f;
    geo.routed_scale_normalized = true;
    geo.gating_func = 2;  // Sigmoid.
    geo.eos_token_id = 120025;
    geo.stop_tokens = {geo.eos_token_id};

    geo.q_lora_rank = 1024;
    geo.kv_lora_rank = 512;
    geo.qk_head_dim = 256;
    geo.v_head_dim = 256;
    geo.kv_a_width = geo.kv_lora_rank + geo.rope_dim;
    geo.idx_heads = 16;
    geo.idx_key_dim = 128;
    geo.idx_kpool = 4;
    geo.idx_top_k = 2048;
    // Sushi hy4_forward.zig reserves one of the 512 groups for the causal tail.
    geo.idx_select_tail = true;

    geo.block_count = kTrunkBlocks;
    geo.trunk_blocks = kTrunkBlocks;
    geo.n_mtp_blocks = 0;  // This checkpoint describes DFly2 separately.
    geo.dense_blocks = 1;
    geo.sliding_window = 512;
    geo.max_context = context == 0 ? kLearnedContext : context;
    geo.layers.reserve(kTrunkBlocks);
    for (int32_t block = 0; block < kTrunkBlocks; ++block) {
        const bool sparse = std::binary_search(kSparseBlocks.begin(), kSparseBlocks.end(), block);
        LayerSpec layer;
        layer.block_id = block;
        layer.attention = sparse ? Attention::Sparse : Attention::Sliding;
        layer.n_heads = sparse ? 32 : 64;
        layer.expert_ordinal = block - geo.dense_blocks;
        layer.has_indexer = sparse;
        geo.layers.push_back(layer);
    }

    geo.engram_ngram_size = 3;
    geo.engram_heads_per_ngram = 8;
    geo.engram_heads = (geo.engram_ngram_size - 1) * geo.engram_heads_per_ngram;
    geo.engram_head_dim = 96;
    geo.engram_pad_token = 120002;
    geo.engram_table_type = ref::BF16;
    geo.engram_row_bytes = geo.engram_head_dim * sizeof(uint16_t);
    geo.engram_tables.reserve(kEngram.size());
    for (const EngramSpec& spec : kEngram) {
        EngramTable table;
        table.block_id = spec.block;
        table.row_base = geo.engram_total_rows;
        table.head_bucket_counts.assign(spec.buckets.begin(), spec.buckets.end());
        table.multipliers.assign(spec.multipliers.begin(), spec.multipliers.end());
        table.head_offsets.reserve(spec.buckets.size());
        for (uint64_t buckets : spec.buckets) {
            // Forward's segmented reader subtracts this table's row base.
            table.head_offsets.push_back(table.row_base + table.rows);
            table.rows += buckets;
        }
        geo.engram_total_rows += table.rows;
        geo.layers[static_cast<std::size_t>(spec.block)].engram_table = static_cast<int32_t>(geo.engram_tables.size());
        geo.engram_tables.push_back(std::move(table));
    }

    out = std::move(geo);
    return {};
}

}  // namespace stark::hy4::stq
