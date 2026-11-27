// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "stq_expert.h"
#include "stq_embedding.h"
#include <map>
#include <memory>

namespace strata::hy4_flash { struct Model; }

namespace stark::hy4::stq {

struct ProjectionSource {
    uint32_t inputs, outputs, rate;
    TensorSlice trellis, suh, svh;
};

enum class WeightUse : uint32_t { raw, widen, linear, absorbed_key, absorbed_value };

struct WeightSource {
    const char* alias;
    WeightUse use;
    uint32_t type, rank;
    uint32_t shape[3]; // GGML order, with the contiguous dimension first
    const TensorSlice* pieces;
    size_t piece_count;
    ProjectionSource projection;
};

struct Trunk {
    bool has_dfly = false;
    ExpertProgram program;
    EmbeddingProgram embedding;
    std::map<std::string, std::unique_ptr<DenseBank>> dense;
    std::array<std::array<ExpertBank, 3>, 41> experts;
    bool begin_residency(mtl::Context& metal) const;
};

std::string load_trunk(strata::hy4_flash::Model& model, mtl::Context& metal,
                       std::span<const WeightSource> weights,
                       std::span<const ProjectionSource> experts, bool round_bf16 = false,
                       bool pair1_decode = false, bool fused_finish_reduce = false, bool split_spec1 = false,
                       bool wide_attention = false);

} // namespace stark::hy4::stq
