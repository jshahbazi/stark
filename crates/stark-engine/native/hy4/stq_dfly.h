// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mtl_rt.h"

#include <array>
#include <memory>
#include <span>
#include <string>

namespace strata::hy4_flash { struct Model; }

namespace stark::hy4::stq {

inline constexpr std::array<int32_t, 5> kDflyTargetBlocks{25, 29, 33, 37, 41};
inline constexpr size_t kDflyMaxDrafts = 5;

struct DflyTargetLayer {
    const mtl::Buffer* kv = nullptr;
    size_t byte_offset = 0;
    // Absolute target row IDs in the target attention's read order. Empty
    // means the full available prefix, only valid through the IndexPool cutoff.
    std::span<const uint32_t> selected;
};

struct DflyTargetView {
    uint32_t position = 0;                    // the next absolute target position
    std::span<const float> result_norm;        // committed row at position - 1
    std::array<DflyTargetLayer, kDflyTargetBlocks.size()> layers;
};

class Dfly2 {
public:
    Dfly2();
    ~Dfly2();
    Dfly2(const Dfly2&) = delete;
    Dfly2& operator=(const Dfly2&) = delete;

    // Uses the existing packed trunk weights/program for embeddings and head.
    // The model and Metal context outlive the drafter.
    std::string open(strata::hy4_flash::Model& model, mtl::Context& metal, int shared_library);
    bool ready() const { return bool(impl_); }

    // Propose for the causal block [anchor, mask ...]. Uses scratch only. The
    // caller keeps the committed target view live until this call returns.
    std::string propose(const DflyTargetView& target, uint32_t anchor, std::span<int32_t> proposals,
                        std::span<float> confidence = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace stark::hy4::stq
