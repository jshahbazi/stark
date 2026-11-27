// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "forward.h"
#include "stq_vision.h"
#include "vis_tower.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class StarkHy4ImagePrompt;

// The Metal context must outlive its resident projector and scratch buffers.
class StarkHy4Vision {
public:
    StarkHy4Vision() = default;
    ~StarkHy4Vision();
    StarkHy4Vision(const StarkHy4Vision&) = delete;
    StarkHy4Vision& operator=(const StarkHy4Vision&) = delete;

    std::string open(mtl::Context& context, const char* trunk_path, const char* projector_path,
                     int32_t trunk_hidden, int32_t rope_dim);
    std::string open_stq(mtl::Context& context, std::span<const stark::hy4::stq::TensorSlice> weights);
    std::string build(mtl::Context& context, int library);
    const strata::hy4_flash::VisionGeometry& geometry() const { return shard_.geo; }
    uint64_t reserved_bytes() const;

private:
    friend class StarkHy4ImagePrompt;
    mtl::Context* context_ = nullptr;
    strata::hy4_flash::VisionShard shard_;
    strata::hy4_flash::VisionTower tower_;
    bool ready_ = false;
};

// Owns all storage borrowed by the decoder until the image request ends.
class StarkHy4ImagePrompt {
public:
    StarkHy4ImagePrompt() = default;
    ~StarkHy4ImagePrompt();
    StarkHy4ImagePrompt(const StarkHy4ImagePrompt&) = delete;
    StarkHy4ImagePrompt& operator=(const StarkHy4ImagePrompt&) = delete;

    std::string attach(StarkHy4Vision& vision, mtl::Context& context, strata::hy4_flash::Forward& forward,
                       const int32_t* tokens, size_t token_count, const uint8_t* rgb, size_t rgb_len,
                       uint32_t width, uint32_t height);

private:
    strata::hy4_flash::Forward* forward_ = nullptr;
    std::vector<float> rows_;
    std::vector<strata::hy4_flash::Forward::ImageSpanRows> spans_;
    std::vector<int32_t> axes_;
};
