// SPDX-License-Identifier: Apache-2.0
#include "stark_hy4_vision.h"

#include "vis_pos.h"

#include <limits>
#include <utility>

using strata::hy4_flash::Forward;
using strata::hy4_flash::ImageSpan;

namespace {
// Tencent Hy4 Flash's text_config has the same interleaved sections for both
// attention types. The trunk GGUF describes the total rotary width only.
constexpr int32_t kRopeTime = 12;
constexpr int32_t kRopeHeight = 10;
constexpr int32_t kRopeWidth = 10;
constexpr int32_t kImageStart = 120118;
constexpr int32_t kImageEnd = 120119;
constexpr int32_t kImagePad = 120120;
constexpr int32_t kPatch = 16;
constexpr int32_t kMerge = 2;
constexpr int64_t kMinPixels = 4096;
constexpr int64_t kMaxPixels = 4194304;
}  // namespace

StarkHy4Vision::~StarkHy4Vision() {
    if (context_ != nullptr) {
        tower_.close(*context_);
        close_vision(shard_, *context_);
    }
}

std::string StarkHy4Vision::open(mtl::Context& context, const char* trunk_path, const char* projector_path,
                                int32_t trunk_hidden, int32_t rope_dim) {
    if (context_ != nullptr || trunk_path == nullptr || projector_path == nullptr || projector_path[0] == '\0') {
        return "invalid vision initialization";
    }
    if (rope_dim != 2 * (kRopeTime + kRopeHeight + kRopeWidth)) {
        return "incompatible vision rotary width";
    }
    context_ = &context;
    if (const std::string error = open_vision(shard_, context, projector_path, trunk_path, trunk_hidden);
        !error.empty()) {
        return error;
    }
    const auto& geo = shard_.geo;
    if (geo.image_token != kImagePad || geo.patch != kPatch || geo.merge != kMerge ||
        geo.min_pixels != kMinPixels || geo.max_pixels != kMaxPixels) {
        return "incompatible AngelSlim vision geometry";
    }
    return {};
}

std::string StarkHy4Vision::build(mtl::Context& context, int library) {
    if (context_ != &context || !shard_.loaded() || ready_) return "invalid vision initialization";
    const std::string error = tower_.open(context, shard_, library);
    ready_ = error.empty();
    return error;
}

uint64_t StarkHy4Vision::reserved_bytes() const {
    const auto& geo = shard_.geo;
    if (!shard_.loaded()) return 0;
    const uint64_t pixels = kMaxPixels;
    const uint64_t patches = pixels / (kPatch * kPatch);
    const uint64_t cells = patches / (kMerge * kMerge);
    const uint64_t rows = 2 * cells;  // one newline per merged row, at most one per cell
    // The loaded projector mapping is already counted by the Metal context.
    // Reserve peak tower and request scratch.
    // The score block in the pinned tower is capped at 16 Mi float elements.
    constexpr uint64_t score_floats = 16ull << 20;
    const uint64_t patch_planes = patches * (3 * kPatch * kPatch + 5 * geo.hidden + geo.ffn);
    const uint64_t projector_planes = cells * (2 * geo.hidden * kMerge * kMerge + geo.out_hidden);
    const uint64_t output_planes = 2 * rows * geo.out_hidden;  // Metal output and owned CPU rows
    return 6 * pixels + sizeof(float) * (patch_planes + projector_planes + output_planes + score_floats);
}

StarkHy4ImagePrompt::~StarkHy4ImagePrompt() {
    if (forward_ != nullptr) forward_->set_image_prompt(nullptr, nullptr, 0, 0, 0, 0);
}

std::string StarkHy4ImagePrompt::attach(StarkHy4Vision& vision, mtl::Context& context, Forward& forward,
                                       const int32_t* tokens, size_t token_count, const uint8_t* rgb,
                                       size_t rgb_len, uint32_t width, uint32_t height) {
    if (forward_ != nullptr || forward.has_image_prompt() || !vision.ready_ || vision.context_ != &context ||
        tokens == nullptr || token_count == 0 || token_count > std::numeric_limits<int32_t>::max() ||
        rgb == nullptr || width > std::numeric_limits<int32_t>::max() || height > std::numeric_limits<int32_t>::max()) {
        return "invalid vision request";
    }
    const auto& geo = vision.geometry();
    const auto w = static_cast<int32_t>(width);
    const auto h = static_cast<int32_t>(height);
    if (const std::string error = check_picture_size(geo, h, w, geo.max_pixels); !error.empty()) return error;
    if (rgb_len != static_cast<size_t>(w) * h * 3) return "invalid vision pixel count";

    std::vector<ImageSpan> spans;
    const std::vector<std::pair<int32_t, int32_t>> grid{{h / geo.patch, w / geo.patch}};
    const auto n = static_cast<int64_t>(token_count);
    if (const std::string error = build_image_spans(spans, tokens, n, geo.image_token, grid, geo.merge);
        !error.empty()) {
        return error;
    }
    const auto& span = spans.front();
    if (span.start == 0 || span.start + span.count >= n || tokens[span.start - 1] != kImageStart ||
        tokens[span.start + span.count] != kImageEnd) {
        return "invalid vision prompt markers";
    }
    if (const std::string error = build_mrope_positions(axes_, n, spans, geo.merge); !error.empty()) return error;
    if (const std::string error = vision.tower_.encode(context, vision.shard_, rgb, h, w, rows_); !error.empty()) {
        return error;
    }
    if (rows_.size() != static_cast<size_t>(span.count) * geo.out_hidden) return "invalid vision output row count";
    spans_ = {{span.start, span.count, rows_.data()}};
    const auto delta = static_cast<int32_t>(axes_[token_count - 1] + 1 - n);
    forward.set_image_prompt(&spans_, axes_.data(), n, delta, 3 * kRopeHeight, 3 * kRopeWidth);
    forward_ = &forward;
    return {};
}
