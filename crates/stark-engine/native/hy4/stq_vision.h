// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "stq_expert.h"

#include <span>
#include <string>

namespace strata::hy4_flash { struct VisionShard; }

namespace stark::hy4::stq {

// The slices are the verified BF16 vit.* tensors in source-name order, all
// from model-globals. Only the file-backed ranges survive this call.
std::string open_vision(strata::hy4_flash::VisionShard& shard, mtl::Context& metal,
                        std::span<const TensorSlice> slices);

}  // namespace stark::hy4::stq
