// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <string>

#include "stq_expert.h"

namespace strata::hy4_flash { struct Model; }

namespace stark::hy4::stq {

// The slices are in geometry table order. They come from the checked STQ
// source descriptors, which stay open until EngramSupply duplicates them.
// The map includes the identity tail for tokens past the stored lookup tensor.
std::string bind_engram(strata::hy4_flash::Model& model, std::span<const TensorSlice> tables,
                        std::span<const int32_t> token_map);

}  // namespace stark::hy4::stq
