// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "strata/hy4flash_gguf/geometry.hpp"

namespace stark::hy4::stq {

// Call only after the source reader validates the pinned STQ config and headers.
// A zero context selects the checkpoint's learned maximum. A nonzero context
// is a caller-selected runtime cap. Returns an empty string on success.
[[nodiscard]] std::string make_geometry(int32_t context, strata::hy4_flash::Geometry& out);

}  // namespace stark::hy4::stq
