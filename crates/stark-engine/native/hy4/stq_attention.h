// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "stq_codec.h"

namespace stark::hy4::stq {

// Decode the joint public [head * (192 + 256), 512] SPEC1 projection
// directly into the native absorbed K [head, 512, 192] and V [head, 256, 512]
// BF16 layouts. Uses the F32 reference reconstruction before BF16 rounding.
[[nodiscard]] bool split_kv_b(const TrellisView& weight, std::size_t heads,
                              std::span<const uint16_t> suh, std::span<const uint16_t> svh,
                              std::span<uint16_t> key, std::span<uint16_t> value) noexcept;

}  // namespace stark::hy4::stq
