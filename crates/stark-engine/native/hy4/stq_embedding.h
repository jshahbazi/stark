// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "stq_expert.h"

namespace stark::hy4::stq {

class EmbeddingProgram {
public:
    static constexpr uint32_t width = 3072, vocabulary = 120832, max_rows = 4096;

    bool open(mtl::Context& metal);

    // Gather host-validated I32 token IDs into F32 [rows, width] in token order.
    // The shader also bounds-checks IDs to protect file-backed reads.
    bool append(std::vector<mtl::Dispatch>& commands, const DenseBank& bank, const mtl::Buffer& token_ids,
                uint32_t rows, const mtl::Buffer& output) const;

private:
    mtl::Pipeline gather_;
};

}  // namespace stark::hy4::stq
