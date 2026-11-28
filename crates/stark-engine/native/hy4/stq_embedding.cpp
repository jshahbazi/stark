// SPDX-License-Identifier: Apache-2.0
#include "stq_embedding.h"

namespace stark::hy4::stq {

bool EmbeddingProgram::open(mtl::Context& metal) {
    try {
        const int library = metal.compile_try({"stq_embedding.metal"}, "#define NO_GGML_COMMON", false, nullptr);
        if (library < 0) return false;
        gather_ = metal.pipeline(library, "stark_hy4_stq_embedding_k6");
        return gather_.max_threads >= 128 && gather_.simd_width == 32;
    } catch (...) {
        return false;
    }
}

bool EmbeddingProgram::append(std::vector<mtl::Dispatch>& commands, const DenseBank& bank,
                               const mtl::Buffer& token_ids, uint32_t rows, const mtl::Buffer& output) const {
    if (!bank.ready() || bank.format != Format::spec1_k6 || bank.inputs != width || bank.outputs != vocabulary ||
        gather_.max_threads < 128 || gather_.simd_width != 32 || rows == 0 || rows > max_rows ||
        !token_ids.handle || token_ids.bytes < size_t(rows) * sizeof(int32_t) || !output.handle ||
        output.bytes < size_t(rows) * width * sizeof(float)) return false;

    using mtl::Arg;
    const uint32_t shape[2] = {width, vocabulary};
    commands.push_back({&gather_, {bank.trellis.argument(), bank.suh.argument(), bank.svh.argument(),
                                   Arg::B(token_ids), Arg::B(output), Arg::V(shape)},
                        {width / 128, rows, 1}, {128, 1, 1}});
    return true;
}

}  // namespace stark::hy4::stq
