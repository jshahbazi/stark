// SPDX-License-Identifier: Apache-2.0
#include "stq_expert.h"

#include <algorithm>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace stark::hy4::stq {

std::optional<ExpertRoutes> plan_routes(std::span<const int32_t> expert_ids) {
    constexpr uint32_t topk = 16, max_rows = 4096, experts = ExpertBank::experts;
    if (expert_ids.empty() || expert_ids.size() > max_rows * topk || expert_ids.size() % topk != 0) return std::nullopt;
    std::array<uint32_t, experts> counts{}, first{};
    for (int32_t expert : expert_ids) {
        if (expert < -1 || expert >= int32_t(experts)) return std::nullopt;
        if (expert >= 0) ++counts[size_t(expert)];
    }
    ExpertRoutes plan;
    uint32_t total = 0;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        first[expert] = total;
        for (uint32_t offset = 0; offset < counts[expert]; offset += 32) {
            plan.jobs.push_back({expert, total + offset, std::min(32u, counts[expert] - offset), 0});
        }
        total += counts[expert];
    }
    plan.rows.resize(total);
    plan.inverse.resize(expert_ids.size(), std::numeric_limits<uint32_t>::max());
    for (uint32_t slot = 0; slot < expert_ids.size(); ++slot) {
        const int32_t expert = expert_ids[slot];
        if (expert < 0) continue;
        const uint32_t index = first[size_t(expert)]++;
        plan.rows[index] = {uint32_t(expert), slot / topk};
        plan.inverse[slot] = index;
    }
    return plan;
}

MappedPlane::~MappedPlane() { close(); }

void MappedPlane::close() {
    if (metal_ == nullptr) return;
    void* address = buffer_.ptr;
    const size_t bytes = buffer_.bytes;
    metal_->free(buffer_);
    munmap(address, bytes);
    metal_ = nullptr;
    bias_ = 0;
}

bool MappedPlane::open(mtl::Context& metal, TensorSlice slice, uint64_t expected_bytes) {
    if (ready() || expected_bytes == 0 || slice.size != expected_bytes || slice.offset % 4 != 0) return false;
    struct stat info {};
    if (fstat(slice.fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0) return false;
    const auto size = static_cast<uint64_t>(info.st_size);
    if (slice.offset > size || slice.size > size - slice.offset) return false;
    const int page_size = getpagesize();
    if (page_size <= 0) return false;
    const auto page = static_cast<uint64_t>(page_size);
    const uint64_t aligned = slice.offset / page * page;
    const uint64_t bias = slice.offset - aligned;
    const uint64_t mapped = (bias + slice.size + page - 1) / page * page;
    if (mapped > metal.max_buffer_length()) return false;
    void* address = mmap(nullptr, static_cast<size_t>(mapped), PROT_READ, MAP_PRIVATE, slice.fd,
                         static_cast<off_t>(aligned));
    if (address == MAP_FAILED) return false;
    try {
        buffer_ = metal.wrap(address, static_cast<size_t>(mapped));
    } catch (...) {
        munmap(address, static_cast<size_t>(mapped));
        return false;
    }
    metal_ = &metal;
    bias_ = static_cast<size_t>(bias);
    return true;
}

bool ExpertBank::open(mtl::Context& metal, TensorSlice words, TensorSlice input_scale, TensorSlice output_scale) {
    if (trellis.ready() || suh.ready() || svh.ready()) return false;
    constexpr uint64_t words_bytes = uint64_t(experts) * (width / 32) * (width / 16) * 2 * 64 * sizeof(uint16_t);
    constexpr uint64_t scale_bytes = uint64_t(experts) * width * sizeof(uint16_t);
    if (trellis.open(metal, words, words_bytes) && suh.open(metal, input_scale, scale_bytes) &&
        svh.open(metal, output_scale, scale_bytes)) return true;
    svh.close();
    suh.close();
    trellis.close();
    return false;
}

bool DenseBank::open(mtl::Context& metal, Format rate, uint32_t input_width, uint32_t output_width,
                     TensorSlice words, TensorSlice input_scale, TensorSlice output_scale) {
    if (trellis.ready() || suh.ready() || svh.ready() ||
        (rate != Format::spec1_k4 && rate != Format::spec1_k6) || input_width == 0 || output_width == 0 ||
        input_width % 128 != 0 || output_width % 128 != 0) return false;
    const uint64_t word_bytes = uint64_t(input_width / 16) * (output_width / 16) *
                                tile_word_count(rate) * sizeof(uint16_t);
    if (trellis.open(metal, words, word_bytes) && suh.open(metal, input_scale, uint64_t(input_width) * 2) &&
        svh.open(metal, output_scale, uint64_t(output_width) * 2)) {
        format = rate;
        inputs = input_width;
        outputs = output_width;
        return true;
    }
    svh.close();
    suh.close();
    trellis.close();
    return false;
}

bool ExpertProgram::open(mtl::Context& metal, bool round_bf16, bool pair1_decode, bool fused_finish_reduce,
                         bool split_spec1) {
    if (!metal.supports_metal4()) return false;
    try {
        std::vector<std::string> sources{"pair1_k4_grouped.metal", "pair1_k4_io.metal", "spec1_dense.metal"};
        if (pair1_decode) sources.emplace_back("pair1_k4_decode.metal");
        const int library = metal.compile_try(sources, "#define NO_GGML_COMMON", false, nullptr);
        if (library < 0) return false;
        mtl::Constants boundary;
        if (round_bf16) boundary.b(0, true);
        prepare_ = metal.pipeline(library, "stark_hy4_stq_pair1_prepare", boundary);
        project_ = metal.pipeline(library, "stark_hy4_stq_pair1_k4_grouped");
        finish_ = metal.pipeline(library, "stark_hy4_stq_pair1_finish");
        dense_finish_ = round_bf16 ? metal.pipeline(library, "stark_hy4_stq_pair1_finish", boundary) : finish_;
        prepare_down_ = metal.pipeline(library, "stark_hy4_stq_pair1_prepare_down");
        combine_ = metal.pipeline(library, "stark_hy4_stq_pair1_combine");
        finish_reduce_ = fused_finish_reduce ? metal.pipeline(library, "stark_hy4_stq_pair1_finish_reduce") : mtl::Pipeline{};
        dense_k4_ = metal.pipeline(library, "stark_hy4_stq_spec1_k4_dense");
        dense_k6_ = metal.pipeline(library, "stark_hy4_stq_spec1_k6_dense");
        dense_split_ = split_spec1 ? metal.pipeline(library, "stark_hy4_stq_spec1_k6_split") : mtl::Pipeline{};
        dense_split_finish_ = split_spec1 ? metal.pipeline(library, "stark_hy4_stq_spec1_split_finish", boundary) : mtl::Pipeline{};
        decode_ = pair1_decode ? metal.pipeline(library, "stark_hy4_stq_pair1_k4_decode") : mtl::Pipeline{};
        return prepare_.max_threads >= 128 && project_.max_threads >= 128 && finish_.max_threads >= 128 &&
               prepare_down_.max_threads >= 128 && combine_.max_threads >= 128 &&
               dense_k4_.max_threads >= 128 && dense_k6_.max_threads >= 128 && dense_finish_.max_threads >= 128 &&
               (!pair1_decode || (decode_.max_threads >= 128 && decode_.simd_width == 32)) &&
               (!fused_finish_reduce || (finish_reduce_.max_threads >= 512 && finish_reduce_.simd_width == 32)) &&
               (!split_spec1 || (dense_split_.max_threads >= 128 && dense_split_.simd_width == 32 &&
                               dense_split_finish_.max_threads >= 128 && dense_split_finish_.simd_width == 32));
    } catch (...) {
        return false;
    }
}

bool ExpertProgram::append(std::vector<mtl::Dispatch>& commands, const ExpertBank& bank, const mtl::Buffer& input,
                           const mtl::Buffer& row_map, const mtl::Buffer& jobs, uint32_t rows, uint32_t job_count,
                           const mtl::Buffer& prepared, const mtl::Buffer& raw, const mtl::Buffer& output) const {
    constexpr uint32_t width = ExpertBank::width, blocks = width / 128;
    const size_t elements = size_t(rows) * width;
    if (!bank.ready() || prepare_.max_threads < 128 || project_.max_threads < 128 || finish_.max_threads < 128 ||
        rows == 0 || job_count == 0 || job_count > rows || !input.handle || input.bytes < width * sizeof(float) ||
        !row_map.handle || !jobs.handle || !prepared.handle || !raw.handle || !output.handle ||
        row_map.bytes < size_t(rows) * 8 ||
        jobs.bytes < size_t(job_count) * 16 || prepared.bytes < elements * sizeof(uint16_t) ||
        raw.bytes < elements * sizeof(uint16_t) || output.bytes < elements * sizeof(float)) return false;

    using mtl::Arg;
    const uint32_t shape[2] = {width, width};
    commands.push_back({&prepare_, {Arg::B(input), bank.suh.argument(), Arg::B(row_map), Arg::B(prepared), Arg::V(width)},
                        {blocks, rows, 1}, {128, 1, 1}});
    commands.push_back({&project_, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::B(jobs), Arg::V(shape)},
                        {blocks, job_count, 1}, {128, 1, 1}});
    commands.push_back({&finish_, {Arg::B(raw), bank.svh.argument(), Arg::B(row_map), Arg::B(output), Arg::V(width)},
                        {blocks, rows, 1}, {128, 1, 1}});
    return true;
}

bool ExpertProgram::append_routed(std::vector<mtl::Dispatch>& commands, const ExpertBank& gate,
                                  const ExpertBank& up, const ExpertBank& down, const mtl::Buffer& input,
                                  const mtl::Buffer& row_map, const mtl::Buffer& jobs, const mtl::Buffer& inverse,
                                  const mtl::Buffer& route_weights, uint32_t tokens, uint32_t rows, uint32_t job_count,
                                  const mtl::Buffer& prepared, const mtl::Buffer& gate_raw, const mtl::Buffer& up_raw,
                                  const mtl::Buffer& routed, const mtl::Buffer& output,
                                  const mtl::Buffer* projection_grid) const {
    constexpr uint32_t width = ExpertBank::width, blocks = width / 128, topk = 16, max_tokens = 4096;
    constexpr float clamp = 10.0f;
    const size_t positions = size_t(tokens) * width;
    const size_t slots = size_t(tokens) * topk;
    const size_t elements = size_t(rows) * width;
    const bool fused_finish = rows != 0 && fuses_finish_reduce(tokens);
    auto fits = [](const mtl::Buffer& buffer, size_t bytes) { return buffer.handle && buffer.bytes >= bytes; };
    if (tokens == 0 || tokens > max_tokens || rows > slots || job_count > rows || (rows != 0 && job_count == 0) ||
        combine_.max_threads < 128 || !fits(inverse, slots * sizeof(uint32_t)) ||
        !fits(route_weights, slots * sizeof(float)) || !fits(output, positions * sizeof(float)) ||
        (!fused_finish && !fits(routed, elements * sizeof(float))) ||
        (projection_grid && !fits(*projection_grid, 12))) return false;

    using mtl::Arg;
    if (rows != 0) {
        if (!gate.ready() || !up.ready() || !down.ready() || prepare_.max_threads < 128 ||
            project_.max_threads < 128 || finish_.max_threads < 128 || prepare_down_.max_threads < 128 ||
            !fits(input, positions * sizeof(float)) || !fits(row_map, size_t(rows) * 8) ||
            !fits(jobs, size_t(job_count) * 16) || !fits(prepared, elements * sizeof(uint16_t)) ||
            !fits(gate_raw, elements * sizeof(uint16_t)) || !fits(up_raw, elements * sizeof(uint16_t))) return false;
        const uint32_t shape[2] = {width, width};
        const bool use_decode = pair1_decode_enabled() && tokens == 1 && rows == topk && job_count == topk &&
                                projection_grid == nullptr;
        auto project = [&](const ExpertBank& bank, const mtl::Buffer& raw, bool use_gemv = false) {
            const auto& projection = use_gemv ? decode_ : project_;
            const uint32_t projection_width = use_gemv ? width / 32 : blocks;
            commands.push_back({&projection, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::B(jobs),
                                               Arg::V(shape)}, {projection_width, job_count, 1}, {128, 1, 1}});
            commands.back().indirect = projection_grid;
        };
        auto prepare = [&](const ExpertBank& bank) {
            commands.push_back({&prepare_, {Arg::B(input), bank.suh.argument(), Arg::B(row_map), Arg::B(prepared),
                                           Arg::V(width)}, {blocks, rows, 1}, {128, 1, 1}});
        };
        prepare(gate);
        project(gate, gate_raw);
        prepare(up);
        project(up, up_raw);
        commands.push_back({&prepare_down_, {Arg::B(gate_raw), Arg::B(up_raw), gate.svh.argument(),
                                            up.svh.argument(), down.suh.argument(), Arg::B(row_map),
                                            Arg::B(prepared), Arg::V(width), Arg::V(clamp)},
                            {blocks, rows, 1}, {128, 1, 1}});
        project(down, gate_raw, use_decode);
        if (fused_finish) {
            commands.push_back({&finish_reduce_, {Arg::B(gate_raw), down.svh.argument(), Arg::B(row_map),
                                                   Arg::B(inverse), Arg::B(route_weights), Arg::B(output), Arg::V(width)},
                                {blocks, tokens, 1}, {512, 1, 1}});
            return true;
        }
        commands.push_back({&finish_, {Arg::B(gate_raw), down.svh.argument(), Arg::B(row_map), Arg::B(routed),
                                       Arg::V(width)}, {blocks, rows, 1}, {128, 1, 1}});
    }
    commands.push_back({&combine_, {Arg::B(routed), Arg::B(inverse), Arg::B(route_weights), Arg::B(output), Arg::V(width)},
                        {blocks, tokens, 1}, {128, 1, 1}});
    return true;
}

bool ExpertProgram::append_dense(std::vector<mtl::Dispatch>& commands, const DenseBank& bank,
                                 const mtl::Buffer& input, const mtl::Buffer& row_map, uint32_t rows,
                                 const mtl::Buffer& prepared, const mtl::Buffer& raw, const mtl::Buffer& output) const {
    const auto& project = bank.format == Format::spec1_k4 ? dense_k4_ : dense_k6_;
    const size_t in_elements = size_t(rows) * bank.inputs, out_elements = size_t(rows) * bank.outputs;
    auto fits = [](const mtl::Buffer& buffer, size_t bytes) { return buffer.handle && buffer.bytes >= bytes; };
    if (!bank.ready() || rows == 0 || rows > 4096 || prepare_.max_threads < 128 || project.max_threads < 128 ||
        dense_finish_.max_threads < 128 || !fits(input, in_elements * sizeof(float)) ||
        !fits(row_map, size_t(rows) * 8) || !fits(prepared, in_elements * sizeof(uint16_t)) ||
        !fits(raw, out_elements * sizeof(uint16_t)) || !fits(output, out_elements * sizeof(float))) return false;

    uint32_t splits = 0;
    if (dense_split_.handle && bank.format == Format::spec1_k6 && rows <= 6) {
        // Fixed geometry from the real packed projection sweep at these row counts.
        constexpr uint32_t tested_shapes[][3] = {
            {1024, 8192, 4}, {1024, 16384, 4},
            {1536, 3072, 8},
            {3072, 1024, 16}, {3072, 1536, 16}, {3072, 8192, 3}, {3072, 16384, 4},
            {8192, 3072, 8}, {16384, 3072, 8},
        };
        for (const auto& [inputs, outputs, split_count] : tested_shapes) {
            if (bank.inputs == inputs && bank.outputs == outputs) {
                splits = split_count;
                break;
            }
        }
        if (raw.chunks || raw.bias % alignof(float) != 0 || !fits(raw, out_elements * splits * sizeof(float)))
            splits = 0;
    }

    using mtl::Arg;
    const uint32_t shape[4] = {bank.inputs, bank.outputs, rows, splits};
    commands.push_back({&prepare_, {Arg::B(input), bank.suh.argument(), Arg::B(row_map), Arg::B(prepared),
                                   Arg::V(bank.inputs)}, {bank.inputs / 128, rows, 1}, {128, 1, 1}});
    if (splits != 0) {
        commands.push_back({&dense_split_, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::V(shape)},
                            {bank.outputs / 128, splits, 1}, {128, 1, 1}});
        commands.push_back({&dense_split_finish_, {Arg::B(raw), bank.svh.argument(), Arg::B(output), Arg::V(shape)},
                            {bank.outputs / 128, rows, 1}, {128, 1, 1}});
        return true;
    }
    commands.push_back({&project, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::V(shape)},
                        {bank.outputs / 128, (rows + 31) / 32, 1}, {128, 1, 1}});
    commands.push_back({&dense_finish_, {Arg::B(raw), bank.svh.argument(), Arg::B(row_map), Arg::B(output),
                                        Arg::V(bank.outputs)}, {bank.outputs / 128, rows, 1}, {128, 1, 1}});
    return true;
}

}  // namespace stark::hy4::stq
