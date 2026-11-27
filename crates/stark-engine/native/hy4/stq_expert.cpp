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
                         bool split_spec1, bool wide_attention, bool wide_routed, bool wide_ffn,
                         bool fast_verify, bool register_dense_io) {
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
        project_k32_ = wide_routed || (split_spec1 && fast_verify) ?
            metal.pipeline(library, "stark_hy4_stq_pair1_k4_prefill") : mtl::Pipeline{};
        prepare_register_ = wide_routed || register_dense_io ?
            metal.pipeline(library, "stark_hy4_stq_pair1_prepare_register", boundary) : mtl::Pipeline{};
        prepare_down_register_ = wide_routed ?
            metal.pipeline(library, "stark_hy4_stq_pair1_prepare_down_register") : mtl::Pipeline{};
        finish_ = metal.pipeline(library, "stark_hy4_stq_pair1_finish");
        dense_finish_ = round_bf16 ? metal.pipeline(library, "stark_hy4_stq_pair1_finish", boundary) : finish_;
        if (register_dense_io) {
            dense_finish_register_ = metal.pipeline(library, "stark_hy4_stq_dense_finish_register", boundary);
            attention_gate_prepare_register_ = metal.pipeline(library, "stark_hy4_stq_attention_gate_prepare_register", boundary);
        }
        prepare_down_ = metal.pipeline(library, "stark_hy4_stq_pair1_prepare_down");
        combine_ = metal.pipeline(library, "stark_hy4_stq_pair1_combine");
        finish_reduce_ = fused_finish_reduce ? metal.pipeline(library, "stark_hy4_stq_pair1_finish_reduce") : mtl::Pipeline{};
        dense_k4_ = metal.pipeline(library, "stark_hy4_stq_spec1_k4_dense");
        dense_k6_ = metal.pipeline(library, "stark_hy4_stq_spec1_k6_dense");
        dense_split_ = split_spec1 ? metal.pipeline(library, "stark_hy4_stq_spec1_k6_split") : mtl::Pipeline{};
        dense_split_k32_ = split_spec1 && fast_verify ?
            metal.pipeline(library, "stark_hy4_stq_spec1_k6_split_k32") : mtl::Pipeline{};
        dense_split_finish_ = split_spec1 ? metal.pipeline(library, "stark_hy4_stq_spec1_split_finish", boundary) : mtl::Pipeline{};
        dense_prefill_ = wide_attention || wide_ffn ?
            metal.pipeline(library, "stark_hy4_stq_spec1_k6_prefill") : mtl::Pipeline{};
        ffn_prepare_down_ = wide_ffn ?
            metal.pipeline(library, "stark_hy4_stq_ffn_prepare_down", boundary) : mtl::Pipeline{};
        attention_gate_prepare_ = wide_attention ?
            metal.pipeline(library, "stark_hy4_stq_attention_gate_prepare", boundary) : mtl::Pipeline{};
        attention_gate_prepare_split_ = split_spec1 && fast_verify ?
            metal.pipeline(library, "stark_hy4_stq_attention_gate_prepare_split", boundary) : mtl::Pipeline{};
        decode_ = pair1_decode ? metal.pipeline(library, "stark_hy4_stq_pair1_k4_decode") : mtl::Pipeline{};
        return prepare_.max_threads >= 128 && project_.max_threads >= 128 && finish_.max_threads >= 128 &&
               prepare_down_.max_threads >= 128 && combine_.max_threads >= 128 &&
               dense_k4_.max_threads >= 128 && dense_k6_.max_threads >= 128 && dense_finish_.max_threads >= 128 &&
               (!(wide_routed || (split_spec1 && fast_verify)) ||
                   (project_k32_.max_threads >= 128 && project_k32_.simd_width == 32)) &&
               (!(wide_routed || register_dense_io) ||
                   (prepare_register_.max_threads >= 128 && prepare_register_.simd_width == 32)) &&
               (!wide_routed || (prepare_down_register_.max_threads >= 128 && prepare_down_register_.simd_width == 32)) &&
               (!register_dense_io ||
                   (dense_finish_register_.max_threads >= 128 && dense_finish_register_.simd_width == 32 &&
                    attention_gate_prepare_register_.max_threads >= 128 && attention_gate_prepare_register_.simd_width == 32)) &&
               (!pair1_decode || (decode_.max_threads >= 128 && decode_.simd_width == 32)) &&
               (!fused_finish_reduce || (finish_reduce_.max_threads >= 512 && finish_reduce_.simd_width == 32)) &&
               (!split_spec1 || (dense_split_.max_threads >= 128 && dense_split_.simd_width == 32 &&
                               dense_split_finish_.max_threads >= 128 && dense_split_finish_.simd_width == 32)) &&
               (!(wide_attention || wide_ffn) || (dense_prefill_.max_threads >= 128 && dense_prefill_.simd_width == 32)) &&
               (!wide_ffn || (ffn_prepare_down_.max_threads >= 128 && ffn_prepare_down_.simd_width == 32)) &&
               (!wide_attention || (attention_gate_prepare_.max_threads >= 128 &&
                                    attention_gate_prepare_.simd_width == 32)) &&
               (!(split_spec1 && fast_verify) || (attention_gate_prepare_split_.max_threads >= 128 &&
                                                 attention_gate_prepare_split_.simd_width == 32 &&
                                                 dense_split_k32_.max_threads >= 128 && dense_split_k32_.simd_width == 32));
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
    constexpr uint32_t min_wide_tokens = 1024;
    static_assert(width % 512 == 0);
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
        const bool wide = prepare_down_register_.handle && tokens >= min_wide_tokens;
        // Narrow indirect verification uses the same grouped layout. Keep
        // its existing input preparation and the separate direct GEMV.
        const bool wide_verify = dense_split_k32_.handle && projection_grid && tokens <= 6;
        const auto& grouped_project = wide || wide_verify ? project_k32_ : project_;
        const auto& prepare_input = wide ? prepare_register_ : prepare_;
        const auto& prepare_down = wide ? prepare_down_register_ : prepare_down_;
        const uint32_t prepare_blocks = wide ? blocks / 4 : blocks;
        auto project = [&](const ExpertBank& bank, const mtl::Buffer& raw, bool use_gemv = false) {
            const auto& projection = use_gemv ? decode_ : grouped_project;
            const uint32_t projection_width = use_gemv ? width / 32 : blocks;
            commands.push_back({&projection, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::B(jobs),
                                               Arg::V(shape)}, {projection_width, job_count, 1}, {128, 1, 1}});
            commands.back().indirect = projection_grid;
        };
        auto prepare = [&](const ExpertBank& bank) {
            commands.push_back({&prepare_input, {Arg::B(input), bank.suh.argument(), Arg::B(row_map), Arg::B(prepared),
                                               Arg::V(width)}, {prepare_blocks, rows, 1}, {128, 1, 1}});
        };
        prepare(gate);
        project(gate, gate_raw);
        prepare(up);
        project(up, up_raw);
        commands.push_back({&prepare_down, {Arg::B(gate_raw), Arg::B(up_raw), gate.svh.argument(),
                                           up.svh.argument(), down.suh.argument(), Arg::B(row_map),
                                           Arg::B(prepared), Arg::V(width), Arg::V(clamp)},
                            {prepare_blocks, rows, 1}, {128, 1, 1}});
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
                                 const mtl::Buffer& prepared, const mtl::Buffer& raw, const mtl::Buffer& output,
                                 DenseBoundaries boundaries, bool prefill) const {
    // Complete tiles use the wider K32 product on the verified shapes.
    // The same dispatch uses the incumbent K16 product on its partial tile.
    const auto attention_width = [](uint32_t width) { return width == 8192 || width == 16384; };
    const auto ffn_width = [](uint32_t width) { return width == 1536 || width == 13824; };
    const bool attention_shape = ((bank.inputs == 1024 || bank.inputs == 3072) && attention_width(bank.outputs)) ||
                                 (attention_width(bank.inputs) && bank.outputs == 3072);
    const bool ffn_shape = (bank.inputs == 3072 && ffn_width(bank.outputs)) ||
                           (ffn_width(bank.inputs) && bank.outputs == 3072);
    const bool wide_prefill = prefill && dense_prefill_.handle && bank.format == Format::spec1_k6 &&
                              rows >= 1024 && ((attention_gate_prepare_.handle && attention_shape) ||
                                               (ffn_prepare_down_.handle && ffn_shape));
    const auto& project = bank.format == Format::spec1_k4 ? dense_k4_ : wide_prefill ? dense_prefill_ : dense_k6_;
    const bool needs_prepare = boundaries != DenseBoundaries::prepared_input;
    const bool needs_finish = boundaries != DenseBoundaries::raw_output && boundaries != DenseBoundaries::split_output;
    const size_t in_elements = size_t(rows) * bank.inputs, out_elements = size_t(rows) * bank.outputs;
    auto fits = [](const mtl::Buffer& buffer, size_t bytes) { return buffer.handle && buffer.bytes >= bytes; };
    if (!bank.ready() || rows == 0 || rows > 4096 || project.max_threads < 128 ||
        (needs_prepare && (prepare_.max_threads < 128 || !fits(input, in_elements * sizeof(float)))) ||
        (needs_finish && (dense_finish_.max_threads < 128 || !fits(output, out_elements * sizeof(float)))) ||
        !fits(row_map, size_t(rows) * 8) || !fits(prepared, in_elements * sizeof(uint16_t)) ||
        !fits(raw, out_elements * sizeof(uint16_t))) return false;

    const uint32_t splits = boundaries == DenseBoundaries::raw_output ? 0 : dense_splits(bank, rows, raw);
    if (boundaries == DenseBoundaries::split_output && !splits) return false;

    using mtl::Arg;
    const uint32_t shape[4] = {bank.inputs, bank.outputs, rows, splits};
    const bool register_io = prefill && rows >= 1024 && dense_finish_register_.handle;
    if (needs_prepare) {
        const bool use_register = register_io && bank.inputs % 512 == 0;
        const auto& prepare = use_register ? prepare_register_ : prepare_;
        commands.push_back({&prepare, {Arg::B(input), bank.suh.argument(), Arg::B(row_map), Arg::B(prepared),
                                      Arg::V(bank.inputs)}, {bank.inputs / (use_register ? 512u : 128u), rows, 1}, {128, 1, 1}});
    }
    if (splits != 0) {
        const bool narrow_k32 = dense_split_k32_.handle && bank.inputs % (splits * 32) == 0 &&
            (attention_shape || (bank.inputs == 3072 && bank.outputs == 1024));
        const auto& split_project = narrow_k32 ? dense_split_k32_ : dense_split_;
        commands.push_back({&split_project, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::V(shape)},
                            {bank.outputs / 128, splits, 1}, {128, 1, 1}});
        if (needs_finish) {
            commands.push_back({&dense_split_finish_, {Arg::B(raw), bank.svh.argument(), Arg::B(output), Arg::V(shape)},
                                {bank.outputs / 128, rows, 1}, {128, 1, 1}});
        }
        return true;
    }
    commands.push_back({&project, {bank.trellis.argument(), Arg::B(prepared), Arg::B(raw), Arg::V(shape)},
                        {bank.outputs / 128, (rows + 31) / 32, 1}, {128, 1, 1}});
    if (needs_finish) {
        const bool use_register = register_io && bank.outputs % 512 == 0;
        const auto& finish = use_register ? dense_finish_register_ : dense_finish_;
        commands.push_back({&finish, {Arg::B(raw), bank.svh.argument(), Arg::B(row_map), Arg::B(output),
                                     Arg::V(bank.outputs)}, {bank.outputs / (use_register ? 512u : 128u), rows, 1}, {128, 1, 1}});
    }
    return true;
}

uint32_t ExpertProgram::dense_splits(const DenseBank& bank, uint32_t rows, const mtl::Buffer& raw) const {
    if (!dense_split_.handle || bank.format != Format::spec1_k6 || rows > 6 ||
        !raw.handle || raw.chunks || raw.bias % alignof(float) != 0) return 0;
    // Fixed geometry from the real packed projection sweep at these row counts.
    constexpr uint32_t tested_shapes[][3] = {
        {1024, 8192, 4}, {1024, 16384, 4},
        {1536, 3072, 8},
        {3072, 1024, 16}, {3072, 1536, 16}, {3072, 8192, 3}, {3072, 16384, 4},
        {8192, 3072, 8}, {16384, 3072, 8},
    };
    for (const auto& [inputs, outputs, splits] : tested_shapes) {
        if (bank.inputs == inputs && bank.outputs == outputs &&
            raw.bytes >= size_t(rows) * outputs * splits * sizeof(float)) return splits;
    }
    return 0;
}

uint32_t ExpertProgram::verify_attention_gate_splits(const DenseBank& gate, const DenseBank& output,
                                                     uint32_t rows, const mtl::Buffer& raw) const {
    if (!attention_gate_prepare_split_.handle || !dense_splits(output, rows, raw)) return 0;
    return dense_splits(gate, rows, raw);
}

bool ExpertProgram::append_attention_gate_prepare(std::vector<mtl::Dispatch>& commands,
                                                  const DenseBank& gate_bank, const DenseBank& output_bank,
                                                  uint32_t rows, const mtl::Buffer& raw, const mtl::Buffer& value,
                                                  const mtl::Buffer& prepared, uint32_t splits) const {
    const uint32_t width = gate_bank.outputs;
    const size_t elements = size_t(rows) * width;
    const bool available = splits ? splits == verify_attention_gate_splits(gate_bank, output_bank, rows, raw)
                                  : fuses_attention_gate(rows);
    if (!available || !gate_bank.ready() || !output_bank.ready() ||
        gate_bank.format != Format::spec1_k6 || output_bank.format != Format::spec1_k6 ||
        width != output_bank.inputs || width == 0 || width % 128 != 0 ||
        !raw.handle || raw.bytes < elements * sizeof(uint16_t) ||
        !value.handle || value.bytes < elements * sizeof(float) ||
        !prepared.handle || prepared.bytes < elements * sizeof(uint16_t)) return false;

    using mtl::Arg;
    const bool use_register = !splits && attention_gate_prepare_register_.handle && width % 512 == 0;
    const auto& prepare = use_register ? attention_gate_prepare_register_ : attention_gate_prepare_;
    mtl::Dispatch command{&prepare,
                          {Arg::B(raw), gate_bank.svh.argument(), Arg::B(value),
                           output_bank.suh.argument(), Arg::B(prepared)},
                          {width / (use_register ? 512u : 128u), rows, 1}, {128, 1, 1}};
    if (splits) {
        const uint32_t shape[] = {gate_bank.inputs, width, rows, splits};
        command.p = &attention_gate_prepare_split_;
        command.args.push_back(Arg::V(shape));
    } else {
        command.args.push_back(Arg::V(width));
    }
    commands.push_back(std::move(command));
    return true;
}

bool ExpertProgram::append_ffn(std::vector<mtl::Dispatch>& commands, const DenseBank& gate, const DenseBank& up,
                               const DenseBank& down, const mtl::Buffer& input, const mtl::Buffer& row_map,
                               uint32_t rows, const mtl::Buffer& prepared, const mtl::Buffer& gate_raw,
                               const mtl::Buffer& raw, const mtl::Buffer& output) const {
    const uint32_t width = gate.outputs;
    if (!fuses_ffn(rows) || !down.ready() || gate.inputs != up.inputs || gate.inputs != down.outputs ||
        width != up.outputs || width != down.inputs || width == 0 || width % 512 != 0) return false;
    if (!append_dense(commands, gate, input, row_map, rows, prepared, gate_raw, output,
                      DenseBoundaries::raw_output, true) ||
        !append_dense(commands, up, input, row_map, rows, prepared, raw, output,
                      DenseBoundaries::raw_output, true)) return false;

    using mtl::Arg;
    commands.push_back({&ffn_prepare_down_, {Arg::B(gate_raw), Arg::B(raw), gate.svh.argument(), up.svh.argument(),
                                             down.suh.argument(), Arg::B(prepared), Arg::V(width)},
                        {width / 512, rows, 1}, {128, 1, 1}});
    return append_dense(commands, down, input, row_map, rows, prepared, raw, output,
                        DenseBoundaries::prepared_input, true);
}

}  // namespace stark::hy4::stq
