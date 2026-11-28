// SPDX-License-Identifier: Apache-2.0
#include "stq_vision.h"

#include "vision.h"

#include <algorithm>
#include <bit>
#include <functional>
#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace stark::hy4::stq {
namespace {

constexpr uint64_t hidden = 1152, ffn = 4304, output = 3072, layers = 27;
constexpr uint64_t patch = 16, merge = 2, edge = 128, cells = hidden * merge * merge;

struct Layout {
    std::string source;
    std::string alias;
    std::vector<uint64_t> shape;  // GGUF order, with the innermost source axis first.
    bool widen;
    uint64_t skip = 0;
};

std::vector<Layout> vision_layout() {
    std::vector<Layout> layout{
        {"vit.embeddings.patch_embedding.weight", "v.patch_embd.weight", {patch, patch, 3, hidden}, false},
        {"vit.embeddings.patch_embedding.bias", "v.patch_embd.bias", {hidden}, true},
        {"vit.embeddings.position_embedding.weight", "v.position_embd.weight", {hidden, edge * edge}, true, hidden},
        {"vit.perceive.image_newline", "v.image_newline", {output}, true},
        {"vit.perceive.norm.weight", "mm.pre_norm.weight", {hidden}, true},
        {"vit.perceive.proj.0.weight", "mm.0.weight", {merge, merge, hidden, cells}, false},
        {"vit.perceive.proj.0.bias", "mm.0.bias", {cells}, true},
        {"vit.perceive.proj.2.weight", "mm.2.weight", {1, 1, cells, output}, false},
        {"vit.perceive.proj.2.bias", "mm.2.bias", {output}, true},
    };
    const Layout block[]{
        {"input_layernorm.weight", "ln1.weight", {hidden}, true},
        {"input_layernorm.bias", "ln1.bias", {hidden}, true},
        {"post_attention_layernorm.weight", "ln2.weight", {hidden}, true},
        {"post_attention_layernorm.bias", "ln2.bias", {hidden}, true},
        {"self_attn.q_proj.weight", "attn_q.weight", {hidden, hidden}, false},
        {"self_attn.q_proj.bias", "attn_q.bias", {hidden}, true},
        {"self_attn.k_proj.weight", "attn_k.weight", {hidden, hidden}, false},
        {"self_attn.k_proj.bias", "attn_k.bias", {hidden}, true},
        {"self_attn.v_proj.weight", "attn_v.weight", {hidden, hidden}, false},
        {"self_attn.v_proj.bias", "attn_v.bias", {hidden}, true},
        {"self_attn.o_proj.weight", "attn_out.weight", {hidden, hidden}, false},
        {"self_attn.o_proj.bias", "attn_out.bias", {hidden}, true},
        {"mlp.dense_h_to_4h.weight", "ffn_up.weight", {hidden, ffn}, false},
        {"mlp.dense_h_to_4h.bias", "ffn_up.bias", {ffn}, true},
        {"mlp.dense_4h_to_h.weight", "ffn_down.weight", {ffn, hidden}, false},
        {"mlp.dense_4h_to_h.bias", "ffn_down.bias", {hidden}, true},
    };
    for (uint64_t layer = 0; layer < layers; ++layer) {
        const std::string index = std::to_string(layer) + ".";
        for (const auto& tensor : block) {
            layout.push_back({"vit.layers." + index + tensor.source, "v.blk." + index + tensor.alias,
                              tensor.shape, tensor.widen});
        }
    }
    std::sort(layout.begin(), layout.end(), [](const auto& a, const auto& b) { return a.source < b.source; });
    return layout;
}

}  // namespace

std::string open_vision(strata::hy4_flash::VisionShard& shard, mtl::Context& metal,
                        std::span<const TensorSlice> slices) {
    if (shard.loaded() || shard.tensors.map != nullptr) return "STQ vision is already open";
    const auto layout = vision_layout();
    if (slices.size() != layout.size()) return "STQ vision tensor count differs from the pinned pack";

    const int fd = slices.front().fd;
    struct stat info{};
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0) return "invalid STQ vision descriptor";
    const auto length = static_cast<uint64_t>(info.st_size);
    std::vector<TensorSlice> ranges(slices.begin(), slices.end());
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
    const uint64_t first = ranges.front().offset;
    uint64_t end = first;
    for (const auto& slice : ranges) {
        if (slice.fd != fd || slice.offset != end || slice.offset % 4 != 0 ||
            slice.offset > length || slice.size > length - slice.offset) return "invalid STQ vision tensor range";
        end += slice.size;
    }
    const int page_size = getpagesize();
    if (page_size <= 0) return "invalid vision mapping page size";
    const auto page = static_cast<uint64_t>(page_size);
    const uint64_t file_start = first / page * page;
    const uint64_t mapped_bytes = (end - file_start + page - 1) / page * page;
    if (mapped_bytes > metal.max_buffer_length()) return "STQ vision mapping exceeds Metal buffer capacity";

    std::map<std::string, gguf::Tensor> directory;
    uint64_t expanded_bytes = 0;
    for (std::size_t i = 0; i < layout.size(); ++i) {
        const auto& format = layout[i];
        const auto count = std::accumulate(format.shape.begin(), format.shape.end(), uint64_t(1), std::multiplies<>());
        if (slices[i].size != (count + format.skip) * sizeof(uint16_t)) {
            return "wrong BF16 extent for STQ vision tensor " + format.source;
        }
        const uint64_t bytes = count * (format.widen ? sizeof(float) : sizeof(uint16_t));
        const uint64_t offset = format.widen ? expanded_bytes : slices[i].offset - file_start;
        directory.emplace(format.alias, gguf::Tensor{format.alias, format.shape, format.widen ? 0 : 30, offset, bytes});
        if (format.widen) expanded_bytes += bytes;
    }

    void* address = mmap(nullptr, mapped_bytes, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(file_start));
    if (address == MAP_FAILED) return "could not map the STQ vision tensor range";
    mtl::Buffer weights, expanded;
    try {
        weights = metal.wrap(address, mapped_bytes);
        expanded = metal.alloc(expanded_bytes);
    } catch (...) {
        // These allocations are unwound together so a caller can retry safely.
    }
    if (!weights.handle || !expanded.handle) {
        if (expanded.handle) metal.free(expanded);
        if (weights.handle) metal.free(weights);
        munmap(address, mapped_bytes);
        return "could not allocate STQ vision buffers";
    }

    // The position grid is the only widened tensor with a skipped prefix.
    // Source row 0 is the unused class embedding, as in Sushi hy4_vision.zig.
    const auto* source = static_cast<const uint8_t*>(address);
    auto* target = expanded.as<float>();
    for (std::size_t i = 0; i < layout.size(); ++i) {
        const auto& format = layout[i];
        if (!format.widen) continue;
        const auto& tensor = directory.find(format.alias)->second;
        const auto* bf16 = reinterpret_cast<const uint16_t*>(source + slices[i].offset - file_start) + format.skip;
        auto* dst = target + tensor.offset / sizeof(float);
        for (std::size_t j = 0; j < tensor.bytes / sizeof(float); ++j) {
            dst[j] = std::bit_cast<float>(static_cast<uint32_t>(bf16[j]) << 16);
        }
    }

    shard.tensors.map = address;
    shard.tensors.map_bytes = mapped_bytes;
    shard.tensors.file_bytes = end - file_start;
    shard.tensors.tensors.swap(directory);
    shard.weights = std::move(weights);
    shard.expanded = std::move(expanded);
    shard.geo = {.hidden = hidden, .heads = 16, .head_dim = hidden / 16, .ffn = ffn, .layers = layers,
                 .patch = patch, .merge = merge, .out_hidden = output, .pos_edge = edge, .eps = 1.0e-5f,
                 .min_pixels = 4096, .max_pixels = 4194304, .image_token = 120120};
    shard.projector_tanh = true;
    return {};
}

}  // namespace stark::hy4::stq
