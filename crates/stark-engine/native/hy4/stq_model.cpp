// SPDX-License-Identifier: Apache-2.0
#include "stq_model.h"
#include "model.h"
#include "stq_attention.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <limits>
#include <unistd.h>

namespace stark::hy4::stq {
namespace {

bool read(TensorSlice slice, void* target, uint64_t bytes) {
    if (slice.size != bytes || slice.offset > uint64_t(INT64_MAX) ||
        bytes > uint64_t(INT64_MAX) - slice.offset) return false;
    auto* dst = static_cast<uint8_t*>(target);
    while (bytes != 0) {
        const ssize_t n = pread(slice.fd, dst, std::min(bytes, uint64_t(16 << 20)), slice.offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        dst += n;
        slice.offset += n;
        bytes -= n;
    }
    return true;
}

bool load_raw(const WeightSource& weight, const gguf::Tensor& tensor, void* dst) {
    const bool widen = weight.use == WeightUse::widen;
    uint64_t left = widen ? tensor.bytes / 2 : tensor.bytes;
    auto* at = static_cast<uint8_t*>(dst);
    for (const auto& piece : std::span(weight.pieces, weight.piece_count)) {
        if (!piece.size || piece.size > left || (widen && (piece.size % 2))) return false;
        if (widen) {
            std::vector<uint16_t> data(piece.size / 2);
            if (!read(piece, data.data(), piece.size)) return false;
            auto* out = reinterpret_cast<float*>(at);
            for (size_t i = 0; i < data.size(); ++i)
                out[i] = std::bit_cast<float>(uint32_t(data[i]) << 16);
            at += piece.size * 2;
        } else {
            if (!read(piece, at, piece.size)) return false;
            at += piece.size;
        }
        left -= piece.size;
    }
    return left == 0;
}

bool load_absorbed(const WeightSource& weight, const gguf::Tensor& key, const gguf::Tensor& value,
                   void* dense) {
    const auto& p = weight.projection;
    if ((p.rate != 4 && p.rate != 6) || p.inputs != 512 || key.shape.size() != 3 || value.shape.size() != 3)
        return false;
    const auto format = p.rate == 4 ? Format::spec1_k4 : Format::spec1_k6;
    const size_t heads = key.shape[2];
    if ((heads != 32 && heads != 64) || p.outputs != heads * (192 + 256) ||
        key.shape != std::vector<uint64_t>{192, 512, heads} ||
        value.shape != std::vector<uint64_t>{512, 256, heads} ||
        p.trellis.size != size_t(p.inputs / 16) * (p.outputs / 16) * tile_word_count(format) * sizeof(uint16_t))
        return false;
    std::vector<uint16_t> words(p.trellis.size / 2), suh(p.inputs), svh(p.outputs);
    if (!read(p.trellis, words.data(), words.size() * 2) ||
        !read(p.suh, suh.data(), suh.size() * 2) || !read(p.svh, svh.data(), svh.size() * 2)) return false;
    auto plane = [&](const gguf::Tensor& tensor) {
        return std::span(static_cast<uint16_t*>(dense) + tensor.offset / 2, tensor.bytes / 2);
    };
    return split_kv_b({words, format, p.inputs, p.outputs, 1}, heads, suh, svh, plane(key), plane(value));
}

} // namespace

bool Trunk::begin_residency(mtl::Context& metal) const {
    std::vector<mtl::Buffer> buffers;
    buffers.reserve(3 * (experts.size() * 3 + dense.size()));
    const auto add = [&](const auto& bank) {
        buffers.push_back(bank.trellis.buffer());
        buffers.push_back(bank.suh.buffer());
        buffers.push_back(bank.svh.buffer());
    };
    for (const auto& block : experts)
        for (const auto& bank : block) add(bank);
    for (const auto& entry : dense) add(*entry.second);
    std::vector<std::span<const uint8_t>> pages;
    pages.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        if (!buffer.ptr || !buffer.bytes) return false;
        pages.emplace_back(buffer.as<uint8_t>(), buffer.bytes);
    }
    return strata::metal_engine::prepare_mapped_pages(pages) && metal.begin_expert_residency(buffers);
}

std::string load_trunk(strata::hy4_flash::Model& model, mtl::Context& metal,
                       std::span<const WeightSource> weights,
                       std::span<const ProjectionSource> experts, bool round_bf16, bool pair1_decode,
                       bool fused_finish_reduce, bool split_spec1, bool wide_attention, bool wide_routed, bool wide_ffn,
                       bool fast_verify) {
    constexpr size_t trunk_weights = 1256, draft_weights = 92;
    if ((weights.size() != trunk_weights && weights.size() != trunk_weights + draft_weights) ||
        experts.size() != 123 || model.stq) return "invalid STQ weight inventory";
    auto trunk = std::make_unique<Trunk>();
    trunk->has_dfly = weights.size() == trunk_weights + draft_weights;
    uint64_t bytes = 0;
    for (const auto& weight : weights) {
        if (!weight.alias || !weight.rank || weight.rank > 3) return "invalid STQ weight descriptor";
        gguf::Tensor tensor;
        tensor.name = weight.alias;
        tensor.shape.assign(weight.shape, weight.shape + weight.rank);
        tensor.type = weight.type;
        uint64_t elements = 1;
        for (uint64_t dim : tensor.shape) {
            if (!dim || elements > UINT32_MAX / dim) return "invalid STQ weight dimensions";
            elements *= dim;
        }
        if (weight.use != WeightUse::linear) {
            if (weight.type != 0 && weight.type != 30) return "invalid STQ raw tensor type";
            tensor.bytes = elements * (weight.type == 0 ? 4 : 2);
            tensor.offset = bytes;
            bytes += (tensor.bytes + 63) & ~uint64_t(63);
        }
        if (!model.tensors.tensors.emplace(weight.alias, std::move(tensor)).second)
            return "duplicate STQ alias";
    }
    if (!bytes || bytes > metal.max_buffer_length()) return "STQ dense allocation exceeds the Metal limit";
    model.dense = metal.alloc(bytes);
    if (!model.dense.ptr) return "STQ dense allocation failed";
    model.dense_bytes = bytes;
    for (const auto& weight : weights) {
        const auto& tensor = *model.find(weight.alias);
        const auto& p = weight.projection;
        switch (weight.use) {
        case WeightUse::raw:
        case WeightUse::widen:
            if (!load_raw(weight, tensor, static_cast<uint8_t*>(model.dense.ptr) + tensor.offset))
                return std::string("cannot read STQ tensor ") + weight.alias;
            break;
        case WeightUse::absorbed_key: {
            const auto* value = model.find(tensor.name.substr(0, tensor.name.size() - (sizeof("attn_k_b.weight") - 1))
                                          + "attn_v_b.weight");
            if (!value || !load_absorbed(weight, tensor, *value, model.dense.ptr))
                return std::string("cannot split STQ absorbed attention ") + weight.alias;
            break;
        }
        case WeightUse::absorbed_value:
            break;
        case WeightUse::linear: {
            auto bank = std::make_unique<DenseBank>();
            if ((p.rate != 4 && p.rate != 6) || tensor.shape != std::vector<uint64_t>{p.inputs, p.outputs} ||
                !bank->open(metal, p.rate == 6 ? Format::spec1_k6 : Format::spec1_k4,
                            p.inputs, p.outputs, p.trellis, p.suh, p.svh))
                return std::string("cannot map STQ projection ") + weight.alias;
            trunk->dense.emplace(tensor.name, std::move(bank));
            break;
        }
        default:
            return "invalid STQ weight use";
        }
    }
    for (size_t i = 0; i < experts.size(); ++i) {
        const auto& p = experts[i];
        if (p.inputs != ExpertBank::width || p.outputs != ExpertBank::width || p.rate != 4 ||
            !trunk->experts[i / 3][i % 3].open(metal, p.trellis, p.suh, p.svh))
            return "cannot map STQ expert bank " + std::to_string(i);
    }
    if (!trunk->program.open(metal, round_bf16, pair1_decode, fused_finish_reduce, split_spec1,
                             wide_attention, wide_routed, wide_ffn, fast_verify) ||
        !trunk->embedding.open(metal))
        return "cannot compile STQ Metal projections";
    model.stq = std::move(trunk);
    return {};
}
} // namespace stark::hy4::stq
