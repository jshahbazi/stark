// SPDX-License-Identifier: Apache-2.0
#include "stq_attention.h"

#include <array>
#include <bit>
#include <cmath>

namespace stark::hy4::stq {

bool split_kv_b(const TrellisView& weight, std::size_t heads,
                std::span<const uint16_t> suh, std::span<const uint16_t> svh,
                std::span<uint16_t> key, std::span<uint16_t> value) noexcept {
    constexpr std::size_t latent = 512, nope = 192, v = 256;
    constexpr std::size_t head_rows = nope + v;
    if ((heads != 32 && heads != 64) || weight.banks != 1 || weight.format != Format::spec1_k6 ||
        weight.inputs != latent || weight.outputs != heads * head_rows || !weight.valid() ||
        suh.size() != weight.inputs || svh.size() != weight.outputs ||
        key.size() != heads * latent * nope || value.size() != heads * latent * v) {
        return false;
    }

    std::array<float, block_values> tile;
    for (std::size_t input = 0; input < weight.inputs; input += block_size) {
        for (std::size_t output = 0; output < weight.outputs; output += block_size) {
            if (!decode_block(weight, 0, input / block_size, output / block_size, suh, svh, tile)) return false;
            for (std::size_t r = 0; r < block_size; ++r) {
                for (std::size_t c = 0; c < block_size; ++c) {
                    const float x = tile[r * block_size + c];
                    if (!std::isfinite(x)) return false;
                    const uint32_t bits = std::bit_cast<uint32_t>(x);
                    const uint16_t bf16 = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
                    const std::size_t head = (output + c) / head_rows;
                    const std::size_t within = (output + c) % head_rows;
                    const std::size_t channel = input + r;
                    if (within < nope) {
                        key[(head * latent + channel) * nope + within] = bf16;
                    } else {
                        value[(head * v + within - nope) * latent + channel] = bf16;
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace stark::hy4::stq
