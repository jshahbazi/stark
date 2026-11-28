// SPDX-License-Identifier: MIT
// Adapted from Sushi v1.8.1 (b7b7975cc017acba1b4c0877d65543c4c5a4a381):
// src/exl3/expert_exl3.zig, codebook_tables.zig, and linear.zig.
// Sushi's EXL3 layout and Hadamard reference follows turboderp/exllamav3.
//
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "stq_codec.h"

#include <algorithm>
#include <array>
#include <bit>

namespace stark::hy4::stq {
namespace {

constexpr std::size_t tiles_per_block = block_size / tile_size;

// pair1-w16-v1 (c089b4eb), K4.
constexpr std::uint32_t pair1_multiplier = 0xe2c7fc0d;
constexpr int pair1_rotation = 19;
constexpr std::uint32_t pair1_mask0 = 0x8f408fa0;
constexpr std::uint32_t pair1_mask1 = 0x8f118fc9;

// spec1-w15 (37f72431), rows n=64 and n=96 respectively.
struct HashMask { std::uint32_t multiplier; std::uint32_t mask; };
constexpr HashMask spec1_k4{0x3fed4cef, 0x8ff08ff0};
constexpr HashMask spec1_k6{0xd0003751, 0x8ff98fff};

constexpr float codebook_scale = 4096.0f;
constexpr float hadamard_scale = 0.08838834764831845f;

float half_to_float(std::uint16_t bits) noexcept {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

std::uint16_t half_sum_scaled(std::uint32_t bits) noexcept {
    const float lo = half_to_float(static_cast<std::uint16_t>(bits));
    const float hi = half_to_float(static_cast<std::uint16_t>(bits >> 16));
    const auto sum = static_cast<_Float16>(lo + hi);
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(static_cast<float>(sum) * codebook_scale));
}

std::uint32_t word(std::span<const std::uint16_t> words, std::size_t i) noexcept {
    return words[2 * i] | (static_cast<std::uint32_t>(words[2 * i + 1]) << 16);
}

std::uint16_t codeword(std::span<const std::uint16_t> words, std::size_t bit_end) noexcept {
    const std::size_t current = (bit_end - 1) / 32;
    const std::size_t previous = current == 0 ? words.size() / 2 - 1 : current - 1;
    const std::size_t shift = (current + 1) * 32 - bit_end;
    // A 16-bit window ending at bit_end, with the first windows using the tile's tail.
    const auto bits = (static_cast<std::uint64_t>(word(words, previous)) << 32) | word(words, current);
    return static_cast<std::uint16_t>(bits >> shift);
}

constexpr std::array<std::size_t, tile_values> tile_permutation = [] {
    std::array<std::size_t, tile_values> permutation{};
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        const std::size_t lane = i / 8;
        const std::size_t element = i % 8;
        const std::size_t row = (lane % 4) * 2 + (element & 1) + (element & 2) * 4;
        const std::size_t column = lane / 4 + (element & 4) * 2;
        permutation[i] = row * tile_size + column;
    }
    return permutation;
}();

std::span<const std::uint16_t> tile_words(const TrellisView& trellis, std::size_t bank,
                                          std::size_t input_tile, std::size_t output_tile) noexcept {
    const std::size_t it = trellis.inputs / tile_size;
    const std::size_t ot = trellis.outputs / tile_size;
    const std::size_t n = tile_word_count(trellis.format);
    const std::size_t bank_base = bank * it * ot;
    const std::size_t tile = trellis.format == Format::pair1_k4
        ? (output_tile / 2 * it + input_tile) * 2 + output_tile % 2
        : input_tile * ot + output_tile;
    return trellis.words.subspan((bank_base + tile) * n, n);
}

void decode_tile_words(Format format, std::span<const std::uint16_t> words,
                       std::span<std::uint16_t, tile_values> tile) noexcept {
    if (format == Format::pair1_k4) {
        for (std::size_t i = 0; i < tile_values; i += 2) {
            const std::uint32_t product = codeword(words, (i + 2) * 4) * pair1_multiplier;
            tile[tile_permutation[i]] = half_sum_scaled(product & pair1_mask0);
            tile[tile_permutation[i + 1]] = half_sum_scaled(std::rotl(product, pair1_rotation) & pair1_mask1);
        }
        return;
    }
    const auto hash = format == Format::spec1_k4 ? spec1_k4 : spec1_k6;
    const std::size_t bits_per_weight = words.size() / tile_size;
    for (std::size_t i = 0; i < tile_values; ++i) {
        const std::uint32_t code = codeword(words, (i + 1) * bits_per_weight) & 0x7fff;
        tile[tile_permutation[i]] = half_sum_scaled((code * hash.multiplier) & hash.mask);
    }
}

}  // namespace

bool TrellisView::valid() const noexcept {
    const std::size_t n = tile_word_count(format);
    if (n == 0 || banks == 0 || inputs == 0 || outputs == 0 ||
        inputs % block_size != 0 || outputs % block_size != 0 || words.size() % n != 0) {
        return false;
    }
    // Validate using division so malformed dimensions cannot overflow the byte count.
    const std::size_t it = inputs / tile_size;
    const std::size_t ot = outputs / tile_size;
    const std::size_t total_tiles = words.size() / n;
    const std::size_t output_tiles = total_tiles / it;
    return total_tiles % it == 0 && output_tiles % ot == 0 && output_tiles / ot == banks;
}

bool decode_tile(Format format, std::span<const std::uint16_t> words,
                 std::span<std::uint16_t, tile_values> tile) noexcept {
    if (words.empty() || words.size() != tile_word_count(format)) return false;
    decode_tile_words(format, words, tile);
    return true;
}

bool decode_tile(const TrellisView& trellis, std::size_t bank, std::size_t input_tile,
                 std::size_t output_tile, std::span<std::uint16_t, tile_values> tile) noexcept {
    if (!trellis.valid() || bank >= trellis.banks ||
        input_tile >= trellis.inputs / tile_size || output_tile >= trellis.outputs / tile_size) {
        return false;
    }
    decode_tile_words(trellis.format, tile_words(trellis, bank, input_tile, output_tile), tile);
    return true;
}

void hadamard128(std::span<float, block_size> values) noexcept {
    std::array<float, block_size> transformed{};
    for (std::size_t input = 0; input < block_size; ++input) {
        const float weighted = values[input] * hadamard_scale;
        for (std::size_t output = 0; output < block_size; ++output) {
            transformed[output] += std::popcount(input & output) % 2 == 0 ? weighted : -weighted;
        }
    }
    std::copy(transformed.begin(), transformed.end(), values.begin());
}

bool decode_block(const TrellisView& trellis, std::size_t bank, std::size_t input_block,
                  std::size_t output_block, std::span<const std::uint16_t> suh,
                  std::span<const std::uint16_t> svh, std::span<float, block_values> block) noexcept {
    if (!trellis.valid() || bank >= trellis.banks || input_block >= trellis.inputs / block_size ||
        output_block >= trellis.outputs / block_size ||
        suh.size() % trellis.inputs != 0 || suh.size() / trellis.inputs != trellis.banks ||
        svh.size() % trellis.outputs != 0 || svh.size() / trellis.outputs != trellis.banks) {
        return false;
    }
    const std::size_t first_input = input_block * tiles_per_block;
    const std::size_t first_output = output_block * tiles_per_block;
    std::array<std::uint16_t, tile_values> tile;
    for (std::size_t i = 0; i < tiles_per_block; ++i) {
        for (std::size_t o = 0; o < tiles_per_block; ++o) {
            decode_tile_words(trellis.format, tile_words(trellis, bank, first_input + i, first_output + o), tile);
            for (std::size_t r = 0; r < tile_size; ++r) {
                const std::size_t offset = (i * tile_size + r) * block_size + o * tile_size;
                for (std::size_t c = 0; c < tile_size; ++c) {
                    block[offset + c] = half_to_float(tile[r * tile_size + c]);
                }
            }
        }
    }

    std::array<float, block_size> column;
    for (std::size_t o = 0; o < block_size; ++o) {
        for (std::size_t i = 0; i < block_size; ++i) column[i] = block[i * block_size + o];
        hadamard128(column);
        for (std::size_t i = 0; i < block_size; ++i) block[i * block_size + o] = column[i];
    }

    const std::size_t su_offset = bank * trellis.inputs + input_block * block_size;
    const std::size_t sv_offset = bank * trellis.outputs + output_block * block_size;
    for (std::size_t i = 0; i < block_size; ++i) {
        const auto row = std::span<float, block_size>(block.data() + i * block_size, block_size);
        hadamard128(row);
        const float su = half_to_float(suh[su_offset + i]);
        for (std::size_t o = 0; o < block_size; ++o) row[o] *= su * half_to_float(svh[sv_offset + o]);
    }
    return true;
}

}  // namespace stark::hy4::stq
