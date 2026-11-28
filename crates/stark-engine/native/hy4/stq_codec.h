// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace stark::hy4::stq {

inline constexpr std::size_t tile_size = 16;
inline constexpr std::size_t tile_values = tile_size * tile_size;
inline constexpr std::size_t block_size = 128;
inline constexpr std::size_t block_values = block_size * block_size;

enum class Format { pair1_k4, spec1_k4, spec1_k6 };

constexpr std::size_t tile_word_count(Format format) noexcept {
    switch (format) {
        case Format::pair1_k4:
        case Format::spec1_k4: return 64;
        case Format::spec1_k6: return 96;
    }
    return 0;
}

// Native little-endian U16 payload as halfwords. PAIR1 uses [banks, OT/2, IT, 2, 64].
// SPEC1 uses [banks, IT, OT, n]; a 3-D tensor has one bank. IT/OT count 16-wide tiles.
struct TrellisView {
    std::span<const std::uint16_t> words;
    Format format;
    std::size_t inputs;
    std::size_t outputs;
    std::size_t banks;

    bool valid() const noexcept;
};

// Decode to binary16 bits in row-major [input, output] order in the Hadamard domain.
[[nodiscard]] bool decode_tile(Format format, std::span<const std::uint16_t> words,
                               std::span<std::uint16_t, tile_values> tile) noexcept;
[[nodiscard]] bool decode_tile(const TrellisView& trellis, std::size_t bank,
                               std::size_t input_tile, std::size_t output_tile,
                               std::span<std::uint16_t, tile_values> tile) noexcept;

void hadamard128(std::span<float, block_size> values) noexcept;

// One public-weight block, in row-major [input, output] order, with the CPU
// reference's F32 arithmetic. suh and svh are F16 bits in [banks, features] order.
[[nodiscard]] bool decode_block(const TrellisView& trellis, std::size_t bank,
                                std::size_t input_block, std::size_t output_block,
                                std::span<const std::uint16_t> suh,
                                std::span<const std::uint16_t> svh,
                                std::span<float, block_values> block) noexcept;

}  // namespace stark::hy4::stq
