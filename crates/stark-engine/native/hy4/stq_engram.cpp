// SPDX-License-Identifier: Apache-2.0
#include "stq_engram.h"

#include "model.h"
#include "ref_quant.h"

#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>

namespace stark::hy4::stq {

std::string bind_engram(strata::hy4_flash::Model& model, std::span<const TensorSlice> tables,
                        std::span<const int32_t> token_map) {
    const auto& geometry = model.geo;
    constexpr uint64_t row_bytes = 192;
    if (geometry.engram_table_type != ref::BF16 || geometry.engram_row_bytes != row_bytes ||
        tables.size() != 4 || geometry.engram_tables.size() != tables.size() || geometry.vocab_size != 120832 ||
        token_map.size() != static_cast<size_t>(geometry.vocab_size) || !model.engram_segments.empty()) {
        return "invalid STQ Engram geometry";
    }
    if (std::any_of(token_map.begin(), token_map.end(), [&](int32_t id) { return id < 0 || id >= geometry.vocab_size; }))
        return "invalid STQ Engram token map";

    std::vector<strata::hy4_flash::EngramSegment> segments;
    segments.reserve(tables.size());
    for (size_t index = 0; index < tables.size(); ++index) {
        const auto& slice = tables[index];
        const auto& geometry_table = geometry.engram_tables[index];
        const int flags = fcntl(slice.fd, F_GETFL);
        struct stat info {};
        if (flags < 0 || (flags & O_ACCMODE) != O_RDONLY ||
            fstat(slice.fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0)
            return "invalid STQ Engram file descriptor";
        const auto bytes = static_cast<uint64_t>(info.st_size);
        if (slice.offset > bytes || slice.size > bytes - slice.offset ||
            slice.offset % alignof(uint16_t) != 0 || slice.size % row_bytes != 0 ||
            slice.size / row_bytes != geometry_table.rows)
            return "invalid STQ Engram tensor span";
        segments.push_back({geometry_table.block_id, "/dev/fd/" + std::to_string(slice.fd), slice.offset,
                            geometry_table.rows});
    }

    std::vector<int32_t> map(token_map.begin(), token_map.end());
    model.engram_segments = std::move(segments);
    model.engram_token_map = std::move(map);
    model.engram_row_bytes = row_bytes;
    model.engram_rows = geometry.engram_total_rows;
    return {};
}

}  // namespace stark::hy4::stq
