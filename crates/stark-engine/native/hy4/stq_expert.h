// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mtl_rt.h"
#include "stq_codec.h"
#include <array>
#include <optional>
#include <span>

namespace stark::hy4::stq {

struct TensorSlice {
    int fd;
    uint64_t offset;
    uint64_t size;
};

/// Pins the exact read-only file range for the lifetime of its Metal buffer.
/// The caller keeps the verified artifact immutable and waits for GPU work before destruction.
class MappedPlane {
public:
    MappedPlane() = default;
    ~MappedPlane();
    MappedPlane(const MappedPlane&) = delete;
    MappedPlane& operator=(const MappedPlane&) = delete;

    bool open(mtl::Context& metal, TensorSlice slice, uint64_t expected_bytes);
    void close();
    mtl::Arg argument() const { return mtl::Arg::B(buffer_, bias_); }
    const mtl::Buffer& buffer() const { return buffer_; }
    bool ready() const { return metal_ != nullptr; }

private:
    mtl::Context* metal_ = nullptr;
    mtl::Buffer buffer_;
    size_t bias_ = 0;
};

struct ExpertBank {
    static constexpr uint32_t width = 1536, experts = 512;
    MappedPlane trellis, suh, svh;

    bool open(mtl::Context& metal, TensorSlice words, TensorSlice input_scale, TensorSlice output_scale);
    bool ready() const { return trellis.ready() && suh.ready() && svh.ready(); }
};

struct DenseBank {
    Format format = Format::spec1_k6;
    uint32_t inputs = 0, outputs = 0;
    MappedPlane trellis, suh, svh;

    bool open(mtl::Context& metal, Format rate, uint32_t input_width, uint32_t output_width,
              TensorSlice words, TensorSlice input_scale, TensorSlice output_scale);
    bool ready() const { return trellis.ready() && suh.ready() && svh.ready(); }
};

struct ExpertRoutes {
    std::vector<std::array<uint32_t, 2>> rows;
    std::vector<std::array<uint32_t, 4>> jobs;
    std::vector<uint32_t> inverse;
};

// Stable expert grouping of the 16 original router slots per row. -1 is
// the invalid-route sentinel kept out of the grouped projection.
std::optional<ExpertRoutes> plan_routes(std::span<const int32_t> expert_ids);

class ExpertProgram {
public:
    // Round incoming activations for both formats and final dense outputs to BF16.
    bool open(mtl::Context& metal, bool round_bf16 = false, bool pair1_decode = false,
              bool fused_finish_reduce = false, bool split_spec1 = false);
    bool pair1_decode_enabled() const { return decode_.handle != nullptr; }
    static constexpr uint32_t kFusedFinishMinTokens = 1024;
    bool fuses_finish_reduce(uint32_t tokens) const {
        return finish_reduce_.handle && tokens >= kFusedFinishMinTokens;
    }

    // Appends one PAIR1 projection to the caller's encoder. Each sorted row has
    // `(expert, source_row)` and each job has `(expert, first_row, count, 0)`.
    // Job spans are disjoint and cover the sorted rows, with up to 32 per job.
    bool append(std::vector<mtl::Dispatch>& commands, const ExpertBank& bank, const mtl::Buffer& input,
                const mtl::Buffer& row_map, const mtl::Buffer& jobs, uint32_t rows, uint32_t job_count,
                const mtl::Buffer& prepared, const mtl::Buffer& raw, const mtl::Buffer& output) const;

    // A direct single-token launch with exactly 16 rows and jobs proves one
    // row per expert and is eligible for the opt-in GEMV. Indirect callers
    // can supply a job upper bound and always use the grouped MMA grid.
    bool append_routed(std::vector<mtl::Dispatch>& commands, const ExpertBank& gate, const ExpertBank& up,
                       const ExpertBank& down, const mtl::Buffer& input, const mtl::Buffer& row_map,
                       const mtl::Buffer& jobs, const mtl::Buffer& inverse, const mtl::Buffer& route_weights,
                       uint32_t tokens, uint32_t rows, uint32_t job_count, const mtl::Buffer& prepared,
                       const mtl::Buffer& gate_raw, const mtl::Buffer& up_raw, const mtl::Buffer& routed,
                       const mtl::Buffer& output, const mtl::Buffer* projection_grid = nullptr) const;

    // The row map contains (0, row) for each source row. The head may submit a
    // short window without materializing logits for the rest of the prompt.
    // Narrow split-K uses raw as F32 partial planes when it has room.
    bool append_dense(std::vector<mtl::Dispatch>& commands, const DenseBank& bank, const mtl::Buffer& input,
                      const mtl::Buffer& row_map, uint32_t rows, const mtl::Buffer& prepared,
                      const mtl::Buffer& raw, const mtl::Buffer& output) const;

private:
    mtl::Pipeline prepare_, project_, decode_, finish_, prepare_down_, combine_, finish_reduce_;
    mtl::Pipeline dense_k4_, dense_k6_, dense_finish_, dense_split_, dense_split_finish_;
};

}  // namespace stark::hy4::stq
