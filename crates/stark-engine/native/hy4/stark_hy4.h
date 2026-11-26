// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

struct StarkHy4;

struct StarkHy4ContextMemory {
    uint32_t checkpoint_context;
    uint32_t engine_context;
    uint32_t pool_size;
    uint32_t draft_slack;
    uint64_t budget;
    uint64_t cache_bytes_per_token;
    uint64_t draft_cache_bytes_per_token;
    uint64_t pool_bytes_per_row;
    uint64_t draft_pool_bytes_per_row;
};

struct StarkHy4VisionGeometry {
    uint32_t image_token;
    uint32_t patch;
    uint32_t merge;
    uint64_t min_pixels;
    uint64_t max_pixels;
};

struct StarkHy4ImageInput {
    const uint8_t* rgb;
    size_t rgb_len;
    uint32_t width;
    uint32_t height;
};

struct StarkHy4Stats {
    uint64_t offered;
    uint64_t accepted;
    uint64_t rounds;
    double draft_seconds;
    double verify_seconds;
    double catchup_seconds;
    double prefill_gpu_span_seconds;
    double prefill_encode_seconds;
    double prefill_expert_wait_seconds;
    double prefill_engram_seconds;
    double decode_gpu_span_seconds;
    double decode_host_cut_seconds;
    uint64_t prefill_expert_bytes_read;
    uint64_t decode_expert_bytes_read;
    double prefill_attention_seconds;
    double prefill_dense_seconds;
    double prefill_moe_route_seconds;
    double prefill_moe_expert_seconds;
    double prefill_other_seconds;
    uint64_t prefill_profiled_blocks;
    uint64_t prefill_grouped_tiles;
    uint64_t prefill_grouped_tail_tiles;
    double decode_encode_seconds;
    double decode_commit_seconds;
    double decode_completion_wait_seconds;
    double decode_engram_seconds;
    uint64_t decode_command_buffers;
    uint64_t decode_encoders;
    uint64_t decode_dispatches;
    uint64_t prefill_invalid_gpu_spans;
    uint64_t decode_invalid_gpu_spans;
    double prefill_attention_dense_seconds;
    double prefill_attention_index_seconds;
    double prefill_attention_mla_seconds;
    double prefill_attention_misc_seconds;
};

extern "C" {
int stark_hy4_open(const char* model, const char* projector, const char* pack, int chunk,
                   int storage, int grouped, int gpu_plan, int profile_prefill, uint64_t arena_bytes_requested,
                   int demand_retention, int routed_half, int dense_packed, int resident_token_embeddings,
                   uint32_t (*select_context)(const StarkHy4ContextMemory*, void*), void* sizing,
                   StarkHy4** output, StarkHy4VisionGeometry* vision, int* mapped,
                   uint64_t* arena_bytes) noexcept;
void stark_hy4_close(StarkHy4* session) noexcept;
void stark_hy4_clear(StarkHy4* session) noexcept;
int stark_hy4_retain_prefix(StarkHy4* session, int position) noexcept;
int stark_hy4_begin(StarkHy4* session, const uint32_t* tokens, int count, int context, int reuse, int drafts,
                    const StarkHy4ImageInput* image) noexcept;
int stark_hy4_prefill(StarkHy4* session, int rows) noexcept;
int stark_hy4_greedy(const StarkHy4* session) noexcept;
const float* stark_hy4_logits(const StarkHy4* session) noexcept;
int stark_hy4_append(StarkHy4* session, uint32_t token) noexcept;
int stark_hy4_verify(StarkHy4* session, int count, uint32_t* output) noexcept;
StarkHy4Stats stark_hy4_stats(StarkHy4* session) noexcept;
}
