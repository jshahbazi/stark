// SPDX-License-Identifier: Apache-2.0
#include "stark_hy4.h"
#include "stark_hy4_vision.h"

#include "experts.h"
#include "engine/sysmem.h"
#include "forward.h"
#include "model.h"
#include "mtp.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace strata::hy4_flash;

namespace {
constexpr int kVocab = 120832;
constexpr int kRecommendedDrafts = 2;
constexpr int kMaxDrafts = 7;
static_assert(kMaxDrafts + 1 <= Forward::kMaxWindow && kMaxDrafts < Mtp::kCacheSlack);
constexpr uint64_t kGiB = 1ull << 30;

double mapped_expert_room(const ArenaPlan& arena) {
    const double system_reserve = 24.0 * kGiB;
    const double live_room = strata::metal_engine::sys_file_backed_room(arena.avail) - 4.0 * kGiB;
    return std::min({arena.gpu_room, live_room, arena.phys - arena.base - system_reserve});
}

StarkHy4ContextMemory context_memory(const Geometry& g, const mtl::Context& metal, int chunk, uint64_t budget) {
    const auto trunk_end = g.layers.begin() + g.trunk_blocks;
    const uint64_t indexed = std::count_if(g.layers.begin(), trunk_end,
                                           [](const LayerSpec& layer) { return layer.has_indexer; });
    const uint64_t kv = sizeof(uint16_t) * g.kv_a_width;
    const uint64_t index = sizeof(uint16_t) * 3 * g.idx_key_dim;
    const uint64_t draft_hidden = sizeof(float) * g.hidden_dim;
    const uint64_t pooled = sizeof(float) * g.idx_key_dim;
    const uint64_t max_buffer = metal.max_buffer_length();
    const uint64_t slack = Mtp::kCacheSlack;
    if (max_buffer < draft_hidden * (slack + 1)) return {};
    const uint64_t draft_rows = std::min({max_buffer / kv, max_buffer / index, max_buffer / draft_hidden - 1});
    const uint64_t max_pool_rows = max_buffer / (sizeof(float) * chunk);
    if (draft_rows <= slack || max_pool_rows == 0) return {};
    // All of these buffers are contiguous, even though their offsets select different layers or tokens.
    const uint64_t engine_context = std::min({
        max_buffer / (kv * g.trunk_blocks),
        max_buffer / (index * indexed),
        draft_rows - slack,
        max_pool_rows * g.idx_kpool - 1,
        static_cast<uint64_t>(std::numeric_limits<int>::max()) - slack,
    });
    return {
        static_cast<uint32_t>(g.max_context),
        static_cast<uint32_t>(engine_context),
        static_cast<uint32_t>(g.idx_kpool),
        static_cast<uint32_t>(slack),
        budget,
        // Full trunk KV and IndexPool state plus token and image-position bookkeeping on the host.
        kv * g.trunk_blocks + index * indexed + 6 * sizeof(int32_t),
        draft_hidden + kv + index,
        pooled + sizeof(float) * (chunk + Forward::kMaxWindow),
        pooled + sizeof(float),
    };
}
}

// Members are destroyed in reverse dependency order. Metal submissions complete
// before the calls that encode them return.
struct StarkHy4 {
    mtl::Context metal;
    Model model;
    Forward forward;
    Mtp mtp;
    StarkHy4Vision vision;
    std::unique_ptr<StarkHy4ImagePrompt> image;
    std::vector<int32_t> tokens;
    int capacity = 0;
    int context = 0;
    int position = 0;
    int drafts = 0;
    int max_drafts = 0;
    StarkHy4Stats start_stats{};
    mtl::Context::SegTiming decode_start{};
    double decode_engram_start_ms = 0;
    uint64_t request_start_invalid_spans = 0;
    ~StarkHy4() { metal.stop_keepalive(); }
};

extern "C" int stark_hy4_open(const char* model, const char* projector, const char* pack, int chunk,
                               int storage, int grouped, int gpu_plan, int profile_prefill, uint64_t arena_bytes_requested,
                               int demand_retention, int routed_half, int dense_packed, int resident_token_embeddings,
                               int tensor_mla, int max_drafts, int routed_fragments, int dense_device_views,
                               int resident_draft_experts, int draft_head_q5,
                               uint32_t (*select_context)(const StarkHy4ContextMemory*, void*), void* sizing,
                               StarkHy4** output, StarkHy4VisionGeometry* vision, int* mapped,
                               uint64_t* arena_bytes, int* actual_draft_head_q5) noexcept {
    *output = nullptr;
    try {
        if ((chunk != 1024 && chunk != 4096) ||
            storage < 0 || storage > 3 ||
            ((arena_bytes_requested || demand_retention) && storage != 1) ||
            (draft_head_q5 != 0 && draft_head_q5 != 1) ||
            max_drafts < kRecommendedDrafts || max_drafts > kMaxDrafts) return 1;
        auto session = std::make_unique<StarkHy4>();
        auto& s = *session;
        const bool resident_experts = storage == 3;
        const bool require_mapped = storage == 2 || resident_experts;
        s.metal.lock_safe_math();
        if (!open(s.model, model, 0, true, resident_token_embeddings != 0).empty()) return 1;
        const auto& geo = s.model.geo;
        if (geo.vocab_size != kVocab || geo.hidden_dim != 3072 || geo.idx_kpool <= 0 || geo.idx_key_dim <= 0 ||
            geo.kv_a_width <= 0 || geo.max_context < 2 || geo.trunk_blocks < 1 ||
            chunk % geo.idx_kpool != 0 ||
            std::none_of(geo.layers.begin(), geo.layers.begin() + geo.trunk_blocks,
                         [](const LayerSpec& layer) { return layer.has_indexer; }))
            return 1;
        if (!load_dense(s.model, s.metal).empty() || !open_experts(s.model).empty()) return 2;
        if (!s.vision.open(s.metal, model, projector, geo.hidden_dim, geo.rope_dim).empty()) return 3;
        // The prompt may need a standalone workspace when it cannot borrow expert slots. Its score
        // rows grow with context and are included in context_memory. Reserve the fixed planes, the
        // draft expert pool and fixed draft scratch, and the largest supported image before fitting.
        const size_t dense_view_bytes = dense_device_views ? Forward::dense_device_scratch_bytes(geo, chunk) : 0;
        const double future = 2.0 * kGiB * (chunk / 1024.0) + s.vision.reserved_bytes() + dense_view_bytes;
        const uint64_t draft_bank_bytes = resident_draft_experts ? Mtp::resident_expert_bytes(geo) : 0;
        if (resident_draft_experts && (draft_bank_bytes == 0 || draft_bank_bytes > s.metal.max_buffer_length()))
            return 5;
        const double draft_and_decode_reserve = 3.0 * kGiB +
            std::max(0.0, static_cast<double>(draft_bank_bytes) - Mtp::kExpertPoolBudgetBytes);
        const auto initial = plan_arena(s.model, s.metal);
        const double workspace = future + draft_and_decode_reserve;
        const double room =
            std::min({initial.gpu_room, initial.ram_room, initial.default_cap - initial.base}) - workspace;
        double context_room = room - std::min(initial.arena, room / 2.0);
        if (arena_bytes_requested != 0) context_room = room - static_cast<double>(arena_bytes_requested);
        if (require_mapped) {
            // The context and scratch use the anonymous budget. The clean expert pages
            // also have to fit within the separate file-backed and Metal headroom.
            context_room = std::min(room, mapped_expert_room(initial) - initial.expert_total - workspace);
        }
        if (context_room <= 0) return arena_bytes_requested != 0 ? 9 : require_mapped ? 8 : 10;
        const auto memory = context_memory(geo, s.metal, chunk, static_cast<uint64_t>(context_room));
        if (memory.engine_context < 2) return 10;
        const uint32_t selected = select_context(&memory, sizing);
        if (selected < 2 || selected > memory.engine_context || selected > memory.checkpoint_context) return 10;
        const int context = static_cast<int>(selected);
        s.model.geo.max_context = context;
        Prefill prefill;
        prefill.chunk = chunk;
        prefill.grouped_k_quant = grouped != 0;
        prefill.mapped_gpu_plan = gpu_plan != 0;
        prefill.optimize_mapped_decode = resident_experts && prefill.mapped_gpu_plan;
        prefill.profile_stages = profile_prefill != 0;
        prefill.routed_half = routed_half != 0;
        prefill.routed_fragments = routed_fragments != 0;
        prefill.dense_packed = dense_packed != 0;
        prefill.dense_device_views = dense_device_views != 0;
        prefill.tensor_mla = tensor_mla != 0;
        if (!s.forward.open(s.model, s.metal, pack, prefill).empty()) return 4;
        const bool resident_nextn = resident_draft_experts != 0;
        if (!s.mtp.open(s.model, s.metal, s.forward.library(), 1024, pack, resident_nextn,
                        resident_nextn && prefill.dense_packed, draft_head_q5 != 0).empty()) return 5;
        if (!s.vision.build(s.metal, s.forward.library()).empty()) return 3;
        auto arena = plan_arena(s.model, s.metal);
        // Stark's resident expert paths budget shared allocation and leave
        // clean file mappings outside the anonymous arena. Count the pending
        // mapped-prefill workspace and image scratch before admitting banks.
        const uint64_t pool_rows = selected / memory.pool_size + 1;
        const double pending = future + sizeof(float) * chunk * pool_rows;
        const double mapped_room = mapped_expert_room(arena) - pending;
        const bool loaded_mapped = storage != 1 && mapped_room > 0 &&
                                   s.model.stream.map_experts(s.metal, static_cast<uint64_t>(mapped_room), resident_experts);
        if (require_mapped && !loaded_mapped) return 8;
        *mapped = loaded_mapped ? 1 : 0;
        const int slots = 2 * s.model.stream.n_experts();
        if (!*mapped) {
            if (arena_bytes_requested != 0) {
                const double candidate_bytes = static_cast<double>(arena_bytes_requested);
                const double spare = std::min({arena.gpu_room, arena.ram_room, arena.default_cap - arena.base}) - pending;
                if (spare < candidate_bytes || arena.expert_total < candidate_bytes) return 9;
                // The existing live anonymous-arena shrink and grow guards
                // remain active while the larger candidate allocates.
                arena.arena = candidate_bytes;
            }
            arena.arena = std::min(arena.arena, std::min({arena.gpu_room, arena.ram_room,
                                                        arena.default_cap - arena.base}) - pending);
            if (arena.arena <= 0 || !make_arena(s.model, s.metal, arena, "", slots + 64).empty()) return 6;
            s.model.stream.set_demand_retention(demand_retention != 0);
            for (int c = 0; c < s.model.stream.n_classes(); ++c) {
                if (s.model.stream.class_slots(c) < slots) return 6;
            }
        }
        if (!s.forward.attach_prefill().empty()) return 6;
        if (!s.mtp.load_resident_experts().empty()) return 5;
        *arena_bytes = s.model.stream.arena_used_bytes();
        const auto& vg = s.vision.geometry();
        *vision = {static_cast<uint32_t>(vg.image_token), static_cast<uint32_t>(vg.patch),
                   static_cast<uint32_t>(vg.merge), static_cast<uint64_t>(vg.min_pixels),
                   static_cast<uint64_t>(vg.max_pixels)};
        s.capacity = context;
        s.max_drafts = max_drafts;
        s.tokens.reserve(static_cast<size_t>(context));
        *actual_draft_head_q5 = s.mtp.uses_q5_draft_head();
        *output = session.release();
        return 0;
    } catch (...) {
        return 7;
    }
}

extern "C" void stark_hy4_close(StarkHy4* session) noexcept { delete session; }

extern "C" void stark_hy4_clear(StarkHy4* session) noexcept {
    auto& s = *session;
    s.image.reset();
    s.forward.reset();
    s.mtp.reset();
    s.tokens.clear();
    s.position = 0;
    s.context = 0;
    s.drafts = 0;
}

extern "C" int stark_hy4_retain_prefix(StarkHy4* session, int position) noexcept {
    auto& s = *session;
    try {
        if (s.image || position < 1 || position > s.position || !s.forward.rewind(position).empty()) return 1;
        s.mtp.truncate(position);
        s.tokens.resize(static_cast<size_t>(position));
        s.position = position;
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int stark_hy4_begin(StarkHy4* session, const uint32_t* tokens, int count, int context, int reuse,
                                int drafts, const StarkHy4ImageInput* image) noexcept {
    auto& s = *session;
    try {
        if (context > s.capacity || count < 1 || count >= context || reuse < 0 || reuse >= count ||
            drafts < 0 || drafts > s.max_drafts || (image && (reuse || drafts))) return 1;
        const auto valid = [](uint32_t token) { return token < kVocab; };
        if (!std::all_of(tokens, tokens + count, valid)) return 1;
        if (reuse == 0) {
            stark_hy4_clear(session);
        } else {
            if (context != s.context || drafts != s.drafts || s.image ||
                static_cast<size_t>(reuse) > s.tokens.size() ||
                !std::equal(tokens, tokens + reuse, s.tokens.begin()) ||
                !s.forward.rewind(reuse).empty()) return 1;
            s.mtp.truncate(reuse);
            s.mtp.stats() = {};
        }
        s.tokens.assign(tokens, tokens + count);
        s.position = reuse;
        s.context = context;
        s.drafts = drafts;
        s.decode_start = s.metal.seg_timing();
        s.request_start_invalid_spans = s.decode_start.invalid_gpu_spans;
        s.decode_engram_start_ms = s.forward.timing().engram_wait_ms;
        s.start_stats = {};
        s.start_stats = stark_hy4_stats(session);
        if (image) {
            s.image = std::make_unique<StarkHy4ImagePrompt>();
            if (!s.image->attach(s.vision, s.metal, s.forward, s.tokens.data(), s.tokens.size(), image->rgb,
                                 image->rgb_len, image->width, image->height).empty()) return 2;
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int stark_hy4_prefill(StarkHy4* session, int rows) noexcept {
    auto& s = *session;
    try {
        if (rows < 1 || rows > s.forward.prefill_chunk() ||
            s.position + rows > static_cast<int>(s.tokens.size())) return 1;
        Forward::PrefillSink sink;
        if (s.drafts) {
            sink.hidden = [&](int pos, int n, const float* h) { s.mtp.set_hidden(pos + 1, n, h); };
        }
        if (!s.forward.prefill(s.tokens.data(), s.position, rows, sink).empty()) return 1;
        s.position += rows;
        const auto& timing = s.forward.timing();
        s.start_stats.decode_host_cut_seconds = timing.host_cut_ms / 1e3;
        s.decode_engram_start_ms = timing.engram_wait_ms;
        s.decode_start = s.metal.seg_timing();
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int stark_hy4_greedy(const StarkHy4* session) noexcept {
    return session->position > 0 ? session->forward.last_token() : -1;
}

extern "C" const float* stark_hy4_logits(const StarkHy4* session) noexcept {
    return stark_hy4_greedy(session) >= 0 ? session->forward.last_logits() : nullptr;
}

extern "C" int stark_hy4_append(StarkHy4* session, uint32_t token) noexcept {
    auto& s = *session;
    try {
        if (token >= kVocab || s.position >= s.context || s.position != static_cast<int>(s.tokens.size())) return 1;
        s.tokens.push_back(static_cast<int32_t>(token));
        if (s.drafts == 0) {
            if (!s.forward.window(s.tokens.data(), s.position, 1).empty()) return 1;
            ++s.position;
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

// One exact greedy verification round adapted from Strata's speculative_greedy.
// The last committed token becomes the next input. Rejected rows are overwritten
// by the next window and the index pool is clamped at that window's start.
extern "C" int stark_hy4_verify(StarkHy4* session, int count, const uint32_t* lookup, uint32_t* output) noexcept {
    auto& s = *session;
    try {
        const int start = s.position;
        if (!s.drafts || start + 1 != static_cast<int>(s.tokens.size()) ||
            count < 1 || count > s.drafts + 1 || start + count >= s.context) return -1;
        const int k = count - 1;
        auto& stats = s.mtp.stats();
        double t = mtl::now_s();
        std::array<int32_t, kMaxDrafts> proposals{};
        if (lookup != nullptr) {
            if (k == 0 || !std::all_of(lookup, lookup + k, [](uint32_t t) { return t < kVocab; })) return -1;
            std::copy_n(lookup, k, proposals.begin());
            s.tokens.insert(s.tokens.end(), proposals.begin(), proposals.begin() + k);
            // The draft cache can lag here. The target history below contains
            // every committed hidden row that NextN needs to catch up later.
        } else {
            for (int i = 0; i < k; ++i) {
                const int row = start + i;
                if (i > 0) s.mtp.set_hidden(row, 1, s.mtp.state());
                if (!s.mtp.step(s.tokens.data(), row, &proposals[i]).empty()) return -1;
                s.tokens.push_back(proposals[i]);
            }
        }
        s.mtp.truncate(start + 1);
        stats.draft_ms += (mtl::now_s() - t) * 1e3;
        t = mtl::now_s();
        if (!s.forward.window(s.tokens.data(), start, count).empty()) return -1;
        stats.verify_ms += (mtl::now_s() - t) * 1e3;
        const int32_t* actual = s.forward.argmax();
        if (!std::all_of(actual, actual + count, [](int32_t t) { return t >= 0 && t < kVocab; })) return -1;
        const int accepted = accept_prefix(proposals.data(), actual, k);
        s.mtp.set_hidden(start + 1, accepted + 1, s.forward.result_norm());
        s.tokens.resize(static_cast<size_t>(start + 1));
        for (int i = 0; i <= accepted; ++i) {
            output[i] = static_cast<uint32_t>(actual[i]);
            s.tokens.push_back(actual[i]);
        }
        s.position += accepted + 1;
        stats.offered += static_cast<uint64_t>(k);
        stats.accepted += static_cast<uint64_t>(accepted);
        ++stats.rounds;
        return accepted + 1;
    } catch (...) {
        return -1;
    }
}

extern "C" StarkHy4Stats stark_hy4_stats(StarkHy4* session) noexcept {
    const auto& draft = session->mtp.stats();
    const auto& graph = session->forward.timing();
    const auto& base = session->start_stats;
    const auto& schedule = session->metal.seg_timing();
    const auto& schedule_base = session->decode_start;
    return {
        draft.offered, draft.accepted, draft.rounds,
        draft.draft_ms / 1e3, draft.verify_ms / 1e3, draft.catchup_ms / 1e3,
        graph.pf_gpu_ms / 1e3 - base.prefill_gpu_span_seconds,
        graph.pf_encode_ms / 1e3 - base.prefill_encode_seconds,
        graph.pf_wait_ms / 1e3 - base.prefill_expert_wait_seconds,
        graph.pf_engram_ms / 1e3 - base.prefill_engram_seconds,
        graph.gpu_ms / 1e3 - base.decode_gpu_span_seconds,
        graph.host_cut_ms / 1e3 - base.decode_host_cut_seconds,
        graph.pf_bytes_read - base.prefill_expert_bytes_read,
        session->model.stream.decode_read_bytes() - base.decode_expert_bytes_read,
        graph.pf_attention_ms / 1e3 - base.prefill_attention_seconds,
        graph.pf_dense_ms / 1e3 - base.prefill_dense_seconds,
        graph.pf_moe_route_ms / 1e3 - base.prefill_moe_route_seconds,
        graph.pf_moe_expert_ms / 1e3 - base.prefill_moe_expert_seconds,
        graph.pf_other_ms / 1e3 - base.prefill_other_seconds,
        graph.pf_profiled_blocks - base.prefill_profiled_blocks,
        graph.pf_grouped_tiles - base.prefill_grouped_tiles,
        graph.pf_grouped_tail_tiles - base.prefill_grouped_tail_tiles,
        (schedule.encode_ms - schedule_base.encode_ms) / 1e3,
        (schedule.commit_ms - schedule_base.commit_ms) / 1e3,
        (schedule.wait_ms - schedule_base.wait_ms) / 1e3,
        (graph.engram_wait_ms - session->decode_engram_start_ms) / 1e3,
        schedule.n_cb - schedule_base.n_cb,
        schedule.n_enc - schedule_base.n_enc,
        schedule.n_dis - schedule_base.n_dis,
        schedule_base.invalid_gpu_spans - session->request_start_invalid_spans,
        schedule.invalid_gpu_spans - schedule_base.invalid_gpu_spans,
        graph.pf_attention_dense_ms / 1e3 - base.prefill_attention_dense_seconds,
        graph.pf_attention_index_ms / 1e3 - base.prefill_attention_index_seconds,
        graph.pf_attention_mla_ms / 1e3 - base.prefill_attention_mla_seconds,
        graph.pf_attention_misc_ms / 1e3 - base.prefill_attention_misc_seconds,
    };
}
