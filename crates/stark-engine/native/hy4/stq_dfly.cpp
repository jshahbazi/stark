// SPDX-License-Identifier: Apache-2.0
#include "stq_dfly.h"

#include "attn.h"
#include "dispatch.h"
#include "model.h"
#include "stq_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#pragma clang fp contract(off)

namespace stark::hy4::stq {
namespace {

using mtl::Arg;
using mtl::Buffer;
using mtl::Dispatch;
using mtl::Pipeline;
using strata::hy4_flash::DenseGemvF32;
using strata::hy4_flash::GemvHeads;

constexpr uint32_t kWidth = 3072, kVocab = 120832, kMask = 120009;
constexpr uint32_t kHeads = 32, kNope = 192, kRope = 64, kQk = kNope + kRope;
constexpr uint32_t kQr = 1024, kLatent = 512, kKv = kLatent + kRope, kValue = 256, kFf = 9216;
constexpr uint32_t kRows = kDflyMaxDrafts + 1, kPoolCutoff = 2048, kMaxSelection = 2051;
constexpr uint32_t kMaxKeys = kMaxSelection + kRows;
constexpr uint32_t kMlaPartition = 256, kMlaPartialWidth = kLatent + 2;
constexpr uint32_t kMaxMlaParts = (kMaxKeys + kMlaPartition - 1) / kMlaPartition;
constexpr size_t kKvBytes = kKv * sizeof(uint16_t);
constexpr uint32_t kTg = 256;
constexpr float kEps = 1e-5f;

// Shared Metal library ABI (hy4_flash_norm.metal and hy4_flash_attn.metal).
struct NormArgs { uint32_t n, x_stride, y_stride; float eps; };
struct RopeArgs {
    uint32_t n_rot, n_head, head_dim, off, x_stride, y_stride;
    float theta_scale;
    uint32_t pos_stride, mlen1, mlen2;
};
struct KvArgs {
    uint32_t n, n_rot, row, kvn_stride;
    float eps, theta_scale;
    uint32_t pos_stride, mlen1, mlen2;
};
struct AttnArgs {
    uint32_t heads, d_qk, d_v, kv_stride, q_stride, o_stride, sel_stride, sc_stride, window;
    float scale;
};
struct ElemArgs { uint32_t n, a_stride, b_stride, y_stride; };
struct GluArgs { uint32_t n, g_stride, u_stride, y_stride; float limit; };
struct ArgmaxArgs { uint32_t n, stride; };

void push(std::vector<Dispatch>& commands, const Pipeline& pipeline, std::vector<Arg> args,
          uint32_t x, uint32_t y, uint32_t threads) {
    Dispatch d{&pipeline, std::move(args), {x, y, 1}, {threads, 1, 1}};
    mtl::require_fit(d, pipeline.name.c_str());
    commands.push_back(std::move(d));
}

} // namespace

struct Dfly2::Impl {
    struct Layer {
        const gguf::Tensor *input, *q_norm, *kv_a, *kv_norm, *key, *value, *sinks, *attention_norm, *ffn_norm;
        const DenseBank *q_a, *q_b, *attention_gate, *o, *ffn_gate, *ffn_up, *ffn_down;
    };

    explicit Impl(strata::hy4_flash::Model& source, mtl::Context& context) : model(source), metal(context) {}
    ~Impl() { for (Buffer* buffer : owned) metal.free(*buffer); }

    strata::hy4_flash::Model& model;
    mtl::Context& metal;
    std::array<Layer, kDflyTargetBlocks.size()> layers;
    const gguf::Tensor *enorm, *hnorm, *fusion_gate, *final_norm, *embed_norm, *hidden_norm, *conf_weight, *conf_bias;
    const DenseBank *fusion, *correction_gate, *correction_up, *correction_down, *embed, *head;
    Pipeline p_norm, p_rope, p_kv, p_copy, p_add, p_gate, p_glu, p_argmax;
    Pipeline p_register_mla, p_mla_finish, p_stitch;
    DenseGemvF32 raw_kv, key_absorb, value_absorb;
    strata::hy4_flash::AttnPipes attention;
    Buffer ids, row_map, positions, compact_positions, unused, bad, best;
    Buffer x, normed, hidden, concat, fused, previous;
    Buffer q_low, q_full, q_abs, kv_raw, kv_norm, score, attended;
    Buffer gate, up, product, linear_out, logits, packed_input, packed_raw;
    std::array<Buffer, kDflyTargetBlocks.size()> stitched;
    std::vector<Buffer*> owned;
    std::vector<Dispatch> commands;

    const gguf::Tensor* raw(const std::string& alias, int type, std::initializer_list<uint64_t> shape) const {
        const auto* tensor = model.find(alias);
        const uint64_t offset = tensor ? tensor->offset : 0;
        uint64_t bytes = type == ref::BF16 ? sizeof(uint16_t) : sizeof(float);
        for (uint64_t dim : shape) bytes *= dim;
        if (!tensor || tensor->type != type || tensor->shape.size() != shape.size() ||
            !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin()) ||
            tensor->bytes != bytes || offset > model.dense.bytes || bytes > model.dense.bytes - offset) {
            throw std::runtime_error("invalid STQ DFly2 raw binding " + alias);
        }
        return tensor;
    }

    const DenseBank* packed(const std::string& alias, uint32_t inputs, uint32_t outputs, Format format) const {
        const auto bank = model.stq->dense.find(alias);
        if (bank == model.stq->dense.end() || !bank->second->ready() || bank->second->format != format ||
            bank->second->inputs != inputs || bank->second->outputs != outputs) {
            throw std::runtime_error("invalid STQ DFly2 packed binding " + alias);
        }
        return bank->second.get();
    }

    void bind() {
        const std::string base = "model.dfly_layers.";
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto p = base + std::to_string(i) + ".";
            const auto a = p + "self_attn.";
            layers[i] = {
                raw(p + "input_layernorm.weight", ref::F32, {kWidth}),
                raw(a + "q_a_layernorm.weight", ref::F32, {kQr}),
                raw(a + "kv_a_proj_with_mqa.weight", ref::BF16, {kWidth, kKv}),
                raw(a + "kv_a_layernorm.weight", ref::F32, {kLatent}),
                raw(a + "attn_k_b.weight", ref::BF16, {kNope, kLatent, kHeads}),
                raw(a + "attn_v_b.weight", ref::BF16, {kLatent, kValue, kHeads}),
                raw(a + "learnable_sink_param", ref::F32, {kHeads}),
                raw(a + "post_proj_norm.weight", ref::F32, {kWidth}),
                raw(p + "post_attention_layernorm.weight", ref::F32, {kWidth}),
                packed(a + "q_a_proj.weight", kWidth, kQr, Format::spec1_k4),
                packed(a + "q_b_proj.weight", kQr, kHeads * kQk, Format::spec1_k4),
                packed(a + "linear_gate.weight", kWidth, kHeads * kValue, Format::spec1_k4),
                packed(a + "o_proj.weight", kHeads * kValue, kWidth, Format::spec1_k4),
                packed(p + "mlp.gate_proj.weight", kWidth, kFf, Format::spec1_k4),
                packed(p + "mlp.up_proj.weight", kWidth, kFf, Format::spec1_k4),
                packed(p + "mlp.down_proj.weight", kFf, kWidth, Format::spec1_k4),
            };
        }
        enorm = raw(base + "mtp_fusion.enorm.weight", ref::F32, {kWidth});
        hnorm = raw(base + "mtp_fusion.hnorm.weight", ref::F32, {kWidth});
        fusion_gate = raw(base + "mtp_fusion.gate", ref::F32, {kWidth});
        final_norm = raw(base + "final_norm.weight", ref::F32, {kWidth});
        embed_norm = raw(base + "hidden_correction.embed_norm.weight", ref::F32, {kWidth});
        hidden_norm = raw(base + "hidden_correction.hidden_norm.weight", ref::F32, {kWidth});
        conf_weight = raw(base + "confidence_head.proj.weight", ref::BF16, {kWidth, 1});
        conf_bias = raw(base + "confidence_head.proj.bias", ref::F32, {1});
        fusion = packed(base + "mtp_fusion.eh_proj.weight", 2 * kWidth, kWidth, Format::spec1_k4);
        correction_gate = packed(base + "hidden_correction.gate_proj.weight", 2 * kWidth, kWidth, Format::spec1_k4);
        correction_up = packed(base + "hidden_correction.up_proj.weight", 2 * kWidth, kWidth, Format::spec1_k4);
        correction_down = packed(base + "hidden_correction.down_proj.weight", kWidth, kWidth, Format::spec1_k4);
        embed = packed("token_embd.weight", kWidth, kVocab, Format::spec1_k6);
        head = packed("output.weight", kWidth, kVocab, Format::spec1_k6);
    }

    void allocate(Buffer& buffer, size_t bytes) {
        owned.push_back(&buffer);
        buffer = metal.alloc(bytes);
        if (!buffer.ptr) throw std::runtime_error("STQ DFly2 scratch allocation failed");
    }

    void initialize(int library, bool register_mla) {
        bind();
        auto build = [&](const char* name) { return metal.pipeline(library, name); };
        p_norm = metal.pipeline(library, "hy4_flash_rmsnorm", mtl::Constants().b(40, true));
        p_rope = build("hy4_flash_rope_norm");
        p_kv = build("hy4_flash_kv_finish");
        p_copy = build("hy4_flash_copy_f32");
        p_add = build("hy4_flash_add_f32");
        p_gate = build("hy4_flash_gate_mul");
        p_glu = build("hy4_flash_swiglu_clamp");
        p_argmax = build("hy4_flash_argmax");
        attention = strata::hy4_flash::attn_pipes(&metal, library, false);
        if (register_mla) {
            p_register_mla = metal.pipeline(library, "stark_hy4_stq_register_mla", mtl::Constants().b(43, false));
            p_mla_finish = build("stark_hy4_stq_register_mla_finish");
            p_stitch = build("stark_hy4_stq_dfly_stitch");
            if (p_register_mla.max_threads < 288 || p_mla_finish.max_threads < kTg || p_stitch.max_threads < 64)
                throw std::runtime_error("STQ DFly2 register MLA pipeline unavailable");
        }
        raw_kv = DenseGemvF32(metal, library, ref::BF16, kWidth, kKv, kWidth * sizeof(uint16_t));
        key_absorb = DenseGemvF32(metal, library, ref::BF16, kNope, kLatent, kNope * sizeof(uint16_t),
                                 GemvHeads{kLatent * kNope * sizeof(uint16_t), kQk, kKv});
        value_absorb = DenseGemvF32(metal, library, ref::BF16, kLatent, kValue, kLatent * sizeof(uint16_t),
                                   GemvHeads{kValue * kLatent * sizeof(uint16_t), kLatent, kValue});

        auto floats = [&](Buffer& b, size_t n) { allocate(b, n * sizeof(float)); };
        allocate(ids, kRows * sizeof(int32_t));
        allocate(row_map, kRows * 2 * sizeof(uint32_t));
        allocate(positions, kRows * sizeof(int32_t));
        allocate(compact_positions, layers.size() * kRows * sizeof(int32_t));
        allocate(unused, sizeof(int32_t));
        allocate(bad, sizeof(uint32_t));
        allocate(best, sizeof(int32_t));
        for (uint32_t i = 0; i < kRows; ++i) {
            row_map.as<uint32_t>()[2 * i] = 0;
            row_map.as<uint32_t>()[2 * i + 1] = i;
        }
        floats(x, kRows * kWidth);
        floats(normed, kRows * kWidth);
        floats(hidden, kDflyMaxDrafts * kWidth);
        floats(concat, 2 * kWidth);
        floats(fused, kWidth);
        floats(previous, kWidth);
        floats(q_low, kRows * kQr);
        floats(q_full, kRows * kHeads * kQk);
        floats(q_abs, kRows * kHeads * kKv);
        floats(kv_raw, kRows * kKv);
        floats(kv_norm, kRows * kLatent);
        floats(score, kRows * kHeads * (register_mla ? kMaxMlaParts * kMlaPartialWidth : kMaxKeys));
        floats(attended, kRows * kHeads * kLatent);
        floats(gate, kRows * kFf);
        floats(up, kRows * kFf);
        floats(product, kRows * kFf);
        floats(linear_out, kRows * kWidth);
        floats(logits, kVocab);
        floats(packed_input, kRows * kFf);
        floats(packed_raw, std::max(kRows * kFf, kVocab));
        for (Buffer& rows : stitched) allocate(rows, kMaxKeys * kKvBytes);
    }

    void rms(const Buffer& input, size_t in, const gguf::Tensor* w, const Buffer& output, size_t out,
             uint32_t width, uint32_t rows) {
        const NormArgs args{width, width, width, kEps};
        push(commands, p_norm, {Arg::B(input, in), Arg::B(model.dense, w->offset), Arg::B(output, out), Arg::V(args)},
             rows, 1, kTg);
    }

    void elementwise(const Pipeline& p, const Buffer& a, size_t a_offset, const Buffer& b, size_t b_offset,
                     const Buffer& output, uint32_t width, uint32_t rows) {
        const ElemArgs args{width, width, width, width};
        push(commands, p, {Arg::B(a, a_offset), Arg::B(b, b_offset), Arg::B(output), Arg::V(args)},
             (width + kTg - 1) / kTg, rows, kTg);
    }

    void project(const DenseBank* bank, const Buffer& input, const Buffer& output, uint32_t rows) {
        if (!model.stq->program.append_dense(commands, *bank, input, row_map, rows, packed_input, packed_raw, output))
            throw std::runtime_error("invalid STQ DFly2 projection");
    }

    void embed_rows(const Buffer& output, uint32_t rows) {
        if (!model.stq->embedding.append(commands, *embed, ids, rows, output))
            throw std::runtime_error("invalid STQ DFly2 embedding");
    }

    void swiglu(const Buffer& input, const DenseBank* g, const DenseBank* u, const DenseBank* d,
                const Buffer& output, uint32_t rows) {
        const uint32_t n = g->outputs;
        project(g, input, gate, rows);
        project(u, input, up, rows);
        const GluArgs args{n, n, n, n, std::numeric_limits<float>::infinity()};
        push(commands, p_glu, {Arg::B(gate), Arg::B(up), Arg::B(product), Arg::V(args)},
             (n + kTg - 1) / kTg, rows, kTg);
        project(d, product, output, rows);
    }

    void execute() {
        metal.run_graph(commands);
        commands.clear();
    }

    uint32_t stitch_target(size_t layer, const DflyTargetView& target, uint32_t rows) {
        const auto& source = target.layers[layer];
        if (!source.kv || !source.kv->ptr || source.byte_offset % alignof(uint16_t) != 0 ||
            source.byte_offset > source.kv->bytes ||
            target.position > (source.kv->bytes - source.byte_offset) / kKvBytes ||
            source.selected.size() > kMaxSelection || (source.selected.empty() && target.position > kPoolCutoff)) {
            throw std::runtime_error("invalid STQ DFly2 target KV view");
        }
        const auto* from = source.kv->as<const uint8_t>() + source.byte_offset;
        auto* dst = stitched[layer].as<uint8_t>();
        const uint32_t n = source.selected.empty() ? target.position : static_cast<uint32_t>(source.selected.size());
        if (source.selected.empty()) {
            std::memcpy(dst, from, n * kKvBytes);
        } else if (p_stitch.handle && source.selected_rows && source.selected_rows->handle &&
                   source.byte_offset % 16 == 0) {
            if (source.selected_offset % alignof(uint32_t) != 0 ||
                source.selected_offset > source.selected_rows->bytes ||
                n > (source.selected_rows->bytes - source.selected_offset) / sizeof(uint32_t))
                throw std::runtime_error("invalid STQ DFly2 selected rows view");
            push(commands, p_stitch, {Arg::B(*source.kv, source.byte_offset),
                Arg::B(*source.selected_rows, source.selected_offset), Arg::B(stitched[layer]), Arg::B(bad),
                Arg::V(target.position)}, n, 1, 64);
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                const auto row = source.selected[i];
                if (row >= target.position) throw std::runtime_error("uncommitted STQ DFly2 target row");
                std::memcpy(dst + i * kKvBytes, from + row * kKvBytes, kKvBytes);
            }
        }
        // Already-rotated target F16 rows are copied verbatim. Only locally
        // projected rows rotate using absolute target positions, below. The
        // attention mask operates on this compact buffer's row coordinates.
        for (uint32_t i = 0; i < rows; ++i)
            compact_positions.as<int32_t>()[layer * kRows + i] = static_cast<int32_t>(n + i);
        return n;
    }

    void encode_layer(size_t i, uint32_t shared, uint32_t rows) {
        const auto& l = layers[i];
        const float theta = std::pow(1e7f, -2.0f / kRope);
        rms(x, 0, l.input, normed, 0, kWidth, rows);
        project(l.q_a, normed, q_low, rows);
        rms(q_low, 0, l.q_norm, q_low, 0, kQr, rows);
        project(l.q_b, q_low, q_full, rows);
        const RopeArgs rope{kRope, kHeads, kQk, kNope, kHeads * kQk, kHeads * kQk, theta, 0, 0, 0};
        Dispatch rotated{&p_rope, {Arg::B(q_full), Arg::B(q_full), Arg::B(positions), Arg::V(rope)},
                         {kRope / 2, kHeads, rows}, {32, 1, 1}};
        mtl::require_fit(rotated, "STQ DFly2 RoPE");
        commands.push_back(std::move(rotated));
        key_absorb.append_heads(commands, model.dense, l.key->offset, q_full, kHeads * kQk,
                                q_abs, kHeads * kKv, 0, rows, kHeads);
        const ElemArgs tail{kRope, kQk, 0, kKv};
        push(commands, p_copy, {Arg::B(q_full, kNope * sizeof(float)), Arg::B(q_abs, kLatent * sizeof(float)), Arg::V(tail)},
             1, rows * kHeads, kTg);

        raw_kv.append(commands, model.dense, l.kv_a->offset, normed, kWidth, kv_raw, kKv, 0, rows);
        const KvArgs kv{kLatent, kRope, kKv, kLatent, kEps, theta, 0, 0, 0};
        push(commands, p_kv, {Arg::B(kv_raw), Arg::B(model.dense, l.kv_norm->offset), Arg::B(kv_norm),
                             Arg::B(stitched[i], shared * kKvBytes), Arg::B(positions), Arg::B(bad), Arg::V(kv)}, rows, 1, kTg);
        const AttnArgs args{kHeads, kKv, kLatent, kKv, kHeads * kKv, kHeads * kLatent, kMaxKeys, kMaxKeys,
                            0, 1.0f / std::sqrt(float(kQk))};
        const auto position = Arg::B(compact_positions, i * kRows * sizeof(int32_t));
        if (p_register_mla.handle) {
            const uint32_t parts = (shared + rows + kMlaPartition - 1) / kMlaPartition;
            Dispatch d{&p_register_mla, {Arg::B(q_abs), Arg::B(stitched[i]), Arg::B(model.dense, l.sinks->offset),
                        Arg::B(unused), Arg::B(unused), position, Arg::B(score), Arg::V(args), Arg::V(parts)},
                       {rows, kHeads / 16, parts}, {288, 1, 1}};
            mtl::require_fit(d, "STQ DFly2 partitioned MLA");
            commands.push_back(std::move(d));
            push(commands, p_mla_finish, {Arg::B(score), Arg::B(attended), Arg::V(args), Arg::V(parts)},
                 rows, kHeads, kTg);
        } else {
            strata::hy4_flash::append_attn(commands, attention, {
                Arg::B(q_abs), Arg::B(stitched[i]), Arg::B(model.dense, l.sinks->offset), Arg::B(unused), Arg::B(unused),
                position, Arg::B(score), Arg::B(attended), Arg::V(args)}, kHeads, rows, shared + rows, kMaxKeys,
                kKv, kLatent, kKv, "STQ DFly2 absorbed MLA");
        }
        value_absorb.append_heads(commands, model.dense, l.value->offset, attended, kHeads * kLatent,
                                  up, kHeads * kValue, 0, rows, kHeads);
        project(l.attention_gate, normed, gate, rows);
        elementwise(p_gate, up, 0, gate, 0, product, kHeads * kValue, rows);
        project(l.o, product, linear_out, rows);
        rms(linear_out, 0, l.attention_norm, linear_out, 0, kWidth, rows);
        elementwise(p_add, x, 0, linear_out, 0, x, kWidth, rows);

        rms(x, 0, l.ffn_norm, normed, 0, kWidth, rows);
        swiglu(normed, l.ffn_gate, l.ffn_up, l.ffn_down, linear_out, rows);
        elementwise(p_add, x, 0, linear_out, 0, x, kWidth, rows);
    }

    float confidence() const {
        const auto* w = reinterpret_cast<const uint16_t*>(model.dense.as<const uint8_t>() + conf_weight->offset);
        const auto* bias = reinterpret_cast<const float*>(model.dense.as<const uint8_t>() + conf_bias->offset);
        const auto* row = fused.as<const float>();
        float sum = 0.0f;
        for (uint32_t d = 0; d < kWidth; ++d) sum += row[d] * ref::bf16_to_f32(w[d]);
        return 1.0f / (1.0f + std::exp(-(sum + *bias)));
    }

    void propose(const DflyTargetView& target, uint32_t anchor, std::span<int32_t> proposals,
                 std::span<float> confidence_out) {
        const uint32_t rows = static_cast<uint32_t>(proposals.size() + 1);
        *bad.as<uint32_t>() = 0;
        std::array<uint32_t, kDflyTargetBlocks.size()> shared;
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = stitch_target(i, target, rows);
        for (uint32_t i = 0; i < rows; ++i) positions.as<int32_t>()[i] = static_cast<int32_t>(target.position + i);
        std::fill_n(ids.as<int32_t>(), rows, kMask);
        ids.as<int32_t>()[0] = static_cast<int32_t>(anchor);
        std::memcpy(fused.ptr, target.result_norm.data(), kWidth * sizeof(float));
        embed_rows(x, rows);
        rms(x, 0, enorm, concat, 0, kWidth, 1);
        rms(fused, 0, hnorm, concat, kWidth * sizeof(float), kWidth, 1);
        project(fusion, concat, fused, 1);
        execute();
        if (*bad.as<uint32_t>() != 0) throw std::runtime_error("uncommitted STQ DFly2 target row");

        // Preserve the unmodified anchor embedding for hidden correction.
        std::memcpy(previous.ptr, x.ptr, kWidth * sizeof(float));
        const auto* fg = reinterpret_cast<const float*>(model.dense.as<const uint8_t>() + fusion_gate->offset);
        for (uint32_t j = 0; j < kWidth; ++j) x.as<float>()[j] += fg[j] * fused.as<float>()[j];
        *bad.as<uint32_t>() = 0;
        for (size_t i = 0; i < layers.size(); ++i) encode_layer(i, shared[i], rows);
        rms(x, kWidth * sizeof(float), final_norm, hidden, 0, kWidth, rows - 1);
        execute();
        if (*bad.as<uint32_t>() != 0) throw std::runtime_error("nonfinite STQ DFly2 local KV");

        std::array<int32_t, kDflyMaxDrafts> draft{};
        std::array<float, kDflyMaxDrafts> conf{};
        for (size_t j = 0; j < proposals.size(); ++j) {
            if (j != 0) {
                *ids.as<int32_t>() = draft[j - 1];
                embed_rows(previous, 1);
            }
            const size_t offset = j * kWidth * sizeof(float);
            rms(hidden, offset, hidden_norm, concat, 0, kWidth, 1);
            rms(previous, 0, embed_norm, concat, kWidth * sizeof(float), kWidth, 1);
            swiglu(concat, correction_gate, correction_up, correction_down, linear_out, 1);
            elementwise(p_add, hidden, offset, linear_out, 0, fused, kWidth, 1);
            project(head, fused, logits, 1);
            push(commands, p_argmax, {Arg::B(logits), Arg::B(best), Arg::V(ArgmaxArgs{kVocab, kVocab})}, 1, 1, kTg);
            execute();
            draft[j] = *best.as<int32_t>();
            if (draft[j] < 0 || uint32_t(draft[j]) >= kVocab || !std::isfinite(logits.as<float>()[draft[j]]))
                throw std::runtime_error("nonfinite STQ DFly2 proposal");
            if (!confidence_out.empty()) {
                conf[j] = confidence();
                if (!std::isfinite(conf[j])) throw std::runtime_error("nonfinite STQ DFly2 confidence");
            }
        }
        std::copy_n(draft.begin(), proposals.size(), proposals.begin());
        std::copy_n(conf.begin(), confidence_out.size(), confidence_out.begin());
    }
};

Dfly2::Dfly2() = default;
Dfly2::~Dfly2() = default;

std::string Dfly2::open(strata::hy4_flash::Model& model, mtl::Context& metal, int shared_library, bool register_mla) {
    if (impl_ || !model.stq || shared_library < 0 || model.geo.trunk_blocks != 42 || model.geo.layers.size() != 42 ||
        model.geo.hidden_dim != kWidth || model.geo.vocab_size != kVocab || model.geo.kv_a_width != kKv ||
        !std::all_of(kDflyTargetBlocks.begin(), kDflyTargetBlocks.end(), [&](int32_t block) {
            return model.geo.block(block).has_indexer && model.geo.block(block).n_heads == kHeads;
        })) return "invalid STQ DFly2 model";
    try {
        auto impl = std::make_unique<Impl>(model, metal);
        impl->initialize(shared_library, register_mla);
        impl_ = std::move(impl);
        return {};
    } catch (const std::exception& e) {
        return e.what();
    }
}

std::string Dfly2::propose(const DflyTargetView& target, uint32_t anchor, std::span<int32_t> proposals,
                           std::span<float> confidence) {
    if (!impl_ || anchor >= kVocab || target.position == 0 || target.result_norm.size() != kWidth ||
        proposals.empty() || proposals.size() > kDflyMaxDrafts ||
        (!confidence.empty() && confidence.size() != proposals.size()) ||
        uint64_t(target.position) + proposals.size() + 1 > uint64_t(impl_->model.geo.max_context)) {
        return "invalid STQ DFly2 proposal request";
    }
    try {
        impl_->propose(target, anchor, proposals, confidence);
        return {};
    } catch (const std::exception& e) {
        impl_->commands.clear();
        return e.what();
    }
}

} // namespace stark::hy4::stq
