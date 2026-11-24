// Source-bound experiment client for Unsloth llama.cpp b12572-mix-8eebad4.
// Link against its released dylibs and compile against the matching source headers.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "llama.h"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Capture {
    int layer, down, tokens, channels, width;
    std::vector<float> input;
    std::vector<int32_t> ids;
};

std::vector<Capture> captures;
ggml_backend_t metal = nullptr;
struct Span { int visit, stage; double seconds; };
std::vector<Span> spans;
struct Readout {
    int count, finite;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
};
std::vector<Readout> readouts;
bool profile = false;
Clock::time_point previous;

void silent_log(ggml_log_level, const char *, void *) {}
bool record_readout(llama_context * context, int count) {
    const float * logits = llama_get_logits_ith(context, -1);
    if (!logits || count <= 0) return false;
    Readout result{count, 0, {}};
    for (int i = 0; i < count; ++i) result.finite += std::isfinite(logits[i]);
    CC_SHA256(logits, count*sizeof(float), result.digest);
    readouts.push_back(result);
    return true;
}
bool target_bank(const ggml_tensor * node, int & layer, int & down) {
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0]->type != GGML_TYPE_NVFP4) return false;
    for (int physical : {6, 11}) {
        const std::string prefix = "blk." + std::to_string(physical) + ".ffn_";
        const std::string name = ggml_get_name(node->src[0]);
        if (name == prefix + "gate_up_exps.weight" || name == prefix + "down_exps.weight") {
            layer = physical;
            down = name == prefix + "down_exps.weight";
            return true;
        }
    }
    return false;
}

template<class T>
std::vector<T> copy_rows(const ggml_tensor * t, int width, int channels, int tokens) {
    std::vector<T> dest(static_cast<size_t>(width) * channels * tokens);
    if (t->nb[0] != sizeof(T)) throw std::runtime_error("unsupported capture layout");
    // ids may be a strided top-k view into all 128 expert slots.
    for (int z = 0; z < tokens; ++z) {
        for (int y = 0; y < channels; ++y) {
            ggml_backend_tensor_get(t, dest.data() + (z*channels + y)*width,
                                    z*t->nb[2] + y*t->nb[1], width*sizeof(T));
        }
    }
    return dest;
}

bool capture_node(ggml_tensor * node, bool ask, void *) {
    if (profile) {
        const std::string name = ggml_get_name(node);
        int visit, length = 0, stage = -1;
        if (std::sscanf(name.c_str(), "ffn_moe_logits-%d%n", &visit, &length) == 1 &&
            size_t(length) == name.size()) stage = 0;
        else if (std::sscanf(name.c_str(), "l_out-%d%n", &visit, &length) == 1 &&
                 size_t(length) == name.size()) stage = 1;
        if (stage < 0) return false;
        if (!ask) {
            auto now = Clock::now();
            spans.push_back({visit, stage, std::chrono::duration<double>(now-previous).count()});
            previous = now;
        }
        return true;
    }
    int layer = 0, down = 0;
    if (!target_bank(node, layer, down)) return false;
    if (ask) return true;
    const ggml_tensor * x = node->src[1];
    const ggml_tensor * ids = node->src[2];
    if (x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        x->ne[3] != 1 || ids->ne[0] != 8 || ids->ne[1] != x->ne[2] ||
        x->ne[1] != (down ? 8 : 1)) throw std::runtime_error("unexpected captured tensor");
    Capture c{layer, down, int(x->ne[2]), int(x->ne[1]), int(x->ne[0]), {}, {}};
    c.input = copy_rows<float>(x, c.width, c.channels, c.tokens);
    c.ids = copy_rows<int32_t>(ids, 8, c.tokens, 1);
    captures.push_back(std::move(c));
    return true;
}
}

extern "C" int capture_visits(const char * path, const char * fixture, int rows) {
    ggml_log_set(silent_log, nullptr);
    llama_log_set(silent_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    auto * model = llama_model_load_from_file(path, mp);
    if (!model) return -1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_seq_max = 1;
    cp.cb_eval = capture_node;
    auto * context = llama_init_from_model(model, cp);
    if (!context) { llama_model_free(model); return -2; }
    std::vector<llama_token> tokens(std::strlen(fixture) + 16);
    const int n = llama_tokenize(llama_model_get_vocab(model), fixture, std::strlen(fixture),
                                 tokens.data(), tokens.size(), true, false);
    int status = -3;
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n >= rows + 2) {
        status = llama_decode(context, llama_batch_get_one(tokens.data(), rows));
        if (status == 0 && !record_readout(context, vocab_size)) status = -5;
        if (status == 0) status = llama_decode(context, llama_batch_get_one(tokens.data()+rows, 1));
        llama_synchronize(context);
        if (status == 0 && !record_readout(context, vocab_size)) status = -5;
        if (!status) {
            profile = true;
            previous = Clock::now();
            status = llama_decode(context, llama_batch_get_one(tokens.data()+rows+1, 1));
            llama_synchronize(context);
            profile = false;
            if (status == 0 && !record_readout(context, vocab_size)) status = -5;
        }
    }
    llama_free(context);
    llama_model_free(model);
    return status == 0 ? int(captures.size()) : -4;
}
extern "C" int readout_count() { return int(readouts.size()); }
extern "C" void readout_meta(int index, int32_t * counts, unsigned char * digest) {
    const Readout & result = readouts.at(index);
    counts[0] = result.count;
    counts[1] = result.finite;
    std::memcpy(digest, result.digest, sizeof(result.digest));
}
extern "C" int span_count() { return int(spans.size()); }
extern "C" double read_span(int index, int32_t * visit, int32_t * stage) {
    const Span & s = spans.at(index);
    *visit = s.visit;
    *stage = s.stage;
    return s.seconds;
}

extern "C" void capture_meta(int index, int32_t * out) {
    const Capture & c = captures.at(index);
    const int32_t meta[]{c.layer,c.down,c.tokens,c.channels,c.width};
    std::memcpy(out, meta, sizeof(meta));
}
extern "C" void capture_copy(int index, float * input, int32_t * ids) {
    const Capture & c = captures.at(index);
    std::memcpy(input, c.input.data(), c.input.size()*sizeof(float));
    std::memcpy(ids, c.ids.data(), c.ids.size()*sizeof(int32_t));
}

extern "C" int probe_projection(int width, int rows, int experts, int tokens, int channels,
    const void * bank, size_t bytes, const float * input, const int32_t * ids, const float * scales,
    float * output, double * seconds, int repeats, int n_fixtures, const float * input_scales, int replay_rows) {
    ggml_log_set(silent_log, nullptr);
    if (!metal) metal = ggml_backend_metal_init();
    if (!metal || ggml_backend_dev_type(ggml_backend_get_device(metal)) != GGML_BACKEND_DEVICE_TYPE_GPU) return -1;
    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, width, rows, experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, tokens);
    ggml_tensor * i = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, tokens);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, experts);
    ggml_tensor * outer = input_scales ? ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,channels,tokens) : nullptr;
    auto project = [&](ggml_tensor * values, ggml_tensor * ids, ggml_tensor * factors) {
        ggml_tensor * product = ggml_mul_mat_id(ctx,w,values,ids);
        ggml_prec_set_acc(product, GGML_PREC_BF16);  // Match build_lora_mm_id for NVFP4.
        auto * gathered = ggml_get_rows(ctx,scale,ggml_reshape_1d(ctx,ids,ggml_nelements(ids)));
        auto * combined = ggml_reshape_3d(ctx,gathered,1,8,ids->ne[1]);
        if (factors) combined = ggml_mul(ctx,combined,factors);
        return ggml_mul(ctx,product,combined);
    };
    auto * graph = ggml_new_graph(ctx);
    ggml_tensor * y;
    if (!replay_rows) {
        y=project(x,i,outer);
        ggml_build_forward_expand(graph,y);
    } else {
        y=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,rows,8,tokens);
        for (int t=0; t<tokens; ++t) {
            auto view = [&](ggml_tensor * a) {
                return ggml_view_3d(ctx,a,a->ne[0],a->ne[1],1,a->nb[1],a->nb[2],t*a->nb[2]);
            };
            auto * row_ids = ggml_view_2d(ctx,i,8,1,i->nb[1],t*i->nb[1]);
            auto * result = project(view(x),row_ids,outer ? view(outer) : nullptr);
            ggml_build_forward_expand(graph,ggml_cpy(ctx,result,view(y)));
        }
    }
    if (ggml_nbytes(w) != bytes) return -2;
    for (int index=0; index<ggml_graph_n_nodes(graph); ++index) {
        if (!ggml_backend_supports_op(metal,ggml_graph_node(graph,index))) return -3;
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, metal);
    if (!buffer) return -4;
    ggml_backend_tensor_set(w, bank, 0, bytes);
    ggml_backend_tensor_set(scale, scales, 0, ggml_nbytes(scale));
    for (int repeat = -2; repeat <= repeats; ++repeat) {
        // The final, untimed fixture is the one independently checked by the caller.
        const int fixture = repeat < 0 || repeat == repeats ? 0 : repeat % n_fixtures;
        ggml_backend_tensor_set(x, input + fixture*ggml_nelements(x), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(i, ids + fixture*ggml_nelements(i), 0, ggml_nbytes(i));
        if (outer) ggml_backend_tensor_set(outer, input_scales+fixture*ggml_nelements(outer), 0, ggml_nbytes(outer));
        auto begin = Clock::now();
        if (ggml_backend_graph_compute(metal,graph) != GGML_STATUS_SUCCESS) return -5;
        ggml_backend_synchronize(metal);
        if (repeat >= 0 && repeat < repeats)
            seconds[repeat] = std::chrono::duration<double>(Clock::now()-begin).count();
    }
    ggml_backend_tensor_get(y, output, 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return 0;
}
