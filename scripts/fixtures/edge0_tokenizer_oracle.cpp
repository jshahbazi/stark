// Stark-authored vocabulary-only qualification bridge; no upstream implementation copied.
// Format/API authority: ggml-org/llama.cpp at
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT reference; provenance/UPSTREAM.md).
// The original private-run source was 4176 bytes, SHA-256
// 6bcce07c37928babf8b7b7ec61af24609b0c95a0d10ea6bc2fb1d203e899db11.
// The original body remains below. A separate 03B Unicode property API follows it.
//
// Build the exact checkout, without tests/tools/server or GPU backends:
// cmake -S SOURCE -B BUILD -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
//   -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF
//   -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF
//   -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF
//   -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
//   -DLLAMA_OPENSSL=OFF
// cmake --build BUILD --target llama -j4
// macOS bridge/probe recipe (substitute actual paths; run serialized):
// clang++ -std=c++17 -O2 -dynamiclib THIS.cpp -I SOURCE/include
//   -I SOURCE/ggml/include -I SOURCE/src -L BUILD/bin -lllama -Wl,-rpath,ABS_BUILD/bin
//   -o libedge0_oracle.dylib
// Replace -dynamiclib with -DEDGE0_PROBE_MAIN to build the native smoke executable.
// Use only a sealed, derived metadata-only GGUF with tensor_count=0 and no
// tensor descriptors/payloads. The API retains the original vocabulary metadata.
// Logging is silenced before loading. No llama_context or inference is created.
// Tokenization uses add_special=false and exposes parse_special independently;
// token-to-piece returns raw bytes with an explicit render-special flag.
// Any caller supplying a UTF-8 replacement policy qualifies that boundary separately.
// Keep comparison text/ID vectors in memory; persist hashes and aggregate results.
//
// Local metadata-only qualification shim; no inference/context API is used.
#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

static void quiet_log(enum ggml_log_level, const char *, void *) {}

extern "C" {
void * edge0_load(const char * path) {
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.vocab_only = true;
    params.check_tensors = false;
    params.load_mtp = false;
    return llama_model_load_from_file(path, params);
}
void edge0_free(void * model) { llama_model_free(static_cast<llama_model *>(model)); }
int32_t edge0_n_vocab(void * model) {
    return llama_vocab_n_tokens(llama_model_get_vocab(static_cast<llama_model *>(model)));
}
int32_t edge0_encode(void * model, const char * text, int32_t len, int32_t * tokens, int32_t capacity, bool parse) {
    return llama_tokenize(llama_model_get_vocab(static_cast<llama_model *>(model)), text, len, tokens, capacity, false, parse);
}
int32_t edge0_piece(void * model, int32_t token, char * text, int32_t capacity, bool special) {
    return llama_token_to_piece(llama_model_get_vocab(static_cast<llama_model *>(model)), token, text, capacity, 0, special);
}
int32_t edge0_attr(void * model, int32_t token) {
    return llama_vocab_get_attr(llama_model_get_vocab(static_cast<llama_model *>(model)), token);
}
bool edge0_is_eog(void * model, int32_t token) {
    return llama_vocab_is_eog(llama_model_get_vocab(static_cast<llama_model *>(model)), token);
}
int32_t edge0_special(void * model, int32_t code) {
    const auto * v = llama_model_get_vocab(static_cast<llama_model *>(model));
    switch (code) {
        case 0: return llama_vocab_bos(v);
        case 1: return llama_vocab_eos(v);
        case 2: return llama_vocab_eot(v);
        case 3: return llama_vocab_sep(v);
        case 4: return llama_vocab_nl(v);
        case 5: return llama_vocab_pad(v);
        case 6: return llama_vocab_mask(v);
        case 7: return llama_vocab_fim_pre(v);
        case 8: return llama_vocab_fim_suf(v);
        case 9: return llama_vocab_fim_mid(v);
        case 10: return llama_vocab_fim_pad(v);
        case 11: return llama_vocab_fim_rep(v);
        case 12: return llama_vocab_fim_sep(v);
        default: return -1;
    }
}
}

#ifdef EDGE0_PROBE_MAIN
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    void * model = edge0_load(argv[1]);
    if (!model) return 3;
    int32_t ids[1024];
    const char input[3] = { 'a', 0, 'b' };
    const auto * v = llama_model_get_vocab(static_cast<llama_model *>(model));
    const auto * special = llama_vocab_get_text(v, llama_vocab_eos(v));
    int32_t encode_plain = edge0_encode(model, input, 3, ids, 1024, false);
    int32_t encode_parse = edge0_encode(model, input, 3, ids, 1024, true);
    int32_t special_plain = edge0_encode(model, special, std::strlen(special), ids, 1024, false);
    int32_t special_parse = edge0_encode(model, special, std::strlen(special), ids, 1024, true);
    char bytes[4096];
    int32_t rendered = edge0_piece(model, llama_vocab_eos(v), bytes, 4096, true);
    int32_t skipped = edge0_piece(model, llama_vocab_eos(v), bytes, 4096, false);
    int32_t unused_rendered = edge0_piece(model, edge0_n_vocab(model)-1, bytes, 4096, true);
    int32_t unused_skipped = edge0_piece(model, edge0_n_vocab(model)-1, bytes, 4096, false);
    int32_t n_eog = 0;
    for (int32_t i = 0; i < edge0_n_vocab(model); ++i) n_eog += edge0_is_eog(model, i);
    const bool ok = encode_plain > 0 && encode_parse > 0 && special_plain > 0 && special_parse > 0;
    std::printf("{\"loaded\":true,\"n_vocab\":%d,\"encode_modes_work\":%s,\"ordinary_parse_false_tokens\":%d,\"ordinary_parse_true_tokens\":%d,\"eos_parse_false_tokens\":%d,\"eos_parse_true_tokens\":%d,\"eos_render_bytes\":%d,\"eos_skip_bytes\":%d,\"last_unused_render_bytes\":%d,\"last_unused_skip_bytes\":%d,\"eog_count\":%d}\n", edge0_n_vocab(model), ok ? "true" : "false", encode_plain, encode_parse, special_plain, special_parse, rendered, skipped, unused_rendered, unused_skipped, n_eog);
    edge0_free(model);
    return ok ? 0 : 4;
}
#endif

// 20261112-EDGE0-03B: source property inventory, not tokenizer/Unicode data copied
// into Stark. The caller supplies a bounded buffer and retains results in memory.
#include "unicode.h"
extern "C" int32_t edge0_unicode_flags(uint16_t * output, uint32_t capacity) {
    constexpr uint32_t scalar_slots = 0x110000;
    if (!output || capacity < scalar_slots) return -static_cast<int32_t>(scalar_slots);
    for (uint32_t cp = 0; cp < scalar_slots; ++cp) {
        output[cp] = unicode_cpt_flags_from_cpt(cp).as_uint();
    }
    return static_cast<int32_t>(scalar_slots);
}

// 20261112-EDGE0-03B.1: consecutive raw UTF-8 piece lengths from the pinned
// qwen35 splitter, before BPE or added-token extraction. No bytes are emitted.
extern "C" int32_t edge0_split_lengths(const char * text, int32_t text_size,
                                        const char * pattern, int32_t pattern_size,
                                        uint32_t * output, int32_t capacity) {
    if (!text || !pattern || text_size < 0 || text_size > 1000000 ||
        pattern_size < 1 || pattern_size > 100000 || capacity < 0) return INT32_MIN;
    try {
        const auto pieces = unicode_regex_split(std::string(text, text_size),
                                                 {std::string(pattern, pattern_size)}, false);
        if (pieces.size() > static_cast<size_t>(INT32_MAX)) return INT32_MIN;
        const auto count = static_cast<int32_t>(pieces.size());
        if (count > capacity || (count && !output)) return -count;
        size_t bytes = 0;
        for (int32_t i = 0; i < count; ++i) {
            output[i] = static_cast<uint32_t>(pieces[i].size());
            bytes += pieces[i].size();
        }
        if (bytes != static_cast<size_t>(text_size)) return INT32_MIN;
        return count;
    } catch (...) {
        return INT32_MIN; // No exception text or corpus bytes leave this bridge.
    }
}
