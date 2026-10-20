// Thin measurement driver for unmodified llama.cpp b11747.
// Fetch include/llama.h and its ggml includes at commit
// f711c8f3038020d71f69b8ced09162b014fd7335. Build with:
// clang++ -std=c++17 -O3 -I INCLUDE scripts/stock_compare_llama.cpp \
//   -L RELEASE -lllama -Wl,-rpath,ABSOLUTE_RELEASE -o DRIVER
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Readout {
    std::vector<double> logits, log_probabilities, probabilities;
};

static void quiet(enum ggml_log_level, const char *, void *) {}

static void error(const char *code) {
    std::printf("{\"error\":\"%s\"}\n", code);
    std::fflush(stdout);
}

static bool normalize(const float *full, int vocabulary,
                      const std::vector<int> &candidates, double temperature,
                      Readout &out) {
    if (vocabulary <= 0 || candidates.empty() || !(temperature > 0) ||
        !std::isfinite(temperature)) return false;
    double maximum = -INFINITY;
    for (int i = 0; i < vocabulary; ++i) {
        if (!std::isfinite(full[i])) return false;
        maximum = std::max(maximum, double(full[i]));
    }
    double total = 0;
    for (int i = 0; i < vocabulary; ++i)
        total += std::exp(double(full[i]) - maximum);
    const double log_normalizer = maximum + std::log(total);
    double candidate_maximum = -INFINITY;
    out = {};
    for (int id : candidates) {
        if (id < 0 || id >= vocabulary) return false;
        out.logits.push_back(full[id]);
        out.log_probabilities.push_back(double(full[id]) - log_normalizer);
        candidate_maximum = std::max(candidate_maximum, double(full[id]));
    }
    double candidate_total = 0;
    for (double logit : out.logits) {
        const double value = std::exp((logit - candidate_maximum) / temperature);
        out.probabilities.push_back(value);
        candidate_total += value;
    }
    for (double &value : out.probabilities) value /= candidate_total;
    return true;
}

static void vector_json(const std::vector<double> &values) {
    std::printf("[");
    for (size_t i = 0; i < values.size(); ++i)
        std::printf("%s%.17g", i ? "," : "", values[i]);
    std::printf("]");
}

static void score(llama_model *model, int context, double temperature,
                  const std::vector<llama_token> &ids,
                  const std::vector<int> &candidates) {
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (ids.empty() || ids.size() >= size_t(context) || candidates.empty()) {
        error("invalid_request");
        return;
    }
    for (int id : ids) {
        if (id < 0 || id >= vocabulary) {
            error("input_out_of_range");
            return;
        }
    }
    for (int id : candidates) {
        if (id < 0 || id >= vocabulary) {
            error("candidate_out_of_range");
            return;
        }
    }
    const auto started = std::chrono::steady_clock::now();
    auto params = llama_context_default_params();
    params.n_ctx = context;
    params.n_batch = std::min<uint32_t>(params.n_batch, context);
    params.n_ubatch = std::min(params.n_ubatch, params.n_batch);
    params.n_seq_max = 1;
    params.no_perf = true;
    llama_context *state = llama_init_from_model(model, params);
    if (!state) {
        error("context_init_failed");
        return;
    }
    auto batch = llama_batch_init(params.n_batch, 0, 1);
    bool decoded = true;
    for (size_t at = 0; at < ids.size();) {
        const int count = std::min<size_t>(params.n_batch, ids.size() - at);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = ids[at + i];
            batch.pos[i] = at + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = at + i + 1 == ids.size();
        }
        if (llama_decode(state, batch) != 0) {
            decoded = false;
            break;
        }
        at += count;
    }
    Readout output;
    const float *logits = decoded ? llama_get_logits_ith(state, -1) : nullptr;
    const bool valid = logits && normalize(logits, vocabulary, candidates, temperature, output);
    llama_batch_free(batch);
    llama_free(state);
    if (!valid) {
        error(decoded ? "nonfinite_output" : "decode_failed");
        return;
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::printf("{\"input_tokens\":%zu,\"logits\":", ids.size());
    vector_json(output.logits);
    std::printf(",\"log_probabilities\":");
    vector_json(output.log_probabilities);
    std::printf(",\"candidate_probabilities\":");
    vector_json(output.probabilities);
    std::printf(",\"score_seconds\":%.17g}\n", elapsed);
    std::fflush(stdout);
}

static int self_test() {
    const float values[] = {0, 1, 2};
    Readout result;
    if (!normalize(values, 3, {0, 2}, 2, result)) return 1;
    if (std::abs(result.log_probabilities[0] + 2.4076059644443803) > 1e-12 ||
        std::abs(result.log_probabilities[1] + 0.4076059644443803) > 1e-12 ||
        std::abs(result.probabilities[0] - 0.2689414213699951) > 1e-12 ||
        std::abs(result.probabilities[1] - 0.7310585786300049) > 1e-12)
        return 1;
    const float shifted[] = {1000, 1001, 1002};
    Readout stable;
    if (!normalize(shifted, 3, {0, 2}, 2, stable) ||
        std::abs(stable.probabilities[0] - result.probabilities[0]) > 1e-12)
        return 1;
    const float invalid[] = {0, INFINITY};
    if (normalize(invalid, 2, {0}, 1, stable) ||
        normalize(values, 3, {3}, 1, stable) ||
        normalize(values, 3, {0}, 0, stable))
        return 1;
    std::puts("self_test_passed");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
    if (argc != 2) {
        std::fprintf(stderr, "usage: stock_compare_llama MODEL | --self-test\n");
        return 2;
    }
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = -1;
    llama_model *model = llama_model_load_from_file(argv[1], params);
    if (!model) {
        error("model_load_failed");
        llama_backend_free();
        return 1;
    }
    std::printf("{\"ready\":true,\"vocab_size\":%d}\n",
                llama_vocab_n_tokens(llama_model_get_vocab(model)));
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string operation, extra;
        int context, count, candidate_count;
        double temperature;
        if (!(input >> operation >> context >> temperature >> count >> candidate_count) ||
            operation != "score" || context <= 0 || context > 16384 ||
            count <= 0 || count >= context || candidate_count <= 0 ||
            candidate_count > 128 || !std::isfinite(temperature) || temperature <= 0) {
            error("invalid_request");
            continue;
        }
        std::vector<llama_token> ids(count);
        std::vector<int> candidates(candidate_count);
        bool valid = true;
        for (auto &id : ids) valid = bool(input >> id) && valid;
        for (auto &id : candidates) valid = bool(input >> id) && valid;
        if (!valid || input >> extra) {
            error("invalid_request");
            continue;
        }
        score(model, context, temperature, ids, candidates);
    }
    llama_model_free(model);
    llama_backend_free();
}
