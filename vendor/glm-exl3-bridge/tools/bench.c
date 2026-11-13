#define _POSIX_C_SOURCE 200809L
#include "stark_glm.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct bench_args {
    const char *model_path;
    const char *tokens_path;
    uint64_t context;
    uint64_t ceiling;
    uint32_t runs;
    uint32_t decode;
} bench_args;

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long value;
    if (!s || !*s || *s == '-') return 0;
    errno = 0;
    value = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int set_arg(bench_args *a, int argc, char **argv, int *i) {
    const char *key = argv[*i];
    uint64_t value;
    if (*i + 1 >= argc) return 0;
    if (strcmp(key, "--model") == 0) a->model_path = argv[++*i];
    else if (strcmp(key, "--tokens") == 0) a->tokens_path = argv[++*i];
    else if (strcmp(key, "--context") == 0) {
        if (!parse_u64(argv[++*i], &a->context)) return 0;
    } else if (strcmp(key, "--ceiling") == 0) {
        if (!parse_u64(argv[++*i], &a->ceiling)) return 0;
    } else if (strcmp(key, "--runs") == 0) {
        if (!parse_u64(argv[++*i], &value) || value > UINT32_MAX) return 0;
        a->runs = (uint32_t)value;
    } else if (strcmp(key, "--decode") == 0) {
        if (!parse_u64(argv[++*i], &value) || value > UINT32_MAX) return 0;
        a->decode = (uint32_t)value;
    } else return 0;
    return 1;
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s --model PATH --tokens U32LE_FILE --context N "
            "--ceiling BYTES --runs N --decode N (use --tokens - for stdin)\n", program);
}

static uint32_t read_le_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int load_tokens(const char *path, size_t max_tokens, uint32_t **tokens, size_t *count) {
    if (max_tokens == 0 || max_tokens > 1048576) return 0;
    const size_t max_bytes = max_tokens * sizeof(uint32_t);
    FILE *f = strcmp(path, "-") == 0 ? stdin : fopen(path, "rb");
    if (!f) return 0;
    unsigned char *bytes = malloc(max_bytes);
    if (!bytes) { if (f != stdin) fclose(f); return 0; }
    size_t length = 0;
    while (length < max_bytes) {
        size_t got = fread(bytes + length, 1, max_bytes - length, f);
        length += got;
        if (got == 0) break;
    }
    int over_bound = length == max_bytes && fgetc(f) != EOF;
    int failed = ferror(f) || over_bound || length == 0 || length % 4 != 0;
    if (f != stdin) fclose(f);
    if (failed) { free(bytes); return 0; }
    *tokens = (uint32_t *)bytes;
    *count = length / 4;
    for (size_t i = 0; i < *count; ++i) (*tokens)[i] = read_le_u32(bytes + i * 4);
    return 1;
}

/* FNV-1a over each F32 bit pattern serialized low byte first. */
static uint64_t readout_hash(const float *values, size_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t i;
    for (i = 0; i < count; ++i) {
        uint32_t bits;
        unsigned byte_index;
        memcpy(&bits, &values[i], sizeof(bits));
        for (byte_index = 0; byte_index < 4; ++byte_index) {
            hash ^= (uint8_t)(bits >> (byte_index * 8));
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static uint32_t argmax_finite(const float *values, size_t count, int *valid) {
    uint32_t best = 0;
    size_t i;
    *valid = count > 0;
    if (!*valid) return 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) { *valid = 0; return 0; }
        if (values[i] > values[best]) best = (uint32_t)i;
    }
    return best;
}

static int fail_status(const char *operation, stark_glm_status status) {
    fprintf(stderr, "%s failed: %s (%u)\n", operation,
            stark_glm_status_name((uint32_t)status), (unsigned)status);
    return 0;
}

int main(int argc, char **argv) {
    bench_args args = {0};
    uint32_t *tokens = NULL;
    size_t token_count = 0;
    stark_glm_model *model = NULL;
    stark_glm_load_options options = {0};
    stark_glm_info info = {0};
    uint32_t prompt_rows = 0, decode_rows = 0;
    uint64_t wired_capacity = 0, previous_wired_capacity = 0;
    float *readout = NULL;
    stark_glm_status status;
    int result = 1;
    for (int i = 1; i < argc; ++i) {
        if (!set_arg(&args, argc, argv, &i)) { usage(argv[0]); return 2; }
    }
    if (!args.model_path || !args.tokens_path || args.context == 0 ||
        args.context > 1048576 || args.ceiling == 0 || args.runs < 2 ||
        args.runs > 1000 || args.decode == 0 || args.decode >= args.context) {
        usage(argv[0]); return 2;
    }
    /* Admission bounds the file before either token allocation. */
    if (!load_tokens(args.tokens_path, (size_t)(args.context - args.decode),
                     &tokens, &token_count)) {
        fputs("invalid or over-context U32LE token file\n", stderr); return 2;
    }
    options.struct_size = (uint32_t)sizeof(options);
    options.abi_version = STARK_GLM_ABI_VERSION;
    options.context_tokens = args.context;
    options.gpu_ceiling_bytes = args.ceiling;
    info.struct_size = (uint32_t)sizeof(info);
    info.abi_version = STARK_GLM_ABI_VERSION;
    double load_start = now_seconds();
    status = stark_glm_load((const uint8_t *)args.model_path,
                            strlen(args.model_path), &options, &model);
    double load_seconds = now_seconds() - load_start;
    if (status != STARK_GLM_OK) { fail_status("load", status); goto done; }
    status = stark_glm_get_info(model, &info);
    if (status != STARK_GLM_OK) { fail_status("get_info", status); goto done; }
    status = stark_glm_get_rows(model, &prompt_rows, &decode_rows);
    if (status != STARK_GLM_OK) { fail_status("get_rows", status); goto done; }
    status = stark_glm_get_wired_capacity(model, &wired_capacity, &previous_wired_capacity);
    if (status != STARK_GLM_OK) { fail_status("get_wired_capacity", status); goto done; }
    if (info.vocab_size == 0 || info.vocab_size > 1048576 ||
        (info.capabilities & STARK_GLM_CAP_FULL_F32_READOUT) == 0) {
        fputs("unsupported readout metadata\n", stderr); goto done;
    }
    readout = malloc((size_t)info.vocab_size * sizeof(float));
    if (!readout) { fputs("readout allocation failed\n", stderr); goto done; }
    printf("abi_version=%u warmup_runs=1 measured_runs=%u prompt_tokens=%zu "
           "decode_tokens_per_run=%u load_seconds=%.6f "
           "expert_prefill_rows=%u expert_rows=%u "
           "wired_capacity_bytes=%llu previous_wired_capacity_bytes=%llu "
           "hash_encoding=fnv1a64-f32-little-endian\n",
           stark_glm_abi_version(), args.runs, token_count, args.decode, load_seconds,
           prompt_rows, decode_rows,
           (unsigned long long)wired_capacity, (unsigned long long)previous_wired_capacity);
    fflush(stdout);
    for (uint32_t run = 0; run <= args.runs; ++run) {
        status = stark_glm_reset(model);
        if (status != STARK_GLM_OK) { fail_status("reset", status); goto done; }
        double prefill_start = now_seconds();
        status = stark_glm_prefill(model, tokens, token_count);
        if (status == STARK_GLM_OK)
            status = stark_glm_readout_f32(model, readout, info.vocab_size);
        double prefill_seconds = now_seconds() - prefill_start;
        if (status != STARK_GLM_OK) { fail_status("prefill_readout", status); goto done; }
        uint64_t prefill_hash = readout_hash(readout, info.vocab_size);
        int valid = 0;
        uint32_t next = argmax_finite(readout, info.vocab_size, &valid);
        if (!valid) { fputs("non-finite readout\n", stderr); goto done; }
        double decode_start = now_seconds();
        for (uint32_t i = 0; i < args.decode; ++i) {
            status = stark_glm_step(model, next);
            if (status == STARK_GLM_OK)
                status = stark_glm_readout_f32(model, readout, info.vocab_size);
            if (status != STARK_GLM_OK) { fail_status("decode_readout", status); goto done; }
            next = argmax_finite(readout, info.vocab_size, &valid);
            if (!valid) { fputs("non-finite readout\n", stderr); goto done; }
        }
        status = stark_glm_drain(model);
        double decode_seconds = now_seconds() - decode_start;
        if (status != STARK_GLM_OK) { fail_status("drain", status); goto done; }
        uint64_t final_hash = readout_hash(readout, info.vocab_size);
        if (run == 0) {
            puts("warmup_complete=1 excluded_from_measurements=1");
        } else {
            printf("run=%u prefill_seconds=%.6f prefill_tokens_per_second=%.3f "
                   "decode_seconds=%.6f decode_tokens_per_second=%.3f "
                   "prefill_full_readout_fnv1a64=%016llx final_full_readout_fnv1a64=%016llx\n",
                   run, prefill_seconds, token_count / prefill_seconds,
                   decode_seconds, args.decode / decode_seconds,
                   (unsigned long long)prefill_hash, (unsigned long long)final_hash);
        }
        fflush(stdout);
    }
    result = 0;
done:
    if (model) {
        status = stark_glm_destroy(&model);
        if (status != STARK_GLM_OK) { fail_status("destroy", status); result = 1; }
    }
    free(readout);
    free(tokens);
    return result;
}
