#include "stark_glm.h"

#include <stdio.h>
#include <string.h>

static int check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 0;
    }
    return 1;
}

int main(void) {
    int ok = 1;
    stark_glm_model *model = NULL;
    stark_glm_load_options options;
    stark_glm_info info;
    uint64_t position = 0;
    float logit = 0.0f;
    uint32_t token = 0;
    stark_glm_prefix_snapshot *snapshot = NULL;

    memset(&options, 0, sizeof(options));
    options.struct_size = (uint32_t)sizeof(options);
    options.abi_version = STARK_GLM_ABI_VERSION;
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    info.abi_version = STARK_GLM_ABI_VERSION;

    ok &= check(stark_glm_abi_version() == STARK_GLM_ABI_VERSION,
                "ABI version is 1");
    ok &= check(strcmp(stark_glm_status_name(STARK_GLM_OK), "OK") == 0,
                "OK status name");
    ok &= check(strcmp(stark_glm_status_name(STARK_GLM_INVALID_ARGUMENT),
                       "INVALID_ARGUMENT") == 0,
                "invalid argument status name");
    ok &= check(stark_glm_status_name(UINT32_MAX) != NULL,
                "unknown status has a stable name");

    ok &= check(stark_glm_load(NULL, 0, NULL, NULL) ==
                    STARK_GLM_INVALID_ARGUMENT,
                "load rejects null arguments");
    options.struct_size--;
    ok &= check(stark_glm_load(NULL, 0, &options, &model) ==
                    STARK_GLM_ABI_MISMATCH,
                "load validates struct size before path admission");
    options.struct_size++;

    ok &= check(stark_glm_destroy(NULL) == STARK_GLM_INVALID_ARGUMENT,
                "destroy rejects null pointer-to-handle");
    ok &= check(stark_glm_reset(NULL) == STARK_GLM_INVALID_ARGUMENT,
                "reset rejects null handle");
    ok &= check(stark_glm_drain(NULL) == STARK_GLM_INVALID_ARGUMENT,
                "drain rejects null handle");
    ok &= check(stark_glm_cancel(NULL) == STARK_GLM_INVALID_ARGUMENT,
                "cancel rejects null handle");
    ok &= check(stark_glm_prefill(NULL, &token, 1) ==
                    STARK_GLM_INVALID_ARGUMENT,
                "prefill rejects null handle");
    ok &= check(stark_glm_step(NULL, token) == STARK_GLM_INVALID_ARGUMENT,
                "step rejects null handle");
    ok &= check(stark_glm_readout_f32(NULL, &logit, 1) ==
                    STARK_GLM_INVALID_ARGUMENT,
                "readout rejects null handle");
    ok &= check(stark_glm_get_position(NULL, &position) ==
                    STARK_GLM_INVALID_ARGUMENT,
                "position rejects null handle");
    ok &= check(stark_glm_get_info(NULL, &info) ==
                    STARK_GLM_INVALID_ARGUMENT,
                "info rejects null handle");
    ok &= check(stark_glm_get_rows(NULL, &token, &token) == STARK_GLM_INVALID_ARGUMENT,
                "admitted rows rejects null handle");
    ok &= check(stark_glm_capture_prefix_at(NULL, 16384) == STARK_GLM_INVALID_ARGUMENT,
                "capture rejects null model");
    ok &= check(stark_glm_take_prefix_snapshot(NULL, &snapshot) == STARK_GLM_INVALID_ARGUMENT,
                "snapshot take rejects null model");
    ok &= check(stark_glm_restore_prefix(NULL, NULL, NULL, 0) == STARK_GLM_INVALID_ARGUMENT,
                "restore rejects null model");
    ok &= check(stark_glm_prefix_snapshot_free(&snapshot) == STARK_GLM_OK && snapshot == NULL,
                "empty CPU snapshot release is idempotent");
    ok &= check(stark_glm_validation_state_sha256(NULL, NULL, 0) == STARK_GLM_INVALID_ARGUMENT,
                "validation state hash rejects null model");

    if (!ok) return 1;
    puts("stark_glm ABI smoke passed; no model or MLX runtime was used");
    return 0;
}
