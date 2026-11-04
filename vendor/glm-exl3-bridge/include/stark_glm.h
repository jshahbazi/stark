#ifndef STARK_GLM_H
#define STARK_GLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STARK_GLM_ABI_VERSION 1u

typedef struct stark_glm_model stark_glm_model;
typedef struct stark_glm_prefix_snapshot stark_glm_prefix_snapshot;

typedef uint32_t stark_glm_status;
enum {
    STARK_GLM_OK = 0,
    STARK_GLM_INVALID_ARGUMENT = 1,
    STARK_GLM_ABI_MISMATCH = 2,
    STARK_GLM_CONTEXT_LIMIT = 3,
    STARK_GLM_CANCELLED = 4,
    STARK_GLM_UNSUPPORTED = 5,
    STARK_GLM_OUT_OF_MEMORY = 6,
    STARK_GLM_IO_FAILURE = 7,
    STARK_GLM_DEVICE_FAILURE = 8,
    STARK_GLM_RELOAD_REQUIRED = 9,
    STARK_GLM_BUSY = 10,
    STARK_GLM_INVALID_STATE = 11
};

#define STARK_GLM_CAP_FULL_F32_READOUT UINT64_C(1)
#define STARK_GLM_CAP_SERIAL_DECODE UINT64_C(2)
#define STARK_GLM_CAP_PREFIX_SNAPSHOT UINT64_C(4)

typedef struct stark_glm_load_options {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t context_tokens;
    uint64_t gpu_ceiling_bytes;
    uint64_t wired_margin_bytes;
    uint32_t expert_rows;
    uint32_t expert_prefill_rows;
} stark_glm_load_options;

typedef struct stark_glm_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t vocab_size;
    uint32_t prefill_chunk_tokens;
    uint64_t context_tokens;
    uint64_t position;
    uint64_t capabilities;
    uint32_t fenced;
    uint32_t reserved;
} stark_glm_info;

uint32_t stark_glm_abi_version(void);
const char *stark_glm_status_name(uint32_t status);

stark_glm_status stark_glm_load(const uint8_t *path_utf8, size_t path_len,
                                const stark_glm_load_options *options,
                                stark_glm_model **out_model);
stark_glm_status stark_glm_destroy(stark_glm_model **model);
stark_glm_status stark_glm_reset(stark_glm_model *model);
stark_glm_status stark_glm_drain(stark_glm_model *model);
stark_glm_status stark_glm_cancel(stark_glm_model *model);
stark_glm_status stark_glm_prefill(stark_glm_model *model,
                                   const uint32_t *token_ids, size_t count);
stark_glm_status stark_glm_step(stark_glm_model *model, uint32_t token_id);
stark_glm_status stark_glm_readout_f32(stark_glm_model *model, float *dst,
                                       size_t count);
stark_glm_status stark_glm_get_position(stark_glm_model *model,
                                        uint64_t *out_position);
stark_glm_status stark_glm_get_info(stark_glm_model *model,
                                    stark_glm_info *info);
/* Actual admitted expert rows from the installed module plan. */
stark_glm_status stark_glm_get_rows(stark_glm_model *model,
                                    uint32_t *prompt_rows, uint32_t *decode_rows);
/* Installed process-local MLX capacity and its saved pre-load setting. The
 * current capacity is bounded by this model's admitted budget. Previous may
 * exceed that budget; destruction restores it after freeing model buffers.
 */
stark_glm_status stark_glm_get_wired_capacity(stark_glm_model *model,
                                            uint64_t *capacity, uint64_t *previous);

/* Compiled with -Dexperimental-prefix=true; advertised by PREFIX_SNAPSHOT.
 * This reports implementation availability. Artifact qualification is separate.
 * Capture must be armed before prefill at a positive 16384-token boundary.
 * Restore follows reset and receives the complete new prompt to verify source
 * ownership privately. Snapshots contain private CPU state and token IDs.
 * Never persist/log them. Any fresh prompt, cancellation, failure or model
 * destruction invalidates reuse. Free CPU snapshot ownership explicitly.
 * take returns OK and a null output when no new capture is available.
 */
stark_glm_status stark_glm_capture_prefix_at(stark_glm_model *model,
                                            uint64_t position);
stark_glm_status stark_glm_take_prefix_snapshot(stark_glm_model *model,
                                               stark_glm_prefix_snapshot **out);
stark_glm_status stark_glm_restore_prefix(stark_glm_model *model,
                                          const stark_glm_prefix_snapshot *snapshot,
                                          const uint32_t *full_prompt, size_t count);
stark_glm_status stark_glm_prefix_snapshot_free(stark_glm_prefix_snapshot **snapshot);
/* Validation build only; SHA256 of complete live numerical KDA/DSA state and
 * position/context, with no token IDs. Expensive, synchronous, never timed.
 * Returns UNSUPPORTED in a normal serial build. count must equal 32.
 */
stark_glm_status stark_glm_validation_state_sha256(stark_glm_model *model,
                                                  uint8_t *dst, size_t count);

#ifdef __cplusplus
}
#endif

#endif
