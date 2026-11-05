#ifndef STARK_GLM_H
#define STARK_GLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STARK_GLM_ABI_VERSION 1u
#define STARK_GLM_NATIVE_DECODE_API_VERSION 1u

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
/* Reference builds leave these bits clear. The explicit experimental native
 * profile or a private qualification
 * stage may expose implementation availability for an admitted head before
 * worker/seeded/quality qualification. This is not production qualification. */
#define STARK_GLM_CAP_NATIVE_DECODE UINT64_C(8)
#define STARK_GLM_CAP_MTP_PREFIX_SNAPSHOT UINT64_C(16)

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

typedef struct stark_glm_native_sampling {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t mode; /* 0: greedy; 1: temperature/top-p, position-keyed seed */
    uint32_t reserved;
    float temperature;
    float top_p;
    uint64_t seed;
} stark_glm_native_sampling;

typedef struct stark_glm_native_round_result {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t count;
    uint32_t accepted;
    uint32_t verified_drafts; /* gate-admitted proposals, not all head work */
    uint32_t next_pending;
    uint32_t tokens[4]; /* committed outputs only; tokens[0] is input pending */
    uint64_t position;
} stark_glm_native_round_result;

uint32_t stark_glm_abi_version(void);
const char *stark_glm_status_name(uint32_t status);

stark_glm_status stark_glm_load(const uint8_t *path_utf8, size_t path_len,
                                const stark_glm_load_options *options,
                                stark_glm_model **out_model);
/* Separate native loader. The bf16-native-experimental stage profile advertises
 * native/head-prefix implementation availability with quality still experimental.
 * The original loader remains the headless F32
 * reference. Head preload is fully admitted even while parked for MTP-off.
 * Request MTP defaults off and may change only after load/reset, before prefill.
 * These entrypoints do not imply advertised or numerical qualification.
 */
stark_glm_status stark_glm_load_native(const uint8_t *path_utf8, size_t path_len,
    const stark_glm_load_options *options, uint32_t load_mtp_head,
    stark_glm_model **out_model);
stark_glm_status stark_glm_set_mtp_enabled(stark_glm_model *model, uint32_t enabled);
/* Call after raw prefill/restore, once. Sampling is copied, never retained by
 * pointer. The result is pending: it is not committed to native state yet.
 */
stark_glm_status stark_glm_native_initial(stark_glm_model *model,
    const stark_glm_native_sampling *sampling, uint32_t *out_pending);
/* Call with the previous pending ID and cap<=3. Returned tokens alone count
 * against the native output budget; next_pending is input to a later round.
 * For total budget B, cap=min(depth,B-committed-1). A budget of one requires
 * an eager cap0 commit even if the initial token was already made visible.
 * No extra stream-wide fence is added after the module's decision completes.
 */
stark_glm_status stark_glm_native_round(stark_glm_model *model,
    uint32_t pending, uint32_t accepted_cap, stark_glm_native_round_result *out);
/* Drain/request-end only: never samples or commits. Position and taken prompt
 * snapshot ownership survive a successful end; next request still needs reset.
 * Partial-block continuation after end is unsupported. End is idempotent.
 */
stark_glm_status stark_glm_native_end_request(stark_glm_model *model);
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
 * original loader bounds capacity by its admitted bill; the staged native
 * loader uses Metal's recommended maximum while allocations remain admitted.
 * Destruction restores the previous setting after freeing model buffers.
 */
stark_glm_status stark_glm_get_wired_capacity(stark_glm_model *model,
                                            uint64_t *capacity, uint64_t *previous);

/* Compiled with -Dexperimental-prefix=true; headless availability is advertised
 * by PREFIX_SNAPSHOT. The staged MTP implementation is callable for direct
 * qualification before the MTP_PREFIX_SNAPSHOT capability is advertised.
 * This reports implementation availability. Artifact qualification is separate.
 * Capture must be armed before prefill at a positive 16384-token boundary.
 * MTP capture requires a nonfinal boundary, after the head has appended known
 * pairs and retains exactly one pending row. MTP mode must match on restore.
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
