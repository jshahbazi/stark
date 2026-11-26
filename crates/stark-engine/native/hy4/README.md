# Hy4 Flash native Metal sources

The direct GGUF graph under `vendor/strata` comes from [Strata at `20610a5442f713be26625b0f587bd52252225e4c`](https://github.com/Niko1221/Strata/tree/20610a5442f713be26625b0f587bd52252225e4c). Strata’s [MIT notice](vendor/strata/LICENSE) and the [GGML MIT notice](vendor/strata/engine/third_party/ggml/LICENSE) travel with the imported code. The import follows Strata’s `strata_hy4_flash_metal_host` source list and its `mtl_rt` dependency, with paths preserved for source comparison. Stark supplies request handling and image preprocessing.

The build embeds the quant definitions and Metal shader sources in the library. The Metal compiler assembles them in the upstream order and uses safe math. The embedded graph takes its settings from Stark and uses the upstream defaults for the remaining tuning. It returns Metal failures to Stark and releases Metal resources and anonymous expert mappings with the loaded graph. MTP’s derived Q8_0 expert file is stored under the caller’s model artifact directory and is written through a temporary file.

The original expert stream reads packed banks into its Metal arena. Stark also wraps read-only GGUF expert ranges directly when the mapped banks fit the live host and Metal budgets. The layer records retain the original gate, up and down offsets and strides. The vision integration reuses Strata’s GGUF projector, vision tower and M-RoPE encoder. Stark provides the resized RGB input and token spans through the native request interface.

The generation worker offers `STARK_HY4_LOAD_PROFILE=streamed-prefill` and
`STARK_HY4_LOAD_PROFILE=resident-decode` for the measured configurations.
Both use the existing context fitter before allocating the graph and accept
different request windows within the selected capacity. See [tested load
profiles](../../../../docs/GENERATION.md#tested-hy4-load-profiles).

The pinned GGUF graph stores latent MLA rows by absolute position, including the sliding-attention blocks. Its IndexPool cache writes a pooled key into the first row of a complete group. At a complete group boundary Stark can rewind the text sequence by clamping the pooled-key counts and replacing the suffix. The next prefill supplies a fresh head output. MTP truncates its row count and retains the verified trunk hidden rows before the same boundary. Request identity and invalidation are tracked by the Stark caller.

The grouped prefill path extends Strata’s Metal 4 matrix multiply to the Q3_K and Q4_K gate/up tensors in this artifact. It uses the upstream Q4_K down projection and top-k combine. When every trunk expert block is covered, the planner groups the whole chunk into 64-entry by 64-output tiles. It stages dequantized weights and activations in F32 and accumulates in F32. The reduction order differs from GEMV. The worker uses this path by default on supported devices.

With mapped expert banks, `Prefill::mapped_gpu_plan` lets the GPU build the expert plan for decode and verification windows of up to eight rows. The GPU sorts routes by expert and selection slot, then writes the same group and entry layout and indirect grid sizes as the host. Routed GEMV arithmetic and slot-order reduction are shared with the host-planned path. Prefill and the streaming expert arena use the host planner.

Set `STARK_HY4_EXPERT_STORAGE=mapped-resident` to prepare the mapped GGUF expert banks before the worker reports readiness. The worker faults in clean pages in bounded intervals and requests Metal residency for the expert buffers. The mode requires live host and Metal headroom and reports an admission failure when preparation cannot complete. Startup with the original projector took 61–70 seconds on the measured M5 Max. [Mapped residency experiment](../../../../docs/model-experiments/2026-11-25-hy4-grouped-fragments.md#20261125-hy4frag-20--prepare-the-mapped-expert-banks-before-generation)

With GPU planning enabled, ordinary mapped-resident verification uses separate Engram inputs per table so all target blocks can share one command buffer. The verifier pairs eligible Q8 GEMV rows and uses the contiguous causal range when the IndexPool selector would return every cell in that range. [Verifier qualification](../../../../docs/model-experiments/2026-11-25-hy4-grouped-fragments.md#qualify-the-prepared-worker)

Set `STARK_HY4_MTP_RESIDENT_EXPERTS=1` to load the original NextN block's derived Q8_0 expert bank ahead of generation. The GPU builds the draft expert plan and preserves top-k slot order. Context admission reserves about 1.85 GB more than for the lazy draft pool. [NextN residency experiment](../../../../docs/model-experiments/2026-11-25-hy4-grouped-fragments.md#20261125-hy4frag-21--keep-nextn-experts-resident)

Set `STARK_HY4_MTP_HEAD_Q5=1` to build a Q5_0 vocabulary projection for NextN. It uses the existing F32 GEMV activations and accumulation. The target verifier uses the original BF16 vocabulary weights. Leaving the switch unset or setting it to `0` selects Q8_0. The worker reports the loaded copy as `draft_head_format`. [Draft vocabulary qualification](../../../../docs/model-experiments/2026-11-25-hy4-decode-cache-construction.md#qualify-the-typed-q5-option)

Stark adds opt-in Metal stage-boundary samples to the prefill graph. They retain the command buffer and host events while splitting compute encoders. The worker exports aggregate stage durations and tile occupancy. Target decode timing excludes draft-graph commands, and invalid command-buffer timestamp intervals are counted and excluded from the GPU span totals. The original streamed arena also accepts a bounded 36 GB request through the worker. These local changes and their measurements are recorded in the [4K experiment](../../../../docs/model-experiments/2026-11-25-hy4-prefill-decode-profile.md).

An optional prefill policy selects the existing Strata 64 × 32 F16 operand, F32 accumulation kernels for grouped routed experts. The packed GGUF banks stay in their original format. The optional streamed decode policy ranks repeatedly demanded resident experts ahead of speculative loads while preserving the existing fence and in-flight read checks. [Precision and retention validation](../../../../docs/model-experiments/2026-11-25-hy4-dense-prefill-expert-retention.md)

With resident NextN experts and packed dense prefill enabled together, bulk NextN cache windows use the same packed
Q8_0 matrix multiply for eligible cache projections. The kernel stages weights and activations in F16 and
accumulates in F32. Cache windows of up to eight rows use GEMV. [Packed cache qualification](../../../../docs/model-experiments/2026-11-25-hy4-decode-cache-construction.md#qualify-the-packed-cache-in-the-rebuilt-worker)

The optional dense device views pack the attention gate and output projection operands into contiguous 64-element K slabs. The Metal tensor products keep their F16 inputs, F32 accumulation order and scratch allocation. [Dense layout qualification](../../../../docs/model-experiments/2026-11-25-hy4-grouped-fragments.md#qualify-the-rebuilt-dense-layout)

Set `STARK_HY4_PREFILL_DENSE_FFN_DEVICE_VIEWS=1` together with `STARK_HY4_PREFILL_DENSE_DEVICE_VIEWS=1` to use the same
workspace and kernels for aligned trunk Q8_0 FFN projections on text prefill chunks of at least 512 rows. This option uses
F16 operands with F32 accumulation and applies only when the loaded context capacity is at most 8,192 tokens. Larger
loaded contexts use the original FFN projection for every chunk, including short requests and prefixes. Shapes that
exceed the existing workspace also use the original projection. The worker reports the requested setting as
`prefill_dense_ffn_device_views_requested`.
