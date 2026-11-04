# Experimental GLM wide prefill

The native GLM chunk-prefill path has independent experimental selectors for
wider token scheduling, batched mHC controls and register KDA. All are opt-in;
engine defaults and the existing optimized wrapper's T8 selection remain
unchanged. Artifact-bound numerical, lifecycle, prefix-reuse and timing
qualification is required for each selected combination. This reference
describes implementation and acceptance requirements; dated run records carry
results.

## Selectors

Start with [`with_glm53_optimized.sh`](../../scripts/with_glm53_optimized.sh),
whose existing recipe enables `STARK_GLM53_PREFILL_MMA=1` and
`STARK_GLM53_PREFILL_CHUNK=8`. Retain that recipe's context limit of 4096.

| Variable | Values | Effect |
| --- | --- | --- |
| `STARK_GLM53_PREFILL_WIDE` | Unset, `16`, `32`, `64`, `128` | Unset retains the existing chunk route. A value selects the maximum wider token tile; admission and tails can select smaller tiles. Other values fail validation. |
| `STARK_GLM53_CONTROL_PREFILL_BATCH` | `1` enables | Encodes independent mHC rows together, retaining each row's mix until its matching residual update. Available with existing T4/T8 chunks and wider tiles. |
| `STARK_GLM53_KDA_REGISTER_PREFILL` | `1` enables | Selects the separately admitted [register KDA operator](GLM_KDA_PREFILL.md); unsupported geometry or pipeline capacity retains chronological per-token KDA. |
| `STARK_GLM53_PREFILL_DENSE_WIDE` | `1` enables | Reuses decoded Q5/Q6 weights across a true T32/T64/T128 input tile for the admitted M5 Max matrix shapes. Other shapes retain T8/T4 projections. |
| `STARK_GLM53_EXPERT_PREFILL_BATCH` | `1` enables | Groups routes across tokens and shares IQ2_XS/IQ3_XXS weight decoding across four input rows. Gate, up and down must all be admitted; other layers retain the per-token route. |
| `STARK_GLM53_MLA_PREFILL_BATCH` | `1` enables | Batches independent MLA projections and compatible Q8/float matrix operations, while cache append and attention remain chronological. Selected Q5/Q6 MLA projections also share decoded weights. |
| `STARK_GLM53_PREFILL_CHUNK` | `4`, `8` | Existing selector, still validated when wide prefill is selected. It defaults to four without the wrapper. |

For example, run a worker with T64 scheduling and batched controls:

```sh
env STARK_GLM53_PREFILL_WIDE=64 STARK_GLM53_CONTROL_PREFILL_BATCH=1 \
  scripts/with_glm53_optimized.sh /absolute/path/to/stark-generate \
  --model-path /absolute/path/to/model
```

For attribution, compare the same source and artifact with all experimental
selectors unset, then enable one selector at a time before measuring intended
combinations. Restart between selector changes. The existing wrapper leaves
these experimental selectors to the caller.

## Execution and admission

Wide prefill requires the admitted native GLM graph, resident encoder banks,
greedy overlap, a serial wide encoder and one complete admitted bank group.
Concurrent encoders, parallel projections and diagnostic phase/encoder timing
are incompatible with this route. The existing bank, device, tensor-shape and
context checks remain authoritative.

T16 through T128 describe layer-major token scheduling. Without the additional
matrix selectors, dense projections write disjoint output slices using the
existing T8/T4 packed kernels and scalar tails. The dense-wide selector instead
shares a decoded weight fragment across T32, T64 or T128 input rows inside one
K loop. Each row retains the existing K partition, partial-sum order and T8
extreme-input fallback boundary.

Grouped experts pack raw integer route IDs on the GPU, build four-row jobs and
use bounded indirect dispatch. Projection results return to the original
token/top-k slots before the existing weighted reduction. There is no CPU
route readback. Unsupported bank formats decline before grouping starts.

MLA batches independent input/query/head projections, normalizations, value
projections and the final output projection. Appending each token's KV/index
state, selecting sparse rows and attending still happens in token order. No
query sees a future cache row. Its compatible matrix kernels retain the
selected F32 reduction and extreme-input fallback policies. Register KDA has
its own [precision policy and gates](GLM_KDA_PREFILL.md).

Before state mutation, conservative layer workspace estimates choose bounded
layer groups and reduce oversized tiles. Each command owner admits at most
448 MiB total for graph scratch and accounted external MLA allocations, and retains
at most 32 MiB of incoming stream backing. Scratch requests retain the existing
8-million-F32-element bound. The two live owners fit the existing two-512-MiB
overlap allowance; the global graph cap remains 512 MiB. Attention growth stays
within the existing context-derived cache inventory.

One group can execute while its successor is encoded. The predecessor drains
before the successor submits; cancellation is checked again after that wait.
On an encoding failure, the unsubmitted successor drops before pending writers
drain and model reset runs. A completion failure fences the model. Every writer
finishes before a tile returns or a prefix snapshot is captured.

## Qualification

Tiles stop at the absolute 1024-token prefix grid, and all supported widths
divide that grid. Existing snapshots retain the same state representation.
Qualify fresh and restored execution with the same selectors, artifact,
context, precision and model instance; a compatible layout alone does not
establish prefix exactness.

Follow [model onboarding](../MODEL_ONBOARDING.md). Cover incoming recurrence,
tails, cache growth, nonzero starting positions, cancellation after waits,
failure with a predecessor pending, reset/replay, and capture/restore across
1024 tokens. Batched controls require comparison with the selected row control's
F32 reduction tree, including nonfinite checks. Register KDA additionally needs
the operator and full-model numerical gates described in its reference.

Bind records to source and executable hashes, exact GGUF/config/tokenizer
identity, flags, toolchain, device and OS. Verify actual route admission with
aggregate counters, compare full-model readouts and generation, then measure
whole-request prefill separately on a quiet host. Keep private model data out
of records. Publish aggregate results and hashes only; component correctness
and fewer dispatches alone do not establish an integrated throughput gain.

## GPU stage profiling

`STARK_GLM53_PREFILL_PROFILE=1` uses retained graphs with GPU timestamp
boundaries. Set `STARK_GLM53_UNRETAINED_GRAPH=0` and
`STARK_GLM53_SCRATCH_HEAP=0` after the optimized wrapper, and provide the
existing private `STARK_NATIVE_PROFILE_OUTPUT` destination. Only fixed-name
aggregate durations are recorded. This diagnostic transport adds encoder
boundaries and is separate from throughput qualification.
