# 20261109-QMEM-01: F32 Qwen scalar attention workspace

## Identity and state

| Field | Value |
| --- | --- |
| Registered date / owner | 2026-11-09 / Codex; registered after source implementation and before execution |
| Kind / evidence scope | Allocation correctness and memory; structural and synthetic component checks |
| Attempt state / result | Implemented; structural, Metal component and synthetic full-forward checks passed |
| Promotion / disposition | Adopted on branch; production artifact speed/memory qualification pending |
| Baseline / candidate | `1ac5b9dc`; `1a9c36b5` in `qwen.rs`, `qwen_graph.rs` and `qwen_graph_tests.rs` |
| Related source | [Performance triage](../archive/runs/2026-11-09-model-launch-performance-triage.md) |

## Hypothesis and change

The F32 graph allocated two `capacity * heads` F32 arrays for scalar attention
whenever its persistent KV cache grew. Normal decode uses those arrays only
below 256 tokens. Split attention uses separate command scratch, bounded to
8,000,000 F32 elements. The report's 304 MB split-scratch claim does not describe
this baseline.

The candidate gives scalar scores/probabilities their own lazily allocated
workspace and stride. Normal split decode retains at most 256 scalar rows per
head. The explicit scalar override and oversized-split fallback can grow that
workspace to the request's actual count. Allocation admission inventories KV and
scalar workspace growth independently, including predecessor-owner overlap.
The shader arithmetic, selected positions, reduction order and split threshold
are unchanged.

An initial suggestion to put both arrays in the command arena was rejected
before implementation: the long scalar fallback may require buffers exceeding
the arena limit. Separate persistent owners preserve that fallback.

## Artifact, source and environment

- Intended production path: PipeNetwork
  `Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, revision
  `b2c422f3c643e36f04227a64d61796b44a4b1029`, native F32 `qwen4_exp`
  decoder. This change does not switch it to the separate Lily backend.
- Architecture provenance remains PipeNetwork MLX revision
  `0352280b927a731fd35276ac622ef2f28da5c936`; existing Apache-2.0 notices stay
  intact.
- Weight bytes and quantization are unchanged. Activations, accumulation,
  attention caches and scalar workspace remain F32; recurrent precision is
  unchanged.
- Planned full-forward component fixture: private complete synthetic
  `qwen-fixture-f32-v1`, with context increased from 128 to 512 solely for the
  scalar-to-split transition test. This is not a production artifact run.
- Toolchain: `nightly-2026-10-01` engine tests, existing locked dependencies.
  Host: Mac17,7, 128 GiB unified memory, macOS 26.7.2 (25G332). Executable and
  fixture hashes are in the [validation record](../archive/runs/2026-11-09-model-launch-correctness.md).
  No speed or RSS measurements have been made.

## Fixed correctness controls

- Model-free checked inventory: normal long contexts keep a 256-row scalar
  workspace; explicit scalar mode and the 8,000,000-element split ceiling
  require full-count capacity. Checked overflow must reject unsafe shapes.
- Metal allocation component: grow the KV cache to 4,096 rows while scalar
  owners stay at 256; then request scalar capacity 4,096 and check both owners.
- Scalar kernel component: scores/probabilities use a different row stride
  from KV storage while preserving masked, grouped-head output against the
  existing scalar CPU reference at absolute error at most `2e-6`.
- Private synthetic full forward: 257 sequential prefixes cross the
  255/256 route transition. Every logit must match the unchanged staged path
  within absolute error `2e-5`. No output/token values are published.
- Run all GPU checks serially. Stop on allocation rejection or a numerical
  mismatch; do not widen tolerances after execution.

## Lifecycle and limits

Reset still drops the complete graph and both workspace owners. No cache rows
or activations cross requests. F32 Qwen prefix reuse stays disabled: no snapshot
type or dispatch hooks restore its Delta recurrence/convolution, QSA attention
and indexer state, and PLE/ngram history. The separate Lily prefix evidence
does not qualify this artifact.

Images and MTP are unavailable on this existing F32 path; the change neither
adds them nor removes a supported head/tower. End-to-end speed, production
checkpoint exactness and observed peak memory are unmeasured. Static payload
reduction is not a measured throughput claim.

## Execution follow-up

The native hybrid suite passed 240 tests with 60 ignored. This includes scalar
workspace admission, independent KV/workspace growth and scalar stride parity.
The complete synthetic 257-prefix scalar/split transition check passed, then
passed separately with forced scalar attention. The tolerance remained `2e-5`
as registered before execution. All GPU checks ran serially. Full readout
values remain private.

No failed numerical controls or tolerance changes occurred. The production
PipeNetwork artifact was not loaded, and peak RSS/GPU memory and throughput
were not measured. The memory change is supported by allocation/inventory
assertions, not a production memory or speed benchmark. Final-source hashes,
other model checks and the interrupted broad-engine-suite scope are in the
[validation record](../archive/runs/2026-11-09-model-launch-correctness.md).
