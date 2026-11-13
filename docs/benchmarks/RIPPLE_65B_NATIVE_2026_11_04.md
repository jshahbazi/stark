# Ripple-65B Preview native Mac validation, 2026-11-04

Native Q8_0 text generation completes on the Apple M5 Max with 128 GiB unified
memory. Full-checkpoint readouts pass a bounded comparison with the publisher's
independent F32 reference; elementary replies, request isolation and recovery
pass. Observed longer-run decode is **0.71–1.54 tok/s under the measured desktop
load**, with a two-run warm median of **1.13 tok/s**. Quiet-machine screening
did not admit a timed run, so these are observations rather than a qualified
throughput floor or maximum.

The [machine-readable aggregates](data/ripple-65b-native-2026-11-04.json) contain
errors, counters, timings and hashes. The [validation record](../archive/runs/2026-11-04-ripple-native-support.md)
records the test scopes, source changes and executable identities. Existing
other-model benchmark records remain unchanged.

## Fixed model, build and machine

- Checkpoint: `AtomicChat/Ripple-65B-Preview-GGUF`, Q8_0, revision
  `57066fdb2348a53f2261f096e0a793b9554b4106`.
- Metadata and reference: `ricursiveAI/Ripple-65B-Preview`, revision
  `e27e171792f303d74c048184e8259ba663cfa0d5`.
- Two weight files: 68,840,113,568 bytes, 64.11 GiB. Actual content hashes match
  both pinned Hugging Face LFS hashes in the validation record.
- Architecture: dense 64.78B parameters; 80 layers, 80 query heads, eight KV
  heads and 128-dimensional heads. Sliding layers use a 1,024-token window;
  every fifth layer uses global causal attention without RoPE.
- Native engine source: `4cb209cdb638ee5d8107cd50eb128bacabbef25b`.
- Rust 1.98.1 release builds with locked dependencies; generator built
  separately from the optimized scorer. No speculative decoding or diagnostic
  profiling is enabled in the worker timing run.
- Host: Apple M5 Max, 18 CPU cores, 40 GPU cores, `Mac17,7`, 128 GiB unified
  memory, macOS 26.7.1 (`25G241`). Power settings were not changed.

Weights retain their Q8_0 blocks and F32 vectors. Matrix results, residuals,
normalization, gates, attention and KV state are F32. Attention and elementwise
operations execute in Rust; matrix products use Stark Metal kernels. Prefill
advances one token at a time. This precision/execution policy is separate from
the publisher's BF16 inference and quality scores.

## Correctness and memory

The independent oracle decodes the same quantized weights, streams all 80
publisher decoder layers through CPU PyTorch and compares eight full-vocabulary
readouts with native Metal. Maximum absolute error is **0.0002365112**,
relative L2 error **0.0000030137**, and top-token agreement **8/8**. Reset replay
and recovery after an invalid token are exact. This is bounded numerical
evidence for the same quantized checkpoint; long-context quality and original
BF16 quality equivalence remain unqualified.

The final signed app worker passes two elementary response checks plus exact
replay after another request and after a rejected request. All reply identities
match the initial implementation. These checks do not qualify agent tasks or
structured tool parsing.

Peak reported Metal allocation during those checks was **68,295,262,208 bytes
(63.605 GiB)**. Peak worker RSS was 153,501,696 bytes. File-backed shared-memory
accounting means these should not be added to infer physical memory use.
System-wide swap-out increased by 10,567,680 bytes during the interval; that is
not a worker-specific measurement. The model completed without allocation
failure on the 128 GiB host.

The device reports a recommended working set of 115,448,725,504 bytes
(107.520 GiB). Ripple's complete matrix-view inventory uses 68,294,410,240
page-rounded bytes (63.604 GiB), below the policy cap of 77,996,609,536 bytes
(72.640 GiB). Admission reserves 8 GiB after taking three quarters of the
recommended working set, validates per-buffer limits and excludes CPU-only
embeddings. Larger inventories retain the original streaming policy.

## Weight-view diagnostic

The initial fixed 8 GiB view cache evicted every projection during a scan of
the dense model. The bounded retained-view policy changes buffer lifetime only.
Complete native readouts remain bit-identical across the policy change.

| Last warm diagnostic replay | Initial 8 GiB cache | Retained complete inventory |
| --- | ---: | ---: |
| Input / decode steps | 2 / 2 | 2 / 2 |
| Decode rate | 0.528 tok/s | 2.925 tok/s |
| Matrix-view hits during decode | 0 | 1,282 |
| Matrix-view misses / evictions | 1,282 / 1,282 | 0 / 0 |
| Peak registered view bytes | 8,585,592,832 | 68,294,410,240 |

Both use the same checkpoint, greedy continuation and output hash. The two-step
profile is diagnostic, includes instrumentation and was not host-screened.
Its rate change is not a general performance comparison; the counter change
directly verifies that repeated registrations were removed.

## Longer observed timing

One worker handles three serialized, identical greedy requests. Requested
context is 512 tokens, actual input is ten tokens, and the output limit is 32
sampled tokens. All runs reach that limit. The first request is warmup; the two
remaining requests have identical output hashes. Each 32-token run performs
31 decode steps after its first sample from prefill. Rates below use
`31 / decode_seconds`.

| Replay | Role | Prefill seconds | Decode seconds | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| 0 | Warmup | 27.013 | 42.026 | 0.738 |
| 1 | Measured | 13.741 | 43.556 | 0.712 |
| 2 | Measured | 13.826 | 20.137 | 1.539 |

Warm median: **1.126 tok/s**; pooled warm rate: **0.973 tok/s**. Warm median
prefill rate is 0.725 tok/s. These shared-load measurements varied materially.

The preceding screened attempt completed warmup and then timed out waiting
for ten seconds of quiet conditions. The monitor reported GPU utilization of
57–64% in its last window, above the 5% admission limit; CPU limits were also
exceeded in that window. Other applications and jobs were not stopped. The
subsequent observed run is explicitly unscreened. There is no matched
llama.cpp, MLX or other-engine result in this study.

The first elementary response check also spent 56.3 seconds in prefill,
including initial weight preparation. Short warm elementary replies reached
2.58–2.86 decode tok/s. Those one/two-output-token cases are correctness smoke
checks; they do not replace the longer timing above.
