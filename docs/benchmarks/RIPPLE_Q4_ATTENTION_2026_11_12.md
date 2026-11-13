# Ripple attention decode tuning, 2026-11-12

The selected native path parallelizes attention preparation and softmax,
coalesces short score dots, and partitions longer weighted outputs. It keeps
the original command transport. Weights and their packed codes are unchanged.
All activations, KV state and accumulation remain F32 with Safe Metal math.

## Long-prefix decode result

A single loaded Q4_K_M model compared the previous scalar attention path with
the new controls. It computed a synthetic 1,024-token prefix once using the
baseline and retained an in-memory diagnostic snapshot of its KV cache.
Each trial restored that exact prefix and replayed the same 57 continuation
inputs. The continuation token vector and trial readouts stayed in memory,
never logged or persisted.
This isolates native decoding; it is not worker end-to-end throughput, a
prefill benchmark or production prefix reuse.

After one warmup per variant, four balanced ABBA/BAAB rounds produced eight
measured trials per variant. Validation occurred after each decode timer.
Both variants retained readouts in memory identically during measurement.

| Measurement | Baseline | New attention |
| --- | ---: | ---: |
| Median native decode, tokens/s | 6.639 | 7.796 |
| Greedy agreement with baseline continuation | 57/57 | 57/57 |

The median **paired round improvement was 16.3%**. Every round favored the
candidate, ranging from 7.8% to 19.7%. Each round's ratio divides the sum of its
two candidate rates by the sum of its two baseline rates; the primary statistic
is the median of those four ratios, rather than the ratio of pooled medians.

Baseline restoration was bit-exact. Candidate readouts had maximum absolute
error `0.000579834`, mean absolute error `0.0000120248`, and relative L2 error
`0.0000008357` against the native baseline. All passed the fixed per-element
tolerance `0.001 + 0.001 * abs(reference)`. The 640 MiB diagnostic snapshot stayed
under a separate 1 GiB test-only payload cap. It validates State identity,
geometry, capacity and finiteness before transactional restoration. The
production cache admission/accounting and request lifecycle integration needed
for prefix reuse remain unimplemented; the diagnostic interface is test-only.

The host was the 128 GiB M5 Max (`Mac17,7`), macOS 26.7.2, on AC power mode 2.
No power settings or other applications changed. Unrelated GPU clients were
observed during the session. Longer process-based trials showed substantial
drift, so they were not used to select the new transport or claim a stable chat
rate. These are local, paired diagnostic results on a working Mac, with no
quiet-host or sustained production throughput claim.

## Selection and precision

- Head preparation uses one 32-lane SIMDgroup per head at dimension 128. Lane
  zero retains the original ordered RMS sum; lanes normalize and rotate pairs
  independently, reusing each pair's trigonometry. Other dimensions retain the
  scalar kernel.
- Softmax uses SIMDgroups for counts at least eight, retaining the scalar
  maximum and denominator accumulation order. Shorter counts retain the
  original kernel.
- Score dots use coalesced SIMD32 only at dimension 128 and counts 1–64.
  Larger counts retain the original score kernel; the SIMD variant lost at
  count 128 and above.
- Weighted output uses partitions at counts 256–4096 and dimension 128.
  Tile size is 64 through count 1024, and 128 above it. Head/KV/hidden/FFN/layer
  bounds limit extra scratch; the existing 512 MiB graph workspace ceiling is
  still enforced. Unsupported shapes/counts retain the original output kernel.

The last two paths reassociate F32 sums. Their envelope guards select the
original ordered computation for extreme operands, preserving overflow
behavior. The output's NaN sentinel is control metadata: neither partial nor
reduction arithmetic consumes it as a numeric model value. Nonfinite model
operands retain sticky failure detection. Cache indices and the existing
32 GiB cache-payload limit are unchanged.

Representative component medians, measured by sealed test executables with
alternating ABBA schedules and GPU timestamps:

| Component | Original, µs | Selected, µs |
| --- | ---: | ---: |
| Head preparation, sliding RoPE | 34.82 | 7.37 |
| Softmax, count 1024 | 245.37 | 81.53 |
| Scores, count 8 | 14.36 | 3.76 |
| Scores, count 64 | 8.17 | 5.02 |
| Weighted output, count 256 | 32.54 | 13.53 |
| Weighted output, count 1024 | 125.38 | 39.59 |
| Weighted output, count 4096 | 784.45 | 133.83 |

Component gains are not whole-model speedups. Scalar-bit comparisons cover the
ordered preparation/softmax paths; F64 oracles cover reassociated score/output
paths, including ring wrap, non-power-of-two capacities, count-257 tails,
optional gates, extreme serial fallback and finite-failure behavior.

## Artifact qualification

Separate independent publisher-decoder references cover 257 positions across
all 80 layers, using the same quantized weights with F32 CPU execution.
The helper now permits at most 257 positions and independently caps readout
allocation at 256 MiB; the previous default remains eight positions. New
source-bound receipts preserve their updated helper hash. Earlier validation
records remain unchanged.

| Artifact | Maximum absolute error | Relative L2 error | Top-token agreement |
| --- | ---: | ---: | ---: |
| Q4_K_M | 0.0016689301 | 0.0000021211 | 257/257 |
| Q8_0 | 0.0011444092 | 0.0000014517 | 257/257 |

Both pass `0.001 + 0.001 * abs(reference)` per element, reset replay,
invalid-token rejection and failure recovery. The absolute maximum alone is
not the acceptance threshold: the relative term is evaluated at each readout.
Q8's historical fixture was missing its weight links; a fresh replay directory
was created after independently verifying both cached shard hashes against the
original receipt. No historical fixture or result was rewritten.

The standard Q4 file remains 38,798,820,576 bytes, SHA-256
`51de49f0833201d82e624c1273a0316dca6c281616371770213f0c6d277b8652`,
from AtomicChat revision `df550a7178a87bbfb6ee2bdbd37603c553d536ee`.
Official metadata and unchanged publisher code remain pinned at
`e27e171792f303d74c048184e8259ba663cfa0d5`.

The release generator SHA-256 is
`3932888cfce850a2ea41f1e642c5baf55bc20f3c7bc22ad913f029a0d408ff46`;
the parity executable is
`f8188921a7c720e905c161a6fe755e55200584e4ef32bf6bf0c5fb6c14cee527`.
Both used locked Rust 1.98.1. The native generator passed bounded arithmetic,
literal-instruction, isolation, invalid-request and recovery checks.

## Rejected experiments and regression checks

Serial projection scheduling, a single concurrent encoder, explicit dense
weight residency and unchecked bounded-input kernels were evaluated separately.
None showed a clear repeatable whole-model advantage. They remain test-only or
unselected. Residency safety checks cover complete admission, budget failures,
mapping lifetime and teardown; it is not enabled in production.

The full CPU/Metal suite passed **784 tests**, with 89 ignored, plus seven
integration tests and three doctests. CPU/grammar compilation, 30 focused
reference/preparation tests, formatting and repository policy passed.
The full-suite log SHA-256 is
`f9d8a69dad6a4951073a1d3660530f762611893eb418f0d67c15968e1f7ba334`.
Independent source review found no pending production correctness or resource
bound findings.

Aggregate evidence and executable/source seals are recorded in
[the data receipt](data/ripple-q4-attention-2026-11-12.json). Weights, private
fixtures, snapshots and runtime artifacts stay ignored. The installed app was
not rebuilt. This supplements the earlier
[packed-projection report](RIPPLE_Q4_KERNELS_2026_11_12.md); the distinct
workloads and statistics should not be multiplied into a combined speedup.

To reproduce the offline long-prefix diagnostic after preparing a private
source-bound fixture with the full-reference helper:

```sh
STARK_RIPPLE_CONTROL_FIXTURE="$PWD/runtime/ripple-control-reference" cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal full_model_long_context_control_snapshot_comparison -- --ignored --nocapture --test-threads=1
```

The sealed diagnostic executable SHA-256 is
`534c3e0fd8e5abaf437719ab9a7aea6dff21c78683a3944fa78bd67bdbf8e40a`.
