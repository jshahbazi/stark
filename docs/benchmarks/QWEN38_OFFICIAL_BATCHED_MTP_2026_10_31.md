# Qwen3.8 mixed-4-8bit exact batched MTP

Date: 2026-10-31. Execution source:
`959ec981354acab313a9a8d709933fb2f2649b17`.
[Machine-readable aggregates](data/qwen38-official-batched-mtp-2026-10-31.json).

This study adds batched exact MTP to the same
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` checkpoint used in the
[matched Stark/MLX-Serve comparison](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md).
The stored quantization and ordinary target distribution are unchanged.

## Final matched results

Across 16 same-loaded pairs, two-draft MTP averaged **65.02 tok/s**, versus
**56.69 off**, a **14.69% gain** with **16/16 pair wins**. Median whole-call
latency was **6.615 versus 7.135 seconds**, a 7.29% reduction. Both routes
retain the MTP head in memory.

| Route | Pooled tok/s | Minimum / median / maximum tok/s | First / last eight pairs, pooled tok/s |
| --- | ---: | --- | --- |
| MTP off | 56.69 | 50.28 / 57.90 / 63.43 | 60.03 / 53.70 |
| MTP on, two drafts | **65.02** | 53.25 / 67.76 / 74.53 | 70.44 / 60.38 |

Every measured request and both warmups matched its strict route reference.
This sustained window shows a consistent relative win and substantial
absolute timing drift. **It does not reach 77–80 tok/s on this checkpoint.**

A separate original/new/new/original load comparison used the frozen
optimized non-MTP-only worker from the earlier MLX-Serve study. Each load
excluded one warmup and measured four requests, giving eight measured
requests per worker. The new worker used two MTP drafts; the old protocol
omitted that unsupported field and loaded no MTP head.

| Worker | Pooled tok/s | Minimum / maximum tok/s | First / second load pooled tok/s | Median whole-call seconds |
| --- | ---: | --- | --- | ---: |
| Original optimized, no MTP head | 52.00 | 48.81 / 54.74 | 51.44 / 52.57 | 8.038 |
| New exact MTP | **57.65** | 56.80 / 58.45 | 57.70 / 57.61 | 7.602 |

The new MTP worker was **10.88% faster** by pooled decode rate in this
separate ABBA window. Both candidate loads exceeded both baseline loads;
all output hashes matched their strict route references. This comparison
includes the extra loaded head and its memory cost. Model loading is
excluded from the timers. It did not rerun MLX-Serve.

## Implementation

The preceding official MTP path executed one ordinary target decode per
proposed token. The new `batched-affine-decode-exact-v1` verifier evaluates
the pending input and up to three proposed successors in a layer-major graph:

- Stock HC normalization/down/up expressions retain per-row reduction and
  BF16 boundaries. ND uses its existing row axis; U2V shares weight loads.
  Per-row partials are bounded to four positions. Finished-gate writes retain
  the separate BF16 product and residual-add rounding.
- GDN convolution preserves the original window and writes its alternate.
  ValueKey recurrence scans at most four positions, retains F32 state and
  captures every rollback prefix in its physical layout. Q8 projections use
  the ordinary stock helper's vector-grid axis.
- Attention retains ordinary Q/K normalization, RoPE, affine-64 K/V writes,
  sparse visibility and split/reduction plans. Dense rows use ordinary decode.
- MoE keeps each row's official router, shared gate, BF16 activation table,
  affine dot expressions and expert-order down accumulation. HC injection
  remains a separate operation.
- The checkpoint's actual vocabulary head is **Q8/group64**, with 675,430,400
  bytes of stored codes and affine planes. Its global Q4 default does not
  describe this storage group. The final verifier shares those Q8 loads
  across input rows and fuses the existing integer BF16-value conversion.
  It retains the generic native Q8 dot expression, SIMD reduction and
  PlatformDefault compiler policy. Ordinary one-row decode is unchanged.

The existing host rejection sampler, F64 probability arithmetic, filtering,
random streams and owned readout lifetime are unchanged. Rejected suffixes
restore recurrent and convolution state; partial delivery, stopping and
cancellation restore only the committed prefix. Same-seed on/off token
sequences may differ because MTP uses its own proposal/acceptance draws.
Exactness refers to preserving the ordinary target distribution.

## Engineering steps

Every row below uses four same-loaded sampled on/off pairs, one excluded
warmup per route, 4K input and 256 sampled outputs. Rates pool 255 useful
post-first-token decode advances per request. All completions and warmups
match their respective strict historical route hashes.

| Worker/change | Drafts | Off tok/s | On tok/s | On/off change | Pair wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved final serial verifier | 2 | 62.61 | 43.48 | -30.55% | 0/4 |
| Initial official batch | 1 | 65.15 | 68.62 | +5.33% | 4/4 |
| Initial official batch | 2 | 57.62 | 63.32 | +9.90% | 4/4 |
| Initial official batch | 3 | 64.18 | 55.89 | -12.91% | 0/4 |
| Expert-union gate/up candidate | 2 | 59.16 | 63.29 | +6.96% | 4/4 |
| Q8 head reuse, original row-wise expert scheduling | 2 | 60.05 | 69.06 | +15.00% | 4/4 |

These are different desktop timing windows. Differences between their
absolute rates or uplift percentages do not isolate a source-level gain.
The expert-union candidate retained exact outputs but did not show a clear
incremental benefit in its short run. Its extra route scans and registers
were omitted from the delivered path. Its frozen worker and exact source
archive remain preserved privately.

The initial batch profile attributed 37.31% of verifier GPU time to MoE,
21.59% to GDN, 16.49% to HC, 14.39% to attention and 9.72% to the vocabulary
head. Timestamp profiling is excluded from throughput runs. It identified
the Q8 head's remaining per-token projections.

At the actual Q8 head geometry, synthetic component ABBA controls measured
three-row projection at **3,411.78 versus 1,170.21 microseconds** for serial
ordinary projection versus fused reuse, a 2.92x component speedup. All
744,960 F32 readout values matched exactly. Four-row values also matched.
These are component timings, separate from full-model throughput.

## Identity and protocol

The immutable checkpoint revision is
`788c07a8725642bccf13a0a3b8a21954ab84b4f2`. Its 113 files total
107,326,639,344 bytes; the pinned manifest SHA-256 is
`11246ebbea8a24a988517ebeb59623b857df89120e5ae3becd48807194542ef6`.
All files retain the size, nanosecond mtime and inode in the prior complete
payload-hash receipt. Configuration and index hashes were rechecked; payloads
were not rehashed or rewritten. Stored Q4/Q8 weights, BF16 activations and
operation boundaries, F32 ValueKey state and affine-64 Q8 K/V are retained.

The host is an Apple M5 Max, 128 GiB, macOS 26.7.1 (25G241), with AC Energy
Mode Automatic. Desktop apps and power settings were left as found. Full
model jobs were serialized, with competing-worker checks around requests.
Other desktop GPU activity and clock drift remain uncontrolled.

Sampling uses temperature 1, top-k 20, top-p 0.95, seed 42, 4,096 input
tokens, 256 sampled outputs and context capacity 4,352. Prefix reuse and
profiling are disabled. The input is the same public Lily source corpus:
`src/qwen4exp/model.rs` at `e22d58fa7cc488493fb1b5d810a198bd29730aeb`,
SHA-256 `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955`.
The 4K input-token hash is
`fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`.

The final worker SHA-256 is
`cd0b3a5a9fbcc61ca4f064b2536cdaf613d3c1348324757eac956d48e383a66d`;
the final native input inventory SHA-256 is
`2402173e67fcbb0b34a642aac1c53a7f97eacd1b3cf0a3f5a9972f07fe31cbbe`.
Builds use Rust 1.98.1 and locked dependencies. Compiler/math policies and
upstream licenses are described in [the provenance record](../../vendor/lily-core/UPSTREAM.md).

## Validation and preservation

The committed source passes **140 complete target readout comparisons**:
34,764,800 F32 values with zero differing bits, zero target error and zero
target-distribution divergence. Widths 1–4, every short accepted prefix,
partial-group drops, dense/sparse attention boundaries, rollback
continuations and failure fencing are covered. The initial stage diagnostic
also matched all 554 captured stages at widths one and four. Draft-head
catch-up readouts are recorded separately; exact correction retains their
actual proposal distributions.

Top-k and nucleus lifecycle controls each pass 23 requests, including
15 bounded output tails, greedy on/off parity, partial-group stopping,
invalid-request recovery and streaming cancellation followed by an identical
fresh request.

Final regression checks pass: 78 generator tests, 41 scorer test executions,
50 model-free benchmark checks, and both worker Clippy checks with warnings
denied. The physical CPU/Metal suite passes 673 tests with 48 ignored, plus
seven CXX and three documentation tests. Vendor defaults pass 75 tests with
51 ignored; 12 additional exact MTP component controls execute separately,
including the retained Lily kernels. Timing experiments are excluded from
these regression totals. Repository policy, formatting and scoped Ruff checks
also pass. Independent source review found no actionable correctness issue.

The final 811 native input files match their receipt and execution-commit
Git blobs byte for byte. The private final source archive passes a full
round-trip hash check, alongside the earlier batch and rejected expert-union
archives. Benchmark reports and executable hashes remain bound separately.

The original `codex/qwen38-exact-mtp` remains at `a060ff31500f2338fdff7c309e3081f833ecdf73`;
its frozen worker retains SHA-256
`81945ba24640c044ca43dd91b76577d361f063c290a7a905181da8183c4b6f06`.
The preceding `codex/qwen38-mlx-mtp` remains at
`4ebdbf15fce9fe6dca584a8bb9df7c183c55aaba`, with its frozen final worker
`7142467b0036a99af262a903cd8c85d3a1768da0a1c67813f5feab5fa857e09a`.
The earlier [Lily result](QWEN38_MTP_COMBINATION_2026_10_31.md) uses a
different weight pack and remains separate evidence.

MTP remains **off by default**. This workload uses `mtp_drafts: 2`.
Prompt, context, sampling and acceptance affect its benefit; these controls
do not establish a throughput floor or an advantage across arbitrary work.
The same-loaded off/on comparison holds the loaded MTP head's memory cost
constant. It does not measure the memory saving of omitting that head.

Numerical readouts and generated tokens are compared in memory. Ignored
private records retain aggregates, hashes, frozen workers and exact source
archives. Checkpoint payloads remain unchanged in their existing directory.
Public results contain aggregate metrics and identities.
