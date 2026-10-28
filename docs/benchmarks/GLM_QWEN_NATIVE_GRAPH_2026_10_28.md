# Native GLM and Qwen GPU graph optimization

Date: 2026-10-28. Host: Apple M5 Max, 128 GiB unified memory.
Baseline source: `8056d605d49e4876093206fb2d9fb18c1c799c25`.
Candidate source: the commit introducing this report.

This follow-up covers the exact GLM-5.3-Flash UD-Q2_K_XL and Qwen3.8-Flash-Next
mixed 4/8-bit checkpoints recorded in [Native hybrid models](../NATIVE_HYBRID_MODELS.md).
The [preceding optimization study](GLM_QWEN_NATIVE_OPTIMIZATION_2026_10_28.md)
preserves its own build identities, validation and measurements. This record
measures additional native Stark work. Reference-engine literature informs the
implementation; throughput comparisons here use Stark workers.

## Final benchmark results

| Model | Previous warm decode | Final warm decode | Ratio | Previous warm prefill | Final warm prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash UD-Q2_K_XL | 2.4849 | 7.5790 | 3.05× | 2.4684 | 7.9342 |
| Qwen3.8-Flash-Next mixed 4/8-bit | 4.3180 | 12.5153 | 2.90× | 4.4515 | 12.6666 |

Rates are tokens/second. Final warm decode repeats span 7.4841–7.6739 for GLM
and 12.3752–12.6555 for Qwen. GLM uses 14 input tokens and samples 32 tokens
with a length finish; Qwen uses 54 inputs and samples 31 tokens with an EOS
finish. Every completion hash matches the preceding worker across all three
fresh-state repeats. Input, configuration and source-receipt hashes also match.

The baseline columns reuse the same-host, source-bound reports from the
preceding study; they were not rerun interleaved with this final candidate.
The final requests ran after builds and competing test/model jobs stopped.
These short-workload ratios do not establish sustained or long-context speed,
or equivalence to another engine or activation policy.

First-request wall times were 9.00 seconds for GLM and 36.78 seconds for Qwen,
versus the preceding 27.65 and 38.35 seconds. File caches were not cleared;
these are first-use observations rather than controlled cold-load results.
The accepted default enables concurrent routed FFN encoding. Neither rejected
residency requests nor selected-slice caching is present in production.

## Measurement method

Preserve each release worker before measuring it. Build the generator separately
from the optimized scorer with locked dependencies and Rust 1.98.1. Stop builds,
GPU tests and other model runs before the final benchmark pass. Run one complete
model at a time, keeping the checkpoint, template, context limit, sampling mode
and output limit fixed between workers. No external runtime executes requests.

The existing [worker harness](../../scripts/benchmark_hybrid.py) sends three
fresh greedy requests to one loaded worker, with a 512-token context limit and
at most 32 sampled tokens. Each request resets attention and recurrent state.
It reports the first replay separately and uses the median of the second and
third replays for warm rates. File caches are not forcibly cleared; first-use
figures include lazy shader and resource setup and are not controlled cold-disk
measurements. The harness records aggregate counts, timing and hashes.

Worker throughput uses `first-token-wall-clock-v1`: prefill ends at the first
sampled token, and decode counts subsequent sampled tokens over the decode
interval. A sampled EOS counts toward the sample total. A comparison must retain
matching input hashes, configuration and receipt hashes, sample counts and
finish reasons. Exact completion hashes provide a bounded replay check.

The opt-in [phase profiler](../../crates/stark-engine/examples/hybrid_profile.rs)
uses a separate short synthetic workload. Architecture scopes are inclusive.
GPU duration is contained within command execution and waiting; these values
cannot be summed as independent costs. Its short warming runs guide experiments
and do not establish final worker throughput. Process RSS excludes substantial
GPU and operating-system file-cache memory.

Models, binaries, raw validation artifacts and runtime reports remain ignored.
Inputs, responses, token vectors and probabilities are not published. Weight
files stay unchanged and retain the previously verified source manifests.

## Native execution changes

The accepted GPU stage graphs cover routed feed-forward, GDN, hyperconnection
and KDA operations. The routed feed-forward path keeps selected gate/up
projections, activation, down projections and ordered expert/shared reduction
within one command buffer.
Routing remains on the CPU. Qwen's initial router/shared projection batch and
the combined feed-forward command replace the previous five completion
boundaries per layer. GLM uses the same graph with its separate routed/shared
SwiGLU clamp parameters. A GPU status word preserves rejection of nonfinite
intermediate projections even when subsequent clipping could conceal them.

Qwen's hyperconnection graph includes down projection, SiLU, up/injection
projections and stream mixing in one submission. Grouped normalization and
residual injection retain their scalar paths. Its GDN graph includes the input
projections, depthwise convolution, recurrent update, normalization/output gate
and output projection in one submission. Convolution history and recurrent F32
state persist in owned GPU buffers between steps. The private recurrent layout
is `[head, key, value]` for coalesced access; reset clears the complete state.

GLM's KDA graph includes its input projections, dependent low-rank projections,
convolution, L2 normalization, decay/delta update, output RMS/gate and output
projection. Its convolution history and `[head, value, key]` recurrent F32 state
remain in GPU buffers. Per-row scalar reduction order is retained inside the
recurrent kernel. Persistent state is request execution state and is cleared by
reset and failure recovery.

The vector8 GEMV kernels assign one output row to each 32-lane SIMD group and
eight rows to each threadgroup. Eight adjacent weights share affine scale/bias
loads or IQ grid/sign/scale decoding. Affine 4/8-bit and IQ2_XS/IQ3_XXS use this
schedule. Q5_K/Q6_K use specialized 64-thread, two-SIMD-group block/lane
kernels, described below. Other formats retain scalar scheduling.
A shared host geometry helper keeps pipeline selection and dispatch shape
consistent across standalone projections and graph intermediates. Byte-safe
loads support unaligned tensor spans. Partial row tiles and batch crossings are
covered by the numerical tests.

Storage remains quantized and all activations, recurrent state and accumulation
remain F32. Metal uses safe math mode. SIMD-row GEMV changes summation order;
the source-reference error criteria remain unchanged. This work introduces no
BF16 activation narrowing. CPU paths still own token/history control, exact
ngram/PLE hashing and sparse table gathers, routing, scalar normalization and
remaining attention/indexer operations. Prefill remains sequential and only its
final input token computes vocabulary readout. Whole-model graph capture,
chunked matrix prefill and MTP are outside this candidate.

## Weight views and experimental controls

The accepted policy keeps a static 8 GiB allowance and a whole-bank registry
only when the entire rank-three inventory fits within 75% of the device's
recommended working set minus that 8 GiB. Every bank must also fit the device
buffer limit and the kernels' u32 view offsets. Larger inventories use
transient checked selected-range views. There is no production residency flag,
residency-set module or selected-slice cache. The rejected residency and cache
experiments below remain separate candidates.

Views retain source mappings through Metal resource ownership. Repeated expert
selections can reuse registered whole-bank wrappers and return checked
subviews; retaining a wrapper does not promise physical page residency. These
registrations retain immutable weights, not inputs, decisions or completions.
Their byte accounting measures retained registrations; live operations and
other process memory have separate lifetimes. This estimate does not reserve
memory against other apps.

`STARK_HYBRID_CONCURRENT_FFN` defaults on for the accepted concurrent routed
gate/up and down projections. Buffer barriers order gate/up → activation → down
→ final reduction. Set it to `0` for a diagnostic serial override. Other graph
passes stay serial.

| Environment variable | Enabled by `1` |
| --- | --- |
| `STARK_HYBRID_CONCURRENT_FFN` | Defaults on. Set to `0` for a diagnostic serial override of routed FFN projections. |

Apple describes residency preparation as dependent on current system
conditions. It is not a guaranteed page pin. See
[requestResidency](https://developer.apple.com/documentation/metal/mtlresidencyset/requestresidency%28%29)
and [recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).
Final A/B results must record the concurrent FFN setting.

## Synthetic GEMV experiments

The [component runner](../../crates/stark-engine/examples/hybrid_gemv_experiment.rs)
compares the original scalar reduction with independent SIMD-row kernels.
The initial expert-shape experiment rotates 64 distinct synthetic compressed
weight banks. It drops one setup replay and records eight warm GPU-duration
samples per kernel, divided by the number of banks. A dependent readout follows
an explicit buffer barrier. The table below is synthetic component timing in
microseconds per matrix, not model tokens/second or hardware peak bandwidth.

| Matrix, output × input | Scalar serial | Vector8 serial | Vector8 concurrent |
| --- | ---: | ---: | ---: |
| Qwen affine 4-bit, 640 × 2560 | 25.26 | 8.48 | 1.85 |
| Qwen affine 4-bit, 2560 × 640 | 15.36 | 4.52 | 2.80 |
| Qwen affine 8-bit, 640 × 2560 | 12.30 | 5.43 | 3.46 |
| Qwen affine 8-bit, 2560 × 640 | 13.03 | 4.97 | 5.94 |
| GLM IQ2_XS, 2048 × 4096 | 56.81 | 15.27 | 10.95 |
| GLM IQ3_XXS, 4096 × 2048 | 55.62 | 12.72 | 9.59 |

Affine cases use BF16 scale/bias planes and groups of 64. Error comparisons
cover every bank: vector8 relative RMS error is approximately `1.35e-7` to
`2.39e-7` against the scalar kernel. Maximum absolute errors range from
`1.43e-6` to `4.58e-5`, depending on format and scale. These bounds describe
the synthetic data. The 8-bit down case regresses under concurrency, so this
experiment does not justify enabling concurrent encoding for every operation.
Final selection depends on full-checkpoint measurements.

## Dense packed formats

A bounded production-header inventory explains why expert-only optimization
leaves substantial GLM work. It includes trunk layers 0–44 and vocabulary
readout, excludes CPU hyperconnection weights, compressor positional controls
and embedding row gathers, and counts complete compressed MLA head banks.
These are metadata-derived unique matrix sizes, not measured memory traffic.

| Static matrix storage | Encoded bytes |
| --- | ---: |
| Q5_K | 2,753,822,720 |
| Q6_K | 2,351,677,440 |
| Rank-two Q8_0 | 536,444,928 |
| F32 | 203,948,032 |
| Q4_K vocabulary readout | 356,843,520 |
| Q8_0 MLA head banks | 196,083,712 |
| Total | 6,398,820,352 |

The same inventory estimates 2,750,939,136 encoded bytes for eight selected
experts per routed layer, including mixed IQ2_XS, IQ3_XXS and IQ4_XS storage.
The actual registered spans and GPU traffic include alignment and execution
details beyond these metadata sums.

A generic eight-adjacent-value schedule calls the existing scalar decoder
eight times. It regresses the tested Q5_K SSM/shared shapes by 9–16%, Q4_K
readout by about 10%, and small wide-input Q8_0 and F32 controls more severely.
It remains an experiment. Q5_K/Q6_K instead need decoding aligned with their
block layout: each SIMD lane owns a column within a 32-column section, reuses
the block's high-bit bytes and scale/minimum fields, and evaluates its eight
values with coalesced input access. The per-value dequantization arithmetic is
unchanged. Each threadgroup owns two rows and submits 64 threads.

The second component experiment uses 1–64 distinct banks, choosing enough
banks to approach 128 MiB of encoded weight storage. It drops one replay and
records eight warm GPU-duration samples. These synthetic serial timings are
separate from the 64-bank experiment above:

| Matrix, output × input | Scalar, microseconds | Block/lane, microseconds | Speedup |
| --- | ---: | ---: | ---: |
| Q5_K SSM input, 8192 × 4096 | 382.71 | 112.41 | 3.40× |
| Q5_K SSM output, 4096 × 8192 | 489.38 | 135.87 | 3.60× |
| Q5_K shared gate, 2048 × 4096 | 62.62 | 20.87 | 3.00× |
| Q6_K SSM input, 8192 × 4096 | 298.34 | 112.53 | 2.65× |
| Q6_K shared down, 4096 × 2048 | 50.01 | 22.58 | 2.21× |

Two, four and eight SIMD groups per threadgroup were compared. Two groups win
or closely match the larger groups at these Q5_K/Q6_K shapes and are selected.
Relative RMS differences from the scalar reduction are approximately
`1.02e-7` to `2.37e-7` on this synthetic data. Q8_0 gains depend strongly on
shape; its 128 × 4096 control matrix regresses more than four times under the
block/lane schedule. Production therefore preserves its scalar path. The
original CPU/Metal source-error gates also cover the accepted Q5_K/Q6_K path.

## Rejected memory-policy experiments

The accepted implementation includes none of these candidate patches. Each
was applied and checked independently; their measurements are candidate A/B
evidence, not final benchmark results.

- `complete-bank-admission.patch` relaxed the GLM complete-bank budget. No
  first request completed after 310.75 seconds, so the trial was terminated.
  It produced no tokens/second result. The rejected near-limit trial used an
  8 GiB static allowance and 4 GiB additional space from the device
  recommendation while admitting 101,884,624,896 bytes of GLM rank-three
  banks.
- `sharded-residency.patch` prototyped command-attached residency sets sized
  at 5% of the recommended working set per set, capped at 32 sets. Qwen's full
  worker measured 12.3024 tokens/second without the hint and 7.7890 with it,
  about 37% slower, with matching output hashes. This result does not rule out
  an MLX-style queue-attached whole-graph residency policy.
- `selected-slice-cache.patch` prototyped cached selected views. On GLM using
  default transient views, throughput was 7.78835 tokens/second versus
  7.02956 with the FIFO cache, about 10% slower with the cache. Prefill was
  7.74873 versus 5.52945 tokens/second, respectively; output hashes matched.

These outcomes reject the tested variants; they do not change the accepted
whole-bank inventory admission rule. The MLX residency reference is pinned to
[`resident.h`](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/resident.h)
and [`resident.cpp`](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/resident.cpp),
revision `7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a`. The completed [Qwen
3.8 literature review](../archive/runs/2026-10-28-qwen38-performance-literature-review.md)
and [GLM 5.3 literature review](../archive/runs/2026-10-28-glm53-performance-literature-review.md)
preserve related external-policy context.

## Source-reference validation

| Fixture or invariant | Observed candidate result |
| --- | --- |
| GLM corrected quantized reference, pool 2 | 94 checked positions; max absolute error `2.8014183044433594e-5`; relative RMS `2.2505577930217008e-6` |
| GLM corrected quantized reference, pool 4 | 94 checked positions; max absolute error `3.826618194580078e-5`; relative RMS `2.8248789638360066e-6` |
| GLM acceptance criterion | Existing absolute and relative-RMS gates `1e-4`; serial and concurrent FFN errors match |
| Qwen mixed/BF16 groups | 13/13 greedy agreement; max absolute error `2.384185791015625e-7`; existing `2e-5` gate |
| Qwen mixed/BF16 four streams | 13/13 greedy agreement; max absolute error `2.384185791015625e-7`; existing `2e-5` gate |
| Qwen concurrent FFN | Both 13-position source fixtures retain the same max error and greedy agreement; reset replay and failure clearing pass |
| Packed/affine kernels | All 13 checkpoint formats and affine 4/8-bit groups 32/64/128 with F32/F16/BF16 planes pass existing F64-oracle gates, including tails, batch crossings and independent expert planes |
| Graph state | Multistep KDA comparison, GDN history/state reset, prefix ingestion, context rejection and failure recovery pass focused checks |

These fixtures and exact full-model hash replays bound regression evidence.
They do not establish general answer quality or equivalence to another
runtime's BF16 execution policy. Complete engine, worker and production-smoke
results belong with the final worker identities below.

## Reference engines and provenance

The new graph orchestration and SIMD-row kernels are original Stark code.
The external sources below informed execution patterns; no external serving
implementation or kernel body was copied or launched. Existing GGML format
decoders and lookup tables retain revision
`8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` and
[LICENSE.ggml-hybrid](../../LICENSE.ggml-hybrid). GLM and Qwen architecture
contracts retain the source revisions in [Native hybrid models](../NATIVE_HYBRID_MODELS.md).

- [Lily Qwen architecture](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/docs/architecture.md),
  revision `e22d58fa7cc488493fb1b5d810a198bd29730aeb`: GPU-resident recurrent
  state, longer encoded operation chains, resident expert banks and sparse
  host access to the ngram table informed the graph and memory experiments.
- [MLX quantized kernels](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/kernels/quantized.h),
  revision `7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a`: packed-word decoding,
  activation reuse and SIMD-level reduction supplied useful scheduling
  references. Stark's vector8 implementation keeps its own F32 arithmetic
  and source-bound GGML decoder conventions.
- [MLX-Serve Qwen expert path](https://github.com/ddalcu/mlx-serve/blob/e1d2013c1a321ba9e04324d790254e3f5428265c/src/plugins/qwen4_exp/trunk/moe.zig),
  revision `e1d2013c1a321ba9e04324d790254e3f5428265c`: resident expert indexing
  and dependent feed-forward stages informed the combined native submission.
  Its rounding policy remains separate from Stark's F32 source gates.

The earlier Uzu-derived engine provenance and rejected four-row/polling
experiments remain in the preceding study. Persistent prompt/session caching
from other engines is not adopted; Stark's immediately-consecutive identical
scoring-cache policy remains unchanged.

## Final artifact identities and validation

| Final check | Result |
| --- | --- |
| Physical CPU/Metal engine suite | 616 passed, 36 ignored; 7 CXX tests and 3 doctests passed |
| GLM comprehensive source references | 94 positions each for pools 2 and 4; errors in the table above; unchanged `1e-4` gates |
| Qwen source references | Both 13-position fixtures pass the unchanged `2e-5` gate; exact reset, prefix ingestion and failure recovery |
| Release generator / optimized scorer tests | 43 generator tests; scorer experiment plus 19 scorer and 19 digest tests passed |
| Worker release Clippy, warnings denied | Generator, default scorer and optimized scorer passed |
| CPU/grammar-only engine build | Passed with Rust 1.98.1 |
| Final worker fixture integration | Both families passed four greedy reference checks, EOS union, request isolation, consecutive score-cache policy, failure recovery and clean shutdown |
| Full production checkpoints | Generation and exact replay passed; invalid requests recovered; raw scoring was finite and candidate probabilities normalized |

Final short phase captures measure 180 submissions/token for GLM, down from
464, and 255 for Qwen, down from 532. These captures use two input tokens and
four decode steps, with separate warmup and clock behavior from the worker
benchmark. Their elapsed GPU/wait times must not be extrapolated into worker
throughput. They show the remaining CPU-visible completion boundaries; a
complete device decode graph and fused expert gathers remain further work.

The independent aggregate audit matched all old/new input, configuration,
receipt, sample-count, finish-reason and completion hashes. Full-model output
hashes are `5377eeb8513c16603090026f7a432327b456842a96d54565044e24bdb5986e1c`
for GLM and `aca3d65e215fc3c055839f338dd8c040531a677970415c0d410fe5c7391f6c7b`
for Qwen. Small fixtures and hash replays do not establish general answer
quality or installed-app qualification. No installed app was rebuilt.

All raw artifacts remain private and ignored. SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Previous generation worker | `eb1640f078f4b8871a94ffbc8e7a95224b83da2f9c1711ead105c108fcee386d` |
| Final generation worker | `c157f22eb82104ca22e324ab1a8f345a1db2349ee1a88888917cdad20f34a7b3` |
| Final optimized scoring worker | `61fa1d94b416fdba68236e2b68b83e5eea7c0731da5a5203c2d089cd48a16661` |
| Final phase profiler | `d037f6c10569af6f48a2c7abac62311cbc0da49b1c1fd6828bfef307d6024002` |
| Unchanged worker harness | `a55be8c1dbb896ee53c4ec6408a653394cd564536c55847e07f0ffa713825c56` |
| GLM final worker report | `a05c0cf8117527d76dd50b8ba11ef8e7bc24eaf9a7be75c077c7ab85c051b3e6` |
| Qwen final worker report | `b358062626100bfea44d902d3018a1bab78d906001cd5bcee28eb6f3ad9f05b5` |
| Physical engine test log | `4267f35d208d32e975500fee549e35d312fac526b97ac3d5a8f958a6b5dd47c8` |
| GLM final production smoke | `617e4dd087cc2b1c652cce148cb89c4dcaa0381b25c9a702b981428640315c0b` |
| Qwen final production smoke | `273e5ead5d37dc06109013eed5ad3e12983e744fce4c751de93114db10844d25` |
| GLM final worker fixture integration | `0e8844ac4128fd6454fdd3b1b16ece458ca817706750293ee3e173d78f8f215b` |
| Qwen final worker fixture integration | `dad5353e25421953e0ff2cf132cfd8777c00f8a26a414aa221ee5a4c95143d85` |
| GLM final phase capture | `6838878eb144a6ca8149a70102f774fd6aaa36414334620284708234c071a40c` |
| Qwen final phase capture | `e910738bc43a2ffbfc6a5960c99083dc49f8aa62effd9cf93a4bdc04c9bc9fd7` |

Reproduce worker measurements after stopping competing builds and GPU jobs:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
env -u STARK_HYBRID_CONCURRENT_FFN uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/glm --worker target/release/stark-generate --output runtime/native-hybrid/glm-graph-replay.json --tokens 32 --replays 3
env -u STARK_HYBRID_CONCURRENT_FFN uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/qwen38 --worker target/release/stark-generate --output runtime/native-hybrid/qwen-graph-replay.json --tokens 32 --replays 3
cargo +1.98.1 build --release --locked -p stark-score --features optimized
```

The default concurrent FFN policy is used here. Set
`STARK_HYBRID_CONCURRENT_FFN=0` to repeat the diagnostic serial variant. Rejected
cache and residency variants require their independent archived source patches.
