# Native GLM and Qwen GPU decode graphs

Date: 2026-10-28. Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.1.
Branch: `codex/native-glm53-qwen38`, following the Flint merge `10ce86c1`.

The native graph improves Qwen warm decode by **2.05x** and GLM by **11.1%**
on this short workload. Qwen completes one GPU command per token; GLM completes
43, retaining CPU routing to respect its existing expert-memory admission
budget. Every production benchmark completion matches the preceding
implementation's token hash.

This follows the [graph and kernel study](GLM_QWEN_NATIVE_GRAPH_2026_10_28.md).
Its measurements remain historical evidence for that earlier build. Checkpoint
revisions and preparation are listed in [native hybrid models](../NATIVE_HYBRID_MODELS.md).

## Matched worker results

Both modes use the same release generator, SHA-256 `e3e4023c…`. Each mode
sends five fresh greedy requests to one loaded model, context 512, maximum
32 output tokens. Warm rates are medians of replays 1 through 4.

| Model / mode | Input / sampled output | Finish | Prefill tok/s | Decode tok/s |
| --- | ---: | --- | ---: | ---: |
| GLM-5.3-Flash UD-Q2_K_XL, staged | 14 / 32 | length | 8.695 | 8.134 |
| GLM-5.3-Flash UD-Q2_K_XL, graph | 14 / 32 | length | 9.674 | 9.041 |
| Qwen3.8-Flash-Next mixed 4/8-bit, staged | 54 / 31 | EOS | 11.603 | 11.214 |
| Qwen3.8-Flash-Next mixed 4/8-bit, graph | 54 / 31 | EOS | 23.981 | 22.936 |

GLM warm decode ranged from 8.059–8.245 staged and 8.728–9.234 graph.
Qwen graph warm decode ranged from 22.668–23.069. All five outputs in each
mode are identical, and each model's two modes have matching input,
configuration and source-receipt hashes. Completion hashes are:

- GLM: `5377eeb8513c16603090026f7a432327b456842a96d54565044e24bdb5986e1c`.
- Qwen: `aca3d65e215fc3c055839f338dd8c040531a677970415c0d410fe5c7391f6c7b`.

The [unchanged harness](../../scripts/benchmark_hybrid.py) uses the checkpoint's
chat template with its fixed synthetic request and thinking enabled.
Generation resets state before and after each request. Prefill ends with the
first sampled token; subsequent sampled tokens determine decode throughput.
A sampled EOS contributes to the sample count.

The staged runs set both `STARK_QWEN38_DECODE_GRAPH=0` and
`STARK_GLM53_DECODE_GRAPH=0`; graph runs unset them. Both modes unset
`STARK_HYBRID_ORDERED_RMS` and retain the default concurrent-FFN setting.
The task's builds, tests and other model runs were stopped for timing. The
workers were serialized. Ordinary desktop activity was uncontrolled, and the
OS file cache was not cleared. First-request timings are not cold-disk
measurements. No generation output or prompt cache is used.

## What changed

A deferred encoder retains checked device-vector views until completion.
Dependent operations share command-local scratch in chunks of at least 1 MiB,
capped at 512 MiB per command. Persistent recurrent and attention buffers have
separate ownership. Sticky GPU failure status catches invalid intermediates,
including values later concealed by activation clipping. Failure resets the
complete sequence. Per-token autorelease pools bound temporary Metal object
lifetimes, including encoder transitions and error cleanup.

Qwen runs normalization, hyperconnections, GDN, QSA attention and caches, PLE
control/convolution, routing, indexed expert products, residuals and readout
on the GPU. Rust prepares token embeddings and exact I64 n-gram lookups before
submission. Vocabulary logits return to the existing CPU sampler. Attention
storage grows with the visible prefix, copying only valid cache rows.

GLM runs mHC/Sinkhorn, normalization, KDA, pooled sparse selection, absorbed MLA,
expert products, residuals and readout on the GPU. Its router logits cross a
completion boundary for CPU correction-biased selection and combination
weights. Rust registers the selected immutable expert views before continuing.
The full 101.9 GB rank-three inventory still exceeds the conservative admission
budget; this change does not force whole-bank registration. Three dense layers
and 42 routed layers produce 42 routing completions plus one final completion.

Selected GLM expert groups execute concurrently inside the same command, with
explicit buffer barriers before consumers. `STARK_HYBRID_CONCURRENT_FFN=0`
selects serial expert batches. Qwen's admitted banks instead use device-indexed
expert GEMV with checked indices and matching compressed-code/scale/bias planes.

RMS uses an F32 SIMD reduction. Its sum tree differs from the staged scalar
implementation and is separately qualified against wide F64 oracles and the
unchanged source-model limits. `STARK_HYBRID_ORDERED_RMS=1` retains the slower,
source-ordered GPU diagnostic; this choice is captured once at native Metal
executor initialization. Control projection prefetch and parallel Sinkhorn
retain bitwise agreement with their scalar GPU references.

New shaders are original Stark implementations. Existing GGML decoding
attribution and licenses remain intact. The earlier
[Qwen review](../archive/runs/2026-10-28-qwen38-performance-literature-review.md)
and [GLM review](../archive/runs/2026-10-28-glm53-performance-literature-review.md)
retain external source revisions and license boundaries.

## Validation

| Check on the final source | Result |
| --- | --- |
| Full physical CPU/Metal engine suite | 640 passed, 38 ignored; 7 CXX and 3 documentation tests passed; engine suite 213.97 s |
| Generation release tests | 43 passed |
| Default and optimized scoring release tests | Each: 1 experiment, 19 unit and 19 digest tests passed |
| Configured release Clippy | Generator, scorer default and scorer optimized passed with warnings denied |
| Pure CPU/grammar build | Passed |
| Source fixtures, GLM pool 2 and pool 4 | 94 positions each; unchanged absolute and relative-RMS `1e-4` limits |
| Source fixtures, Qwen grouped and four-stream | 13 positions each; unchanged `2e-5` comparison; 13/13 greedy agreement |
| Qwen graph/staged growth and reset | 33 positions cross 16/32-token capacity boundaries; exact reset and failure recovery |
| Worker fixture integration | GLM/Qwen: 4/4 greedy checks each, EOS union, isolation, consecutive score-cache policy, recovery and clean exit |
| Flint integration after merge | Four 13-position source fixtures and F32/Q4-BF16 worker fixtures passed |
| RMS and projection kernels | F64 wide/tail RMS oracles; ordered RMS, prefetch and Sinkhorn bitwise controls passed |

GLM pool-2 maximum absolute error is `3.3378601074440795e-5`, relative RMS
`2.1092784081480096e-6`; pool-4 values are `3.910064697265625e-5` and
`2.977030357176713e-6`. Qwen maximum errors are `2.980232238769531e-7`
and `2.2351741790771484e-7`. Both families retain exact reset/failure replay;
the Qwen source fixtures also check greedy agreement at all 13 positions.

The independent review caught a GLM error-path bounds defect: nonfinite index
scores could let a padding index reach the visibility mask. Scores are now
sanitized after flagging failure, and the mask writer separately checks bounds.
Regression tests cover overflow, padding canaries and recovery.

Final full-checkpoint smokes passed for both families using the generation and
optimized scoring workers listed below. Each generated eight tokens, repeated
exactly, recovered from an invalid request and exited cleanly. Scoring produced
finite vocabulary readouts and valid normalized candidate probabilities. These
checks reuse the previously verified weight manifests, guarded by unchanged
file sizes and modification times; they do not rehash the complete checkpoints.

Three four-step release phase profiles per model confirm **43 completions per
GLM token** (172 submissions and waits per replay) and **one per Qwen token**
(four per replay). The profile workload uses a different prefix from the worker
benchmark. CPU release-test builds briefly overlapped these diagnostic profiles;
their timings are excluded from the throughput comparison. Models, binaries,
source-fixture data and raw runtime captures remain ignored; only aggregates
and digests are published.

## Rejected candidate and numerical limits

An earlier graph candidate, generator `9ccc2cf2…`, regressed GLM warm decode
from 7.656 to 4.261 tok/s despite matching completion hashes. Source-ordered
memory loads and lost expert concurrency prompted the subsequent kernel and
scheduling changes. Those measurements are not used as the final baseline.

A periodic synthetic MLA test exposed a different limitation: CPU/GPU
exponential rounding changed selection at nearly tied pool scores, producing
a 0.2186 output difference at position 67. Growth/sort tests now isolate exact
dyadic pooling, nonuniform pooling, signed-zero ties and adjacent F32 scores.
No source tolerance was widened. This does not establish identical selection
at every possible near tie.

Prefill remains sequential. Qwen's over-budget sparse-index selection remains
a scalar GPU kernel, so these short requests do not qualify its long-context
cost. GLM's transient expert registration and CPU routing remain performance
constraints. Further work includes bounded bank/gather execution, fused expert
kernels and chunked prefill. This run does not establish sustained thermals,
answer quality, another engine's precision policy, MTP, vision or installed-app
qualification.

## Artifact identities and reproduction

The final source follows `10ce86c1` and includes the Flint integration.
The source-bundle digest covers all 22 `.rs` and `.metal` files immediately
under `crates/stark-engine/src/engine/hybrid`, sorted by repository-relative
path. Hash input is UTF-8 path, NUL, file bytes, NUL for each file. The separate
RMS test shader is listed independently.

| Artifact | SHA-256 |
| --- | --- |
| Generation worker | `e3e4023c625785059e903d3ca6b3c83633653499e64d2b1cc2859fb9dcc83893` |
| Optimized scoring worker | `d9e0cf8774afbe3a86666e8fdf469b9f56b677f825450c9dc78e7f91ce5c57ff` |
| Phase profiler | `6f7e26d8b6fb3e0165a377eb805d27439f6fee414ec3ed3b4ab14e7ecda34218` |
| Reference comparison example | `444c9f3aea4c3f3483d82425207da8dc01a0328097372d9ee0ac7bec6ee3003f` |
| Hybrid source bundle | `dee5d6f5e5207f0ec74797a1522c901a1fac895447a562facb1172c91dbdad37` |
| RMS reference test shader | `d7ca8796ae4989b56920adf83635cb995919b47663648a0d2cc2e2b35d13f21b` |
| Benchmark harness | `a55be8c1dbb896ee53c4ec6408a653394cd564536c55847e07f0ffa713825c56` |

Private result receipts:

| Receipt | SHA-256 |
| --- | --- |
| GLM staged benchmark | `2afbdbb395abdbd22625c14913dbfea51997e071a10e57883614dad1c63e2bec` |
| GLM graph benchmark | `645b1d18037e9e39a933fbe414153a39c460761bdd3c3b57c621af1a381d3137` |
| Qwen staged benchmark | `49fa7cf0030b7021f5628f8f1901dab4116407c9cc21a7053d61865c6c8e060e` |
| Qwen graph benchmark | `6c26232325658a8f930ad44ea90d4a948ef0db5b292b822c16be5e12cf00754b` |
| GLM release phase profile | `78558506eb8f1014ca5d2e43b25538a7ac713d4a8c8acbe21376fe06cf33e810` |
| Qwen release phase profile | `b4cb8b74093ef0c2b58de4b71f9d969a759a53c5dcf1d3e472361a68275b4f5b` |
| GLM full-checkpoint smoke | `63ed59ef5decb617f3685093d3f54838bdfcfc51961fbde11d36a3139a276642` |
| Qwen full-checkpoint smoke | `e4265bd8fe0b84705618d1bc4359778ad3062cbe75f87f29f86e9ef35c2c5457` |

Build generator and scorer separately. With the pinned checkpoints already
prepared, repeat each model's staged and graph commands:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
env -u STARK_HYBRID_ORDERED_RMS -u STARK_HYBRID_CONCURRENT_FFN STARK_QWEN38_DECODE_GRAPH=0 STARK_GLM53_DECODE_GRAPH=0 uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/glm --worker target/release/stark-generate --output runtime/native-hybrid/glm-staged-replay.json --tokens 32 --replays 5
env -u STARK_HYBRID_ORDERED_RMS -u STARK_HYBRID_CONCURRENT_FFN -u STARK_QWEN38_DECODE_GRAPH -u STARK_GLM53_DECODE_GRAPH uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/glm --worker target/release/stark-generate --output runtime/native-hybrid/glm-graph-replay.json --tokens 32 --replays 5
env -u STARK_HYBRID_ORDERED_RMS -u STARK_HYBRID_CONCURRENT_FFN STARK_QWEN38_DECODE_GRAPH=0 STARK_GLM53_DECODE_GRAPH=0 uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/qwen38 --worker target/release/stark-generate --output runtime/native-hybrid/qwen-staged-replay.json --tokens 32 --replays 5
env -u STARK_HYBRID_ORDERED_RMS -u STARK_HYBRID_CONCURRENT_FFN -u STARK_QWEN38_DECODE_GRAPH -u STARK_GLM53_DECODE_GRAPH uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/qwen38 --worker target/release/stark-generate --output runtime/native-hybrid/qwen-graph-replay.json --tokens 32 --replays 5
cargo +1.98.1 build --release --locked -p stark-score --features optimized
```
