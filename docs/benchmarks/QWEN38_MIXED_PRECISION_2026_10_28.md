# Qwen3.8-Flash-Next native inference optimization

This study follows the [initial decode-graph result](GLM_QWEN_DECODE_GRAPH_2026_10_28.md).
It preserves that build's evidence and records the subsequent F32 optimizations
and the separately qualified mixed-precision integration.

## Host and comparison boundary

Apple M5 Max, 40 GPU cores, 128 GB unified memory, macOS 26.7.1, connected to
power. Rust builds use 1.98.1 with locked dependencies. Engine tests and
formatting use `nightly-2026-10-01`; Python uses the locked converter environment
through uv 0.12.24 and Python 3.13.13. Native checkpoints and raw validation
records remain private under ignored `runtime/`. Published evidence contains
aggregate metrics and hashes only.

[Lily's pinned performance report](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/README.md#performance)
is the strongest explicit non-speculative Apple silicon result found in this
investigation: the author reports 77–80 tok/s for its q4-xl conversion at
4K–64K context on an M5 Max with 40 GPU cores and 128 GB memory. Its smaller q4
conversion reports 83–89 tok/s. These are upstream measurements, not Stark
measurements. The table describes the author's October 12 HTTP series at
`f8ff23e52dd72aa762ab5871f542ec8f870ed16f`, retained in the later pinned README.
That loop overlaps ordinary single-token encoding and submission; it does not
require draft tokens. Higher MTP/speculative numbers are outside this comparison.

The original Pipe mixed checkpoint and Lily q4-xl are distinct artifacts.
Pipe uses 8-bit GDN projections; Lily q4-xl uses 4-bit GDN projections. Both
use mixed quantization elsewhere. Similar total stored size does not imply
equal bytes read by active projections, equal arithmetic, or equal accuracy.
Switching to Lily's checkpoint is an explicit preparation choice.

## Original checkpoint, F32 graph

The original checkpoint is
`pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit` at
`b2c422f3c643e36f04227a64d61796b44a4b1029`.
The short worker workload uses 54 input tokens and requests 32 output tokens;
all five requests stop at EOS after 31 sampled tokens. The reported warm median
excludes the first request. Decode counts the 30 forward steps after the
prefill's first sampled token. No speculation is used.

| Build/artifact | Warm decode tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Preserved baseline binary, original artifact | 23.1563 | 24.2699 |
| Optimized F32 binary, original artifact | 43.7012 | 48.9193 |
| Optimized F32 binary, losslessly aligned artifact | 45.6490 | 51.3529 |

The original-artifact comparison improves warm decode by 88.72%. All requests
have the same completion hash. This short workload does not establish a 4K
context rate. The aligned artifact changes only Safetensors JSON-header padding;
all 11 tensor payloads were copied and independently hash-verified. Its temporary
duplicate was subsequently removed to make room for the Lily checkpoint.

| Identity | SHA-256 |
| --- | --- |
| Baseline generator | `e3e4023c625785059e903d3ca6b3c83633653499e64d2b1cc2859fb9dcc83893` |
| F32 comparison generator | `67df6a8de0bd2c658890716b9032232eea06860ec7b6087d654bae766d8452ab` |
| Original configuration | `e94bbec7dad4e51de2866a6fc5a91ac24f566e07d7c3c4b1cc5646b760e53466` |
| Original preparation receipt | `271ff6df40dd7b95b1c445e0f6bc7021a2cdad1fcf45d2c4f51ace4f82241a98` |
| Input IDs | `f8d93cdb09e0b761f2547a818ff0b8b55cfe2b20018fe225203457211c6c1eac` |
| Completion IDs | `aca3d65e215fc3c055839f338dd8c040531a677970415c0d410fe5c7391f6c7b` |
| Aligned preparation receipt | `0536ac0ba5e78862404a8b2766a9cc43cf26fc5eb00cd249ffd4027303d0bb89` |

The accepted F32 changes parallelize independent projections, normalization and
sparse selection, cache completed pooled index keys, split long QSA attention,
stream the GDN state update in blocks, fuse affine finite checks into the
reduction, use wider loads when every packed plane is aligned, and avoid
repeating immutable weight-geometry admission. They retain source numerical
gates and F32 activations. An F32 hyperconnection fusion regressed its component
benchmark and was removed; its [patch and evidence](../archive/experiments/qwen38-f32-hc-2026-10-28/README.md)
remain archived.

Stage-boundary GPU counters preserve one command completion per token, but add
instrumentation overhead. In the diagnostic run, GDN accounted for 6.942 ms,
MoE 5.188 ms, two hyperconnection stages together 4.898 ms, QSA 2.122 ms, and
readout/remainder 1.272 ms per warm step. The sum of measured encoder intervals
was 20.514 ms and completed-command time 23.081 ms. Those numbers identify
remaining work; the instrumented 37.10 tok/s is not a production benchmark.

## Mixed-precision backend and reference contract

Stark now admits the explicit Lily checkpoint format through an in-process
Rust/Metal 4 core extracted from Lily revision
`e22d58fa7cc488493fb1b5d810a198bd29730aeb`. The source has no server, session
persistence, diagnostic captures or expert-cache service. All 14 upstream Metal
math files are unchanged. Stark adds request adaptation, execution-fault draining
and an integer-bit finite-logit guard for greedy readout. See
[vendored provenance](../../vendor/lily-core/UPSTREAM.md) and its license notices.

The selected policy is BF16 activations, F32 reductions/logits/GDN recurrent
state, and explicitly selected Q8 attention K/V. MTP and vision are disabled.
The n-gram table is mapped read-only; experts remain resident. Each request has
fresh logical state and a maximum of 16,384 tokens. Greedy decoding reads one
token from the GPU; scoring and stochastic generation read full F32 logits.
Existing F32 validation remains reference evidence under the revised
[mixed-precision policy](../../AGENTS.md).

The final greedy worker overlaps CPU command encoding with GPU execution.
Future encoding remains unsubmitted until its input ID has been validated.
This prevents an invalid readout sentinel from reaching the embedding gather.
The iterator is lazy, uses the collector's exact remaining budget, and drains
submitted work on an early stop. Its cleanup failures remain visible to the
worker. Completion feedback is checked before outputs are accepted or command
memory is released. No draft model, speculative candidate, or multirow batching
is involved. Scoring and stochastic generation retain synchronous full readout.

The matching artifact is
`fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`: 55 weight shards totaling
106,905,155,192 bytes. The preparation recipe retains quantized tensor bytes and
adds Stark's tokenizer/EOS metadata and a separate preparation receipt.

An independent harness links the untouched pinned upstream library. Both it
and Stark tokenize the same public fixed input without special tokens, explicitly
select Q8 K/V and capacity 16,384, and perform ordinary synchronous decode.
Teacher forcing compares all F32 logit bytes and lowest-ID argmax hashes over
32 readouts, including inputs that differ from the previous greedy choice.
A separate greedy run emits 256 IDs without EOS stopping, with no full-logit
readback. These diagnostic sequences and the 4K worker throughput workload have
different input construction and must be reported separately.

All 55 downloaded weight shards match their published LFS SHA-256 values.
The complete verification receipt has SHA-256
`13e5743960944d0ee491b3fed3981405202f67c9ce09d8e264bcbc3f6ab182c3`.
The prepared configuration is
`2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899`;
its preparation receipt is
`7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30`.

The independent upstream and Stark teacher runs match every F32 logit byte in
all 64 readouts, over vocabulary size 248,320: 32 at a two-token input and
32 after a 4,096-token input. Forced-input and argmax hashes also match.
Stark's repeat/reset, invalid-ingest recovery and invalid-step recovery checks
all pass exactly at both lengths. This establishes equivalence on those
sequences under the selected precision and checkpoint, not equivalence to the
original Pipe/F32 model or a broad answer-quality evaluation.

| Teacher case | Full-logit SHA-256 | Greedy-ID SHA-256 |
| --- | --- | --- |
| Two-token input, 32 readouts | `01e0d05f93b6f986cd35a2974908a52e84ac76e89eec20aa9dfdea0b095da40d` | `fdd2446110c657a2f68b58d215c33b71a71a0ae178fa64e5a0bb60789a5f8dd8` |
| 4K input, 32 readouts | `04e8a4ddd7a69f156ade5d021bf474930c01ff737545ea2fdadd23c70561d7ef` | `012ccc45ff17ca33a8487b13fa9ee036b78bdb0ecf6d630e6a364685bf6bddc7` |

The final binary repeats the same exact teacher comparison. Its aggregate
receipt is kept privately as `lily-final-parity-summary.json` alongside the
source and binary receipts; the earlier synchronous receipt is retained.
Throughput is measured separately from these readback/hash/recovery controls.

## Final worker throughput

The final worker measures **77.1836 tok/s warm median** at a 4,096-token input,
within the cited 77–80 tok/s range. Warm samples span 76.8097–79.2509 tok/s.
The checkpoint, precision profile and 16K Stark context limit are explicit;
this run does not establish performance at 64K. No speculation is enabled.

The workload tokenizes the pinned upstream `src/qwen4exp/model.rs` as public
corpus text without special tokens and takes its first 4,096 IDs. It requests
256 outputs, with a total context budget of 4,352. Five fresh requests run
serially against one loaded worker. All produce 256 tokens, finish at length,
and have identical output hashes. The first request is warmup; the four remaining
requests supply the warm median. Decode counts the 255 ordinary steps following
the prefill's first draw. Loading, prefill and reset are excluded from decode
rate. No builds, tests, model downloads or other inference runs overlap timing.

| Same Lily q4-xl checkpoint and public-corpus workload | Warm decode tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Initial synchronous adapter | 74.0307 | 2,012.1084 |
| Final safe encoding overlap and verified completion | 77.1836 | 2,050.7693 |

The final warm decode improvement over the synchronous adapter is 4.26%.
Its median including warmup is 77.4950 tok/s. The two worker builds produce the
same 256-token output hash on this workload. A separate cyclic-input diagnostic
also matches all 256 greedy IDs between untouched upstream, synchronous Stark,
and the final iterator. That diagnostic's single-run decode rates are 76.19,
75.86 and 79.38 tok/s respectively; they are not repeated HTTP/server benchmarks.

| Identity | SHA-256 |
| --- | --- |
| Final generator | `21a8fdd1983b2c9579c6af08b6ae172f2d2240dd77ba6e47fa320e0d17911cbb` |
| Final optimized scorer | `6deaa23bf7607f3b40e63d6d310645fa5d996c57a7a15fbe763c8e89dbb6f245` |
| Final native validation executable | `cab0773360f6f6027c9fa148122ef7467961c28838f26ba9b1792e3b25190d29` |
| Untouched-library reference executable | `60880083b6fadadf5cd742447d59a1b27a45ac202641fcab11257fec217cc0bc` |
| Measured source inventory | `2646b88ecd3de1e06c303e11b3fd9761380c973926d771df4be586938f626f14` |
| Final source inventory, including lock/scan metadata | `e230b274c5b7b5f4810f41df58788de1046822d21903a0e45f32c36428313c6b` |
| Public corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| Worker input IDs | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |
| Worker output IDs, both builds | `3a94579eb57b2d943bfb8d6e1b9dc65bbaa4588b98fef7f96895b28bb2073e5a` |
| Cyclic-input greedy IDs, all three paths | `bea6c3782d5fb1a6c8f257c69d9934d97d4d9a034ea27c203237416bdd51fdf4` |
| Final 4K worker benchmark receipt | `e0361ca1c0b756d3834bf158ac1d2fbb8400fa4c1955712fa8131814aa9e5a50` |
| Final mixed-precision worker validation receipt | `8264881a15902726a6cd25a0b733e0b53df08ffbb40432fec5521b9d1f9f445a` |

The source inventory binds 965 source, manifest, policy and test files over base
commit `9e786991885d69506f5d4740c8af11535844c7de`. Its private manifest sorts
paths and hashes `UTF8(path) + NUL + raw_SHA256(file)` for each entry. The
benchmark report itself is excluded to avoid a circular hash. Detailed runtime
receipts remain ignored. These results supersede the earlier Qwen decode-rate
row; the GLM measurements and all dated source validation remain unchanged.
After measurement, the standalone validation lock was aligned to the product
lock and registered with the existing dependency scans. All 31 standalone
registry dependencies now match the product lock, including checksums. The
completion and GPU finite-guard controls pass again under that lock. Runtime
Rust sources and the product lock are unchanged by this metadata update.

## Build and model-free validation

- Locked release generator and optimized scorer builds pass. The workers were
  built separately to avoid unifying the scorer's optimized features into the
  generator. Release Clippy passes for both with warnings denied.
- Full physical CPU/Metal engine suite before adding the request iterator:
  663 passed, 43 ignored; seven CXX tests and three documentation tests passed.
  The opt-in factored-affine controls pass their F64, cancellation, layout and
  nonfinite checks. The final iterator then passed all four targeted Lily tests,
  including the two real-checkpoint tests: lazy execution, exact budgets,
  synchronous equivalence, early-drop draining, reset and caller-selected input.
- Generator unit suite: 43 passed. Optimized scorer: one experiment test,
  19 unit tests and 19 digest tests passed.
- Converter: 5,001 passed, 88 skipped. Repository policy: 41 passed.
- Distribution packaging: 19 passed. Lossless alignment utility: eight passed.
  Context benchmark report test: one passed. Configured Ruff and pinned Rust
  formatting pass; the converter lock check passes.
- The vendored core builds warning-free with Rust 1.98.1. Its GPU finite guard
  passed 126 cases covering NaN/Inf encodings, finite extremes, tails, view
  offsets and output canaries. Three adapter-harness hash/argmax tests pass.
  Two model-free completion tests pass, including a delayed error callback
  arriving after the GPU fence.
- CPU-only engine admission compiles. Dependency advisories, bans, licenses and
  sources pass the configured cargo-deny policy, including its existing
  unmaintained-dependency warnings.

The final workers also pass both original F32 source fixtures: four reference
greedy decisions per fixture, EOS union, tokenization, request isolation,
invalid-input recovery, exact consecutive score-cache behavior and clean exits.
Maximum absolute logit errors are `1.78814e-7` and `1.63913e-7`, within their
unchanged gates. Earlier full source-graph controls covered 13 readout positions
per fixture at `2e-5`, with 13/13 greedy agreement, exact reset replay and failure
state clearing. Their maximum absolute errors were `3.72529e-7` and `2.38419e-7`.
These F32 fixture checks do not qualify the new BF16 path; its matching upstream
comparison is separate.

The final mixed-precision workers pass 13 generation requests and 12 scoring
requests on the complete Lily checkpoint. Checks cover greedy and fixed-seed
stochastic replay and isolation, invalid-ID recovery, streaming equality,
acknowledged cancellation and subsequent recovery, two natural EOS terminations,
finite scoring readouts, independently recomputed candidate softmax, consistent
full-vocabulary log partitions, and exact consecutive-cache hit/miss/reset/error
behavior. Both workers exit cleanly and are loaded serially. The candidate
protocol exposes at most 16 entries, so this worker check does not independently
sum the entire vocabulary's probabilities; full logits are covered by the
separate upstream comparison.
