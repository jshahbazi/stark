# Retained GLM BF16 native profile

Date: 2026-11-05. Branch: `codex/glm-prefill-throughput`.
Implementation: `00b13d30`, merged with main at `5daf939f`, plus the reviewed
receipt check requiring explicit `private_native_caps: false` for the named
experimental profile. This is a new artifact validation record; earlier
measurements retain their original source and tooling bindings.

## Selected configuration

`prepare_stage.py --profile bf16-native-experimental` selects the BF16 draft
head, depth three, gate 0.5, original `native-target-draw-match-v1` acceptance,
transition-weighted expert retention, direct expert reads and eager final-head
append. It includes 16,384-token prefix support. The worker defaults omitted
MTP settings to three drafts for this profile; explicit `mtp: false` and
`mtp_drafts: 0` remain serial. Sampling and seed remain request choices.

The stage uses normal strict admission: 112 GiB ceiling, 2 GiB margin and
source-bound state/head reserves. Expert counts are automatic. The earlier
129/156 counts are an observation, not an operator pin. Default headless stages,
private validation stages and other model families keep their existing defaults.
The public profile does not require validation-only environment flags.

This preserves the strongest completed long sampled diagnostic: **922.6615
prefill / 20.3003 decode tok/s**, raw 16,384-token input, 1,024 outputs,
T1/top-p 0.95/seed 1, context 32,768 and no warmup. That observation failed the
external CPU screen and has unverified clocks. It is not a qualified throughput
claim for this new build. The earlier quality smoke passed 22/24 across both
modes (11/12 each); the all-pass quality gate remains unmet.

The [experiment index](../../benchmarks/2026-11-04-glm-mlx-stream-comparison.md#experiment-index-and-retained-conclusion)
records the serial baseline, native/MTP diagnostics, read and residency controls,
cache frontier, High Power comparisons, rejected Q8 and frequency-first paths,
uncompleted transition-only scorer, guarded 217-entry attempts, private P/Q
sampler and unrun profiler proposal. None of those alternatives is promoted by
this profile. The approximately 1,000/33 target remains unmet.

## Preparation and source bindings

The owned bridge source bundle is
`85bc8d8986897597d193968083f86652ad5cd7305287a736068f363cea217413`.
The pinned model remains the unchanged EXL3/FP8 pack at
`OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw`, revision
`50237d5c26ceb98bfd85f779990b0f5e6d7af144`. Upstream revisions, imported file
hashes and licenses remain recorded in the bridge lock and provenance files.

The first preparation attempt had a Python variable-shadowing bug: a header
path loop overwrote the requested prefix boolean with a string. Its v1 receipts
are retained privately and were not used for inference. Preparation now uses a
separate boolean with explicit assertions. Both corrected v2 stages passed C ABI
smoke checks and five pure native protocol tests. Independent review rehashed
all 26 owned sources and all five artifacts in each stage, verified the receipt
flags, and reviewed default/override dispatch and strict admission.

| Corrected v2 artifact | SHA-256 |
| --- | --- |
| BF16 experimental stage receipt | `411aeecdf3c179eaf840df7f8173ad26020a54c5afb4e5ac632450e8dea5c915` |
| BF16 experimental bridge archive | `bdfe288e268600ec7458611b4da124cc3dd0651f01b454beb1a79a9c9ebf0040` |
| Headless reference stage receipt | `f32462ef8361774bc123ed7ac446eb796145b97f83291b678c6e48dda20b190c` |
| Headless reference bridge archive | `f4511759f8724839c43b2dd7a3785414fb7dd8fbf01bbd0d6975019a66b3898b` |
| Model-bound worker executable | `eb0368157549d0c1fb14b52183986bc713622784ada60fb3d363791254b581df` |

The experimental stage has native profile/prefix enabled, private capabilities
disabled, API version one, and explicit false quality/throughput qualification.
The reference stage has both profile and prefix disabled. Missing or malformed
experimental disposition metadata is rejected; legacy reference receipt
compatibility is retained.

## Actual-model smoke

The release worker was built with Rust 1.98.1, locked dependencies, `glm-exl3`
and the corrected explicit experimental stage, without native validation flags.
The worker ran with `--prefix-cache`. The frozen public-source 16,384-token
fixture was reconstructed in RAM, extended to 16,896 tokens, and given a changed
tail for the reuse case. Requests used context 32,768, eight output tokens,
T1/top-p 0.95 and seed one. Prompts, replies and token vectors stayed in RAM;
only aggregate counts, booleans and hashes were written.

| Check | Result |
| --- | --- |
| Readiness | MTP supported, default three drafts, prefix enabled, chunk 16,384 |
| Omitted settings | Three drafts, zero reused input tokens |
| Changed tail with matching prefix | Three drafts, 16,384 reused input tokens |
| Reset and fresh replay | Zero reused tokens; generated IDs, text and finish reason equal cached execution |
| Explicit `mtp: false` | Serial; native snapshot invalidated, zero reused tokens |
| Explicit `mtp_drafts: 0` | Serial; 16,384 reused tokens, exact output equality with the preceding serial request |
| Subsequent omitted settings | Three drafts restored, zero reused tokens for the shorter input |

All six generation checks passed. The global GPU lock and live memory/paging
guard remained active. The observed maximum non-file memory floor was 93.967
GiB across 219 samples, below 112 GiB; pageout and swapout counters did not
increase. The worker was stopped and reaped, and model identity plus worker,
stage and harness hashes matched after execution. This validates these dispatch
and reuse cases only; it is not a new performance or broad quality benchmark.

The first smoke attempt successfully loaded the model and checked readiness,
then rejected the harness's invalid 64-character cache scope before generation.
The corrected harness uses the required 32-character scope. That failed attempt
remains a separate receipt, not a model regression or a passing case.

| Private aggregate receipt | SHA-256 |
| --- | --- |
| Failed smoke v1 | `c6af99d2e37322f9ac887af501069fe40c44fa4fdec1ee507b3097a43ec9833f` |
| Passed smoke v2 | `dc37af1bcd4340744e196cc9f5bd3bc0564831c7274a021ab1a53f50d2535697` |

## Local checks

The model-free profile policy tests passed 2/2. The default worker suite passed
117 tests with two ignored, and the no-default-feature protocol suite passed
77 tests. The final `glm-exl3` feature worker suite also passed 117 with two
ignored, and the Cargo profile-policy integration suite passed 2/2. Feature
worker Clippy passed with warnings denied. Formatting, converter Ruff and
repository policy checks passed. The complete converter suite plus the readout
comparison and prefix-validator script regressions passed 6,017 tests and seven
subtests, with 89 skipped, using the locked uv/Python 3.13.13 environment and
offline Hugging Face mode.

The independent source review found one narrow missing-field acceptance gap in
the new receipt contract; requiring explicit non-private disposition closed it.
The reviewer cleared the final change. Historical model-bound matrices and
quality results remain bound to their original artifacts and are not silently
reissued for this worker.
