# Qwen3.8 Uzu loading and sampled generation, 2026-10-29–30

This follows the [Uzu format qualification](QWEN38_UZU_FORMAT_2026_10_29.md).
The converted text artifact, weight quantization and no-speculation policy
remain unchanged. These experiments do not establish consistent 77–80 tok/s.

## Workload and controls

The host was an M5 Max with 40 GPU cores, 128 GiB unified memory and macOS
26.7.1 (25G241), using Automatic AC Energy Mode. The converted artifact contains
104,165,407,864 bytes of tensor storage. Its model hash is
`c493d4a3f876f6495f42127744b6415fe4cd3287f017fbc4b12ecd6d1245e657`;
configuration hash is
`cd9fef82e5016e282dc1920600776a1fee5c3560b86bc7787b883b83c732410e`.
The source checkpoint revision and complete conversion provenance remain in
the preceding qualification record.

Generation uses 4,096 input tokens, a 4,352-token context, 256 sampled tokens,
temperature 1, top-p 0.95 and seed 42. The two cases use top-k 20 and unbounded
nucleus sampling. Rates count the 255 continuation steps after the first
sample. Input token hash:
`fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`.
Expected output hashes:

| Case | SHA-256 |
| --- | --- |
| Top-k 20 | `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0` |
| Nucleus | `9c33d3a4ef1fef97562e00f7f88b5dc287aacadb16b50c4739aec28dbea1b614` |

BF16 activations, F32 reductions/logits/GDN state, Q8 attention caches,
resident experts and paged n-grams retain the preceding precision policy.
No MTP, speculation or re-quantization is used. Rust 1.98.1 builds use separate
Cargo targets for every source workspace. Root-owned performance runs execute
without concurrent compilation or other model/GPU experiments. Private records
retain aggregate metrics and hashes; model content and token vectors are not
published.

## Restoring converted affine parameters

The artifact stores affine BF16 parameters in canonical group-major order.
The loader restores the logical order once while loading weights. The original
group-outer loop writes two bytes at widely separated destination addresses.
The new AArch64 path transposes 8×8 integer-word tiles with NEON, within
64-row tiles. It performs no floating-point arithmetic. Unaligned buffers,
unsupported CPUs and remaining rows/groups use byte copies. Invalid padding
retains the original error and earlier-group partial writes.

Component controls use three physical parameter planes from this converted
artifact, four distinct input/output copies, a 256 MiB cache scrub outside the
timer, two warm and twelve balanced ABC/CBA observations per arm. All output
bytes match the reference, including exceptional BF16 encodings.

| Plane | Original median µs | Blocked byte copy µs | NEON median µs |
| --- | ---: | ---: | ---: |
| Expert down, 1,310,720 rows × 10 groups | 4,259.08 | 4,332.23 | 1,866.29 |
| Expert gate, 327,680 rows × 40 groups | 10,861.66 | 4,821.90 | 1,644.47 |
| HC down, 320 rows × 160 groups | 19.61 | 22.02 | 9.76 |

Six focused production tests cover every 16-bit encoding, tails, independent
source/destination alignment, canaries, checked vector bounds and overflow,
padding error/partial-write parity, and read failure ordering. These gains
apply to loading and the first request's startup cost. Decode does not repeat
this transpose.

A fresh-process ABBA control uses one executable with its load-time selector,
the same artifact, and two reference/two NEON loads. Each load runs one warm
and two measured requests for each sampling case. File identity checks bracket
the complete experiment. All 24 generated sequences match the expected hashes.

| Load route | Process-to-ready observations, seconds | Median seconds |
| --- | --- | ---: |
| Original | 20.858, 20.213 | 20.535 |
| NEON | 18.213, 17.969 | 18.091 |

The median startup reduction is 2.444 seconds, or 11.90%. It includes worker
startup and complete model loading, and does not claim cold-disk isolation.
Warm pooled top-k rates are 68.18 reference / 64.61 NEON tok/s, and nucleus
rates are 64.09 / 63.51 tok/s. Decode kernels and loaded bytes are identical;
these between-load observations do not establish a decode improvement.

## Exact nucleus prefix research

The Top64 candidate selects the same ordered leading tokens and computes all
rounded F64 exponential terms in token-ID order. An outward-rounded interval
bounds the original sorted sum. A prefix is accepted only when that entire
interval proves the original stopping decision; otherwise the full ordering
is required. The CDF accumulation, strict draw comparison and SplitMix64 state
remain identical. The positive-summation error bound follows the standard
analysis in [Higham (1993), section 2](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf),
with a conservative margin and directed endpoint rounding. The fallback
preserves exact top-p filtering.

The initial V4 research worker's scalar exponential helper and original
expression compile to F32 subtraction/division, F64 conversion and the same
platform C `exp`. Model-bound controls compare every probability's bits,
the sorted total's interval containment, the kept prefix, sampled IDs and RNG
state. Frozen real readouts at 32 and 4,096 tokens pass all controls.

| Frozen readout | Original nucleus median µs | V4 shortcut median µs |
| --- | ---: | ---: |
| 32 tokens | 888.83 | 350.71 |
| 4,096 tokens | 910.94 | 350.29 |

These are CPU component measurements: eight warm and 32 alternating AB/BA
pairs per readout. All 32 measured shortcut draws use a proved prefix for
each frozen readout. Top-k 20 follows the original algorithm.

The complete 256-token generation exercises a wider distribution: 194 draws
prove the Top64 prefix and 62 require the full ordering. A same-loaded V4
control has eight warm requests and 32 balanced measured requests, with exact
hash parity for all 40 requests:

| Case | Original pooled tok/s | V4 pooled tok/s |
| --- | ---: | ---: |
| Top-k 20, unchanged algorithm | 58.31 | 59.02 |
| Nucleus | 55.56 | 57.51 |

The nucleus differences across four balanced cycles are −3.56%, +1.86%,
+4.79% and +9.77%. The unchanged top-k control also varies. Rootless IOReport
observations over complete request windows show active-residency-weighted GPU
frequency declining from about 1,133 MHz early to about 750 MHz late. Those
windows include prefill, and the observations do not identify a cause. The
pooled 3.51% nucleus difference is not a qualified stable end-to-end gain.

The retained sampler adds an exact reuse fallback: when the nucleus exceeds
64 or a comparison is uncertain, the existing full ordering indexes the
already-rounded probabilities for the original sum, filter and draw. Selector
budget exhaustion falls back before computing probabilities. The shortcut is
enabled only for Qwen3.8's Lily sampled-decode path. Other hybrid models retain
their reference route. The reference and shortcut share one scalar exp helper.
See [the numerical argument and lifetime rules](../NUMERICAL_SAMPLING.md).

The final V5b source passes new model-bound CPU controls with the same two
readout hashes as V4. Eight warm and 32 alternating AB/BA pairs compare against
the original exp-expression sampler. Probability bits, exact prefix,
seeded token/RNG state and cleared scratch lengths all match.

| Frozen readout | Original nucleus median µs | V5b median µs |
| --- | ---: | ---: |
| 32 tokens | 899.04 | 350.02 |
| 4,096 tokens | 920.48 | 349.77 |

The corresponding top-k medians are 79.417 / 79.500 µs at 32 tokens and
78.375 / 78.375 µs at 4,096 tokens. Each nucleus readout exercises the proved
prefix on all 32 measured draws. The retained probability payload is 1,986,560
bytes at this vocabulary, plus at most 64 retained IDs; lengths clear after
draws, errors and request boundaries. Full fallback ordering remains temporary.

The final V5b same-loaded control repeats the eight warm and 32 measured
request schedule. Its reference arm uses the common scalar helper, whose bits
are checked against the original expression above. All 40 output hashes and
scratch-cleanup acknowledgements pass.

| Case | Reference pooled tok/s | V5b pooled tok/s |
| --- | ---: | ---: |
| Top-k 20, unchanged algorithm | 57.743 | 57.249 |
| Nucleus | 55.648 | 55.634 |

Nucleus cycle differences are +1.38%, −2.58%, −0.96% and +1.84%.
The complete-request GPU frequency observations decline from about 1,150 MHz
early to 704 MHz late. Sustained nucleus throughput is effectively unchanged
in this control. The retained optimization has a qualified CPU component
gain; a consistent end-to-end decode gain and the 77–80 tok/s target remain
unestablished.

Separate production-binary replays check each retained change without private
selectors: three requests per case, with the first treated as warmup. Every
request retains the expected output hash and 256 sampled tokens.

| Production build | Top-k warm median tok/s | Nucleus warm median tok/s |
| --- | ---: | ---: |
| NEON loader alone | 75.06 | 72.64 |
| NEON and final sampler | 63.88 | 74.96 |

These separate-load windows qualify the production path and parity. They are
not matched controls for a throughput improvement. One additional NEON-only
run initially retained the benchmark CLI's default top-k 20; it is recorded
separately and excluded from the nucleus results above. The corrected nucleus
run explicitly disables top-k.

## Rejected and unqualified trials

Simple sampler buffer reuse regresses the same-loaded pooled controls by
0.55% for top-k and 1.40% for nucleus. Keeping packed IDs saves about 3.15%
of nucleus CPU component time, but its generation controls remain noisy.
Neither is retained as an independent optimization.

A stride-aware port of the standard Uzu GEMV core is tested with several
threadgroup geometries and Metal language versions. It produces no gain on
the actual Q4 GDN shape and loses on QSA QKV; a few smaller synthetic shapes
improve by only a few percent. A fused MoE gate/up port also loses on actual
compact expert weights. These kernels are not enabled. The preceding
[Metal hotpath record](QWEN38_METAL_HOTPATHS_2026_10_29.md) remains applicable.

A separate BF16 attention-cache experiment changes full-vocabulary readouts
materially and is not retained. The shipped Q8 cache policy is unchanged.

## Source binding and validation

The base is `66b29b16f684f8f237d775c96849cb9f34d6f5d7`. The production
generator contains the NEON loader and the Lily-scoped nucleus shortcut, with
no private benchmark selectors or response fields. The private V5b control
adds request-ID arm selection and aggregate scratch acknowledgements; its
sampling kernels and NEON loading match the production implementation.

| Build | Generator SHA-256 | Native source manifest SHA-256 |
| --- | --- | --- |
| Production NEON loader alone | `eab1f1654a69870bb2be71ef1d8570902f3c9ad6596c6e9d6edbef69bef309f5` | `3016898d11bb4c083d67ca87e59797b3a18dc28c7098dc440966242c2a705a5b` |
| NEON private load control | `a95c1cddbabab85daf499e39d10974b1e9c37d509883599f6baed6b3fe8c146b` | `e9f55bca5d75fa7c2f63f998e815a68b553fc79897591b308d258a1bb1b04071` |
| Production NEON and sampler | `3fca1cb56dc0ab2d8ab9b870a2c5e0ae9a92289900e88cf70768f237f3445544` | `8fffa7222e115d450919de48d3791b6ddc14c37d584ead12a03aca396ac7a64e` |
| V5b private sampler control | `97302f00efc763594662b1efd3f90b8d552606b99c8ed467353e00d6e0f3e57f` | `ad4ddc9297dd9452f8c3ee32c53f271f62e0d4aae7f02e03e9eae49c18890786` |

Private aggregate evidence is sealed separately:

| Artifact | SHA-256 |
| --- | --- |
| Actual-plane transpose component summary | `f733d9a4c0607315b0709d390799bd2a35b90237f48371d603d5dce07e1d2e7a` |
| Fresh-load ABBA requests | `ec636115064049bfbb60016ed43dd12b369f272cd1ed68c5040fa06941e7a3c6` |
| V5b real-readout CPU summary | `01611224482b8bf56c35a4edbacb4b0dc1cc19fec98d7f1d3fbf3f18b3bc84ac` |
| V5b same-loaded generation requests | `3ee10028d930c47000d4372a062762f25cf64871e7b464648482667372800d4d` |
| Final production top-k replays | `fcc71de62a802de8fe7226b36ee912a67e1470ddd69e9ceb2f6435b24d5c79c8` |
| Final production nucleus replays | `efc816f60d03e8de470e302038042efd4bd6250f5ff7cd46cb472faa0e2938ca` |

Production checks include 62 generator unit tests (two ignored), 16 core tests
(nine ignored), six worker process cases, all-target release generator Clippy
with warnings denied, formatting, repository policy and its 41 self-tests.
Seven new sampler tests compare original probability expressions, exact prefix
and draw/RNG behavior across random, tied, signed-zero, threshold, underflow,
parallel, fallback, unsupported and failure cases. The six new loader tests
and the model-bound CPU/load controls are described above. Independent source
review covers NEON pointer bounds/permutation and sampler numerical/lifetime
contracts. No Metal decode kernel is changed by the retained patch.
