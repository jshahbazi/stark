# Qwen3 GGUF prefill optimization

Date: 2026-10-13. Selected kernel implementation:
`b4bebb5021fe68d3a6ba86ca8f363eb53f063f38`, restored by
`68374278ec2dc2bd07d02a44e587fadc5cbdcb67` with identical engine files.

The selected change improves imported Qwen3-0.6B Q8_0 prefill throughput by
15.7% at 512 input tokens and 14.9% at 1,024 tokens in the primary six-round
comparison. The remaining throughput gap against native MLX 8-bit is about
7–8% at 512 tokens; 1,024-token throughput is approximately equal. A separate
four-round comparison confirmed improvements of 15.1% and 14.7%.

The compressed model files are unchanged. Full F32 logits and fixed-count
32-token generation match the original Stark worker exactly on the measured
input lengths and the six source-reference cases. The original import's BF16
scale adaptation and its one divergent sequence against llama.cpp remain.

This report adds new source-bound evidence to the
[original import experiment](QWEN3_GGUF_IMPORT_EXPERIMENT.md). Its
[aggregate evidence](experiments/qwen3-prefill-2026-10-13/aggregate.json)
contains model/binary/receipt hashes, per-round measurements, numerical
comparisons, and rejected experiments. Native artifacts remain private.

## Diagnosis and selected change

The imported Q8_0 matrices retain 32-element quantization groups. The existing
MLX 8-bit model uses 128-element groups. The original quantized MXU schedule
coupled its staging depth to the quantization group size, giving the imported
model a 32-column K stage and MLX a 128-column stage. These representations also
differ in coefficient count, coefficient layout, quantizer, and weight bytes.

The original policy also gives eligible large Q8_0 projections a 128×128 tile
with 16 SIMDgroups, or 512 threads. Its quantization legality check rejects that
tile for group sizes above 64, so the group-128 MLX projections fall back to a
64×64 tile with four SIMDgroups, or 128 threads. This makes the formats exercise
different dispatch geometry as well as different staging depths. At 32 and
128 input tokens both use 128-thread geometry; the 32-token tile is 32×64.
The policy and legality contracts are in
[GemmTiling](../crates/stark-engine/src/backends/common/gpu_types/gemm.rs)
and [GEMM policy](../crates/stark-engine/src/backends/metal/kernel/matmul/gemm/policy.rs).

Candidate C jointly changes the group-aware loader and dispatch geometry:

- Stage 64 K columns together while loading two independent 32-element scale
  groups. Every code and coefficient retains its original addressing and
  dequantization arithmetic.
- Dispatch the eligible 128×128 policy tile through the measured 64×64 tile.
  Both retain a 32×32 per-SIMDgroup MMA shape.
- Calculate split-K from the original policy tile, preserving the original
  split count, accumulation boundaries, and BF16 partial-result rounding.
- Require complete 64-column stages. Other operand formats retain their
  existing selection and group-sized staging.

The owning code is
[selection.rs](../crates/stark-engine/src/backends/metal/kernel/matmul/gemm/selection.rs),
[gemm.metal](../crates/stark-engine/src/backends/metal/kernel/matmul/gemm/gemm.metal),
and the
[staged loader](../crates/stark-engine/src/backends/metal/kernel/matmul/gemm/common/schedules/staged.h).

Source inspection and the controlled interventions identify a useful change to
the group-32 schedule. They do not isolate the contribution of the loader versus
tile geometry or identify the hardware bottleneck. No per-kernel GPU counters
were collected. Barrier-only pairing did not improve performance, and widening
the selected stage to 128 columns made performance substantially worse.

## Method and provenance

The host is an Apple M5 Max with 18 CPU cores and 128 GiB unified memory,
Darwin 25.6.0 arm64. Workers use Rust 1.98.1, locked dependencies, release mode,
and the existing optimized features. Model sources and immutable revisions
remain those of the original import experiment.

The native performance harness uses input lengths 32, 128, 512, and 1,024;
context capacity 2,048; two warmups; seven measured scoring repeats per length;
and two measured fixed-count generation repeats after a separate scoring phase.
Every model round runs in a fresh process, serially. Fixed-seed cyclic orders
balance every complete process-order cycle; input-length order rotates by round.
The primary C run has six rounds, and the independent C/E comparison has four.

Scoring creates fresh KV state. Its native prefill timer excludes state creation,
readout, JSON, and private pipe overhead. Paired speedup is the original
round-median latency divided by the candidate's corresponding round-median
latency. Reported milliseconds below are medians of the per-round medians;
their ratios can differ slightly from medians of paired speedup ratios.

Power, temperature, and filesystem cache were uncontrolled. CPU processing of
full logits between native calls affects the workload's duty cycle. Short
32/128-token observations were noisy and do not support a reliable improvement
claim.

| Worker | Immutable source revision | Worker SHA-256 |
| --- | --- | --- |
| Original | `2642d705f63e8080b434d7ec835749d5c71ad77c` | `0cb4e1fcc02286af21213cf9c9b4871c758143839d570069d70baa56f4f5fc80` |
| Selected C | `b4bebb5021fe68d3a6ba86ca8f363eb53f063f38` | `2a83202b52208686c6e076c02d949dc8960c74f608708809eaa7a5f1a452ba56` |
| Rejected A | `b42b52652e17c848ca1f8eeb8a73961639ed314f` | `d7ad71f6e5392f3626ed50797e2bea6e51d0410a8c5e53974ff92b422ce8941a` |
| Rejected E | `7d181d06d6fad558eee58b02496e896755ac43cc` | `b2a0f284a71664c2703c24932641302b99d50b875bc668d393e8207a814f07d1` |

The final production worker built from the restored source is byte-identical to
the measured C worker. Caller-declared source revisions and directly verified
binary hashes are recorded separately in the evidence.

## Performance and memory

Primary six-round C run:

| Input tokens | Original Q8_0 prefill (ms) | Selected C prefill (ms) | MLX 8-bit prefill (ms) | C paired speedup over original |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 5.301 | 5.036 | 5.227 | 1.028×; noisy |
| 128 | 6.916 | 7.357 | 7.700 | 1.009×; noisy |
| 512 | 19.544 | 16.881 | 15.539 | 1.157×, range 1.145–1.196× |
| 1,024 | 33.360 | 29.050 | 28.983 | 1.149×, range 1.121–1.191× |

The controlled original-versus-MLX baseline had throughput gaps of 18.8% at
512 tokens and 12.9% at 1,024 tokens. In the primary C comparison, paired
throughput relative to MLX was 0.923× and 1.001×, respectively.

The separate four-round comparison measured:

| Input tokens | Original Q8_0 (ms) | Selected C (ms) | Rejected E (ms) | MLX 8-bit (ms) | C paired speedup over original |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 19.406 | 16.855 | 27.736 | 15.694 | 1.151×, range 1.143–1.153× |
| 1,024 | 33.207 | 28.956 | 49.357 | 28.872 | 1.147×, range 1.146–1.148× |

Memory maxima from the primary six-round comparison:

| Representation | Weight file (MiB) | Peak process RSS (MiB) | Peak tracked Metal allocation (MiB) |
| --- | ---: | ---: | ---: |
| Original imported Q8_0 | 604.2 | 805.3 | 862.7 |
| Selected C, same Q8_0 artifact | 604.2 | 794.8 | 862.7 |
| Native MLX 8-bit | 586.5 | 781.8 | 828.9 |

The Q8_0 weight file remains 633,562,928 bytes with SHA-256
`d866fa25264aa5244de920525c747d6be8d9afbdb78624f2e09a762750178725`.
All Q8_0 variants used identical config, tokenizer, and weight hashes. No
requantization, dense weight expansion, persistent cache, or model conversion
was introduced.

Original and selected C both peaked at 904,593,408 tracked Metal bytes. Keeping
the original split counts retains their existing partial-result scratch
allocation. RSS varies across process rounds, so the observed lower C maximum
does not establish a general memory reduction. RSS and tracked Metal allocations
share unified-memory pages and must not be added. The larger context and input
lengths here also prevent direct comparison with the original report's RSS
maxima.

## Correctness and rejected alternatives

The primary six-round run had 168 measured score-hash agreements and 48
generation-hash agreements per model. C and the original matched full logits and
32-token generation at all four measured lengths. The independent four-round
run reproduced those exact comparisons, including for E.

A separate source-correctness run used the original six authored cases, context
1,024, 32 generated tokens, and three repeats per case. All Stark workers closed
before llama.cpp started with the original source GGUF.

| Comparison | Full-vocabulary F32 logit hashes | Maximum logit error | Exact 32-token generation hashes |
| --- | ---: | ---: | ---: |
| Selected C versus original Stark | 6/6 | 0 | 6/6 |
| Rejected E versus original Stark | 6/6 | 0 | 6/6 |

Each worker passed all 18 score-repeat, generation-repeat, and generation-first
token consistency checks. Against llama.cpp, each worker had 6/6 tokenizer and
first-token agreements, mean top-32 overlap 97.40%, mean top-32 log-probability
RMSE 0.09475, and 5/6 exact generation-hash agreements. The optimization preserves
the original import's source-runtime differences. It does not establish
bit-exact source-runtime equivalence or general generation quality.

Rejected alternatives are retained as independent evidence:

- A paired two group-sized stages to reduce synchronization. Its paired
  throughput ratios against the original were 0.994× at 512 and 0.979× at 1,024.
  It provided no useful improvement.
- D proposed reducing split partitions. A deterministic 30-case NumPy BF16
  emulation had 0 mismatches against its F32-to-BF16 oracle with the original 12
  partitions and 6 mismatches with three partitions. D was rejected before
  native execution; it has no native correctness or performance result. The
  aggregate records the synthetic recipe for reproduction without raw arrays.
- E widened C's staging depth to 128 while preserving output arithmetic.
  Its 512-token throughput was about 30% below the original and 39% below C.
  The exact outputs did not compensate for the performance regression, and E
  was removed from the selected default.

## Validation and reproduction

On the restored C source, 106 scoped quantized-dispatch tests and six host GEMM
selection tests passed. The twelve new prefill harness tests and seven original
GGUF harness tests passed. Python lint/format, scoped Rust formatting, diff,
and repository policy checks passed. A broader 474-test engine run was stopped
early; no full-suite pass is claimed. The final Rust 1.98.1 locked optimized production worker build
completed and matched C's measured binary hash.

Use the unchanged native model artifacts and separately preserved original/C
binaries with
[qwen3_prefill_experiment.py](../scripts/qwen3_prefill_experiment.py). Its
`--binary NAME=PATH` arguments permit immutable worker versions in one run.
For the primary comparison, use six rounds, two warmups, seven scoring repeats,
two generation repeats, lengths `32 128 512 1024`, context `2048`, and 32 output
tokens. Compare C against both the original Q8_0 worker and the original MLX
worker. The harness rejects existing aggregate destinations and requires ignored
private runtime output directories.

Use
[qwen3_prefill_correctness.py](../scripts/qwen3_prefill_correctness.py) with the
same imported Q8_0 model, named original/C binaries, the original GGUF, and the
installed llama.cpp server to reproduce the separate six-case reference check.
It defaults to three repeats, context `1024`, and 32 output tokens. Prompts,
responses, logits, and token vectors remain in memory and private pipes.

| Evidence | Immutable private receipt SHA-256 |
| --- | --- |
| Original/MLX controlled baseline | `59b0eed291e66938c12d5d62a954e085493f5b390b8022ef73e06756614204b5` |
| Rejected A | `a0be40996d0c785d5b86600ea1b16643efae23ecf01594d863d2e303c63fc9bc` |
| Primary six-round C | `82c9c91c2abef8fc756c744d24c37f6417c88e77b8817c145e07f927bfeb0ac3` |
| Four-round C/E confirmation | `fe60bbcb5ad1208449478144d6f7fb1b018dadaba729de8965a484d8fea207d3` |
| Six-case source correctness | `77682f5559bc9cf375a5bd8f2fd55b83302ba7c18e470dd9a0a3c614e176805f` |

The results qualify the selected change for this model, workload, source-bound
build, and host. Larger models, other Apple GPUs, sustained power conditions,
general group-32 performance, semantic accuracy, and broader GGUF support remain
unqualified.
