# Qwen3.8 sampled generation and BF16 profiling

This follows the [mixed-precision integration](QWEN38_MIXED_PRECISION_2026_10_28.md).
All comparisons here use the same Lily q4-xl artifact and precision policy.
No weight conversion, quantization change, draft model or speculation is used.
The earlier greedy results remain source-bound evidence for their named build.

## Fixed model, host and workload

- Model: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`; 55 shards,
  106,905,155,192 bytes. All shard stat identities still match the previous
  complete SHA-256 verification; no model file was modified.
- Execution: BF16 activations, F32 reductions/logits/GDN recurrent state,
  explicitly selected Q8 attention K/V. Resident experts, read-only paged
  n-gram table, MTP and vision disabled. Context remains capped at 16,384.
- Host: Apple M5 Max, 40 GPU cores, 128 GB memory, macOS 26.7.1 (25G241), AC
  power. No thermal or performance warning was recorded at baseline.
- Build: locked Rust 1.98.1. Python runs through uv 0.12.24 with the locked
  converter environment and Python 3.13.13.
- Workload: the first 4,096 token IDs from the same pinned upstream
  `src/qwen4exp/model.rs` public corpus used in the preceding study, tokenized
  without added special tokens; 256 requested outputs, context budget 4,352.
- Sampling: temperature 1.0, top-p 0.95, seed 42; top-k 20 is the checkpoint's
  default. A separate case omits top-k. Every replay starts with the same seed
  and fresh request state. Decode rate counts 255 steps after the prefill draw.
- The first request is warmup. Uninstrumented measurements run serially,
  without builds, tests, downloads or other inference in progress. Raw runtime
  records remain ignored; only aggregate metrics and hashes are published.

## Initial sampled baseline and exact partial selection

The preserved generator from `8dc6ccd0769df9261b4fe9454f0861b4c9853836` has
SHA-256 `21a8fdd1983b2c9579c6af08b6ae172f2d2240dd77ba6e47fa320e0d17911cbb`.
Its sampled path reads the full F32 logits, sorts all 248,320 vocabulary IDs,
then truncates to top-k. Greedy decoding's GPU token readout and encoding overlap
were not used by sampled requests.

| Initial comparison | Requests | Warm decode tok/s |
| --- | ---: | ---: |
| Preserved worker, top-k 20 | 5 | 47.0501 |
| Exact partial selection, top-k 20 | 5 | 68.3783 |
| Preserved worker, no top-k | 3 | 36.0387 |

Partial selection improves the top-k 20 case by 45.33%. All ten top-k requests
produce the same 256-token output hash. It uses the existing descending F32
`total_cmp` order and ascending-ID tie break, selects the required prefix, and
sorts that prefix. The F32 subtraction/division, serial F64 exponent sums,
filtering, cumulative draw and 53-bit SplitMix64 RNG are unchanged. Differential
controls compare candidate order, seeded choices and RNG advancement with the
original full-sort algorithm, including ties, signed zero and finite extremes.

The partial-selection/profile worker has SHA-256
`d28a1ee212131b3cbe007c79fdee9fb37b10ce8760364b43e63d5d57d643af73`.
These initial measurements are retained separately from the later changes.

## Initial native BF16 profile

An explicit `STARK_QWEN38_PROFILE_OUTPUT` setting enables private aggregate
JSONL output. The destination must be new and is created with mode `0600`.
Default execution has no profiling file or GPU timestamp writes. The record
contains fixed category names, counts, byte totals and durations; it contains no
prompts, responses, token vectors, logits or decision probabilities.

The profiler adds precise Metal 4 counter-heap timestamps at decode stage
boundaries, retains one command buffer per token, and preserves the existing
dispatch concurrency and dependency barriers. Heap indices and result sizes
are bounded and checked. Counter memory is read only after verified completion
feedback and remains owned through failure draining. GPU ticks are converted
using the device's reported timestamp frequency and checked against the
independent completed-command duration. All 14 upstream math shaders are
byte-identical to the preceding build.

Profiles use 4K input and 64 outputs, three requests, discarding the first.
Each warm request has 63 decode passes and 64 CPU sampler/readout calls: the
first sampled token comes from prefill. Instrumented and uninstrumented controls
produce identical 64-token output hashes for both sampling cases. Timestamp
writes can perturb scheduling. Stage-boundary intervals include GPU scheduling
gaps. Production throughput comes from separate runs without timestamp writes.

| Instrumented top-k 20 GPU interval | Mean ms/decode | Share |
| --- | ---: | ---: |
| MoE | 4.4161 | 31.4% |
| Hyperconnections | 3.1828 | 22.6% |
| GDN | 3.0060 | 21.4% |
| QSA | 2.1965 | 15.6% |
| Vocabulary head | 1.1396 | 8.1% |
| PLE | 0.1000 | 0.7% |
| GPU draw/finite guard | 0.0135 | 0.1% |
| Embedding | 0.0053 | <0.1% |
| Measured span / completed-command span | 14.0599 / 14.0599 | 100.0% coverage |

| Instrumented CPU category | Top-k 20, mean ms/call | No top-k, mean ms/call |
| --- | ---: | ---: |
| Candidate selection | 0.2900 | 3.7582 |
| Exponents and filtering | 0.000409 | 0.5353 |
| Complete CPU sampler, inclusive | 0.3512 | 4.3514 |
| Command encoding | 0.6397 | 0.5353 |
| N-gram staging | 0.0446 | — |
| F32 logit copy | 0.0146 | — |
| Logit finite validation | 0.0610 | — |
| Timestamp resolution | 0.0067 | — |

CPU wait time is inclusive blocking wall time, mostly GPU execution; it is not
additional CPU-active work. Nested CPU counters and GPU intervals must not be
summed into an end-to-end duration. The two sampling cases follow different
token trajectories, so their GPU-time difference is not an isolated sampler
comparison. These measurements guided faster full-vocabulary ordering and
overlapping CPU encoding around the existing host sampler.

## Final sampler and scheduling changes

Full-vocabulary ordering now uses four stable 8-bit radix passes over a
descending monotonic F32 key. IDs start in ascending order, preserving the old
tie break, including signed zeros. Bounded top-k retains exact partial selection.
The exponent, filtering, cumulative sum and random draw code remains unchanged.

A request-scoped sampled decoder pre-encodes the next command while the current
GPU pass executes. It waits for verified completion, copies and validates the
full logits, invokes Stark's existing CPU sampler, and stages that actual
selected ID before submitting the encoded future. The unused GPU argmax is
never consumed as sampled input. Dispatch is bounded by the exact output budget.
Early cleanup drains submitted work without calling the sampler or consuming
randomness. Callback errors remain recoverable after draining; GPU failures
remain fenced. Greedy and scoring behavior retain their existing contracts.

The final worker is `f15b471baf1bd929d0d0233f3d9894042276d10e839cc1a9d7ae3148e0264627`.
An additional release rebuild after adding the ignored component benchmark
produces the same executable bytes.

### Throughput and repeatability

Both the preserved baseline and final candidate were loaded again using the
same configuration. The repeat-window comparison uses five requests per worker,
with four warm measurements after one warmup:

| Sampling | Preserved worker tok/s | Final worker tok/s | Warm median gain |
| --- | ---: | ---: | ---: |
| Temperature 1, top-k 20, top-p 0.95 | 50.6156 | 75.0780 | 48.33% |
| Temperature 1, no top-k, top-p 0.95 | 49.7349 | 65.5722 | 31.84% |

All completed windows are retained below. Rates vary with the measurement
window; the initial no-top-k baseline is substantially slower than its repeat.
These are sequential measurements on one host, without randomized request order.

| Case/window | Warm requests | Warm median tok/s | Warm range tok/s |
| --- | ---: | ---: | ---: |
| Top-k 20, initial baseline | 4 | 47.0501 | 44.3169–49.9563 |
| Top-k 20, initial final-candidate run | 4 | 73.1550 | 72.8474–74.5352 |
| Top-k 20, repeat baseline | 4 | 50.6156 | 46.8847–54.2414 |
| Top-k 20, repeat final candidate | 4 | 75.0780 | 74.4328–75.8917 |
| No top-k, initial baseline | 2 | 36.0387 | 35.7896–36.2877 |
| No top-k, initial final-candidate run | 4 | 54.6887 | 51.2084–59.9460 |
| No top-k, repeat baseline | 4 | 49.7349 | 49.3558–50.6314 |
| No top-k, repeat final candidate | 4 | 65.5722 | 65.1855–66.6610 |

Every throughput request produces 256 outputs and finishes at the requested
length. All 256-output baseline, intermediate, final and instrumented runs have
identical output hashes within each sampling mode. The separate 64-output
profiles match their corresponding uninstrumented controls. A three-request
greedy regression control measures
78.4778 tok/s warm and retains the preceding build's completion hash; it does
not supersede the earlier five-request greedy study.

### Paired CPU component control

An ignored native worker test compares the old full sort with radix ordering
on the same real F32 logits at each of 256 readouts. It follows the same 4K
corpus, seed and nucleus-only trajectory. Comparison order alternates AB/BA to
reduce order effects, and each complete index vector must match. Only timing
aggregates are emitted.

| Selector | Mean ms/readout | Median ms/readout |
| --- | ---: | ---: |
| Original full sort | 3.874821 | 3.851750 |
| Stable radix | 1.146534 | 1.127105 |

Mean CPU selection time falls by 70.41% (3.38x faster); all 256 index orders
match. This isolates the selector comparison from the between-run variation in
end-to-end throughput. Run the control with an exclusive GPU slot:

```sh
STARK_LILY_TEST_MODEL_PATH=/absolute/checkpoint STARK_SAMPLE_CORPUS_PATH=/absolute/pinned-corpus.rs cargo +1.98.1 test --release --locked -p stark-generate private_model_selection_cpu_benchmark -- --ignored --nocapture --test-threads=1
```

### Final BF16 profile

Final profiles use the complete 256-output workload, three requests per case,
with the first discarded. Each warm record has 255 decode passes and 256 CPU
sampler/readout calls. GPU phase sums cover 99.99998–100.00000% of the independent
completed-command span. Instrumented output hashes match the uninstrumented
final worker exactly in both sampling modes.

| Instrumented GPU interval | Top-k 20, mean ms/decode | No top-k, mean ms/decode |
| --- | ---: | ---: |
| MoE | 4.4197 | 4.4841 |
| Hyperconnections | 3.1569 | 3.1889 |
| GDN | 2.9893 | 3.0928 |
| QSA | 2.2270 | 2.2242 |
| Vocabulary head | 1.1381 | 1.1525 |
| PLE | 0.1007 | 0.1132 |
| GPU draw/finite guard | 0.0137 | 0.0138 |
| Embedding | 0.0051 | 0.0144 |
| Measured / command | 14.0504 / 14.0504 | 14.2839 / 14.2839 |

| Instrumented CPU category | Top-k 20, mean ms/call | No top-k, mean ms/call |
| --- | ---: | ---: |
| Candidate selection | 0.2867 | 1.3688 |
| Complete sampler, inclusive | 0.3485 | 2.1778 |
| Command encoding, overlapped | 0.7072 | 0.8089 |
| Wait, inclusive wall time | 13.5911 | 14.1247 |

The nested partial/radix counters are included in candidate-selection time.
Encoding overlaps GPU execution, and waits include remaining GPU work. The
largest attributed GPU intervals remain MoE, hyperconnections and GDN under
this fixed quantization and workload.

## Validation and identities

- Generator release suite: 51 passed; the ignored real-model selector benchmark
  was run separately and passed. Differential controls compare complete ranking,
  seeded choices and RNG advancement with the old sampler across production
  vocabulary size, finite IEEE edge patterns, signed zeros, ties and filters.
- Targeted native Lily suite: six passed, including three complete-checkpoint
  controls. These cover arbitrary non-argmax selected inputs, complete F32
  readout hashes, lazy execution, budgets, early drops, callback counts and
  recovery from callback failures and invalid selected IDs.
- Vendored core: five passed, including counter parsing, late-error feedback,
  fault-safe draining and the real GPU finite-readout guard.
- Default and optimized scorer suites each pass 39 tests. Clippy passes with
  warnings denied for generator and both scorer configurations under Rust
  1.98.1. CPU-only engine admission compiles.
- Context benchmark controls: two passed. Configured Ruff, root/vendor pinned
  formatting, repository policy and staged whitespace checks pass.
- Real worker integration: 13 generation and 12 scoring requests pass, with
  greedy/stochastic replay, request isolation, invalid-ID recovery, streaming
  equality, acknowledged cancellation and recovery, two natural EOS cases,
  finite scoring, candidate softmax, log-partition consistency and consecutive
  cache hit/miss/reset/error behavior. Workers load serially and exit cleanly.
  The protocol exposes up to 16 candidates; this worker control does not
  independently sum all vocabulary probabilities.

The preceding run's model configuration, tokenizer and preparation hashes are
unchanged. All 55 shard stat identities still match the earlier full hash
verification. All 15 vendored Metal files (14 upstream plus Stark's existing
finite guard) are byte-identical to base commit
`8dc6ccd0769df9261b4fe9454f0861b4c9853836`. Dependency locks are unchanged.

| Identity | SHA-256 |
| --- | --- |
| Final generator | `f15b471baf1bd929d0d0233f3d9894042276d10e839cc1a9d7ae3148e0264627` |
| Final optimized scorer | `324cab7677721d711839ac3175f4855c651dc22cb59fa130c993c70237053cf8` |
| Source inventory | `a59e20189a308d95af4ae6815737ad95462d4135d55920513b7db2b13e2cdb09` |
| Public corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| Input IDs | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |
| Top-k 20 output IDs | `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0` |
| Nucleus-only output IDs | `9c33d3a4ef1fef97562e00f7f88b5dc287aacadb16b50c4739aec28dbea1b614` |
| Top-k 20 repeat baseline receipt | `fe6fca866e76c79cce1fe7f21103e95b8a45ae353f2a1e14301e1812f1351d72` |
| Top-k 20 repeat candidate receipt | `688338c997a0bdfafaa4a8efaa9f6a1fb25caa18cacf5e74d5377531a6444215` |
| Nucleus-only repeat baseline receipt | `7311c6cd6a31aaf2079a40b2e893de6ed2ff8ab106b3f64ed40d9e9f934e8849` |
| Nucleus-only repeat candidate receipt | `a24d4ab81d039827097e2ba73565ff151b4eb119727c2800aafa974fadd73668` |
| Top-k 20 final profile | `b946abba7fbc4c17d86852c8b0102ac5a4311bf5553e853c600001849c91b1a4` |
| Nucleus-only final profile | `43c5b5c90c84e7e0ca1503f3d9f38db7e0eeab0adc5ea5361fedf5f12c858fc0` |
| Worker integration receipt | `a325b700dd83dba6b6454bd95774dd489426ed2fc9c5008567857b0edeb20999` |
| Selector component log | `047d497f97ced7c4a4bf490596420bddc04024d97b7ec1e09ae9240682acf345` |

The private source inventory binds 969 source, manifest, toolchain, policy and
test files over the named base commit. It sorts paths and hashes
`UTF8(path) + NUL + raw_SHA256(file)` per entry. The report itself is excluded.
The corpus is the [public source at the pinned upstream revision](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/qwen4exp/model.rs).
Earlier validation and performance records remain separate. This study updates
sampled execution; it makes no new context-length or quantization qualification.
