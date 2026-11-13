# Ripple Q4_K_M kernel performance, 2026-11-12

Stark's native Ripple Q4_K_M path now uses grouped32 Q4_K and grouped16 Q6_K
projections. The earlier block schedule repeated unnecessary dequantization
work. The mixed artifact's output head is Q6_K; the Q4_K vocabulary-shaped
tensor is the embedding. Both actual projection formats are covered here.

Attention output also replaces a per-token integer modulo with a conditional
subtraction. Host bounds establish `start < capacity`, `count <= capacity` and
a bounded context, so the sum wraps at most once without integer overflow.
Indices and floating-point operation order remain identical, including for
non-power-of-two sliding windows. This precision-neutral indexing change also
applies to Ripple's other graph weight formats.

## Saved-binary comparison

The same 38,798,820,576-byte AtomicChat Q4_K_M artifact was used throughout:
revision `df550a7178a87bbfb6ee2bdbd37603c553d536ee`, SHA-256
`51de49f0833201d82e624c1273a0316dca6c281616371770213f0c6d277b8652`.
Prior content verification was bound to size, modification time and inode
before and after every cohort. The original official metadata revision remains
`e27e171792f303d74c048184e8259ba663cfa0d5`.

Four serialized cohorts ran in baseline/candidate/candidate/baseline order.
Each had one excluded warmup and two measured requests, giving four warm
measurements per arm. Each request used ten input tokens, context 512, greedy
sampling, no speculation and a 64-token output limit. All twelve generations
finished at EOS after 58 sampled tokens. Decode rate counts the 57 tokens
after the prefill token. Every output token hash was identical across arms.

| Warm measurement | Saved baseline | Candidate |
| --- | ---: | ---: |
| Median decode tokens/s | 4.559 | 7.460 |
| Decode range, tokens/s | 4.288–5.268 | 7.129–7.794 |
| Median ten-token prefill, seconds | 2.068 | 1.309 |
| Median full request, seconds | 14.602 | 8.962 |

The observed median decode improvement is **1.64×**, about **64%**. These are
short, unscreened runs on a working M5 Max with 128 GiB unified memory and
macOS 26.7.2. Rates drifted during the session: the baseline cohort medians
were 5.024 and 4.313 tokens/s, versus candidate medians 7.285 and 7.637. The
comparison establishes a local improvement; it does not establish constant
clocks, quiet-host throughput or sustained production speed. AC power and
power mode 2 were observed; no power settings or other applications changed.

The saved baseline comes from `33d336f4`, executable SHA-256
`8c1327616fbf007fb379f3b579fdcc8aea59e6f0461c6bd27acf504a9c886dd8`.
The final release candidate SHA-256 is
`58ec7859301c005fa2f35b9bf3f7937e6d3070b3eb74df1aaf638ac08343f142`.
Generation and example executables used locked Rust 1.98.1 builds. Production
execution uses the Safe Metal library and requires no diagnostic selectors.

## Kernel and numerical evidence

Rotating-bank component measurements used at least 320 MiB of packed weights,
seven alternating ABBA replays and GPU timestamps. Selected median results:

| Projection | Block kernel, µs | Selected grouped kernel, µs |
| --- | ---: | ---: |
| Q4 gate/up, 40960 × 5120 | 378.57 | 204.43 |
| Q4 down, 5120 × 40960 | 398.51 | 217.63 |
| Q4 Q/gate, 10240 × 5120 | 98.77 | 52.72 |
| Q4 KV, 1024 × 5120 | 13.72 | 7.69 |
| Q4 O, 5120 × 10240 | 101.46 | 56.01 |
| Q6 down, 5120 × 40960 | 548.16 | 323.81 |
| Q6 KV, 1024 × 5120 | 17.13 | 12.09 |
| Q6 output head, 100352 × 5120 | 1197.02 | 730.06 |

All decoded-product F64 checks passed. Component relative RMS errors were below
`6.7e-7`. New test-only Q4 row-reuse variants were slower and were not promoted.
The initial component executable was not sealed before later test-harness
additions; its timing body remained unchanged. The separate Q6-head test
executable was sealed. These component observations guided selection; the
saved release-binary comparison above is the primary performance evidence.

The unchanged full 80-layer independent CPU oracle compared eight vocabulary
readouts against the final candidate. Maximum absolute error was
`0.0001506805`, mean absolute error `0.0000279765`, relative L2 error
`0.0000015347`, and greedy agreement 8/8 at the fixed tolerance
`0.001 + 0.001 * abs(reference)`. Reset replay, invalid-token rejection and
failure recovery passed. Elementary arithmetic, literal instruction,
intervening-request isolation and worker failure recovery also passed.

Weights remain packed and unchanged. Activations, accumulation, attention,
controls and KV state remain F32. The grouped matrix kernels reassociate F32
operations and retain their existing extreme-input block fallback. Each
Ripple-specific schedule has its own pipeline cache; other graph defaults
retain their previous routing. Complete-weight admission, streaming fallback,
cache bounds and fresh-per-request state policies are preserved.

The full native readout hash is
`e7f77d011a35d20b7ebf279ff34e520a6b885f02f1dcce0b763d5e791b7a4183`;
the independent reference hash remains
`e7929a809b495f0b85042c4a79f37ca0b4e709487bce8bb7236c2b54045970b3`.

Short 16-step instrumented profiles reached roughly 11–12 tokens/s. A longer
57-step profile measured about 8.2 tokens/s before the attention-index change
and 9.5–9.8 in warm runs afterward. These profiling workloads use a synthetic
repeated prefix and are supplementary; they are not the chat worker result.

This report supersedes timing observations in the
[initial Q4_K_M report](RIPPLE_Q4_K_M_2026_11_12.md). Its earlier source-bound
numerical evidence remains unchanged. The installed app was not rebuilt;
measurements use the native worker. Long-context quality and sustained
throughput remain unqualified. Existing prefix-reuse blockers are unchanged.

Private fixtures, weights, executables and captures stay ignored. Published
evidence contains aggregate results and hashes only.

## Regression validation

The full native CPU/Metal suite passed 776 tests with 82 ignored, plus seven
integration tests and three doctests. This includes non-power-of-two sliding
ring eviction, cache growth, F64 projection checks, extreme-input behavior and
default-route cache isolation. The full log SHA-256 is
`9379cd3324cd19897f3c53658063f84cf30864c9d68f535ea58fe3967828b6b8`.
Rust formatting, repository policy and whitespace checks passed. Independent
review found no actionable issue in dispatch, fallback or index bounds.

The [aggregate evidence](data/ripple-q4-kernels-2026-11-12.json) records cohort
samples, executable and source hashes, component selection, profile summaries
and reference results. No Python or dependency changes were needed.
