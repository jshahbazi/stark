# Qwen3.8 MTP with the optimized Metal branch

Date: 2026-10-31. Final execution source:
`2c78ea6cecc85393d8688cc73b20ecd5ed97ae04`.
[Machine-readable aggregates](data/qwen38-mtp-combination-2026-10-31.json).

The final Lily configuration averaged **82.85 tok/s with MTP**, versus
**72.97 off**, across 16 same-loaded pairs. All 16 MTP requests exceeded
77 tok/s. The original **78.33 tok/s** implementation remains preserved on
`codex/qwen38-exact-mtp` at `a060ff31500f2338fdff7c309e3081f833ecdf73`,
with its frozen executable and source-bound records intact.

The official mixed-affine checkpoint uses a serial verifier to preserve its
new target arithmetic. MTP was slower there. This study therefore retains
both checkpoint paths and records their results separately.

## Final longer Lily runs

Each request uses 4,096 input tokens, 256 sampled outputs, temperature 1,
top-p 0.95, seed 42, and two MTP drafts when enabled. Top-k is either 20 or
omitted. Rates count 255 post-first-token decode advances per request and
exclude prefill. Each mode has one excluded warmup per route and randomized
balanced ABBA pair order, using one loaded worker for all its requests.

| Sampling | Measured pairs | MTP off, pooled tok/s | MTP on, pooled tok/s | On/off change | MTP pair wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k 20 | 16 | 72.97 | **82.85** | **+13.54%** | 16/16 |
| Nucleus, no top-k | 8 | 73.28 | **74.53** | +1.70% | 7/8 |

| Sampling | MTP minimum / median / maximum tok/s | MTP requests at least 77 tok/s | Median whole-call seconds, off / on |
| --- | --- | ---: | --- |
| Top-k 20 | 78.61 / 84.14 / 85.82 | 16/16 | 5.504 / 5.143 |
| Nucleus | 73.39 / 74.72 / 75.30 | 0/8 | 5.481 / 5.485 |

Top-k MTP pooled at 85.18 tok/s in the first eight pairs and 80.64 in the
last eight. The observed floor holds for this window; these results do not
establish a floor across arbitrary prompts, contexts, desktop GPU activity
or sustained hours. The earlier 78.33 result and this result come from
different timing windows. Their difference does not isolate a source gain.
Nucleus whole-call latency was essentially unchanged despite its small
decode-rate gain.

The final worker SHA-256 is
`7142467b0036a99af262a903cd8c85d3a1768da0a1c67813f5feab5fa857e09a`.
Its native input inventory SHA-256 is
`5c09cd52b2fb8ad85e8b2734693e2df1a4304ec3b9f698a715f4bb238ea0ddba`.
Builds use Rust 1.98.1 and locked dependencies. Documentation updates after
the measured source do not change execution code.

## What transferred

The optimized non-MTP branch was `codex/qwen38-mlx-benchmark`, with runtime
changes in `74e6c5c2d612f58555e8a85a2013dad23f02d9d7`. The combined branch
retains its official checkpoint adapter and precision policies, plus the
earlier exact MTP implementation. Most of its new GPU kernels explicitly
select the official policy. Lily continues using its qualified batched
target arithmetic.

Two host changes apply to the older Lily MTP path:

1. `615355a6b2022e0fb5a1a0d4566652925f5cf71d` transfers the exact chunked
   finite-value guard to draft and verifier readouts. Verification retains
   one owned flat F32 buffer and borrows its rows during sampling, removing
   the second allocation/copy of every vocabulary row. Ownership continues
   through acceptance, rollback and early termination. No GPU arithmetic
   changes.
2. `2c78ea6cecc85393d8688cc73b20ecd5ed97ae04` makes the newer nucleus-prefix
   proof available to MTP distributions. It returns the same retained IDs,
   F64 weight bits and cumulative total as the reference. Uncertain cases
   use the reference order. Draft acceptance, residual correction and both
   RNG streams retain their previous behavior. Top-k 20 retains the
   reference distribution route.

The nucleus change uses the existing probability terms and a proof of the
retained boundary. It adds no approximation, candidate cap, requantization
or alternative target distribution.

## Source-comparison controls

Each comparison loads reference/candidate/candidate/reference. Every load
excludes an off and an on warmup, then measures two off/on pairs with
opposite route orders. This gives four measured requests per source and
route, 16 measured requests and eight warmups per comparison. All 24 calls
must match their historical route hashes. Model loads and requests are
serialized; the private harness checks for competing model workers before
and after every request.

| Change | Sampling | Reference off / on tok/s | Candidate off / on tok/s | Candidate-on versus reference-on |
| --- | --- | --- | --- | ---: |
| Combined branch plus host readout change, versus preserved MTP | Top-k 20 | 71.61 / 79.12 | 68.94 / 81.91 | +3.54% |
| Nucleus distribution proof, versus host-change worker | Nucleus | 69.02 / 67.00 | 71.34 / 75.59 | +12.82% |

The host comparison's reference-on load rate fell 7.10% between its first
and last loads. Its observed 3.54% uplift does **not establish a reliable
3% source gain**. In the nucleus comparison both candidate-on loads beat
both reference-on loads, while the off control also differed by 3.37%.
The bounded nucleus result supports a benefit; the full 12.82% cannot be
attributed solely to the source change. The longer same-loaded runs above
measure the resulting on/off benefit separately.

| Executable | Native source | SHA-256 |
| --- | --- | --- |
| Preserved original MTP | `a1a2a5f4168235680164e0bffb6dd874df4771d2` | `81945ba24640c044ca43dd91b76577d361f063c290a7a905181da8183c4b6f06` |
| Initial combined worker | `4ba628e1efbe56005feda9ded03ae15b7374aa17` | `7ddc77fe64a51ff676634ece31eb5f69506a0a613a649a90ef9970fab5c03dd6` |
| Host readout change | `615355a6b2022e0fb5a1a0d4566652925f5cf71d` | `3d75a47eefbab5cd25aff2aaae67aa6969c032ebda0d8417501c5e6c1928cd60` |
| Final nucleus transfer | `2c78ea6cecc85393d8688cc73b20ecd5ed97ae04` | `7142467b0036a99af262a903cd8c85d3a1768da0a1c67813f5feab5fa857e09a` |

The original branch's later `a060ff31` commit is documentation; its frozen
executable remains the original `a1a2a5f4` native build.

## Official mixed-affine checkpoint

The checkpoint includes 72 MTP tensor planes and one full-attention MTP
layer. The adapter now admits that existing head and shares its trunk
vocabulary head, preserving stored tensor ranges and precision.

Its optimized target uses stock HC boundaries, F32 ValueKey GDN state,
affine-64 Q8 K/V, and BF16-valued vocabulary-head outputs. The earlier Lily
batch kernels do not preserve these boundaries. Official host verification
therefore executes ordinary decode per row and reports
`serial-decode-exact-v1`. The incompatible batched route fails closed.
Lily continues reporting `batched-decode-exact-v1`.

| Drafts | Measured pairs | MTP off tok/s | MTP on tok/s | On/off change | MTP pair wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 44.22 | 36.83 | -16.73% | 0/4 |
| 2 | 8 | 52.37 | 36.47 | -30.36% | 0/8 |
| 3 | 4 | 49.76 | 30.85 | -38.02% | 0/4 |

These initial-combination measurements use the frozen `7ddc77fe` worker,
before the host and nucleus followups. They qualify the serial fallback;
they do not rule out a future exact batched official verifier. That work
needs official-policy GDN, HC, MoE, QSA and head row kernels, followed by
full readout and rollback validation. The earlier
[non-speculative maximum 66.81 tok/s](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md)
uses a different window and a maximum-request statistic.

## Checkpoints, host and input identity

No weights were converted or requantized during this study.

| Path | Repository and immutable revision | Verified payload inventory |
| --- | --- | --- |
| Lily | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | 55 shards, 106,905,155,192 bytes |
| Official | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` at `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | 113 files, 107,326,639,344 bytes |

Lily retains its BF16 activations, F32 reductions/logits/GDN state and Q8
K/V policy. Official precision remains the policy qualified by its
[matched-checkpoint study](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md).
The packs have different quantization profiles; these results do not rank
engines independently of checkpoint choice.

Both inventories reuse prior complete payload-hash receipts. All Lily
shards match their recorded inode, size and nanosecond mtime; device IDs
differ across mount sessions and are explicitly excluded. All 113 official
file identities match. Configuration and index hashes were checked; the
full weight payloads were not rehashed in this run.
Original source-receipt verification labels remain unchanged; these local
validation runs have separate identities and records.

The host is an M5 Max with 40 GPU cores, 128 GiB, macOS 26.7.1 (25G241),
on AC power in Automatic mode. Desktop apps and power settings were not
changed. Two interrupted benchmark attempts overlapped a leftover model
worker and were discarded without performance claims. The stopped job's
surviving processes were cleaned up before the retained timing series.
Other desktop GPU activity and clock drift remain uncontrolled.

The public corpus is upstream Lily `src/qwen4exp/model.rs` at
`e22d58fa7cc488493fb1b5d810a198bd29730aeb`. Its SHA-256 is
`daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955`;
the 4,096-token compact-JSON SHA-256 is
`fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`.
All timing runs disable stage profiling and prefix reuse.

## Numerical and lifecycle controls

Both checkpoint policies pass **140 complete target readout comparisons**,
covering widths 1–4, accepted prefixes, dense/sparse boundaries and rollback
continuations. Each policy compares 34,764,800 target F32 values with zero
differing bits, zero errors and zero target-distribution divergence.
Four additional draft-head comparisons can differ after catch-up; their
actual proposal distributions are retained for exact correction.

The sampler tests compare complete ordered distributions, weight/total bits,
acceptance, rejection, bonus draws and both RNG states. The finite guard
tests IEEE edge cases, chunk boundaries and tails. Final real-worker
lifecycle controls pass for top-k and nucleus, each with 23 requests,
15 bounded tails, greedy parity, partial-group stopping, invalid-request
recovery and stream cancellation followed by an identical fresh request.
All measured and warmup completions match their strict route references.

Final regression checks: generator 78 passed; scorer 41 test executions;
31 Python benchmark checks; both worker Clippy checks, formatting and
repository policy passed. The physical CPU/Metal suite passed 673 tests
with 48 ignored, plus seven CXX and three documentation tests. Vendor
default tests passed 72 with 47 ignored; nine selected ignored MTP kernel
correctness controls also passed. Timing experiments were excluded.

The separate source review found no actionable correctness issues.
These are bounded numerical, sampling, lifecycle and local regression
controls. They do not establish cross-engine quality equivalence or
performance over arbitrary workloads.

## Default policy

`mtp_drafts` remains **0 by default**. Two drafts are useful for the measured
Lily top-k workload. Nucleus has a small on/off gain and essentially equal
whole-call latency; the official serial path loses throughput. A general
default is not supported by these measurements.

MTP keeps the target sampling distribution and adds draft computation,
verification, state/capture memory and prompt preparation. Its benefit
depends on acceptance and batching. A model-specific default needs broader
prompt, context and short-request latency checks. The MTP-capable worker
loads its head at startup, so the paired off/on controls hold that memory
cost constant; setting drafts to zero does not unload it.

## Reproduction

Use the pinned Lily artifact with its native source receipt, the public
corpus revision/hash above, and a Rust 1.98.1 locked generator build:

```sh
mkdir -p runtime/mtp-reproduction
chmod 700 runtime/mtp-reproduction
CARGO_TARGET_DIR=runtime/mtp-reproduction/build cargo +1.98.1 build --release --locked -p stark-generate
curl --fail --location --silent --show-error --output runtime/mtp-reproduction/pinned-lily-model.rs https://raw.githubusercontent.com/fabiogreter/lily-qwen3.8-flash-next/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/qwen4exp/model.rs
uv run --project tools/converter --locked --python 3.13.13 python -c 'from pathlib import Path; from hashlib import sha256; assert sha256(Path("runtime/mtp-reproduction/pinned-lily-model.rs").read_bytes()).hexdigest() == "daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955"'
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_mtp.py --model models/qwen38-lily-q4-xl --worker runtime/mtp-reproduction/build/release/stark-generate --corpus runtime/mtp-reproduction/pinned-lily-model.rs --input-tokens 4096 --tokens 256 --blocks 16 --drafts 2 --top-k 20 --top-p 0.95 --seed 42 --output runtime/mtp-reproduction/top20.json
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_mtp.py --model models/qwen38-lily-q4-xl --worker runtime/mtp-reproduction/build/release/stark-generate --corpus runtime/mtp-reproduction/pinned-lily-model.rs --input-tokens 4096 --tokens 256 --blocks 8 --drafts 2 --top-k 0 --top-p 0.95 --seed 42 --output runtime/mtp-reproduction/nucleus.json
```

Official receipt-bound engineering controls use `--checkpoint-receipt`
with the pinned complete-hash receipt and unchanged local file identities.
The private ABBA checker, raw timing reports, executable/source archives,
native validation records and preservation index remain ignored. The
machine-readable record publishes their aggregate identities and hashes.
Prompts, generated text, token vectors, logits and probabilities are not
published.
