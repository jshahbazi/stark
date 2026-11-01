# Qwen3.8 exact MTP generation

This study implements checkpoint-backed multi-token prediction and compares
sampled generation with MTP enabled and disabled on one loaded model. The
implementation is `495f2ea3cb9f98be00ac0a5e57a06687576170ea`, based on
`main` at `5d7a97ec`. MTP is optional and disabled by default.

The exact implementation is slower in every paired block measured here:
22.6–40.1% lower decode throughput. It does not recover the 77–80 tok/s
normal-desktop target. The faster batched-verifier candidate failed numerical
qualification and is excluded.

## Checkpoint and precision

The checkpoint remains `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`. Its 55 shards total
106,905,155,192 bytes. All current file stat identities match the prior
complete SHA-256 verification; this run does not rehash the full checkpoint.
No weights are converted or requantized. Its existing one-layer MTP head,
including its separately quantized readout, is loaded from the same files.

Target execution retains BF16 activations, F32 reductions/logits/GDN state,
Q8 attention K/V, resident experts and a read-only paged n-gram table.
Verification uses the ordinary decode graph and its existing arithmetic.
The extra copies retain recurrent state, projected convolution inputs,
residuals and complete F32 readouts for rollback. Requests stay serialized;
state and draft distributions do not survive a request.

The generation worker loads the declared head once. Each request selects
`mtp_drafts: 0` for off, or 1–3 for on. Off requests skip head execution but
retain the same head allocation as the on requests. Scoring continues to load
without the head.

## Exact sampling

Draft and target share Stark's qualified ordering, temperature and filtering
implementation: F32 subtraction/division, F64 exponentials and ordered sums,
top-k before top-p mass calculation, and optional min-p. There is no
1,024-candidate cap. Proposals are sampled from the actual filtered draft
distribution `q`; verification accepts with `min(1, p/q)` and resamples a
rejection from normalized `max(p-q, 0)`. A fully accepted group gets a target
bonus draw. The algorithm follows
[Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192).

Draft randomness has a separate deterministic stream. MTP can change the
seeded output trajectory while preserving the target distribution. Tests
therefore require reproducibility within each route, rather than equal
sampled hashes across routes. Cleanup never samples. An early stop may discard
already sampled outputs in its verification group; their request-local RNG
state is discarded with the request.

## Numerical validation and rejected batched candidate

The retained verifier matches ordinary target execution on 56 complete
248,320-element readouts: 13,905,920 F32 values, zero unequal bytes. Controls
cover acceptance counts 0–3, partial iterator drops, reset/failure handling,
and prefixes of 4,095 and 4,096 tokens across sparse-attention/indexer
boundaries. Dropping a group invokes no sampling callbacks.

The draft head can differ numerically after different catch-up shapes. The
observed maximum full-vocabulary TV is 0.00403 and there are no changed top
predictions in the four head comparisons. This changes proposal efficiency;
the exact sampler records and verifies the distribution actually used.

An earlier candidate reused batched prefill arithmetic for target verification.
Its MoE router/shared gate and GDN normalization introduced extra BF16
boundaries, and its MoE combination rounded before the shared-expert add.
Correcting these boundaries did not qualify the complete batched path:
the maximum full-vocabulary KL was 0.03337, full-vocabulary TV 0.09650 and
nucleus-filtered TV 0.10138. A rollback comparison also changed one top
prediction. That candidate is rejected and excluded from production. Its
source snapshot and aggregate evidence remain private and ignored.

The retained verifier executes the existing target graph once per proposed
position. It does not amortize target weight traffic through batched arithmetic.
Drafting, unused verification rows and rollback can consequently outweigh
the accepted tokens.

## Paired benchmark protocol

The host is an Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1 (`25G241`),
on AC with Automatic power preferences. No host settings or desktop apps are
changed. Other desktop activity is uncontrolled; no claim of isolated GPU
conditions or sustained clock behavior is made.

Each case loads one worker and runs both routes against the same 4,096-token
public-code prefix, 256 sampled outputs, context 4,352, temperature 1,
top-p 0.95 and seed 42. Top-k 20 and disabled top-k are separate cases.
One warmup per route is excluded, followed by four balanced pairs, eight
measured requests. Both routes reuse the same loaded worker, model, weights
and sampling settings; each request resets logical decode state. The benchmark
removes stage profiling before launch.

Decode rates count 255 advances after the prefill draw. Pooled throughput
divides 1,020 advances per route by summed decode time. All replies must
acknowledge the requested route, return all 256 samples and pass per-route
hash reproducibility. Different on/off trajectories can visit different
experts; these are complete sampled-generation measurements, not an isolated
kernel benchmark.

## Results

| Sampling | Drafts | Off tok/s | On tok/s | Relative change | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k 20, top-p 0.95 | 1 | 66.2539 | 50.9553 | -23.09% | 428 / 592 |
| Top-k 20, top-p 0.95 | 2 | 57.9369 | 44.8290 | -22.62% | 632 / 772 |
| Top-k 20, top-p 0.95 | 3 | 61.1821 | 36.6205 | -40.15% | 640 / 1,136 |
| No top-k, top-p 0.95 | 1 | 60.5458 | 45.8241 | -24.31% | 428 / 588 |

All 40 requests pass, including eight excluded warmups. All 32 measured
requests return 256 samples and are deterministic within their route. Every
one of the 16 pairs has lower throughput with MTP. Baseline rates drift
between cases; each percentage uses its own same-loaded off control. These
workloads do not establish a universal slowdown percentage for other content.

Accepted/proposed measures utilization of all proposed tokens, including
proposals beyond a rejected prefix; it is not a conditional acceptance
probability. The verifier executes 1,184 / 1,160 / 1,516 / 1,180 target rows
for each case's 1,020 measured decode advances. With three drafts, it performs
48.6% more ordinary target forwards than the off route, before drafting and
rollback. Its phase wall times are:

| Sampling / drafts | Draft seconds | Verify seconds | Rollback seconds |
| --- | ---: | ---: | ---: |
| Top-k 20 / 1 | 0.9338 | 18.5146 | 0.4262 |
| Top-k 20 / 2 | 1.3156 | 20.9256 | 0.3667 |
| Top-k 20 / 3 | 1.7941 | 25.5251 | 0.3945 |
| No top-k / 1 | 1.4557 | 19.2235 | 0.5733 |

These aggregate phase times include CPU encoding, GPU waits and readback;
they are not GPU-only counters. Target sampling and iterator overhead are
outside these phases. They explain the current implementation's additional
work without isolating clock or desktop-contention effects.

The three top-k off cases preserve the earlier trajectory hash
`621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0`.
The nucleus off hash is
`9c33d3a4ef1fef97562e00f7f88b5dc287aacadb16b50c4739aec28dbea1b614`.
MTP hashes differ as expected from exact rejection sampling's different RNG
consumption. Each MTP route repeats its own hash across warmup and measurement.

Making MTP faster requires sharing target computation across positions while
preserving the qualified target arithmetic. The current exact route provides
the control and optional execution path for that work; enabling it by default
would reduce throughput on these measurements.

Run the paired control from the repository root after building the worker:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_mtp.py --model models/qwen38-lily-q4-xl --worker target/release/stark-generate --corpus vendor/lily-core/src/qwen4exp/model.rs --input-tokens 4096 --tokens 256 --blocks 4 --drafts 1 --top-k 20 --top-p 0.95 --seed 42 --output runtime/mtp-top20-d1.json
```

The example's current source corpus differs from this study's pinned older
public-code corpus. The [pinned upstream model source](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/qwen4exp/model.rs)
has the recorded corpus hash and reproduces the same input. Use `--top-k 0`
for full-vocabulary nucleus sampling.

## Regression validation

Validation on the implementation source passes:

- Native CPU/Metal engine: 666 tests, 46 ignored; seven CXX tests and three
  documentation tests pass. The model-bound MTP control is run separately.
- Explicit final-source MTP replay: 56 target readouts and 13,905,920 F32
  values remain byte-identical, including rollback and sparse-context controls.
- Release generator: 68 tests pass, two private benchmarks ignored. Exact
  sampler tests include analytic probability-mass checks, 120,000 sampled
  trials, support holes, underflow, tiny residuals and full-vocabulary behavior.
- Optimized scorer and its probes: 39 test executions pass.
- Benchmark harness controls: 15 Python tests pass, including 12 new paired
  MTP harness cases.
- Real-worker lifecycle: 23 generation requests pass, including 15 short
  context-tail cases across depths 1–3, greedy on/off hash parity, invalid
  request recovery, an early stop within a verified group and streaming
  cancellation followed by a matching fresh request.
- Generator and optimized-scorer Clippy pass with warnings denied. Pinned
  Rustfmt, configured Ruff checks/formatting, repository policy and diff checks
  pass. Host power preferences match the initial snapshot.

The benchmark scripts receive a later formatting-only follow-up. Their Python
ASTs match the measured versions at `495f2ea3`; native execution sources and
the measured worker are unchanged. The aggregate source inventory above refers
to the measured revision, including its original harness formatting.

## Evidence identities

| Artifact | SHA-256 |
| --- | --- |
| Release generation worker | `ad065cc47e9bdee9aedeaf155ab34ebe4e33c19faa28769d9ab1acacb443cc87` |
| 734-file build source inventory | `1e948f2e80dffd635f47586f639b728703ec00049366f41f16f9bd2373ab6dba` |
| Strict verifier aggregate control | `53760ec81365890781ddb37d44944f8c83130a5f639c5591cf9cc6b84dd18b98` |
| Final-source regression and strict replay record | `b8f07a2d1210fde241ea91337020df9d2d7b573929448f48daa22d9bfb182be5` |
| Real-worker lifecycle aggregate control | `c2cd32f6bf13cefddf08a3994d334dd64a9e45dcfb62b74266b54953d93e1ae8` |
| Rejected batched source snapshot | `7f7b2d426a925a8d1eefc901c82fdd5a1d9f28873c6afb48eb9bb46d3f867be7` |
| Rejected batched aggregate metrics | `b3f97e1dbde51866d0601e75ec311e995277a6390320e030ccbba636bfc81084` |
| Checkpoint configuration | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Checkpoint source receipt | `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30` |
| Public code corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| 4K input prefix | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |
| Top-k 20, one draft paired report | `98f258d72841e7d33bc254b3fa0d572419c53d845fe097b7622fde8386b364d1` |
| Top-k 20, two drafts paired report | `e3e47e130567fe5d3b495bc0cbe48be47716a6af3da0d7222da3dfd70f2de624` |
| Top-k 20, three drafts paired report | `022db1678641ef65ab39c748451e504f07abb2fb97344d3ccb6ac3129d316946` |
| Nucleus, one draft paired report | `0a82cc253c3258927e8c52b95e83cb72327fbd9f96bc857b33fe3dd0c4fed235` |

Private aggregate reports are mode 0600 under an ignored mode-0700 directory.
No native prompt, output text, token vector, logits or probabilities are
persisted. Earlier non-speculative results remain source-bound historical
records; this study does not supersede their numerical qualification.
