# Qwen3.8 batched exact MTP verification

The layer-major verifier passes byte-identical controls against ordinary
target execution on 90 complete readouts. The final build at `6dbda328`
improves sampled top-k 20 generation with two drafts by **12.44%**, from
59.51 to 66.91 tok/s across eight same-loaded pairs. An earlier four-pair
window gains 16.93%. One draft has little benefit and three drafts reduce
throughput on this workload. MTP remains optional and off by default.
The 77–80 tok/s normal-desktop target is still unqualified.

This study supersedes the throughput conclusion for the serial verifier in
[the earlier exact-MTP study](QWEN38_EXACT_MTP_2026_10_30.md). That report's
source-bound serial results and rejected prefill experiment remain historical
evidence. They do not qualify the new batched implementation.

## Checkpoint and exactness contract

The checkpoint is unchanged: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`,
revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, 55 shards totaling
106,905,155,192 bytes. All 55 current stat identities match the prior complete
SHA-256 verification. This study does not rehash all weights. No weights are
converted or requantized; the existing one-layer MTP head and its separately
quantized readout come from the same checkpoint.

Target execution retains BF16 activations, F32 reductions/logits/GDN state,
Q8 attention K/V and resident experts. The unchanged host sampler draws from
the actual filtered draft distribution `q`, accepts with `min(1, p/q)`,
resamples rejections from normalized `max(p-q, 0)`, and draws a target bonus
after a fully accepted group. Temperature, top-k, top-p and min-p have the
same semantics as ordinary sampling. Exact sampling means preservation of
the ordinary target distribution; separate draft randomness can change the
seeded trajectory across on/off routes.

Requests execute serially against one loaded model. Each request creates
fresh logical decode state. Off requests retain the same loaded head weights
as on requests but skip head execution. Draft distributions and request RNG
state never survive a request.

## Why the original batched verifier changed the distribution

The rejected prefill-based verifier used different numerical execution:

- Skinny projections changed quantized affine accumulation order and lane
  reductions. Projecting a slice could also choose a different packed-row
  route from ordinary decode's full stacked projection.
- MoE prefill rounded router/shared-gate values and expert combination at
  different BF16 boundaries.
- GDN used differently rounded normalized inputs and recurrent reductions.
- Dense attention staged Q8 K/V through BF16 and used a different attention
  implementation.
- Small-row HC mixing could differ by one BF16 unit under compiler scheduling.

Rejection sampling requires the actual target probabilities. Treating these
different readouts as ordinary target probabilities changes the distribution.
The new verifier therefore preserves each ordinary arithmetic boundary while
sharing scheduling and weight locality across positions.

## Batched execution

Verification executes the pending token and up to three proposals in one
GPU graph, layer by layer. Row-axis projection and MoE kernels reuse ordinary
decode helper bodies and lane reductions. Full stacked projections preserve
their original dispatch choices. Routing/shared-gate values stay F32 and the
expert combination keeps the original BF16 branch boundary.

The final build also reuses ordinary decode's fused MoE down/combine/HC
injection body across rows, removing the separate injection dispatch and its
dependency barrier in each layer. Its isolated GPU timing gains are small
and variable; the measured end-to-end claim is for the complete verifier.

GDN scans one to four positions with each state column held in registers,
retaining original F32 expression order, BF16 raw-output rounding and F32
intermediate captures for rollback. Attention batches normalization, RoPE and
cache scatter while retaining the ordinary causal decode attention path for
each query. HC row-axis wrappers execute the ordinary single-row down and
cached-up bodies. PLE projections retain ordinary arithmetic.

The verifier captures recurrent state, convolution inputs, residuals and
complete F32 target readouts. Accepted prefixes restore matching state and
catch up the MTP head through committed rows. Cleanup performs no sampling.
The strict serial verifier remains a reference. Stage diagnostics are
debug-only and have no release environment or persistence interface.

## Numerical controls

The complete checkpoint control compares **90 full-vocabulary target
readouts**, 22,348,800 F32 values, with **zero unequal bytes**, zero measured
KL/TV and zero changed top predictions. It covers widths 1–4, every accepted
prefix from zero through width minus one, post-rejection continuation,
4,095/4,096-token sparse-attention boundaries, the strict serial verifier,
and iterator drops after zero through four delivered tokens. Before fusion,
the stage diagnostic also matches all 554 tested boundaries for row zero at
widths one and four.

This full control passes independently before fusion at `1586c3f8` and after
fusion at `6dbda328`, with separately sealed test binaries and receipts.

Independent component controls pass:

| Component | Cases | Compared bytes | Unequal bytes |
| --- | ---: | ---: | ---: |
| Q4/Q8 projections and complete MoE | 40 | 10,486,140 | 0 |
| GDN register scan, outputs and captured states | 64 | 312,532,992 | 0 |
| Attention Q/K normalization, RoPE and cache scatter | 128 | 570,828,800 | 0 |
| HC down, injection values, cached up and mix | 128 | 699,360 | 0 |
| Fused MoE down/combine/HC injection | 72 | 4,915,200 | 0 |

These are finite controls over the specified shapes and boundaries. They
support the preserved-arithmetic implementation; they do not exhaust all
possible model histories. The draft head retains the earlier catch-up-shape
differences (maximum measured full-vocabulary TV 0.004025); the sampler uses
the actual `q`, so those differences affect acceptance efficiency.

## Sampled benchmark protocol

The host is Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1 (`25G241`),
AC power, Automatic energy preferences. No desktop apps or host settings are
changed. Desktop GPU activity remains uncontrolled.

Each case loads one release worker and runs both routes with the pinned
public-code 4,096-token prefix, 256 sampled outputs, context 4,352,
temperature 1, top-p 0.95 and seed 42. One warmup per route is excluded,
followed by balanced pairs in randomized ABBA order. Four-pair cases divide
1,020 decode advances per route by summed decode time; the final top-k
two-draft case extends this to eight pairs and 2,040 advances per route.
The prefill draw is excluded. Every request must return all 256 samples,
acknowledge its route
and `verification: batched-decode-exact-v1`, and reproduce its route hash.
Stage profiling is removed before worker launch.

Both routes share loaded weights and sampling settings. Each request resets
logical decode state. Different sampled trajectories can visit different
experts. Each percentage uses its own same-loaded off control; comparisons
between separately loaded windows do not isolate an implementation effect.

## First batched build: `1586c3f8`

| Sampling | Drafts | Off tok/s | On tok/s | Change | Faster pairs | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k 20, top-p 0.95 | 1 | 50.2070 | 50.9761 | +1.53% | 3 / 4 | 428 / 592 |
| Top-k 20, top-p 0.95 | 2 | 45.7265 | 53.4667 | +16.93% | 4 / 4 | 632 / 772 |
| Top-k 20, top-p 0.95 | 3 | 53.1117 | 46.2024 | -13.01% | 0 / 4 | 640 / 1,136 |
| No top-k, top-p 0.95 | 1 | 53.3609 | 52.0839 | -2.39% | 2 / 4 | 428 / 588 |

All 40 requests pass, including eight warmups. Both routes in every case
reproduce the corresponding earlier strict-verifier hash. This checks
end-to-end sampling and accepted-prefix behavior in addition to the complete
target-readout controls. Equal hashes across on/off are not required.

Three drafts verify 1,516 rows for 1,020 emitted advances; two drafts verify
1,160. The larger unused suffix explains why increasing speculation depth
can reduce throughput even with a faster verifier. Accepted/proposed is
utilization over all proposals, including unused suffixes after rejection;
it is not a conditional acceptance probability.

| Sampling / drafts | Draft seconds | Verify seconds | Rollback seconds |
| --- | ---: | ---: | ---: |
| Top-k 20 / 1 | 1.2893 | 17.8877 | 0.5456 |
| Top-k 20 / 2 | 1.4632 | 17.0262 | 0.4410 |
| Top-k 20 / 3 | 2.0054 | 19.4420 | 0.4816 |
| No top-k / 1 | 1.7850 | 15.7362 | 0.6442 |

Phase times include host encoding, GPU waits and readback. Sampling/iterator
work outside these phases remains in decode time. Component timing gains
are separate evidence: row-batched readout projection is 1.71–1.89 times as
fast on its synthetic two-to-four-row control, and register GDN improves its
measured component windows. Neither number is an end-to-end speedup claim.

## Final fused build: `6dbda328`

The preferred two-draft top-k case is repeated with eight measured pairs
after two excluded warmups, 18 requests total and 2,040 advances per route.
All eight pairs favor MTP. Both routes reproduce their earlier strict
verifier hashes. Median request wall time falls from 6.9645 to 6.5022 seconds
(6.64%); this includes prefill and transport, while decode throughput excludes
the initial prefill draw.

| Sampling | Drafts | Pairs | Off tok/s | On tok/s | Change | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k 20, top-p 0.95 | 2 | 8 | 59.5082 | 66.9095 | +12.44% | 8 / 8 |
| No top-k, top-p 0.95 | 2 | 4 | 42.9124 | 43.8071 | +2.09% | 3 / 4 |

The nucleus case has four measured pairs and two excluded warmups, 10
requests total and 1,020 advances per route. Its small pooled gain includes
one slower pair and does not establish a reliable improvement across
workloads. All outputs are reproducible within their route. Its accepted
utilization is 600/832, compared with 1,264/1,544 for the longer top-k case.
Median nucleus request wall time falls from 9.8906 to 9.4543 seconds (4.41%).

All six cases together contain 68 requests, including 12 excluded warmups;
all 56 measured requests return 256 samples. Phase times and hashes are
retained in separate private receipts for each build and sampling case.

The final build's off rate differs materially from earlier windows. These
measurements establish a positive same-loaded MTP effect for the tested
workload in both windows; they do not isolate the small incremental effect
of injection fusion. The 54.55–72.20 tok/s per-request MTP range in the
longer top-k window also falls short of consistent 77–80 tok/s.

The nucleus two-draft case is also replayed separately on the preserved
strict worker: both off and on hashes match the final batched worker.
Together with the initial four cases and final top-k repeat, every sampled
mode/depth measured here reproduces its strict-reference route hash.

## Regression validation

Validation passes on the final native sources at
`6dbda32819426c97251c3461eeefdd928b63f04c`:

- Full physical CPU/Metal engine: 666 tests pass, 47 ignored; seven CXX tests
  and three documentation tests pass. The ignored checkpoint control is run
  separately and passes its 90 complete target comparisons.
- Release generator: 68 tests pass, two private benchmarks ignored. This
  includes the exact rejection sampler's analytic and empirical controls.
- Optimized scorer and probes: 39 test executions pass. Generator and scorer
  are built separately to avoid feature unification.
- Final-source vendor controls: all component rows above pass on one freshly
  built test binary, with the 909-file inventory checked before and after.
- Benchmark harness: 17 Python tests pass. Configured Ruff checking and
  formatting, pinned workspace Rustfmt, repository policy and diff checks pass.
- Real-worker lifecycle: 23 requests pass, including 15 short context-tail
  cases across depths 1–3, greedy on/off hash parity, invalid request recovery,
  early stopping inside a verified group, streaming cancellation, and matching
  fresh requests afterward.
- Generator and optimized-scorer Clippy pass with warnings denied.

All 909 inventoried source files and the release executable still match their
sealed identities after validation. All 55 shard stat identities and host
power preferences remain unchanged. Source inventories include crate/vendor
sources and build inputs; they use the same scope for both candidate builds.
The later documentation commit does not alter measured native execution.

## Evidence identities

The initial implementation is `1586c3f81d52b668fbe8cf9d15e1ba997eb59d79`;
the final implementation is `6dbda32819426c97251c3461eeefdd928b63f04c`.
Both are built with Rust 1.98.1. Engine/component tests use the pinned
`nightly-2026-10-01`; Python tools use uv with locked Python 3.13.13 or the
policy script's pinned 3.12.13 environment.

| Artifact | SHA-256 |
| --- | --- |
| Initial release worker | `f0522a9c1e612bee51599f36c293852d276b1954b6aa5ebbfc615f75c5df75f8` |
| Final release worker | `ec4afa5cdb579b47ae64f1e1c3b8b31eab537ac5b873b68f6dd06e17f1ff0a23` |
| Initial 909-file source inventory | `1c22b3a5ee9dd6f592a4b5028dba54d812876f9fe79b8b25d96914ee5fe2c3fb` |
| Final 909-file source inventory | `fc34f8d6b65e485de9dc4d811f2369435a02d56aeaf754c3ee9fa6339bc1f0ff` |
| Final 90-readout checkpoint control | `8baa98cc7dfe44520981e08805c0b6612b69bcd1dd9557cde8d4a79cee5dd874` |
| Final vendor controls | `4e8b4e8adf88e93031fd681dee2e5e25a4df7ead0dce8a84daff541e51e1a282` |
| CPU regression control | `92618dc7d27163dac942fdce49cb3822b77bd1cd32b061fea8827b52d058bbbe` |
| Full engine regression control | `ea15a93904e20713c2646bacbb30c2bcaa0780cc9c489f190e9d36ef91d0a052` |
| Static validation control | `768e785f0334127ee5ace058bdf5be8bf929bfbb764968581cc4b912cc211efd` |
| Real-worker lifecycle control | `1dea6b32da526f6246fa32a271867018d81a46f948b8ec71260c43c18f3a1ede` |
| Strict nucleus two-draft replay | `7d46da77ca58d1c8d9a35caae4c233763e90c585ee3aa5adc40e85fbe99afab2` |
| Final validation and identity seal | `9ab1ffff72ee31c0de7a70afab1393695ddf21fccbfec4d9e99a5525952e9686` |
| Initial top-k 20, one draft report | `50ec6abcd9d6f6079742ff743680b78ece7a18948d1ca1a3e8e72d0245ea915f` |
| Initial top-k 20, two drafts report | `c72d72e7ba83ba76c363cad5571e6589308bf5ce2bf04303a79741c81a695a0e` |
| Initial top-k 20, three drafts report | `a29b7ba69198c222483f59e6f4688303414a5fbbd87479fc2f60fdfe1e55d391` |
| Initial nucleus, one draft report | `7191e13ef501222073645bf510064e386902a0c43c5c4e8794273ea5455d4bb2` |
| Final top-k 20, two drafts report | `0d083f939440cbed4bcb1f0fa2b83599cb725efcfec9ef1630e3c40c670e3174` |
| Final nucleus, two drafts report | `4b813fd9f68059860328bae23f833cc7f52ebecd213e35c51722aba82e09ff53` |
| Checkpoint configuration | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Checkpoint source receipt | `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30` |
| Public-code corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| 4K input prefix | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |

## Reproduction

Build the generator separately from optimized scoring. Use the pinned
[upstream corpus](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/qwen4exp/model.rs)
to reproduce this study's input:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_mtp.py --model models/qwen38-lily-q4-xl --worker target/release/stark-generate --corpus runtime/pinned-lily-model.rs --input-tokens 4096 --tokens 256 --blocks 4 --drafts 2 --top-k 20 --top-p 0.95 --seed 42 --output runtime/mtp-batched-top20-d2.json
```

Use `--top-k 0` for full-vocabulary nucleus sampling. Private aggregate
receipts remain in an ignored mode-0700 directory with mode-0600 files.
Prompts, generated text, token vectors, logits and decision probabilities
are not persisted.
