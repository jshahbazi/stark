# Native GLM and Qwen execution optimization

Date: 2026-10-28. Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.1.
Baseline source: `959c57664a3e569fb17676a014ab89868b11e117`.
Candidate source: the commit introducing this report.

This study covers GLM-5.3-Flash UD-Q2_K_XL and Qwen3.8-Flash-Next mixed 4/8-bit,
using the exact [pinned checkpoints](../NATIVE_HYBRID_MODELS.md). It supersedes
the short throughput observations in the
[initial full-checkpoint smoke record](../archive/runs/2026-10-28-native-hybrid-full-checkpoints.md).
Those source-bound correctness results remain recorded against their original
builds. This study compares Stark builds; it does not compare other runtimes or
measure answer quality.

## Results

Median warm rates, tokens/second:

| Model | Input / sampled output | Prefill baseline → final | Decode baseline → final | Decode speedup |
| --- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash UD-Q2_K_XL | 14 / 32 | 0.422 → 2.468 | 0.470 → 2.485 | 5.282× |
| Qwen3.8-Flash-Next mixed 4/8-bit | 54 / 31 | 0.757 → 4.451 | 0.672 → 4.318 | 6.423× |

Warm decode ranges across the two measured warm replays are 0.453–0.488 versus
2.281–2.688 for GLM, and 0.667–0.677 versus 4.281–4.355 for Qwen. Prefill
speedups are 5.845× and 5.877×. Both final workloads reproduce all three baseline
output hashes exactly, with identical input hashes, configuration hashes, sample
counts and finish reasons.

First requests after loading take 148.72 → 27.65 seconds for GLM and
137.32 → 38.35 seconds for Qwen. These are observed first-request times, including
lazy pipeline/resource setup; operating-system file caches were not forcibly
cleared. They are not controlled cold-disk latency comparisons.

## Measurement method

The original release workers were preserved before changing code. Each model
runs alone, with three greedy requests to the same loaded worker, at most 32
output tokens and a 512-token context budget. Every request starts with fresh
recurrent and attention state. The first replay is reported separately; warm
rates are the median of the second and third replays. This is operating-system
and resource warming, without a prompt or response cache.

Rates use the worker's `first-token-wall-clock-v1` method: prefill ends at the
first sampled token, and decode throughput counts subsequent sampled tokens
over the decode interval. Sample counts include a sampled EOS token. GLM samples
32 tokens at the length limit; Qwen samples 31 and stops at EOS. Their rendered
prompts contain 14 and 54 input tokens respectively. Comparisons use each
model's identical input and output counts.

The harness is [benchmark_hybrid.py](../../scripts/benchmark_hybrid.py). It uses
the checkpoint's template with a fixed synthetic input and enabled thinking.
It records counts, times and hashes. Inputs, responses, token vectors and scores
are not published. Raw aggregate reports and profile captures remain ignored.
Model files were previously verified against pinned LFS SHA-256 values and are
unchanged during this study. Full model runs and GPU tests are serialized.

Opt-in [hybrid_profile](../../crates/stark-engine/examples/hybrid_profile.rs)
separates uploads, command execution, GPU duration, readback, expert gathering
and architecture phases. Its short phase profiles use two input tokens and
four decode steps, with the second replay shown below. Nested architecture
scopes are inclusive and must not be added together. Five-second macOS sampling
intervals also observe each baseline worker; this sampling adds a small amount
of profiling overhead to those intervals. CPU fixture checks and setup builds
may overlap baseline setup; these are development measurements, not hardware
peak-throughput claims. No thermal or performance warning was recorded by
`pmset -g therm` during setup.

## Profile and changes

| Baseline phase, seconds per decode token | GLM | Qwen |
| --- | ---: | ---: |
| Wall time | 2.014 | 1.051 |
| Compressed uploads | 0.559 | 0.363 |
| Command execution including waits | 1.295 | 0.606 |
| GPU duration within command execution | 1.012 | 0.390 |
| Selected-expert gathering | 0.060 | 0.021 |
| CPU recurrence | 0.025 | 0.020 |
| Compressed bytes uploaded per token | 9.22 GB | 6.02 GB |
| Command submissions per token | 810 | 917 |

Independent process sampling finds 61.3% waits and 31.1% memory copies for GLM;
Qwen shows 58.6% waits and 34.1% copies. Recurrence fusion was measured on CPU
and omitted because its small improvement would not address the dominant cost.

The packed shader previously divided a 64-bit element index by a runtime block
or group size for each weight. All supported block/group sizes are powers of
two. Equivalent shifts and masks preserve addresses and floating reductions.
This isolated change reduced four-step GLM GPU duration from 4.048 to 0.974
seconds, and Qwen from 1.562 to 0.749 seconds, with identical output hashes.
Wall improvements were smaller because copies and command waits remained.

Mapped Metal views retain an owned file mapping through the buffer's deallocator.
Selected experts use only their checked compressed ranges, with bounded active
views, rather than registering an entire expert tensor. Independent projections
share a command submission while retaining their individual kernels and reduction
order. Prefill advances intermediate tokens without projecting discarded
vocabulary logits; the final input token still performs the full readout.

The first full worker runs after these changes improved decode by 3.7 times for
GLM and 5.1 times for Qwen, with identical output hashes. A subsequent residency
experiment avoids registering static weight views again on every call. Its
model-local FIFO retains at most 8 GiB of page-rounded immutable weight views;
selected experts remain transient. It stores no inputs, recurrent state, scores
or completions. Live operations retain their own views through completion, and
model unload releases the cache.

In adjacent short profiles this cache raises GLM's warm median from 2.528 to
3.274 tokens/second and Qwen from 3.599 to 4.708. Retained registrations peak at
6,447,644,672 and 4,697,325,568 bytes respectively, within the bound, without
evictions on these workloads. Independent ownership review and physical tests
cover page-rounded accounting, eviction with live views, over-budget fallback,
expert bypass, distinct inputs, and release after dropping the store. The final
worker comparison below includes the retained-view cache.

Kernel specialization fixes storage kind, bit width, group size and scale dtype
at pipeline creation, so each kernel sheds unrelated decoder paths. The cache
contains bounded shader layouts and schedules, without model data or decisions.
The original F32 math mode and per-row reduction order remain unchanged.

A four-row scheduling experiment improves synthetic IQ2_XS and four-bit affine
component cases at 2,048 by 4,096. It regresses floating, Q8 and 8-bit affine
components, and four-bit affine regresses about 5% at Qwen's actual 640 by 2,560
and 2,560 by 640 expert shapes. Selecting it only for large IQ2_XS also fails
the full GLM check: adjacent three-replay profiles measure a warm median of
2.312 tokens/second with tiling versus 2.589 with the original row schedule.
The experiment is excluded from production and retained only for test-side
reproduction. Synthetic hot codebook data did not predict real-weight behavior.

A bounded 200-microsecond completion-polling experiment also fails its gate.
Adjacent short-profile warm medians are 2.528 versus 2.399 tokens/second for
GLM, and 3.599 versus 3.495 for Qwen (blocking wait versus polling). User CPU
time for each complete three-replay profiling process rises from 1.51 to 3.18
seconds for GLM and from 1.01 to 2.92 for Qwen. These CPU totals include startup
and the first replay; cold file-cache differences make total process wall time
unsuitable for the throughput comparison. Production keeps blocking completion
waits. The rejected [polling patch](../archive/experiments/native-hybrid-2026-10-28/poll200.patch)
and test-only row-tile source preserve the experiments without selecting them
for native execution.

## Upstream code examined

The existing Uzu-derived GEMV implementation reuses packed-word and activation
loads across output rows in `gemv/common/quant_slice.h`. Its command layer also
reuses allocators and scratch buffers, and its dense decoder encodes longer
operation chains. Local ancestry is Uzu 0.6.1,
`8d01581474760bd1fc7f4a547ff1d02f7dada437`, MIT, plus the recorded Stark patches.
The hybrid row-tile experiment uses that scheduling idea while retaining its
existing per-row arithmetic and GGML decoder helpers.

MLX-Serve at `e1d2013c1a321ba9e04324d790254e3f5428265c` indexes resident expert
banks and fuses expert gate/up/activation and down/weighted reduction. Its
[Qwen expert path](https://github.com/ddalcu/mlx-serve/blob/e1d2013c1a321ba9e04324d790254e3f5428265c/src/plugins/qwen4_exp/trunk/moe.zig#L780)
and affine kernel geometry are useful references for reducing further readback
boundaries. Its [fusion correctness note](https://github.com/ddalcu/mlx-serve/blob/e1d2013c1a321ba9e04324d790254e3f5428265c/src/engines/mlx/compiled.zig#L28)
documents greedy divergence after changed intermediate rounding. Its BF16 casts
and sigmoid tables are not imported into Stark's F32 path. This project's own
code is MIT licensed; no external serving implementation is copied or launched.

MLX-LM at `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`, MIT, reserves vocabulary
readout for the final prompt token in its
[generation loop](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/generate.py#L442).
Its chunked prefill and GPU-resident GDN demonstrate larger graph work that is
separate from this sequential-prefill implementation.

## Reproduction

Build the generation worker separately from the optimized scorer, as required by
[Contributing](../../CONTRIBUTING.md). Run each model separately:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/glm --worker target/release/stark-generate --output runtime/native-hybrid/glm-benchmark.json --tokens 32 --replays 3
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/qwen38 --worker target/release/stark-generate --output runtime/native-hybrid/qwen-benchmark.json --tokens 32 --replays 3
cargo +1.98.1 run --release --locked -p stark-engine --no-default-features --features cpu,metal --example hybrid_profile -- runtime/native-hybrid/glm 4 2
```

These are short-context, single-request text measurements. Long-context memory,
concurrent application pressure, other checkpoints and installed-app throughput
require separate qualification. The F32 execution policy and CPU recurrence are
unchanged.

## Correctness and build validation

| Check | Result |
| --- | --- |
| Complete physical CPU/Metal engine suite | 609 passed, 35 ignored; 7 CXX tests and 3 documentation tests passed |
| Source-bound GLM reference, pool 2 and pool 4 | 94 checked positions per profile; maximum errors `3.30210e-5` / `3.38554e-5`, unchanged from the previous implementation |
| GLM numerical criterion | Existing absolute and relative-RMS gate `1e-4`; no gate changed for this optimization |
| Source-bound Qwen mixed/BF16 and four-stream profiles | 13/13 greedy agreement each; maximum errors `2.98023e-7` / `2.38419e-7` |
| Prefix ingestion | Exact full-step readouts, position accounting, context rejection, reset and failure recovery |
| Final worker fixture replay, both families | Four greedy source checks each; EOS union, tokenization/decode, A/B/A isolation, consecutive score-cache policy and failure recovery passed |
| Full production checkpoint smoke, both families | Generation, exact replay, recovery, finite scoring and normalized candidate scores passed; eight-token completion hashes match the original source checkpoint |
| Release generation / optimized scoring unit suites | 43 / 19 passed; scoring digest repeats 19 and its experiment test passes |
| Release worker Clippy with warnings denied | Both passed |
| Pinned Rust format, benchmark-helper Ruff, repository policy and whitespace | Passed |

The full benchmark pass runs after builds, unit tests and source-bound reference
checks have finished. Original weights remain unchanged; production smoke reuses
their previously verified LFS manifests with size/mtime checks. No reference
runtime executes the native requests. The small-model references and full-model
hash replays do not establish general answer quality or BF16-runtime parity.
Process RSS excludes substantial GPU and operating-system file-cache memory and
must not be interpreted as total model memory. The cache budget measures retained
Metal registrations, with separate live-operation bounds.

## Artifact identities

All binaries, model files and raw validation artifacts remain ignored. SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Baseline generation worker | `10b0956567edfdc8a6074c6161a4745e3f280f330aabf9c0dac461ce45e36b82` |
| Final generation worker | `eb1640f078f4b8871a94ffbc8e7a95224b83da2f9c1711ead105c108fcee386d` |
| Final optimized scoring worker | `4e9cc8ed8f5ccf5a1612e50355fd32c74ef7fec06d6f0ab13c160c3a282f3ad8` |
| Benchmark harness | `a55be8c1dbb896ee53c4ec6408a653394cd564536c55847e07f0ffa713825c56` |
| GLM baseline aggregate report | `1aa8ed9e6eac0a71d581ce72dee776239b624872561c0ec298f7390f4ae1ae31` |
| GLM final aggregate report | `f0f1ecd2ac26adab553bb4cf1aceae6e699eab6d8ad3acbdb0ba4fdfb9cf6a07` |
| Qwen baseline aggregate report | `a05f601349a85c1f0b7dd149392a4da8e4c788b196132ba10a4c3499d4df92bc` |
| Qwen final aggregate report | `f5abe84bfc76b17b9f0066206eabda03af3c14821b4072da19d33e43dfc649f9` |
| Engine test log | `b89c8cd9404082958bc85b93f1c45fc1f1d3a809718c2d92e6eb048907a6caa9` |
| GLM final production smoke | `65e961c08c949cc7be4fda77795adfefe0e12371b02c0c8148f092d2c3fae48e` |
| Qwen final production smoke | `d92ecf74d7e92e30fdd3e272bd05098095b194a4638eb04223f6a86d7e14fa81` |

Baseline and final GLM output hash:
`5377eeb8513c16603090026f7a432327b456842a96d54565044e24bdb5986e1c`.
Baseline and final Qwen output hash:
`aca3d65e215fc3c055839f338dd8c040531a677970415c0d410fe5c7391f6c7b`.
