# Flash-Next expert gate/up and shared-down experiment

Date: 2026-11-07. Scope: the native Lily-format Qwen3.8-Flash-Next Q4 XL
artifact, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`.
Engine baseline: `72b5aaef96c939809bc906d10d45a0eef70386c0`.

[Aggregate receipts](data/flashnext-moe-projections-2026-11-07.json) and
[applyable source patches](../archive/experiments/2026-11-07-flashnext-moe/README.md)
bind the variants and measurements separately.

Status: the final candidate completed an observed ABBA comparison at **81.23
decode tok/s with two MTP drafts**, versus **85.50 tok/s for baseline**. With
MTP off it measured **75.63 versus 80.33 tok/s**. Both candidates pass the
bounded numerical controls. The completed run has no quiet admission and four
host-ineligible arms, so no whole-model speedup is qualified. The
fixed-gate/shared-first implementation remains experimental.

## Hypothesis

The [optimization pilot](2026-11-07-flashnext-optimization-pilot.md) identified
expert gate/up plus shared down as 16.8% of the instrumented GPU command span.
The existing gate/up shader already fuses both projections and SwiGLU, and its
dispatch overlaps the independent shared-down projection. Earlier independent
row-packing trials had no demonstrated benefit.

The first candidate specializes integer addressing for the checkpoint's actual
geometry: Q4/group64 gate/up, hidden width 2,560 and expert width 640; and
Q8/group64 shared down, 640 inputs and 2,560 outputs. Constant dimensions allow
the Metal compiler to simplify group division and loop bounds. The gate/up
entry points share one arithmetic body. The shared-down entry point uses the
ordinary Q8 arithmetic macro, including its lane reduction and BF16 output
boundary. The experimental selectors have exact shape guards and retain the
generic fallback. The official mixed-affine numerical policy is excluded from
the shared-down selector.

The second candidate retains the fixed gate/up kernel, restores the ordinary
Q8 shared-down kernel, and queues shared down before expert gate/up within
their existing concurrent dependency level. Both inputs are already ready;
the existing barrier remains after both outputs, and no additional barrier is
introduced. The separately qualified official path retains its original order.
The synthetic correctness control follows this candidate order as well.

Both candidates preserve weight codes and coefficients, BF16 activations,
F32 reductions and recurrent state, Q8 attention cache, and the existing
sampling algorithm. They add no retained state. Existing prefix snapshots and
cache keys remain applicable because the change affects local projection
addressing, with no new request option or persistent buffer. Qualification is
specific to this artifact and build.

The ordinary target decode path uses these projections. MTP target
verification uses its separate row kernels, so ordinary-decode component gains
cannot be transferred to the batched verifier. The matched workload measures
MTP off as the primary route and two drafts as a regression control.

## Source and artifact binding

Both workers use Rust 1.98.1 and locked release builds in distinct Cargo target
directories. The baseline is built from a detached checkout of the named base.
The candidate is identified by its source patch and executable SHA-256. The
full checkpoint hash receipt from the pilot verified all 55 weight shards,
106,905,155,192 bytes, against the pinned upstream manifest. This experiment
revalidates that receipt's file size, nanosecond mtime and inode guards before
and after the benchmark; it does not claim a new payload rehash.

The machine is an Apple M5 Max (`Mac17,7`), 128 GiB, arm64, macOS 26.7.2.
The baseline worker SHA-256 is
`ec755e73a1d27b7d3ee13b36cf3673f8cd21fb57ef9a491b0cfc9dcfb2abe6a3`.
The final candidate worker SHA-256 is
`58c00756c8fc2de7cca4bf72987a8f1f094a96710e20ffcbe8a0cc53434ecd87`;
its implementation patch SHA-256 is
`34640e336d7ad228fc0368d5790f1df726af7ccb0ba3651ef39d22394b601c19`.
The final test-only route supplement has patch SHA-256
`a897e69e365721bae1bc56e9975b53d3bfb29d968ca125439b5cac215f27c723`.
Its production-source patch sections are byte-identical to candidate 2, and
a generator rebuild retains the same executable hash. Original receipts
remain bound to their original test sources.
The source corpus is `vendor/lily-core/src/qwen4exp/model.rs`, SHA-256
`33dcb30e762d63db4f16aaef21034fbaab93dd56dfd9b555e6da38f5db3238e3`.

Raw inputs, outputs, token vectors and probabilities remain in memory. Native
test receipts stay under ignored `runtime/flashnext-moe-2026-11-07/`; public
records contain aggregate results and hashes only. Historical validation
records remain unchanged.

## Numerical controls

The original synthetic native control compares both projections at the actual
dimensions, with permuted expert routes, offset views and output canaries.
Its 12 cases include signed zeros, subnormal BF16 inputs and
cancellation patterns. All **99,840 BF16 values** match the generic reference
byte-for-byte, with identical SHA-256
`a2eb851dd775700f3e68b3b3849c1fcc822088d56dac3accdd667de34e95ea29`.
The complete ordinary-versus-row MoE controls also pass, including the
separate official-policy control.

Final review identified that the original fixture's route permutation did
not contain duplicates. A separate test-only supplement adds a repeated-route
case for each input style. The expanded 16-case run compares **135,680 BF16
values** byte-for-byte with matching SHA-256
`463aae4ad8d05e3d7abbcc969c8aca85e33861451f83fbd9db0c067437729cb1`.
This supplement changes no production source or timing fixture. The earlier
12-case receipts remain unchanged.

The checkpoint-bound serial/verify/rollback audit compares 35,758,080
vocabulary values. Baseline and both candidates produce identical aggregate audit
results. Verification, rollback and host-serial phases have zero unequal
values. The existing draft-head seam has 248,320 unequal values, maximum
absolute error 0.0533607 and no top-1 mismatches, identically across the builds;
this experiment does not repair or requalify that separate head policy.

The checkpoint-bound prefix capture/restore control passes for both candidates, comparing full
readouts after a captured and restored recurrent prefix with fresh execution.
The final candidate's native engine suite passes 817 tests, with 91 ignored;
nine integration checks and three documentation checks also pass. The final
suite log SHA-256 is
`3879eae333a504d19754e3fb806e58089431e0f3244ab5236fd75fcb80ece9a4`.
These bounded controls do
not establish general answer quality or equivalence to another runtime.

The final candidate also passes an independent scoring replay using separate
Rust 1.98.1 `optimized` builds of baseline and candidate `stark-score-digest`.
Four source-derived input lengths (1, 4, 64 and 512 tokens) cover the single-row
decode and prefill paths. Both repetitions per input use a fresh-state scorer
with caching disabled. The complete-vocabulary digests match for all
**993,280 compared F32 values**; repeated requests also match. Requests and
replies remain in private pipes, with only digests and counts retained.
The ordinary score worker's shared implementation passes its 25 model-free
tests in each of its two binary targets, plus the separate probe test.

## Component measurements

Sixteen independent synthetic weight banks rotate through the actual
projection dimensions. Each command encodes one layer from every bank. Four
warmups per arm precede 16 alternating ABBA-ordered pairs. Timings are GPU
command duration divided by 16 layers. They exclude command encoding and
worker transport and are component evidence only.

| Scope | Reference median microseconds/layer | Candidate median microseconds/layer | Median paired speedup | Pair wins |
| --- | ---: | ---: | ---: | ---: |
| Expert gate/up | 44.82 | 43.45 | 1.0321x | 15/16 |
| Gate/up plus shared down | 37.53 | 37.19 | 1.0074x | 15/16 |

These two scopes use separate windows; their absolute times cannot be
subtracted to estimate shared-down cost. Neither window establishes a
whole-model speedup or a quiet-screened production rate.

A second component window, after all builds stopped, measured the
fixed-gate/up plus shared-first candidate:

| Scope | Reference median microseconds/layer | Candidate median microseconds/layer | Median paired speedup | Pair wins |
| --- | ---: | ---: | ---: | ---: |
| Expert gate/up | 45.64 | 43.89 | 1.0231x | 15/16 |
| Gate/up plus ordinary shared down queued first | 39.39 | 38.42 | 1.0221x | 16/16 |

An earlier development timing while a build was active is retained privately
and excluded from these tables. The final synthetic ordering control still
matches all 99,840 values and the same output hash.

## Matched whole-model protocol

The existing pilot harness runs baseline/candidate/candidate/baseline sessions,
with one warmup per MTP route and four paired blocks per session. Both builds
receive identical corpus-derived 512-token inputs, 256 sampled outputs,
context 768, seed 42, temperature 1, top-k 20 and top-p 0.95. In the initial
per-request protocol each arm requires
the existing ten-second quiet-admission streak and a passing request-time host
screen. Measurements use uninstrumented workers. No observed-only substitute
is used for the requested quiet-window comparison.

The predeclared acceptance rule requires passing numerical/prefix controls,
matching same-route output hashes, eligible timing arms and a repeatable
end-to-end benefit relative to control drift. A component gain alone is
insufficient to promote the candidate. Incomplete or ineligible attempts
remain separate receipts.

Repeated per-request quiet-admission attempts were interrupted by compiler
activity, another model process, or other host load. The first candidate's
completed ABBA run had 31 of 32 eligible arms; its first arm exceeded the CPU
limit at 154.8% versus 150%. It is retained as an ineligible complete run.
Its pooled decode rates were 82.61 to 82.92 tok/s with MTP off and 95.37 to
95.54 with two drafts. The small differences are not a qualified benefit.

In these initial screened attempts, the second candidate did not complete a
candidate session before admission
timed out. Completed baseline cells and partial requests remain in their
original receipts and are not used to claim a candidate gain.
The five attempts retain 47 measured arms in total; available same-route
generation hashes agree across baseline and both candidates.

The harness was then extended with explicit `--quiet-scope session`. Its
default remains `request`. Session mode requires the same ten-second quiet
streak before the first measured request of every loaded worker, then screens
every measured arm using the same CPU, compiler, model-activity, thermal,
sampling-gap and paging rules. It measures continuous paired work and is a
different timing protocol from the per-request-admission attempts. It never
disables request-time screens. All 62 paired/pilot/context harness tests pass,
including admission-count and per-arm-screen coverage for both modes.

The session-level attempt also timed out before its first measurement. An
idle diagnostic then confirmed `IOConsoleLocked: true` and
`CGSSessionScreenIsLocked: true`. Device GPU utilization ranged from 0% to
85%, with the largest observed Metal-client counter increase attributed to
`loginwindow`. Those per-client counter units are unverified; they identify
the active client without supplying a utilization percentage. The earlier
compiler/model observations are recorded separately and are not attributed
to the lock screen.

A temporary `caffeinate -d -i -u -t 1800` assertion was started to keep the
display/system awake during the requested window. It does not unlock the
console or change persistent lock settings. Manual unlock was requested before
another quiet-admitted attempt. No observed-only run or relaxed CPU/GPU threshold
has been substituted for the requested quiet comparison.
The temporary assertion was released before handoff.

### Completed observed whole-model run

A separate `--observe-host` run completed all four ABBA sessions and all 32
measured arms on the final candidate. It uses the same pinned workers,
checkpoint, 512-token source input, 256 sampled outputs, context 768, seed 42,
temperature 1, top-k 20, top-p 0.95 and four blocks per loaded worker. A
`caffeinate -d -i` assertion was owned by the benchmark process and ended
with it. The console was still locked before this run. No quiet admission
was attempted, and the host-screen thresholds were unchanged. This is a
separate observed protocol from the initial screened attempts.

The [complete aggregate receipt](data/flashnext-moe-matched-observed-2026-11-07.json)
retains every measured arm, worker and harness hashes, artifact guards,
sampling configuration, output hashes and host screens. Rates pool all eight
requests per worker and MTP route as total decode steps divided by total
decode seconds. Wall times are medians across those same eight requests.

| Route | Baseline decode tok/s | Final candidate decode tok/s | Candidate rate change | Baseline median wall seconds | Candidate median wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP off | 80.33 | 75.63 | -5.86% | 3.5000 | 3.6358 |
| MTP two drafts | 85.50 | 81.23 | -4.99% | 3.4038 | 3.4021 |

| Session | Worker | MTP off decode tok/s | Two-draft decode tok/s | Host-eligible arms |
| --- | --- | ---: | ---: | ---: |
| 0 | Baseline | 80.82 | 86.75 | 7/8 |
| 1 | Final candidate | 74.02 | 80.98 | 8/8 |
| 2 | Final candidate | 77.31 | 81.49 | 6/8 |
| 3 | Baseline | 79.86 | 84.28 | 7/8 |

All same-route output hashes match across workers and repeats. The artifact
and worker identity guards pass. No active compiler or external model CPU
activity was observed, and thermal state remained nominal or fair. Four arms
exceeded the 150% external CPU threshold, with a maximum of 222.73%; one also
had two pageouts. Thus 28/32 arms pass their individual screens, and the run
retains `timing_eligible: false`. No arm was removed or retried.

The final candidate's observed decode rate is lower than baseline in both
routes. Two-draft median complete-request wall time is essentially unchanged;
MTP-off wall time is 3.88% higher. These measurements provide no evidence for
promoting the candidate. The earlier roughly 95 tok/s measurements belong to
the separate first-candidate run under its recorded machine conditions.

## Disposition

The combined component improved by about 2.2% in the second synthetic window.
The completed observed whole-model run measured 81.23 tok/s with two drafts
and 75.63 tok/s with MTP off, both below its matched baseline. No whole-model
benefit is demonstrated. The code, candidate patches, numerical evidence and
all benchmark receipts are retained for review. Performance promotion is
unsupported by the available results; a quiet-qualified comparison remains
outstanding.

## Reproduction

Apply exactly one archived implementation patch to the named baseline in a
separate checkout. Build each worker with its own absolute `CARGO_TARGET_DIR`
using `cargo +1.98.1 build --release --locked -p stark-generate`. Build scoring
workers separately with `-p stark-score --features optimized --bin stark-score
--bin stark-score-digest`. Do not build or run other GPU work during timing.

With `MODEL`, `BASELINE_WORKER`, `CANDIDATE_WORKER` and `WEIGHTS_RECEIPT` set
to the pinned local artifact, sealed workers and guarded prior hash receipt,
the new session-admission protocol is:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_pilot.py \
  --model "$MODEL" --worker "$BASELINE_WORKER" --candidate-worker "$CANDIDATE_WORKER" \
  --corpus vendor/lily-core/src/qwen4exp/model.rs --weights-receipt "$WEIGHTS_RECEIPT" \
  --input-lengths 512 --draft-depths 2 --tokens 256 --blocks 4 \
  --screened --quiet-scope session \
  --output runtime/flashnext-moe-repeat/summary.json --cells runtime/flashnext-moe-repeat/cells
```

Use a new output directory for each attempt. Omitting `--quiet-scope session`
reproduces the default per-request-admission protocol. Both require matching
same-route hashes and every measured arm to pass its host screen. The
completed observed run replaces `--screened --quiet-scope session` with
`--observe-host`, retaining all arms and explicitly unqualified timing. The
synthetic tests are ignored native tests under `stark_lily_core`:
`flashnext_gate_up_shared_down_match_reference_bytes` and
`flashnext_gate_up_shared_down_streaming_timing`. Run those separately from
whole-model timing with `--ignored --nocapture --test-threads=1`. The
checkpoint audit and prefix test names and their aggregate receipts are in
the public JSON. Model inputs, responses and readouts must remain private.
