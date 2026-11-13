# 20261113-EDGE0-17: measured-arm task attribution and CPU audit

## Identity and state

| Field | Value |
| --- | --- |
| Attempt / registered and executed date / owner | `20261113-EDGE0-17` / 2026-11-13 / Codex |
| Kind / scope | Memory and host-observation diagnostics; local non-commercial evaluation only |
| State / result | Attempted; initial parent resource-deferred without children; separately authorized direct run completed 12 matching warmups and eight measured arms, then five groups deferred |
| Evidence scope | Model-free diagnostic checks, actual-artifact output agreement and partial end-to-end timing; no complete eligible timing group |
| Promotion / disposition | Diagnostic implemented in private qualification harness; performance qualification deferred; MTP default off |
| Source | `8704f04c969de2c7e03345829be2f5956c93a78d`; unchanged inference binary and precision policy |
| Predecessor | [Attempt 16](2026-11-13-edge0-quiet-window.md) |
| Governing plan / related evidence | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp); [benchmark record](../benchmarks/2026-11-13-edge0-arm-attribution.md) |

## Hypothesis and change

Attempt 16 completed 40 measured arms with 26 decompression-positive arms and
13 CPU-positive arms, including 12 with both conditions. Its zero worker-task
decompressions did not identify the owners of global activity. This follow-up
adds fixed task-inventory endpoints around each measured arm and reconstructs
CPU rejection contributors from retained scrubbed host rows. Neither diagnostic
subtracts counters, changes thresholds or establishes complete global ownership.

The new `qualify_worker_v7.py` uses `arm_diagnostics.py`. Each measured arm gets
one before endpoint after the existing worker-memory observation, followed by
the existing first-arm admission-freshness check. The after endpoint runs after
the unchanged host screen has been stored, under an explicit `arm-diagnostics`
phase. This placement prevents the added after-arm census from extending that
arm's stored host screen. Each task census has its own global-counter brackets;
the wider arm comparison includes both endpoints, native work and intervening
observation/screening work. Full endpoint warmup precedes the host Monitor.

Endpoint CPU uses `os.times`: the whole Python parent, including concurrent
observer threads, plus reaped children. It excludes running-worker CPU. Endpoint
totals are components of the wider totals and must not be added to them again.
This accounting does not isolate census overhead or attribute external service
activity caused by observation.

## Artifact, source and environment

The artifact remains
`Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`.
Native preparation receipt SHA-256 is
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`;
Rust 1.98.1 worker SHA-256 is
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
Existing complete payload hashes retain their explicit immutability assumption;
this attempt performs no new large-payload rehash or build.

Retain original source weight formats, virtual Metal, F32 activation,
accumulation, recurrent-state and cache policy, selected-record bindings
(`STARK_EDGE0_EXPERT_WINDOWS=1`), context 512 and unavailable prefix reuse.
No new state-affecting inference path is introduced. The predecessor's concrete
prefix-restoration blocker and image/text scope remain unchanged.

The registered source manifest SHA-256 is
`72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47`;
private v7 harness SHA-256 is
`040564ee6163a42395d68d7c5d9daabe02c2baa823fd7b3e2c2a56a409e8f540`;
arm helper SHA-256 is
`ad564d6b23a69552a369ad915cfbaaf7b582a2ad2341fcebb238e27321fe5558`.
These identify the qualification recipe separately from the unchanged worker.
The previously recorded host remains the 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332; no fresh hardware/OS identity probe is claimed.

## Workload and controls

Retain the predecessor's matched matrix: input lengths 32/128, context 512,
greedy seed 42, maximum 16 outputs with matching natural EOS, draft depths one
through three, one warmup for each off/on route and four balanced pairs per
group. Intended work is 12 warmups and 48 measured arms. Private corpus SHA-256
remains `1a681566d9fcf584e42df629c610513a786648976d5365687a2e6465b6fc504c`.
Loading and warmups remain outside timing summaries. This run introduces no
new quality, image, stochastic-sampling or lifecycle qualification.

Each group still needs ten fresh contiguous quiet seconds within its strict
120-second admission deadline: all four global paging deltas zero, external
CPU at most 150%, individual-process CPU at most 80%, model CPU at most 10%,
GPU at most 5% and thermal at most fair. Existing two-second freshness and gap
bounds remain. Measured arms retain external CPU at most 150%, model CPU at
most 10%, thermal at most fair, no compiler, monitor failure or excessive gap,
zero compression/decompression/pageout/swapout deltas, request-boundary coverage
and final monitor join. No baseline subtraction or selective screen removal is
permitted.

Retain startup available memory at least 64 GiB, sampled owned RSS at most
48 GiB, a 180-second request limit and 1,800-second worker limit. A depth still
requires complete eligible groups at both input lengths, off-wall spread at
most 15%, median paired wall/decode speedups at least 1.05 and median prefill
on/off at most 1.05. No applications, foreign workers or system settings were
changed as part of this attempt.

## Attempt history and failures

| Trial | Observation | Validity and disposition |
| --- | --- | --- |
| `arm-attribution-1` parent | Initial no-worker quiet preflight deferred after 120.227 seconds; 279 global decompressions, CPU rejection and maximum combined quiet 6.738 seconds; zero children launched | Preserve the deferred parent. No observer control or benchmark ran under this registration. |
| Explicit user steering | After that deferral, the user instructed “just launch it” | A separately sealed direct benchmark registration omits the outer entry preflight and six-treatment observer control. This is explicit subsequent authorization; the failed entry did not become a pass. |
| `arm-attribution-direct-1` | Twelve predecessor-matching warmups; eight measured arms in input-32/depth-one; five later groups deferred | Partial measured evidence; no complete eligible timing group. Every group admission and measured-arm screen remains active. |
| Source review | Initial placement ran the after endpoint before closing the measured host phase | Corrected before registration/execution; wiring test requires stored screen, diagnostic phase, then after endpoint. This was a review finding, not a reported native failure. |

The direct launch does not inherit a passing entry condition or a successful
observer control from the deferred parent. It establishes only its own recorded
per-group admission and measured-screen results. No automatic additional run
or relaxed threshold follows from these outcomes.

## Correctness and lifecycle

Prelaunch validation passed 117 model-free checks covering diagnostic brackets,
privacy, fixed error codes, scope, registration and retained admission/screen
behavior. These checks qualify harness behavior separately from model numerical
or performance evidence. No inference source or binary changed.

All 12 actual-artifact warmups matched predecessor output hashes, emitted and
sampled counts, and stop conditions. The eight measured arms retained paired
agreement, useful-work checks, real speculative activity and target-verification
checks. Input 32 emits six tokens and samples seven with natural EOS; input 128
uses the retained 16-token length stop. This is bounded greedy agreement;
prior lifecycle and numerical evidence retain their original scope.

All 600 worker memory snapshots recorded zero worker-task decompressions.
The final host monitor joined after 1,313 samples with zero failures. Worker
teardown used stdin EOF, exit code zero and successful reap. No new restart,
cancellation, larger-context or prefix-restoration test is claimed.

## Performance and memory

The first group admitted and completed eight measured arms. Three individual
arms passed their screens; five failed solely for native-bracket global
decompressions, totaling 36 pages. All eight measured CPU maxima were below
150%; the largest was 82.017%. Independent reconstruction uses each unique
phase label and saved `covered_end`, excluding any subsequently retained
crossing-phase row absent when the screen was computed. All sample counts,
maxima and eligibility decisions reconstruct exactly.

Five later groups deferred at admission and produced no measured arms. Their
admission windows contain 4,364 global decompressions and overlapping CPU/GPU
rejections. Quiet intervals and wider task-inventory intervals have different
boundaries; no event-ownership subtraction is valid.

| Metric / boundary | Observation | Eligibility and conclusion |
| --- | --- | --- |
| Sixteen task endpoints, summed parent/reaped CPU; wall | 1.370 CPU seconds; 1.357 wall seconds | Includes concurrent monitor work; no isolated census-overhead estimate |
| Eight wider arm brackets, parent/reaped CPU | 13.110 CPU seconds | Includes endpoints and screening waits; do not add endpoint CPU again |
| Startup available / sampled peak owned RSS / final available | 95,657,328,640 / 788,774,912 / 91,095,105,536 bytes | Gauges and sampled process RSS; no complete Metal residency or reservation claim |

Named CPU contributions from the predecessor remain censored below 5% per
original process per host interval. Basename collisions combine different
processes; external Python names do not identify the harness. Current measured
arms have zero CPU-rejection intervals. Neither result identifies all global
decompression owners or demonstrates an observation-caused decompression rate.

## Evidence, decision and limits

Private receipts remain under ignored
`runtime/edge0-arm-attribution-2026-11-13/`. Publish only aggregates and hashes:

| Receipt | SHA-256 |
| --- | --- |
| Deferred parent report | `8970bf7654d918fa87dd2e1e88e95d9bed61a931c624871b9b2d82d61cc857a9` |
| Direct benchmark registration | `d7d98095e4a1069a962bdd2c7ae531347c7ba33bc1d711bb1906aa26a13623fb` |
| Direct benchmark report | `bb66d06c123b6e46b52f88bb18fa5f0f29ebcb14ae9b1f0cc80ab014b36a29ac` |
| Previous exact CPU audit | `bae4a7daa526d1d54c7156923e1cd810734206d13c07859ade34a53d0f5d19f0` |
| Current exact CPU audit | `141e393ae6897ff197c7005e9afc4bfb8ad14a1fa2709edcaed2d4825254e9f4` |
| Current task-attribution audit | `c163a247e0b2036066d1784c9b49e551fff9beafef6c06b2b88bd3644405c5c9` |

The endpoint diagnostic and offline rejection reconstruction are implemented
and exercised within their stated scopes. Performance remains unqualified.
Keep selected-record bindings opt-in and MTP default off; Phase 5 remains open.
No release, installation, product admission, broader model support or complete
global activity attribution follows. Further measurement needs a fresh
registration and source-bound run that satisfies the existing full matched matrix.
