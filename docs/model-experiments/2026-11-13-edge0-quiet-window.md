# 20261113-EDGE0-16: quiet entry before observer control and matched MTP

## Identity and scope

| Field | Value |
| --- | --- |
| Owner / authorization | 2026-11-13 / Codex; fresh quiet-window no-worker control followed by matched MTP with unchanged gates |
| Scope | Local non-commercial evaluation only |
| Source | `f28753754c3a58b5ac291edd7f028abb0553301d`; unchanged inference implementation/binary from `7c900c69` |
| Predecessor | [Attempt 15](2026-11-13-edge0-host-interference.md) |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| State / result | First registration deferred before control; separately authorized second registration passed both entry preflights and the control, then completed 12 matching warmups and 40 measured arms with zero complete eligible timing groups; final group deferred |
| Promotion / disposition | No new qualification; record policy opt-in, MTP default off |

Attempt 15 failed its observer control's final no-worker census. Its separate
matched rerun completed 12 matching warmups, six admission deferrals and zero
measured arms. This follow-up requires strict quiet admission and a no-worker
census before entering the control sequence and again before benchmark launch.
It does not change the previous inference binary, observer treatments, measured
matrix or acceptance thresholds. The separate
[benchmark record](../benchmarks/2026-11-13-edge0-quiet-window.md) records the
result and evidence bindings.

## Artifact and precision

Reuse `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`,
native receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`,
and prior complete payload hashes with their explicit immutability assumption.
No new large-payload rehash is claimed. The Rust 1.98.1 worker remains SHA-256
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
The corpus SHA-256 remains
`1a681566d9fcf584e42df629c610513a786648976d5365687a2e6465b6fc504c`.

Retain original source weight formats, virtual Metal, F32 activation,
accumulation, recurrent-state and cache policy, selected-record bindings
(`STARK_EDGE0_EXPERT_WINDOWS=1`), context 512 and unavailable prefix reuse.
The previously recorded host is the 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332; this identity is not a new host measurement.

## New source-bound workflow

The private `run_quiet.py` parent and `quiet_window.py` preflight live under
`runtime/edge0-quiet-qualification-2026-11-13/`. They use the unchanged
`control_residual.py`, `qualify_worker_v6.py`, persistent paging sampler,
strict admission helper and host Monitor from attempt 15. A fresh parent
registration seals both new helpers and tests, previous helper bindings and the
two child registrations before execution. Previous reports remain immutable.

The fixed stage sequence is:

1. Strict no-worker quiet preflight before the control sequence.
2. The unchanged six-treatment observer control.
3. A fresh strict no-worker quiet preflight before benchmark launch.
4. The unchanged matched MTP benchmark.

A failed or deferred preflight stops the workflow before its following stage.
A failed control prevents benchmark launch. There are no retries, relaxed
thresholds, counter-baseline subtraction or selectively removed treatments.
No applications are closed, foreign workers terminated, or OS memory/power
settings changed.

Each preflight requires ten fresh contiguous seconds within the strict
120-second admission deadline. All four global paging deltas must remain zero;
external CPU must be at most 150%, individual-process CPU at most 80%, model CPU
at most 10%, GPU utilization at most 5% and thermal state at most fair. Existing
freshness and two-second maximum-gap requirements remain. Census checks occur
before admission, during its one-second diagnostic cadence and after observer
cleanup. Unavailable inventory fails closed. The census detects visible exact
`stark-generate` basenames; renamed workers and activity between samples remain
outside its coverage.

The preflight retains raw counter observations privately, host observations,
census statuses, deadline evidence and monitor/sampler cleanup. Admission must
remain fresh within two seconds after cleanup and immediately before child
launch. A census, native call or cleanup can finish after the admission deadline;
that cannot turn a late observation into admission.

## Control and matched acceptance

The control retains one warmed persistent Python process and the fixed seed-42
order **P, E, F, F, P, E**, with six 60-second treatments. E uses global
endpoints and sleep; P adds direct reads every 0.5 seconds; F additionally uses
the unchanged complete host Monitor. Task/GPU endpoints, CPU reads and census
checks stay outside the central global brackets. The child control checks for
workers around each treatment and launches no model.

Quiet admission establishes entry conditions for the sequence. It does not
establish continuous host quiet throughout the endpoint-only E treatments or
all later treatments. Preserve polling/wakeup differences, adjacent carryover,
uncontrolled host drift and two repetitions as limits on causal interpretation.
Task/GPU endpoint intervals differ from global brackets, and their incomplete
coverage cannot support baseline subtraction or complete event attribution.

The matched matrix retains prompts 32/128, context 512, greedy seed 42,
maximum 16 outputs with matching natural EOS, drafts one through three,
one warmup per off/on route and four balanced pairs per group. Intended work
is 12 warmups and 48 measured arms. All predecessor reference hashes, useful
work, stopping, real speculative activity and target-verification checks remain.

Every measured group still requires combined quiet admission under the limits
above. Measured arms retain external CPU at most 150%, model CPU at most 10%,
thermal at most fair, no active compiler, no monitor failure or gap over two
seconds, zero global compression/decompression/pageout/swapout deltas,
request-boundary coverage and final monitor join. Retain startup available
memory at least 64 GiB, sampled owned
RSS at most 48 GiB, 180-second request and 1,800-second worker limits.

A depth can qualify only with complete eligible groups for both prompt lengths,
off wall spread at most 15%, median paired wall and decode speedups at least
1.05, and median prefill on/off at most 1.05. Warmup agreement, successful
workflow execution and model-free tests do not independently establish a speed
benefit or broader model qualification.

## Registered execution and tests

The first registered case, `quiet-window-1`, deferred at its first preflight
after 120.059 seconds. The interval recorded 270 global decompressions, zero
compression/pageout/swapout increases and CPU/GPU gate failures. Maximum
combined quiet was 0.643 seconds. All 113 census checks passed, the monitor
joined with 198 samples and zero failures, and no child stage or model launched.
Preserve this deferred result.

The user subsequently declared a fresh quiet window and explicitly authorized
another attempt. `quiet-window-2` uses its own parent and child registrations;
this is a separately authorized run, not an automatic retry. Initial
unregistered host observations are excluded from both registered results.

Both second-case preflights admitted. Their selected zero-paging streaks were
10.441/10.400 seconds, with contiguous host coverage of 10.356/10.300 seconds.
The wider waits contained rejected observations before those selected streaks.
All 75/25 census checks passed. Both monitors joined with zero failures;
post-cleanup launch ages were below 0.214 seconds. These are entry conditions,
without a claim of continuous quiet throughout the subsequent control.

The child control passed all six fixed treatments and its no-worker census
checks, with zero model launches. P/E/F/F/P/E global decompressions were
99/43/49/195/72/85, totaling 543 over 360.200 treatment seconds; compression,
pageout and swapout deltas stayed zero. The control supports the registered
sampled no-worker condition. It does not identify the cause of those global
events or establish a monitor-caused decompression total.

The matched child completed all 12 warmups with predecessor-matching hashes,
emitted/sampled counts and stops. Prompt 32 emitted six and sampled seven tokens
with EOS; prompt 128 emitted/sampled 16 with a length stop. Forty of 48 intended
measured arms completed across the first five groups. Thirteen individual arms
passed their screens, distributed 2/8, 5/8, 5/8, 0/8 and 1/8 across
32/depths-one-through-three and 128/depths-one-through-two. Every complete group
was timing-ineligible. No eligible subset is promoted to a complete matched
comparison.

The final 128/depth-three group deferred before any measured arm, after
120.027 seconds and 153 global decompressions. Other paging deltas stayed zero.
Counter-only quiet reached 15.844 seconds, while combined quiet reached only
7.889 seconds; CPU/GPU conditions also failed. Preserve all completed arms and
the deferred group. No additional attempt or threshold change followed.

All 402 worker memory snapshots recorded zero worker-task decompressions.
The benchmark's final monitor joined with 1,310 samples and zero failures.
Startup available memory was 90,447,953,920 bytes and sampled peak owned RSS
786,595,840 bytes. The worker exited zero on stdin EOF and was reaped. These
gauges do not establish physical Metal residency or complete page ownership.

The 26 new private model-free checks passed, and 47 existing gate/observer
checks were rerun successfully. They qualify the stated orchestration and
admission checks separately from numerical or timing evidence. The parent and
benchmark reports both retain `deferred`; default draft decision remains zero.

## Disposition and evidence

Keep selected-record bindings opt-in and MTP default off. No new throughput,
draft-depth ranking, product admission, installation, broader-context or prefix
reuse qualification follows from this workflow. Detailed native records
remain private and ignored; publish aggregates and hashes only.

The [benchmark evidence table](../benchmarks/2026-11-13-edge0-quiet-window.md#evidence-bindings-and-limits)
binds both parent runs, the completed child control, partial benchmark, source
manifest and new helper/tests. Preserve all predecessor records, including
attempt 15's failed control.
