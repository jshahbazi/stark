# Edge0 quiet-entry control passed; matched timing remains unqualified

Attempt [20261113-EDGE0-16](../model-experiments/2026-11-13-edge0-quiet-window.md),
2026-11-13, local non-commercial evaluation only. This record describes the
fixed fresh quiet-entry workflow and both separately registered runs. The first
run deferred before any child launch. After explicit user authorization for a
fresh quiet window, the second run passed both entry preflights and its
no-worker control, then completed 12 matching warmups and 40 measured arms.
No complete measured group passed timing eligibility; the final group deferred.
MTP remains default off.

## Identity

Source `f28753754c3a58b5ac291edd7f028abb0553301d`, unchanged inference
implementation/binary from `7c900c69`, Rust 1.98.1. Worker SHA-256:
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
Pinned source `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`;
native receipt SHA-256:
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
Corpus SHA-256:
`1a681566d9fcf584e42df629c610513a786648976d5365687a2e6465b6fc504c`.
Prior complete payload hashes are reused under the recorded immutability
assumption; no new large-payload rehash is claimed.

Original weight formats, F32 arithmetic/state/cache, virtual Metal,
selected-record bindings, context 512 and unavailable prefix reuse remain.
The predecessor host identity is 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332. Record bindings remain opt-in and MTP default off.

## Fixed workflow and screens

The new private parent registration seals the quiet runner, preflight helper,
their tests, the unchanged attempt-15 helpers and both child registrations.
The parent runs strict no-worker quiet admission before the complete observer
control sequence, then repeats that preflight before benchmark launch. A failed
or deferred preflight launches no following stage; failed control launches no
benchmark. There are no retries, threshold changes or baseline subtraction.

Both preflights use ten fresh contiguous seconds and a 120-second admission
deadline. Limits remain external CPU 150%, individual-process CPU 80%, model
CPU 10%, GPU utilization 5%, thermal fair, two-second maximum sampling gap,
and zero global compression, decompression, pageout and swapout deltas.
Worker census checks run before admission, during the existing one-second
diagnostic cadence and after cleanup. Inventory failure or a detected worker
fails closed. Census coverage is sampled visible exact basenames, without
coverage of renamed workers or all between-sample activity. Observer cleanup
must complete and admission remain fresh within two seconds before child launch.

The unchanged control uses warmed observation paths and six 60-second
treatments in fixed seed-42 order P/E/F/F/P/E. E sleeps between global endpoints,
P adds direct reads every 0.5 seconds, and F adds the complete host Monitor.
Task/GPU endpoints, CPU reads and treatment census checks sit outside the global
brackets. Quiet-admitted sequence entry does not establish continuous quiet
during endpoint-only E treatments. Two repetitions, polling/wakeup differences,
carryover and uncontrolled host drift limit causal contrasts.

The unchanged benchmark intends 12 warmups and 48 measured arms: prompts 32/128,
context 512, greedy seed 42, maximum 16 outputs with matching natural EOS,
draft depths one through three, one off/on warmup per route and four balanced
pairs per group. Every group retains strict combined admission under the limits
above. Measured arms retain external CPU at most 150%, model CPU at most 10%,
thermal at most fair, no active compiler, no monitor failure or gap over two
seconds, all four zero-global-paging screens, request-boundary coverage and
final monitor join. Resource limits remain 64 GiB
minimum startup available memory, 48 GiB sampled owned RSS, 180 seconds per
request and 1,800 seconds per worker session.

Performance acceptance requires complete eligible groups at both prompt lengths,
off wall spread at most 15%, median paired wall and decode speedups at least
1.05 and median prefill on/off at most 1.05. Correctness agreement and warmup
timings do not establish performance acceptance.

## Registered observations and result

### First registered quiet preflight

`quiet-window-1` deferred after 120.059 seconds at its first preflight, with
270 global decompressions and zero compression/pageout/swapout increases.
Maximum external CPU was 1,496.971%, individual-process CPU 361.551% and GPU
39%; thermal remained nominal. Combined quiet reached 0.643 seconds and
counter-only quiet 7.936 seconds. Overlapping rejection counts were external
CPU 194, individual-process CPU 94 and GPU 103. All 113 census checks passed;
the monitor joined with 198 samples and zero failures. No child or model launched.

The user subsequently declared a fresh quiet window and authorized the second
run. Its separate registration preserves the first deferral. There was no
automatic retry, threshold change or removal of rejected observations.
Unregistered initial host observations are excluded from both results.

### Second registered entry preflights

Both `quiet-window-2` preflights admitted under the unchanged limits:

| Preflight | Admission elapsed s | Selected zero-paging streak s | Contiguous host coverage s | Census checks, all passed | Final monitor samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before control | 79.538 | 10.441 | 10.356 | 75 | 131 |
| Before benchmark | 26.349 | 10.400 | 10.300 | 25 | 44 |

Both monitors joined with zero failures and both samplers released. Selected
streaks have zero deltas for all four global paging counters; earlier rejected
observations remain part of the wider preflight record. The complete preflight
counter traces contain 1,172/30 decompressions across 131/44 observations;
their scope differs from the selected zero-paging streaks. The maximum recorded
counter polling gaps were 0.724352/0.752560 seconds; diagnostic round trips were
0.053294/0.049504 seconds. Child launch ages after cleanup were below
0.214 seconds. This establishes sampled quiet entry without continuous quiet
coverage of the later endpoint-only treatments.

### Passed no-worker observer control

The control passed its six registered treatments and census checks, launching
zero models. No foreign worker was terminated.

| Session / condition | Treatment seconds | Intermediate global reads | Global decompressions |
| --- | ---: | ---: | ---: |
| 0 / P | 60.002 | 119 | 99 |
| 1 / E | 60.003 | 0 | 43 |
| 2 / F | 60.089 | 119 | 49 |
| 3 / F | 60.094 | 119 | 195 |
| 4 / P | 60.002 | 119 | 72 |
| 5 / E | 60.009 | 0 | 85 |
| Sum | 360.200 | 476 | 543 |

Compression, pageout and swapout deltas were zero in every treatment. Across
their two treatments, F incurred 2.47 seconds of observer CPU and 16.24 seconds
of reaped-child CPU; P incurred 0.03/0 seconds and E 0/0 at `os.times`
resolution. A displayed zero does not establish no work. These descriptive
observations do not identify the source of global decompressions. Entry
admission, sampled worker absence, different polling/wakeup behavior and two
repetitions do not establish a monitor-caused decompression total. No counter
baseline is subtracted.

Both F monitors joined with 99 samples and zero failures. Their external-CPU
maxima were 290.090%/294.336%, individual-process maxima 149.097%/125.354%
and GPU maxima 1%/39%, with nominal thermal state. The passed no-worker
protocol therefore does not establish continuous quiet during the treatments.
Accessible matched task endpoints contain 271 positive decompressions and no
positive observer-task row, but all six comparisons have lifecycle changes and
wider sequential bounds. They cannot be expressed as shares of the global count
or exclude short-lived observer-child activity. GPU comparisons match 21 clients
each; 58 clients per endpoint lack counters and four of twelve endpoints contain
an unavailable prior identity. Raw GPU units remain unverified.

### Partial matched benchmark

All 12 warmups match predecessor hashes, counts and stopping reasons. Prompt 32
emitted six and sampled seven tokens with EOS; prompt 128 emitted/sampled 16
with a length stop. Off speculative work remains zero; on routes propose and
verify real work. The first five groups completed 40 measured arms:

| Prompt / on drafts | Completed measured arms | Individually eligible arms | Eligible pairs | Complete-group outcome |
| --- | ---: | ---: | ---: | --- |
| 32 / 1 | 8 | 2 | 1 | Timing-ineligible |
| 32 / 2 | 8 | 5 | 2 | Timing-ineligible |
| 32 / 3 | 8 | 5 | 1 | Timing-ineligible |
| 128 / 1 | 8 | 0 | 0 | Timing-ineligible |
| 128 / 2 | 8 | 1 | 0 | Timing-ineligible |
| 128 / 3 | 0 | 0 | 0 | Admission-deferred |
| Sum | 40 of 48 intended | 13 | 4 | Zero complete eligible groups |

Individual eligible arms do not establish a complete paired group or a depth
ranking. Preserve every rejected arm; no eligible subset is selected for a
performance claim. Twenty-six arms failed zero-decompression screening and
13 failed external-CPU screening, with 12 arms in both sets. All measured-arm
coverage, gap, failure, compiler, model-CPU and thermal checks passed. Group
summary timings remain private observation-only data.
The 40 measured-arm brackets total 766 global decompressions and zero changes
in the other three paging counters. This total has a different scope from
admission searches and the complete benchmark trace.

The final 128/depth-three admission deferred after 120.027 seconds, with
153 global decompressions and zero compression/pageout/swapout increases.
Counter-only quiet reached 15.844 seconds but combined quiet only 7.889 seconds.
External CPU reached 713.716%, individual-process CPU 339.711% and GPU 94%,
with thermal at most fair. Overlapping rejection counts were external CPU 61,
individual-process CPU 16 and GPU 103, plus one phase-boundary observation.
No freshness or contiguity rejection was recorded. Its maximum polling gap was
0.752684 seconds and diagnostic round trip 0.164008 seconds.

All 402 worker memory snapshots record zero task decompressions. Startup
available memory was 90,447,953,920 bytes and sampled peak owned RSS
786,595,840 bytes. The final monitor joined with 1,310 samples and zero failures.
The worker exited zero on stdin EOF and was reaped. The child and parent both
retain `deferred`; the default draft decision remains zero.

The benchmark's complete 1,448-observation paging trace contains 2,276 global
decompressions and four pageouts, with zero compression or swapout increases.
This spans warmups, admission searches, measured requests and inter-group work;
it differs from both measured-arm brackets and the final deferred admission.
The first five successful admission reports' zero deltas describe their winning
streaks, not every earlier observation in their search windows. Worker task
page-ins increase by 1,529,598 across the observed span. Task execution counters
do not identify the cause of global events or distinct physical page ownership.

Twenty-six new model-free checks passed, and 47 existing gate/observer checks
were rerun successfully. Those checks provide orchestration and gate evidence,
without new numerical or performance qualification.

## Evidence bindings and limits

| Binding | SHA-256 |
| --- | --- |
| Source manifest | `72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47` |
| First parent registration | `4d07e9678e86f78a73bba16cd603a193aa5ce873fc6db11f1924954eefae2693` |
| First deferred parent report | `9731a73851b5e9672c218c5957c893cd224c92b7fdecc1c49dd4bfefdbbf8eec` |
| Second parent registration | `9163eecb70cd51482196e2101d7ccfde8b902245bb95684a94008d23b6882a0f` |
| Second deferred parent report | `1126f6bd4c0b715bde710b912d90eaeddfc48b7b8fc52cfe1f4893c3a9b0efc4` |
| Second child control registration | `c226f2435367adac3cf499dd97f7399229033204eb23b4d60baf803007f69f50` |
| Passed child control report | `9c3e23d1c0ddb9d312cb07ed8ce1ceabd0e9c5c5546d2e4b67d9e266ab471b89` |
| Second child benchmark registration | `fc8aebc4b1154b3e6048a51d052897a8d6d11366b569c2acbace1c52dcee8e23` |
| Deferred child benchmark report | `2dc2376f989a81c577a6dede893505bdda99a28d2e757947a87d5c1c57937e81` |
| Quiet runner | `1c4ed4b7ea8a9594bb4d3e699c05d28f3cfba0684865158ac190d88985050411` |
| Quiet runner tests | `2a87040355da3b5379b9d6fc877ae6d92c2cdec320b5fbbcd4c97797973435f3` |
| Quiet preflight helper | `32e1dc0a35200c82ab6d78ce77dc86514057e4c7a57db11f79d016fda0abc543` |
| Quiet preflight tests | `2d0d19ce705712ca8a7690983b48b19f3627fdddb94e288b55e3c088fd33b778` |

Independent read-only review verified all 2,185 source-manifest entries,
including 18 internal symlinks, previous/new helpers and tests, seven SDK headers,
worker, exact worktree Python executable, native receipt and parent/child
registration/report hashes. The complete payload immutability assumption
remains explicit; this review performs no new large-payload rehash.

Preserve the
[failed predecessor control](2026-11-13-edge0-host-interference.md) and all
prior source-bound receipts. Accessible task and GPU-client endpoint diagnostics
retain their incomplete coverage and different intervals; no subtraction,
attribution percentages or verified GPU counter units are implied. RSS and
registration extents do not establish physical Metal residency.

No applications are closed, foreign workers terminated, or OS memory/power
settings changed. Detailed native records stay private and ignored. No new
performance, draft-depth, broader-context, prefix reuse, installation or product
admission qualification is claimed. MTP remains default off.
