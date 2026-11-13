# Edge0 host interference: failed observer control, timing deferred

Attempt [20261113-EDGE0-15](../model-experiments/2026-11-13-edge0-host-interference.md),
2026-11-13, local non-commercial evaluation only. All six observer-control
treatments completed, but the final census found an unrelated evaluation worker.
The registered no-worker control **failed**. Its preserved observations cannot
establish an observer-caused decompression total. The separately registered
strict matched rerun completed 12 matching warmups and six admission deferrals,
with zero measured arms. MTP remains off by default.

## Identity

Source `0047285f`, unchanged inference implementation/binary from `7c900c69`,
Rust 1.98.1. Worker SHA-256:
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
Pinned source `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`;
native receipt SHA-256:
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
The prior complete payload hashes and immutability assumption are reused; no new
large-payload rehash is claimed. Recorded host: 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332. Original weights/F32 arithmetic, virtual Metal,
record policy 1, context 512 and unavailable prefix reuse are unchanged.

## Fixed observer-control observations

One persistent Python process warmed imports, APIs and commands before the
registered intervals. Two seed-42 shuffled permutations fixed order P/E/F/F/P/E.
E sleeps between shared global endpoints; P adds direct reads every 0.5 seconds;
F adds the unchanged host Monitor, including startup/final tick/join inside the
global bracket. Task/GPU endpoints, CPU reads and worker census checks are outside
that bracket. The control itself launches no model.

| Session / condition | Treatment seconds | Intermediate global reads | Global decompressions |
| --- | ---: | ---: | ---: |
| 0 / P | 60.005 | 119 | 153 |
| 1 / E | 60.001 | 0 | 274 |
| 2 / F | 60.099 | 119 | 891 |
| 3 / F | 60.089 | 119 | 125 |
| 4 / P | 60.002 | 119 | 237 |
| 5 / E | 60.003 | 0 | 591 |
| Sum | 360.199 | 476 | 2,271 |

All six intervals have zero compression, pageout and swapout deltas. The wider
first-global-before to last-global-after interval, including gaps, contains
2,274 decompressions over 362.152 seconds; its scope differs from the table.
No counter baseline is subtracted and no rejected treatment is removed.

| Condition, two treatments | Observer CPU seconds | Reaped-child CPU seconds |
| --- | ---: | ---: |
| E | 0.00 | 0.00 |
| P | 0.04 | 0.00 |
| F | 2.32 | 14.34 |

CPU values use `os.times` resolution; a displayed zero does not prove no work.
The F monitors recorded 100/102 samples, zero failures, joined threads and maximum
intervals 0.622631/0.622300 seconds. External CPU maxima were 394.651%/347.857%,
single-process maxima 133.238%/127.265%, GPU maxima 41%/39%, and thermal nominal.
External CPU excludes the harness and its descendants. Basename CPU buckets
retain only process samples at least 5%; their integrals are censored, and
snapshot observations can miss exited short-lived processes.

## Task/GPU diagnostic limits and failed census

Accessible same-user task endpoints recorded 1,516 positive task decompressions
across matched identities, including Finder, Docker and other applications.
The observer task has no positive matched delta. All six comparisons have
incomplete lifecycle coverage because tasks enter or leave the endpoint subset.
These are charged execution events across sequential endpoint intervals wider
than the central global interval. They cannot be divided by the global total,
subtracted from it or used to exclude short-lived observer-child activity.

GPU endpoints retain 21 matched clients per comparison, with one additional
after-only client in the final E session. Sixty registry entries lacked usable
Metal counters at each endpoint, increasing to 61 at the final endpoint. Positive
matched raw deltas include WindowServer, Claude Helper and Codex basenames.
`accumulatedGPUTime` units remain unverified, coverage is an incomplete client
subset, and endpoint intervals differ from both task and global intervals.
This identifies observed client activity without a complete GPU ownership or
utilization attribution.

The final census found a foreign `stark-generate` after session 5. Report status
is **failed**, failure code `evaluation_worker_present`. Its exact overlap with
the last treatment is not established by this report. Preserve the last E
session and the failed whole-control result; the remaining rows do not become
a clean observer-causality experiment by omission. Two repetitions, uncontrolled
host drift, polling/wakeup changes and adjacent carryover further limit contrasts.
Monitoring CPU overhead was measured; decompression causality was not established.
No foreign worker was terminated and no OS memory/power settings changed.

## Strict matched rerun

The registered matrix retains prompts 32/128, greedy seed 42, maximum 16 outputs
with matching natural EOS, draft depths one through three, one warmup per route
and four balanced pairs per group. Intended work is 12 warmups and 48 measured
arms. All four zero-global-paging gates and the CPU/GPU/thermal/compiler/gap/
failure screens remain, with fresh contiguous ten-second combined admission,
120-second group deadlines, request-boundary coverage and final monitor join.

The private helper tightens deadline enforcement after sleeps, diagnostic and
counter reads and immediately before admission return, rejecting a late
observation. Polling-gap and diagnostic-latency maxima and additive raw traces
are newly retained. Eighty-two private model-free tests passed. This is separately
bound tooling evidence, with no new inference or numerical qualification.

The startup 64 GiB available-memory gate passed after recovery. The unrelated
worker remained recorded; its interference is subject to the unchanged screens.
The earlier post-control available-memory observation is not the launch value.

All 12 warmups match predecessor hashes, counts and stops: six emitted/seven
sampled/EOS for prompt 32 and 16/16/length for prompt 128. Off speculative work
is zero; on routes propose and verify real work. All six groups deferred before
any of the intended 48 measured arms, with no retry or threshold change.

| Prompt / on drafts | Global decompressions | Global pageouts | Maximum counter-only streak s | Maximum combined streak s | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 32 / 1 | 351 | 0 | 4.943 | 1.150 | Deferred |
| 32 / 2 | 354 | 0 | 4.705 | 1.789 | Deferred |
| 32 / 3 | 232 | 0 | 12.190 | 4.703 | Deferred |
| 128 / 1 | 187 | 45 | 8.659 | 6.733 | Deferred |
| 128 / 2 | 132 | 0 | 11.915 | 6.034 | Deferred |
| 128 / 3 | 199 | 0 | 9.720 | 6.736 | Deferred |
| Sum | 1,455 | 45 | Two groups reach ten seconds | Every group below ten seconds | Zero measured arms |

Admission waits totaled 720.367 seconds; compressions and swapouts stayed
unchanged in every wait. Counter-only quiet briefly exceeded ten seconds in two
groups, while every group also breached host CPU/GPU conditions. Paging alone
does not explain all admission failures. Overlapping rejection counts were 453
external CPU, 272 single-process CPU, one external-model CPU and 605 GPU, plus
two phase-boundary observations. Maxima were 479.523%, 191.772%, 11.383% and 92%
respectively; thermal remained nominal. No freshness or contiguity rejection was
recorded. The maximum recorded strict-admission polling gap was 0.752114 seconds
and the maximum diagnostic round trip 0.156958 seconds. The sole model-name
heuristic breach had basename `Stark`; it does not identify a particular model
workload.

CPU-name buckets from the 1,194 reconstructed admission observations identify
additional host activity. Reconstruction reproduces each group's sample count,
maxima and rejection map, retaining crossing-phase observations. The phase tag
alone is insufficient because it persists into later warmups. These totals
retain only per-process samples at least 5% CPU and can combine several processes
with the same basename. They are censored sampled CPU seconds, with no job or
command-argument attribution:

| Basename | Observed CPU seconds | Maximum individual-process CPU % |
| --- | ---: | ---: |
| WindowServer | 224.07 | 53.068 |
| python | 94.08 | 112.757 |
| Python | 85.90 | 105.595 |
| Google Chrome renderer helper | 74.73 | 141.761 |
| Claude renderer helper | 42.93 | 139.364 |
| mds_stores | 34.15 | 191.772 |

The first-to-last 2,516 persistent paging observations span 833.694 seconds and
contain 1,837 decompressions and 321 pageouts, with zero compression or swapout
increase. This includes warmups and inter-group activity and differs from the
admission table. All 658 worker snapshots were valid; task decompressions stayed
zero throughout. Task page-ins increased by 1,529,598 over the observed span,
with zero during each admission wait. These are execution counters and gauges,
without complete GPU/driver ownership accounting.

Accessible task comparisons around admission contain 518 positive task
decompressions across wider sequential intervals, with lifecycle changes and
one denied task read preserved. Leading basename aggregates include Chrome
renderer 160, `claude` 81, Signal 45 and `node` 43 charged events. These are not
shares of the global count. Each GPU comparison matches 23 clients, while 61
registry entries at each endpoint lack usable Metal counters. Positive raw
deltas include application/graphics and `stark-generate` activity. The latter
basename aggregate does not separate the owned worker from a foreign worker
with the same name. These incomplete endpoint subsets do not establish a causal
source for all global events and are never subtracted from a timing gate.

Final retained bank-registration bytes were 19,286,458,368 across 9,488 windows;
Metal allocated-resource bytes were 25,544,294,400, with complete prospective
allowance 44,210,061,312 and scratch high-water 10,485,760 bytes. Registration and
allocation extents do not establish physical residency or a universal memory
bound. Startup available memory was 82,959,859,712 bytes; sampled peak owned RSS
was 757,071,872 bytes. Whole-session parent CPU was 116.128 seconds; reaped-child
CPU was 114.720 seconds, including the worker and observer children, so this
whole-session diagnostic does not isolate monitoring overhead.

The final monitor joined with 1,383 samples and zero failures. The owned worker
exited zero on stdin EOF and was reaped. The control's failed no-worker condition
remains explicit; no clean observer contrast or eligible MTP ranking follows
from this rerun.

## Evidence bindings

Independent review rehashed all 2,185 source entries with the sealed symlink
contract, helpers/tests, SDK headers, worker, corpus, native receipt and Python
executable, and recomputed every control global delta. No discrepancy was found.
Detailed endpoint records stay private and ignored; process identities remain
transient. Publish aggregates and hashes only. Locked Python 3.13.13 / psutil
7.2.2 were used.

| Binding | SHA-256 |
| --- | --- |
| Source manifest | `72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47` |
| Observer-control registration | `8600018d653353582b65ff228a4bd1b5ec2d92239458301bb02907f9629a4950` |
| Failed observer-control report | `abc14e49c128f46e88b4eb7b495e22a9bbe7bcaba20fbddba64b93260d26be9c` |
| Matched-rerun registration | `f133f88ace50f3b67c56501cfc4de1e1c1351f8a39402b26d75db8918abc42a7` |
| Deferred matched-rerun report | `e8b935b8e0309eefb72591c04dbdd9e4847507114ac863b6633a487a242dfe1d` |
| Observer-control helper | `25529ef606aadfe670b55b139687cf07511bb5f0f9a28a91598a67051fdd49fc` |
| Matched-rerun helper | `92e65427a8a1bf8d0ed694c5f95e32d4ad60191837dc411e1c20d5985e141daa` |
| Strict admission helper | `ade422496357f800cfad01a9f078a248ed16c0375c4ceee008954266f04d6d49` |
| Same-user task endpoints | `08acf5a8001cffc6fb23597e86948f87bfefa8c117990c6c2a5ca7212d815139` |
| GPU client endpoints | `2b469b3d794e81b75b70516e2c5616a25cee4d9ec7788104cf37b3508e917179` |

Record bindings remain opt-in and MTP remains default off. Complete eligible
matched groups are required before any performance or draft-depth decision.
