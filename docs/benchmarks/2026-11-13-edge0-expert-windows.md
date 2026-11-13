# Edge0 expert windows: bounded churn reduction, timing still deferred

Attempt [20261113-EDGE0-14](../model-experiments/2026-11-13-edge0-expert-windows.md),
2026-11-13, local non-commercial evaluation only. Record-sized Metal bindings
avoided the post-idle worker decompressions reproduced by both baseline sessions.
The separate MTP performance attempt still deferred all six groups: **12 warmups,
zero measured arms and no qualified timing**. MTP remains off by default.

## Identity and changed path

Implementation `7c900c69`, Rust 1.98.1, 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332. Worker SHA-256:
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
Pinned source `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`,
native receipt SHA-256:
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
Prior full payload hashes and their immutability assumption are reused; no new
large-payload rehash is claimed.

`STARK_EDGE0_EXPERT_WINDOWS=1` binds an expert's original 1,630,720-byte
gate/up/down record instead of its 834,928,640-byte layer. Projection bindings,
F32 arithmetic, kernels, routing and sampling are unchanged. Unset/zero keeps
the layer policy; the record policy remains opt-in. Page-rounded complete
admission increases by 390,070,272 bytes, under the existing cap/reserves.
This changes neither model bytes nor OS memory/power settings or explicit
residency. Backend diagnostics expose aggregate registration/allocation gauges.

## Fixed-idle diagnostic control

One sealed binary, fresh workers, policy order layer/record/record/layer. Each
session uses the same 32-token prefix, context 512, seed 42, greedy sampling,
maximum 16 outputs and natural EOS. Run off/on, wait at least 120 seconds with
identical diagnostic scheduling, then run off/on again. Query waits are one
second after completion, an approximate cadence rather than exact 1 Hz.

| Session / policy | Warm off task decompressions | Warm on | Post-idle off | Post-idle on |
| --- | ---: | ---: | ---: | ---: |
| 0 / layer | 2,148 | 0 | 458,036 | 0 |
| 1 / record | 0 | 0 | 0 | 0 |
| 2 / record | 0 | 0 | 0 | 0 |
| 3 / layer | 89,213 | 0 | 446,679 | 0 |

All 16 requests match the same input/output hashes and six emitted/seven sampled
tokens with EOS; off speculative activity is zero and on proposals/verification
are real. Fixed idle waits were 120.114–120.152 seconds. The control contains
472 valid snapshots; all workers exited normally and were reaped. Idle task
decompressions were zero except eight in the last baseline session.

| Retained resource gauge after on | Layer policy | Record policy |
| --- | ---: | ---: |
| Complete prospective bank allowance, bytes | 43,819,991,040 | 44,210,061,312 |
| Registered bank bytes for these requests | 43,819,991,040 | 11,956,895,744 |
| Registered windows | 96 | 5,036 |
| Metal allocated resource bytes | 50,077,827,072 | 18,214,731,776 |
| Scratch high-water bytes | 10,485,760 | 10,485,760 |

Observed retained bank-registration bytes fell 72.714% for this workload. The
larger complete prospective allowance includes overlapping page edges. These
are registration/allocation extents, not physical residency or distinct pages.
The source change and ABBA result support binding extent as an actionable cause
of excess churn in this bounded control. Complete OS/driver fault ownership is
unestablished. Four sessions, fixed off-before-on route order and uncontrolled
host activity establish no MTP efficiency ranking, throughput or general claim
that decompressions are eliminated. Task/global intervals differ; never subtract
their counters or form attribution percentages.

## Correctness and lifecycle

The record policy passed 26 actual lifecycle cases, including nine greedy hash
comparisons, stochastic sampling, budget tails, custom stopping,
streaming, prompt/decode cancellation with two same-worker recoveries,
cancellation-channel EOF, three invalid requests and two context-512 plans.
No lifecycle memory trace was captured. The worker exited normally and was reaped.

Native engine checks passed; final focused Edge0 checks passed 52 tests, with
11 additional artifact-dependent cases ignored. The new GPU transport test validates page
prefixes/tails, identical original bytes and products, and mapping ownership
through both completions after file unlink. Worker tests passed 163 with seven
ignored. Six startup cases, 55 private observer/control tests, formatting,
repository policy and production-target Clippy passed.

## Unchanged-screen performance attempt

Original weights/F32, virtual Metal, record policy, context 512, no prefix reuse;
prompts 32/128, greedy seed 42, maximum 16 outputs with matching natural EOS,
depths one through three, one warmup per route and four intended balanced pairs
per group. Planned work was 12 warmups plus 48 measured arms. All four global
zero-paging gates, CPU/GPU/thermal/compiler/gap/failure screens, fresh contiguous
ten-second combined quiet admission, 120-second deadlines, boundary coverage,
post-diagnostic freshness and final monitor join are retained. There were no
retries, threshold changes, baseline subtraction or removal of rejected arms.

| Prompt / on drafts | Admission global decompressions | Maximum sampled counter-only streak s | Maximum combined streak s | Outcome |
| --- | ---: | ---: | ---: | --- |
| 32 / 1 | 5,726 | 2.954 | 2.429 | Deferred |
| 32 / 2 | 1,722 | 5.442 | 4.251 | Deferred |
| 32 / 3 | 1,404 | 3.597 | 1.780 | Deferred |
| 128 / 1 | 4,384 | 8.481 | 8.481 | Deferred |
| 128 / 2 | 3,118 | 9.759 | 8.601 | Deferred |
| 128 / 3 | 3,632 | 7.290 | 5.560 | Deferred |
| Sum | 19,986 | Below ten seconds | Below ten seconds | Zero measured arms |

Admission windows totaled 720.392 seconds. Compressions, pageouts and swapouts
stayed unchanged in every window. Whole-observer global decompressions were
20,843 from the first pre-query through the final post-query counter snapshot;
this includes warmups, gaps and diagnostic query brackets and differs
from the admission table. Query-correlated activity establishes no observer
causation.

All 657 performance snapshots were valid; task decompressions stayed zero
throughout the observed span. Task page-ins totaled 199,722, with zero page-ins
or decompressions during all admission waits. Every group also breached CPU/GPU
gates. Overlapping rejection counts were external CPU 433, process CPU 272,
external model CPU two and GPU 297. Maxima were 690.922%, 413.818%, 11.401% and
93% respectively; thermal stayed nominal. No freshness/contiguity rejection was
recorded, without a separate maximum counter-polling-gap measurement.

All 12 warmups match predecessor hashes/counts/stops: 6/7/EOS for prompt 32 and
16/16/length for 128. Off activity was zero; on routes proposed and verified real
work. Retained registration grew to 19,286,458,368 bytes / 9,488 windows with
the longer prefix; Metal allocation reached 25,544,294,400 bytes. Future prompts
can populate more of the admitted inventory. The short control is not a universal
memory bound.

The final monitor joined with 1,365 samples and zero failures. Worker stdin EOF
exit was zero and reaping succeeded. Available memory before launch was
82,907,873,280 bytes; sampled peak owned RSS was 747,405,312 bytes, within limits.
These gauges do not establish complete Metal residency.

## Evidence and disposition

Independent reviews regenerated all 2,185 source entries, rehashed the sealed
binary/helpers/tests/SDK/artifact bindings and recomputed control/performance
trace deltas and brackets. No discrepancy was found. Detailed receipts remain
private and ignored; predecessor evidence is preserved.

| Binding | SHA-256 |
| --- | --- |
| Source manifest | `72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47` |
| ABBA registration | `930c3a51a0a2a4c87a3b6dafba54f26e6b863216816a2c68e7fe69c22e5ab16c` |
| ABBA aggregate report | `dbc8f16c33c9836db0146ea786cc77343ddbd3d504b49579d8ced9e284b81ad9` |
| Lifecycle aggregate report | `976a1656e938d818fffe1b60b4d068e53d9261103198624214a41a1318d8afcd` |
| Performance registration | `a891aaaaef2c7bf8b45e335ddd66a1cb802ddcca97291a0f2b277f1712da33b0` |
| Performance aggregate report | `d0a4c8d15ea07490eb66afd23aca79fc1595b67e868fec64f733189cdeaa3b4d` |

The binding experiment reduced the reproduced worker churn within its tested
scope. A qualified MTP decision still needs complete eligible matched groups;
residual global decompressions and host activity prevent that here. Keep record
bindings opt-in, MTP default zero, context 512, unavailable prefix reuse and local
non-commercial scope. Wider contexts, other artifacts and product admission
remain outside this result.
