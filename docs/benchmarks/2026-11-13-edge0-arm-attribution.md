# Edge0 arm attribution: partial host evidence, timing remains deferred

Attempt [20261113-EDGE0-17](../model-experiments/2026-11-13-edge0-arm-attribution.md),
2026-11-13, local non-commercial evaluation only. The new endpoint diagnostics
preserve the strict native timing screens. The direct registered run completed
12 matching warmups and eight measured arms. Three arms and one pair passed
their individual screens; zero complete groups qualified. Five later groups
deferred at admission. MTP remains default off.

## Identity and scope

Registered source `8704f04c969de2c7e03345829be2f5956c93a78d`, unchanged inference
implementation/binary from `7c900c69`, Rust 1.98.1. Worker SHA-256:
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
Pinned source remains
`Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`;
native receipt SHA-256:
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
Corpus SHA-256:
`1a681566d9fcf584e42df629c610513a786648976d5365687a2e6465b6fc504c`.

Original weight formats, F32 arithmetic/state/cache, virtual Metal,
selected-record bindings, context 512 and unavailable prefix reuse remain.
The recorded predecessor host is 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332. Python 3.13.13 and psutil 7.2.2 remain locked.
Prior full payload hashes are reused under the existing immutability assumption;
this record claims no new large-payload rehash. Registration/source-manifest
agreement and the post-run offline audits have separately recorded scopes.

## Prior evidence investigated offline

The preceding [quiet-window run](2026-11-13-edge0-quiet-window.md) retained
40 measured arms. Two new offline audits reproduce their rejection cross-tab:

| Decompression rejection | External-CPU rejection | Arms |
| --- | --- | ---: |
| No | No | 13 |
| No | Yes | 1 |
| Yes | No | 14 |
| Yes | Yes | 12 |

The result remains four eligible pairs and zero eligible complete groups.
The 26 decompression rejections and 13 CPU rejections overlap in 12 arms.
These counts preserve every arm and do not select a subset for performance.

The exact CPU audit reconstructs stored phase samples through each saved
coverage endpoint. Across 48 unique CPU-rejected intervals, leading retained
basename contributions are WindowServer 6.66 CPU seconds, `duetexpertd` 6.52
and `com.aikido.endpoint.proxy.l4.dist.extension` 4.82. These totals include
full sampling intervals and censor original process samples below 5%; basename
collisions remain possible. The observer excludes its own root and descendants
from external CPU, so retained external Python names do not identify harness
work. Direct observer CPU is measurable; observation-induced external activity
and causal ownership of decompressions remain unresolved.

That run's task endpoints surround admissions only. Their 943 positive charged
task decompressions include ChatGPT 612, Chrome renderer 152, Codex renderer 54,
IMAutomaticHistoryDeletionAgent 37 and IMDPersistenceAgent 26. Those wider
intervals do not attribute the 766 global decompressions inside measured arms.
All six comparisons have lifecycle changes, totaling 58 after-only and
49 before-only identities. All 402 worker snapshots report zero task
decompressions.

The prior passed no-worker control records 543 global decompressions across
six treatments, including 43 and 85 during endpoint-only E treatments. Its
accessible task comparisons record 271 charged events, led by Chrome renderer
117, Signal 43, `codex` 20 and 1Password 19. No positive observer-task row is
present. F treatments record 2.47 seconds of parent CPU and 16.24 seconds of
reaped-child CPU; P records 0.03/0 and E 0/0 at `os.times` resolution.
These observations do not establish observer-caused decompression totals.
Sequential bounds, lifecycle gaps, inaccessible tasks and short-lived children
prevent complete ownership attribution. No baseline is subtracted.

## New protocol and launch history

The registered parent `arm-attribution-1` deferred at its initial quiet
preflight and launched zero children. Its receipt remains preserved. After the
user explicitly directed a launch, a separate `arm-attribution-direct-1`
registration launched the benchmark directly. This direct run has no new passed
parent preflight or no-worker observer control. Its per-group admission and
measured-arm screens remain enforced. The earlier control results retain their
original scope.

The matrix remains prompts 32/128, context 512, greedy seed 42, at most
16 outputs with matching natural EOS, draft depths one through three, one
off/on warmup per route, and four balanced pairs per group. Intended work is
12 warmups and 48 measured arms. Every group requires ten fresh contiguous
seconds under external CPU 150%, individual-process CPU 80%, model CPU 10%,
GPU utilization 5%, thermal at most fair and zero global compression,
decompression, pageout and swapout deltas, within a 120-second deadline.

Measured arms retain external CPU at most 150%, model CPU at most 10%, thermal
at most fair, no active compiler, no sampling failures or gaps over two seconds,
all four zero-global-paging gates, request-boundary coverage and final monitor
join. Resource limits remain 64 GiB minimum startup available memory,
48 GiB sampled owned RSS, 180 seconds per request and 1,800 seconds per session.
There is no threshold relaxation, automatic retry or subtraction.

The new diagnostics warm their complete two-endpoint path before the measured
sequence. Each measured arm receives one same-user task endpoint before the
existing first-arm freshness check and one after the unchanged native screen
has been stored and its phase closed. Native wall/counter brackets are
unchanged. No periodic task inventories are added. Task identities remain
transient; records contain basename aggregates, coverage statuses and hashes.
Unexpected diagnostic exceptions produce fixed failure codes.

Each task endpoint also has global-counter and parent/reaped-child CPU brackets.
The wider arm diagnostic includes endpoints, memory diagnostics and the host
screen's trailing-coverage wait. Its CPU record includes the concurrent Monitor
thread in the same parent process and reaped children. It does not isolate the
causal cost of task inventory or account for a running worker's CPU.

## Direct-run result

All 12 warmups match predecessor output hashes, counts and stopping reasons:
prompt 32 emits six and samples seven tokens with EOS; prompt 128 emits/samples
16 with a length stop. Off speculative work remains zero; on routes propose
and verify real work. This extends unchanged-work evidence without qualifying
performance.

| Prompt / on drafts | Measured arms | Eligible arms | Eligible pairs | Outcome |
| --- | ---: | ---: | ---: | --- |
| 32 / 1 | 8 | 3 | 1 | Timing-ineligible |
| 32 / 2 | 0 | 0 | 0 | Admission-deferred |
| 32 / 3 | 0 | 0 | 0 | Admission-deferred |
| 128 / 1 | 0 | 0 | 0 | Admission-deferred |
| 128 / 2 | 0 | 0 | 0 | Admission-deferred |
| 128 / 3 | 0 | 0 | 0 | Admission-deferred |

Five measured arms fail only the decompression screen. No measured arm fails
the external-CPU screen or the other recorded screen conditions. The eight
native brackets contain 36 global decompressions and zero changes in the other
three paging counters. Group timing summaries remain private observation-only
data. Three arms and one pair do not qualify the complete group.

The five deferred admissions have the following different scope:

| Prompt / drafts | Wait s | Global decompressions | Maximum counter-only streak s | Maximum combined streak s |
| --- | ---: | ---: | ---: | ---: |
| 32 / 2 | 120.096 | 117 | 12.834 | 5.430 |
| 32 / 3 | 120.101 | 1,135 | 10.985 | 6.642 |
| 128 / 1 | 120.006 | 100 | 14.516 | 1.898 |
| 128 / 2 | 120.061 | 211 | 9.619 | 1.357 |
| 128 / 3 | 120.079 | 2,801 | 7.361 | 6.102 |

Their other three global paging counters stay unchanged. Overlapping host
rejection counts total 592 external CPU, 326 individual-process CPU and
496 GPU intervals, plus one phase-boundary interval. Model-CPU, thermal,
freshness and contiguity rejection counts are zero. Counter-only quiet exceeds
ten seconds in three groups while combined quiet stays below ten seconds in
every deferred group. Paging alone does not explain these admission failures.

## Attribution and observer coverage

| Scope | Global decompressions | Positive matched task decompressions |
| --- | ---: | ---: |
| Eight native measured brackets | 36 | No exactly coextensive task measurement |
| Eight wider arm diagnostics | 41 | 1 across their sequential task endpoints |
| Sixteen task-inventory endpoint global brackets | 5 | No separate coextensive task total |
| Six admission task comparisons | Different admission intervals | 2,030 |
| Complete 2,318-observation paging trace | 4,642 | No complete task attribution |

The wider arm task comparisons identify one charged decompression in a Chrome
renderer. Five of eight comparisons have lifecycle changes, totaling four
after-only and seven before-only identities. The 16 endpoint inventories record
11,753 successful task reads, 6,020 other-UID exclusions and four processes gone
during reads; no native error code is recorded. Counts can include the same
task at several endpoints. No positive observer-task row is present.

The endpoint brackets span 1.357371 seconds in total, ranging from
0.080291 to 0.091628 seconds each. Parent CPU increases total 1.34 seconds and
reaped-child CPU 0.03 seconds during these brackets. These are process-level
observations with concurrent Monitor activity, without isolated endpoint-cost
attribution. Wider arm brackets total 38.120061 seconds, with 9.00 seconds of
parent CPU and 4.11 seconds of reaped-child CPU. Endpoint CPU is nested within
these wider totals and must not be added to them. CPU uses `os.times`
resolution; a displayed zero does not establish no work.

The six admission comparisons record 2,030 positive charged task events. Leading
basename aggregates are Finder 785, Chrome renderer 572, CSExattrCryptoService
119, iconservicesagent 100, XprotectService 79, Codex helper 64 and ChatGPT 58.
Their lifecycle totals are 125 after-only and 99 before-only identities.
These wider admission events do not identify the source of measured-arm global
events. Task counters describe charged execution events; they are not a complete
physical-page ownership ledger. The difference between task and global totals
is not an attributable residual, and no attribution percentages are calculated.

All 600 worker memory snapshots report zero task decompressions. Worker task
page-ins increase by 1,529,598 over the observed span. The full paging trace spans
796.078 seconds, including warmups, admission searches, measured requests and
diagnostic work; its 4,642 global decompressions have a different scope from
the native-arm and wider-arm totals. Its other three counters remain unchanged.

Startup available memory is 95,657,328,640 bytes and sampled peak owned RSS
788,774,912 bytes. The final monitor joins with 1,313 samples and zero failures.
The worker exits zero on stdin EOF and is reaped. RSS, task accounting and
registration extents do not establish physical Metal residency.

## Evidence bindings and interpretation

| Binding | SHA-256 |
| --- | --- |
| Source manifest | `72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47` |
| Deferred parent registration | `83d5757035ac8cee2729315ced5b1db4242f0875d2e616fff6be21962046ba7d` |
| Deferred parent report | `8970bf7654d918fa87dd2e1e88e95d9bed61a931c624871b9b2d82d61cc857a9` |
| Direct registration | `d7d98095e4a1069a962bdd2c7ae531347c7ba33bc1d711bb1906aa26a13623fb` |
| Deferred direct report | `bb66d06c123b6e46b52f88bb18fa5f0f29ebcb14ae9b1f0cc80ab014b36a29ac` |
| Prior offline rejection audit | `609f705235a4a44bcbcf6d0558d8b321193ba5b87bc69421b0999d72ca0c33fa` |
| Prior exact CPU audit | `bae4a7daa526d1d54c7156923e1cd810734206d13c07859ade34a53d0f5d19f0` |
| Current exact CPU audit | `141e393ae6897ff197c7005e9afc4bfb8ad14a1fa2709edcaed2d4825254e9f4` |
| Current offline audit helper | `1288bfb80415b1735b4ab59def2d457b3450912e0a59e1709433bd10cd63deba` |
| Current offline attribution audit | `c163a247e0b2036066d1784c9b49e551fff9beafef6c06b2b88bd3644405c5c9` |
| Post-run integrity audit | `9afa7b7f48100e243991a95f60d455bbd183fe97b25462cbe34274914bf39c0b` |

The offline current audit binds its own source, imported offline audit helper,
input reports/registrations and measurement helper hashes. It reconstructs
eligibility, counter arithmetic, bracket order, warmup agreement, task aggregates
and lifecycle coverage. Its identity check verifies recorded bindings and
registration hashes; it does not newly verify every payload or source-manifest
entry. The pre-run model-free suite passes 117 tests, including nine endpoint
tests and 20 parent-workflow tests. Independent review checks the final endpoint
placement and unchanged admission/screen helpers.
The separately recorded post-run integrity audit passes 2,617 checks; its
scope and source bindings remain distinct from the aggregate attribution audit.

Preserve the deferred parent, direct run, rejected arms and all predecessor
receipts. Detailed records remain private and ignored. No applications are
closed, foreign workers terminated or OS memory/power settings changed.
Complete eligible matched groups at both prompt lengths remain required for a
performance or draft-depth decision. No new performance, broader-context,
prefix-reuse, installation or product-admission qualification is claimed.
Record bindings remain opt-in and MTP remains default off.
