# Edge0 worker memory diagnostics: screened MTP rerun deferred

Attempt [20261113-EDGE0-13](../model-experiments/2026-11-13-edge0-worker-memory.md),
2026-11-13, local non-commercial evaluation only. **Twelve warmups matched; all
six groups failed strict admission; zero measured arms ran.** MTP stays off by
default. Worker memory diagnostics are locally validated; Phase 5 performance
qualification remains open.

## Identity and controls

Implementation `590b185d`, Rust 1.98.1, 128 GiB Apple M5 Max / Mac17,7,
macOS 26.7.2 build 25G332. New worker SHA-256:
`c9ab3bb7c469f56e7fee1954ebc3f778d293e35da978d03ae58469dfede8f846`.
Pinned artifact `Edge0/Edge0-145B-q3k` at
`d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, native receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
The prior full payload receipt and immutability assumption are reused; no new
large-payload rehash is claimed. Inference arithmetic and precision are unchanged.

One serialized JSONL worker, original weights/F32 policy, virtual Metal, context
512, prefix reuse disabled, greedy seed 42, prompt lengths 32/128, maximum 16
outputs with matching natural EOS. Depths one through three each have one warmup
per route followed by an intended four balanced pairs. The full intended matrix
is 12 warmups plus 48 measured arms. Warmup order is always off then on.

`STARK_WORKER_MEMORY_DIAGNOSTICS=1` enables explicit self-task queries around
every request and approximately once per second during admission. All four
global paging counters, CPU/GPU/thermal thresholds, compiler/failure/gap screens,
fresh contiguous ten-second admission, 120-second admission deadline, request
boundary coverage and final monitor join are retained. Query costs and unrelated
activity are never subtracted from global screens. Added pre-request queries
must leave the first measured request's admission endpoints at most two seconds
old. The default-selection criteria are unchanged. No selective retries occurred.

## Admission outcomes

These are global page-counter deltas over six admission windows, excluding
warmups/gaps. All waits reached the registered 120-second deadline. Counter-only
streaks are sampled fresh streaks; the combined screen also includes host gates.

| Prompt / on drafts | Global decompressions | Global compressions | Global pageouts | Max counter-only streak s | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| 32 / 1 | 34,570 | 572,336 | 477 | 1.109 | Deferred |
| 32 / 2 | 17,493 | 913,718 | 786 | 1.203 | Deferred |
| 32 / 3 | 3,137 | 0 | 34 | 1.115 | Deferred |
| 128 / 1 | 14,846 | 1,238,712 | 255 | 0.629 | Deferred |
| 128 / 2 | 21,640 | 0 | 5 | 0.730 | Deferred |
| 128 / 3 | 3,394 | 0 | 0 | 1.112 | Deferred |
| Sum | 95,080 | 2,724,766 | 1,557 | — | Zero measured arms |

Swapouts remained unchanged in every window. Admission waits totaled 720.322
seconds and 1,191 counter observations, with 1,059 nonzero-counter intervals.
The maximum combined quiet streak was 1.115312 seconds. Even the sampled
counter-only screen failed to reach ten seconds in every group.

Host gates also rejected intervals: external CPU 415, single-process CPU 154,
external model CPU four, and GPU utilization 327. Counts overlap and do not
establish unique causes. Maxima were 885.318%, 493.787%, 59.821% and 93%
respectively, versus unchanged admission limits 150%, 80%, 10% and 5%. Thermal
remained nominal. No freshness/contiguity rejection was recorded; the receipt
does not retain a separate maximum admission counter-polling gap, so no stronger
gap-free claim follows. Device utilization is not process-attributed GPU use.

## Worker-task observations

All 686 diagnostic snapshots had valid VM/events statuses, 87/eight returned
words, 16,384-byte pages and monotonic counters. Maximum diagnostic round-trip
was 0.189 seconds, including IPC and scheduling. These are task execution/event
counts and accounting gauges, not an inventory of model-owned or distinct pages.

| Warmup prompt / on drafts | Off task decompressions | On task decompressions |
| --- | ---: | ---: |
| 32 / 1 | 20,608 | 0 |
| 32 / 2 | 476,988 | 2 |
| 32 / 3 | 900,294 | 0 |
| 128 / 1 | 38 | 2 |
| 128 / 2 | 1,166,054 | 0 |
| 128 / 3 | 0 | 0 |

Warmup endpoint deltas totaled 2,563,986 task decompressions and 3,035,620 task
page-ins. During admission, task decompressions increased by 1/0/2/0/0/0 and
page-ins stayed unchanged. This establishes worker-charged decompression
activity in some warmups and near-flat task counters during global admission
activity. It does not establish complete GPU/driver attribution. The task and
global counter endpoints have different brackets; do not subtract them or form
attribution percentages.

All off warmups precede on warmups, and cold/idle histories differ. These counts
cannot establish route-specific memory efficiency or speed. No warmup timing is
promoted. The six pairs matched predecessor and within-prompt output hashes,
emitted/sample counts and stop reasons: 6/7/EOS at prompt 32; 16/16/length at 128.
Off speculative counters were zero; on routes had real proposals and sufficient
verification.

Task gauge ranges were compressed balance 0–282,722,304 bytes, physical footprint
297,748,568–685,865,168 bytes, resident 253,722,624–701,022,208 bytes and virtual
550,675,890,176–550,917,718,016 bytes. Their different accounting scopes do not
measure complete Metal residency, physical compressed storage or model working
set. The whole observer span recorded global 2,663,628 decompressions, 2,892,960
compressions, 2,068 pageouts and zero swapouts; that includes warmups, gaps and
query brackets and must not be confused with the admission table. There is no
diagnostics-disabled control or proof that query-correlated events were caused
by the observer.

## Validation and disposition

The monitor stopped with zero failures and 1,392 host samples. The worker exited
normally on stdin EOF, code zero, and was reaped. Available memory before launch
was 92,184,526,848 bytes; sampled peak owned RSS was 827,342,848 bytes. Registered
resource limits passed; RSS is not complete GPU residency.

Independent review rehashed the exact 2,184-file source inventory and all sealed
bindings, then checked every diagnostic record and derived delta. Worker tests
passed 161 with seven ignored; the new ignored live self-task test passed
separately. Six startup cases, 38 private observer/screen tests, SDK static ABI
assertions, formatting, repository policy and production-target Clippy passed.
The existing all-target test-placement lint is retained in the attempt record.

| Binding | SHA-256 |
| --- | --- |
| Source manifest | `1742e9532a5828c3c546b9e7234582f095ac03e28a48de29c6699a72b4629dbb` |
| Registration | `e00bfb86cd66fdb1527cc1c74fb5f179ccd1047a7c8301341a08d2b30cf7d786` |
| Aggregate report | `e3bd3459b821e9f3f72376074a4edbb29344fc9074c9ee5dac08cff9e92e2634` |
| Harness | `6bfbd77325e11dace60ad9f515fc028d6d96cfdc0213544f86ff77904874bd2e` |

Detailed diagnostics remain private and ignored. A performance decision needs
complete eligible matched groups under the same screens. Default zero, the
512-token limit, unavailable prefix reuse and local non-commercial scope remain
in force; no performance ranking or product admission follows.
