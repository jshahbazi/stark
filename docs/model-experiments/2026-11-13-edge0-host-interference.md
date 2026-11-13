# 20261113-EDGE0-15: residual host activity and observer controls

## Identity and scope

| Field | Value |
| --- | --- |
| Owner / authorization | 2026-11-13 / Codex; investigate residual global decompressions and CPU/GPU interference, then rerun unchanged strict matched tests |
| Scope | Local non-commercial evaluation only |
| Source | `0047285f`; unchanged inference implementation/binary from `7c900c69` |
| Predecessor | [Attempt 14](2026-11-13-edge0-expert-windows.md) |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| State / result | Observer control failed its final no-worker census; 12 matching warmups, six admission deferrals and zero measured arms in the separate rerun |
| Promotion / disposition | Diagnostic only; record policy opt-in, MTP default off; no performance promotion |

Attempt 14 removed the reproduced post-idle worker decompressions within its
bounded binding-policy control, while global decompressions and CPU/GPU activity
still prevented timing admission. This follow-up investigates accessible task
and GPU-client activity and observation overhead. It changes private diagnostics
and tightens admission deadline enforcement; inference code and model policy are
unchanged. The separate [benchmark record](../benchmarks/2026-11-13-edge0-host-interference.md)
contains the aggregates and evidence bindings.

## Artifact and instrumentation

Reuse `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`,
native receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`,
and the prior complete payload hashes with their explicit immutability
assumption. No new large-payload rehash is claimed. The Rust 1.98.1 worker remains
SHA-256 `0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.
Original source weight formats, virtual Metal, F32 activation/accumulation/state/
cache policy, record bindings (`STARK_EDGE0_EXPERT_WINDOWS=1`), context 512 and
unavailable prefix reuse are retained. The recorded host remains the 128 GiB
Apple M5 Max / Mac17,7, macOS 26.7.2 build 25G332.

The existing persistent global `host_statistics64` sampler and unchanged host
Monitor remain. Add separately bound endpoint diagnostics:

- Same-user task metadata uses `task_name_for_pid` and `TASK_VM_INFO` revision 5.
  Process identity is checked around each read and between endpoints. Public
  records contain status counts and basename aggregates, without PIDs or task
  ports. Access failures, lifecycle changes and invalid counters stay explicit.
- GPU endpoints read Metal `AppUsage.accumulatedGPUTime` from
  `IOGPUDeviceUserClient`, matching both registry-client and process identities.
  Missing counters remain unavailable. Raw units are unverified; no conversion
  to seconds or complete GPU ownership claim is permitted.
- `os.times` records observer CPU and CPU charged to reaped children. Host CPU
  name buckets retain samples where a process reaches at least 5% CPU, so those
  bucket integrals are censored. External CPU excludes the harness and its
  descendants. Snapshot sampling can miss exited short-lived processes.

All task, GPU and global reads have sequential, different bounds. Task and GPU
endpoint comparisons span more than the central global interval. They cannot be
subtracted from global counters or expressed as attribution percentages. Task
decompressions describe charged execution events, not ownership of distinct pages.

## Preregistered observer control

Use one persistent Python process, warmed imports/native APIs/commands, no model
launches, and two seed-42 shuffled E/P/F permutations. The sealed order is
**P, E, F, F, P, E**, each with a fixed 60-second treatment:

| Condition | Treatment inside the global counter bracket |
| --- | --- |
| E | Shared global endpoints and sleep; zero intermediate counter reads |
| P | Shared endpoints plus persistent direct counter reads every 0.5 seconds |
| F | P plus the unchanged host Monitor, including startup, final tick and join |

GPU/task endpoints, self CPU reads and the same-user worker census lie outside
the global brackets. Refuse a live `stark-generate` in the census; never stop
another task's process. Record every treatment, including failures. E/P changes
polling and wakeups together; F adds the complete monitor. Two repetitions,
adjacent carryover and uncontrolled host activity limit any causal contrast.
No control result changes timing eligibility.

## Correctness, deadline enforcement and tests

The private admission helper now rechecks its fixed 120-second deadline after
sleep, diagnostic work and counter acquisition, and before returning admission.
An observation completed after the deadline cannot admit a group. This closes a
late-return path and tightens the existing deadline; no acceptance threshold is
relaxed. Polling-gap and diagnostic-latency maxima and additive raw traces are
retained under a new helper identity. Previous source-bound records remain intact.

The locked Python environment passed 82 private model-free tests covering the
samplers, endpoint identity/error handling, control privacy/order, strict screens
and qualification harness. No new numerical or lifecycle qualification is
claimed from those tests. The unchanged worker retains attempt 14's evidence.

## Control result and failed condition

All six treatments were recorded, totaling 360.199 seconds and 2,271 global
decompressions. Compression, pageout and swapout deltas were zero. F incurred
2.32 seconds of observer CPU and 14.34 seconds of reaped-child CPU; P recorded
0.04/0 seconds and E 0/0 at `os.times` resolution. Both F monitors joined with
zero failures. Accessible task comparisons contain positive charged events in
several other applications; the observer-task comparison has no positive delta.
This does not exclude decompressions charged to short-lived observer children.

The final census found an unrelated live evaluation worker after the last E
treatment. The report therefore has status **failed** with
`evaluation_worker_present`. All six treatments and the final E result are
preserved. The registered no-worker condition was not established for the full
experiment; this is not a clean observer-causality contrast. No worker was
terminated. Background task events and measurable monitor CPU cost remain
descriptive findings, without assigning the global decompressions to the monitor.

## Matched rerun and disposition

The separately registered rerun retains prompt lengths 32/128, context 512,
greedy seed 42, maximum 16 outputs with matching natural EOS, drafts one through
three, one warmup per route and four balanced pairs per group: 12 intended
warmups and 48 intended measured arms. Preserve all four zero-global-paging
screens, existing CPU/GPU/thermal/compiler/failure/gap limits, fresh contiguous
ten-second combined admission, 120-second group deadlines, boundary coverage
and final monitor join. Keep the 64 GiB startup available-memory minimum,
48 GiB sampled owned RSS, 180-second request and 1,800-second worker limits.

An unrelated worker remained present when the rerun began. The registered startup
available-memory gate passed after available memory recovered; the earlier
post-control observation is not the benchmark launch value. Foreign activity
remains subject to the unchanged screens. No application closure or OS
memory/power change was performed.

The rerun completed all 12 warmups with predecessor-matching hashes, emitted/
sampled counts and stopping reasons. The 32-token cases emitted six and sampled
seven tokens with EOS; the 128-token cases emitted/sampled 16 with a length stop.
Off speculative work was zero and every on route proposed and verified real work.
All six groups deferred before any of the intended 48 measured arms. Admission
waits totaled 720.367 seconds, 1,455 global decompressions and 45 pageouts, with
zero compression or swapout increases. The pageouts occurred in 128/depth-one
admission. There were no retries, threshold changes or removed rejected groups.

Sampled counter-only quiet streaks reached 12.190 seconds for 32/depth-three and
11.915 seconds for 128/depth-two. Every group also breached host CPU/GPU gates;
the longest combined streak was 6.736 seconds, below ten. Global paging alone
does not explain every admission failure. All 658 worker snapshots were valid,
with zero task decompressions throughout the observed span and zero task page-ins
during every admission wait. Task gauges do not establish full GPU/driver
accounting or the cause of residual global events.

The 2,516 persistent global observations span 833.694 seconds and contain 1,837
decompressions and 321 pageouts, including warmups and inter-group observations.
Their scope differs from the admission total. Additional accessible task
comparisons contain 518 positive task decompressions across wider sequential
admission endpoint intervals, with explicit lifecycle changes and one denied
task read. GPU-client counters show application/graphics activity and positive
`stark-generate` basename deltas. The latter name aggregate cannot distinguish
the owned worker from another worker with the same basename. Neither diagnostic
permits subtraction, ownership percentages or a complete causal attribution.

The final host monitor joined with 1,383 samples and zero failures. The owned
worker exited zero on stdin EOF and was reaped. Startup available memory was
82,959,859,712 bytes; sampled peak owned RSS was 757,071,872 bytes, within the
registered limits. RSS does not establish Metal residency. Whole-session CPU
includes the parent and reaped worker/observer children; it cannot isolate
monitor cost from inference. The separate failed control remains failed.

Keep record bindings opt-in and MTP default zero. No depth ranking, speed benefit,
broader context, prefix reuse, other-artifact qualification, product admission or
installation claim follows from the control. Complete eligible matched groups
remain required for a performance decision.

## Evidence bindings

The [benchmark record](../benchmarks/2026-11-13-edge0-host-interference.md#evidence-bindings)
contains the source/helper/registration hashes. The completed performance report
SHA-256 is `e8b935b8e0309eefb72591c04dbdd9e4847507114ac863b6633a487a242dfe1d`.
Detailed native receipts remain private and ignored. Preserve the failed control
and all predecessor evidence; this attempt establishes no MTP timing ranking.
