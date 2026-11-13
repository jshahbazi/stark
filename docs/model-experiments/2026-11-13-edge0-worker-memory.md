# 20261113-EDGE0-13: worker memory diagnostics and screened MTP rerun

## Identity and scope

| Field | Value |
| --- | --- |
| Owner / authorization | 2026-11-13 / Codex; requested worker memory diagnostics and matched rerun |
| Scope | Local non-commercial evaluation only |
| Baseline | `5ace423e`; inference arithmetic from `bdf3b62e` retained |
| Predecessor | [Attempt 12](2026-11-13-edge0-mtp-paging.md) |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| State / promotion | Implementation and preregistration in progress; default off |

Attempt 12 deferred all six groups before measured work. Its global counters
could not attribute the large compression/decompression changes around warmups,
and timeout receipts lacked host-gate reason counts. The owner authorized
worker-level diagnostics and a separately bound rerun with every screen retained.

## Instrumentation contract

Add an operator-only `memory_diagnostics` operation under
`STARK_WORKER_MEMORY_DIAGNOSTICS=1`; normal workers have no task-memory sampling.
The persistent sampler reads only its own task via Darwin `TASK_VM_INFO` rev5
and `TASK_EVENTS_INFO`. It accepts no foreign port, reports only aggregate
counters/gauges and fixed statuses, and preserves inference and prepared-image
state. Strict parsing and output-failure invalidation remain in force.

Local SDK layout checks bind the packed four-byte ABI: 87 VM words, signed
decompressions at byte 344, physical footprint at byte 144, eight events words,
page-ins at byte four. Kernel errors, short replies, negative/saturated counters
and cumulative regressions never produce a fabricated zero delta. Returned word
counts are retained. Per-task counter saturation is distinct from global
64-bit lifetime VM counters.

The [Apple task implementation](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/task.c)
and [fault accounting](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_fault.c)
explain the limits: task decompressions are charged execution events, not a page
ownership inventory. Compressed bytes are an accounting balance; resident,
virtual and footprint gauges have different scopes and do not establish complete
Metal residency or model working-set size. The two native flavors are sequential.
Diagnostic query costs are included in diagnostic endpoint deltas.

The private successor records worker/global snapshots before and after every
warmup/measured request and approximately once per second during admission.
Queries are bounded to five seconds and bracketed with monotonic clocks. Task
and global read intervals differ; neither subtraction nor an attribution ratio
is allowed. Native unavailable/malformed diagnostics fail this attempt closed.
Host maxima and overlapping per-gate rejection counts are retained on admission
timeout, including a separate sampled counter-only streak.

## Unchanged gates and intended rerun

Keep all [attempt 12 mandatory gates](2026-11-13-edge0-mtp-paging.md#unchanged-mandatory-gates):
zero measured Pageouts/Swapouts/Compressions/Decompressions; unchanged CPU,
GPU, thermal, compiler, monitor failure/gap limits; fresh contiguous ten-second
combined quiet admission, 120-second group deadlines; both request boundaries
covered; final monitor joined. Diagnostic reads cannot exempt any global event.
No baseline subtraction, threshold change, selective arm retry, application
closure or OS memory/power policy change is authorized or planned.

The exact artifact/receipt and large-payload immutability assumption are reused.
Use a separately built Rust 1.98.1 worker, same source weights/F32 policy, virtual
Metal backend, context 512 and no prefix reuse. Prompt lengths 32/128, greedy
seed 42, maximum 16 outputs with matching natural EOS, drafts one through three,
one warmup per route and four balanced pairs per group: 12 warmups, intended
48 measured arms. Compare useful-work hashes/counts/stops against the predecessor
as well as within each group. Retain 64 GiB minimum available before launch,
48 GiB sampled owned RSS, 180-second request and 1,800-second worker limits.
Serialize loaded model execution and exclude builds/probes during timing.

All source, worker, corpus, sampler, observer, helper/test, SDK header and
registration identities are sealed before launch and checked after teardown.
Default selection still requires complete eligible groups at both prompts,
at least 5% paired wall/decode gains, at most 5% prefill regression and 15%
off wall spread. Keep default zero when no depth qualifies. Prefix reuse,
resident MTP, wider context, quality and product admission are outside this
attempt; the diagnostic operation creates no new state-affecting inference path.

## Results

Model-free worker validation passed: 161 native worker tests (six prior ignored
artifact cases; a new ignored live-task test is separate), the explicit live
self-task ABI/counter test, six startup-process cases, 38 private observer/screen
tests, production-target Clippy with warnings denied, formatting and repository
policy. The broader all-target Clippy invocation found the pre-existing test
placement lint in unchanged `hybrid/edge0.rs`; that invocation is not reported
as passing. Independent source review corrected diagnostic interval labeling,
per-query global-counter validation and post-admission freshness before sealing.

New release worker SHA-256:
`c9ab3bb7c469f56e7fee1954ebc3f778d293e35da978d03ae58469dfede8f846`.

Pending sealed model execution. Keep predecessor runs immutable; record actual
warmup, measured and deferred counts separately. Publish aggregates and hashes
only.
