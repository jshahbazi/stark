# 20261113-EDGE0-13: worker memory diagnostics and screened MTP rerun

## Identity and scope

| Field | Value |
| --- | --- |
| Owner / authorization | 2026-11-13 / Codex; requested worker memory diagnostics and matched rerun |
| Scope | Local non-commercial evaluation only |
| Baseline | `5ace423e`; inference arithmetic from `bdf3b62e` retained |
| Predecessor | [Attempt 12](2026-11-13-edge0-mtp-paging.md) |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| State / promotion | Diagnostics validated; six rerun groups admission-deferred, zero measured arms; default off |

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

The [Apple task implementation](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c)
and [fault accounting](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_fault.c)
explain the limits: task decompressions are charged execution events, not a page
ownership inventory. Compressed bytes are an accounting balance; resident,
virtual and footprint gauges have different scopes and do not establish complete
Metal residency or model working-set size. The two native flavors are sequential.
Diagnostic endpoint deltas can include query overhead.
These explanatory sources are pinned to XNU
`f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`; no exact running-kernel source match
is claimed. The actual local SDK header digests are separately sealed.

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

The [separate rerun record](../benchmarks/2026-11-13-edge0-worker-memory.md)
is bound to implementation commit `590b185d`. All 12 warmups matched the
predecessor's useful-work hashes/counts/stops, with zero speculative activity off
and real proposals on. All six groups exhausted strict admission, before any of
the intended 48 measured arms. No retry or threshold relaxation occurred.

All 686 worker snapshots had valid VM/events statuses and monotonic counters.
Warmup endpoint deltas totaled 2,563,986 decompressions charged to worker-task
execution; the six admission waits totaled three. The separately
bracketed global admission counters totaled 95,080 decompressions, 2,724,766
compressions, 1,557 pageouts and zero swapouts over 720.322 seconds. Sampled
counter-only streaks never exceeded 1.203 seconds; combined host/paging streaks
never exceeded 1.116 seconds. Every window also had CPU/GPU threshold breaches;
thermal remained nominal. These outcomes establish worker-execution churn during
some warmups and continuing global activity during admission, without a page
ownership inventory or a complete attribution of GPU/driver work.

Warmups always run off before on, and loading/idle history differs between them.
Their memory deltas cannot establish an MTP route ranking or timing benefit.
Task/global endpoint intervals differ and must not be subtracted or used for an
attribution percentage. Activity within query brackets is not proof of observer
causation; this attempt has no diagnostics-disabled matched control.

The monitor stopped with zero failures and 1,392 samples. Worker stdin EOF exit
was zero and reaping succeeded. Available memory before launch was
92,184,526,848 bytes; sampled peak owned RSS was 827,342,848 bytes. Task gauges
and RSS are not complete Metal residency. Independent review regenerated the
2,184-file source inventory and checked all sealed bindings and trace deltas;
no discrepancy was found. The predecessor records remain immutable. Performance
qualification remains open, default stays zero, and no product admission follows.

## Evidence bindings

Detailed native receipts, diagnostics, ABI fixture and observer tests remain
private and ignored. Python was 3.13.13 with psutil 7.2.2 in the locked converter
environment. Six SDK headers, the Python executable and the golden predecessor
report are bound inside the registration.

| Binding | SHA-256 |
| --- | --- |
| Source manifest, 2,184 files | `1742e9532a5828c3c546b9e7234582f095ac03e28a48de29c6699a72b4629dbb` |
| Registration | `e00bfb86cd66fdb1527cc1c74fb5f179ccd1047a7c8301341a08d2b30cf7d786` |
| Aggregate report | `e3bd3459b821e9f3f72376074a4edbb29344fc9074c9ee5dac08cff9e92e2634` |
| Harness | `6bfbd77325e11dace60ad9f515fc028d6d96cfdc0213544f86ff77904874bd2e` |
| Native paging sampler, unchanged | `0740fb04e91d7391db9cefa55a03bac73a6bc19d74edecfb68cb33d9a5883244` |
| Worker observer | `d5bee18cb63dfe1c980d8495dbdedf3a2a0ef248f2ffeae99328ea8138049de0` |
| Strict screens and reason counters | `e9d6658601b10adc778be8fe7648cdb946b929720bf9c841af28ec09bda049b0` |
| Worker observer tests | `566969b6e25cfbf4294220c2eabbc33da469f983a6415d20fc6019693cd870e5` |
| SDK ABI fixture | `6d73d40e59f8750a7c192b18336a97075b0f66d2cf5111c7f085541e68d6f499` |

## Remaining qualification work

An eligible timing comparison still requires a host that reaches the unchanged
combined quiet window after warmups. These diagnostics support separating
worker-execution churn from idle global activity; they do not identify particular
pages, allocation classes, GPU-driver accounting or every competing workload.
Further diagnosis could correlate explicit Metal allocation telemetry with the
self-task gauges in a separately registered run. Any memory-policy or residency
change needs its own correctness and matched-performance evidence. Keep the
512-token context, unavailable Edge0 prefix reuse and local opt-in boundaries.
