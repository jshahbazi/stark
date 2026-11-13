# 20261113-EDGE0-12: paging investigation and unchanged-screen rerun

## Identity and scope

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `7d5205f5`; inference code `bdf3b62e`, unchanged release worker |
| Predecessor | [Attempt 11](2026-11-13-edge0-mtp-batched-worker.md), [ineligible matched run](../benchmarks/2026-11-13-edge0-worker-mtp.md) |
| State / result | Investigation complete; all six rerun groups deferred at strict admission, zero measured arms |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| Promotion | Default off; no timing promotion before every existing gate passes |

The owner requested paging/decompression investigation and a matched rerun
without weakening the screens. Inference arithmetic, weights, artifact, context,
MTP implementation and default are unchanged. All new native probes, fixtures
and detailed runtime receipts remain ignored; publish aggregates and hashes only.

## Questions and preliminary findings

Attempt 11 recorded 995 decompressions and 363 pageouts across measured arms.
Decompressions affected 23 of 24 off arms and 23 of 24 on arms. Counts normalized
by arm wall time were approximately 2.25 pages/second off and 2.27 on. That
association does not attribute global activity to either route or another process.
The earlier report lacks absolute counter endpoints and actual interval bounds.

The old quiet-admission rule checked CPU/GPU/thermal conditions, not paging.
`vm_stat` endpoints and the legacy observer also launch subprocesses. A bounded
model-free ABBA control compared ten-second idle / old-monitor / old-monitor /
idle epochs. Direct `host_statistics64` and `vm_stat` deltas agreed throughout;
decompression counts were 515 / 126 / 29 / 68, with zero compression, pageout
and swapout increases. All eight endpoint bracket comparisons passed.
Decompression activity was observed without a live evaluation worker. This
experiment does not establish observer causation or the identity of the source.

The monitor epochs had process/child page-in and I/O overhead. Those counters
are additive diagnostics and cannot attribute or excuse global decompressions.
Counter page units come from the local SDK; the probe recorded 16,384-byte pages.
These are not disk-byte measurements or counts of distinct pages.

## Unchanged mandatory gates

Retain all predecessor thresholds and reject missing, malformed or reset counters:

- Each measured arm requires **zero** Pageouts, Swapouts, Compressions and
  Decompressions. No baseline subtraction, rate allowance, worker attribution
  exemption, counter removal or tolerance is allowed.
- External CPU at most 150%, external model CPU at most 10%, thermal at most
  fair, no compiler activity, no sampling failures or interval gaps over two
  seconds. Preserve crossing-boundary host observations.
- Quiet admission retains external CPU at most 150%, one process at most 80%,
  external model CPU at most 10%, idle GPU at most 5% and thermal at most fair.
  Add a fresh, contiguous ten-second all-four-counters-zero window; stale
  observations do not advance admission.
- Verify host observation coverage at both request boundaries and final monitor
  termination. A late monitor failure invalidates timing; it cannot be hidden by
  an earlier arm screen.
- Preserve matching output hashes, emitted/sample counts, stopping reasons,
  real proposals on, zero speculative activity off, and every existing default
  gate. No selective arm removal or retry.

## Successor instrumentation and rerun

Use a persistent, warmed `host_statistics64(HOST_VM_INFO64)` reader with the same
four global lifetime page counters. Verify its ABI against the current SDK and
its output against bracketing `vm_stat` reads. Retain the legacy CPU, GPU,
thermal and compiler observer. The changed instrumentation has separate source
bindings and receipts; it does not replace the old evidence.

Record counter endpoints and monotonic times, actual host interval bounds,
boundary coverage, admission resets and final monitor status. Optional owned
process rusage is descriptive only. Do not close other applications, kill foreign
workers, purge memory, alter swap or wired-memory settings, or change power policy.

The intended matched matrix remains prompt lengths 32/128, context 512, greedy
seed 42, maximum 16 outputs with matching natural EOS, depths one through three,
one warmup per route, four balanced pairs per group: 12 warmups and 48 measured
arms. Loading/cold requests remain excluded. Stricter admission can defer a group;
record that outcome rather than calling unexecuted requests successful.

The same exact artifact/receipt and large-payload immutability assumption apply.
Reuse worker SHA-256
`056839034938df8d4ffa19ec55ee0be0923e4559004acbc688dc1035ff07ede2`.
Require at least 64 GiB available before launch, sampled owned RSS below 48 GiB,
180-second request and 1,800-second worker limits; serialize model processes.
Seal code, worker, corpus, observer and registration before launch and afterward.

The default-selection criteria remain at least 5% median paired decode/wall gain,
at most 5% prefill regression and 15% off wall spread, with complete eligible
groups at both prompts. Retain off if no depth qualifies. Prefix reuse, resident
MTP, wider context, quality and product admission remain outside this attempt.

## Results and disposition

Twenty-six deterministic private tests pass: native ABI/error/counter handling,
single-page rejection, stale observations, skipped busy rows, continuous host
coverage, crossing-boundary samples and failure propagation. Independent source
review identified and corrected skipped-row admission and malformed-observation
handling before sealing. The same worker, source manifest and artifact receipt
were verified before launch and after normal teardown. Independent review
rehashed all 2,182 sealed source files, the worker, corpus, private helpers/tests
and three SDK headers; no integrity drift was found.

The [separate rerun record](../benchmarks/2026-11-13-edge0-mtp-paging.md)
records **12 matching warmups, six admission deferrals and zero measured arms**.
Every group exhausted its preregistered 120-second admission limit. The sum of
the six admission windows was 33,588 decompressions, 690 pageouts, 513,776
compressions and zero swapouts over 720.262 seconds. These counts exclude
warmups and between-group activity. The largest recorded combined host/paging
quiet streak was 4.898 seconds; the receipt does not retain enough host detail to
isolate CPU/GPU/thermal contributions or establish a counter-only quiet maximum.

Across the earliest admission counter snapshot through the final snapshot,
including intervening warmups/gaps, the deltas were 545,760 decompressions,
773 pageouts, 516,960 compressions and zero swapouts. In particular, 511,490
decompressions occurred between the end of 128/two-draft admission and the start
of 128/three-draft admission. Global counters cannot attribute that change to
the warmup, worker state, shared GPU memory, observer or another process.

After the worker was reaped, a separate 20.155-second counter-only control
recorded 476 decompressions and zero changes in the other three counters, with
40 samples and a 0.505-second maximum gap. It launched no subprocesses or model.
Together with the pre-run idle controls, this confirms decompression activity
without a live evaluation worker. It neither excludes worker-related memory
churn during execution nor establishes the background source.

The final monitor stopped with zero failures and 1,380 samples. Worker EOF exit
was zero and reaping succeeded. Available memory before worker construction was
95,064,489,984 bytes; sampled peak owned RSS was 792,166,400 bytes, under the
48 GiB limit. RSS does not measure complete GPU residency. No inference source, arithmetic,
weight, model policy or default changed. No qualified timing or depth ranking
results from this attempt. Phase 5 performance qualification remains open;
default stays zero and the prior correctness/lifecycle evidence is preserved.

## Evidence bindings

Only aggregate records and hashes are published. Detailed native artifacts,
probes and fixtures remain private and ignored. Python was 3.13.13 with psutil
7.2.2 under the locked converter environment. SDK header digests and executable
identity are included in the private registration.

| Binding | SHA-256 |
| --- | --- |
| Model-free ABBA probe | `cbd5b7e43b0bf290c3c262e8d69e26997408edb25ad7143e5b099460111e6ca8` |
| Model-free ABBA report | `94b78b8a5b143c340ae0a72c52375428e163b942b5d5be62b48427d144d0847d` |
| Source manifest, 2,182 files | `3626c3b1a05bcd9ef5965cead2e168a8978963cc45888dc8f94cf273fd7912fd` |
| Rerun registration | `25fceabe165c667716457f6f95a23e21c6627dc20d029c0e5f35c10bc12a91c6` |
| Rerun aggregate report | `0725c92f366295394f674182c8dce572d157035fe919460f726a1d582818b6b7` |
| Rerun harness | `5d32eeccd7a1be2a3b25940259f95179a9817b35fc62d7e6249105b7a583bc3d` |
| Native counter sampler | `0740fb04e91d7391db9cefa55a03bac73a6bc19d74edecfb68cb33d9a5883244` |
| Additional strict screens | `1170c887479093319f4eaf3203559f3d38d3b98c5f025fd68ff4a421779755d4` |
| Sampler tests | `c1e066acce034ba881d992d42357d3187e4382b9783bced58515ff6b3df33027` |
| Strict-screen tests | `723e7c25fc942ffc19ba97735d97abcd1aa010ade8f76f66eafec9c25e17a3f0` |
| Post-exit control | `3d05569fc8f45ec7a1c18bb3cb9967d18938ea478437b63bd9d9d853e451b624` |
| Post-exit aggregate report | `1185ef513a74c6b4a06063f731553ce9336c9efeabdcd45b4b727db314f05146` |

## Remaining blocker

The host did not establish strict admission after any warmup pair. A further
matched performance run needs complete eligible groups under every existing
screen. Increasing paging allowances or subtracting the idle control would
invalidate that comparison. A successor observer should retain per-gate timeout
reason counts and host maxima to distinguish admission causes. If activity
persists, a separately bound in-worker self-task diagnostic could distinguish
worker decompressions from global activity; this attempt did not implement or
qualify that instrumentation. Other applications and OS memory/power policy were
left under the owner's control.
