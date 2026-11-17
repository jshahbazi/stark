# Flash Next 12K midpoint capture attribution

**20261117-FLASHSPEED-14 · November 17, 2026**

The [single-midpoint A/B](2026-11-17-flashnext-midpoint.md) saw a +5.51% median paired first-token change on fresh 12,288-token inputs. The median changed direction with launch order. This diagnostic measures the work that scoped requests add on the same warmed worker and uses the two unchanged prefill chunks to track drift.

Both arms use the pinned retained-vision Lily Q4-XL artifact at revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` with 4,096 physical rows, 40,960 context, two exact MTP drafts, and seed 42 at temperature 0.6, top-k 20 and top-p 0.95. The stock worker takes an aligned snapshot after token 8,192. The midpoint worker captures its snapshot at token 10,240 inside the third physical pass. Alternate unscoped requests, which create no snapshot, with unique scoped requests on each resident worker. Require zero reused tokens, correct snapshot counts, three full physical prefill passes with valid timestamps, and identical input, generated output, sample count, finish and MTP work for each comparison. Retain the scope order and warmup exclusions. Use four launches in baseline/midpoint/midpoint/baseline order, with both scope orders, and keep one full model resident at a time.

Measure host snapshot allocation and zeroing, midpoint n-gram history, the stock synchronous copy and wait, and GPU timestamps for the final pass and the first two passes. Keep existing host staging, encoding and wait counters and system paging alongside the new measurements. Compare scoped minus unscoped within each launch before comparing the two workers. The final pass contributes all normal prefill and MTP catch-up work as well as capture when it is enabled. Attribute a GPU signal only when its direction repeats and it is larger than changes in the untouched passes. Treat telemetry and paired medians as descriptive. The prior 5% decision gate stays with the uninstrumented timing comparison. Restore the installed Lily worker and verify the local API after the run.

## Execution and controls

The four launches on the M5 Max, 128 GiB, macOS 26.7.2 (25G332) on AC used one worker built with Rust 1.98.1 from `14475c65d5aa0d2d2b3e2b84c546afdced16580e`. Its SHA-256 was `1bb3f519aa890473d32c8974481074cca28b7cf16840b6cf07db50d2da5c6dc5`. The artifact SHA-256 was `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. Both workers used the existing opt-in native aggregate profile output. The new tracker reads the Metal command's existing feedback and the baseline blit's GPU span. Its wall time includes encoding, submission and fence completion.

Each launch excluded its first unscoped 12K request and then ran two repetitions of three repository-source coding fixtures. The sequence alternated scoped/unscoped in the first launch pair and unscoped/scoped in the second. All 48 measured requests reported zero reused tokens. The 24 scoped requests used distinct scopes and each captured exactly one 115,662,848-byte snapshot of the expected type. All 24 scoped/unscoped pairs and 12 cross-arm comparisons matched inputs, emitted output, finish, sample counts and two-draft MTP work. All 144 profiled prefill GPU spans and all 12 aligned blit GPU timestamps were present. The guarded release build, repository policy and model-free runner checks passed. The aggregate receipt SHA-256 is `b4306e2a0c6b4fc3d39a0f44645f7701dae2bac2064ab651407462e234ed4190`.

## Results

All values are medians of paired scoped-minus-unscoped differences. Positive numbers indicate more time for the scoped request. The untouched row sums the GPU spans of the first two physical passes. The baseline snapshot copy runs separately between its second and third passes.

| Component | Stock aligned snapshot | Midpoint snapshot |
| --- | ---: | ---: |
| CPU snapshot allocation and zeroing | +7.18 ms | +7.63 ms |
| Synchronous snapshot blit wall time | +6.47 ms | — |
| GPU time inside the stock blit | +1.06 ms | — |
| Final physical pass GPU time | +42.21 ms | −96.24 ms |
| First two physical passes GPU time | −133.05 ms | −101.17 ms |
| Native first-token time | −48.44 ms | −92.50 ms |

Direct midpoint allocation took 6.20–14.71 ms across the 12 captures. Its paired incremental allocation cost over stock was **0.55 ms** at the median, with order medians of 0.55 and 0.51 ms. This is too small to explain the roughly 335 ms median paired cold gap observed at 12K in the uninstrumented A/B. The midpoint path also avoids the stock synchronous blit. Its additional history step and change in host encoding were under 1 ms at the median. Median n-gram staging decreased by 7.93 ms when comparing the two scope effects.

The final GPU pass's scoped-minus-unscoped change on the midpoint worker reversed from **−229.05 ms** to **+112.75 ms** across launch orders. The first two unchanged GPU passes moved by **+257.30 ms** and **−200.84 ms** on the same pairs of launches. Comparing the complete midpoint scope effect against the aligned scope effect gave +171.83 ms in the first order and −230.02 ms in the second. System paging also varied across controls. Pageouts for stock scoped/unscoped windows were 1,211/727, and for midpoint 293/345. No request window recorded a swapout. Paging is system-wide, and repeated fixtures share worker launches.

**Conclusion.** Allocation adds about half a millisecond over stock. The profile did not reproduce a stable slowdown localized to the capture-bearing GPU pass. GPU timing also varied substantially on passes with identical work, so the earlier cold gap remains unattributed. The data does not support changing snapshot allocation or the GPU kernel based on that gap.

At restoration the installed app showed an operation failure while its worker was stopped. Starting Lily returned the app to Ready. Authenticated health, model listing and an MTP-enabled completion returned HTTP 200. The campaign's unified-log check returned diagnostic-worker process events.

[Reproducer](../archive/experiments/2026-11-17-flashnext-speed/midpoint_profile.py) · [Aggregate results](../benchmarks/data/flashnext-midpoint-attribution-2026-11-17.json)
