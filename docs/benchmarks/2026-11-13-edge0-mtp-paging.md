# Edge0 worker MTP rerun: strict admission deferred

Attempt [20261113-EDGE0-12](../model-experiments/2026-11-13-edge0-mtp-paging.md),
2026-11-13, local non-commercial evaluation only. All six groups failed to reach
strict quiet admission: **12 warmups completed, zero measured arms ran, and no
timing qualified**. MTP remains off by default. This is a bounded rerun attempt,
not a completed 48-arm comparison.

## Matched conditions and unchanged screens

The same release worker, prepared artifact and 128 GiB M5 Max host as the
[predecessor](2026-11-13-edge0-worker-mtp.md) were used. Source baseline was
`7d5205f5`, inference code `bdf3b62e`. Worker SHA-256:
`056839034938df8d4ffa19ec55ee0be0923e4559004acbc688dc1035ff07ede2`.
Artifact: `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`;
receipt SHA-256:
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
The prior complete payload receipt and immutability assumption were reused;
no new large-payload rehash is claimed.

One loaded JSONL worker, original weights/F32 policy, virtual Metal mappings,
context 512, prefix reuse disabled, greedy seed 42, prompt lengths 32/128,
maximum 16 outputs with matching natural EOS, MTP depths one through three.
The intended matrix was one warmup per route and four balanced pairs per group:
12 warmups plus 48 measured arms. There were no arm retries or selective removals.

All predecessor measured-arm screens remain mandatory: zero pageout, swapout,
compression and decompression deltas; external CPU at most 150%, external model
CPU at most 10%, thermal at most fair, no compiler, no monitor failure and no
sample interval above two seconds. All default-selection criteria are unchanged.
Quiet admission retains CPU/GPU/thermal thresholds and adds ten fresh contiguous
seconds with zero change in all four paging counters. Each group has a
120-second admission deadline. A persistent warmed native reader replaces
`vm_stat` launches; the legacy CPU/GPU/thermal observer is retained. Native and
CLI counter equivalence was checked separately. Additional request-boundary
coverage and final-monitor checks fail closed; no baseline is subtracted.

## Admission outcomes

These are global page-counter deltas **during admission windows**, not request
timings, total session counters or process-attributed events. Page size was
16,384 bytes; counts are not distinct-page or disk-byte measurements.

| Prompt / on drafts | Wait s | Decompressions | Pageouts | Compressions | Swapouts | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32 / 1 | 120.054 | 14,995 | 0 | 0 | 0 | Deferred |
| 32 / 2 | 120.039 | 2,417 | 88 | 0 | 0 | Deferred |
| 32 / 3 | 120.019 | 3,783 | 300 | 0 | 0 | Deferred |
| 128 / 1 | 120.046 | 4,072 | 202 | 0 | 0 | Deferred |
| 128 / 2 | 120.053 | 7,116 | 100 | 513,776 | 0 | Deferred |
| 128 / 3 | 120.051 | 1,205 | 0 | 0 | 0 | Deferred |
| Sum | 720.262 | 33,588 | 690 | 513,776 | 0 | Zero measured arms |

The maximum recorded **combined host-and-paging** quiet streak was 4.898 seconds.
Timeout receipts do not retain host maxima/reason counts, so they cannot isolate
CPU/GPU/thermal contributions to deferral or establish a counter-only quiet
maximum. Warmup and between-group deltas are excluded from this table; the
[attempt record](../model-experiments/2026-11-13-edge0-mtp-paging.md) describes
the larger enclosing interval and separate idle controls without causal claims.

The six off/on warmup pairs matched output hashes, emitted/sample counts and
stopping reason, also matching across depths at each prompt length. The short
prompt emitted six tokens and sampled seven including EOS; the long prompt
emitted/sampled 16 and stopped at length. Off speculative counters were all zero;
on routes had real proposals and sufficient target verification. Warmup timings
are excluded and establish no ranking.

The monitor stopped with zero failures and 1,380 observations. Worker EOF exit
was zero and reaping succeeded. Sampled peak owned RSS was 792,166,400 bytes;
available memory before construction was 95,064,489,984 bytes. These satisfy the
registered limits but do not measure complete GPU residency. Source, artifact
receipt, observer, helper and executable bindings matched after teardown.

## Evidence and disposition

| Binding | SHA-256 |
| --- | --- |
| Registration | `25fceabe165c667716457f6f95a23e21c6627dc20d029c0e5f35c10bc12a91c6` |
| Aggregate report | `0725c92f366295394f674182c8dce572d157035fe919460f726a1d582818b6b7` |
| Harness | `5d32eeccd7a1be2a3b25940259f95179a9817b35fc62d7e6249105b7a583bc3d` |
| Source manifest, 2,182 files | `3626c3b1a05bcd9ef5965cead2e168a8978963cc45888dc8f94cf273fd7912fd` |

Twenty-six deterministic private sampler/screen tests passed. Independent source
and receipt review found no remaining eligibility or binding issue. All detailed
artifacts remain ignored; only aggregates/hashes are published. Prior correctness
and lifecycle qualification remains unchanged. Performance qualification requires
a further complete, eligible matched run. No throughput claim, depth selection,
prefix reuse or product admission follows from this attempt.
