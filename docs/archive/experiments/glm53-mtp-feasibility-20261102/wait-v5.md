# MTP Head FFN Wait Policy v5 (2026-11-02)

## Result

Keep the existing blocking completion wait. Four prospective ABBA runs passed their guards, but status polling did not demonstrate a gain. The two-run poll mean was 19.925460 ms per head versus 18.701309 ms for blocking, a 6.55% higher wall mean. B1 was an outlier: it was slowest for both the head and the serial target control. With two runs per policy and visible sequence drift, these results do not establish a stable regression magnitude or a causal completion-wakeup delay.

Both policies retain the final blocking `wait_until_completed` fence, status checks, finite checks, and selected-owner checks. The test-only B policy polls for completion with a 20-microsecond sleep while pending, then still uses that same fence. The production wait policy and engine defaults were not changed. The experiment measured a head component; it did not establish runtime speedup.

## ABBA measurements

Each run contains 24 measured heads. Times below are milliseconds per head, except the serial target control, which is per comparison. The inclusive `wait-return` interval contains polling (for B), the final blocking fence, status and GPU-counter accounting. `Commit-call` and `wait-return` are diagnostic subspans within FFN submit/return; wait-return also overlaps enclosing head intervals. Do not add these values to the enclosing phase totals.

| Run | Policy | Head wall | Head GPU | FFN commit-call wall | FFN wait-return wall | FFN GPU | Serial target wall | Serial target GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | Blocking | 18.875759 | 2.578835 | 0.003455 | 15.988342 | 1.294646 | 32.438536 | 29.008363 |
| B1 | Poll | 21.304609 | 2.619266 | 0.004085 | 18.241434 | 1.317255 | 33.057045 | 29.410868 |
| B2 | Poll | 18.546311 | 2.576675 | 0.003083 | 15.504422 | 1.302835 | 32.332050 | 29.118802 |
| A2 | Blocking | 18.526859 | 2.600113 | 0.003380 | 15.556101 | 1.309203 | 32.393377 | 29.166434 |

| Two-run policy mean | Head wall | Head GPU | FFN wait-return wall | FFN GPU | Serial target wall | Serial target GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blocking | 18.701309 | 2.589474 | 15.772221 | 1.301924 | 32.415957 | 29.087398 |
| Poll | 19.925460 | 2.597970 | 16.872928 | 1.310045 | 32.694548 | 29.264835 |

Polling increased mean head wall by 6.5458% and mean head GPU time by 0.3281%. The pairwise sequence changes were B2 versus B1 −12.9470% and A2 versus A1 −1.8484%. Commit-call wall time was about 0.003–0.004 ms. The recorded GPU time comes from the completed-command counter; a zero commit-call counter delta does not show that device work did not overlap the commit call.

The large wall/GPU gap remains within the inclusive wait-return phase under both policies. This measurement does not distinguish pre-GPU resource scheduling or residency from completion notification latency, and it does not isolate a causal wake delay. Retain blocking. At this v5 checkpoint, the test-only v6 same-view repeat diagnostic had not run and its result was unknown.

## Alignment, guards, and memory

All four runs used the same frozen source, v5 binary and driver, launcher, corpus, model identity, and protocol. Every guard passed: source before and after, four GGUF identities, six metadata-file identities, weight identity, wrapper policy selectors, and result schema. All four runs had 23/24 draft agreements with the same 8/8, 7/8, 8/8 case pattern, the same target continuation hash, and identical memory gauges. Each had eight full-target alignment rows, eight draft-cache alignment rows across three chunk groups, and zero target rollback calls. Across the four runs this is 96 comparisons, 92 agreements, 32 rows of each alignment type, 132 selected-FFN calls, and 132 exact owner-release checks.

The recorded memory values were identical across runs:

- Selected registration peaked at 73,269,248 bytes over at most 24 views. Each run recorded 33 selected calls and release checks, and zero selected bytes live at return.
- The static cache peaked at 609 entries and 6,573,162,496 bytes. Resident expert-bank peak was 99,035,873,280 bytes; residency allocation peak was 105,609,035,776 bytes; sampled Metal allocation peak was 106,012,295,168 bytes.
- Against the 109,676,289,225-byte cap, observed target-required peak was 108,692,065,600 bytes. The probe reservation was 108,767,504,704 bytes, leaving 908,784,521 bytes.
- A prospective production-overlap projection reserves 109,304,375,616 bytes and leaves 371,913,609 bytes. This is a projection, not runtime admission; it does not account for admitting a verifier, rollback state, or production scratch heap.

Overlap, prefill MMA, and scratch heap were disabled for this test-only component study. No production memory, rollback, or throughput claim follows from the projection.

## Validation and source provenance

The model-free parser/schema fixture suite passed 22 cases (six accepted forms and 16 rejected forms). One TLS head-scope/restoration unit passed. The v5 release `--no-run` build completed in 1m58s using `nightly-2026-10-01`; it did not run native tests or execute a model/GPU. The sealed engine binary SHA-256 is `9efd92b9a8bb1e6c7de3943e637f51094b3ade5d344d16f4555979de8a9f5ccd`.

The source freeze is based on `379e5c79d1cfae249e947740732b783b1a6fec82`; its manifest SHA-256 is `8b17a190cc950278d62bc0a224132777652086d56fd95391e0f8e75600c876c9`. The frozen source file hashes, upstream reference revisions, engine binary, driver, launcher, and per-run result/guard hashes are listed in [wait-v5.json](wait-v5.json). The pinned llama.cpp and vLLM comparison revisions are `1b6935b227b9f9aa89deea8016c0e208f547446f` and `3b6eeb060621eba88deb74a146039291a1afbc42`.

The sealed private evidence archive has SHA-256 `8586da4aaa960f27e9d446aef6e8cd1bbe88ab566cc57da003c2697309589a98` (32,135,248 bytes; 47 members). Only its canonical `prototype.patch` was extracted here as [wait-v5.patch](wait-v5.patch). The patch SHA-256 is `d5a355a884f5c291baf721bb432b66826966abf9c46b81ca09e209ca17be4511`; it declares all nine frozen source paths with four new-file sections. The canonical patch was generated with `--no-ext-diff` and `--no-textconv`; the sealed manifest reports forward reconstruction of all nine source hashes and a successful reverse check. The public file contains source and authored fixed fixtures only, with no model binaries, runtime receipts, or captured run payloads.

The same guarded aggregate receipt is preserved by hash as `97e0609aaf03588ec118548d575034b33f8b07f041994087239e7617e0f81182`. No wait-policy change is recommended from this study.
