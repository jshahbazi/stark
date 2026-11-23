# Inkling 2 Mini worker prefill command width and the cache on/off fault

## 20261123-INKWIDTH-01 · Worker selects the widest admitted prefill command width

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKWIDTH-01` / 2026-11-23 |
| Kind | Scheduling and memory admission in the worker |
| State and result | Adopted in the tree (coordinator change); prefix-reuse validation exposed a fault (INKWIDTH-02); re-validation after the fix queued. Not final. |
| Evidence scope | Engine width checks ([INKFASTPRE-04](2026-11-23-inkling-mini-prefill-round4.md#command-width-api-adopted)), worker unit tests, `scripts/validate_generation_prefix_cache.py` on both artifacts. No worker timing study of the width alone. |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`; not installed |

**Change.** In `crates/stark-generate/src/inkling.rs`, `source_command_rows` picks the widest source prefill command width that memory admits among 2,048, 4,096 and 8,192 rows. The base width keeps its original admission rule; a wider width must also leave room for the verifier, the proposal head and a maximum-context request. `InklingBackend::load` calls `Session::set_batched_prefill_width`. Snapshots stay aligned to 2,048 rows.

**Engine evidence.** 8,192-row commands measured 1.033× against 2,048 (8/8 pairs) and 4,096-row commands 1.021× (5/8) on the engine diagnostic; seed hashes are identical across widths. MLX-Serve's default chunk is 8,192 rows.

**Side effect.** Cancellation is checked between commands, so a cancel on a long prompt can take about 3.5 s instead of about 1 s.

## 20261123-INKWIDTH-02 · Cache on and cache off differed on the wide-command worker

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKWIDTH-02` / 2026-11-23 |
| Kind | Prefix reuse, correctness defect and fix |
| State and result | Fault found; fix adopted with unit tests; re-validation queued. Not final. |

**Finding.** `scripts/validate_generation_prefix_cache.py` on the wide-command worker: tokens matched the 2,048-row worker in 37 of 37 cases, but cache on against cache off differed in 34 of 37, on both the original and the Stark 8-bit artifact.

**Cause.** An unscoped request (no cache scope) ran its whole bulk in 8,192-row chunks from position 0, which merged the remainder after the last 2,048-aligned boundary into a wide command. A scoped request runs that remainder as its own command. The prefill router's split-K rounding differs for commands of at most 1,024 rows and above, the one remaining width-class dependence ([INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md)), so the two schedules produced different state.

**Fix.** `unscoped_source_split`: an unscoped request splits at the same boundary as a scoped one, without taking a snapshot. Unit tests added. Re-validation of the fixed worker was still queued when last reported.

**Separate mismatches, not yet attributed.** The same validator reported 5 reuse-rule mismatches that are believed to predate wide commands: the worker advertises 1,024-token reuse chunks while source prefill aligns to 2,048 rows with a 4-token suffix, and it reuses a snapshot across a change of drafting mode where the validator's rule expects none. A control run on the 2,048-row worker is queued. If it confirms them, they are a prefix-reuse qualification gap in the Inkling port, open since before this session.

## Follow-up history

- 2026-11-23: both changes (`source_command_rows` and `unscoped_source_split`, with their unit tests) are in commit `2a1d72a2`. The committed tree passed the coordinator's commit-check gates (167 worker tests; drafting on equals off at 73 to 16,000 on both artifacts; worker smoke on both artifacts; receipts in `runtime/commit-check/`). The prefix-reuse re-validation of the fixed worker and the 2,048-row control are not reported in the sources and remain queued.
- 2026-11-23: a paired run of the same binary at 8,192- against 2,048-row commands measured 1.052× (1.042 to 1.062, 8/8, identical state hashes) on the original artifact at 8,800 tokens, which supports the worker's choice of the widest admitted width ([INKFASTPRE-04 follow-up](2026-11-23-inkling-mini-prefill-round4.md#sources-located-after-these-follow-ups-were-written-2026-11-23)). Wide commands also let prefill use the gathered NVFP4 kernel, which runs only on commands of 768 rows or more ([INKFASTPRE-07](2026-11-23-inkling-mini-gathered-nvfp4.md)).
