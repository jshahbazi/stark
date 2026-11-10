# 2026-11-10 text prefix snapshot capability

## Attempt record

| Field | Value |
| --- | --- |
| Attempt ID | `20261110-PREFIX-01` |
| Scope | Worker dispatch for ordinary text models whose decoder supports exact prefix snapshots but cannot rewind its completed state |
| Status | Implementation in progress; tests and model-bound validation pending |
| Candidate | Current `codex/model-family-correctness` worktree; final commit and executable identity pending |
| Checkpoint | None used for implementation review |
| Baseline | Existing worker admitted snapshot-capable image wrappers and dense-MTP text, but did not advertise or route snapshot-only ordinary text models through capture/restore |

## Observed gap and change

The engine distinguishes `prefix_reuse_supported()` (rewind of the original
state) from `text_prefix_snapshot_supported()` (capture and restore of a
completed text-prefill boundary). The latter was crate-private, so the worker
could not select the already implemented snapshot path for ordinary text model
wrappers. This attempt exposes that capability and routes snapshot-only text
models through snapshot memory admission, restore, capture and pending
publication. Rewind-capable full-attention models retain their existing path.
Snapshot entries use the text execution-graph key.

## Evidence and limits

- Source inspection only; no build, tests, checkpoint load, worker run or timing
  has been performed for this attempt.
- Existing engine tests cover snapshot/restore state behavior for synthetic
  attention models, including ring state. They do not exercise this worker
  routing change.
- Required next checks: model-free worker tests for snapshot-only capability
  routing and publication/invalidation; engine unit tests for rewind and ring
  restore; then checkpoint-bound worker validation for each promoted family and
  separate matched timing before making a performance claim.
- No model, host, or speed qualification transfers from Gemma/Qwen image
  artifacts to text-only checkpoints.

## Promotion decision

Pending source review, worker tests, and artifact-bound validation. This change
establishes a routing capability only; it does not claim qualification or a
latency improvement.
