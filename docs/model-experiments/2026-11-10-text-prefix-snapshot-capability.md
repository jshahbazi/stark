# 2026-11-10 text prefix snapshot capability

## Attempt record

| Field | Value |
| --- | --- |
| Attempt ID | `20261110-PREFIX-01` |
| Scope | Worker dispatch for ordinary text models whose decoder supports exact prefix snapshots but cannot rewind its completed state |
| Status | Implemented; synthetic engine/worker controls passed, production artifact qualification pending |
| Candidate | Runtime commit `b41201523b547ea646f1378104319e7f501b3ac7`; identities in final triage record |
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

- At registration only source inspection had been performed. The final checks
  below supersede that initial validation status.
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

## Final controls

At `b4120152`, all eight existing engine text-prefix controls passed on the
CPU/Metal test target, including the new snapshot-aware context planner budget
control. The worker test passed on two owner-private synthetic mixed-affine
128-dimensional, 256-vocabulary, two-layer artifacts: DeltaNet plus full
attention, then the same target with a 128-token sliding ring. Greedy and
stochastic seeds 0/7/42 compare cached and fresh output exactly, with 1,024
reused rows at the eligible boundary. Divergence, context/scope changes,
scope-less and shorter requests, and explicit clearing behave as required.
The cold context planner now charges the optional snapshot as well as state;
its chosen capacity is checked against actual snapshot-enabled request bytes.

The synthetic target weights hash to
`d22280aec723064cee465ee077f2243cdaca099b24ab76aa9d5ef037ea476ae8`.
Full-attention and ring configs hash to
`cac408f6afd7291fecd7ceb04b9c2c239f6a412f1cdddb7ef095138462088a0d`
and `74ef85cf5a36e1a50940a3bcb0ff41f127a78072131b7d12a4c5262c24998ac1`.
The original setup retained undeclared MTP payloads and failed loading; corrected
fixture construction and the failure history are in the
[final source-bound record](../archive/runs/2026-11-10-model-family-triage.md).
No production Qwen/Gemma artifact or image parity/performance claim follows.
The routing implementation is adopted on this branch; production artifact
qualification and timing remain pending.
