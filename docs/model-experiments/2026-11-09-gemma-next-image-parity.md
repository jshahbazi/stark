# Gemma image-default continuation

Registered 2026-11-09 before new implementation or execution. Baseline is
`900752b4`, combining Gemma `7468dbc4` with reviewed Qwen PR #114 at `c18f1d23`.
The preceding Gemma cache evidence remains attached to its original binaries;
this integration has not yet been executed. No previous result qualifies the
new Gemma3 path or a Gemma4 assistant.

## 20261109-G3IP-01: Gemma3 4B text prefix restoration

State: implementation started; model execution not run. The worker currently
advertises no Gemma3 prefix support although its owning attention layers can
use the exact physical ring snapshots implemented for Gemma4. Add model-bound
capture/restore, scoped worker retention, snapshot memory admission, suffix-only
prefill and the existing actual-prefill-count guard. Preserve image arithmetic.

Qualification starts with the retained original-BF16 4B image artifact, whose
selected files and binary will be sealed before execution. Context 2,048,
nonperiodic input 1,792 tokens, eight greedy/seeded outputs, chunk boundary 1,024.
Require exact fresh/cache hashes, actual suffix counts, scope/context changes,
real image zero reuse, cancellation/error/reset and same-worker recovery. The
cached request must remain within context capacity, making duplicate replay
observable. No MTP capability is inferred for Gemma3.

The bounded speed control uses the same source graph and payload with tower
omitted/resident, and the retained previous text worker as a separate baseline.
ABBA loads, one warmup plus three measurements per load, context 2,048, same
short input, 64 greedy outputs, seed 42, MTP0, no prefix scope. Loading is excluded.
Record first token, decode and whole request; predeclare loss bound as greater
of 5% and twice baseline relative range. Stop on paging growth, insufficient
headroom or host interference. Larger/quantized artifacts remain unqualified.

## 20261109-G4AP-01: execute the official E2B assistant

State: companion converter implemented and CPU controls passed; assistant
payload conversion and native execution are not yet admitted.
Follow the pinned [assistant stages](../GEMMA4_ASSISTANT_PLAN.md), beginning with
typed companion conversion and target/KV handoff, then actual four-layer draft
execution and exact target verification. Preserve target embedding scaling,
normalization and ring/shared-KV ownership; do not enable unsupported partial
ring speculation. E4B requires its own complete intake and execution results.

No MTP capability is advertised before actual proposals and complete session,
cache, memory and image-transition controls pass. Serial proposal execution,
if used for a correctness component, does not qualify useful MTP performance.

The new converter binds exact pinned E2B/E4B source configuration hashes and
complete 50-tensor inventories, preserves source BF16 tensor bytes, validates
the entire I64 ordering as a vocabulary permutation, and binds the companion
to the selected target config/weight identity. Configuration reads are limited
to 1 MiB; the permutation plane is 2 MiB. It is a separate companion plan API,
not a catalog entry, chat model or enabled runtime head.

Eleven CPU converter checks pass under locked Python 3.13.13. Sparse fixtures
use independently fetched pinned source header inventories; they cover both
small-profile payload sizes and target binding, unsupported source, missing,
extra, wrong-dtype/wrong-shape tensors and duplicate/negative/out-of-range
ordering entries. Ruff passes. These are component controls, not a payload
conversion or proposal result.

Pinned Transformers 5.15.1 assistant source SHA-256 is
`a77e67673ce7d2248294e361c3c074c29d8d6a092655d7b27b95fc9777a2d0d1`;
candidate generator SHA-256 is
`411ea9b7c8062344ba6e2ad936c54927c19062fc198ee40f4b6ca50958672c1e`.
All four assistant layers borrow target KV and use a constant position within
a round; the assistant has no independent KV cache. Existing shared-layer
attention expects an uncommitted suffix. Passing the fully committed target
state directly would include an extra row; dedicated external-KV query
semantics must precede native proposals. Target activation, query/readout,
verification/rollback, cache/lifecycle and useful-speed gates remain open.

## Resources and promotion

Source/CPU-light work precedes the GLM → Qwen → Gemma resource handoff. Builds,
Metal, model loads and downloads are serialized. Use a distinct ignored target;
retain 8 GiB disk reserve and preserve all existing weights. No full model
download is authorized by this attempt. Sealed disposable outputs may be
retired only after preserving receipts. Runtime data stays private; publish
aggregate hashes/counts/timings. No PR or release promotion is claimed.
