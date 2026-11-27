# HY4STQ-SERVE-02 · Default shared-weight text slots for Hy4 STQ

This change follows the [two-slot preview](2026-11-27-hy4-stq-dual-slot.md).
The native catalog artifact is
`beamster/Hy4-Flash-Sushi-STQ-4bpw@61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`.
The artifact identity is
`1ed7e181a27f78ffb1e93f9bef2eb00be3c9c5933cf5d7f50494320b45e9a04f`.

The desktop requests four slots by default. The native loader shares one model
and admits the highest slot count the memory budget supports. The API uses a
window that fits the admitted slots, and the primary retains its separately
fitted capacity for Playground and image work. Each text slot keeps its own
context, KV and index state, cancellation pipe, and output stream. A text
request can use a smaller logical window. A load requested with one slot uses
the ordinary automatic fit.

The native phase queue lets the slots share the temporary Metal projection,
routing and attention buffers. Each auxiliary still owns its Engram readers,
draft target, readout and sequence buffers. The fitter reserves its private
buffers and limits common capacity before dropping a slot. Primary prefix
reuse follows the consecutive-request policy. Auxiliary starts and finishes
mark any retained primary state for invalidation before the next primary
request. Limiting auxiliary state to the active request avoids holding
several completed conversations in memory.

## Qualification

- Reuse the 16,229- and 27,229-token OpenCode renders from the two-slot
  preview. Check the same 96-token budgets, input and output hashes, and
  match the expected lane count to actual native admission.
- Record four-agent first-token latency and total group time, retain a solo
  27K control, and keep the existing five-draft and F16 execution path.
- Exercise flexible windows, image handling and consecutive primary prefix
  reuse on the same model instance. Check cancellation and EOF routing as
  model-free controls.

## Result

Qualification is in progress.
