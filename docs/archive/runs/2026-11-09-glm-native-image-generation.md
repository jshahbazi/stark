# 20261109-GLMVIS-06: actual native GLM image generation and text parity

Registered before the new native build/model runs. Integration source baseline:
`9efdb44e`; predecessor: [GLMVIS-05](2026-11-09-glm-image-worker-integration.md).
The four pinned UD-Q2_K_XL decoder shards, separately verified vision sidecar,
private F32 tower reference and qualified component binaries survived the
external capacity recovery. The main checkout's build directory and template
helper are absent. New validation must build the matching helper and worker in
this worktree's own ignored target and retain their identities separately from
the historical results.

## Fixed execution and acceptance scope

Use the unchanged decoder/source tuple and exact schema-7 metadata described
in GLMVIS-05. Create an owned image folder by sharing immutable decoder bytes
and the existing sidecar; perform native content and mapped-descriptor
admission. No further full-model download or model deletion is authorized.
Keep at least 8 GiB free and serialize native build/model work after Gemma's
resource handoff, with Qwen next after GLM workers close.

First compile and run the native profile, source-binding/ownership controls,
fixed F32 tower reference and causal fusion controls. Then execute a complete
installed-model image prepare/generate/discard case, verify finite/nonempty
generation, image request zero drafts/proposals/reused tokens and following
text recovery. Record only hashes, counts, booleans and aggregate timings.

Qualify normal text prefix reuse with images loaded against fresh execution at
the native 1,024-token boundary, including context/scope/state-option changes,
image success/failure, cancellation, delivery failure, restart and idle cleanup
where the existing validator can exercise them. Image requests must invalidate
old snapshots and must never publish image-derived prefixes.

For bounded matched timing, compare a source-identified prior text worker,
candidate text loading and candidate image loading using identical decoder
bytes, target arithmetic, opt-in GLM flags, greedy sampling, context, prompt and
output budget, with MTP zero on all three ordinary GGUF arms. Warm every arm;
collect at least three quiet accepted baseline replays before candidate timing.
Before observing candidate results, fix each metric's allowed relative
regression to `max(0.10, 3 * baseline_MAD / baseline_median)`. Assess prefill /
first token, useful decode and end-to-end independently; distinguish loading
and memory observations. Reject or separately report busy/paging attempts. A
failed quiet-host control is not evidence of a model regression or parity.

Enable the normal catalog/preparation/image-loaded route only after the actual
image case passes and its remaining limits are represented honestly. The
ordinary GGUF source's 29 predictor tensors still require a real execution
implementation; a text-only or image-loaded result with MTP zero does not
qualify that gate. EXL3's image ABI and its retained native-stage/default-three
text MTP require separate implementation and validation. Neither is N/A.

## Status

Registered; native build and actual-model execution pending the resource slot.
The earlier 6,830 converter tests and numerical component results remain bound
to their own sources and binaries. They are not this attempt's full-model proof.

### Fresh native build and controls

The Rust 1.98.1 worker/helper build succeeded. The first nightly engine-test
build found the new controls used ordinary `#[test]` attributes instead of
Stark's custom harness attributes; its failed build record is retained. The
controls now use `#[stark_test]`, and the existing Lily default wrapper remains
the public default entry's delegate. The corrected native test build succeeds.

The test executable is
`bbdb68c18ee6f742b3625ae1a166f8252efbce19b33083168f484d867d2c81ea`.
Ten GLM admission/preprocessor/operator controls pass. Six separately selected
mapped-descriptor, convolution, fixed attention reference, actual PNG/tower
cancellation, exact causal fusion/snapshot fencing and complete causal-reference
controls pass; their private receipt is
`157c61e83d453a3569b66571f6f666e85f820d838a7bf5592e3d4ba2631d4558`.
The separately run actual full tower again matches the fixed F32 reference:
maximum absolute error `2.8312206268310547e-6`, relative RMS
`1.90591745194979e-5`, under the unchanged `1e-3` gates. These are 17 focused
native controls, not the complete engine suite or a full decoder image request.
The matching newly built helper also passes 187 focused host/converter controls,
including the larger-than-one-MiB request boundary. All source/build identities
remain private and separately bound; the final worker is rebuilt/resealed after
the harness/wrapper correction before full-model execution.
