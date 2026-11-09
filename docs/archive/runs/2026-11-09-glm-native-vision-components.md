# 20261109-GLMVIS-02: native GLM image components and fusion

Registered 2026-11-09 before implementation, successor to
[20261109-GLMVIS-01](2026-11-09-glm-image-parity.md). Starting Stark source
`79351278`; branch `codex/glm-image-parity`. Intended implementation scope:
native GGUF sidecar admission, GLM still-image preprocessing and tower, then
request-owned embedding fusion before the existing mHC expansion. This record
does not cover the optional EXL3 image ABI.

## Bound sources and arithmetic

- Decoder and sidecar remain the pinned Unsloth revision from GLMVIS-01.
- Transformers investigation/reference revision:
  `b6401d6605d16620e0f95b54fb2906f92de7e33f`, GLM5-Next model and processor.
  Source files are retained privately; no repository Python is executed.
- GGUF metadata uses spatial pixel budgets 12,544 through 6,272,000. The pinned
  processor expresses budgets with temporal factor 2; dividing its static-image
  frame budget by 2 gives the same spatial values. The earlier ambiguity is
  resolved without changing the recipe.
- Candidate tower arithmetic: original BF16/F32 weights; F32 activations,
  attention scores and accumulations. This requires a matching F32 component
  reference and its own actual-model qualification. It does not inherit the
  source BF16 trajectory's numerical result.
- Existing text arithmetic, request policy and decoder weight storage remain
  unchanged. Image embeddings enter before the normal mHC expansion.

## Fixed first gates and stop conditions

1. Image dimensions, padding and temporal/merge patch order must match the
   pinned recipe on deterministic geometric controls. Reuse existing bounded
   image decoding/bicubic code; reject native processed-patch overflow before
   allocation, without silently changing source resize dimensions.
2. Admit exactly the 348 sidecar tensors and required metadata, complete file
   size and expected SHA256. Keep it separate from the four decoder shards.
   Changed source/profile/tokens/shapes/dtypes/spans must fail closed.
3. Test rotary coordinates, RMS/LayerNorm, clamped SwiGLU and downsample layout
   against independent scalar/reference controls before a full tower run. For
   F32 components use absolute/relative error <= 1e-3, including finite-value
   guards; report controls and any failures separately.
4. Before advertising images, execute an actual sidecar tower and complete
   image request, cancellation/error recovery and text→image→text invalidation.
   No image state or image request may publish text prefix snapshots or MTP.
5. Actual image-loaded text/cache/performance/MTP requirements remain the full
   matrix in IMGPAR-15. No component pass closes those gates. The ordinary GGUF
   MTP implementation gap and EXL3 image/embedding ABI gap stay explicit.
6. Build and GPU/model runs require the serialized resource slot. Keep an 8 GiB
   disk reserve; fetch only the 1,164,010,176-byte sidecar when space permits.
   Preserve all existing models; no full upstream FP8/BF16 source download.

## Component implementation and CPU evidence

The branch contains the following native components:

- Bounded GLM canvas/content sizing, bottom/right black padding, CLIP
  normalization and merge-block/channel/temporal patch order. It shares only
  existing JPEG/PNG decoding and validated bicubic resize primitives.
- Separate exact-size/SHA256 sidecar loading and the complete 348-tensor
  shape/storage contract. The sidecar remains separate from decoder shards.
- The 24-layer encoder, Q/K RMS normalization, axial rotary coordinates,
  noncausal attention, clipped SwiGLU, convolutional downsample and final
  projector/LayerNorm/GELU/SwiGLU merger. This assembled tower has not yet run
  against the actual sidecar; component checks do not establish its full result.
- A bounded floating convolution view in the existing mapped weight executor;
  rank/shape/precision restrictions preserve original weight bytes.
- Crate-private fresh image-feature fusion before existing GLM mHC expansion.
  It clears text-prefix capture, retains the existing text graph, and resets on
  cancellation or invalid features. Production image dispatch is not enabled.

CPU validation used `nightly-2026-10-01`, reporting
`rustc 1.101.0-nightly (21b707e3f 2026-09-30)`, with
`--no-default-features --features cpu,grammar`:

| Check | Result and scope |
| --- | --- |
| Full engine unit suite | 569 passed, 33 ignored, 0 failed |
| Supporting integration/doctest binaries | 12 passed, 0 failed |
| New normal GLM vision controls | Five pass: canvas/padding/temporal layout, rotary coordinates, attention constant-value/cancellation and merger scalar controls |
| New shared weight convolution control | Pass: independently known channel/kernel dot products; invalid width/rank/integer storage rejected |
| Private PyTorch 2.14.1 F32 attention reference | Pass on four deterministic rows / 4,096 output values at the fixed maximum-absolute and relative-RMS bounds of 1e-3 |
| Complete synthetic GLM fusion fixture | Exact logits and subsequent continuation versus identical text embedding rows; cancellation/invalid-feature reset pass |
| Formatting, whitespace and repository policy | Pass |
| Metal and actual sidecar/tower/model | Not run; resource deferred behind the serialized Qwen work |

The attention fixture contains deterministic component data, not prompts or
generated completions. Its SHA256 is
`d969a0018e6f7da6dc5f810ba6760b34ae91b4a252af1719ce2269a5d1f405df`.
The synthetic causal fixture is the existing private
`runtime/glm53-native-float-oracle-v3` artifact; it is not the 320B checkpoint.

The private source/artifact/log manifest is
`runtime/glm-image-source-2026-11-09/cpu-validation-receipt.json`, SHA256
`bf808ba1087a8792418765314561c783f00eacfefb3b378ca93f281079347491`.
The qualified CPU test executable was retained separately with SHA256
`42e4484bf0ca8edea9fe04fef0ffb89f0a9d58525c79cb25c4b6fbc5b6a819ae`.
Only this worktree's disposable CPU build target was retired after sealing
those identities, returning approximately 2 GiB of disk headroom. All models,
private fixtures, logs and the qualified test executable were preserved.

## Remaining gates and disposition

These components are **implemented on the branch / CPU-component checked**.
Full sidecar payload admission, Metal operation, complete tower numerical
reference, actual image generation/lifecycle, default preparation/loading,
image-loaded cache correctness, ordinary text performance and text MTP/image
transitions are **unfinished**. No image runtime capability is advertised.

Next, obtain the serialized resource slot and sufficient disk reserve to fetch
only the pinned 1.164 GB sidecar, then qualify its actual operations. Complete
content-bound artifact/host/worker integration and memory accounting before
enabling image dispatch. Run the full cache/performance/MTP matrix afterward;
GGUF MTP implementation and EXL3 embedding ABI remain distinct open work.
The current shared `load_with_image_support` also passes
`native_decode_stage: false`; extending its family allowlist alone would lose
the EXL3 retained-BF16 text MTP/default policy. GLM integration must preserve
the explicitly qualified native-stage/head load policy with images resident.

## Review follow-up: 20261109-GLMVIS-03

Independent component review found a descriptor-binding gap before any full
sidecar admission was attempted: hashing one opened file, independently opening
the weight mapping by pathname and then checking the pathname again could map a
substituted file if the original name were restored in between. The source now
checks the actual mapped `Mapping::_file` signature against the hashed
descriptor, retaining the separate final pathname check. A dedicated path
replacement control verifies that the signature follows the mapped file.

This changed candidate has its own post-review compile/test receipt;
GLMVIS-02's immutable CPU receipt above remains unchanged. Review otherwise
found normal-text arithmetic, image reset/failure handling, tensor layout and
bounded projection batches consistent within this component-only scope.
The source also checks cancellation before/after every eight-row projection
batch; this prevents submitting the rest of a full projection after cancellation.
Actual device cancellation latency remains unqualified and must be measured
before enabling host execution.

Post-review validation passed: 570 CPU/grammar unit tests, 33 ignored and zero
failures, plus the same 12 supporting integration/doctests. The new directory
swap control observes an unchanged full seven-field signature for the restored
pathname while the intervening mapped descriptor has a different signature.
The private F32 attention and exact embedding-fusion/cancellation controls pass
again. The existing pinned-upstream complete causal GLM synthetic reference was
also explicitly run and passed; this is not an actual 320B decoder result.

The final private review receipt
`runtime/glm-image-source-2026-11-09/cpu-validation-review-receipt.json` has
SHA256 `168f19c90eb41ffbbbbe0a8b878699560ad850cd6203c788e84fd3b690cac4c4`.
Its retained CPU executable has SHA256
`469f3bad62cc4069f3594a57179e7f36985ca52af200d95111da300b2453afcd`.
The disposable review build target was again retired only after sealing the
source/test identities and copying that executable; approximately 12 GiB disk
headroom remained. Metal/full-sidecar/full-model/default-image qualification
remains deferred and no image capability was enabled.
