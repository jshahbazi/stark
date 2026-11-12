# 20261112-FNI-01: general Flash Next image admission

## Identity and intended change

- Registered and started: 2026-11-12; owner: Codex.
- Baseline: Stark `72ad1b09e9f6ebdee269ea46197ccc06a5680c80`.
- State: implementation and qualification in progress; no runtime pass claimed.
- Trigger: Add Source rejects `nace-ai/Qwen3.8-Flash-Next-MLX-KQuant`
  with images enabled because the generic `qwen4_exp` route is text-only.
- Hypothesis: architecture, storage, complete tensor inventory and processing
  metadata can select the existing shared retained image backend without a
  repository allowlist. Preserve content-bound provenance and native admission.
- Candidate: general retained preparation, Add Source routing and automated
  structural controls. Preserve text MTP and existing text-prefix restoration;
  image requests must invalidate text prefixes and use zero speculative drafts.

## Artifact and acceptance gates

- Requested source: `nace-ai/Qwen3.8-Flash-Next-MLX-KQuant`, metadata revision
  `05cb5c56c918b8e32ed440c71d17a8e8f72fff90`.
- Initial metadata investigation reports a distinct KQuant layout with mixed
  widths, separate PLE/MTP files and approximately 100.2 GiB of stored weights.
  The recipe string initially suggested 6-bit projections; actual ranged tensor
  headers establish Q4/Q8 storage instead. No 6-bit inference path is needed.
  Trunk shared experts use Q8/group-128; embeddings and shared gates use Q8.
  PLE uses 128 indexed Q4/group-32 shards in one Safetensors file. These are
  storage-policy and tensor-view adaptations to the existing kernels.
- Before weight I/O: reject unsupported decoder packing, incomplete or
  quantized towers, unsupported processor operations, invalid indices and
  changed immutable source identities. Accept compatible independent sources.
- After preparation: source-bound hashes, regular-file ownership, complete
  native header admission and actual image-worker qualification remain required.
- Actual model gate: successful image generation and text/image/text recovery;
  exact fresh/cached text controls with supported text MTP; matched loading-mode
  performance measured separately with a fixed 5% regression bound.
- Resource stop: do not download more than available disk headroom; initial
  free disk is approximately 53 GiB. No deletion of other tasks' artifacts.

## Results and limits

- Baseline repository policy check passed.
- First full converter regression run: 7,224 passed, 94 skipped. This precedes
  completion of the indexed native adapter; final checks are recorded below.
- Actual pinned remote image preflight is conditional and binds the supported
  processor, template and PLE sidecar; it does not claim runtime execution.
- First live full-header test rejected the source patch-embedding kernel:
  its BF16 shape is `[1152, 2, 16, 16, 3]`, while the native loader originally
  required `[1152, 3, 2, 16, 16]`. The correction is a bounded lossless layout
  transpose at load time; the source payload and its hash remain unchanged.
- Norm-policy investigation rejected inheriting MLX-Serve's folded-norm
  assumption. Three actual KQuant norm tensors match the pinned official source
  byte-for-byte: 20,736 BF16 elements, maximum absolute difference zero. The
  pinned base-pack documentation declares zero-centered norms; the source
  implementation applies F32 `1 + weight`. The indexed adapter must use that
  convention and bypass the folded-gain HC fast path.
  Aggregate private receipt SHA-256:
  `710071d14a6eab7129491e315a0608d51ce766c596697499fe93c973f862f53b`.
- Actual source config SHA-256:
  `53b8832978052c49ad9c66ac64d2f1376e74cca666b001b460ef3b3ff718ad86`.
  The source's `conversion.json` records a different older config hash and
  size. Admission binds the actual files at the immutable repository revision;
  it does not trust the conversion sidecar as execution or payload proof.
- Corrected live remote header gate passed all 3,649 indexed tensors across
  52 shards, including 333 BF16 image tensors, 74 MTP tensors and 128 PLE
  ranges. The index SHA-256 is
  `aa7b502e618504b630d0f528064e8d7f0a450ef651c294cc110a416a8e6485f3`.
- All 35 actual PLE integer controls (280 bytes) match the native seed-1234
  construction exactly. Aggregate private receipt SHA-256:
  `01ddd236d22ae0681332e2750bd69493a34bcd4fb68e8967ea2ed1d2cdba3548`.
- Native core model-free suite: 97 passed, 54 ignored. Ignored physical-device,
  source-fixture and performance tests are not counted as executed checks.
- Storage follow-up: the user freed space, increasing observed free storage to
  approximately 155 GiB. The pinned full checkpoint download started; complete
  execution results follow after download and preparation.
- Full-model execution, numerical comparison, cache lifecycle and performance:
  not run yet. No cross-artifact qualification is inherited.
- Models, metadata scratch and raw runtime evidence stay in ignored/private
  storage. Publish only aggregate results and hashes.

See the [experiment ledger](../../model-experiments/README.md) and
[image-default plan](../../IMAGE_DEFAULT_PARITY_PLAN.md).

## 20261112-FNI-02: actual model lifecycle gate

Registered before loading the full checkpoint. Candidate: the same indexed
storage adapter, native worker and source revision recorded above. State:
pending download/preparation; no pass claimed.

Use `scripts/qualify_flashnext_images.py` with context 8,192, 4,104 nonperiodic
source-derived input tokens and a 64-token output budget. Exercise MTP0 and MTP2
with actual proposal counters, cold/reused/changed-tail exact output hashes,
the 4,096-token boundary, malformed image recovery, a bounded synthetic red PNG
answer check, zero image drafts/reuse, text recovery, explicit reset and
sequential process restart. Keep prompts, token vectors and responses in memory;
save only aggregate counters and hashes. Stop on a failed correctness/resource
gate and retain the partial receipt. Close only the qualification worker.

## 20261112-FNI-03: matched image-residency text check

Registered before timing. State: pending the lifecycle gate. Use the exact same
config and stored payload in text-only and image-loaded worker instances,
sequentially. Four loading blocks: text, image, image, text. For MTP0 and MTP2,
use 64 source-derived input tokens, context 512 and at most 64 output tokens;
one warmup and five measured requests per policy per block. Request scopes
prevent prefix reuse. Require matching greedy output hashes and actual MTP
proposals. Fixed regression bound: image-loaded median decode and end-to-end
rates must be at least 95% of the matching text-only rates. Report dispersion
and paging/compression/swap observations. A short passing result only covers
this workload and build; it is not a general throughput qualification.
