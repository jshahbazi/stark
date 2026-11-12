# 20261112-FNI-01: general Flash Next image admission

## Identity and intended change

- Registered and started: 2026-11-12; owner: Codex.
- Baseline: Stark `72ad1b09e9f6ebdee269ea46197ccc06a5680c80`.
- State: implementation and bounded local qualification completed; limits and
  failed attempts are retained below.
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
- Full source-reference decoder comparison and general visual quality remain
  unassessed. Actual model lifecycle and short timing results are recorded
  below; no cross-artifact qualification is inherited.
- Models, metadata scratch and raw runtime evidence stay in ignored/private
  storage. Publish only aggregate results and hashes.

See the [experiment ledger](../../model-experiments/README.md) and
[image-default plan](../../IMAGE_DEFAULT_PARITY_PLAN.md).

## 20261112-FNI-02: actual model lifecycle gate

Registered before loading the full checkpoint. Candidate: the same indexed
storage adapter, native worker and source revision recorded above. State:
completed for the bounded lifecycle controls below.

Use `scripts/qualify_flashnext_images.py` with context 8,192, 4,104 nonperiodic
source-derived input tokens and a 64-token output budget. Exercise MTP0 and MTP2
with actual proposal counters, cold/reused/changed-tail exact output hashes,
the 4,096-token boundary, malformed image recovery, a bounded synthetic red PNG
answer check, zero image drafts/reuse, text recovery, explicit reset and
sequential process restart. Keep prompts, token vectors and responses in memory;
save only aggregate counters and hashes. Stop on a failed correctness/resource
gate and retain the partial receipt. Close only the qualification worker.

### Candidate identity

| Field | Value |
| --- | --- |
| Implementation commit | `6ddfdaba` |
| Host | Apple M5 Max, 128 GiB RAM, macOS 26.7.2 (25G332) |
| Toolchains | Rust 1.98.1 (`48a229cea`), uv 0.12.24, locked dependencies |
| Worker SHA-256 | `e44fcea263d8ceeea0819a5eebb257b7d2af933235e73cd4380a1eddb4d70543` |
| Lifecycle helper SHA-256 | `5d1c0c2bc2cff49a48c1a8ba70d91a2bd5aa7bb950980988d6d59c2a9458e3c9` |
| Cargo.lock SHA-256 | `f836a99ba1c6bd017fb829cd45f108fd97f9f1fb40c1497564898b3eac625f5f` |
| Converter lock SHA-256 | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |

Before the follow-up host memory fix, 7,241 converter tests passed,
94 skipped; 145 generation-worker tests passed, five ignored; 97 native-core
tests passed, 54 ignored; the new engine source-identity test passed under the
nightly custom harness; seven qualification-helper counterexample tests passed.
Repository policy, Python lint/format, Rust formatting and the release worker
build passed. These counts overlap focused subsets reported during development;
they are not added to those subsets.

### First actual attempt: memory preflight rejection

Full pinned preparation passed with 52 stored weight files totaling
107,614,239,896 bytes. The prepared configuration SHA-256 is
`49bce31896c983ef11d98e51f7f839bc31f2129f273f0254a1d66aab93777b61`;
the image manifest SHA-256 is
`99d8cead8011d2f1f94c54c65c9da9a613245b489782eb52e69a04a0bdf3808f`.

The first lifecycle attempt was rejected before launching a native worker.
The host's unknown-artifact memory estimate charged the entire source payload
plus 2 GiB: 109,761,723,544 bytes, against 99,870,408,704 bytes observed
available. That includes the 32 GB paged PLE table as if fully resident. No
generation or image execution occurred; this is an admission/resource failure,
not a numerical result. Preserve the partial private receipt as
`runtime/kquant-lifecycle-qualification-v1-memory-preflight.json`.

Follow-up `20261112-FNI-04` generalizes the host estimate using validated
tensor headers: resident target/head/tower payloads plus typed startup,
scratch, paged-working-set and driver/host reserves. It must independently
verify the supported layout before excluding PLE bytes; the native loader and
request allocation guards remain authoritative. The frozen exact Lily recipe
and unknown-layout fallback retain their previous behavior.

### Corrected actual lifecycle result

Host fix commit `52e7cf62` passed 116 focused memory/image-check tests. For this
artifact, validated resident payloads plus reserves require 91,585,062,040
bytes at the host's actual 16 KiB page size. The paged working-set allowance
uses that page size rather than the frozen legacy recipe's 4 KiB assumption.
No native worker binary or source payload changed for this correction.

The second full-model attempt passed all 14 registered lifecycle checks across
16 recorded generations and two sequential worker instances. Automatic local
image verification passed. Explicit checks passed for MTP0/MTP2 greedy equality,
actual MTP proposals, 4,096-token prefix restoration, changed-tail fresh
references, malformed-image recovery/invalidation, bounded red-image answering,
zero image drafts/reuse, cleaned image staging, text/MTP recovery after images,
re-established text reuse, explicit reset and process restart. This is bounded
local execution/cache evidence; full source-reference decoder logits and general
visual quality remain unassessed.

Private aggregate receipt:
`runtime/kquant-lifecycle-qualification-v2.json`, SHA-256
`18b96b20bac7f41ad23f4fbe6db01f97a0bd62e857ac2f25c274f377b7a2e57f`.

The actual desktop Add Source controller was then exercised in an isolated
library using the remote repository, immutable revision and `vision-language`
selection. Its real converter child completed, the host admitted the prepared
artifact, automatic local image execution checks passed, and the controller
registered one image artifact with `image_verification: locally_tested`.
The resulting manifest matched the independently prepared artifact above.
This uses the production controller path without replacing the installed app
or modifying its library. The installed Stark app was temporarily closed to
release its loaded model during the native tests.
The controller's private aggregate receipt SHA-256 is
`453843e4bc904addaed53d9a2caab1fe27fe8c15c0bd740ccf441d5feb9fbd37`.

After the memory fix, the full converter suite passed again: **7,249 passed,
94 skipped**. The existing CI benchmark/qualification step, including the new
harness counterexamples, passed **123 tests and 12 subtests** locally. No hosted
CI result is claimed.

## 20261112-FNI-03: matched image-residency text check

Registered before timing; completed after the lifecycle gate. Use the exact same
config and stored payload in text-only and image-loaded worker instances,
sequentially. Four loading blocks: text, image, image, text. For MTP0 and MTP2,
use 64 source-derived input tokens, context 512 and at most 64 output tokens;
one warmup and five measured requests per policy per block. Request scopes
prevent prefix reuse. Require matching greedy output hashes and actual MTP
proposals. Fixed regression bound: image-loaded median decode and end-to-end
rates must be at least 95% of the matching text-only rates. Report dispersion
and paging/compression/swap observations. A short passing result only covers
this workload and build; it is not a general throughput qualification.

### Results

Completed on the same exact prepared payload and worker. All 48 requests (eight
warmups and 40 measured requests) had identical greedy output hashes across
loading modes and MTP0/MTP2. Prefix reuse was zero for every request; MTP2
actually proposed drafts. The four sequential loading blocks used matched
configuration and payload identities. Loading/hash checks were outside timing.

| Policy | Text-only median decode tok/s | Images-loaded median decode tok/s | Image/text ratio | Text-only median wall s | Images-loaded median wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP0 | 62.9046 | 63.2476 | 1.00545 | 1.14388 | 1.13855 |
| MTP2 | 51.0214 | 52.8509 | 1.03586 | 1.39743 | 1.34835 |

Both fixed 5% timing bounds passed in this short observation. This does not
establish a general speedup or an MTP benefit: MTP2 was slower than MTP0 on this
workload. Ten measured requests per policy/loading mode contributed to each
median. Full dispersion and per-block observations remain in the private
aggregate receipt.

The measurement endpoints recorded no pageouts, swapouts or compression, and
thermal state remained zero. They recorded 12,710 decompressed pages across
the 40 measured requests, so **no request passed the strict clean-endpoint
screen**. CPU/GPU contention and clocks were not isolated. Disposition:
**bounded unscreened observation**, not production throughput qualification or
evidence of broad loading-mode performance equivalence. MTP1/MTP3, other
contexts/prompt lengths and sustained/image throughput remain separately
unqualified.

Private helper SHA-256:
`d181518a45f08f8f2ef91fe5d03e7566e0ba149da28eeafcb0fa56a10906aca3`.
Private aggregate receipt `runtime/kquant-matched-text-performance.json`, SHA-256
`323a4a60dab24832ef3f8e4fdedb5f9356208753aeea17f718c2e19a6121b6f5`.

## Delivery status

The shared implementation and follow-up host memory fix are committed on
`codex/flashnext-general-images`. Models, binaries and raw/private qualification
receipts remain ignored. The original installed application and its library
were not replaced; Stark was reopened after the sequential native runs. Use a
build containing this branch to receive the Add Source fix. No PR was requested.

## 20261112-FNI-05: independent follow-up review

The user requested another review and clarification of whole-family coverage.
Baseline for this review: pushed commit `e051a74e`. Independent native, host and
coverage reviews found the following admission issues to repair:

- The retained Lily route reused a text-only planner that intentionally ignores
  `mtp.*`. A malformed declared Lily head could pass pre-download header checks
  and preparation, then fail native loading. The fix must validate declared MTP
  metadata, complete head shapes/dtypes/packing and the native draft precision
  policy before downloading payloads. Typed text-only conversion keeps its
  existing omission contract.
- Merely finding an unrelated `ple_manifest.json` made processor inspection
  invoke the indexed KQuant validator for other storage dialects. The check must
  select the manifest by the declared format and require it for indexed PLE,
  while ignoring unselected auxiliary sidecars for other admitted dialects.
- The typed text planner allowed F32 controls that retained native RMS/HC,
  attention and other kernels require in BF16. A retained-only control-dtype
  check now preserves the native F32/BF16 GDN exceptions while rejecting
  unsupported control inputs before payload downloads. Typed text conversion
  and its recorded F32 reference evidence remain unchanged.
- Actual metadata inspection found that both pinned Lily and MLX-Serve sources
  omit processor sidecars. The generic route originally rejected those sources,
  despite the existing pinned preparation path deriving the fixed processor.
  The shared route now derives that same family recipe only after validating
  the image architecture/native geometry, validates any explicit metadata
  strictly, and records generated-file provenance separately from upstream
  source hashes. No new pixel-processing arithmetic was introduced.

These are host admission changes. They do not alter inference arithmetic or
the already recorded worker executable. Retain all earlier receipts; record
new regression/header checks and any additional actual-image smoke separately.

Whole-family support is not established. The shared implementation admits
specific compatible Lily, external-table affine and indexed affine contracts,
complete BF16 towers, native Flash Next geometry and supported processor/template
programs. Family membership alone does not establish those properties. The
original 14 checks cover one actual checkpoint and one small valid PNG plus a
malformed PNG; they are lifecycle assertions, not 14 image fixtures or a general
visual-quality evaluation.

### Actual source audit after the fixes

These are pinned metadata and complete tensor-header checks only. The review
does not transfer the actual `nace-ai` runtime result to other artifacts.

| Source and immutable revision | Image preflight | Complete indexed header check |
| --- | --- | --- |
| `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | Conditional, family-default processor | Passed: 55 shards, 3,845 indexed tensors |
| `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | Conditional, family-default processor | Passed: 101 shards, 3,167 indexed tensors, plus the external table planes |
| `nace-ai/Qwen3.8-Flash-Next-MLX-KQuant`, `05cb5c56c918b8e32ed440c71d17a8e8f72fff90` | Conditional, explicit processor | Passed: 52 shards, 3,649 indexed tensors |
| `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, `b2c422f3c643e36f04227a64d61796b44a4b1029` | Unsupported vision weight format | Not attempted |
| `Qwen/Qwen3.8-Flash-Next`, `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Unsupported vision weight format | Not attempted |

Private source-audit receipt SHA-256:
`500b93caeacc0ff27f71f5c797675b55abe20c274ed3cae75755ab70754c37f2`.
The missing-processor diagnosis is independently retained with SHA-256
`dfa76bb40469dfaad6d57dc622de7af181bdb45a2c6b47a11027accf1ac0f1c8`.

## 20261112-FNI-06: additional actual image fixtures

The original worker executable and exact `nace-ai` artifact were exercised again
at context 2,048, maximum 64 output tokens, greedy sampling. Four synthetic
fixtures passed 12 image/recovery/cleanup checks:

| Fixture | Format / dimensions | Image tokens | Result |
| --- | --- | ---: | --- |
| Solid red square | PNG, 256 × 256 | 64 | Passed |
| Solid green portrait | PNG, 256 × 512 | 128 | Passed |
| Solid blue landscape | PNG, 512 × 256 | 128 | Passed |
| Solid yellow square | JPEG, 256 × 256 | 64 | Passed |

Each image required its unique expected color label, no negated answer, zero
image drafts/proposals/reuse, valid token accounting, cleaned staging and
subsequent text generation with MTP2 enabled. Responses and token vectors were
kept in memory; receipts contain only hashes, counts and checks. This is
transport/aspect-ratio smoke coverage. General vision quality, reference
parity, multiple-image execution and high-resolution behavior remain unqualified.

Setup history is retained: the first private helper incorrectly assumed Pillow
was installed in the locked converter environment; it failed before launching
a worker. A replacement Swift/ImageIO fixture generator initially produced
black images because its drawing context did not fill the bitmap. Two attempts
with those fixtures failed their expected-color gates and are invalid setup,
not model correctness results. The black fixtures and failed receipts were
preserved. The generator was corrected to fill RGB bytes directly; corrected
fixtures were checked before the passing run. The final semantic gate accepts
an unambiguous, non-negated color answer rather than requiring exact one-word
formatting. No generation output was logged.

Passing private aggregate receipt:
`runtime/kquant-review-image-fixtures-v3.json`, SHA-256
`e7b6894d3641cf4d95fe33e90b13902beb7489f2daf4a701e994251926092bf4`.

### Follow-up validation and delivery

With the review edits frozen, the complete offline converter suite passed
**7,306 tests, 94 skipped**. The qualification harness counterexample suite
passed all seven tests. Repository policy and its 55 self-tests passed; Ruff
checks, formatting for all 219 converter Python files and `git diff --check`
passed. No Rust code or dependency locks changed in this follow-up, so the
earlier native test results and executable identity remain separate evidence.

An earlier full-suite attempt overlapped a regression-test edit and reported
one failing stale test assertion about source-file receipt keys. The final
frozen-source run above includes the corrected provenance assertion and passes.

The follow-up is delivered on the same `codex/flashnext-general-images` branch.
No installed application or user library was replaced. A build containing these
changes is required to use the revised Add Source route.
