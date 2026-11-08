# Gemma image experiments: retrospective and attempt ledger

Date: 2026-11-08. Evidence cutoff: image-family integration and its pre-PR review,
present on main at `7dc51834`. This is a retrospective of preserved runs; no new
model execution, conversion or performance measurement was performed for it.

Gemma 3 4B and Gemma 4 E2B/E4B have artifact-specific numerical and real-worker
evidence. Gemma 4 12B and 31B have qualified image components. The released
larger decoders and 26B expert weights still require execution qualification.
The most useful result of the investigation is the retained chain from a
failed model gate through independent operator witnesses to a passing unchanged
gate. Earlier failures remain evidence.

## Checkpoints and evidence boundaries

Revisions below identify source inputs, not interchangeable family approvals.
Unless stated otherwise, the November reference environment was Torch 2.14.1,
Transformers 5.15.1, Pillow 13.0.0, CPU eager BF16; native execution used Metal
on Apple M5 Max. Actual sealed executable identities belong to each linked run.

| Checkpoint / revision | Configuration and artifact | Established result; remaining scope |
| --- | --- | --- |
| `google/gemma-3-4b-it@093f9f388b31de276ce2de164bdc2081324b9767` | Schema 5 `gemma3-siglip-bf16-v1`; width 2560, 34 decoder layers; 27-layer width-1152 SigLIP; 896px, 4096 patches pooled to 256 embeddings | Full native prefill-logit/eight-step greedy replay and worker/HTTP lifecycle pass. [Gemma 3 run](../archive/runs/2026-11-07-gemma3-images.md). |
| Gemma 3 12B / 27B | Original BF16 multimodal wrapper; decoder 3840/48 layers and 5376/62 layers respectively; same SigLIP geometry | Structural admission only. No exact execution revision/result is claimed. [Contract](../GEMMA3_IMAGES.md). |
| `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7` | Schema 4 `gemma4-e2b-bf16-v1`; width 1536, 35 PLE layers, 20 shared-KV layers; 16-layer width-768/head-64 clipped tower | Original BF16 image artifact `5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993` passes bounded canonical comparison and retained-worker replay. [Original](../archive/runs/2026-10-28-olmo3-gemma4-native.md), [final replay](../archive/runs/2026-11-07-image-family-pr-review.md). |
| `google/gemma-4-E4B-it@4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | Separate schema-4 profile; width 2560, 42 PLE layers, two KV heads, 18 shared-KV layers; E2B tower geometry, separate projector/tokenizer | Artifact `58e1ed2e61286ccc58223bc46a2efe909f11b0b8eca6c4ccb03f00cdb435ea5a` passes its own three canonical cases and retained-worker replay. [E4B](../archive/runs/2026-11-07-gemma4-e4b-images.md). |
| `google/gemma-4-12B-it@43c9d014b231474fa7212dbb48a762566c2b255d` | Encoder-free width-3840 embedder; raw 48px patches and factorized positions; 48 no-PLE decoder layers, global K=V/one KV head | Ten-tensor image component passes; complete checkpoint not downloaded/converted. [Unified controls](../archive/runs/2026-11-07-gemma4-unified-vision-controls.md). |
| `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | Width 2816, 30 no-PLE layers, 128 experts/top 8 plus dense branch; global K=V/two KV heads; 27-layer width-1152/head-72 standardized tower | Runtime/converter and synthetic MoE controls pass. Released expert/vision weights and complete generation unqualified. [Family scope](../archive/runs/2026-11-07-gemma4-image-family-scope.md). |
| `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475` | Width 5376, 60 dense no-PLE layers, global K=V/four KV heads; same larger tower geometry | Extracted official tower/projector passes all three frozen feature gates. Full decoder/generation unqualified. [Component](../archive/runs/2026-11-07-gemma4-larger-vision-qualified-component.md). |

## Ledger conventions

IDs use retrospective registration date 2026-11-08; the linked run dates identify actual experiments (2026-10-28 through 2026-11-07). IDs are stable: append a new attempt when implementation, oracle or fixture
changes. `PASS` means the stated bounded gate passed; `FAIL` preserves its
negative result; `NEGATIVE` identifies a deliberately contrasting control;
`COMPONENT` restricts a pass to operators or extracted weights; `UNVERIFIED`
means no qualifying execution. Each row gives its disposition and next test.
RMS values refer to relative RMS unless the row explicitly says otherwise.

### Gemma 3 and efficient Gemma 4

| ID | Attempt, evidence and outcome | Promotion decision / next test |
| --- | --- | --- |
| IMG-20261108-GI-001 | **FAIL — Gemma 3 nonuniform resize.** Fixed Pillow 22-bit coefficients disagreed with pinned CPU preprocessing in 799 patch elements. Solid-color controls concealed this. [Run](../archive/runs/2026-11-07-gemma3-images.md). | Replaced by adaptive signed-int16 coefficients; retain original failure. Extend nonuniform sizes when processor support changes. |
| IMG-20261108-GI-002 | **FAIL → PASS — Gemma 3 convolution reduction.** Resize-only repair left pattern feature RMS 0.0338022. CPU Conv2D required four streams and bias before BF16 cast; corrected pattern RMS 0.0000713012. Seventeen source stage hashes reproduced. [Run](../archive/runs/2026-11-07-gemma3-images.md). | Retain dedicated patch-convolution dispatch; rerun complete gates after arithmetic changes. |
| IMG-20261108-GI-003 | **PASS — Gemma 3 original-reference resume and full replay.** Resume preserved all 13 original file hashes; an unchanged full forward reproduced pattern features. On `af356919`, all three patches exact, feature RMS 0.0000241230–0.0000713012, centered logit RMS 0.0295242–0.0399148, KL ≤0.000869342, 8/8 greedy per image. [Run](../archive/runs/2026-11-07-gemma3-images.md). | Exact 4B artifact qualified under fixed limits: feature ≤0.02/cosine ≥0.999, centered logits ≤0.1/KL ≤0.03. Other sizes need separate replays. |
| IMG-20261108-GI-004 | **PASS — Gemma 3 real lifecycle.** Worker `9ce9cf93` passed 14 gates, actual HTTP 200 with aggregate usage 273+8, disconnect cancellation/drain, same-instance text recovery, cached-check restart and staging cleanup. Separate completed-tower-block cancellation proof passed. [Run](../archive/runs/2026-11-07-gemma3-images.md). | Bounded request behavior qualified; installed UI, quality, JPEG end-to-end and image throughput remain unmeasured. |
| IMG-20261108-GI-005 | **PASS — E2B canonical BF16 images.** Three exact-patch cases: feature RMS 1.17%, 1.33%, 1.50%, cosine >0.99988; three greedy agreements, maximum KL 0.000028174. Independent CPU BF16/F32 feature differences 1.70–1.78% calibrated the 2% gate. MLX-VLM was a separate comparison. [Original](../archive/runs/2026-10-28-olmo3-gemma4-native.md). | Preserve canonical-HF identity; no image-specific BF16/F32 logit calibration was run. |
| IMG-20261108-GI-006 | **PASS / invalid invocation retained — PLE fusion regression.** Source residual fusions retained BF16 sum boundaries and explicit source-profile selection. Separate canonical BF16 vision artifact still passed 3/3 with aggregate bytes unchanged. One harness selected a text-only artifact and failed loading; corrected invocation passed. [Fusion checkpoint](../archive/runs/2026-10-29-gemma4-source-residual-fusions.md), [fresh checks](../archive/runs/2026-11-02-olmo3-gemma4-source-fusions.md). | Keep canonical/legacy PLE operations separately selected. Text/Q4 timing evidence supplies no image speed result. |
| IMG-20261108-GI-007 | **PASS — E4B independent conversion/reference.** 1234 tensors; peak converter RSS 634,290,176 bytes, 8MiB transform workspace. Three own-tokenizer CPU references: exact patches; feature RMS 0.0114521/0.0130552/0.0147230; centered last-logit RMS 0.0942021/0.0914297/0.0796816; greedy agrees. [E4B](../archive/runs/2026-11-07-gemma4-e4b-images.md). | One output token per fixture; no borrowed E2B features/tokens. Longer generation needs separate qualification. |
| IMG-20261108-GI-008 | **PASS — E4B lifecycle/cancellation.** Worker, cold/cached client, malformed recovery, prepared-image single use, fresh/restart replay and counterfactual pass. Controlled cancellation before/after synchronous projection preserved loaded-instance text execution. [E4B](../archive/runs/2026-11-07-gemma4-e4b-images.md). | Safe-boundary cancellation established; backend/nonfinite/allocation/GPU failures retain fencing. No quiet-host timing claim. |

### Larger decoder and unified-image controls

| ID | Attempt, evidence and outcome | Promotion decision / next test |
| --- | --- | --- |
| IMG-20261108-GI-009 | **COMPONENT PASS / NEGATIVE — no-PLE residual topology.** Explicit post-attention BF16 residual, MLP addition and once-only layer scalar pass six-layer CPU lengths 1/9/31/63 with Q/K gains 0.125: centered RMS ≤0.002327. Identical-weight legacy distributed-scalar selection fails. [Residual](../archive/runs/2026-11-07-gemma4-nople-residual-controls.md). | Select default-false materialization only for declared no-PLE BF16 profiles. Obtain full released decoder evidence. |
| IMG-20261108-GI-010 | **FAIL / NEGATIVE — six-layer unit-Q/K stress.** BF16 RoPE improves a short case without resolving long contexts. Width-512 lengths 31/63 have centered RMS 0.318832/0.312848; BF16 score/probability materialization worsens to 0.434641/0.472615 and stays disabled. Torch math SDPA still fails at 0.276836/0.259122. Direct lower-window predicate agrees. [Unified diagnostics](../archive/runs/2026-11-07-gemma4-unified-vision-controls.md). | Unresolved; retain every reference. Layerwise length-31 residual RMS grows 0.001935→0.51951; first differences precede window boundary. No single-cause conclusion. |
| IMG-20261108-GI-011 | **COMPONENT PASS — trained-gain calibration.** Only 12 Q/K tensors changed in a synthetic six-layer fixture using extracted 31B gains: Q 0.94921875–1.03125, local K 0.12109375–0.1318359375, global K 0.0615234375. CPU lengths 1/9/31/63 pass, centered RMS ≤0.02158. [Residual](../archive/runs/2026-11-07-gemma4-nople-residual-controls.md). | Shows trained gain magnitudes matter; projections/scalars remain synthetic. Unit-gain failure remains open. |
| IMG-20261108-GI-012 | **COMPONENT PASS — MoE mechanics.** Six layers, four experts/top two, nonunit/negative gains/scalars; original residual feeds router, F32 softmax/top-k renormalization, separate dense/expert norms, BF16 accumulator boundaries. CPU/Metal four lengths pass: centered RMS ≤0.010053/0.009049; 124 converter/support tests pass. One/two-layer unit-Q/K controls pass ≤0.004549. [MoE](../archive/runs/2026-11-07-gemma4-moe-component-controls.md). | Actual 128-expert/top-eight 26B weights still unverified. Controls default false outside declared profiles. |
| IMG-20261108-GI-013 | **COMPONENT PASS — encoder-free 12B.** Strict namespace mapping loads all ten tensors; 99,844,608 source payload bytes extracted. Three native-Metal/CPU-reference image cases have exact patches, feature RMS ≤0.00007343, cosine ≥0.9999999973; sealed `af356919` replay confirms. [Unified](../archive/runs/2026-11-07-gemma4-unified-vision-controls.md), [seal](../archive/runs/2026-11-07-gemma4-larger-vision-stage-diagnostics.md). | No random missing-weight substitution. Full 12B decoder and worker remain unverified; file caps are admission bounds. |

### Larger vision: preserved failures and successive isolation

The feature limits stayed **exact patches, RMS ≤0.02, cosine ≥0.999** throughout.
Source-only checkpoints in the dated records were subsequently executed only
where the following rows identify a sealed result.

| ID | Attempt, evidence and outcome | Promotion decision / next test |
| --- | --- | --- |
| IMG-20261108-GI-014 | **FAIL → component loads — 31B namespace.** Official ordinary linears retain `*.linear.weight` wrappers. `a5eaa732` maps them to native leaves. Clone-only derivative changes 189 names, verifies all 356 tensor payloads/offsets, allocates 65,536 physical bytes. [Stages](../archive/runs/2026-11-07-gemma4-larger-vision-stage-diagnostics.md). | Original preserved; corrected component `adcc1e2a…` used subsequently. Check 26B separately. |
| IMG-20261108-GI-015 | **FAIL / diagnostic — initial tower and padding/pooling hypotheses.** Exact patches but RMS 0.07467/0.09914/0.05464. Zero 128-padded tails; patch projection exact; canonical-input pooling RMS only 0.000354–0.000375. Encoder drift 0.0561–0.0854 localizes before pooling. [Stages](../archive/runs/2026-11-07-gemma4-larger-vision-stage-diagnostics.md). | Padding/pooling do not explain measured failure. Preserve bounded-prefix limits. |
| IMG-20261108-GI-016 | **COMPONENT PASS; feature FAIL — RMS cascade.** Width72/1152 source fixtures expose 4/1/112/103 BF16 rounding witnesses against sequential sums. `fcadc29f` CPU/Metal cascade controls and prepared Q/K/V become exact; features still 0.07462/0.10354/0.04970. [Stages](../archive/runs/2026-11-07-gemma4-larger-vision-stage-diagnostics.md). | Retain logical-width normalization before padding; attention becomes next measured difference. |
| IMG-20261108-GI-017 | **COMPONENT PASS; feature FAIL — Flash source policy.** 512-key tiles, source vector exponential, physical BF16 unnormalized exponentials, F32 denominator/PV; six valid-key fixtures include 776 witnesses against ordinary F32 probabilities. `32b33e16` passes CPU/Metal; features still 0.05385/0.07353/0.04158. Full canonical Q/K/V exact, attention has 513 BF16 differences. [Flash](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md). | Keep fixed reused planes and source softmax boundaries. Prefix success alone insufficient. |
| IMG-20261108-GI-018 | **COMPONENT PASS; feature FAIL — F32 QK schedule.** Source F32-output four-part fold replaces BF16-output 32-stream assumption; independent Torch logsumexp exposes 52 F32 witnesses/256 queries. `214c19a9` passes controls/all-head prefix; full attention retains six differences/2,757,888, RMS 0.00000552; features 0.03685/0.04938/0.03614. [Flash](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md), [MLP](../archive/runs/2026-11-07-gemma4-larger-mlp-controls.md). | Retain four-part QK; dense BF16-output projections keep their own schedule. |
| IMG-20261108-GI-019 | **NEGATIVE / COMPONENT PASS — down-projection hypothesis.** `2529098a`: native down input differs in 2819 BF16 values; native projection on canonical input matches all 2,757,888 outputs. Prefix gate/GELU/up/product exact; four-stream prefix control differs once. [MLP](../archive/runs/2026-11-07-gemma4-larger-mlp-controls.md). | No projection change justified for that first-block case; later layers remain separate. |
| IMG-20261108-GI-020 | **COMPONENT PASS; residual FAIL — denominator FMA.** Source threads/strides preserve oracle outputs. `96b6da7d` explicit `fmaf(factor,previous_sum,tile_sum)` makes affected CPU exact, Metal differences six→four; QK and BF16 exponentials exact. Nine/twenty scalar factors differ one F32 ULP. [Flash](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md). | Keep PV rescaling separately rounded; no independent reciprocal defect shown. |
| IMG-20261108-GI-021 | **COMPONENT PASS; feature FAIL — nearest-F32 scalar.** Compensated degree-12 Taylor/range reduction passes 8215 cases on `f67ab9d5`; source libm differs from decimal-nearest in 42 one-ULP cases. Blue first-layer attention becomes wholly exact. Feature RMS 0.0091937972/0.0132609557/0.0264942683: pattern still fails. [Flash](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md). | Retain independent nearest helper/goldens. Mathematical-nearest policy has insufficient source compatibility for this model gate. |
| IMG-20261108-GI-022 | **NEGATIVE / COMPONENT PASS — GELU and whole-row diagnosis.** `bd4ea321` Gemma GELU matches all 65,280 finite BF16 values on CPU/Metal; no GELU change. Whole pattern blocks 0–2 exact; block 3 has 63 differences/2,654,208. `b003b291` finds one layer-3 attention difference with exact Q/K/V; substituting source scalar factors alone removes it. [Stages](../archive/runs/2026-11-07-gemma4-larger-vision-stage-diagnostics.md), [Flash](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md). | Eight-row prefixes had concealed earlier whole-block drift. Keep full-row selected-layer controls. |
| IMG-20261108-GI-023 | **COMPONENT PASS with known counterexample — source-compatible scalar.** Independent 128-entry power/quadratic formula `1+r*(a+a³/32+r*a²/2)`, `a=ln(2)/128`, compensated F32 pairs. `e5dade43` passes 8220 source-libm controls and retained nearest controls. Broader 106,525 CPU inputs retain one one-ULP source-libm discrepancy. [Flash](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md). | Qualified finite compatibility only. Counterexample hash `1bb4d3b41ee73d5a9c43d4ce1092a86181ea30e58d0331f89ac161fdfde7d8ce`; expand independent scalar coverage. |
| IMG-20261108-GI-024 | **COMPONENT PASS — final original three-image gates.** `e5dade43`, executable `85a98ef1…`: exact patches; RMS 0.0010468731/0.0010244444/0.0011068880; cosines 0.9999994520/0.9999994753/0.9999993874. Full captured blue first-layer/pattern layer-3 attention exact. Receipt `ee85488d…`; process RSS 1,237,057,536 bytes. [Qualification](../archive/runs/2026-11-07-gemma4-larger-vision-qualified-component.md). | Passing extracted 31B tower/projector; no complete-model memory, generation, universal exactness or speed claim. |

### Admission and final retained-worker replay

| ID | Attempt, evidence and outcome | Promotion decision / next test |
| --- | --- | --- |
| IMG-20261108-GI-025 | **FAIL → PASS — P2 named-profile decoder admission.** Independent review reproduced altered layer-count/global-RoPE decoder retaining official profile. `f8293f87db7052da910acc378d21470072b1d4a7` binds pinned source config bytes and compares complete typed decoder, including geometry, precision, PLE and MoE. [Review](../archive/runs/2026-11-07-image-family-pr-review.md). | Exact retained E2B embedding-selector pair `(false,false)` remains admitted; canonical `(true,true)` admitted; mixed pairs rejected. Structural fixtures establish compatibility only. |
| IMG-20261108-GI-026 | **PASS — post-fix actual E2B/E4B retained workers.** Release SHA-256 `a49cb0620f5634772712d9ce44c6fbb2c61f845593d2cb05d9b0cb7fc5296988`; engine/worker trees identical across `f8293f87`/`a612db68`; 76-file seal `87f660dd…`. Both pass three greedy cases, discrimination, fresh/restart, malformed recovery, single-use/staging cleanup; zero exit, no guard/timeout. Receipts E2B `0676f408…`, E4B `a52bbfef…`. [Review](../archive/runs/2026-11-07-image-family-pr-review.md). | Retained artifacts function after stricter admission. Newly converted canonical artifacts need their own full-logit qualification. Competing worker prevents performance claims. |

## What to carry forward

Arithmetic policy is part of provenance. PLE/shared-KV efficient decoders,
no-PLE K=V decoders, encoder-free patches, and larger head-72 vision must keep
separate dispatch and evidence. Global no-PLE attention is causal; local image
visibility includes its strict lower-window bound. Gemma 3 global layers use
causal-or-same-image composition. Image requests clear/bypass prefix reuse and
disable image MTP; Gemma sliding-ring restoration remains a concrete blocker.

The [provenance record](../../provenance/UPSTREAM.md) preserves Transformers
Apache-2.0, PyTorch BSD and selected ARM MIT notices, source hashes and independent
derivations. No proprietary host implementation or table bytes were copied for
the scalar formula. Model weights retain separate Gemma terms. Private prompts,
responses, tensors and logs stay ignored; this ledger publishes aggregates.

Next qualification should resolve IMG-20261108-GI-010 and execute exact released larger
decoders before worker/generation claims. Maintain separate identities for
checkpoint, artifact, reference, executable, host package and receipt; preserve
failures and changed-oracle experiments separately. A later source-only status
does not erase an earlier executed result, and a passed synthetic operator does
not certify a checkpoint. Hosted CI, installed UI, answer quality and image
throughput require their own evidence. The [integration](../archive/runs/2026-11-07-image-family-integration.md)
and [review](../archive/runs/2026-11-07-image-family-pr-review.md) retain regression
counts, invalid test-setup attempts, full seals and publication history.
