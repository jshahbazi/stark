# Model change and experiment ledger

Use this directory to track attempts to make a model work correctly or run faster.
Record unsuccessful, invalid and resource-deferred attempts alongside successful
ones. A later passing run does not erase the earlier result.

This ledger connects a change to its evidence and shipping decision. Numerical
receipts, benchmark aggregates, source patches and detailed run records retain
their existing canonical locations under `docs/archive/`, `docs/benchmarks/` and
`provenance/`. Private model payloads and evaluation data stay ignored.

## Starting records

| Record | Scope |
| --- | --- |
| [20261109-GLMMEM-11: nonmodel mapping lifecycle](../archive/runs/2026-11-09-glm-mapping-lifecycle-diagnostic.md) | Review corrected periodic checksum gap; CPU/build and one tiny Metal run pass all four cases/60 copies plus wrong-offset control with zero paging/compression; wrapper zero differs from retained task footprint; full-model failure unreproduced, no production change/retry |
| [20261109-GLMVIS-10: functional-only GLM VM screen](../archive/runs/2026-11-09-glm-functional-pageout-screen.md) | Pinned source distinguishes global Pageouts from Swapouts; separate supervisor/source and child controls pass; one reviewed short run stops on hard compressor growth before first completion/image, owner drained and slot released; no retry, cache/MTP/performance/default promotion |
| [20261109-GLMVIS-09: bounded GLM image/text execution](../archive/runs/2026-11-09-glm-bounded-image-execution.md) | Fresh builds/native accounting/supervisor controls pass; initial and single quiet-host short attempts resource-deferred by strict global pageout growth before completion, owned workers drained; resident plan does not fit sampled bytes; image/cache/MTP/timing open |
| [20261109-GLMVIS-08: pinned artifact hash backend](../archive/runs/2026-11-09-glm-hash-backend.md) | macOS/aarch64 SHA-256 feature only; four saved vectors and fresh complete metadata/sidecar digests match accelerated, force-soft and independent references; full decoder replay/startup/model gates remain open |
| [20261109-GLMVIS-07: default-image admission](../archive/runs/2026-11-09-glm-image-admission.md) | Text and PNG preparation pass; image generation hits admission. Memory draft rejected for missing lazy/expert growth; conservative policy restored and Qwen handed the slot. Hash-backend change remains separate; image/default/cache/performance/MTP gates remain open |
| [20261109-GLMVIS-06: actual native GLM image generation](../archive/runs/2026-11-09-glm-native-image-generation.md) | Registered after external capacity recovery; fresh matching helper/native build, complete GGUF image execution and loaded-image text/cache/performance gates pending the serialized resource slot |
| [20261109-GLMVIS-05: complete GLM image worker integration](../archive/runs/2026-11-09-glm-image-worker-integration.md) | Pinned preparation/native dispatch source wired; 6,830 converter tests pass and independent admission fixes applied; actual GGUF predictor inventory confirms 29 retained head tensors; native compilation/full image/default/cache/performance/MTP gates remain open |
| [20261109-GLMVIS-04: actual GLM vision sidecar on Metal](../archive/runs/2026-11-09-glm-actual-vision-components.md) | Complete actual BF16/F32 sidecar tower matches a private F32 reference; native PNG/tower and synthetic decoder fusion pass; full GLM image requests/default dispatch/cache/performance/MTP remain unfinished |
| [20261109-GLMVIS-02: native GLM vision components](../archive/runs/2026-11-09-glm-native-vision-components.md) | GLM processor/tower/sidecar contract and embedding fusion; CPU/attention/synthetic causal controls pass; actual sidecar, Metal, default image dispatch and cache/performance/MTP qualification remain unfinished |
| [20261109-GLMVIS-01: GLM image source planning](../archive/runs/2026-11-09-glm-image-parity.md) | Pinned vision-sidecar identity/header, bounded source planning and backend-specific implementation sequence; image execution/cache/performance/MTP remain unqualified |
| [20261108-LVMTP-01: retained vision text MTP](../archive/runs/2026-11-08-lily-vision-text-mtp.md) | Lily vision/text MTP parity passed bounded output, 4,096-token prefix and image-transition controls; short paired text speed matched; memory recipe updated |
| [Image-support retrospective](2026-11-08-image-support-retrospective.md) | Conversation-wide findings, model coverage, validation mistakes, review, delivery and remaining work |
| [Qwen image attempts](2026-11-08-qwen-images.md) | BF16 and affine dense Qwen, affine and BF16-source MoE, retained Flash-Next/Lily numerical repairs and resource controls |
| [Gemma image attempts](2026-11-08-gemma-images.md) | Gemma 3 processor/convolution fixes; Gemma 4 efficient, unified, MoE and larger-tower attempts; decoder admission and retained compatibility |
| [20261108-VTP-01/02/04: vision artifact text prefix reuse](../archive/runs/2026-11-08-vision-text-prefix-reuse.md) | First recipe missed reuse coverage with exact outputs; stable-prefix and final restore-fence workers passed all 32 numerical/lifecycle cases on official Qwen3.5-0.8B BF16 |
| [20261108-VTP-03: paired vision text prefix timing](../benchmarks/2026-11-08-qwen35-vision-text-prefix.md) | Four observations per arm; bounded first-token benefit, cache-cold cost and snapshot payload; other artifacts and host-screened performance remain unqualified |
| [20261108-VTP-05: quick Lily text sanity check](../archive/runs/2026-11-08-vision-text-prefix-reuse.md) | Retrospective short unscreened text comparison; identical outputs and no apparent regression; separate from vision qualification |

The `IMG-20261108-QI-*`, `IMG-20261108-GI-*` and `IMG-20261108-XI-*` IDs register
historical attempts on 2026-11-08. The registration date is not a claim about
when each experiment ran. Original dates and identities remain in the linked
records. This image batch covers the image-support conversation and its
explicit prerequisites. The context/resource-budget attempts are indexed below.
These batches are not an exhaustive backfill of all Stark research.

Existing performance work remains indexed in the
[benchmark index](../benchmarks/README.md). The
[dense Qwen3.8 approach ledger](../benchmarks/2026-11-06-qwen38-27b-approaches.md),
[Flash-Next projection experiment](../benchmarks/2026-11-07-flashnext-moe-projections.md),
[sustained-throughput diagnosis](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md)
and [cross-backend idle-wake study](../benchmarks/2026-11-08-cross-backend-idle-wake.md)
are related records from other work. Their findings must retain their own
checkpoint, host-screening and source boundaries; they are not new measurements
from this retrospective.

## Record an attempt

1. Before changing code or running a comparison, copy the
   [attempt template](TEMPLATE.md) into a dated Markdown record. Choose a unique
   `YYYYMMDD-AREA-NN` or `WORKSTREAM-YYYYMMDD-AREA-NNN` ID
   (for example, `20261108-CTX-01` or `IMG-20261108-QI-001`)
   and list the model/checkpoint variants it actually
   targets. Keep source-bound validation records under `docs/archive/runs/` and
   timed studies under `docs/benchmarks/`; the ledger links their canonical
   evidence. State the observed problem, hypothesis, baseline, intended change,
   fixed acceptance limits and stop conditions. Use `not recorded` for missing
   historical details; do not reconstruct them from a different run.
2. Bind the checkpoint revision and artifact hashes, baseline and candidate
   commits/executables, source references, tools, device/OS and precision policy.
   Record conversion, kernel, cache, sampler and MTP choices that can affect the
   result. Separate image/tensor/component checks from full-model execution.
3. Run the smallest useful correctness controls first. For a speed claim, retain
   a matching correctness gate and a controlled end-to-end timing comparison.
   Preserve prompt/input lengths, output budget, warm/cold/reuse mode, seed,
   sampling, draft depth, repetitions, paired order and host/resource screening.
   Publish aggregates and hashes, never prompts, generated text, token vectors,
   probabilities, credentials or raw exception dumps.
4. Append what happened, including every failed gate, rejected arm, invalid setup,
   guard stop and known counterexample. Give each changed candidate or materially
   changed measurement recipe its own attempt or follow-up ID. Keep prior
   receipts immutable and link the predecessor.
5. Record the **outcome** and **promotion decision** separately, with reasons and
   the next test. A component pass, synthetic kernel speedup or unscreened timing
   observation does not establish a production full-model gain. Name each exact
   artifact whose qualification remains missing.
6. Add the entry to this index and link its ID from the implementation PR. Update
   the current model-support/reference page only when the evidence supports a
   capability change. When a new comparison supersedes a benchmark, update its
   current/superseded index without changing the historical measurements.

For a shared infrastructure change, one record may contain a per-artifact
matrix. Different checkpoints, quantizations, precision policies, MTP depths,
workloads or source revisions need separate results; a shared code path or
passing unit suite does not qualify every artifact.

This is a repository documentation workflow, not an automatic experiment
collector or evidence validator. Reviewers must check the linked receipts and
the claimed scope. A record does not authorize a large model run, a download,
publication or promotion by itself.

## Outcome and promotion vocabulary

| Outcome | Meaning |
| --- | --- |
| Planned / untested | Proposed or implemented route without the named execution evidence |
| Passed | The stated gate passed for the exact recorded inputs and build; name whether it is structural, component, full-model, lifecycle or timing evidence |
| Failed | A valid attempt missed its fixed gate; retain the failing metric and diagnostic limits |
| Invalid setup | The intended experiment was not run with matching inputs/tools/builds; results do not decide model correctness or performance |
| Resource-deferred | Admission or a resource guard prevented completion; this is neither a numerical pass nor a numerical failure |
| Inconclusive | Execution completed but controls, coverage or timing eligibility do not support the intended conclusion |

Use a separate decision such as `promoted within named scope`, `default-off`,
`diagnostic only`, `rejected`, `deferred` or `awaiting qualification`. An
intentionally rejected mutation is a passing **admission negative control**, not
a failed implementation. A counterexample can remain unresolved even when a
bounded feature gate passes. Never turn `implemented`, `merged`, `released` or
`CI passed` into a substitute for model qualification.

## Required evidence boundaries

- A checkpoint name, parameter count or catalog row does not identify the actual
  artifact, quantization, processor/template or selected native path.
- Full logit parity, greedy agreement, image-feature similarity, image-answer
  correctness and lifecycle safety answer different questions.
- A BF16 tower with an affine decoder differs from a quantized tower. A typed
  text-only conversion differs from a retained format containing vision.
- Equal output hashes do not establish throughput. Faster component timing does
  not establish faster requests. Memory admission caps and sampled RSS do not
  establish a full-model working-set guarantee.
- Prefix restoration, cancellation, reset, scope ownership and MTP must be
  assessed for every state-affecting route, following
  [model onboarding](../MODEL_ONBOARDING.md).

Backfill older work by linking its existing canonical record first. Mark missing
comparators or receipts explicitly; do not silently fill gaps with current
results from a different build.

## Evidence and status rules

| Dimension | Required distinction |
| --- | --- |
| Attempt state | Proposed, attempted, completed, blocked or superseded. A proposal is not an executed trial. |
| Result | Passed, failed, inconclusive or not measured, with the criterion and evidence scope stated. “Passed” alone is insufficient. |
| Evidence scope | Source review, model-free test, native smoke, numerical qualification, matched performance study or deployment check. |
| Correctness | Exact reference/artifact, aggregate error/hash agreement and tested operator/request scope. Equal errors are not output parity. |
| Performance | Same checkpoint/quantization, workload/options and timing boundary; repeats/order, host eligibility, cache state, dispersion, memory and excluded attempts. A short smoke is not a speed result. |
| Disposition | Experiment only, retained optional, adopted on branch, merged, released, installed, rejected or superseded; attach commit/PR/build evidence. CI success is not deployment. |
| Provenance | Source and dependency/vendor revisions, patch/manifest and executable/model hashes. A malformed or missing identity stays unresolved. |
| Safety/lifecycle | Context/resource admission, cancellation, error recovery, prefix reuse and clearing, scratch lifetime, and known limits. |

Existing studies also include the [Qwen3.8 optimization history](../benchmarks/QWEN38_OPTIMIZATION_HISTORY.md); its findings retain their original scope.

Use totals only within their recorded scope; overlapping reruns must not be summed. Report sampled tokens versus emitted/useful tokens explicitly. Do not use a directory name, available artifact, implemented backend or successful EOS as proof of broad model quality or speed.

## Context/resource-budget attempt index

All rows below link to the [complete findings and per-model matrix](../archive/runs/2026-11-08-context-resource-budget-findings.md). `CTX-NN` expands to `20261108-CTX-NN`. The implementation's runtime revision `9a3e7216` merged through [PR #99](https://github.com/jshahbazi/stark/pull/99) as `ea098a24`. This records runtime adoption; it does not establish release, installation or broader artifact qualification. This ledger and its workflow are a separate documentation addition.

| ID | Scope and attempt | Recorded outcome | Disposition |
| --- | --- | --- | --- |
| CTX-01 | Shared cap audit across model entry points | Source confirmed independent pre-tokenization policies. | Diagnosis recorded; coordinated change in PR. |
| CTX-02 | User-facing fixed resource setting | User rejected the interface requirement. | Superseded by automatic selection; developer override retained. |
| CTX-03 | Shared framing/template/content budget | Model-free large-input, sandbox and recovery controls passed. | Implemented in PR. |
| CTX-04 | Finite Auto and output ceiling removal | Qwen3.5 emitted 4,097 sampled tokens; short HTTP Auto/retry worked. Early identity has CTX-19's limitation. | Implemented; bounded native evidence. |
| CTX-05 | Explicit output requirement for unknown capacity | Did not satisfy automatic behavior requested by the user. | Superseded; legacy update guidance retained. |
| CTX-06 | Unknown-capacity memory planning, including Dense MTP | Synthetic ShortConv HTTP/CLI passed; allowance not filled. Dense MTP coverage is model-free. | Implemented; trained-artifact qualification pending. |
| CTX-07 | macOS memory probe prefix compatibility | Native smoke exposed rejection of a valid older statistics prefix. | Corrected and retested. |
| CTX-08 | Current-main integration/vision variants | Incomplete matches and unrelated branch content required correction. | Clean replay and integration fix; no new model qualification. |
| CTX-09 | Finite generic/GLM/Qwen/Flint/Lily cache admission | Source review found growth gap; native Qwen3.5 smoke and family inventory tests passed. Other families lack runs here. | Implemented; per-artifact limits retained. |
| CTX-10 | Unicode/concurrent HTTP parsing admission | Initial three-times-raw estimate was insufficient. | Replaced by shared twelve-times-raw estimate and atomic reservations. |
| CTX-11 | Hybrid inventory failure recovery | Error classification could unload a healthy model. | Corrected; model-free retry tests passed. |
| CTX-12 | Per-buffer and shader-index bounds | RAM-only fit was insufficient. | Checked geometry/device limits implemented and tested. |
| CTX-13 | Scoring retained-pool admission | Generation's pool-lifetime assumption was insufficient. Dense Qwen3 and hybrid Qwen3.5 exact-fill smokes passed. | Implemented; long scoring workload unqualified. |
| CTX-14 | Scoring overcharge refinement | Draft sampler/per-chunk charges were unnecessarily restrictive. | One final readback charge and final-pool allowance retained. |
| CTX-15 | Private app parsing and draining | Large frames lacked current parsing-headroom admission. | Corrected; worker, Reset, Stop and retry tests passed. |
| CTX-16 | Native/hybrid image decoder admission | Short qualification check did not cover full requested decoder span. | Implemented; model-free recovery coverage only. |
| CTX-17 | Selected sparse virtual cache capacity | Short physical span did not prove full virtual reservation valid. | Separate device/index and predictor-bound checks implemented. |
| CTX-18 | Buffered callback performance | No matched timing study. | Open measurement; no speedup claimed. |
| CTX-19 | Early Qwen3.5 tokenizer provenance | Recorded SHA-256 has 63 hexadecimal characters. | Unresolved; recover original receipt, do not substitute later hash. |

## 2026-11-08 Qwen Beginner memory admission

[Source-bound record](../archive/runs/2026-11-08-qwen-guided-memory.md).

| ID | Scope and attempt | Recorded outcome | Disposition |
| --- | --- | --- | --- |
| 20261108-MEM-01 | Exact Lily Q4-XL and MLX-Serve mixed text artifacts on M5 Max 128 GiB; reconcile Beginner and native memory admission | Final trials passed 9/9 requests per artifact through near-32K, MTP on/off and reuse/fresh-scope hash agreement. Physical footprint stayed below 74 GiB, without additional swap. Earlier admission, compile, test, telemetry and staging failures are retained. | Adopted on branch; signed build installed; Coding + Balanced recommendation and actual Lily startup verified. |
| 20261108-MEM-02 | [Fresh-install recipe-bound planning](../archive/runs/2026-11-08-fresh-guided-memory.md), independent of an installed copy | Corrected empty-registry planning; 73 focused tests and full converter suite (6,791 passed, 96 skipped) cover hardware/resource boundaries and source/artifact drift. No model run. | Adopted on PR #104; signed build installed and opened without loading a model. |

## Public record boundary

Publish aggregate results and hashes only. Keep prompts, completions, token vectors, probabilities, credentials, exception dumps, model files and private evaluation/runtime artifacts out of the ledger. An evidence link can point to an ignored private receipt by a stable identifier/hash without publishing its contents. Preserve upstream licenses and distinguish unmodified source revisions from local patches.
