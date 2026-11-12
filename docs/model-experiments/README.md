# Model change and experiment ledger

- [20261112-CATALOG-01: shared config normalization](2026-11-12-catalog-config-normalization.md): one reviewed rewrite pass before every in-scope family planner, shared by catalogue admission and conversion; unknown keys and real RoPE variants still fail closed. Header admission only; converted artifacts unqualified.
- [20261110-CATALOG-01: catalogue family admission](2026-11-10-catalog-family-admission.md): shared header planners, interleaved discovery and unsupported reasons. Pre-PR review repaired image-to-text fallback after vision header failure. Full checkpoint qualification remains separate.

- [20261110-TOOL-QUALITY-01: bounded tool output and continuation replay](2026-11-10-tool-output-quality-replay.md): five installed artifacts checked; all passed synthetic tool continuation, four passed ordinary-chat exact match; 17×19 arithmetic controls isolated no demonstrated parser failure.
- [20261110-TOOL-QUALITY-02: restored Qwen3 0.6B artifact replay](2026-11-10-tool-quality-qwen06-replay.md): pinned-source recreation failed the tiny-tool control; replay of actual host input fit context and ended at EOS with no calls. Model/workload limitation supported, no parser repair established; temporary import removed.
- [20261110-G4INSTALL-01: Gemma installation repair](2026-11-10-gemma-install-repair.md): bounded HF config-link authorization, E4B conversion/app image smoke, and content-free failure diagnostics.

Current continuation: [Gemma3 cache and Gemma4 assistant implementation](2026-11-09-gemma-next-image-parity.md)
records attempts `20261109-G3IP-01` and `20261109-G4AP-01`; execution is pending.

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
| [20261112-DENSE-MLX-01/02/03: Qwen2 and Mistral MLX affine import](2026-11-12-dense-mlx-qwen2-mistral.md) | Planner gates opened for uniform 4/8-bit MLX; nine live repositories pass real preflight and header planning (re-checked after the main merge). Qwen2.5 0.5B 4-bit fails the stock-MLX logit/greedy gates (0/5), inside MLX's own F16/BF16 envelope; prefix reuse exact. The pinned Mistral 7B v0.3 4-bit export fails conversion (named chat template); a template-substituted copy fails 1/5 logit cases and passes greedy checks. Not qualified; F16 execution and named-template handling are open. |
| [20261112-FNI-01–07: general Flash Next image admission](../archive/runs/2026-11-12-flashnext-general-images.md) | Shared Add Source admission, actual KQuant lifecycle and image fixtures, and contract/harness follow-up; compatibility and artifact-specific qualification remain bounded to the recorded evidence. |
| [20261110-Q38IMPORT-01: external-table affine import](2026-11-10-qwen38-external-import.md) | Live iQ-MLX catalog and preflight, complete header comparison, synthetic import and manifest replay. Full checkpoint execution deferred for storage. |
| [20261110-LILY-SIZE-01: retained image resolution](2026-11-10-lily-image-size.md) | Retained Lily 16 MP processor ceiling, negotiated app limits, 65,536-row component controls and 2.4 MP native execution; signed local build installed, full-ceiling model qualification pending |
| [20261110-FAMILY-RECHECK-01: independent pre-PR review](2026-11-10-family-double-check.md) | Isolated cache invalidation controls, accepted-temperature overflow handling and scratch-pool high-water admission; revised validation tracked separately |
| [20261110-DENSE-01: capped attention and dense sampling](2026-11-10-dense-attention-sampling-correctness.md) | Gemma 2 soft cap, recoverable sampling validation, dense MTP filter semantics, nonfinite readout errors and optional snapshot copying; component controls and artifact limits tracked separately |
| [20261110-CONVERT-01: family conversion regressions](../archive/runs/2026-11-10-family-converter-correctness.md) | Safe dense config annotations, Qwen3.5 norm convention and size-specific metadata; RNJ/Polaris remain planned pending YaRN support |
| [20261110-IMAGE-01: image loading and request admission](2026-11-10-family-image-correctness.md) | Single component scan with mutation checks, skipped PNG ancillary inflation, primary MPO decoding and tower workspace accounting; artifact qualification pending |
| [20261110-PREFIX-01: text-only snapshot routing](2026-11-10-text-prefix-snapshot-capability.md) | Route snapshot-only ordinary text through existing exact state capture/restore; artifact qualification pending |
| [20261109-LAUNCH-RECHECK-01: image-resident Lily text controls](2026-11-09-model-launch-double-check.md) | Same ddalcu payload with explicitly prepared image metadata passes seeded three-draft history, 4,096-token reuse and real image-to-text recovery; original folder is not image-admitted, paging prevents speed claims |
| [20261109-QMEM-01: F32 Qwen scalar attention workspace](2026-11-09-qwen-attention-memory.md) | Independent lazy scalar scores/probabilities and matching allocation inventory; allocation/stride and synthetic full-forward controls passed, production speed unmeasured |
| [20261109-LILY-STARTUP-01/02/03/04: retained image startup](../benchmarks/2026-11-09-lily-image-startup.md) | Initial startup canceled; accelerated full scan passed image smoke but latency rejected; digest reuse/UI cleanup installed, native startup 10.5–11.9 s, routine UI Ready by 14.774 s; final image attachment and copy-removal checks passed |
| [20261109-QDIP-01: dense default-image text parity](../archive/runs/2026-11-09-qwen-dense-image-default-parity.md) | Preserve the supported affine dense text MTP head with the image tower resident; source-bound cache, conversion and performance qualification is tracked separately |
| [20261109-QDIP-02/05/06: dense image-loaded text controls](../benchmarks/2026-11-09-qwen-dense-image-text-parity.md) | Separates tower residency from changed target arithmetic; corrected short 27B comparison fails text speed parity despite exact output hashes and restored MTP |
| [20261109-QDIP-07: shared-weight 27B fast text](../archive/runs/2026-11-09-qwen27-shared-weight-fast-text.md) | Exact Jundot target/head payload and configuration equality; normal text regains Legacy graph and batched MTP with images resident. Paired 64-token MTP0/3 medians pass bounded parity; all 14 nonperiodic near-capacity cache/lifecycle controls pass. Other artifacts retain separate gates. |
| [20261109-QNIP-01/02/03/04/05: remaining Qwen image parity](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md) | Three headless exports pass short speed gates; BF16/affine 0.8B each pass 30 lifecycle controls. Headed BF16 MTP0 passes 34/28 lifecycle checks plus 48/26 geometry/image-recovery checks, ordinary hybrid restoration and strict short speed gates. Test-only reference FC fixes fusion; broader head projections and materialized SiLU still fail full-head gates. MoE cache remains unqualified after foreign/pageout deferrals and a functional-only hard 64 GiB available-memory stop; frozen source rules out full recurrent-state histories. Tiny affine operator measures command-heap growth and exact split/reuse, without full-model attribution or policy adoption. Retained ddalcu has a complete source/header peak plan and remains held; BF16 MoE full payload is absent. |
| [QNIP bounded prefill / source-rounded head continuation](../archive/runs/2026-11-09-qwen-moe-bounded-prefill-memory-plan.md) | Physical 256 preserves the logical 1,024 grid; two CPU controls, four sigmoid controls and 16 synthetic state/cache cases pass. Actual MoE 1,026-token cold/reuse/changed-tail controls pass 4/4 and match the frozen worker's cold output. Short 64-token image/text comparison passes fixed 5% checks with zero observed paging/compression/swap; full 1,792-token lifecycle remains pending. Source-rounded head still fails stage zero hidden absolute error (0.09375 > 0.0625), with two subsequent single-row stages exact; no head arithmetic adoption. |
| [QNIP pinned wide projection / production head continuation](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md) | Pinned MLX reduction and source operators pass the exact 0.8B BF16 unchanged head gate without test selectors. Production selection requires Legacy, qualified dense geometry and full-BF16 target/head metadata; target/MTP0 remain unchanged. Actual 1,026-token lifecycle passes 28 text / 34 image controls. Short 62-of-64-token loading-mode comparisons pass fixed 5% checks for MTP0 and MTP3, with zero observed paging/compression/swap; MTP3 is 49.03% slower on this small model. Initial timing overlapping package provisioning is retained/excluded. Other sizes/layouts remain unqualified. |
| [20261108-LVMTP-01: retained vision text MTP](../archive/runs/2026-11-08-lily-vision-text-mtp.md) | Lily vision/text MTP parity passed bounded output, 4,096-token prefix and image-transition controls; short paired text speed matched; memory recipe updated |
| [Image-support retrospective](2026-11-08-image-support-retrospective.md) | Conversation-wide findings, model coverage, validation mistakes, review, delivery and remaining work |
| [Qwen image attempts](2026-11-08-qwen-images.md) | BF16 and affine dense Qwen, affine and BF16-source MoE, retained Flash-Next/Lily numerical repairs and resource controls |
| [Gemma image attempts](2026-11-08-gemma-images.md) | Gemma 3 processor/convolution fixes; Gemma 4 efficient, unified, MoE and larger-tower attempts; decoder admission and retained compatibility |
| [20261109-G4IP-01: Gemma 4 initial image-loaded text controls](2026-11-09-gemma4-image-text-parity.md) | Ring snapshot components and short loading-mode measurements; original worker cache qualification withdrawn after full-prompt replay was found |
| [20261109-G4IP-02: Gemma 4 warmed cache correction](2026-11-09-gemma4-image-text-parity.md#20261109-g4ip-02-warmed-alternating-cache-timing) | Suffix-dispatch fix, actual-prefill invariant, nonperiodic/capacity regression and eligible warmed alternating controls; separate assistant conversion/execution remains open |
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

## 2026-11-09 request resource ceilings and installation

| ID | Scope and attempt | Recorded outcome | Disposition |
| --- | --- | --- | --- |
| 20261109-ADM-01 | [Initial Playground correction](2026-11-09-resource-admission.md): remove fixed content/generation fractions and correct Lily resident accounting | Model-free checks and short Qwen native controls passed; original private request was not replayed. | Adopted at `b9e904fe`; installation was pending. |
| 20261109-ADM-02 | [Follow-up ceiling audit](2026-11-09-resource-admission.md): Decision fraction, stale Auto allowance and progress cutoff | Native boundary, Auto growth and late-progress controls passed. Python startup failures reproduced by the fixture's two-second deadline; six normal isolated repeats passed. | Adopted at `de60be71`; included in installed build. |
| 20261109-ADM-03 | [Signed installation and actual Flash Next control](2026-11-09-resource-admission.md) | Two short Auto requests admitted the declared 262,144 context; bounded follow-up passed on the same worker. No swap growth/guard trigger; all 459 installed overlay hashes matched. Full-window execution untested. | Signed development build installed and opened; Ready observed. |
| 20261109-ADM-04 | [Non-Lily resident-memory accounting](2026-11-09-non-lily-admission.md): generic/dense-MTP/vision decoder and GLM/Qwen/Flint state ownership | Final-source review, full Metal suite and bounded Qwen/toy controls passed; full hybrid/vision artifacts unqualified. | Adopted on branch at `08fb0075`; PR authorized; no new installation. |

## Public record boundary

Publish aggregate results and hashes only. Keep prompts, completions, token vectors, probabilities, credentials, exception dumps, model files and private evaluation/runtime artifacts out of the ledger. An evidence link can point to an ignored private receipt by a stable identifier/hash without publishing its contents. Preserve upstream licenses and distinguish unmodified source revisions from local patches.
