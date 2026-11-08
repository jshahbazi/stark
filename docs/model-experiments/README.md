# Model experiment ledger

This ledger tracks compatibility, correctness, memory and performance attempts across Stark models, including failed, inconclusive and abandoned work. It links evidence rather than replacing source-bound run records or benchmark reports.

The initial entries cover the context/resource-budget conversation. Historical optimization reports have **not** all been backfilled. Use the [benchmark index](../benchmarks/README.md) for existing studies, including the [Qwen3.8 optimization history](../benchmarks/QWEN38_OPTIMIZATION_HISTORY.md) and [Qwen 27B approach ledger](../benchmarks/2026-11-06-qwen38-27b-approaches.md). Their measurements and dispositions retain their original scope.

## Recording an attempt

1. Assign `YYYYMMDD-AREA-NN` before changing code or running a trial. Create a record using [TEMPLATE.md](TEMPLATE.md), or a dated report containing equivalent fields. Put validation records under `docs/archive/runs/` and timed studies under `docs/benchmarks/`.
2. Record the hypothesis and exact artifact/backend, baseline/candidate source, precision, workload and acceptance criteria. Unknown values must say `not recorded` or `not yet measured`.
3. Update the outcome even when setup fails, admission rejects, outputs differ, performance regresses, host conditions invalidate timing, or the idea is dropped. Keep proposed/not-run ideas distinct from executed attempts.
4. Add a row below, link its detailed evidence, and record the code/PR/default/release disposition separately from the experiment result. Preserve older receipts when a correction or new tool changes the tested source.
5. For a shared infrastructure change, one record may contain a per-artifact matrix. A shared code path or passing unit suite must not mark every model in that matrix as qualified.

An attempt is a particular change under fixed artifact/backend and workload conditions. Different checkpoints, quantizations, precision policies, MTP depths, workloads or source revisions need separate results. A family-wide summary must retain those distinctions.

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

Use totals only within their recorded scope; overlapping reruns must not be summed. Report sampled tokens versus emitted/useful tokens explicitly. Do not use a directory name, available artifact, implemented backend or successful EOS as proof of broad model quality or speed.

## Attempt index

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
| 20261108-MEM-01 | Exact Lily Q4-XL and MLX-Serve mixed text artifacts on M5 Max 128 GiB; reconcile Beginner and native memory admission | Final trials passed 9/9 requests per artifact through near-32K, MTP on/off and reuse/fresh-scope hash agreement. Physical footprint stayed below 74 GiB, without additional swap. Earlier admission, compile, test and telemetry failures are retained. | Adopted on branch; signed installation pending. |

## Public record boundary

Publish aggregate results and hashes only. Keep prompts, completions, token vectors, probabilities, credentials, exception dumps, model files and private evaluation/runtime artifacts out of the ledger. An evidence link can point to an ignored private receipt by a stable identifier/hash without publishing its contents. Preserve upstream licenses and distinguish unmodified source revisions from local patches.
